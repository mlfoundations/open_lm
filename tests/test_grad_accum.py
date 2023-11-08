import torch
import os
from torch import optim
import torch.multiprocessing as mp
import copy

from open_lm.distributed import init_distributed_device
from open_lm.train import train_one_epoch
from open_lm.main import random_seed
from open_lm.model import create_model
from open_lm.data import get_data
from open_lm.scheduler import cosine_lr
from tests.shared import MockTrainArgs

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


def create_grad_acc_fixtures():
    # Setup data, optimizer, and other basic settings
    args = MockTrainArgs("open_lm_11m")

    # only want to look at one batch
    args.train_num_samples = args.batch_size

    # increase learning rate and remove warmup for maximize change to model weights
    args.lr = 2
    args.warmup = 0

    # create base models
    random_seed()
    model = create_model(args).to(args.device)

    # create dataloader
    data = get_data(
        args,
        epoch=0,
        tokenizer=None,
        skip_train=False,
    )

    # create optimizer
    named_parameters = list(model.named_parameters())
    params = [p for _, p in named_parameters if p.requires_grad]
    optimizer = optim.AdamW(
        [
            {"params": params, "weight_decay": args.wd},
        ],
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )

    # create scheduler
    scheduler = cosine_lr(
        optimizer,
        args.lr,
        args.warmup,
        10,
        args.lr_cooldown_end,
        args.force_min_lr,
    )

    # create loss
    loss = torch.nn.CrossEntropyLoss()

    return args, model, data, optimizer, scheduler, loss


def _grad_acc_helper(test_fsdp, accs=[2, 1], threshold=1e-7):
    if test_fsdp:
        world_size = 1
        mp.spawn(
            _grad_acc_helper_fsdp,
            args=(world_size, accs, threshold),
            nprocs=world_size,
            join=True,
        )
    else:
        _grad_acc_helper_single(test_fsdp=False, accs=accs, threshold=threshold)


def _grad_acc_helper_fsdp(rank, world_size, accs, threshold):
    # Initialize distributed training
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:29501",
        rank=rank,
        world_size=world_size,
    )
    _grad_acc_helper_single(test_fsdp=True, accs=accs, threshold=threshold)
    torch.distributed.destroy_process_group()


def _grad_acc_helper_single(test_fsdp, accs=[2, 1], threshold=1e-7):
    args, model, data, optimizer, scheduler, loss = create_grad_acc_fixtures()

    if test_fsdp:
        args.fsdp = True
        args.fsdp_amp = True

    # create models
    random_seed()
    model_accum_grad = copy.deepcopy(model).to(args.device)
    model_no_accum_grad = copy.deepcopy(model_accum_grad).to(args.device)

    # train on mock data with/without grad accumulation for one epoch
    for model, accum_freq in zip([model_accum_grad, model_no_accum_grad], accs):
        if test_fsdp:
            model = FSDP(model)
        args.accum_freq = accum_freq
        train_one_epoch(
            model,
            data,
            loss,
            0,
            optimizer,
            args.scaler,
            scheduler,
            args,
        )

    # check that models weights are similar (within some threshold)
    for p1, p2 in zip(model_accum_grad.parameters(), model_no_accum_grad.parameters()):
        assert torch.allclose(p1, p2, atol=threshold)


def test_grad_acc():
    _grad_acc_helper(test_fsdp=False)


def test_grad_acc_fsdp():
    _grad_acc_helper(test_fsdp=True)
