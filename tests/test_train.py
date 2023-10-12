import torch
from torch import optim

from open_lm.train import train_one_epoch
from open_lm.main import random_seed
from open_lm.model import create_model
from open_lm.data import get_data
from open_lm.scheduler import cosine_lr
from tests.shared import MockArgs


def test_grad_acc():
    args = MockArgs("open_lm_11m")

    # create model
    random_seed()
    model = create_model(args)

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

    # train on mock data without grad accumulation
    train_one_epoch(
        model,
        data,
        loss,
        0,
        optimizer,
    )

    # train on mock data with grad accumulation

    # check that models weights are similar (within some threshold)


def test_grad_acc_fsdp():
    # TODO: similar to above but also init fsdp
    pass
