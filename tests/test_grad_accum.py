import copy
import pytest

import torch
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from open_lm.train import train_one_epoch
from open_lm.main import random_seed
from tests.shared import create_train_fixtures


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
        backend="nccl" if torch.cuda.is_available() else "gloo",
        init_method="tcp://127.0.0.1:29501",
        rank=rank,
        world_size=world_size,
    )
    _grad_acc_helper_single(test_fsdp=True, accs=accs, threshold=threshold)
    torch.distributed.destroy_process_group()


def _grad_acc_helper_single(test_fsdp, accs=[2, 1], threshold=1e-7):
    random_seed()
    args, model_accum_grad, data, optimizer, scheduler, loss = create_train_fixtures()
    random_seed()
    _, model_no_accum_grad, _, _, _, _ = create_train_fixtures()
    model_no_accum_grad.load_state_dict(model_accum_grad.state_dict())

    # check that models weights are similar at the start (within some threshold)
    for p1, p2 in zip(model_accum_grad.parameters(), model_no_accum_grad.parameters()):
        assert torch.allclose(p1, p2, atol=threshold)

    if test_fsdp:
        args.fsdp = True
        args.fsdp_amp = True

    # train on mock data with/without grad accumulation for one epoch
    for model, accum_freq in zip([model_accum_grad, model_no_accum_grad], accs):
        if test_fsdp:
            model = FSDP(model)
        args.accum_freq = accum_freq
        train_one_epoch(
            model=model,
            data=data,
            loss=loss,
            epoch=0,
            step=0,
            optimizer=optimizer,
            scaler=args.scaler,
            scheduler=scheduler,
            total_steps=10,
            args=args,
        )

    # check that models weights are similar (within some threshold)
    for p1, p2 in zip(model_accum_grad.parameters(), model_no_accum_grad.parameters()):
        assert torch.allclose(p1, p2, atol=threshold)


def test_grad_acc():
    _grad_acc_helper(test_fsdp=False)


@pytest.mark.gpu
def test_grad_acc_fsdp():
    _grad_acc_helper(test_fsdp=True)
