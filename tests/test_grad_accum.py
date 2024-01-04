import copy
import pytest

import torch
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from open_lm.model import create_model

from open_lm.train import train_one_epoch
from open_lm.main import random_seed
from tests.shared import create_train_fixtures


def _grad_acc_helper(test_fsdp, accs=[1, 2], threshold=1e-7):
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
    # List of tuples with (args, model, data, optimizer, scheduler, loss)
    fixtures = []
    for _ in accs:
        random_seed()
        (args, model, data, optimizer, scheduler, loss) = create_train_fixtures()

        # HACK: Currently, AdamW optimizer leads to different results with gradient accumulation.
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=args.lr)

        if test_fsdp:
            args.fsdp = True
            args.fsdp_amp = True
            # Required to force distributed mode on 1 gpu.
            args.distributed = True
        fixtures.append((args, model, data, optimizer, scheduler, loss))

    model1 = fixtures[0][1]
    for fixture in fixtures[1:]:
        model2 = fixture[1]
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2, atol=threshold), "Parameter mismatch at init"

    # train on mock data with/without grad accumulation for one epoch
    for fixture, accum_freq in zip(fixtures, accs):
        args, model, data, optimizer, scheduler, loss = fixture
        if test_fsdp:
            model = FSDP(model)
        args.accum_freq = accum_freq
        random_seed()
        train_one_epoch(
            model=model,
            data=data,
            loss=loss,
            epoch=0,
            step=0,
            optimizer=optimizer,
            scaler=None,
            scheduler=scheduler,
            total_steps=10,
            args=args,
        )

    model1 = fixtures[0][1]
    failed_grad = []
    failed_weight = []
    for fixture in fixtures[1:]:
        model2 = fixture[1]
        for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
            if not torch.allclose(p1.grad, p2.grad, atol=threshold):
                failed_grad.append(n1)
                print(f"Gradient mismatch at {n1}, {n2}")

            if not torch.allclose(p1, p2, atol=threshold):
                failed_weight.append(n1)
                print(f"Weight mismatch at {n1}, {n2}")
    assert not failed_grad, f"Failed gradient checks at: {failed_grad}"
    assert not failed_weight, f"Failed weight checks at: {failed_weight}"


def test_no_accumulation_matches():
    _grad_acc_helper(test_fsdp=False, accs=[1, 1])


def test_grad_acc():
    _grad_acc_helper(test_fsdp=False, accs=[1, 2])


@pytest.mark.gpu
def test_grad_acc_fsdp():
    _grad_acc_helper(test_fsdp=True, accs=[1, 2])
