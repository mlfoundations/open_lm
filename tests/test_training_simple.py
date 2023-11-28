from open_lm.main import main, train_one_epoch
import shutil
import pytest
import numpy as np

def test_train_simple():
    seq_len = 16
    num_batches = 5
    batch_size = 1
    # fmt: off
    main([
        "--train-num-samples", str(num_batches * seq_len),
        "--batch-size", str(batch_size),
        "--dataset-type", "synthetic",
        "--model", "open_lm_test_tiny",
        "--epochs", "1",
    ])
    # fmt: on



# =============================================================
# =                 LR Scheduling Tests                       =
# =============================================================
""" Tests for LR getting adjusted correctly. 
General test strat:
- Get to the point where we're about to train and break, getting all args
- Run training 'manually'
- Assert LR's are okay 
"""

@pytest.mark.parametrize('num_batches',
    [10, 100, 1000, 10000])
def test_lr_single_epoch_warmup(num_batches):
    """ Tests that LR gets adjusted correctly for a single epoch
    --
     """
    seq_len = 16
    num_batches = num_batches
    batch_size = 2
    args = ['--train-num-samples', str(num_batches * seq_len * batch_size),
            '--batch-size', str(batch_size),
            '--dataset-type', 'synthetic',
            '--model', 'open_lm_test_tiny',
            '--epochs', '1',
            '--finegrain-debug']

    main_vars = main(args)

    args = main_vars['args']
    assert args.warmup == 10_000 # Default warmup steps
    assert num_batches <= args.warmup

    # Now do training...
    if args.distributed:
        dist.barrier()

    success, global_step = train_one_epoch(
        main_vars['model'],
        main_vars['data'],
        main_vars['loss'],
        epoch=main_vars['start_epoch'],
        step=main_vars['global_step'],
        optimizer=main_vars['optimizer'],
        scaler=main_vars['scaler'],
        scheduler=main_vars['scheduler'],
        total_steps=main_vars['total_steps'],
        args=main_vars['args'],
        tb_writer=main_vars['writer'],
    )
    if args.distributed:
        dist.barrier()

    lrs = [_['lr'] for _ in main_vars['optimizer'].param_groups]
    assert all(abs(lr - num_batches / args.warmup * args.lr) < 1e-9 for lr in lrs)



@pytest.mark.parametrize('total_batches',
    [10, 100, 1000, 10000])
def test_lr_multi_epoch_warmup(total_batches):
    """ Tests that LR gets adjusted correctly for multiple epochs (but still in the warmup)
    """
    seq_len = 16
    num_epochs = 5
    num_batches = total_batches // num_epochs
    batch_size = 2
    args = ['--train-num-samples', str(num_batches * seq_len * batch_size),
            '--batch-size', str(batch_size),
            '--dataset-type', 'synthetic',
            '--model', 'open_lm_test_tiny',
            '--epochs', str(num_epochs),
            '--finegrain-debug']

    main_vars = main(args)

    args = main_vars['args']
    assert args.warmup == 10_000 # Default warmup steps
    assert num_batches <= args.warmup


    global_step = main_vars['global_step']

    for epoch in range(num_epochs):
    # Now do training...
        if args.distributed:
            dist.barrier()
        success, global_step = train_one_epoch(
            main_vars['model'],
            main_vars['data'],
            main_vars['loss'],
            epoch=main_vars['start_epoch'],
            step=global_step,
            optimizer=main_vars['optimizer'],
            scaler=main_vars['scaler'],
            scheduler=main_vars['scheduler'],
            total_steps=main_vars['total_steps'],
            args=main_vars['args'],
            tb_writer=main_vars['writer'],
        )
        if args.distributed:
            dist.barrier()

        lrs = [_['lr'] for _ in main_vars['optimizer'].param_groups]
        assert all(abs(lr - global_step / args.warmup * args.lr) < 1e-9 for lr in lrs)



def test_lr_single_epoch_cyclic():
    """ Tests that LR gets adjust correctly for a single epoch (that overruns the warmup)
    """

    seq_len = 16
    batch_size = 2
    num_batches = 500
    args = ['--train-num-samples', str(num_batches * seq_len * batch_size),
            '--batch-size', str(batch_size),
            '--dataset-type', 'synthetic',
            '--model', 'open_lm_test_tiny',
            '--epochs', '1',
            '--finegrain-debug',
            '--warmup', str(100)] # short warmup
    main_vars = main(args)

    args = main_vars['args']
    assert args.warmup == 100 # Default warmup steps
    assert num_batches > args.warmup


    # Now do training...
    if args.distributed:
        dist.barrier()

    success, global_step = train_one_epoch(
        main_vars['model'],
        main_vars['data'],
        main_vars['loss'],
        epoch=main_vars['start_epoch'],
        step=main_vars['global_step'],
        optimizer=main_vars['optimizer'],
        scaler=main_vars['scaler'],
        scheduler=main_vars['scheduler'],
        total_steps=main_vars['total_steps'],
        args=main_vars['args'],
        tb_writer=main_vars['writer'],
    )
    if args.distributed:
        dist.barrier()

    lrs = [_['lr'] for _ in main_vars['optimizer'].param_groups]

    target = 0.5 * (1 + np.cos(np.pi * (num_batches-1 - args.warmup) / (num_batches - args.warmup))) * args.lr
    assert all(abs(lr-target) < 1e-10 for lr in lrs)



def test_lr_multi_epoch_cyclic():
    """ Tests that LR gets adjust correctly for a single epoch (that overruns the warmup)
    """
    seq_len = 16
    batch_size = 2
    num_epochs = 5
    total_batches = 1000
    num_batches = total_batches // num_epochs
    print("NUM BATCHES", num_batches)
    args = ['--train-num-samples', str(num_batches * seq_len * batch_size),
            '--batch-size', str(batch_size),
            '--dataset-type', 'synthetic',
            '--model', 'open_lm_test_tiny',
            '--epochs', str(num_epochs),
            '--finegrain-debug',
            '--lr', '1e0',
            '--warmup', str(100)] # short warmup
    main_vars = main(args)

    args = main_vars['args']
    assert args.warmup == 100 # Default warmup steps
    assert num_batches > args.warmup


    compute_lr = lambda s: 0.5 * args.lr * (1 + np.cos(np.pi * (s - 100) / (total_batches - 100)))
    # Now do training...
    global_step = main_vars['global_step']
    all_lrs = []
    for epoch in range(num_epochs):
        if args.distributed:
            dist.barrier()

        success, global_step = train_one_epoch(
            main_vars['model'],
            main_vars['data'],
            main_vars['loss'],
            epoch=main_vars['start_epoch'],
            step=global_step,
            optimizer=main_vars['optimizer'],
            scaler=main_vars['scaler'],
            scheduler=main_vars['scheduler'],
            total_steps=main_vars['total_steps'],
            args=main_vars['args'],
            tb_writer=main_vars['writer'],
        )
        if args.distributed:
            dist.barrier()

        lrs = [_['lr'] for _ in main_vars['optimizer'].param_groups]

        all_lrs.append(lrs)
        # return args, all_lrs
        if global_step >= 100:
            assert all(abs(lr - compute_lr(global_step - 1)) < 1e-8 for lr in lrs)


