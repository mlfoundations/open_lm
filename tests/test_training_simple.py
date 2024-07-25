from open_lm.main import main, train_one_epoch, load_model
from open_lm.params import parse_args
import shutil
import pytest
import numpy as np
from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import glob
import os
import torch
import copy
from tests.shared import create_train_fixtures
from tests.utils import download_dl_test_data


LOG_PATH = "./logs/test_logs/test_lr_scheduling_from_main/"

# ==============================================================
# =                 Testing utilities                          =
# ==============================================================


def parse_tensorboard(tb_path):
    ea = event_accumulator.EventAccumulator(
        tb_path,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    _absorb_print = ea.Reload()
    return {k: pd.DataFrame(ea.Scalars(k)) for k in ea.Tags()["scalars"]}


@pytest.fixture
def cleanup_test_logs():
    print("RUNNING FIXTURE")
    shutil.rmtree(LOG_PATH, ignore_errors=True)


# ==============================================================
# =                    Simple Test                             =
# ==============================================================


def test_train_simple():
    seq_len = 16
    num_batches = 5
    batch_size = 1
    # fmt: off
    main([
        "--train-num-samples", str(num_batches * seq_len),
        "--global-batch-size", str(batch_size),
        "--dataset-type", "synthetic",
        "--model", "open_lm_test_tiny",
        "--epochs", "1",

    ])
    # fmt: on


# ==============================================================
# =             Resuming + Determinism Tests                   =
# ==============================================================


def test_training_deterministic():
    download_dl_test_data()
    name1 = "test_training_deterministic_test1"
    name2 = "test_training_deterministic_test2"

    logdir = "tests/assets/"
    args = [
        "--train-num-samples",
        150 * 16,  # seq_len is 16 for open_lm_test_tiny
        "--global-batch-size",
        4,
        "--workers",
        1,
        "--model",
        "open_lm_test_tiny",
        "--dataset-manifest",
        "tests/assets/source_seq_len_16/manifest.jsonl",
        "--logs",
        logdir,
        "--seed",
        42,
        "--data-key",
        "json",
        "--model-norm",
        "gain_only_layer_norm",
    ]
    args = [str(x) for x in args]
    args2 = copy.deepcopy(args)

    try:
        # Train for one epoch each model, with the same seed.
        # Make sure that the models are the same.
        main(args + ["--name", name1, "--epochs", "2"])
        main(args + ["--name", name2, "--epochs", "2"])

        _, model, _, _, _, _ = create_train_fixtures("open_lm_test_tiny")
        _, model2, _, _, _, _ = create_train_fixtures("open_lm_test_tiny")

        args = parse_args(args)
        args.resume = f"{logdir}{name1}/checkpoints/epoch_1.pt"
        args.seed = 42
        args.distributed = False
        load_model(args, model)
        args2 = parse_args(args2)
        args2.resume = f"{logdir}{name2}/checkpoints/epoch_1.pt"
        args2.seed = 42
        args2.distributed = False
        load_model(args2, model2)

        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
            assert torch.allclose(p1, p2, atol=1e-6)

    finally:
        shutil.rmtree(f"{logdir}{name1}", ignore_errors=True)
        shutil.rmtree(f"{logdir}{name2}", ignore_errors=True)


def test_good_resume_shard_shuffle():
    download_dl_test_data()
    name1 = "test_good_resume_test1"
    name2 = "test_good_resume_test2"

    logdir = "tests/assets/"
    args = [
        "--train-num-samples",
        150 * 16,  # seq_len is 16 for open_lm_test_tiny
        "--global-batch-size",
        4,
        "--workers",
        1,
        "--model",
        "open_lm_test_tiny",
        "--dataset-manifest",
        "tests/assets/source_seq_len_16/manifest.jsonl",
        "--logs",
        logdir,
        "--seed",
        42,
        "--data-key",
        "json",
        "--model-norm",
        "gain_only_layer_norm",
    ]
    args = [str(x) for x in args]

    try:
        # Train for one epoch each model, with the same seed.
        # Make sure that the models are the same.
        main(args + ["--name", name1, "--epochs", "1"])
        main(args + ["--name", name1, "--epochs", "2", "--resume", "latest"])
        main(args + ["--name", name2, "--epochs", "2"])

        logfile1 = f"{logdir}{name1}/out.log"
        with open(logfile1, "r") as f:
            lines1 = f.read()
        lines1 = lines1.splitlines()

        logfile2 = f"{logdir}{name2}/out.log"
        with open(logfile2, "r") as f:
            lines2 = f.read()
        lines2 = lines2.splitlines()

        # Make sure that the same set of shards was chosen in each setting, regardless of resuming
        for t in range(2):
            splitting_str = f"=> epoch {t}, training on"  # Shard ids are printed after this.
            choice1 = [x.split(splitting_str)[1] for x in lines1 if splitting_str in x][0]
            choice2 = [x.split(splitting_str)[1] for x in lines2 if splitting_str in x][0]
            assert choice1 == choice2

    finally:
        shutil.rmtree(f"{logdir}{name1}", ignore_errors=True)
        shutil.rmtree(f"{logdir}{name2}", ignore_errors=True)


# =============================================================
# =                 LR Scheduling Tests                       =
# =============================================================
""" Tests for LR getting adjusted correctly. 
General test strat:
- Get to the point where we're about to train and break, getting all args
- Run training 'manually'
- Assert LR's are okay 
"""


@pytest.mark.parametrize("num_batches", [10, 100, 1000])
def test_lr_single_epoch_warmup(num_batches, cleanup_test_logs):
    """Tests that LR gets adjusted correctly for a single epoch
    --
    """
    seq_len = 16
    num_batches = num_batches
    batch_size = 2
    args = [
        "--train-num-samples",
        str(num_batches * seq_len * batch_size),
        "--global-batch-size",
        str(batch_size),
        "--dataset-type",
        "synthetic",
        "--model",
        "open_lm_test_tiny",
        "--epochs",
        "1",
        "--lr",
        "1e0",  # Artificially high LR
        "--report-to",
        "tensorboard",
        "--log-every-n-steps",
        str(1),
        "--logs",
        LOG_PATH,
        "--name",
        "test_lr_single_epoch_warmup_%03d" % num_batches,
    ]
    output_args = main(args)

    tb_data = parse_tensorboard(glob.glob(os.path.join(output_args.tensorboard_path, "*"))[0])
    lr_array = np.array(tb_data["train/lr"]["value"])
    assert len(lr_array) == num_batches  # Make sure we've flushed TB
    expected_lr_array = np.array([(i + 1) / 10_000 for i in range(len(lr_array))])

    assert abs(lr_array - expected_lr_array).max() < 1e-6


@pytest.mark.parametrize("total_batches", [10, 100, 1000])
def test_lr_multi_epoch_warmup(total_batches, cleanup_test_logs):
    """Tests that LR gets adjusted correctly for multiple epochs (but still in the warmup)"""
    seq_len = 16
    num_epochs = 5
    num_batches = total_batches // num_epochs
    batch_size = 2
    args = [
        "--train-num-samples",
        str(num_batches * seq_len * batch_size),
        "--global-batch-size",
        str(batch_size),
        "--dataset-type",
        "synthetic",
        "--model",
        "open_lm_test_tiny",
        "--epochs",
        str(num_epochs),
        "--lr",
        "1e0",
        "--report-to",
        "tensorboard",
        "--log-every-n-steps",
        str(1),
        "--logs",
        LOG_PATH,
        "--name",
        "test_lr_multi_epoch_warmup_%03d" % total_batches,
    ]

    output_args = main(args)

    tb_data = parse_tensorboard(glob.glob(os.path.join(output_args.tensorboard_path, "*"))[0])
    lr_array = np.array(tb_data["train/lr"]["value"])

    assert len(lr_array) == total_batches
    expected_lr_array = np.array([(i + 1) / 10_000 for i in range(len(lr_array))])
    assert abs(lr_array - expected_lr_array).max() < 1e-6


def test_lr_single_epoch_cyclic(cleanup_test_logs):
    """Tests that LR gets adjust correctly for a single epoch (that overruns the warmup)"""

    seq_len = 16
    batch_size = 2
    num_batches = 500
    warmup = 100
    lr = 1e0
    args = [
        "--train-num-samples",
        str(num_batches * seq_len * batch_size),
        "--global-batch-size",
        str(batch_size),
        "--dataset-type",
        "synthetic",
        "--model",
        "open_lm_test_tiny",
        "--epochs",
        "1",
        "--warmup",
        str(warmup),  # short warmup
        "--lr",
        str(lr),  # artificially high LR
        "--report-to",
        "tensorboard",
        "--log-every-n-steps",
        str(1),
        "--logs",
        LOG_PATH,
        "--name",
        "test_lr_single_epoch_cyclic",
    ]
    output_args = main(args)

    tb_data = parse_tensorboard(glob.glob(os.path.join(output_args.tensorboard_path, "*"))[0])
    lr_array = np.array(tb_data["train/lr"]["value"])

    assert len(lr_array) == num_batches
    es = num_batches - warmup
    lr_calc = lambda i: 0.5 * (1 + np.cos(np.pi * (i - warmup) / es)) * lr
    expected_lr_array = [(1 + i) / warmup for i in range(warmup)]
    expected_lr_array = expected_lr_array + [lr_calc(i) for i in range(warmup, num_batches)]
    expected_lr_array = np.array(expected_lr_array)
    assert len(expected_lr_array) == len(lr_array)
    assert abs(lr_array - np.array(expected_lr_array)).max() < 1e-6


def test_lr_multi_epoch_cyclic(cleanup_test_logs):
    """Tests that LR gets adjust correctly for a single epoch (that overruns the warmup)"""
    seq_len = 16
    batch_size = 2
    num_epochs = 5
    total_batches = 1000
    warmup = 100
    lr = 1e0
    num_batches = total_batches // num_epochs
    print("NUM BATCHES", num_batches)
    args = [
        "--train-num-samples",
        str(num_batches * seq_len * batch_size),
        "--global-batch-size",
        str(batch_size),
        "--dataset-type",
        "synthetic",
        "--model",
        "open_lm_test_tiny",
        "--epochs",
        str(num_epochs),
        "--lr",
        str(lr),
        "--warmup",
        str(warmup),
        "--report-to",
        "tensorboard",
        "--log-every-n-steps",
        str(1),
        "--logs",
        LOG_PATH,
        "--name",
        "test_lr_multi_epoch_cyclic",
    ]

    output_args = main(args)

    tb_data = parse_tensorboard(glob.glob(os.path.join(output_args.tensorboard_path, "*"))[0])
    lr_array = np.array(tb_data["train/lr"]["value"])
    assert len(lr_array) == total_batches

    es = total_batches - warmup
    lr_calc = lambda i: 0.5 * (1 + np.cos(np.pi * (i - warmup) / es)) * lr
    expected_lr_array = [(1 + i) / warmup for i in range(warmup)]
    expected_lr_array = expected_lr_array + [lr_calc(i) for i in range(warmup, total_batches)]
    expected_lr_array = np.array(expected_lr_array)
    assert len(expected_lr_array) == len(lr_array)
    assert abs(lr_array - np.array(expected_lr_array)).max() < 1e-6


# =========================================================
# =                main.py tests for LR                   =
# =========================================================


def test_lr_scheduling_from_main(cleanup_test_logs):
    # Then do a training run
    seq_len = 16
    batch_size = 5
    num_epochs = 1
    total_batches = 1000
    num_batches = total_batches // num_epochs
    args = [
        "--train-num-samples",
        str(num_batches * seq_len * batch_size),
        "--global-batch-size",
        str(batch_size),
        "--dataset-type",
        "synthetic",
        "--model",
        "open_lm_test_tiny",
        "--epochs",
        str(num_epochs),
        "--debug",
        "--lr",
        "1e0",  # artificially high LR
        "--warmup",
        str(100),  # short warmup
        "--log-every-n-steps",
        str(1),
        "--report-to",
        "tensorboard",
        "--logs",
        LOG_PATH,
        "--name",
        "test_lr_scheduling_from_main",
    ]
    output_args = main(args)

    tb_data = parse_tensorboard(glob.glob(os.path.join(output_args.tensorboard_path, "*"))[0])
    lr_array = np.array(tb_data["train/lr"]["value"])
    assert len(lr_array) == 1000  # Make sure we've flushed TB

    compute_lr = lambda s: 0.5 * 1e0 * (1 + np.cos(np.pi * (s - 100) / (total_batches - 100)))
    expected_lr = np.array([1e0 * i / 100 for i in range(1, 101)] + [compute_lr(_) for _ in range(100, 1000)])
    max_diff = (abs(expected_lr - lr_array)).max()
    assert max_diff < 1e-7


# =========================================================
# =             main.py tests for no positional           =
# =========================================================
def test_train_no_positional():
    seq_len = 16
    num_batches = 5
    batch_size = 1
    # fmt: off
    main([
        "--train-num-samples", str(num_batches * seq_len),
        "--global-batch-size", str(batch_size),
        "--dataset-type", "synthetic",
        "--model", "open_lm_test_tiny",
        "--epochs", "1",
        "--positional-embedding-type", "none",
    ])
    # fmt: on
