import pytest
import os
import shutil

import torch.multiprocessing as mp

from open_lm.main import main


def tiny_save_load(fsdp=False, distributed=False):
    """
    This test checks that the model can be saved and loaded without changing the parameters.
    """
    name = "test_tiny_save_load"
    # fmt: off
    logdir = "tests/assets/"
    args = [
        "--train-num-samples", 64 * 16,  # seq_len is 16 for open_lm_test_tiny
        "--global-batch-size", 4,
        "--name", name,
        "--model", "open_lm_test_tiny",
        "--dataset-type", "synthetic",
        "--logs", logdir,
    ]
    args = [str(x) for x in args]
    # fmt: on

    if fsdp:
        args += ["--fsdp", "--fsdp-amp", "--precision", "amp_bf16"]
        assert distributed

    if distributed:
        args += ["--force-distributed"]
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "12301"

    try:
        # Train for one epoch, load the model, then train for another epoch.
        from torch.distributed.distributed_c10d import GroupMember, _world
        main(args + ["--epochs", "1"])

        import subprocess
        subprocess.run(["ls", "--recursive", logdir])
        print(f"Abspath: {os.path.abspath(logdir)=}")
        # Loading saved tiny model
        resume_args = args + ["--resume", "latest", "--epochs", "2"]
        main(resume_args)
    finally:
        shutil.rmtree(f"{logdir}{name}", ignore_errors=True)


def _save_load_helper_dist(rank, fsdp):
    tiny_save_load(fsdp=fsdp, distributed=True)


def test_tiny_save_load_no_distributed():
    tiny_save_load(fsdp=False, distributed=False)


@pytest.mark.parametrize("fsdp", [False, True])
def test_tiny_save_load_dist_fsdp(fsdp):
    mp.spawn(_save_load_helper_dist, args=(fsdp,), nprocs=1, join=True)


if __name__ == "__main__":
    pytest.main([__file__])
