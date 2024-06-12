import pytest

from open_lm.data import get_wds_dataset
from open_lm.file_utils import get_string_for_epoch
from open_lm.train import train_one_epoch
from tests.shared import create_train_fixtures
from tests.utils import download_dl_test_data
from torch.cuda.amp import GradScaler

SOURCE_MANIFEST = ["tests/assets/source_3/manifest.jsonl"]


@pytest.mark.gpu
@pytest.mark.parametrize(
    "test_case",
    [
        (100, 2, 1000, 4, [20, 40]),  # Easy case.
        (100, 2, 1200, 4, [20, 40, 48]),  # End before consuming all in a shard.
        (100, 2, 1500, 4, [20, 40, 54, 60]),  # One of the shards here is smaller. 54 instead of 56 because of workers.
        (85, 2, 1000, 4, [22, 44, 47]),  # Batch weirdness, total_steps = 1000 * 4 // 85 = 47,
        # steps_epoch = 2000 // (85 * 2) * 2 = 22
    ],
)
def test_token_count(test_case):
    """Test if the correct number of steps are performed.

    Run training several times, and make sure that the expected number of steps is done each time.
    Having the same number of steps guarantees that the same number of tokens/samples are seen.

    TODO: this test seems to break for some reason, if test_training_simple.py is run along with it.
    It works fine when run by itself and if the other tests pass, and it does not affect CI, so it is fine for now.
    """
    batch_size, workers, desired_sequences_per_epoch, desired_epochs, expected_result = test_case

    download_dl_test_data()
    args, model, _, optimizer, scheduler, loss = create_train_fixtures("open_lm_11m")
    args.global_batch_size = batch_size
    args.per_gpu_batch_size = args.global_batch_size // args.world_size
    args.workers = workers
    args.train_data = None
    args.dataset_manifest = SOURCE_MANIFEST
    args.epochs = desired_epochs
    args.train_num_samples = desired_sequences_per_epoch
    args.scaler = None if args.precision != "amp" else GradScaler()

    total_samples = desired_sequences_per_epoch * desired_epochs
    total_steps = total_samples // (args.global_batch_size)
    global_step = 0
    next_shard_per_source = [0]
    epoch = 0
    data = None

    while True:
        if data is not None:
            del data

        shard_string_for_epoch, num_samples_per_source, next_shard_per_source = get_string_for_epoch(
            args.train_num_samples,
            next_shard_per_source,
            SOURCE_MANIFEST,
            weights=None,
            num_workers_per_gpu=args.workers,
            world_size=args.world_size,
        )
        args.train_data = shard_string_for_epoch
        print(args.train_data)
        data = {}
        data["train"] = get_wds_dataset(
            args, True, epoch, floor=True, force_num_samples=num_samples_per_source, data_key=args.data_key
        )

        success, global_step = train_one_epoch(
            model, data, loss, epoch, global_step, optimizer, args.scaler, scheduler, total_steps, args
        )

        assert success

        assert global_step == expected_result[epoch]

        epoch += 1

        if global_step == total_steps:
            break

    assert epoch == len(expected_result)
