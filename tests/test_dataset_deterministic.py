import pytest
import argparse
import random
import os
import webdataset as wds
import glob
from open_lm.model import _MODEL_CONFIGS
from open_lm.main import random_seed
from open_lm.data import get_wds_dataset
from open_lm.file_utils import (
    get_string_for_epoch,
    get_metadata_file,
    get_shards_for_chunk,
)
from open_lm.params import parse_args
from pathlib import Path
from tests.utils import download_dl_test_data
from time import sleep

NUM_SAMPLES = 1000
NUM_SAMPLES_TO_CHECK = 5

# Update this to two data sources with webdataset, each with their own manifest.
INPUT_PATHS = [
    "tests/assets/source_1/manifest.jsonl",
    "tests/assets/source_2/manifest.jsonl",
]


def retrieve_dataset(epoch, next_shard, weights, seed, disable_buffer, min_shards_needed=2):
    args = parse_args("")

    train_data_string_per_source, num_samples_per_source, _ = get_string_for_epoch(
        NUM_SAMPLES, [next_shard, next_shard], INPUT_PATHS, weights, min_shards_needed, world_size=1
    )
    args.train_num_samples = NUM_SAMPLES
    args.train_data = train_data_string_per_source
    args.workers = 2
    args.global_batch_size = 2
    args.seed = seed
    args.dataset_resampled = False
    args.disable_buffer = disable_buffer
    args.vocab_size = _MODEL_CONFIGS[args.model]["vocab_size"]
    args.seq_len = _MODEL_CONFIGS[args.model]["seq_len"]
    args.world_size = 1
    args.rank = 0
    args.per_gpu_batch_size = 2
    data = get_wds_dataset(args, is_train=True, epoch=epoch, force_num_samples=num_samples_per_source)
    dl = data.dataloader

    return dl


def retrieve_dataset_resampled(epoch, next_shard, weights, seed, min_shards_needed=2):
    args = parse_args("")
    train_data_string_per_source, _, _ = get_string_for_epoch(
        NUM_SAMPLES, [next_shard, next_shard], INPUT_PATHS, weights, min_shards_needed, world_size=1
    )
    args.train_num_samples = NUM_SAMPLES
    args.train_data = train_data_string_per_source
    args.num_workers = 2
    args.global_batch_size = 2
    args.seed = seed
    args.dataset_resampled = True
    args.vocab_size = _MODEL_CONFIGS[args.model]["vocab_size"]
    args.seq_len = _MODEL_CONFIGS[args.model]["seq_len"]
    args.world_size = 1
    args.rank = 0
    args.per_gpu_batch_size = 2
    data = get_wds_dataset(args, is_train=True, epoch=epoch)
    dl = data.dataloader

    return dl


@pytest.mark.parametrize("next_shard", [0, 2])
@pytest.mark.parametrize("weights", [[0.5, 0.5], [0.9, 0.1]])
@pytest.mark.parametrize("seed", [0, 17])
def test_deterministic_no_buffer(next_shard, weights, seed):
    download_dl_test_data("tests/assets")
    disable_buffer = True
    random_seed(seed)
    dl1 = retrieve_dataset(0, next_shard, weights, seed, disable_buffer)
    dl2 = retrieve_dataset(0, next_shard, weights, seed, disable_buffer)

    iter1 = iter(dl1)
    iter2 = iter(dl2)

    for _ in range(NUM_SAMPLES_TO_CHECK):
        item1 = next(iter1)
        item2 = next(iter2)
        assert item1 == item2
        sleep(0.001)


@pytest.mark.parametrize("next_shard", [0, 2])
@pytest.mark.parametrize("weights", [[0.5, 0.5], [0.9, 0.1]])
@pytest.mark.parametrize("seed", [0, 17])
def test_deterministic_with_buffer(next_shard, weights, seed):
    download_dl_test_data("tests/assets")
    disable_buffer = False
    random_seed(seed)
    dl1 = retrieve_dataset(0, next_shard, weights, seed, disable_buffer)
    dl2 = retrieve_dataset(0, next_shard, weights, seed, disable_buffer)

    iter1 = iter(dl1)
    iter2 = iter(dl2)

    for _ in range(NUM_SAMPLES_TO_CHECK):
        item1 = next(iter1)
        item2 = next(iter2)
        assert item1 == item2
        sleep(0.001)


@pytest.mark.parametrize("next_shard", [0, 2])
@pytest.mark.parametrize("weights", [[0.5, 0.5], [0.9, 0.1]])
@pytest.mark.parametrize("seed", [0, 17])
def test_deterministic_resampled(next_shard, weights, seed):
    download_dl_test_data("tests/assets")
    random_seed(seed)
    dl1 = retrieve_dataset_resampled(0, next_shard, weights, seed)
    dl2 = retrieve_dataset_resampled(0, next_shard, weights, seed)

    iter1 = iter(dl1)
    iter2 = iter(dl2)

    for _ in range(NUM_SAMPLES_TO_CHECK):
        item1 = next(iter1)
        item2 = next(iter2)
        assert item1 == item2
        sleep(0.001)


@pytest.mark.parametrize("next_shard", [0, 2])
@pytest.mark.parametrize("weights", [[0.5, 0.5], [0.6, 0.4]])
@pytest.mark.parametrize("min_shards_needed", [2, 4])
def test_min_shards(next_shard, weights, min_shards_needed):
    download_dl_test_data("tests/assets")
    shard_strings, _, _ = get_string_for_epoch(
        NUM_SAMPLES, [next_shard, next_shard], INPUT_PATHS, weights, min_shards_needed, world_size=1
    )
    for item in shard_strings:
        num_shards = len(item.split(","))
        assert num_shards >= min_shards_needed


def test_count_manifest():
    download_dl_test_data("tests/assets")
    manifest_path = INPUT_PATHS[0]
    metadata = get_metadata_file(manifest_path)
    idx = random.randint(0, len(metadata))
    item = metadata[idx]
    shard_path = os.path.join(str(Path(INPUT_PATHS[0]).parent), item["shard"] + ".tar")
    shard_ds = wds.WebDataset(str(shard_path))
    count = 0
    for _ in iter(shard_ds):
        count += 1
    assert count == item["num_sequences"]
