import pytest
import argparse
import random
import os
import webdataset as wds

from open_lm.model import _MODEL_CONFIGS
from open_lm.main import random_seed
from open_lm.data import get_wds_dataset
from open_lm.file_utils import get_string_for_epoch, get_metadata_file
from open_lm.params import parse_args
from pathlib import Path

NUM_SAMPLES = 100000

# Update this to two data sources with webdataset, each with their own manifest.
INPUT_PATHS = [
    "path1/manifest.jsonl",
    "path2/manifest.jsonl",
]


def retrieve_dataset_once(epoch, weights, seed, disable_buffer):
    args = parse_args("")
    random_seed(seed)
    train_data_string_per_source, num_samples_per_source = get_string_for_epoch(
        NUM_SAMPLES, epoch, INPUT_PATHS, weights
    )
    args.train_num_samples = NUM_SAMPLES
    args.train_data = train_data_string_per_source
    args.num_workers = 2
    args.batch_size = 2
    args.seed = seed
    args.dataset_resampled = False
    args.disable_buffer = disable_buffer
    args.vocab_size = _MODEL_CONFIGS[args.model]["vocab_size"]
    args.seq_len = _MODEL_CONFIGS[args.model]["seq_len"]
    args.world_size = 1
    data = get_wds_dataset(
        args, is_train=True, epoch=epoch, force_num_samples=num_samples_per_source
    )
    dl = data.dataloader
    iterator = iter(dl)
    item = next(iterator)
    return item


def retrieve_dataset_once_resampled(epoch, weights, seed):
    args = parse_args("")
    random_seed(seed)
    train_data_string_per_source, _ = get_string_for_epoch(
        NUM_SAMPLES, epoch, INPUT_PATHS, weights
    )
    args.train_num_samples = NUM_SAMPLES
    args.train_data = train_data_string_per_source
    args.num_workers = 2
    args.batch_size = 2
    args.seed = seed
    args.dataset_resampled = True
    args.vocab_size = _MODEL_CONFIGS[args.model]["vocab_size"]
    args.seq_len = _MODEL_CONFIGS[args.model]["seq_len"]
    args.world_size = 1
    data = get_wds_dataset(args, is_train=True, epoch=epoch)
    dl = data.dataloader
    iterator = iter(dl)
    item = next(iterator)
    return item


@pytest.mark.parametrize("epoch", [0, 2])
@pytest.mark.parametrize("weights", [[0.5, 0.5], [0.9, 0.1]])
@pytest.mark.parametrize("seed", [0, 17])
def test_deterministic_no_buffer(epoch, weights, seed):
    disable_buffer = True
    output1 = retrieve_dataset_once(epoch, weights, seed, disable_buffer)
    output2 = retrieve_dataset_once(epoch, weights, seed, disable_buffer)
    assert output1 == output2


@pytest.mark.parametrize("epoch", [0, 2])
@pytest.mark.parametrize("weights", [[0.5, 0.5], [0.9, 0.1]])
@pytest.mark.parametrize("seed", [0, 17])
def test_deterministic_with_buffer(epoch, weights, seed):
    disable_buffer = False
    output1 = retrieve_dataset_once(epoch, weights, seed, disable_buffer)
    output2 = retrieve_dataset_once(epoch, weights, seed, disable_buffer)
    assert output1 == output2


@pytest.mark.parametrize("epoch", [0, 2])
@pytest.mark.parametrize("weights", [[0.5, 0.5], [0.9, 0.1]])
@pytest.mark.parametrize("seed", [0, 17])
def test_deterministic_resampled(epoch, weights, seed):
    output1 = retrieve_dataset_once_resampled(epoch, weights, seed)
    output2 = retrieve_dataset_once_resampled(epoch, weights, seed)
    assert output1 == output2


def test_count_manifest():
    manifest_path = INPUT_PATHS[0]
    metadata = get_metadata_file(manifest_path)
    idx = random.randint(0, len(metadata))
    item = metadata[idx]
    shard_path = os.path.join(
        str(Path(INPUT_PATHS[0]).parent), "shard_" + item["shard"] + ".tar"
    )
    shard_ds = wds.WebDataset(str(shard_path))
    count = 0
    for _ in iter(shard_ds):
        count += 1
    assert count == item["num_chunks"]
