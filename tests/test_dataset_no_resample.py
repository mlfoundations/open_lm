""" TEST STRATEGY:
Tests to just handle the edge cases in sampling WITHOUT replacement.

We are just going to cover the path where we have:
- Exactly one source 
- Never request more data than we have available

And generally we want to assert that:
- We never see any repeated data 
- We see exactly the right number of samples as desired (some math here)

"""

import pytest
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
from tests.utils import download_dl_test_data, make_fake_tarfiles
import json
from collections import Counter, defaultdict

import numpy as np

SEQ_LEN = 2048
INPUT_PATHS = [
    "tests/assets/source_id_00/manifest.jsonl",
    "tests/assets/source_id_01/manifest.jsonl",
]
SINGLE_SOURCE = [INPUT_PATHS[0]]

TOTAL_SEQ_COUNT = 666 * 2
SINGLE_SEQ_COUNT = 666


# ===============================================
# =                Helpers                      =
# ===============================================


def get_dataset_size(paths, count_tokens=False):
    """Computes dataset size (given list of paths to the manifest)
    Returns number of sequences by default, but can multiply by 2048 if you ask it to =)
    """
    total_seq_count = 0
    for filename in paths:
        for line in open(filename, "r").readlines():
            total_seq_count += json.loads(line)["num_chunks"]
    return total_seq_count * 2048 if count_tokens else total_seq_count


def retrieve_dataset_once(
    total_seqs, paths, epoch, next_shard, weights, seed, disable_buffer, batch_size, num_workers, min_shards_needed=1
):
    """Returns the output of get_wds_dataset -- not dataloader or nothing fancy
    Only works for a single source.
    """

    assert len(paths) == 1
    args = parse_args("")
    random_seed(seed)

    train_data_string_per_source, num_seqs_per_source, _ = get_string_for_epoch(
        num_samples=total_seqs,
        starting_points=[next_shard],
        paths=paths,
        weights=weights,
        num_workers_per_gpu=num_workers,
        world_size=1,
        multi_epoch=False,
    )

    args.train_num_samples = total_seqs
    args.train_data = train_data_string_per_source
    args.workers = num_workers
    args.global_batch_size = args.per_gpu_batch_size = batch_size
    args.seed = seed
    args.dataset_resampled = False
    args.disable_buffer = disable_buffer
    args.vocab_size = _MODEL_CONFIGS[args.model]["vocab_size"]
    args.seq_len = _MODEL_CONFIGS[args.model]["seq_len"]
    args.world_size = 1
    args.rank = 0
    data = get_wds_dataset(args, is_train=True, epoch=epoch, force_num_samples=[total_seqs])
    return data


# ======================================================
# =           Single Source Test Cases                 =
# ======================================================


@pytest.mark.parametrize(
    "num_samples,next_shard,batch_size,min_shards_needed",
    [(10, 0, 1, 1), (100, 0, 25, 2), (150, 1, 50, 4), (666, 0, 111, 3)],
)
def test_singleSource_singleWorker_perfectBatch(num_samples, next_shard, batch_size, min_shards_needed):
    """Cases where:
        - Only a single source
        - Only a single worker
        - The batch_size perfectly divides the number of samples requested

    Should see exactly num_samples at the end, and no repeats
    """
    download_dl_test_data()
    make_fake_tarfiles()

    data = retrieve_dataset_once(
        num_samples,
        SINGLE_SOURCE,
        0,
        next_shard,
        None,
        seed=42,
        disable_buffer=True,
        batch_size=batch_size,
        num_workers=1,
        min_shards_needed=min_shards_needed,
    )

    data_ids = []
    for batch in data.dataloader:
        for seq in batch[0]:
            data_ids.append(tuple(seq[:3]))

    assert len(set(data_ids)) == len(data_ids) == num_samples  # Checking no repeats and exact


@pytest.mark.parametrize(
    "num_samples,next_shard,batch_size, min_shards_needed", [(50, 0, 7, 1), (250, 2, 13, 4), (666, 0, 13, 1)]
)
def test_singleSource_singleWorker_imperfectBatch(num_samples, next_shard, batch_size, min_shards_needed):
    """Cases where:
        - Only a single source
        - Only a single worker
        - Batch size does NOT divide the number of samples

    Should see no repeats, but the greatest multiple of batch_size <= num_samples
    """
    download_dl_test_data()
    make_fake_tarfiles()

    data = retrieve_dataset_once(
        num_samples,
        SINGLE_SOURCE,
        0,
        next_shard,
        None,
        seed=42,
        disable_buffer=True,
        batch_size=batch_size,
        num_workers=1,
        min_shards_needed=min_shards_needed,
    )

    data_ids = []
    for batch in data.dataloader:
        for seq in batch[0]:
            data_ids.append(tuple(seq[:3]))

    assert len(set(data_ids)) == len(data_ids)  # Check no repeats
    assert len(data_ids) == batch_size * (num_samples // batch_size)  #


def test_singleSource_multiWorker_0():
    """
    Asking for 200 samples with 2 workers and a batchsize of 10.
    So we should get 2 shards, distribute 1 to each worker and each should fully consume it

    We should see all samples from the first two shards
    """
    download_dl_test_data()
    make_fake_tarfiles()

    num_samples = 200
    epoch = 0
    next_shard = 0
    batch_size = 10
    min_shards_needed = num_workers = 2

    data = retrieve_dataset_once(
        num_samples,
        SINGLE_SOURCE,
        epoch,
        next_shard,
        None,
        seed=42,
        disable_buffer=True,
        batch_size=batch_size,
        num_workers=num_workers,
        min_shards_needed=min_shards_needed,
    )

    data_ids = []
    for batch in data.dataloader:
        for seq in batch[0]:
            data_ids.append(tuple(seq[:3]))

    assert len(set(data_ids)) == len(data_ids)  # Check no repeats
    # Now reasoning about this, I should consume each fully

    target_data_ids = [(0, i, j) for i in range(2) for j in range(100)]  # All of chunk 0000, chunk 0001
    assert sorted(target_data_ids) == sorted(data_ids)


def test_singleSource_multiWorker_1():
    """
    Asking for 150 samples from 2 workers.
    - Should get 2 shards, and give 1 to each worker
    - Each worker should see 15 batches
    Output should be the first 75 examples from each of the first 2 shards
    """
    download_dl_test_data()
    make_fake_tarfiles()

    num_samples = 150
    epoch = 0
    next_shard = 0
    batch_size = 5
    min_shards_needed = num_workers = 2

    data = retrieve_dataset_once(
        num_samples,
        SINGLE_SOURCE,
        epoch,
        next_shard,
        None,
        seed=42,
        disable_buffer=True,
        batch_size=batch_size,
        num_workers=num_workers,
        min_shards_needed=min_shards_needed,
    )

    data_ids = []
    for batch in data.dataloader:
        for seq in batch[0]:
            data_ids.append(tuple(seq[:3]))

    assert len(set(data_ids)) == len(data_ids)  # Check no repeats

    target_data_ids = [(0, i, j) for i in range(2) for j in range(75)]  # 3/4 of chunk 0000, chunk 0001
    assert sorted(target_data_ids) == sorted(data_ids)


def test_singleSource_multiWorker_2():
    """
    Asking for 150 samples from 2 workers
    - Should get 2 shards and give 1 to each worker
    - Each worker should see 150 // (2 * 10) = 7 batches
    Output should be the first 70 examples from each of the first 2 shards
    """
    download_dl_test_data()
    make_fake_tarfiles()

    num_samples = 150
    epoch = 0
    next_shard = 0
    batch_size = 10
    min_shards_needed = num_workers = 2

    data = retrieve_dataset_once(
        num_samples,
        SINGLE_SOURCE,
        epoch,
        next_shard,
        None,
        seed=42,
        disable_buffer=True,
        batch_size=batch_size,
        num_workers=num_workers,
        min_shards_needed=min_shards_needed,
    )

    data_ids = []
    for batch in data.dataloader:
        for seq in batch[0]:
            data_ids.append(tuple(seq[:3]))

    assert len(set(data_ids)) == len(data_ids)  # Check no repeats
    # Now reasoning about this, I should consume each partially. Each worker gets 150/20=7.5->7 batches
    target_data_ids = [(0, i, j) for i in range(2) for j in range(70)]
    assert sorted(target_data_ids) == sorted(data_ids)


def test_singleSource_multiWorker_3():
    """
    Asking for 300 samples from 2 workers
    - Should get 3 shards and give 1 to each worker (and throw the last one away b/c need: #shards % #workers == 0)
    - Each worker should see 300 // (2 * 10) = 15 batches
    Output should be the full first 2 shards

    """
    download_dl_test_data()
    make_fake_tarfiles()

    num_samples = 300
    epoch = 0
    next_shard = 0
    batch_size = 10
    min_shards_needed = num_workers = 2

    data = retrieve_dataset_once(
        num_samples,
        SINGLE_SOURCE,
        epoch,
        next_shard,
        None,
        seed=42,
        disable_buffer=True,
        batch_size=batch_size,
        num_workers=num_workers,
        min_shards_needed=min_shards_needed,
    )

    data_ids = []
    for batch in data.dataloader:
        for seq in batch[0]:
            data_ids.append(tuple(seq[:3]))
    target_data_ids = [(0, i, j) for i in range(2) for j in range(100)]  # all of shard 000, 001
    assert sorted(target_data_ids) == sorted(data_ids)


def test_singleSource_multiWorker_4():
    """
    Asking for 256 samples from 2 workers
    - Should get 3 shards and give 1 to each worker (and throw the last one away b/c need: #shards % #workers == 0)
    - Each worker should see 256 // (2 * 10) = 12 batches
    Output should be the full first 2 shards

    """
    download_dl_test_data()
    make_fake_tarfiles()

    num_samples = 256
    epoch = 0
    next_shard = 0
    batch_size = 10

    min_shards_needed = num_workers = 2

    data = retrieve_dataset_once(
        num_samples,
        SINGLE_SOURCE,
        epoch,
        next_shard,
        None,
        seed=42,
        disable_buffer=True,
        batch_size=batch_size,
        num_workers=num_workers,
        min_shards_needed=min_shards_needed,
    )

    data_ids = []
    for batch in data.dataloader:
        for seq in batch[0]:
            data_ids.append(tuple(seq[:3]))
    target_data_ids = [(0, i, j) for i in range(2) for j in range(100)]  # all of shard 000, 001
    assert sorted(target_data_ids) == sorted(data_ids)
