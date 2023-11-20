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

SEQ_LEN = 2048
INPUT_PATHS = [
    "tests/assets/source_id_00/manifest.jsonl",
    "tests/assets/source_id_01/manifest.jsonl",
]
SINGLE_SOURCE = [INPUT_PATHS[0]]

TOTAL_SEQ_COUNT = 666 * 2
SINGLE_SEQ_COUNT = 666

""" TEST STRATEGY:
Tests to just handle the edge cases in sampling WITHOUT replacement.

We are just going to cover the path where we have:
- Exactly one source 
- Never request more data than we have available

And generally we want to assert that:
- We never see any repeated data 
- We see exactly the right number of samples as desired (some math here)

"""



# ===============================================
# =                Helpers                      =
# ===============================================



def get_dataset_size(paths, count_tokens=False):
    """ Computes dataset size (given list of paths to the manifest)
        Returns number of sequences by default, but can multiply by 2048 if you ask it to =)
    """
    total_seq_count = 0
    for filename in paths:
        for line in open(filename, 'r').readlines():
            total_seq_count += json.loads(line)['num_chunks']
    return total_seq_count * 2048 if count_tokens else total_seq_count



def retrieve_dataset_once(total_seqs, paths, epoch, next_shard, weights, seed, disable_buffer, batch_size, num_workers, 
                          min_shards_needed=1):
    """ Returns the output of get_wds_dataset -- not dataloader or nothing fancy
    ONLY WORKS FOR A SINGLE SOURCE
    """

    assert len(paths) == 1
    args = parse_args("")
    random_seed(seed)


    train_data_string_per_source, num_seqs_per_source, _ = get_string_for_epoch(
        total_seqs, [next_shard], paths, weights, min_shards_needed, world_size=1)

    args.train_num_samples = total_seqs
    args.train_data = train_data_string_per_source
    args.num_workers = num_workers
    args.batch_size = batch_size
    args.seed = seed
    args.dataset_resampled = False
    args.disable_buffer = disable_buffer
    args.vocab_size = _MODEL_CONFIGS[args.model]["vocab_size"]
    args.seq_len = _MODEL_CONFIGS[args.model]["seq_len"]
    args.world_size = 1 
    
    data = get_wds_dataset(args, is_train=True, epoch=epoch, force_num_samples=[total_seqs])
    return data



# ======================================================
# =           Single Source Test Cases                 =
# ======================================================

@pytest.mark.parametrize('num_samples,next_shard,batch_size,min_shards_needed',
                         [(10, 0, 1, 1),
                          (100, 0, 25, 2),
                          (150, 1, 50, 4),
                          (666, 0, 111, 3)])
def test_singleSource_singleWorker_perfectBatch(num_samples, next_shard, batch_size, min_shards_needed):
    """ Cases where:
        - Only a single source
        - Only a single worker 
        - The batch_size perfectly divides the number of samples requested

    Should see exactly num_samples at the end, and no repeats
    """
    data = retrieve_dataset_once(num_samples, SINGLE_SOURCE, 0, next_shard, None, 
                                 seed=42, disable_buffer=True, batch_size=batch_size,
                                 num_workers=1, min_shards_needed=min_shards_needed)

    data_ids = []
    for batch in data.dataloader:
        for seq in batch[0]:
            data_ids.append(tuple(seq[:3]))

    assert len(set(data_ids)) == len(data_ids) == num_samples# Checking no repeats and exact




@pytest.mark.parametrize('num_samples,next_shard,batch_size, min_shards_needed',
                         [(50, 0, 7, 1),
                          (250, 2, 13, 4),
                          (666, 0, 13, 1)])
def test_singleSource_singleWorker_imperfectBatch(num_samples, next_shard, batch_size, min_shards_needed):
    """ Cases where:
        - Only a single source
        - Only a single worker
        - Batch size does NOT divide the number of samples

    Should see no repeats, but the greatest multiple of batch_size <= num_samples
    """
    data = retrieve_dataset_once(num_samples, SINGLE_SOURCE, 0, next_shard, None, 
                                 seed=42, disable_buffer=True, batch_size=batch_size,
                                 num_workers=1, min_shards_needed=min_shards_needed)

    data_ids = []
    for batch in data.dataloader:
        for seq in batch[0]:
            data_ids.append(tuple(seq[:3]))

    assert len(set(data_ids)) == len(data_ids) # Check no repeats
    assert len(data_ids) == batch_size * (num_samples // batch_size) #




def test_singleSource_multiWorker_perfectBatch_0():
    """ 
    Not using pytest.mark.parametrize because I need to be careful about the reasoning
    Start with a simple case:
    - 2 workers, each should consume (fully) one tar file
    """
    num_samples = 200
    epoch = 0
    next_shard = 0
    batch_size = 10
    min_shards_needed = num_workers = 2

    data = retrieve_dataset_once(num_samples, SINGLE_SOURCE, epoch, next_shard, None, 
                                 seed=42, disable_buffer=True, batch_size=batch_size,
                                 num_workers=num_workers, min_shards_needed=min_shards_needed)

    data_ids = []
    for batch in data.dataloader:
        for seq in batch[0]:
            data_ids.append(tuple(seq[:3]))

    assert len(set(data_ids)) == len(data_ids) # Check no repeats
    # Now reasoning about this, I should consume each fully


    target_data_ids = [(0, i, j) for i in range(2) for j in range(100)]
    assert sorted(target_data_ids) == sorted(data_ids)



def test_singleSource_multiWorker_0():
    """ 
    Not using pytest.mark.parametrize because I need to be careful about the reasoning
    - 2 workers, each should consume (fully) one tar file
    """
    num_samples = 200
    epoch = 0
    next_shard = 0
    batch_size = 10
    min_shards_needed = num_workers = 2

    data = retrieve_dataset_once(num_samples, SINGLE_SOURCE, epoch, next_shard, None, 
                                 seed=42, disable_buffer=True, batch_size=batch_size,
                                 num_workers=num_workers, min_shards_needed=min_shards_needed)

    data_ids = []
    for batch in data.dataloader:
        for seq in batch[0]:
            data_ids.append(tuple(seq[:3]))

    assert len(set(data_ids)) == len(data_ids) # Check no repeats
    # Now reasoning about this, I should consume each fully

    target_data_ids = [(0, i, j) for i in range(2) for j in range(100)] # All of chunk 0000, chunk 0001
    assert sorted(target_data_ids) == sorted(data_ids)


def test_singleSource_multiWorker_1():
    """ 
    Not using pytest.mark.parametrize because I need to be careful about the reasoning
    - 2 workers, each should consume (partially) one tar file (but no batch weirdness)
    """
    num_samples = 150
    epoch = 0
    next_shard = 0
    batch_size = 5
    min_shards_needed = num_workers = 2

    data = retrieve_dataset_once(num_samples, SINGLE_SOURCE, epoch, next_shard, None, 
                                 seed=42, disable_buffer=True, batch_size=batch_size,
                                 num_workers=num_workers, min_shards_needed=min_shards_needed)

    data_ids = []
    for batch in data.dataloader:
        for seq in batch[0]:
            data_ids.append(tuple(seq[:3]))

    assert len(set(data_ids)) == len(data_ids) # Check no repeats

    target_data_ids = [(0, i, j) for i in range(2) for j in range(75)] # 3/4 of chunk 0000, chunk 0001
    assert sorted(target_data_ids) == sorted(data_ids)



def test_singleSource_multiWorker_2():
    """ 
    Not using pytest.mark.parametrize because I need to be careful about the reasoning
    - 2 workers, each should consume (partially) one tar file (but yes batch weirdness)
    """
    num_samples = 150
    epoch = 0
    next_shard = 0
    batch_size = 10
    min_shards_needed = num_workers = 2

    data = retrieve_dataset_once(num_samples, SINGLE_SOURCE, epoch, next_shard, None, 
                                 seed=42, disable_buffer=True, batch_size=batch_size,
                                 num_workers=num_workers, min_shards_needed=min_shards_needed)

    data_ids = []
    for batch in data.dataloader:
        for seq in batch[0]:
            data_ids.append(tuple(seq[:3]))

    assert len(set(data_ids)) == len(data_ids) # Check no repeats
    # Now reasoning about this, I should consume each partially. Each worker gets 150/20=7.5->7 batches
    target_data_ids = [(0, i, j) for i in range(2) for j in range(70)] 
    assert sorted(target_data_ids) == sorted(data_ids)


def test_singleSource_multiWorker_3():
    """ 
    Not using pytest.mark.parametrize because I need to be careful about the reasoning

    - 2 workers, one worker should get 1 tarfile, the other worker should get 2 tarfiles
    - no batch weirdness

    So if I ask for 300 samples, but each shard has 100, then I need 3 shards
    But then if I distribute across 2 workers, i should get 
    worker_0:= [shard_0000], worker_1:=[shard_0001, shard_0002] (or some such thing)
    and then each worker should ask to see 300/(batch_size * num_workers) = 15 batches
    so worker_0 exhausts its supply
    and worker_1 only sees 15 batches
    """
    num_samples = 300
    epoch = 0
    next_shard = 0
    batch_size = 10
    min_shards_needed = num_workers = 2

    data = retrieve_dataset_once(num_samples, SINGLE_SOURCE, epoch, next_shard, None, 
                                 seed=42, disable_buffer=True, batch_size=batch_size,
                                 num_workers=num_workers, min_shards_needed=min_shards_needed)

    data_ids = []
    for batch in data.dataloader:
        for seq in batch[0]:
            data_ids.append(tuple(seq[:3]))

    assert len(set(data_ids)) == len(data_ids) # Check no repeats
    assert len(data_ids) == 250


def test_singleSource_multiWorker_4():
    """
    - 2 workers, one worker gets one tarfile, the other gets 2 tarfiles
    - yes batch weirdness

    I ask for 256 samples, so I should get two workers with 1,2 tars each
    ??? 
    TALKING TO GEORGE ABOUT THE PROPER BEHAVIOR HERE
    """
    num_samples = 256
    epoch = 0
    next_shard = 0
    batch_size = 10

    min_shards_needed = num_workers = 2

    data = retrieve_dataset_once(num_samples, SINGLE_SOURCE, epoch, next_shard, None, 
                                 seed=42, disable_buffer=True, batch_size=batch_size,
                                 num_workers=num_workers, min_shards_needed=min_shards_needed)

    data_ids = []
    for batch in data.dataloader:
        for seq in batch[0]:
            data_ids.append(tuple(seq[:3]))

    assert len(set(data_ids)) == len(data_ids) # Check no repeats
    assert len(data_ids) == 250    


# ====================================================================
# =           Tests when num_train_samples == dataset_size           =
# ====================================================================


@pytest.mark.parametrize('batch_size,num_workers',
                         [(1, 1),
                          (2, 3),
                          (111, 6),
                         ])
def test_wo_resample_exact_a(batch_size, num_workers):
    """
    Case a: we have batch_size/num_workers such that we cover the dataset exactly once 
    i.e, TOTAL_SEQ_COUNT is exactly divisible by number of workers

    Assert:
    - that we get exactly TOTAL_SEQ_COUNT examples
    - that we get the exact dataset that we requested (by ids) 
    
    """
    make_fake_tarfiles()
    seq_count = TOTAL_SEQ_COUNT
    data = retrieve_dataset_once(seq_count, INPUT_PATHS, 0, 0, [0.5, 0.5], 1234, True,
                                 batch_size, num_workers, min_shards_needed=1)
    data_ids = []
    for batch in data.dataloader:
        for seq in batch[0]:
            data_ids.append(tuple(seq[:3]))

    expected_ids = []
    for i in range(seq_count):
        expected_ids.append((i  % 2, (i //2) // 100, (i // 2) % 100))

    assert sorted(expected_ids) == sorted(data_ids)



@pytest.mark.parametrize('batch_size,num_workers',
                         [(7, 1)])
def test_wo_resample_exact_b(batch_size, num_workers):
    """
    Case b: we have batch_size/num_workers such that we cover:
        - up until the last batch: the dataset less than exactly once
        - all batch: some parts of the dataset once or twice

    Assert:
    - max frequency all-but-lasts is 1
    - total frequency of all-but-lasts is greatest bsz*numworkers multiple <= seq_count

    - max frequency of all-ids is 2
    - total frequency of all-ids is smallest bsz*numworkers multiple >= seq_count
    - total number of keys is TOTAL_SEQ_COUNT


    - total set of ids is all ids
    """
    make_fake_tarfiles()
    seq_count = TOTAL_SEQ_COUNT
    total_workers = batch_size * num_workers * len(INPUT_PATHS)
    data = retrieve_dataset_once(seq_count, INPUT_PATHS, 0, 0, [0.5, 0.5], 1234, True,
                                 batch_size, num_workers, min_shards_needed=1)


    all_batches = list(data.dataloader)
    all_ids = defaultdict(int)
    all_but_last_ids = defaultdict(int)
    for i in range(len(all_batches)):
        batch = all_batches[i]
        for seq in batch[0]:
            _id = tuple(seq[:3])
            all_ids[_id] += 1
            
    for i in range(2):
        for j in range(7):
            for k in range(100):
                if j == 0 and k < 6:
                    # First 6 samples per source should have been seen twice
                    assert all_ids[(i,j,k)] == 2
                elif (i,j,k) in all_ids:
                    assert all_ids[(i,j,k)] == 1


    expected_ids_per_source = [[], []]
    for i in range(seq_count):
        expected_ids_per_source[i % 2].append((i  % 2, (i //2) // 100, (i // 2) % 100))

    # Asserts on all ids
    assert max(all_ids.values()) == 2
    assert sum(all_ids.values()) == seq_count + total_workers - (seq_count % total_workers)
    assert len(all_ids) == seq_count

    # And actually get the set of all ids
    assert sorted(list(all_ids.keys())) == expected_ids_per_source[0] + expected_ids_per_source[1]

# ====================================================================
# =           Tests when num_train_samples < dataset_size            =
# ====================================================================


def test_wo_resample_smallTrain():
    assert True



# ====================================================================
# =           Tests when num_train_samples > dataset_size            =
# ====================================================================


def test_wo_resample_bigTrain():
    assert True


# ====================================================================
# =                      Tests for file_utils                        =
# ====================================================================


def test_adjust_samples():
    """Test for adjust_samples.

    This test should verify that adjust_samples returns the same number of samples as when splitting with
    wds.split_by_node and wds.split_by_worker.
    """
    assert True
