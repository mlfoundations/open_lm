""" Test code for file_utils

Mostly just testing the `get_string_for_epoch` method

"""


from open_lm.file_utils import get_string_for_epoch, adjust_samples

import pytest 
import os
import math
import json



# ==============================================================
# =           Single Source get_string_for_epoch               =
# ==============================================================

SINGLE_SOURCE = ["tests/assets/source_id_00/manifest.jsonl"]
# ^ 6 files with 100 sequences, 1 file with 66 sequences

@pytest.mark.parametrize(
    "num_samples,starting_point",
    [(10, 0), (10, 1), (100, 2)]
)
def test_gsfe_ss_0(num_samples, starting_point):
    """ Test case when we want to consume exactly one file, with a single worker """
    shards_ps, nums_ps, next_ps = get_string_for_epoch(num_samples, [starting_point], SINGLE_SOURCE, None,
                                                       num_workers_per_gpu=1, world_size=1, multi_epoch=False)

    assert shards_ps == [os.path.join(os.path.dirname(SINGLE_SOURCE[0]), '%08d.tar' % starting_point)]
    assert nums_ps == ([100] if starting_point < 6 else [66])
    assert next_ps == [starting_point + 1]


@pytest.mark.parametrize(
    "num_samples,starting_point",
    [(101, 0), (250, 1), (150, 5)]
)
def test_gsfe_ss_1(num_samples, starting_point):
    """ Test case when we want to consume multiple files, with a single worker """
    shards_ps, nums_ps, next_ps = get_string_for_epoch(num_samples, [starting_point], SINGLE_SOURCE, None,
                                                       num_workers_per_gpu=1, world_size=1, multi_epoch=False)    


    expected_num_shards = math.ceil(num_samples / 100.0)
    expected_shardlist = ['%08d' % i for i in range(starting_point, starting_point + expected_num_shards)]


    expected_num_samples = expected_num_shards * 100
    if expected_shardlist[-1] == '%08d' % 6:
        expected_num_samples -= 34

    expected_shard_ps = [os.path.join(os.path.dirname(SINGLE_SOURCE[0]), '{%s}.tar' % ','.join(expected_shardlist))]
    expected_nums_ps = [expected_num_samples]
    expected_next_ps = [starting_point + expected_num_shards]

    assert shards_ps == expected_shard_ps
    assert expected_nums_ps == nums_ps
    assert expected_next_ps == next_ps



@pytest.mark.parametrize(
    "num_samples,starting_point",
    [(1000, 0), (200, 5)]
)
def test_gsfe_ss_2(num_samples, starting_point):
    """ Test case when we want to consume too many samples, with a single worker """
    try:
        get_string_for_epoch(num_samples, [starting_point], SINGLE_SOURCE, None,
                                                       num_workers_per_gpu=1, world_size=1, multi_epoch=False)   
    except IndexError:
        assert True


@pytest.mark.parametrize(
    "num_workers,world_size,starting_point",
    [(10, 1, 0), (5, 2, 0), (3, 3, 0), (3, 1, 5)]
)
def test_gsfe_ss_3(num_workers, world_size, starting_point):
    """ Test case when we want to consume data but have too many workers """
    try:
        get_string_for_epoch(42, [starting_point], SINGLE_SOURCE, None,
                                                       num_workers_per_gpu=num_workers, 
                                                       world_size=world_size, multi_epoch=False)   
    except IndexError:
        assert True


def test_gsfe_ss_4():
    """ Test case when we want to consume a small amount of data, but with multiple workers """
    shards_ps, nums_ps, next_ps = get_string_for_epoch(10, [0], SINGLE_SOURCE, None,
                                                       num_workers_per_gpu=2, world_size=1, multi_epoch=False)

    expected_shards_ps = [os.path.join(os.path.dirname(SINGLE_SOURCE[0]), '{%s}.tar' % ','.join(['%08d' % i for i in range(2)]))]
    expected_nums_ps = [200]
    expected_next_ps = [2]

    assert shards_ps == expected_shards_ps
    assert expected_nums_ps == nums_ps
    assert expected_next_ps == next_ps


def test_gsfe_ss_5():
    """ Test case whne we want to consume a reasonable amount of data, multiple workers """
    shards_ps, nums_ps, next_ps = get_string_for_epoch(400, [0], SINGLE_SOURCE, None,
                                                       num_workers_per_gpu=2, world_size=1, multi_epoch=False)

    expected_shards_ps = [os.path.join(os.path.dirname(SINGLE_SOURCE[0]), '{%s}.tar' % ','.join(['%08d' % i for i in range(4)]))]
    expected_nums_ps = [400]
    expected_next_ps = [4]

    assert shards_ps == expected_shards_ps
    assert expected_nums_ps == nums_ps
    assert expected_next_ps == next_ps



def test_gsfe_ss_6():
    """ Test case when we want to reasonable data, but uneven modulo num workers """
    shards_ps, nums_ps, next_ps = get_string_for_epoch(450, [0], SINGLE_SOURCE, None,
                                                       num_workers_per_gpu=2, world_size=1, multi_epoch=False)

    expected_shards_ps = [os.path.join(os.path.dirname(SINGLE_SOURCE[0]), '{%s}.tar' % ','.join(['%08d' % i for i in range(4)]))]
    expected_nums_ps = [400]
    expected_next_ps = [4]

    assert shards_ps == expected_shards_ps
    assert expected_nums_ps == nums_ps
    assert expected_next_ps == next_ps



# ==========================================================
# =           Multi Source get_string_for_epoch            =
# ==========================================================

def test_gsfe_ms_0():
    try:
        get_string_for_epoch([10, 10], [0, 0], SINGLE_SOURCE + SINGLE_SOURCE, None, 1, 1, multi_epoch=False)
    except ValueError as err:
        assert str(err) == "Multiple sources are not supported fully as of now"



# ==================================================
# =           Adjust Samples                       =
# ==================================================


def test_adjust_samples_0():
    """ Standard use case of adjust samples when everything should be balanced """
    shard_list = ['%08d' % i for i in range(1, 5)]
    manifest = [json.loads(_) for _ in open(SINGLE_SOURCE[0], 'r').readlines()]
    starting_idx = 1
    num_samples = adjust_samples(shard_list, manifest, starting_idx, num_workers_per_gpu=2, world_size=2)
    assert num_samples == 400 # 100 is the minimum, 4 workers

def test_adjust_samples_1():
    """ Where we use the final two nodes of the real manifest """
    shard_list = ['%08d' % i for i in range(5 ,7)]
    manifest = [json.loads(_) for _ in open(SINGLE_SOURCE[0], 'r').readlines()]
    starting_idx = 5
    num_workers_per_gpu = 2
    world_size = 1
    num_samples = adjust_samples(shard_list, manifest, starting_idx, num_workers_per_gpu=2, world_size=1)
    assert num_samples == 66 * 2


def test_adjust_samples_2():
    """ Completely manufactured thing with pathological examples """
    shard_list = list(range(2 * 3 * 4 * 5))
    manifest = [{'num_sequences': 77} for i in range(2 * 3 * 4 * 5)]
    manifest[12]['num_sequences'] = 42

    # Each gets 5 shards, with the smallest being [77, 77, 77, 77, 42]
    num_samples = adjust_samples(shard_list, manifest, 0, num_workers_per_gpu=6, world_size=4)

    assert num_samples == 24 * (77 * 4 + 42)




