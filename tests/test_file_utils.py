""" Test code for file_utils

Mostly just testing the `get_string_for_epoch` method

"""

from open_lm.file_utils import get_string_for_epoch

import pytest
import os
import math
import json
from pathlib import Path
from braceexpand import braceexpand

from tests.utils import download_dl_test_data, make_fake_tarfiles


# ==============================================================
# =           Single Source get_string_for_epoch               =
# ==============================================================

SINGLE_SOURCE = ["tests/assets/source_id_00/manifest.jsonl"]
# ^ 6 files with 100 sequences, 1 file with 66 sequences


@pytest.mark.parametrize("num_samples,starting_point", [(10, 0), (10, 1), (100, 2)])
def test_gsfe_ss_0(num_samples, starting_point):
    """Test case when we want to consume exactly one file, with a single worker"""
    download_dl_test_data()
    make_fake_tarfiles()

    shards_ps, nums_ps, next_ps = get_string_for_epoch(
        num_samples, [starting_point], SINGLE_SOURCE, None, num_workers_per_gpu=1, world_size=1, multi_epoch=False
    )

    assert shards_ps == [os.path.join(os.path.dirname(SINGLE_SOURCE[0]), "%08d.tar" % starting_point)]
    assert nums_ps == ([100] if starting_point < 6 else [66])
    assert next_ps == [starting_point + 1]


@pytest.mark.parametrize("num_samples,starting_point", [(101, 0), (250, 1), (150, 5)])
def test_gsfe_ss_1(num_samples, starting_point):
    """Test case when we want to consume multiple files, with a single worker"""
    download_dl_test_data()
    make_fake_tarfiles()

    shards_ps, nums_ps, next_ps = get_string_for_epoch(
        num_samples, [starting_point], SINGLE_SOURCE, None, num_workers_per_gpu=1, world_size=1, multi_epoch=False
    )

    expected_num_shards = math.ceil(num_samples / 100.0)
    expected_shardlist = ["%08d" % i for i in range(starting_point, starting_point + expected_num_shards)]

    expected_num_samples = expected_num_shards * 100
    if expected_shardlist[-1] == "%08d" % 6:
        expected_num_samples -= 34

    expected_shard_ps = [os.path.join(os.path.dirname(SINGLE_SOURCE[0]), "{%s}.tar" % ",".join(expected_shardlist))]
    expected_nums_ps = [expected_num_samples]
    expected_next_ps = [starting_point + expected_num_shards]

    assert shards_ps == expected_shard_ps
    assert expected_nums_ps == nums_ps
    assert expected_next_ps == next_ps


@pytest.mark.parametrize("num_samples,starting_point", [(1000, 0), (200, 5)])
def test_gsfe_ss_2(num_samples, starting_point):
    """Test case when we want to consume too many samples, with a single worker"""
    download_dl_test_data()
    make_fake_tarfiles()

    try:
        get_string_for_epoch(
            num_samples, [starting_point], SINGLE_SOURCE, None, num_workers_per_gpu=1, world_size=1, multi_epoch=False
        )
    except IndexError:
        assert True


@pytest.mark.parametrize("num_workers,world_size,starting_point", [(10, 1, 0), (5, 2, 0), (3, 3, 0), (3, 1, 5)])
def test_gsfe_ss_3(num_workers, world_size, starting_point):
    """Test case when we want to consume data but have too many workers"""
    download_dl_test_data()
    make_fake_tarfiles()

    try:
        get_string_for_epoch(
            42,
            [starting_point],
            SINGLE_SOURCE,
            None,
            num_workers_per_gpu=num_workers,
            world_size=world_size,
            multi_epoch=False,
        )
    except IndexError:
        assert True


def test_gsfe_ss_4():
    """Test case when we want to consume a small amount of data, but with multiple workers"""
    download_dl_test_data()
    make_fake_tarfiles()

    shards_ps, nums_ps, next_ps = get_string_for_epoch(
        10, [0], SINGLE_SOURCE, None, num_workers_per_gpu=2, world_size=1, multi_epoch=False
    )

    expected_shards_ps = [
        os.path.join(os.path.dirname(SINGLE_SOURCE[0]), "{%s}.tar" % ",".join(["%08d" % i for i in range(2)]))
    ]
    expected_nums_ps = [200]
    expected_next_ps = [2]

    assert shards_ps == expected_shards_ps
    assert expected_nums_ps == nums_ps
    assert expected_next_ps == next_ps


def test_gsfe_ss_5():
    """Test case whne we want to consume a reasonable amount of data, multiple workers"""
    download_dl_test_data()
    make_fake_tarfiles()

    shards_ps, nums_ps, next_ps = get_string_for_epoch(
        400, [0], SINGLE_SOURCE, None, num_workers_per_gpu=2, world_size=1, multi_epoch=False
    )

    expected_shards_ps = [
        os.path.join(os.path.dirname(SINGLE_SOURCE[0]), "{%s}.tar" % ",".join(["%08d" % i for i in range(4)]))
    ]
    expected_nums_ps = [400]
    expected_next_ps = [4]

    assert shards_ps == expected_shards_ps
    assert expected_nums_ps == nums_ps
    assert expected_next_ps == next_ps


def test_gsfe_ss_6():
    """Test case when we want to reasonable data, but uneven modulo num workers"""
    download_dl_test_data()
    make_fake_tarfiles()

    shards_ps, nums_ps, next_ps = get_string_for_epoch(
        450, [0], SINGLE_SOURCE, None, num_workers_per_gpu=2, world_size=1, multi_epoch=False
    )

    expected_shards_ps = [
        os.path.join(os.path.dirname(SINGLE_SOURCE[0]), "{%s}.tar" % ",".join(["%08d" % i for i in range(4)]))
    ]
    expected_nums_ps = [400]
    expected_next_ps = [4]

    assert shards_ps == expected_shards_ps
    assert expected_nums_ps == nums_ps
    assert expected_next_ps == next_ps


@pytest.mark.parametrize("seed", [0, 17, 42])
def test_shard_shuffling(seed):
    """Test whether shard shuffling is deterministic, given a seed."""

    download_dl_test_data()
    make_fake_tarfiles()

    shards_ps_1, _, _ = get_string_for_epoch(
        150, [0], SINGLE_SOURCE, None, num_workers_per_gpu=1, world_size=1, multi_epoch=False, shard_shuffle_seed=seed
    )

    shards_ps_2, _, _ = get_string_for_epoch(
        150, [0], SINGLE_SOURCE, None, num_workers_per_gpu=1, world_size=1, multi_epoch=False, shard_shuffle_seed=seed
    )

    assert shards_ps_1 == shards_ps_2


@pytest.mark.parametrize(
    "test_case",
    [
        (2000, 0, 1, ["00000", "00001"]),  # Easy case.
        (2000, 3, 1, ["00003", "00004", "00005"]),  # Shard 00004 is smaller
        (2000, 7, 1, ["00000", "00001"]),  # At the very end of training shards, this should loop over.
        (
            10000,
            0,
            1,
            "Multiple passes over the dataset did not allow for a valid shard string to be created. Try decreasing the number of tokens between checkpoints.",
        ),  # Not enough data in a single pass.
    ],
)
def test_multi_passes(test_case):
    num_samples, starting_shard, num_workers_per_gpu, expected_outputs = test_case

    download_dl_test_data()

    source_manifest = "tests/assets/source_3/manifest.jsonl"

    try:
        shards_ps, _, _ = get_string_for_epoch(
            num_samples=num_samples,
            starting_points=[starting_shard],
            paths=[source_manifest],
            weights=None,
            num_workers_per_gpu=num_workers_per_gpu,
            world_size=1,
            multi_epoch=True,
            shard_shuffle_seed=None,
        )

        shard_ids = sorted([Path(s).with_suffix("").name for s in braceexpand(shards_ps[0])])
        assert shard_ids == expected_outputs
    except ValueError as e:
        assert str(e) == expected_outputs
