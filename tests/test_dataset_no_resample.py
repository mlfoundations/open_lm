import pytest
import argparse

from open_lm.main import random_seed
from open_lm.data import get_wds_dataset
from open_lm.file_utils import get_string_for_epoch
from open_lm.params import parse_args


NUM_SAMPLES = 100000
INPUT_PATHS = [
    "/scratch/08002/gsmyrnis/open_lm_tokenized/rpj/manifest.jsonl",
    "/scratch/08002/gsmyrnis/open_lm_tokenized/not_rpj/manifest.jsonl",
]


def retrieve_dataset_once(epoch, weights, seed):
    args = parse_args("")
    train_data_string_per_source, num_samples_per_source = get_string_for_epoch(
        NUM_SAMPLES, epoch, INPUT_PATHS, weights
    )
    args.train_data = train_data_string_per_source
    args.num_workers = 2
    args.batch_size = 2
    args.seed = seed
    args.dataset_resampled = False
    random_seed(seed)
    data = get_wds_dataset(
        args, is_train=True, epoch=epoch, force_num_samples=num_samples_per_source
    )["dataloader"]
    iterator = iter(data)
    item = next(iterator)
    return item[1]


@pytest.mark.parametrize("epoch", [0, 1, 2])
@pytest.mark.parametrize("weights", [[0.5, 0.5], [0.9, 0.1]])
@pytest.mark.parametrize("seed", [0, 1])
def test_deterministic(epoch, weights, seed):
    output1 = retrieve_dataset_once(epoch, weights, seed)
    output2 = retrieve_dataset_once(epoch, weights, seed)
    assert output1 == output2
