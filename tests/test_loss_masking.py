import json

from torch import Tensor, equal

from open_lm.data import sample_chunk
from tests.shared import create_train_fixtures


def test_target_mask_left():
    args, _, _, _, _, _ = create_train_fixtures()

    special_token_left = 42
    batched_tokens = Tensor(
        [
            [0, 1, 2, 42],
            [0, 1, 42, 2],
            [0, 42, 1, 2],
            [42, 0, 1, 2],
            [0, 1, 2, 3],
            [42, 0, 42, 1],
        ]
    )

    args.target_mask_left = special_token_left
    args.seq_len = 3

    input, target = sample_chunk(batched_tokens, args)

    real_input = Tensor(
        [
            [0, 1, 2],
            [0, 1, 42],
            [0, 42, 1],
            [42, 0, 1],
            [0, 1, 2],
            [42, 0, 42],
        ]
    )

    real_target = Tensor(
        [
            [-100, -100, -100],
            [-100, -100, 2],
            [-100, 1, 2],
            [0, 1, 2],
            [1, 2, 3],
            [-100, -100, 1],
        ]
    )

    assert equal(input, real_input)
    assert equal(target, real_target)


def test_target_mask_individual():
    args, _, _, _, _, _ = create_train_fixtures()
    special_token_individual = 2

    batched_tokens = Tensor(
        [
            [0, 1, 2, 42],
            [0, 1, 42, 2],
            [0, 42, 1, 2],
            [42, 0, 1, 2],
            [0, 1, 2, 3],
            [42, 0, 42, 1],
        ]
    )

    args.target_mask_individual = special_token_individual
    args.seq_len = 3

    input, target = sample_chunk(batched_tokens, args)

    real_input = Tensor(
        [
            [0, 1, 2],
            [0, 1, 42],
            [0, 42, 1],
            [42, 0, 1],
            [0, 1, 2],
            [42, 0, 42],
        ]
    )

    real_target = Tensor(
        [
            [1, -100, 42],
            [1, 42, -100],
            [42, 1, -100],
            [0, 1, -100],
            [1, -100, 3],
            [0, 42, 1],
        ]
    )

    assert equal(input, real_input)
    assert equal(target, real_target)


def test_target_mask_left_individual():
    args, _, _, _, _, _ = create_train_fixtures()
    special_token_left = 42
    special_token_individual = 2

    batched_tokens = Tensor(
        [
            [0, 1, 2, 42],
            [0, 1, 42, 2],
            [0, 42, 1, 2],
            [42, 0, 1, 2],
            [0, 1, 2, 3],
            [42, 0, 42, 1],
            [42, 42, 42, 1],
        ]
    )

    args.target_mask_left = special_token_left
    args.target_mask_individual = special_token_individual
    args.seq_len = 3

    input, target = sample_chunk(batched_tokens, args)

    real_input = Tensor(
        [
            [0, 1, 2],
            [0, 1, 42],
            [0, 42, 1],
            [42, 0, 1],
            [0, 1, 2],
            [42, 0, 42],
            [42, 42, 42],
        ]
    )

    real_target = Tensor(
        [
            [-100, -100, -100],
            [-100, -100, -100],
            [-100, 1, -100],
            [0, 1, -100],
            [1, -100, 3],
            [-100, -100, 1],
            [-100, -100, 1],
        ]
    )

    assert equal(input, real_input)
    assert equal(target, real_target)


def test_target_mask_left_individual_squash():
    args, _, _, _, _, _ = create_train_fixtures()
    special_token_left = 42
    special_token_individual = 2

    batched_tokens = Tensor(
        [
            [0, 1, 2, 42],
            [0, 1, 42, 2],
            [0, 42, 1, 2],
            [42, 0, 1, 2],
            [0, 1, 2, 3],
            [42, 0, 42, 1],
        ]
    )

    args.target_mask_left = special_token_left
    args.target_mask_individual = special_token_individual
    args.seq_len = 3
    args.squash_mask_left = True

    input, target = sample_chunk(batched_tokens, args)

    real_input = Tensor(
        [
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
            [0, 2, 2],
        ]
    )

    real_target = Tensor(
        [
            [-100, -100, -100],
            [-100, -100, -100],
            [1, -100, -100],
            [0, 1, -100],
            [1, -100, 3],
            [-100, 1, -100],
        ]
    )

    assert equal(input, real_input)
    assert equal(target, real_target)


def test_target_mask_left_individual_squash_real_data():
    data = None
    with open("tests/assets/2049_span_pad.json", "r") as f:
        data = json.load(f)

    args, _, _, _, _, _ = create_train_fixtures()

    args.target_mask_left = 50300
    args.target_mask_individual = 50400
    args.seq_len = 2048
    args.squash_mask_left = True

    # skip the pad left token
    real_input = data[:65] + data[66:72]

    # right pad with the target_mask_individual token
    real_input += (2048 - len(real_input)) * [50400]
    real_input = Tensor([real_input])

    input, target = sample_chunk(Tensor([data]), args)
    # print(input.shape)

    # skip the pad left token and mask out the prefix with -100
    real_target = len(data[1:65]) * [-100] + data[66:72]

    # right pad with the ignore xent token (-100)
    real_target += (2048 - len(real_target)) * [-100]
    real_target = Tensor([real_target])

    assert equal(input, real_input)
    assert equal(target, real_target)
