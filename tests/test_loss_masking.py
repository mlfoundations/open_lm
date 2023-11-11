from torch import Tensor, equal

from open_lm.train import replace_before_tok, replace_tok, sample_chunk
from tests.shared import create_train_fixtures


def create_tok_fixtures():
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

    return special_token_left, special_token_individual, batched_tokens


def test_create_replace_before_tok():
    special_token, _, batched_tokens = create_tok_fixtures()

    # masking including special token
    target1 = Tensor(
        [
            [-100, -100, -100, -100],
            [-100, -100, -100, 2],
            [-100, -100, 1, 2],
            [-100, 0, 1, 2],
            [0, 1, 2, 3],
            [-100, -100, -100, 1],
        ]
    )
    assert equal(replace_before_tok(batched_tokens, special_token, -100), target1)

    # masking excluding special token
    target2 = Tensor(
        [
            [-100, -100, -100, 42],
            [-100, -100, 42, 2],
            [-100, 42, 1, 2],
            [42, 0, 1, 2],
            [0, 1, 2, 3],
            [42, -100, 42, 1],
        ]
    )

    assert equal(replace_before_tok(batched_tokens, special_token, -100, excusive=True), target2)

    target3 = Tensor(
        [
            [0, 1, 2, -100],
            [0, 1, -100, 2],
            [0, -100, 1, 2],
            [-100, 0, 1, 2],
            [0, 1, 2, 3],
            [-100, 0, -100, 1],
        ]
    )
    assert equal(replace_tok(batched_tokens, special_token, -100), target3)


def test_target_mask_left():
    args, _, _, _, _, loss = create_train_fixtures()
    special_token, _, batched_tokens = create_tok_fixtures()

    args.target_mask_left = special_token
    args.seq_len = 3

    _, target = sample_chunk(batched_tokens, args)

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

    assert equal(target, real_target)


def test_target_mask_individual():
    args, _, _, _, _, loss = create_train_fixtures()
    _, special_token, batched_tokens = create_tok_fixtures()

    args.target_mask_individual = special_token
    args.seq_len = 3

    _, target = sample_chunk(batched_tokens, args)

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

    assert equal(target, real_target)


def test_target_mask_left_individual():
    args, _, _, _, _, loss = create_train_fixtures()
    special_token_left, special_token_individual, batched_tokens = create_tok_fixtures()

    args.target_mask_left = special_token_left
    args.target_mask_individual = special_token_individual
    args.seq_len = 3

    _, target = sample_chunk(batched_tokens, args)

    real_target = Tensor(
        [
            [-100, -100, -100],
            [-100, -100, -100],
            [-100, 1, -100],
            [0, 1, -100],
            [1, -100, 3],
            [-100, -100, 1],
        ]
    )

    assert equal(target, real_target)
