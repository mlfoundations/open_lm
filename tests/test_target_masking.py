from torch import Tensor, equal
from open_lm.train import replace_before_tok, replace_tok


def create_replace_before_tok_fixtures():
    special_token = 42

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

    return special_token, batched_tokens


def test_create_replace_before_tok():
    special_token, batched_tokens = create_replace_before_tok_fixtures()

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

    assert equal(
        replace_before_tok(batched_tokens, special_token, -100, excusive=True), target2
    )

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
