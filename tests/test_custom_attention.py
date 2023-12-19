import torch

from open_lm.attention import torch_attn, custom_attn, xformers_attn, ATTN_ACTIVATIONS, ATTN_SEQ_SCALARS
from open_lm.precision import get_autocast


def test_custom_attn_matches_softmax_attn(threshold=1e-7):
    for bs, q_seq_len, k_seq_len, h, d in [
        [10, 1024, 2048, 8, 128],
        [10, 2048, 1024, 8, 128],
        [10, 2048, 2048, 8, 128],
        [1, 1024, 2048, 8, 128],
    ]:
        queries = torch.rand(bs, q_seq_len, h, d)
        keys = torch.rand(bs, k_seq_len, h, d)
        values = torch.rand(bs, k_seq_len, h, d)

        for is_causal in [True, False]:
            torch_out = torch_attn(queries.cpu(), keys.cpu(), values.cpu(), is_causal=is_causal)

            my_out = custom_attn(
                queries.cpu(),
                keys.cpu(),
                values.cpu(),
                attn_non_linearity="softmax",
                attn_seq_scalar="none",
                alpha=1.0,
                is_causal=is_causal,
            )

            assert torch.allclose(
                torch_out, my_out, atol=threshold
            ), "custom_attn incorrectly implements softmax attention"

            if torch.cuda.is_available():
                # also test xformers attention
                torch_out = torch_attn(queries.cuda(), keys.cuda(), values.cuda(), is_causal=is_causal)
                # xformers_out = xformers_attn(queries.cuda(), keys.cuda(), values.cuda(), is_causal=is_causal)
                my_out = custom_attn(
                    queries.cuda(),
                    keys.cuda(),
                    values.cuda(),
                    attn_non_linearity="softmax",
                    attn_seq_scalar="none",
                    alpha=1.0,
                    is_causal=is_causal,
                )

                assert torch.allclose(
                    torch_out, my_out, atol=threshold
                ), "custom_attn incorrectly implements softmax attention"


def test_no_failure():
    for nl in ATTN_ACTIVATIONS:
        for os in ATTN_SEQ_SCALARS:
            for bs, q_seq_len, k_seq_len, h, d in [
                [2, 64, 64, 1, 32],
                [2, 64, 16, 1, 32],
                [2, 16, 64, 1, 32],
            ]:
                queries = torch.rand(bs, q_seq_len, h, d)
                keys = torch.rand(bs, k_seq_len, h, d)
                values = torch.rand(bs, k_seq_len, h, d)

                for is_causal in [True, False]:
                    custom_attn(
                        queries,
                        keys,
                        values,
                        attn_non_linearity=nl,
                        attn_seq_scalar=os,
                        alpha=1.0,
                        is_causal=is_causal,
                    )

    assert True


# def test_custom_attn_matches_softmax_attn_amp(threshold=1e-7):
#     for p in ["amp_bf16", "amp"]:
#         autocast = get_autocast(p)
#         with autocast():
#             test_custom_attn_matches_softmax_attn(threshold=threshold)


# def test_no_failure_amp():
#     for p in ["amp", "amp_bf16"]:
#         autocast = get_autocast(p)
#         with autocast():
#             test_no_failure()
