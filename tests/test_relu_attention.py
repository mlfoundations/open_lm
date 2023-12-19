import torch

from open_lm.attention import torch_attn, non_linear_attn, ATTN_NON_LINEARITIES, ATTN_SEQ_SCALARS


def test_non_linear_attn_matches_torch_softmax_attn(threshold=1e-7):
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
            torch_out = torch_attn(queries, keys, values, is_causal=is_causal)
            print(torch_out.shape)

            my_out = non_linear_attn(
                queries, keys, values, attn_non_linearity="softmax", attn_outter_scalar="none", is_causal=is_causal
            )

            assert torch.allclose(
                torch_out, my_out, atol=threshold
            ), "non_linear_attn incorrectly implements softmax attention"


def test_no_failure():
    for nl in ATTN_NON_LINEARITIES:
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
                    non_linear_attn(
                        queries, keys, values, attn_non_linearity=nl, attn_outter_scalar=os, is_causal=is_causal
                    )

    assert True
