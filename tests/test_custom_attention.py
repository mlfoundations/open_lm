import torch

from open_lm.attention import torch_attn, custom_attn, xformers_attn, ATTN_ACTIVATIONS, ATTN_SEQ_SCALARS
from open_lm.model import SwiGLUTorch
from open_lm.precision import get_autocast
from xformers.ops import SwiGLU


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
                attn_activation="softmax",
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
                xformers_out = xformers_attn(queries.cuda(), keys.cuda(), values.cuda(), is_causal=is_causal)
                my_out = custom_attn(
                    queries.cuda(),
                    keys.cuda(),
                    values.cuda(),
                    attn_activation="softmax",
                    attn_seq_scalar="none",
                    alpha=1.0,
                    is_causal=is_causal,
                )

                assert torch.allclose(
                    torch_out, my_out, atol=threshold
                ), "custom_attn incorrectly implements softmax attention"

                assert torch.allclose(
                    xformers_out, my_out, atol=threshold
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
                        attn_activation=nl,
                        attn_seq_scalar=os,
                        alpha=1.0,
                        is_causal=is_causal,
                    )

    assert True


def test_swiglu_torch(threshold=1e-7):
    bsz = 5
    in_feats = 10
    hidden_feats = 30
    out_feats = 10
    num_tries = 5

    xops_swiglu = SwiGLU(in_features=in_feats, hidden_features=hidden_feats, out_features=out_feats)
    torch_swiglu = SwiGLUTorch(in_dim=in_feats, hidden_dim=hidden_feats, out_dim=out_feats)

    # Copy state dict from one swiglu to the other so that they have the same weights

    state_dict = xops_swiglu.state_dict()
    new_state_dict = {
        "w12.weight": state_dict["w12.weight"],
        "w3.weight": state_dict["w3.weight"],
        "w12.bias": state_dict["w12.bias"],
        "w3.bias": state_dict["w3.bias"],
    }
    torch_swiglu.load_state_dict(new_state_dict)

    with torch.no_grad():
        for _ in range(num_tries):
            random_in = torch.rand((bsz, in_feats))
            torch_out = torch_swiglu(random_in)
            xops_out = xops_swiglu(random_in)
            assert torch.allclose(torch_out, xops_out, atol=threshold)
