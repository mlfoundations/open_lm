import torch
import pytest

from open_lm.attention import xformers_attn


@pytest.mark.gpu
def test_attention_masking1():
    n, d = 8, 4
    queries = torch.rand((1, n, 1, d)).cuda()
    keys = torch.rand((1, n, 1, d)).cuda()
    values = torch.rand((1, n, 1, d)).cuda()

    attention_mask = torch.ones((1, n)).cuda()
    # Ignore first elements
    attention_mask[:, :4] = 0

    # Run with only last 4 elements of the sequence
    output1 = xformers_attn(
        queries[:, 4:],
        keys[:, 4:],
        values[:, 4:],
        is_causal=True,
        attention_mask=attention_mask[:, 4:],
    )

    # Run with all elements but mask the first 4 elements
    output2 = xformers_attn(queries, keys, values, is_causal=True, attention_mask=attention_mask)

    output3 = xformers_attn(output2, output2, output2, is_causal=True, attention_mask=attention_mask)
    assert not output3.isnan().any()

    assert torch.allclose(output1, output2[:, 4:])
