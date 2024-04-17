import torch
import pytest

from open_lm.attention import xformers_attn, torch_attn


@pytest.mark.gpu
def test_attention_masking_xformers():
    n, d = 8, 4
    queries = torch.rand((1, n, 1, d)).cuda()
    keys = torch.rand((1, n, 1, d)).cuda()
    values = torch.rand((1, n, 1, d)).cuda()

    attention_mask = torch.ones((1, n)).cuda()
    # Ignore first elements
    attention_mask[:, :4] = 0

    # Run with only last 4 elements of the sequence, no attention mask
    output_no_mask = xformers_attn(
        queries[:, 4:],
        keys[:, 4:],
        values[:, 4:],
        is_causal=True,
        attention_mask=None,
    )

    # Run with only last 4 elements of the sequence
    output_dummy_mask = xformers_attn(
        queries[:, 4:],
        keys[:, 4:],
        values[:, 4:],
        is_causal=True,
        attention_mask=attention_mask[:, 4:],
    )

    # Run with all elements but mask the first 4 elements
    output_mask_initial = xformers_attn(queries, keys, values, is_causal=True, attention_mask=attention_mask)

    # Run with the output of attention again and ensure it looks good (e.g., we don't run into NaNs. This happened in
    # initial implementations where the output had NaNs for certain types of masks.
    output3 = xformers_attn(
        output_mask_initial, output_mask_initial, output_mask_initial, is_causal=True, attention_mask=attention_mask
    )
    assert not output3.isnan().any()

    assert torch.allclose(output_no_mask, output_dummy_mask)
    assert torch.allclose(output_no_mask, output_mask_initial[:, 4:])


def test_attention_masking_torchattn():
    n, d = 8, 4
    queries = torch.rand((1, n, 1, d))
    keys = torch.rand((1, n, 1, d))
    values = torch.rand((1, n, 1, d))

    attention_mask = torch.ones((1, n))
    # Ignore first elements
    attention_mask[:, :4] = 0

    # Run with only last 4 elements of the sequence, no attention mask
    output_no_mask = torch_attn(
        queries[:, 4:],
        keys[:, 4:],
        values[:, 4:],
        is_causal=True,
        attention_mask=None,
    )

    # Run with only last 4 elements of the sequence
    output_dummy_mask = torch_attn(
        queries[:, 4:],
        keys[:, 4:],
        values[:, 4:],
        is_causal=True,
        attention_mask=attention_mask[:, 4:],
    )

    # Run with all elements but mask the first 4 elements
    output_mask_initial = torch_attn(queries, keys, values, is_causal=True, attention_mask=attention_mask)

    output3 = torch_attn(
        output_mask_initial, output_mask_initial, output_mask_initial, is_causal=True, attention_mask=attention_mask
    )
    assert not output3.isnan().any()

    assert torch.allclose(output_no_mask, output_dummy_mask)
    assert torch.allclose(output_no_mask, output_mask_initial[:, 4:])


@pytest.mark.gpu
def test_attention_masking_torchattn_vs_xformers():
    n, d = 8, 4
    queries = torch.rand((1, n, 1, d))
    keys = torch.rand((1, n, 1, d))
    values = torch.rand((1, n, 1, d))

    attention_mask = torch.ones((1, n))
    # Ignore first elements
    attention_mask[:, :4] = 0

    # Run with only last 4 elements of the sequence, no attention mask
    output_no_mask_torch = torch_attn(
        queries[:, 4:],
        keys[:, 4:],
        values[:, 4:],
        is_causal=True,
        attention_mask=None,
    )

    # Run with only last 4 elements of the sequence
    output_dummy_mask_torch = torch_attn(
        queries[:, 4:],
        keys[:, 4:],
        values[:, 4:],
        is_causal=True,
        attention_mask=attention_mask[:, 4:].clone(),
    )

    # Run with all elements but mask the first 4 elements
    output_mask_initial_torch = torch_attn(queries, keys, values, is_causal=True, attention_mask=attention_mask.clone())
    output_mask_initial_fewq_torch = torch_attn(
        queries[:, : n - 2], keys, values, is_causal=True, attention_mask=attention_mask.clone()
    )

    output3_torch = torch_attn(
        output_mask_initial_torch,
        output_mask_initial_torch,
        output_mask_initial_torch,
        is_causal=True,
        attention_mask=attention_mask.clone(),
    )
    assert not output3_torch.isnan().any()

    queries = queries.cuda()
    keys = keys.cuda()
    values = values.cuda()
    attention_mask = attention_mask.cuda()

    # Run with only last 4 elements of the sequence, no attention mask
    output_no_mask_xformers = xformers_attn(
        queries[:, 4:],
        keys[:, 4:],
        values[:, 4:],
        is_causal=True,
        attention_mask=None,
    )

    # Run with only last 4 elements of the sequence
    output_dummy_mask_xformers = xformers_attn(
        queries[:, 4:],
        keys[:, 4:],
        values[:, 4:],
        is_causal=True,
        attention_mask=attention_mask[:, 4:].clone(),
    )

    # Run with all elements but mask the first 4 elements
    output_mask_initial_xformers = xformers_attn(
        queries, keys, values, is_causal=True, attention_mask=attention_mask.clone()
    )
    output_mask_initial_fewq_xformers = xformers_attn(
        queries[:, : n - 2], keys, values, is_causal=True, attention_mask=attention_mask.clone()
    )

    output3_xformers = xformers_attn(
        output_mask_initial_xformers,
        output_mask_initial_xformers,
        output_mask_initial_xformers,
        is_causal=True,
        attention_mask=attention_mask.clone(),
    )
    assert not output3_xformers.isnan().any()

    assert torch.allclose(output_no_mask_torch, output_no_mask_xformers.cpu())
    assert torch.allclose(output_dummy_mask_torch, output_dummy_mask_xformers.cpu())
    assert torch.allclose(output_mask_initial_torch, output_mask_initial_xformers.cpu())
    assert torch.allclose(output3_torch, output3_xformers.cpu())
    assert torch.allclose(output_mask_initial_fewq_torch, output_mask_initial_fewq_xformers.cpu())
