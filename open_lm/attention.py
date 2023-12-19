from functools import partial

import torch
from torch.nn import functional as F
import xformers.ops as xops


def get_rectangular_mask(shape, q_seq_len, k_seq_len, device, dtype):
    # xformers requires the mask to be built with a shape that is a multiple of 8
    # probably because of the way it is implemented in CUDA
    next_multiple_8 = (k_seq_len + 7) // 8 * 8  #
    mask = torch.ones((q_seq_len, next_multiple_8), device=device, dtype=bool)
    mask[:, -q_seq_len:] = torch.tril(mask[:, -q_seq_len:], diagonal=0)
    return torch.zeros((*shape, q_seq_len, next_multiple_8), device=device, dtype=dtype).masked_fill(
        ~mask, float("-inf")
    )[:, :, :, :k_seq_len]


def xformers_attn(queries, keys, values, is_causal):
    # xformers assumes q, k, v are [batch, seq_len, heads, embed_dim]
    # We assume that queries match the last part of the key / value sequences
    # see (https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.fmha.attn_bias.LowerTriangularFromBottomRightMask)
    # we would like to replace the mask generation with: mask = xops.fmha.attn_bias.LowerTriangularFromBottomRightMask()
    # sadly we cannot us this because it needs xformers>=0.0.23 and this is not compatible with torch<2.1.1 while llm-foundry requires torch<2.1.1

    mask = None
    # If queries have shape [batch, 1, heads, dim] it means there is only one query in the sequence.
    # In this case, there is no notion of causal masking, so we can just set the mask to None.
    # This is actually needed to get the desired behavior with seq_len=1.
    if is_causal and queries.shape[1] == keys.shape[1]:
        mask = xops.LowerTriangularMask()
    elif is_causal and queries.shape[1] > 1:
        # Build causal mask that assumes queries are in the end of the sequence.
        batch, q_seq_len, heads, _ = queries.shape
        k_seq_len = keys.shape[1]
        mask = get_rectangular_mask((batch, heads), q_seq_len, k_seq_len, queries.device, queries.dtype)
    return xops.memory_efficient_attention(queries, keys, values, attn_bias=mask)


def torch_attn(queries, keys, values, is_causal):
    # Need to call contiguous in torch >=2.1, otherwise later calls to .view() fail.
    # Possibly related: https://github.com/pytorch/pytorch/issues/110213 - behavior of scaled_dot_product_attention
    # changed between 2.0 and 2.1
    if is_causal and keys.shape[1] > queries.shape[1] > 1:
        q_seq_len = queries.shape[1]
        k_seq_len = keys.shape[1]
        # Same as above, we would like to use:
        # mask = xops.fmha.attn_bias.LowerTriangularFromBottomRightMask().materialize((1, 1, q_seq_len, k_seq_len), queries.dtype, queries.device)
        mask = get_rectangular_mask((1, 1), q_seq_len, k_seq_len, queries.device, queries.dtype)
        return (
            F.scaled_dot_product_attention(
                queries.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2), attn_mask=mask
            )
            .transpose(1, 2)
            .contiguous()
        )
    elif queries.shape[1] == 1:
        return (
            F.scaled_dot_product_attention(queries.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2))
            .transpose(1, 2)
            .contiguous()
        )
    else:
        return (
            F.scaled_dot_product_attention(
                queries.transpose(1, 2), keys.transpose(1, 2), values.transpose(1, 2), is_causal=is_causal
            )
            .transpose(1, 2)
            .contiguous()
        )


ATTN_NON_LINEARITIES = {
    "relu": F.relu,
    "relu_squared": lambda x: torch.pow(F.relu(x), 2),
    # "gelu": F.gelu, # goes to NaN with bais so comment out for now
    "softplus": F.softplus,
    "identity": lambda x: x,
    "relu6": F.relu6,
    "sigmoid": F.sigmoid,
    "softmax": partial(F.softmax, dim=-1),
}

ATTN_SEQ_SCALARS = {
    "max": lambda x: x,
    # "seq": lambda x: torch.arange(x) + 1,  # comment out for now more involved
    "avg": lambda x: (x - 1) / 2 + 1,
    "none": lambda _: 1,
}


def non_linear_attn(
    queries,
    keys,
    values,
    attn_non_linearity,
    attn_seq_scalar,
    alpha,
    is_causal=False,
) -> torch.Tensor:
    # naive reference implementation for relu-attention following: https://arxiv.org/pdf/2309.08586.pdf
    # code modifies: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html

    batch, q_seq_len, heads, embed_dim = queries.shape
    _, k_seq_len, _, _ = keys.shape

    attn_bias = torch.zeros(batch, heads, q_seq_len, k_seq_len, device=queries.device, dtype=queries.dtype)
    if is_causal and queries.shape[1] > 1:
        attn_bias = get_rectangular_mask((batch, heads), q_seq_len, k_seq_len, queries.device, queries.dtype)

    inner_scale = 1 / embed_dim**0.5
    attn_weight = inner_scale * torch.einsum("bqhd,bkhd->bhqk", queries, keys)
    attn_weight += attn_bias

    # scaling by: 1/L^{-\alpha}
    outter_scale = ATTN_SEQ_SCALARS[attn_seq_scalar](k_seq_len) ** -alpha
    attn_weight = outter_scale * ATTN_NON_LINEARITIES[attn_non_linearity](attn_weight)

    return torch.einsum("bhqk,bkhd->bqhd", attn_weight, values)


def get_attn_func(
    attn_name,
    attn_non_linearity=None,
    attn_seq_scalar=None,
    alpha=None,
):
    if attn_name == "xformers_attn":
        return xformers_attn
    elif attn_name == "torch_attn":
        return torch_attn
    elif attn_name == "non_linear_attn":
        assert (
            attn_non_linearity is not None and attn_seq_scalar is not None and alpha is not None
        ), "must provide attn-non-linearity, attn-seq-scalar, attn-seq-scalar-alpha"
        return partial(
            non_linear_attn,
            attn_non_linearity,
            attn_seq_scalar,
            alpha,
        )
    else:
        raise ValueError(f"Unsupported attn-name: {attn_name}")
