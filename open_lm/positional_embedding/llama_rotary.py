# NOTE: 08/31/23, this class is copied from xformers as there is currently a bug related to which channel dim the rotary embedding is applied to.
# when the upstream issue is fixed, this file should be deleted. To track progress, see this issue: https://github.com/facebookresearch/xformers/issues/841

# taken from: https://github.com/facebookresearch/xformers/blob/748c159096d4f9fcfe3eaf22801e5aed4777210b/xformers/components/positional_embedding/rotary.py
from typing import Tuple

import torch


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.




    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    # dynamic shape could be more general but torchscript doesn't support it
    # shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    shape = [1, x.shape[1], 1, x.shape[-1]]
    return freqs_cis.view(shape)


@torch.jit.script
def apply_llama_rotary_pos_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply llama rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.

    """
    xq_even = xq[..., ::2]
    xq_odd = xq[..., 1::2]

    xk_even = xk[..., ::2]
    xk_odd = xk[..., 1::2]

    # Stack them along the last dimension to make it [N, ..., D, 2]
    xq_ = torch.stack([xq_even, xq_odd], dim=-1)
    xk_ = torch.stack([xk_even, xk_odd], dim=-1)
    xq_ = torch.view_as_complex(xq_.float())
    xk_ = torch.view_as_complex(xk_.float())

    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class LLaMARotaryEmbedding(torch.nn.Module):
    """
    The rotary position embeddings from RoFormer_ (Su et. al).
    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.

    Other implementations are available in the Rotary Transformer repo_ and in
    GPT-NeoX_, GPT-NeoX was an inspiration

    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox


    .. warning: Please note that this embedding is not registered on purpose, as it is transformative
        (it does not create the embedding dimension) and will likely be picked up (imported) on a ad-hoc basis
    """

    def __init__(self, head_dim: int, num_heads: int, seq_len: int, *_, **__):
        super().__init__()
        # Generate and save the inverse frequency buffer (non trainable)
        self.freqs_cis = precompute_freqs_cis(
            # Note that self.params.max_seq_len is multiplied by 2 because the token limit for the Llama 2 generation of models is 4096.
            # Adding this multiplier instead of using 4096 directly allows for dynamism of token lengths while training or fine-tuning.
            head_dim,
            seq_len * 2,
        )

    def forward(self, q: torch.Tensor, k: torch.Tensor, start_pos: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[1]
        self.freqs_cis = self.freqs_cis.to(q.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seq_len]

        return apply_llama_rotary_pos_emb(q, k, freqs_cis)


class LLaMARotaryWithCast(LLaMARotaryEmbedding):
    def forward(self, q, k, v):
        q, k = super().forward(q, k)
        return q.to(v.dtype), k.to(v.dtype), v
