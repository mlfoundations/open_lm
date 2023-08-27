# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# CREDITS: the underlying kernel comes straight from the Triton tutorials
# see https://github.com/openai/triton/blob/master/python/tutorials/05-layer-norm.py

import logging
from typing import Optional

import torch
import torch.nn as nn
import triton
from torch.cuda.amp import custom_bwd, custom_fwd

import triton
import triton.language as tl


logger = logging.getLogger("xformers")


_triton_layernorm_fp16_enabled = False  # NOTE: PyTorch keeps layernorm as fp32
_triton_registered_warnings = False

# fmt: off
@triton.jit
def layer_norm_fw(X, Y, W, B, M, V, stride, N, eps, affine: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    # fmt: on
    """
    Fused layernorm kernel over a 3d tensor.
    The layer norm is applied over the last dimension.

    Compute
        y = (x - E(x))/(sqrt(var(x) + epsilon)) * gamma + beta
    """

    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N

    # Move to this row
    x_ptrs = X + row * stride + cols
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Compute mean and variance
    mean = tl.sum(x, axis=0) / N
    x_zm = tl.where(mask, x - mean, 0.0)
    tl.store(M + row, mean)

    x_var = tl.sum(x_zm * x_zm, axis=0) / N
    rstd = 1.0 / tl.sqrt(x_var + eps)

    # Normalize, optionally affine
    y = x_zm * rstd
    tl.store(V + row, rstd)

    mask = cols < N
    if affine:
        w = tl.load(W + cols, mask=mask, other=1.0)
        b = tl.load(B + cols, mask=mask, other=0.0)
        y = y * w + b

    y_ptrs = Y + row * stride + cols
    tl.store(y_ptrs, y, mask=mask)


# Backward pass (DX + partial DW + partial DB)
# fmt: off
@triton.jit
def layer_norm_bwd_dx_fused(
    DX, DY, DW, DB,
    X, W, M, V,
    Lock, stride, N,
    # META-parameters
    affine: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # fmt: on

    # position of elements processed by this program
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N

    # offset data pointers to start at the row of interest
    x_ptrs = X + row * stride + cols
    dy_ptrs = DY + row * stride + cols

    # load data to SRAM
    x = tl.load(x_ptrs, mask=mask, other=0)
    dy = tl.load(dy_ptrs, mask=mask, other=0)
    mean = tl.load(M + row)
    rstd = tl.load(V + row)

    # compute dx
    xhat = (x - mean) * rstd

    if affine:
        w = tl.load(W + cols, mask=mask, other=0)
        wdy = w * dy
    else:
        wdy = dy

    xhat = tl.where(mask, xhat, 0.)
    wdy = tl.where(mask, wdy, 0.)
    mean1 = tl.sum(xhat * wdy, axis=0) / N
    mean2 = tl.sum(wdy, axis=0) / N
    dx = (wdy - (xhat * mean1 + mean2)) * rstd

    # write-back dx
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N  # re-materialize the mask to save registers
    dx_ptrs = DX + row * stride + cols
    tl.store(dx_ptrs, dx, mask=mask)

    if affine:
        # accumulate partial sums for dw/db
        partial_dw = (dy * xhat).to(w.dtype)
        partial_db = dy.to(w.dtype)

        # offset locks and weight/bias gradient pointer
        # each kernel instance accumulates partial sums for
        # DW and DB into one of GROUP_SIZE_M independent buffers
        # these buffers stay in the L2, which allow this kernel
        # to be fast
        lock_id = row % GROUP_SIZE_M
        Lock += lock_id
        Count = Lock + GROUP_SIZE_M

        # - wait for a lock on the accumulated dw/db
        while tl.atomic_cas(Lock, 0, 1) == 1:
            pass
        count = tl.load(Count)

        # - we got the lock, accumulate this kernel's results with
        # the stored values.
        dw_ptrs = DW + lock_id * N + cols
        db_ptrs = DB + lock_id * N + cols

        if count == 0:
            # first store doesn't accumulate
            tl.atomic_xchg(Count, 1)
        else:
            partial_dw += tl.load(dw_ptrs, mask=mask, other=0.)
            partial_db += tl.load(db_ptrs, mask=mask, other=0.)

        tl.store(dw_ptrs, partial_dw, mask=mask)
        tl.store(db_ptrs, partial_db, mask=mask)

        # release lock
        tl.atomic_xchg(Lock, 0)


# Backward pass (total DW + total DB)
# fmt: off
@triton.jit
def layer_norm_bwd_dwdb(
    DW, DB, FINAL_DW, FINAL_DB,
    M, N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    # fmt: on

    pid = tl.program_id(0)

    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_cols = cols < N

    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        offs = rows[:, None] * N + cols[None, :]
        mask_rm = rows < M

        dw += tl.load(DW + offs, mask=mask_rm[:, None] & mask_cols[None, :], other=0.0)
        db += tl.load(DB + offs, mask=mask_rm[:, None] & mask_cols[None, :], other=0.0)

    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)

    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_cols = cols < N

    tl.store(FINAL_DW + cols, sum_dw, mask=mask_cols)
    tl.store(FINAL_DB + cols, sum_db, mask=mask_cols)


class _LayerNorm(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16 if _triton_layernorm_fp16_enabled else None)
    def forward(ctx, x, weight, bias, eps):
        # catch eps being too small if the tensors are fp16
        if x.dtype == torch.float16:
            eps = max(eps, 1.6e-5)

        # allocate output
        y = torch.empty_like(x)

        # reshape input data into 2D tensor
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape

        # allocate mean and std, they'll be used in the backward pass
        mean = torch.empty((M,), dtype=torch.float32, device="cuda")
        rstd = torch.empty((M,), dtype=torch.float32, device="cuda")

        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE_N:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

        if not x_arg.is_contiguous() or not y.is_contiguous():
            global _triton_registered_warnings
            if not _triton_registered_warnings:
                logger.warning(
                    "Non-contiguous input tensor found. Making it contiguous,"
                    + " but could have perf or trainer implications"
                )

                _triton_registered_warnings = True

            x_arg = x_arg.contiguous()
            y = y.contiguous()

        # heuristics for number of warps.
        num_warps = min(max(BLOCK_SIZE_N // 256, 1), 16)

        # enqueue kernel
        # fmt: off
        layer_norm_fw[(M,)](
            x_arg, y, weight, bias, mean, rstd,
            x_arg.stride(0),
            N,
            eps,
            num_warps=num_warps,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            affine=weight is not None
        )
        # fmt: on

        ctx.save_for_backward(x, mean, rstd, weight)
        ctx.BLOCK_SIZE_N = BLOCK_SIZE_N
        ctx.num_warps = num_warps

        return y.reshape_as(x)

    @staticmethod
    @custom_bwd
    def backward(
        ctx, dy
    ):  # pragma: no cover  # this is covered, but called directly from C++
        x, mean, rstd, weight = ctx.saved_tensors

        # flatten the batch dimension, if any.
        # We're interested in 'samples' x norm_dimension
        x = x.reshape(-1, x.size(-1))
        M, N = x.size()

        # heuristics for amount of parallel reduction stream for DG/DB
        GROUP_SIZE_M = 32
        if N <= 8192:
            GROUP_SIZE_M = 64
        if N <= 4096:
            GROUP_SIZE_M = 96
        if N <= 2048:
            GROUP_SIZE_M = 128
        if N <= 1024:
            GROUP_SIZE_M = 256

        if dy.dtype == torch.float32:
            GROUP_SIZE_M = GROUP_SIZE_M // 2

        # allocate output
        locks = torch.zeros(2 * GROUP_SIZE_M, dtype=torch.int32, device="cuda")
        t_args = {"dtype": x.dtype, "device": x.device}
        _dw = torch.empty((GROUP_SIZE_M, x.size(-1)), **t_args)
        _db = torch.empty_like(_dw)
        dw = torch.empty((x.size(-1),), **t_args)
        db = torch.empty_like(dw)
        dy = dy.contiguous()
        dx = torch.empty_like(dy)

        # Check the tensor shapes and layouts
        # we suppose in the kernel that they have the same size and are contiguous
        assert (
            dy.numel() == x.numel()
        ), "Something is wrong in the backward graph, possibly because of an inplace operation after the layernorm"

        # enqueue kernel using forward pass heuristics
        # also compute partial sums for DW and DB
        num_warps = min(max(ctx.BLOCK_SIZE_N // 256, 1), 16)

        # fmt: off
        layer_norm_bwd_dx_fused[(M,)](
            dx, dy, _dw, _db, x,
            weight if weight is not None else x,
            mean, rstd,
            locks,
            x.stride(0),
            N,
            affine=weight is not None,
            GROUP_SIZE_M=GROUP_SIZE_M,
            BLOCK_SIZE_N=ctx.BLOCK_SIZE_N,
            num_warps=num_warps
        )
        # fmt: on

        def grid(meta):
            return [triton.cdiv(N, meta["BLOCK_SIZE_N"])]

        # accumulate partial sums in separate kernel
        # fmt: off
        layer_norm_bwd_dwdb[grid](
            _dw, _db, dw, db,
            GROUP_SIZE_M,
            N,
            BLOCK_SIZE_M=32,
            BLOCK_SIZE_N=64
        )
        # fmt: on

        dx = dx.reshape_as(dy)
        return dx, dw, db, None


class FusedLayerNorm(nn.Module):
    """
    Handle a layer normalization, like torch.nn.LayerNorm_.

    This implementation should be measurably faster than the default PyTorch layernorm (as of PyTorch 1.9),
    both for training and inference worloads.

    .. NOTE: Computations under Torch AMP are kept as float32 by default, one can change this to be float16
        by setting the flag `xformers.triton.k_layer_norm._triton_layernorm_fp16_enabled = True`

    .. _torch.nn.LayerNorm: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html

    """

    def __init__(self, normalized_shape, affine=True, eps=1e-06):
        super().__init__()
        if affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.weight = self.bias = None
        self.epsilon = eps

    def forward(self, x):
        return layer_norm(x, self.weight, self.bias, self.epsilon)

    def init_weights(self, *args, **kwargs):
        with torch.no_grad():
            if self.weight is not None:
                self.weight.fill_(1.0)

            if self.bias is not None:
                self.bias.fill_(0.0)


def layer_norm(
    x: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-06,
) -> torch.Tensor:

    global _triton_registered_warnings

    r"""Applies normalization over a mini batch of inputs"""

    try:
        if (
            not _triton_registered_warnings
            and torch.cuda.is_available()
            and x.is_cuda
            and weight is not None
            and bias is not None
        ):
            return _LayerNorm.apply(x, weight, bias, eps)
    except RuntimeError as e:
        # Catch cases where the current GPU does not have enough registers to hold a full tensor line
        # fallback to PyTorch's implementation, which streams the tensor in and out
        _triton_registered_warnings = True
        logger.warning(
            "Triton layernorm kernel register spillover or invalid image caught. "
            "Deactivating this kernel, please file an issue in the xFormers repository"
        )
        logger.warning(e)

    return torch.nn.functional.layer_norm(
        x, [x.shape[-1]], weight=weight, bias=bias, eps=eps
    )


def test_layer_norm(M, N, dtype, eps=1e-5, device='cuda'):
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    weight = torch.rand(w_shape, dtype=dtype, device='cuda', requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device='cuda', requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device='cuda')
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)
    # forward pass
    y_tri = layer_norm(x, weight, bias, eps)
    y_ref = torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps).to(dtype)

    # backward pass (triton)
    y_tri.backward(dy, retain_graph=True)
    dx_tri, dw_tri, db_tri = [_.grad.clone() for _ in [x, weight, bias]]
    x.grad, weight.grad, bias.grad = None, None, None
    # backward pass (torch)
    y_ref.backward(dy, retain_graph=True)
    dx_ref, dw_ref, db_ref = [_.grad.clone() for _ in [x, weight, bias]]
    # compare
    assert torch.allclose(y_tri, y_ref, atol=3e-2, rtol=0)
    assert torch.allclose(dx_tri, dx_ref, atol=3e-2, rtol=0)
    assert torch.allclose(db_tri, db_ref, atol=3e-2, rtol=0)
    assert torch.allclose(dw_tri, dw_ref, atol=3e-2, rtol=0)

test_layer_norm(1151, 8192, torch.float16)
# bench_layer_norm.run(save_path='.', print_data=True)