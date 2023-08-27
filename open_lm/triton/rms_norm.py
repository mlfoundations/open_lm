# modified from: https://github.com/facebookresearch/xformers/blob/e153e4b4f5d0d821d707696029d84faed11a92bf/xformers/triton/k_layer_norm.py

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


_triton_rmsnorm_fp16_enabled = False  # NOTE: PyTorch keeps layernorm as fp32
_triton_registered_warnings = False

# fmt: off
@triton.jit
def rms_norm_fw(X, Y, W, R, stride, N, eps, affine: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    # fmt: on
    """
    Fused rmsnorm kernel over a 3d tensor.
    The rms norm is applied over the last dimension.

    Compute
        y = x/(rms(x) + epsilon) * gamma + beta
    """

    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N

    # Move to this row
    x_ptrs = X + row * stride + cols
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Compute reciprical rms
    ms = tl.sum(x * x, axis=0) / N
    rrms = 1.0 / tl.sqrt(ms + eps)

    tl.store(R + row, rrms)

    # Normalize, optionally affine
    y = x * rrms

    mask = cols < N
    if affine:
        w = tl.load(W + cols, mask=mask, other=1.0)
        y = y * w

    y_ptrs = Y + row * stride + cols
    tl.store(y_ptrs, y, mask=mask)


# Backward pass (DX + partial DW)
# fmt: off
@triton.jit
def rms_norm_bwd_dx_fused(
    DX, DY, DW,
    X, W, R,
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
    rrms = tl.load(R + row)

    # compute dx
    xhat = x * rrms

    if affine:
        w = tl.load(W + cols, mask=mask, other=0)
        wdy = w * dy
    else:
        wdy = dy

    xhat = tl.where(mask, xhat, 0.)
    wdy = tl.where(mask, wdy, 0.)
    mean1 = tl.sum(xhat * wdy, axis=0) / N
    mean2 = tl.sum(wdy, axis=0) / N
    dx = (wdy - (xhat * mean1 + mean2)) * rrms

    # write-back dx
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N  # re-materialize the mask to save registers
    dx_ptrs = DX + row * stride + cols
    tl.store(dx_ptrs, dx, mask=mask)

    if affine:
        # accumulate partial sums for dw
        partial_dw = (dy * xhat).to(w.dtype)

        # offset locks and weight gradient pointer
        # each kernel instance accumulates partial sums for
        # DW into one of GROUP_SIZE_M independent buffers
        # these buffers stay in the L2, which allow this kernel
        # to be fast
        lock_id = row % GROUP_SIZE_M
        Lock += lock_id
        Count = Lock + GROUP_SIZE_M

        # - wait for a lock on the accumulated dw
        while tl.atomic_cas(Lock, 0, 1) == 1:
            pass
        count = tl.load(Count)

        # - we got the lock, accumulate this kernel's results with
        # the stored values.
        dw_ptrs = DW + lock_id * N + cols

        if count == 0:
            # first store doesn't accumulate
            tl.atomic_xchg(Count, 1)
        else:
            partial_dw += tl.load(dw_ptrs, mask=mask, other=0.)

        tl.store(dw_ptrs, partial_dw, mask=mask)

        # release lock
        tl.atomic_xchg(Lock, 0)


# Backward pass (total DW)
# fmt: off
@triton.jit
def rms_norm_bwd_dw(
    DW, FINAL_DW,
    M, N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    # fmt: on

    pid = tl.program_id(0)

    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_cols = cols < N

    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        offs = rows[:, None] * N + cols[None, :]
        mask_rm = rows < M

        dw += tl.load(DW + offs, mask=mask_rm[:, None] & mask_cols[None, :], other=0.0)

    sum_dw = tl.sum(dw, axis=0)

    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_cols = cols < N

    tl.store(FINAL_DW + cols, sum_dw, mask=mask_cols)


class _RmsNorm(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16 if _triton_rmsnorm_fp16_enabled else None)
    def forward(ctx, x, weight, eps):
        # catch eps being too small if the tensors are fp16
        if x.dtype == torch.float16:
            eps = max(eps, 1.6e-5)

        # allocate output
        y = torch.empty_like(x)

        # reshape input data into 2D tensor
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape

        # allocate rrms, it will be used in the backward pass
        rrms = torch.empty((M,), dtype=torch.float32, device="cuda")

        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE_N:
            raise RuntimeError("This rms norm doesn't support feature dim >= 64KB.")

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
        rms_norm_fw[(M,)](
            x_arg, y, weight, rrms,
            x_arg.stride(0),
            N,
            eps,
            num_warps=num_warps,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            affine=weight is not None
        )
        # fmt: on

        ctx.save_for_backward(x, rrms, weight)
        ctx.BLOCK_SIZE_N = BLOCK_SIZE_N
        ctx.num_warps = num_warps

        return y.reshape_as(x)

    @staticmethod
    @custom_bwd
    def backward(
        ctx, dy
    ):  # pragma: no cover  # this is covered, but called directly from C++
        x, rrms, weight = ctx.saved_tensors

        # flatten the batch dimension, if any.
        # We're interested in 'samples' x norm_dimension
        x = x.reshape(-1, x.size(-1))
        M, N = x.size()

        # heuristics for amount of parallel reduction stream for DG
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
        dw = torch.empty((x.size(-1),), **t_args)
        dy = dy.contiguous()
        dx = torch.empty_like(dy)

        # Check the tensor shapes and layouts
        # we suppose in the kernel that they have the same size and are contiguous
        assert (
            dy.numel() == x.numel()
        ), "Something is wrong in the backward graph, possibly because of an inplace operation after the rmsnorm"

        # enqueue kernel using forward pass heuristics
        # also compute partial sums for DW
        num_warps = min(max(ctx.BLOCK_SIZE_N // 256, 1), 16)

        # fmt: off
        rms_norm_bwd_dx_fused[(M,)](
            dx, dy, _dw, x,
            weight if weight is not None else x,
            rrms,
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
        rms_norm_bwd_dw[grid](
            _dw, dw,
            GROUP_SIZE_M,
            N,
            BLOCK_SIZE_M=32,
            BLOCK_SIZE_N=64
        )
        # fmt: on

        dx = dx.reshape_as(dy)
        return dx, dw, None


def rms_norm(
    x: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
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
        ):
            return _RmsNorm.apply(x, weight, eps)
    except RuntimeError as e:
        # Catch cases where the current GPU does not have enough registers to hold a full tensor line
        # fallback to PyTorch's implementation, which streams the tensor in and out
        _triton_registered_warnings = True
        logger.warning(
            "Triton rmsnorm kernel register spillover or invalid image caught. "
            "Deactivating this kernel, please file an issue in the xFormers repository"
        )
        logger.warning(e)
        raise e


def pytorch_naive_rmsnorm(a: torch.Tensor, weight: torch.Tensor, eps: float):

    variance = a.to(torch.float32).pow(2).mean(-1, keepdim=True)
    tmp = a * torch.rsqrt(variance + eps)

    # convert into half-precision if necessary
    if weight.dtype in [torch.float16, torch.bfloat16]:
        tmp = tmp.to(weight.dtype)

    return weight * tmp

def test_rms_norm(M, N, dtype, eps=1e-5, device='cuda'):
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    weight = torch.rand(w_shape, dtype=dtype, device='cuda', requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device='cuda')
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)
    # forward pass
    y_tri = rms_norm(x, weight, eps)
    y_ref = pytorch_naive_rmsnorm(x, weight, eps).to(dtype)

    # backward pass (triton)
    y_tri.backward(dy, retain_graph=True)
    dx_tri, dw_tri = [_.grad.clone() for _ in [x, weight]]
    x.grad, weight.grad = None, None
    # backward pass (torch)
    y_ref.backward(dy, retain_graph=True)
    dx_ref, dw_ref = [_.grad.clone() for _ in [x, weight]]
    # compare

    print(y_tri, y_ref)
    print(dx_tri, dx_ref)
    print(dw_tri, dw_ref)

    # assert torch.allclose(y_tri, y_ref, atol=1e-2, rtol=0), "y does not match"
    # assert torch.allclose(dx_tri, dx_ref, atol=1e-2, rtol=0), "dx does not match"
    assert torch.allclose(dw_tri, dw_ref, atol=1e-2, rtol=0), "dw does not match"

test_rms_norm(1151, 8192, torch.float16)
# bench_rms_norm.run(save_path='.', print_data=True)