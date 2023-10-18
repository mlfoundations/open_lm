import torch
from open_lm.triton.rms_norm import rms_norm


def pytorch_naive_rmsnorm(a: torch.Tensor, weight: torch.Tensor, eps: float):
    # modified from: https://github.com/ELS-RD/kernl/blob/91e2cd92db44d503874d39a9f6dec42c9f481a8e/src/kernl/implementations/layer_norm.py#L41
    variance = a.to(torch.float32).pow(2).mean(-1, keepdim=True)
    tmp = a * torch.rsqrt(variance + eps)

    # convert into half-precision if necessary
    if weight.dtype in [torch.float16, torch.bfloat16]:
        tmp = tmp.to(weight.dtype)

    return weight * tmp


def test_rms_norm(M=1151, N=8192, dtype=torch.float32, eps=1e-5, device="cuda"):
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1],)
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    dy = 0.1 * torch.randn_like(x)
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

    assert torch.allclose(y_tri, y_ref, atol=1e-2, rtol=0), "y does not match"
    assert torch.allclose(dx_tri, dx_ref, atol=1e-2, rtol=0), "dx does not match"
    assert torch.allclose(dw_tri, dw_ref, atol=1e-2, rtol=0), "dw does not match"
