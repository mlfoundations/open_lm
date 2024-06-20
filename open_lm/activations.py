import torch
import triton
import triton.language as tl

# Forward Triton kernel
@triton.jit
def relu_squared_kernel(X, Y, N, BLOCK_SIZE : tl.constexpr =1024):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < N
    x = tl.load(X + idx, mask=mask, other=0.0)
    y = tl.where(x > 0, x*x, 0.0)
    tl.store(Y + idx, y, mask=mask)

# Backward Triton kernel
@triton.jit
def relu_squared_backward_kernel(X, dY, dX, N, BLOCK_SIZE: tl.constexpr=1024):
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < N
    x = tl.load(X + idx, mask=mask, other=0.0)
    dy = tl.load(dY + idx, mask=mask, other=0.0)
    dx = tl.where(x > 0, 2 * x * dy, 0.0)
    tl.store(dX + idx, dx, mask=mask)

# Custom autograd function
class SquaredReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = torch.empty_like(input)
        n_ele = input.numel()

        grid = lambda meta: (triton.cdiv(n_ele, meta['BLOCK_SIZE']), )
        
        relu_squared_kernel[grid](input, output, n_ele)
        ctx.save_for_backward(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = torch.empty_like(input)
        n_ele = input.numel()
        
        grid = lambda meta: (triton.cdiv(n_ele, meta['BLOCK_SIZE']), )
        
        relu_squared_backward_kernel[grid](input, grad_output, grad_input, n_ele)
        return grad_input

# To use the custom autograd function
def squared_relu(input):
    return SquaredReLUFunction.apply(input)


