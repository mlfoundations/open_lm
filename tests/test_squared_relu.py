import torch 
import triton
import triton.language as tl

from open_lm.activations import squared_relu



@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(12, 28, 1)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='vector squared relu performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(size, provider):
    x = torch.rand(size, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        relu = torch.nn.ReLU()
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: (relu(x))**2, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: squared_relu(x), quantiles=quantiles)
    gbps = lambda ms: 12 * size / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)


def test_squared_relu_perf():
	benchmark.run(print_data=True, show_plots=True)

def test_squared_relu_correctness(): 
	torch.manual_seed(0)
	x = torch.randn(1823, 781, device='cuda')
	y_triton = squared_relu(x)
	relu = torch.nn.ReLU()
	y_torch = (relu(x))**2

	assert torch.allclose(y_triton, y_torch), (y_triton, y_torch)
