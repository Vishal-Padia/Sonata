import torch
import cutlass
import cutlass.cute as cute
import torch.nn.functional as F
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack


_BLOCK_SIZE = 128
_COMPILED_CACHE: dict[tuple, object] = {}


def conv1d_output_length(
    input_length: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
) -> int:
    """Calculate PyTorch Conv1d's output length."""
    return (
        input_length
        + 2 * padding
        - dilation * (kernel_size - 1)
        - 1
    ) // stride + 1


@cute.kernel
def _fused_snake_conv1d_kernel(
    output: cute.Tensor,
    x: cute.Tensor,
    alpha: cute.Tensor,
    weight: cute.Tensor,
    bias: cute.Tensor,
    batch_size: cutlass.Int32,
    input_channels: cutlass.Int32,
    output_channels: cutlass.Int32,
    input_length: cutlass.Int32,
    output_length: cutlass.Int32,
    kernel_size: cutlass.Int32,
    stride: cutlass.Int32,
    padding: cutlass.Int32,
    dilation: cutlass.Int32,
    eps: cutlass.Float32,
):
    """Compute one fused Snake -> Conv1d output element per thread.
    Mapping:
        linear thread index -> (batch, output_channel, output_position)
    Reduction:
        loop over input_channels and kernel_size
    Input index:
        input_position = output_position * stride
                       - padding
                       + kernel_position * dilation
    """
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    tid = bidx * _BLOCK_SIZE + tidx

    total_outputs = batch_size * output_channels * output_length
    if tid < total_outputs:
        op = tid % output_length
        tmp = tid // output_length
        oc = tmp % output_channels
        b  = tmp // output_channels

        acc = cutlass.Float32(0.0)
        
        acc += bias[oc]

        for ic in range(input_channels):
            alpha_val = alpha[ic]
            for k in range(kernel_size):
                ip = op * stride - padding + k * dilation

                x_val = cutlass.Float32(0.0)
                if ip >= 0 and ip < input_length:
                    x_val = x[b, ic, ip]

                sin_val = cute.math.sin(alpha_val * x_val)
                snake = x_val + (sin_val * sin_val) / (alpha_val + eps)

                w_val = weight[oc, ic, k]
                acc += snake * w_val

        output[b, oc, op] = acc

@cute.jit
def _launch_fused_snake_conv1d(
    output: cute.Tensor,
    x: cute.Tensor,
    alpha: cute.Tensor,
    weight: cute.Tensor,
    bias: cute.Tensor,
    batch_size: cutlass.Int32,
    input_channels: cutlass.Int32,
    output_channels: cutlass.Int32,
    input_length: cutlass.Int32,
    output_length: cutlass.Int32,
    kernel_size: cutlass.Int32,
    stride: cutlass.Int32,
    padding: cutlass.Int32,
    dilation: cutlass.Int32,
    eps: cutlass.Float32,
    stream,
):
    """Configure and launch the fused kernel."""
    total_outputs = batch_size * output_channels * output_length
    grid_size = (total_outputs + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    _fused_snake_conv1d_kernel(
        output,
        x,
        alpha,
        weight,
        bias,
        batch_size,
        input_channels,
        output_channels,
        input_length,
        output_length,
        kernel_size,
        stride,
        padding,
        dilation,
        eps,
    ).launch(
        grid=[grid_size, 1, 1],
        block=[_BLOCK_SIZE, 1, 1],
        stream=stream,
    )


def fused_snake_conv1d(
    x: torch.Tensor,
    alpha: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    eps: float = 1e-9,
) -> torch.Tensor:
    """Validate inputs, compile/cache the kernel, and return its output."""
    batch_size, input_channels, input_length = x.shape
    output_channels, weight_channels, kernel_size = weight.shape
    if weight_channels != input_channels:
        raise ValueError("Only groups=1 is currently supported.")
    if alpha.numel() != input_channels:
        raise ValueError("alpha must contain one value per input channel.")
    if bias.numel() != output_channels:
        raise ValueError("bias must contain one value per output channel.")
    output_length = conv1d_output_length(
        input_length,
        kernel_size,
        stride,
        padding,
        dilation,
    )
    output = torch.empty(
        (batch_size, output_channels, output_length),
        device=x.device,
        dtype=x.dtype,
    )
    stream = cutlass_torch.current_stream()
    arguments = (
        from_dlpack(output),
        from_dlpack(x),
        from_dlpack(alpha),
        from_dlpack(weight),
        from_dlpack(bias),
        batch_size,
        input_channels,
        output_channels,
        input_length,
        output_length,
        kernel_size,
        stride,
        padding,
        dilation,
        eps,
        stream,
    )
    cache_key = (
        x.device,
        x.dtype,
        x.shape,
        weight.shape,
        stride,
        padding,
        dilation,
    )
    if cache_key not in _COMPILED_CACHE:
        _COMPILED_CACHE[cache_key] = cute.compile(
            _launch_fused_snake_conv1d,
            *arguments,
        )
    _COMPILED_CACHE[cache_key](*arguments)
    return output

def explicit_snake_conv1d(
    x: torch.Tensor,
    alpha: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    eps: float = 1e-9
) -> torch.Tensor:
    """
    Read x → compute Snake locally → multiply by weights → accumulate y

    Shapes:
        x:      [batch, input_channels, input_length]
        alpha:  one value per input channel
        weight: [output_channels, input_channels, kernel_size]
        bias:   [output_channels]

    Args:
        x: Input tensor of shape (batch_size, in_channels, sequence_length)
        alpha: Alpha tensor of shape (batch_size, out_channels)
        weight: Weight tensor of shape (out_channels, in_channels, kernel_size)
        bias: Bias tensor of shape (out_channels)
        stride: Stride for the convolution
        padding: Padding for the convolution
        dilation: Dilation for the convolution
        eps: Epsilon for the snake activation
    """
    _, input_channels, _ = x.shape
    output_channels, weight_channels, kernel_size = weight.shape

    if weight_channels != input_channels:
        raise ValueError("This oracle supports ungrouped Conv1d only.")
    if alpha.numel() != input_channels:
        raise ValueError("alpha must contain one value per input channel.")
    if bias is not None and bias.numel() != output_channels:
        raise ValueError("bias must contain one value per output channel.")
    
    # padding before SNake is equivalent to Conv1d zero-padding because snake(0) is exactly zero
    padded = F.pad(x, (padding, padding))

    # extract the complete dilated receptive field for every output position
    receptive_field = dilation * (kernel_size - 1) + 1
    windows = padded.unfold(dimension=2, size=receptive_field, step=stride)
    windows = windows[..., ::dilation]

    # broadcast the channel-wise snake param over batch, time and kernel
    alpha = alpha.reshape(1, input_channels, 1, 1)
    activated_windows = windows + torch.sin(alpha * windows).square() / (alpha + eps)

    # sum over input channels and kernel position
    output = torch.einsum(
        "bctk, ock -> bot",
        activated_windows,
        weight,
    )

    if bias is not None:
        output = output + bias.reshape(1, -1, 1)
    
    return output

def reference_snake_conv1d(
    x: torch.Tensor,
    alpha: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    eps: float = 1e-9
) -> torch.Tensor:

    activated = x + torch.sin(alpha.reshape(1, -1, 1) * x).square() / (
        alpha.reshape(1, -1, 1) + eps
    )
    reference = F.conv1d(
        activated,
        weight,
        bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    return reference

def test_snake_conv1d():
    torch.manual_seed(42)
    # Small, reasonable test dimensions
    batch_size = 2
    in_channels = 4
    out_channels = 3
    seq_len = 8
    kernel_size = 3

    x = torch.randn(batch_size, in_channels, seq_len)
    alpha = torch.rand(in_channels) * 2 + 0.3  # avoid 0 for alpha
    weight = torch.randn(out_channels, in_channels, kernel_size)
    bias = torch.randn(out_channels)
    stride = 1
    padding = 1
    dilation = 1

    # Run both functions
    y_explicit = explicit_snake_conv1d(
        x, alpha, weight, bias, stride=stride, padding=padding, dilation=dilation
    )
    y_reference = reference_snake_conv1d(
        x, alpha, weight, bias, stride=stride, padding=padding, dilation=dilation
    )

    print("explicit_snake_conv1d output:")
    print(y_explicit)
    print("reference_snake_conv1d output:")
    print(y_reference)
    max_abs_diff = (y_explicit - y_reference).abs().max().item()
    print(f"Max abs difference: {max_abs_diff:.6g}")
    if max_abs_diff < 1e-5:
        print("PASS: Outputs match closely.")
    else:
        print("FAIL: Outputs diverge.")


def test_fused_kernel():
    """Actually runs the CuTe kernel (fused_snake_conv1d) on the GPU and checks
    it against reference_snake_conv1d. Requires CUDA."""
    if not torch.cuda.is_available():
        print("SKIP: test_fused_kernel requires CUDA, none available.")
        return

    torch.manual_seed(42)
    batch_size, in_channels, out_channels, seq_len, kernel_size = 2, 4, 3, 8, 3
    device = "cuda"
    stride, padding, dilation = 1, 1, 1

    x = torch.randn(batch_size, in_channels, seq_len, device=device, dtype=torch.float32)
    alpha = torch.rand(in_channels, device=device, dtype=torch.float32) * 2 + 0.3
    weight = torch.randn(out_channels, in_channels, kernel_size, device=device, dtype=torch.float32)
    bias = torch.randn(out_channels, device=device, dtype=torch.float32)

    y_kernel = fused_snake_conv1d(x, alpha, weight, bias, stride=stride, padding=padding, dilation=dilation)
    torch.cuda.synchronize()
    y_reference = reference_snake_conv1d(x, alpha, weight, bias, stride=stride, padding=padding, dilation=dilation)

    print("fused_snake_conv1d (CuTe kernel) output:")
    print(y_kernel)
    print("reference_snake_conv1d output:")
    print(y_reference)
    max_abs_diff = (y_kernel - y_reference).abs().max().item()
    print(f"Max abs difference: {max_abs_diff:.6g}")
    if max_abs_diff < 1e-2:
        print("PASS: CuTe kernel matches reference closely.")
    else:
        print("FAIL: CuTe kernel diverges from reference.")


if __name__ == "__main__":
    # test_snake_conv1d()
    test_fused_kernel()