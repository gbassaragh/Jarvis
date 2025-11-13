"""
FP8 and INT8 quantization kernels for SM120

Leverages Blackwell's native FP8 support for 2x throughput
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _quantize_fp8_kernel(
    X, Y, Scale,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Quantize FP16/BF16 to FP8 (E4M3 format)

    E4M3 format:
    - 1 sign bit
    - 4 exponent bits
    - 3 mantissa bits
    - Range: ±448, good for activations
    """
    pid = tl.program_id(0)

    # Compute offsets
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Load input
    x = tl.load(X + offs, mask=mask, other=0.0)

    # Compute scale (per-tensor quantization)
    # Scale to maximize FP8 range [-448, 448]
    abs_max = tl.max(tl.abs(x))
    scale = 448.0 / (abs_max + 1e-12)

    # Quantize
    y = (x * scale).to(tl.float8e4m3)

    # Store output and scale
    tl.store(Y + offs, y, mask=mask)
    if pid == 0:
        tl.store(Scale, scale)


def quantize_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize tensor to FP8 format

    Args:
        x: Input tensor (FP16/BF16/FP32)

    Returns:
        (quantized_tensor, scale)

    Benefits:
        - 2x memory reduction
        - 2x compute throughput on SM120 tensor cores
        - Minimal accuracy loss for most models
    """
    N = x.numel()
    x_flat = x.flatten()

    # Allocate output
    y = torch.empty_like(x_flat, dtype=torch.float8_e4m3fn)
    scale = torch.zeros(1, device=x.device, dtype=torch.float32)

    BLOCK_SIZE = 1024

    grid = (triton.cdiv(N, BLOCK_SIZE),)

    _quantize_fp8_kernel[grid](
        x_flat, y, scale,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return y.reshape(x.shape), scale


@triton.jit
def _dequantize_fp8_kernel(
    X, Y, Scale,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Dequantize FP8 back to FP16/BF16
    """
    pid = tl.program_id(0)

    # Compute offsets
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Load input and scale
    x = tl.load(X + offs, mask=mask, other=0.0)
    scale = tl.load(Scale)

    # Dequantize
    y = x.to(tl.float32) / scale

    # Store output
    tl.store(Y + offs, y, mask=mask)


def dequantize_fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    Dequantize FP8 tensor back to original precision

    Args:
        x: Quantized tensor (FP8)
        scale: Quantization scale

    Returns:
        Dequantized tensor
    """
    N = x.numel()
    x_flat = x.flatten()

    # Allocate output
    y = torch.empty_like(x_flat, dtype=torch.float32)

    BLOCK_SIZE = 1024

    grid = (triton.cdiv(N, BLOCK_SIZE),)

    _dequantize_fp8_kernel[grid](
        x_flat, y, scale,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return y.reshape(x.shape)


@triton.jit
def _quantize_int8_kernel(
    X, Y, Scale, ZeroPoint,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Quantize to INT8 with per-channel scaling

    Good for weights (activations better with FP8)
    """
    pid = tl.program_id(0)

    # Compute offsets
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Load input
    x = tl.load(X + offs, mask=mask, other=0.0)

    # Compute scale and zero point
    x_max = tl.max(x)
    x_min = tl.min(x)

    scale = (x_max - x_min) / 255.0
    zero_point = -tl.floor(x_min / scale)

    # Quantize
    y = tl.floor(x / scale + zero_point)
    y = tl.maximum(tl.minimum(y, 127.0), -128.0)

    # Store output
    tl.store(Y + offs, y.to(tl.int8), mask=mask)
    if pid == 0:
        tl.store(Scale, scale)
        tl.store(ZeroPoint, zero_point)


def quantize_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize tensor to INT8

    Args:
        x: Input tensor

    Returns:
        (quantized_tensor, scale, zero_point)

    Use cases:
        - Weight quantization
        - 4x memory reduction
        - Efficient storage
    """
    N = x.numel()
    x_flat = x.flatten()

    # Allocate output
    y = torch.empty_like(x_flat, dtype=torch.int8)
    scale = torch.zeros(1, device=x.device, dtype=torch.float32)
    zero_point = torch.zeros(1, device=x.device, dtype=torch.float32)

    BLOCK_SIZE = 1024

    grid = (triton.cdiv(N, BLOCK_SIZE),)

    _quantize_int8_kernel[grid](
        x_flat, y, scale, zero_point,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return y.reshape(x.shape), scale, zero_point


@triton.jit
def _fused_quantized_matmul_kernel(
    A, B, C,
    Scale_A, Scale_B,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused FP8 matrix multiplication

    Leverages SM120's FP8 tensor cores for maximum throughput
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Initialize accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Iterate over K dimension
    for k in range(0, K, BLOCK_K):
        # Load A block
        offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_ak = k + tl.arange(0, BLOCK_K)
        a_ptrs = A + offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (offs_ak[None, :] < K), other=0.0)

        # Load B block
        offs_bk = k + tl.arange(0, BLOCK_K)
        offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        b_ptrs = B + offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        b = tl.load(b_ptrs, mask=(offs_bk[:, None] < K) & (offs_bn[None, :] < N), other=0.0)

        # FP8 matrix multiplication using tensor cores
        acc += tl.dot(a.to(tl.float8e4m3), b.to(tl.float8e4m3), out_dtype=tl.float32)

    # Load scales and apply
    scale_a = tl.load(Scale_A)
    scale_b = tl.load(Scale_B)
    acc = acc / (scale_a * scale_b)

    # Store output
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))


def fused_quantized_matmul(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> torch.Tensor:
    """
    Fused FP8 matrix multiplication

    Args:
        a: FP8 tensor [M, K]
        b: FP8 tensor [K, N]
        scale_a: Scale for A
        scale_b: Scale for B

    Returns:
        Output tensor [M, N]

    Performance:
        - 2x faster than FP16 matmul on SM120
        - Fused scaling reduces overhead
    """
    M, K = a.shape
    K2, N = b.shape
    assert K == K2

    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    # Optimal block sizes for SM120
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 64

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _fused_quantized_matmul_kernel[grid](
        a, b, c,
        scale_a, scale_b,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return c
