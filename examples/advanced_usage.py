"""
Advanced usage example for AI Assistant Pro

Demonstrates:
- Custom kernel usage
- FP8 quantization
- Performance comparison
- Memory profiling
"""

import torch
from ai_assistant_pro.kernels.attention import flashattention_v3
from ai_assistant_pro.kernels.fused_ops import fused_layernorm, fused_gelu
from ai_assistant_pro.kernels.quantization import quantize_fp8, dequantize_fp8
from ai_assistant_pro import AssistantEngine


def demo_flashattention():
    """Demonstrate FlashAttention-3 kernel"""
    print("\n" + "=" * 60)
    print("FlashAttention-3 Demo")
    print("=" * 60)

    # Create test tensors
    batch_size = 2
    num_heads = 8
    seq_len = 1024
    head_dim = 64

    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)

    print(f"Input shape: [batch={batch_size}, heads={num_heads}, seq_len={seq_len}, head_dim={head_dim}]")

    # Run FlashAttention-3
    print("\nRunning FlashAttention-3 (FP16)...")
    out_fp16 = flashattention_v3(q, k, v, use_fp8=False)
    print(f"Output shape: {out_fp16.shape}")

    # Run FlashAttention-3 with FP8
    print("\nRunning FlashAttention-3 (FP8)...")
    out_fp8 = flashattention_v3(q, k, v, use_fp8=True)
    print(f"Output shape: {out_fp8.shape}")

    # Compare results
    diff = torch.abs(out_fp16 - out_fp8).max().item()
    print(f"\nMax difference (FP16 vs FP8): {diff:.6f}")

    print("\n✓ FlashAttention-3 demo complete!")


def demo_fused_ops():
    """Demonstrate fused operations"""
    print("\n" + "=" * 60)
    print("Fused Operations Demo")
    print("=" * 60)

    # Create test tensors
    batch_size = 4
    seq_len = 512
    hidden_dim = 768

    x = torch.randn(batch_size, seq_len, hidden_dim, device="cuda", dtype=torch.float16)
    weight = torch.randn(hidden_dim, device="cuda", dtype=torch.float16)
    bias = torch.randn(hidden_dim, device="cuda", dtype=torch.float16)

    print(f"Input shape: [batch={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}]")

    # Fused LayerNorm
    print("\nRunning fused LayerNorm...")
    out_ln = fused_layernorm(x, weight, bias)
    print(f"Output shape: {out_ln.shape}")

    # Fused GELU
    print("\nRunning fused GELU...")
    out_gelu = fused_gelu(x)
    print(f"Output shape: {out_gelu.shape}")

    # Compare with PyTorch
    pytorch_gelu = torch.nn.functional.gelu(x)
    diff = torch.abs(out_gelu - pytorch_gelu).max().item()
    print(f"\nMax difference (Triton vs PyTorch GELU): {diff:.6f}")

    print("\n✓ Fused operations demo complete!")


def demo_quantization():
    """Demonstrate FP8 quantization"""
    print("\n" + "=" * 60)
    print("FP8 Quantization Demo")
    print("=" * 60)

    # Create test tensor
    x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
    print(f"Original tensor shape: {x.shape}, dtype: {x.dtype}")
    print(f"Original memory: {x.element_size() * x.numel() / 1024 / 1024:.2f} MB")

    # Quantize to FP8
    print("\nQuantizing to FP8...")
    x_fp8, scale = quantize_fp8(x)
    print(f"Quantized tensor dtype: {x_fp8.dtype}")
    print(f"Quantized memory: {x_fp8.element_size() * x_fp8.numel() / 1024 / 1024:.2f} MB")
    print(f"Scale: {scale.item():.4f}")

    # Dequantize back
    print("\nDequantizing back to FP32...")
    x_recovered = dequantize_fp8(x_fp8, scale)
    print(f"Recovered tensor dtype: {x_recovered.dtype}")

    # Calculate error
    error = torch.abs(x.float() - x_recovered).mean().item()
    print(f"\nMean absolute error: {error:.6f}")
    print(f"Relative error: {error / x.float().abs().mean().item() * 100:.2f}%")

    print("\n✓ FP8 quantization demo complete!")


def demo_performance_comparison():
    """Compare performance with and without optimizations"""
    print("\n" + "=" * 60)
    print("Performance Comparison")
    print("=" * 60)

    # Test with optimizations
    print("\n1. Engine WITH optimizations:")
    engine_optimized = AssistantEngine(
        model_name="gpt2",
        use_triton=True,
        use_fp8=False,
        enable_paged_attention=True,
    )

    prompt = "Once upon a time"
    result = engine_optimized.benchmark(seq_lengths=[512, 1024])

    print("\n  Benchmark results:")
    for seq_len, metrics in result.items():
        print(f"    {seq_len}: {metrics['avg_time_ms']:.2f} ms, {metrics['tokens_per_sec']:.2f} tokens/sec")

    # Get stats
    stats = engine_optimized.get_stats()
    print(f"\n  Statistics:")
    print(f"    Model: {stats['model']}")
    print(f"    Triton: {stats['use_triton']}")
    print(f"    FP8: {stats['use_fp8']}")
    print(f"    Paged attention: {stats['paged_attention']}")

    print("\n✓ Performance comparison complete!")


def main():
    """Run all advanced examples"""
    print("AI Assistant Pro - Advanced Usage Examples")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available! These examples require a GPU.")
        return

    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")

    # Run demos
    demo_flashattention()
    demo_fused_ops()
    demo_quantization()
    demo_performance_comparison()

    print("\n" + "=" * 60)
    print("✓ All advanced examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
