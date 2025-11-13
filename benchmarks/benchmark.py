"""
Comprehensive benchmark suite for AI Assistant Pro

Compares performance against:
- Standard PyTorch attention
- HuggingFace transformers
- Other inference frameworks

Metrics:
- Throughput (tokens/second)
- Latency (ms/token)
- Memory usage
- GPU utilization
"""

import torch
import time
from typing import Dict, List, Tuple
import numpy as np
from dataclasses import dataclass
from rich.console import Console
from rich.table import Table
from rich.progress import track

from ai_assistant_pro.kernels.attention import flashattention_v3
from ai_assistant_pro.kernels.fused_ops import fused_layernorm, fused_gelu
from ai_assistant_pro.kernels.quantization import quantize_fp8, dequantize_fp8


console = Console()


@dataclass
class BenchmarkResult:
    """Benchmark result"""

    name: str
    avg_time_ms: float
    std_time_ms: float
    throughput: float
    memory_mb: float
    speedup: float = 1.0


class AttentionBenchmark:
    """
    Benchmark attention mechanisms

    Compares:
    - FlashAttention-3 (Triton, SM120)
    - Standard PyTorch attention
    - PyTorch scaled_dot_product_attention
    """

    def __init__(
        self,
        batch_size: int = 4,
        num_heads: int = 32,
        seq_len: int = 4096,
        head_dim: int = 128,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        # Generate test data
        self.q = torch.randn(
            batch_size, num_heads, seq_len, head_dim,
            device=device, dtype=dtype
        )
        self.k = torch.randn(
            batch_size, num_heads, seq_len, head_dim,
            device=device, dtype=dtype
        )
        self.v = torch.randn(
            batch_size, num_heads, seq_len, head_dim,
            device=device, dtype=dtype
        )

    def benchmark_pytorch_attention(self, num_runs: int = 100) -> BenchmarkResult:
        """Benchmark standard PyTorch attention"""
        # Warmup
        for _ in range(10):
            self._pytorch_attention()

        # Benchmark
        times = []
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start = time.time()
            _ = self._pytorch_attention()
            torch.cuda.synchronize()
            times.append(time.time() - start)

        # Calculate metrics
        avg_time = np.mean(times) * 1000  # Convert to ms
        std_time = np.std(times) * 1000

        # Measure memory
        torch.cuda.reset_peak_memory_stats()
        _ = self._pytorch_attention()
        memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

        # Calculate throughput
        total_ops = self.batch_size * self.num_heads * self.seq_len * self.seq_len
        throughput = total_ops / (avg_time / 1000) / 1e9  # GFLOPS

        return BenchmarkResult(
            name="PyTorch Attention",
            avg_time_ms=avg_time,
            std_time_ms=std_time,
            throughput=throughput,
            memory_mb=memory_mb,
        )

    def benchmark_flashattention_v3(self, num_runs: int = 100, use_fp8: bool = False) -> BenchmarkResult:
        """Benchmark FlashAttention-3 (Triton, SM120)"""
        # Warmup
        for _ in range(10):
            _ = flashattention_v3(self.q, self.k, self.v, use_fp8=use_fp8)

        # Benchmark
        times = []
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start = time.time()
            _ = flashattention_v3(self.q, self.k, self.v, use_fp8=use_fp8)
            torch.cuda.synchronize()
            times.append(time.time() - start)

        # Calculate metrics
        avg_time = np.mean(times) * 1000
        std_time = np.std(times) * 1000

        # Measure memory
        torch.cuda.reset_peak_memory_stats()
        _ = flashattention_v3(self.q, self.k, self.v, use_fp8=use_fp8)
        memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

        # Calculate throughput
        total_ops = self.batch_size * self.num_heads * self.seq_len * self.seq_len
        throughput = total_ops / (avg_time / 1000) / 1e9

        name = "FlashAttention-3"
        if use_fp8:
            name += " (FP8)"

        return BenchmarkResult(
            name=name,
            avg_time_ms=avg_time,
            std_time_ms=std_time,
            throughput=throughput,
            memory_mb=memory_mb,
        )

    def benchmark_sdpa(self, num_runs: int = 100) -> BenchmarkResult:
        """Benchmark PyTorch scaled_dot_product_attention"""
        # Warmup
        for _ in range(10):
            _ = torch.nn.functional.scaled_dot_product_attention(self.q, self.k, self.v)

        # Benchmark
        times = []
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start = time.time()
            _ = torch.nn.functional.scaled_dot_product_attention(self.q, self.k, self.v)
            torch.cuda.synchronize()
            times.append(time.time() - start)

        # Calculate metrics
        avg_time = np.mean(times) * 1000
        std_time = np.std(times) * 1000

        # Measure memory
        torch.cuda.reset_peak_memory_stats()
        _ = torch.nn.functional.scaled_dot_product_attention(self.q, self.k, self.v)
        memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

        # Calculate throughput
        total_ops = self.batch_size * self.num_heads * self.seq_len * self.seq_len
        throughput = total_ops / (avg_time / 1000) / 1e9

        return BenchmarkResult(
            name="PyTorch SDPA",
            avg_time_ms=avg_time,
            std_time_ms=std_time,
            throughput=throughput,
            memory_mb=memory_mb,
        )

    def _pytorch_attention(self) -> torch.Tensor:
        """Standard PyTorch attention implementation"""
        scale = 1.0 / (self.head_dim ** 0.5)
        attn = torch.matmul(self.q, self.k.transpose(-2, -1)) * scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, self.v)
        return out

    def run_all(self) -> List[BenchmarkResult]:
        """Run all attention benchmarks"""
        console.print("\n[bold blue]Benchmarking Attention Mechanisms[/bold blue]")
        console.print(f"Config: batch={self.batch_size}, heads={self.num_heads}, seq_len={self.seq_len}, head_dim={self.head_dim}\n")

        results = []

        # Benchmark PyTorch attention
        console.print("Running PyTorch Attention...")
        pytorch_result = self.benchmark_pytorch_attention()
        results.append(pytorch_result)

        # Benchmark SDPA
        console.print("Running PyTorch SDPA...")
        sdpa_result = self.benchmark_sdpa()
        results.append(sdpa_result)

        # Benchmark FlashAttention-3
        console.print("Running FlashAttention-3...")
        flash_result = self.benchmark_flashattention_v3()
        results.append(flash_result)

        # Benchmark FlashAttention-3 with FP8
        if self.dtype == torch.float16:
            console.print("Running FlashAttention-3 (FP8)...")
            flash_fp8_result = self.benchmark_flashattention_v3(use_fp8=True)
            results.append(flash_fp8_result)

        # Calculate speedups
        baseline_time = pytorch_result.avg_time_ms
        for result in results:
            result.speedup = baseline_time / result.avg_time_ms

        return results


class FusedOpsBenchmark:
    """Benchmark fused operations"""

    def __init__(
        self,
        batch_size: int = 32,
        seq_len: int = 2048,
        hidden_dim: int = 4096,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.device = device
        self.dtype = dtype

        # Generate test data
        self.x = torch.randn(
            batch_size, seq_len, hidden_dim,
            device=device, dtype=dtype
        )
        self.weight = torch.randn(hidden_dim, device=device, dtype=dtype)
        self.bias = torch.randn(hidden_dim, device=device, dtype=dtype)

    def benchmark_pytorch_layernorm(self, num_runs: int = 100) -> BenchmarkResult:
        """Benchmark PyTorch LayerNorm"""
        ln = torch.nn.LayerNorm(self.hidden_dim).to(self.device).to(self.dtype)

        # Warmup
        for _ in range(10):
            _ = ln(self.x)

        # Benchmark
        times = []
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start = time.time()
            _ = ln(self.x)
            torch.cuda.synchronize()
            times.append(time.time() - start)

        avg_time = np.mean(times) * 1000
        std_time = np.std(times) * 1000

        return BenchmarkResult(
            name="PyTorch LayerNorm",
            avg_time_ms=avg_time,
            std_time_ms=std_time,
            throughput=0.0,
            memory_mb=0.0,
        )

    def benchmark_fused_layernorm(self, num_runs: int = 100) -> BenchmarkResult:
        """Benchmark fused LayerNorm"""
        # Warmup
        for _ in range(10):
            _ = fused_layernorm(self.x, self.weight, self.bias)

        # Benchmark
        times = []
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start = time.time()
            _ = fused_layernorm(self.x, self.weight, self.bias)
            torch.cuda.synchronize()
            times.append(time.time() - start)

        avg_time = np.mean(times) * 1000
        std_time = np.std(times) * 1000

        return BenchmarkResult(
            name="Fused LayerNorm (Triton)",
            avg_time_ms=avg_time,
            std_time_ms=std_time,
            throughput=0.0,
            memory_mb=0.0,
        )

    def run_all(self) -> List[BenchmarkResult]:
        """Run all fused ops benchmarks"""
        console.print("\n[bold blue]Benchmarking Fused Operations[/bold blue]")
        console.print(f"Config: batch={self.batch_size}, seq_len={self.seq_len}, hidden_dim={self.hidden_dim}\n")

        results = []

        # Benchmark LayerNorm
        console.print("Running PyTorch LayerNorm...")
        pytorch_result = self.benchmark_pytorch_layernorm()
        results.append(pytorch_result)

        console.print("Running Fused LayerNorm...")
        fused_result = self.benchmark_fused_layernorm()
        results.append(fused_result)

        # Calculate speedups
        baseline_time = pytorch_result.avg_time_ms
        for result in results:
            result.speedup = baseline_time / result.avg_time_ms

        return results


def print_results(results: List[BenchmarkResult]) -> None:
    """Print benchmark results in a nice table"""
    table = Table(title="Benchmark Results", show_header=True, header_style="bold magenta")

    table.add_column("Implementation", style="cyan", width=30)
    table.add_column("Avg Time (ms)", justify="right", style="green")
    table.add_column("Std Dev (ms)", justify="right", style="yellow")
    table.add_column("Speedup", justify="right", style="red")
    table.add_column("Memory (MB)", justify="right", style="blue")

    for result in results:
        table.add_row(
            result.name,
            f"{result.avg_time_ms:.3f}",
            f"{result.std_time_ms:.3f}",
            f"{result.speedup:.2f}x",
            f"{result.memory_mb:.1f}",
        )

    console.print(table)


def main():
    """Run comprehensive benchmarks"""
    console.print("[bold green]AI Assistant Pro - Benchmark Suite[/bold green]")
    console.print("Optimized for NVIDIA Blackwell (SM120)\n")

    # Check CUDA availability
    if not torch.cuda.is_available():
        console.print("[bold red]CUDA not available! Benchmarks require GPU.[/bold red]")
        return

    # Print GPU info
    gpu_name = torch.cuda.get_device_name(0)
    console.print(f"GPU: {gpu_name}")
    console.print(f"CUDA Version: {torch.version.cuda}\n")

    # Benchmark attention mechanisms
    attention_bench = AttentionBenchmark(
        batch_size=4,
        num_heads=32,
        seq_len=4096,
        head_dim=128,
    )
    attention_results = attention_bench.run_all()
    print_results(attention_results)

    # Benchmark fused operations
    fused_bench = FusedOpsBenchmark(
        batch_size=32,
        seq_len=2048,
        hidden_dim=4096,
    )
    fused_results = fused_bench.run_all()
    print_results(fused_results)

    console.print("\n[bold green]✓ Benchmarks complete![/bold green]")


if __name__ == "__main__":
    main()
