"""
Benchmarks for Stone Retrieval Function (SRF)

Compares SRF against baseline retrieval methods and demonstrates
performance characteristics.
"""

import torch
import time
import numpy as np
from typing import List, Dict
from rich.console import Console
from rich.table import Table
from rich.progress import track

from ai_assistant_pro.srf import (
    StoneRetrievalFunction,
    SRFConfig,
    MemoryCandidate,
)
from ai_assistant_pro.srf.kernels import srf_batch_score


console = Console()


class SRFBenchmark:
    """Benchmark suite for SRF"""

    def __init__(
        self,
        n_candidates: int = 1000,
        embedding_dim: int = 768,
        device: str = "cuda",
    ):
        self.n_candidates = n_candidates
        self.embedding_dim = embedding_dim
        self.device = device

        # Generate test data
        console.print(f"\n[bold]Generating {n_candidates} test candidates...[/bold]")
        self.embeddings = torch.randn(
            n_candidates, embedding_dim, device=device, dtype=torch.float16
        )
        self.emotional_scores = torch.rand(n_candidates, device=device)
        self.timestamps = torch.tensor([
            time.time() - np.random.exponential(1800) for _ in range(n_candidates)
        ], device=device)
        self.last_access = torch.tensor([
            time.time() - np.random.exponential(900) for _ in range(n_candidates)
        ], device=device)

        self.query = torch.randn(embedding_dim, device=device, dtype=torch.float16)

    def benchmark_baseline_semantic(self, num_runs: int = 100) -> Dict:
        """Benchmark baseline semantic similarity only"""
        console.print("\n[cyan]Benchmarking baseline (semantic similarity only)...[/cyan]")

        # Warmup
        for _ in range(10):
            _ = torch.nn.functional.cosine_similarity(
                self.query.unsqueeze(0),
                self.embeddings,
                dim=1
            )

        # Benchmark
        times = []
        for _ in track(range(num_runs), description="Running baseline"):
            torch.cuda.synchronize()
            start = time.time()

            scores = torch.nn.functional.cosine_similarity(
                self.query.unsqueeze(0),
                self.embeddings,
                dim=1
            )
            top_scores, top_indices = torch.topk(scores, k=10)

            torch.cuda.synchronize()
            times.append(time.time() - start)

        return {
            "name": "Baseline (Semantic Only)",
            "avg_time_ms": np.mean(times) * 1000,
            "std_time_ms": np.std(times) * 1000,
            "top_scores": top_scores.cpu().tolist(),
        }

    def benchmark_srf_cpu(self, num_runs: int = 10) -> Dict:
        """Benchmark SRF on CPU (slower, for comparison)"""
        console.print("\n[cyan]Benchmarking SRF (CPU)...[/cyan]")

        # Create SRF with CPU
        config = SRFConfig(use_triton=False, use_gpu=False)
        srf = StoneRetrievalFunction(config)

        # Add subset of candidates (CPU is slow)
        n_subset = min(100, self.n_candidates)
        for i in range(n_subset):
            candidate = MemoryCandidate(
                id=i,
                content=self.embeddings[i].cpu(),
                emotional_score=self.emotional_scores[i].item(),
                timestamp=self.timestamps[i].item(),
                last_access=self.last_access[i].item(),
            )
            srf.add_candidate(candidate)

        # Warmup
        for _ in range(3):
            _ = srf.retrieve(self.query.cpu(), top_k=10)

        # Benchmark
        times = []
        for _ in track(range(num_runs), description="Running SRF (CPU)"):
            start = time.time()
            results = srf.retrieve(self.query.cpu(), top_k=10)
            times.append(time.time() - start)

        return {
            "name": f"SRF (CPU, {n_subset} candidates)",
            "avg_time_ms": np.mean(times) * 1000,
            "std_time_ms": np.std(times) * 1000,
            "top_scores": [r.score for r in results],
        }

    def benchmark_srf_gpu_triton(self, num_runs: int = 100) -> Dict:
        """Benchmark SRF with Triton GPU kernels"""
        console.print("\n[cyan]Benchmarking SRF (GPU + Triton)...[/cyan]")

        current_time = time.time()

        # Warmup
        for _ in range(10):
            scores, _ = srf_batch_score(
                query=self.query,
                embeddings=self.embeddings,
                emotional_scores=self.emotional_scores,
                timestamps=self.timestamps,
                last_access=self.last_access,
                current_time=current_time,
                alpha=0.3,
                beta=0.2,
                gamma=0.25,
                delta=0.15,
            )
            _ = torch.topk(scores, k=10)

        # Benchmark
        times = []
        for _ in track(range(num_runs), description="Running SRF (GPU+Triton)"):
            torch.cuda.synchronize()
            start = time.time()

            scores, components = srf_batch_score(
                query=self.query,
                embeddings=self.embeddings,
                emotional_scores=self.emotional_scores,
                timestamps=self.timestamps,
                last_access=self.last_access,
                current_time=current_time,
                alpha=0.3,
                beta=0.2,
                gamma=0.25,
                delta=0.15,
            )
            top_scores, top_indices = torch.topk(scores, k=10)

            torch.cuda.synchronize()
            times.append(time.time() - start)

        return {
            "name": f"SRF (GPU+Triton, {self.n_candidates} candidates)",
            "avg_time_ms": np.mean(times) * 1000,
            "std_time_ms": np.std(times) * 1000,
            "top_scores": top_scores.cpu().tolist(),
        }

    def run_all_benchmarks(self):
        """Run all benchmarks and display results"""
        console.print("\n[bold green]SRF Benchmark Suite[/bold green]")
        console.print(f"Configuration: {self.n_candidates} candidates, {self.embedding_dim}D embeddings")
        console.print(f"Device: {self.device}")

        results = []

        # Baseline
        baseline_result = self.benchmark_baseline_semantic()
        results.append(baseline_result)

        # SRF CPU
        if self.n_candidates <= 1000:  # Only run CPU for small datasets
            cpu_result = self.benchmark_srf_cpu()
            results.append(cpu_result)

        # SRF GPU
        if torch.cuda.is_available():
            gpu_result = self.benchmark_srf_gpu_triton()
            results.append(gpu_result)

        # Display results
        self._display_results(results, baseline_result)

    def _display_results(self, results: List[Dict], baseline: Dict):
        """Display benchmark results in table"""
        table = Table(title="SRF Benchmark Results", show_header=True, header_style="bold magenta")

        table.add_column("Method", style="cyan", width=40)
        table.add_column("Avg Time (ms)", justify="right", style="green")
        table.add_column("Std Dev (ms)", justify="right", style="yellow")
        table.add_column("Speedup", justify="right", style="red")

        baseline_time = baseline["avg_time_ms"]

        for result in results:
            speedup = baseline_time / result["avg_time_ms"]
            table.add_row(
                result["name"],
                f"{result['avg_time_ms']:.3f}",
                f"{result['std_time_ms']:.3f}",
                f"{speedup:.1f}x" if speedup != 1.0 else "1.0x (baseline)",
            )

        console.print("\n")
        console.print(table)


class SRFQualityBenchmark:
    """Benchmark retrieval quality of SRF vs baselines"""

    def __init__(self, n_candidates: int = 500, embedding_dim: int = 768):
        self.n_candidates = n_candidates
        self.embedding_dim = embedding_dim

        console.print(f"\n[bold]Generating quality test dataset...[/bold]")

        # Create candidates with known important/relevant items
        self.candidates = []
        self.important_indices = set()

        for i in range(n_candidates):
            # Mark 10% as "important"
            is_important = i < n_candidates // 10

            embedding = torch.randn(embedding_dim)
            candidate = MemoryCandidate(
                id=i,
                content=embedding,
                text=f"Candidate {i}" + (" [IMPORTANT]" if is_important else ""),
                emotional_score=0.9 if is_important else np.random.uniform(0.1, 0.5),
                timestamp=time.time() - np.random.uniform(0, 3600),
            )

            self.candidates.append(candidate)
            if is_important:
                self.important_indices.add(i)

        # Create query biased toward important candidates
        important_embeddings = [
            self.candidates[i].content for i in self.important_indices
        ]
        self.query = torch.stack(important_embeddings).mean(dim=0)
        self.query += torch.randn(embedding_dim) * 0.2  # Add noise

    def evaluate_baseline(self) -> Dict:
        """Evaluate baseline semantic similarity"""
        console.print("\n[cyan]Evaluating baseline retrieval quality...[/cyan]")

        # Compute semantic similarity
        embeddings = torch.stack([c.content for c in self.candidates])
        scores = torch.nn.functional.cosine_similarity(
            self.query.unsqueeze(0),
            embeddings,
            dim=1
        )

        # Get top-k
        top_k = 50
        _, top_indices = torch.topk(scores, k=top_k)
        top_indices = top_indices.tolist()

        # Calculate metrics
        important_retrieved = len(set(top_indices) & self.important_indices)
        precision = important_retrieved / top_k
        recall = important_retrieved / len(self.important_indices)

        return {
            "name": "Baseline (Semantic)",
            "precision": precision,
            "recall": recall,
            "important_retrieved": important_retrieved,
        }

    def evaluate_srf(self) -> Dict:
        """Evaluate SRF retrieval quality"""
        console.print("\n[cyan]Evaluating SRF retrieval quality...[/cyan]")

        # Create SRF
        config = SRFConfig(
            alpha=0.3,
            beta=0.2,
            gamma=0.25,
            delta=0.15,
            use_triton=False,
            use_gpu=False,
        )
        srf = StoneRetrievalFunction(config)

        # Add all candidates
        for candidate in self.candidates:
            srf.add_candidate(candidate)

        # Retrieve
        top_k = 50
        results = srf.retrieve(self.query, top_k=top_k)

        # Calculate metrics
        top_indices = {r.candidate.id for r in results}
        important_retrieved = len(top_indices & self.important_indices)
        precision = important_retrieved / top_k
        recall = important_retrieved / len(self.important_indices)

        return {
            "name": "SRF",
            "precision": precision,
            "recall": recall,
            "important_retrieved": important_retrieved,
        }

    def run_quality_benchmark(self):
        """Run quality benchmark"""
        console.print("\n[bold green]SRF Quality Benchmark[/bold green]")
        console.print(f"Dataset: {self.n_candidates} candidates, {len(self.important_indices)} important")

        baseline_result = self.evaluate_baseline()
        srf_result = self.evaluate_srf()

        # Display results
        table = Table(title="Retrieval Quality Comparison", show_header=True)

        table.add_column("Method", style="cyan")
        table.add_column("Precision@50", justify="right", style="green")
        table.add_column("Recall@50", justify="right", style="yellow")
        table.add_column("Important Retrieved", justify="right", style="magenta")
        table.add_column("Improvement", justify="right", style="red")

        for result in [baseline_result, srf_result]:
            improvement = ""
            if result["name"] == "SRF":
                prec_improvement = (
                    (result["precision"] - baseline_result["precision"])
                    / baseline_result["precision"]
                    * 100
                )
                improvement = f"+{prec_improvement:.1f}%"

            table.add_row(
                result["name"],
                f"{result['precision']:.3f}",
                f"{result['recall']:.3f}",
                f"{result['important_retrieved']}/{len(self.important_indices)}",
                improvement,
            )

        console.print("\n")
        console.print(table)


def main():
    """Run all SRF benchmarks"""
    console.print("[bold blue]Stone Retrieval Function - Comprehensive Benchmarks[/bold blue]")

    if not torch.cuda.is_available():
        console.print("[yellow]Warning: CUDA not available. Some benchmarks will be skipped.[/yellow]")

    # Performance benchmarks
    console.print("\n[bold]Part 1: Performance Benchmarks[/bold]")
    perf_bench = SRFBenchmark(n_candidates=1000, embedding_dim=768)
    perf_bench.run_all_benchmarks()

    # Quality benchmarks
    console.print("\n[bold]Part 2: Quality Benchmarks[/bold]")
    quality_bench = SRFQualityBenchmark(n_candidates=500)
    quality_bench.run_quality_benchmark()

    console.print("\n[bold green]✓ All SRF benchmarks complete![/bold green]")


if __name__ == "__main__":
    main()
