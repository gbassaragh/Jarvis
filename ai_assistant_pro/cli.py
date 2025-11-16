"""
Command-line interface for AI Assistant Pro

Provides easy access to all framework features via CLI.
"""

from pathlib import Path
from rich.console import Console
from rich.table import Table
import time

import click
import torch

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """AI Assistant Pro - High-performance inference framework for NVIDIA Blackwell"""
    pass


@cli.command()
@click.option("--model", "-m", default="gpt2", help="Model name from HuggingFace")
@click.option("--prompt", "-p", required=True, help="Input prompt")
@click.option("--max-tokens", default=100, help="Maximum tokens to generate")
@click.option("--temperature", default=0.8, help="Sampling temperature")
@click.option("--use-triton/--no-triton", default=True, help="Use Triton kernels")
@click.option("--use-fp8/--no-fp8", default=False, help="Use FP8 quantization")
def generate(model, prompt, max_tokens, temperature, use_triton, use_fp8):
    """Generate text from a prompt"""
    from ai_assistant_pro import AssistantEngine

    console.print(f"[bold]Loading model:[/bold] {model}")

    engine = AssistantEngine(
        model_name=model,
        use_triton=use_triton,
        use_fp8=use_fp8,
        enable_paged_attention=True,
    )

    console.print(f"\n[bold]Prompt:[/bold] {prompt}")
    console.print("[dim]Generating...[/dim]\n")

    response = engine.generate(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    console.print(f"[bold green]Response:[/bold green]\n{response}")


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
@click.option("--model", "-m", default="gpt2", help="Model to serve")
@click.option("--use-triton/--no-triton", default=True, help="Use Triton kernels")
@click.option("--use-fp8/--no-fp8", default=False, help="Use FP8 quantization")
@click.option("--dry-run", is_flag=True, help="Validate config and exit without starting server")
def serve(host, port, model, use_triton, use_fp8, dry_run):
    """Start API server"""
    from ai_assistant_pro.serving import serve as start_server

    console.print(f"[bold]Starting API server...[/bold]")
    console.print(f"  Model: {model}")
    console.print(f"  Host: {host}")
    console.print(f"  Port: {port}")
    console.print(f"  Triton: {use_triton}")
    console.print(f"  FP8: {use_fp8}")

    if dry_run:
        console.print("[dim]Dry run only; server not started.[/dim]")
        return

    start_server(
        host=host,
        port=port,
        model_name=model,
        use_triton=use_triton,
        use_fp8=use_fp8,
    )


@cli.command()
@click.option("--seq-lengths", default="128,512,1024,2048", help="Comma-separated sequence lengths")
@click.option("--model", "-m", default="gpt2", help="Model to benchmark")
def benchmark(seq_lengths, model):
    """Run performance benchmarks"""
    from ai_assistant_pro import AssistantEngine

    seq_lens = [int(x) for x in seq_lengths.split(",")]

    console.print("[bold]Running benchmarks...[/bold]")
    console.print(f"  Model: {model}")
    console.print(f"  Sequence lengths: {seq_lens}")

    engine = AssistantEngine(model_name=model, use_triton=True)

    results = engine.benchmark(seq_lengths=seq_lens)

    # Display results
    table = Table(title="Benchmark Results", show_header=True)
    table.add_column("Sequence Length", style="cyan")
    table.add_column("Avg Time (ms)", justify="right", style="green")
    table.add_column("Tokens/sec", justify="right", style="yellow")

    for seq_len in seq_lens:
        key = f"seq_len_{seq_len}"
        if key in results:
            table.add_row(
                str(seq_len),
                f"{results[key]['avg_time_ms']:.2f}",
                f"{results[key]['tokens_per_sec']:.2f}",
            )

    console.print("\n")
    console.print(table)


@cli.command()
@click.option("--n-candidates", default=1000, help="Number of candidates")
@click.option("--embedding-dim", default=768, help="Embedding dimension")
@click.option(
    "--device",
    default="cuda",
    help="Device to run benchmarks on (cuda or cpu). Defaults to cuda if available.",
)
def srf_benchmark(n_candidates, embedding_dim, device):
    """Run SRF benchmarks"""
    from benchmarks.srf_benchmark import SRFBenchmark, SRFQualityBenchmark

    console.print("[bold]Running SRF benchmarks...[/bold]")

    if device == "cuda" and not torch.cuda.is_available():
        console.print("[yellow]CUDA not available; falling back to CPU.[/yellow]")
        device = "cpu"

    # Performance benchmark
    perf_bench = SRFBenchmark(
        n_candidates=n_candidates,
        embedding_dim=embedding_dim,
        device=device,
    )
    perf_bench.run_all_benchmarks()

    # Quality benchmark
    if n_candidates <= 1000:
        quality_bench = SRFQualityBenchmark(n_candidates=n_candidates // 2)
        quality_bench.run_quality_benchmark()


@cli.command()
@click.argument("config_file", type=click.Path(exists=True))
@click.option("--output", "-o", default="config_summary.txt", help="Output file")
def validate_config(config_file, output):
    """Validate configuration file"""
    import yaml

    with open(config_file) as f:
        config = yaml.safe_load(f)

    console.print(f"[bold]Validating configuration:[/bold] {config_file}")

    # Validate SRF config
    if "srf" in config:
        from ai_assistant_pro.srf import SRFConfig

        try:
            srf_config = SRFConfig(**config["srf"])
            srf_config.validate()
            console.print("  [green]✓[/green] SRF config valid")
        except Exception as e:
            console.print(f"  [red]✗[/red] SRF config invalid: {e}")

    # Validate engine config
    if "engine" in config:
        console.print("  [green]✓[/green] Engine config found")

    console.print(f"\n[bold]Summary written to:[/bold] {output}")


@cli.group()
def jarvis():
    """JARVIS - Complete AI assistant (voice, memory, tools, RAG)"""
    pass


@jarvis.command()
@click.option("--model", "-m", default="gpt2", help="Model name")
@click.option("--user-id", "-u", default="default", help="User identifier")
@click.option("--enable-memory/--no-memory", default=True, help="Enable conversational memory")
@click.option("--enable-tools/--no-tools", default=True, help="Enable tool use")
@click.option("--enable-rag/--no-rag", default=False, help="Enable RAG knowledge base")
@click.option("--use-triton/--no-triton", default=True, help="Use Triton kernels")
@click.option("--use-fp8/--no-fp8", default=False, help="Use FP8 quantization")
@click.option("--max-turns", type=int, default=None, help="Auto-exit after N turns (CI-safe).")
@click.option(
    "--exit-after-seconds",
    type=float,
    default=None,
    help="Auto-exit after N seconds without waiting for input (CI-safe).",
)
@click.option("--dry-run", is_flag=True, help="Initialize and exit without starting chat loop.")
@click.option("--exit-phrase", default="quit", show_default=True, help="Phrase to exit the loop.")
def chat(
    model,
    user_id,
    enable_memory,
    enable_tools,
    enable_rag,
    use_triton,
    use_fp8,
    max_turns,
    exit_after_seconds,
    dry_run,
    exit_phrase,
):
    """Start interactive JARVIS chat"""
    from ai_assistant_pro.jarvis import JARVIS

    console.print("[bold]Initializing JARVIS...[/bold]")

    jarvis_instance = JARVIS(
        model_name=model,
        user_id=user_id,
        enable_voice=False,
        enable_tools=enable_tools,
        enable_rag=enable_rag,
        enable_memory=enable_memory,
        use_triton=use_triton,
        use_fp8=use_fp8,
    )

    console.print("[green]✓[/green] JARVIS ready!\n")
    console.print(f"Type '{exit_phrase}' to exit, 'stats' for statistics\n")

    if dry_run:
        console.print("[dim]Dry run only; chat loop not started.[/dim]")
        return

    start_time = time.perf_counter()
    turns = 0

    while True:
        try:
            if exit_after_seconds is not None and (time.perf_counter() - start_time) >= exit_after_seconds:
                console.print("\n[dim]Auto-exit after timeout.[/dim]")
                break

            if max_turns is not None and turns >= max_turns:
                console.print("\n[dim]Auto-exit after max turns.[/dim]")
                break

            message = input("You: ")

            if message.lower() == exit_phrase.lower():
                break

            if message.lower() == "stats":
                stats = jarvis_instance.get_stats()
                console.print("\n[bold]Statistics:[/bold]")
                for key, value in stats.items():
                    console.print(f"  {key}: {value}")
                console.print()
                continue

            result = jarvis_instance.chat(
                message,
                use_memory=enable_memory,
                use_tools=enable_tools,
                use_rag=enable_rag,
            )

            console.print(f"\n[bold green]JARVIS:[/bold green] {result['response']}")

            if result["tool_results"]:
                console.print("\n[dim]Tools used:[/dim]")
                for tr in result["tool_results"]:
                    console.print(f"  - {tr['tool']}: {tr['result']}")

            if result["rag_sources"]:
                console.print(f"\n[dim]Retrieved {len(result['rag_sources'])} knowledge sources[/dim]")

            console.print()

            turns += 1

        except KeyboardInterrupt:
            break

    console.print("\n[bold]Goodbye![/bold]")


@jarvis.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8080, help="Port to bind to")
@click.option("--model", "-m", default="gpt2", help="Model name")
@click.option("--use-triton/--no-triton", default=True, help="Use Triton kernels")
@click.option("--use-fp8/--no-fp8", default=False, help="Use FP8 quantization")
@click.option("--dry-run", is_flag=True, help="Validate and exit without starting web server")
def serve(host, port, model, use_triton, use_fp8, dry_run):
    """Start JARVIS web interface"""
    from ai_assistant_pro.jarvis import JARVIS

    console.print("[bold]Starting JARVIS web interface...[/bold]")

    jarvis_instance = JARVIS(
        model_name=model,
        enable_voice=False,
        enable_tools=True,
        enable_rag=True,
        enable_memory=True,
        use_triton=use_triton,
        use_fp8=use_fp8,
    )

    console.print(f"\n[green]✓[/green] JARVIS initialized")
    console.print(f"[bold]Web interface:[/bold] http://{host}:{port}")
    console.print("\nPress Ctrl+C to stop\n")

    if dry_run:
        console.print("[dim]Dry run only; web interface not started.[/dim]")
        return

    jarvis_instance.start_web_interface(host=host, port=port)


@jarvis.command()
@click.option("--model", "-m", default="gpt2", help="Model name")
@click.option("--user-id", "-u", default="default", help="User identifier")
@click.option("--dry-run", is_flag=True, help="Initialize and exit without starting mic loop")
def voice(model, user_id, dry_run):
    """Start JARVIS voice assistant (requires microphone)"""
    from ai_assistant_pro.jarvis import JARVIS

    console.print("[bold]Starting JARVIS voice assistant...[/bold]")

    jarvis_instance = JARVIS(
        model_name=model,
        user_id=user_id,
        enable_voice=True,
        enable_tools=True,
        enable_memory=True,
    )

    console.print("\n[green]✓[/green] JARVIS ready!")
    console.print("[bold]Say 'Jarvis' followed by your command[/bold]")
    console.print("Press Ctrl+C to stop\n")

    if dry_run:
        console.print("[dim]Dry run only; voice assistant not started.[/dim]")
        return

    jarvis_instance.start_voice_assistant()


@jarvis.command()
@click.option("--model", "-m", default="gpt2", help="Model name")
@click.option("--user-id", "-u", default="default", help="User identifier")
@click.option("--config", "-c", help="Config file path")
@click.option(
    "--wake-words",
    "-w",
    multiple=True,
    default=["hi", "hello", "jarvis"],
    help="Wake words for activation",
)
@click.option(
    "--no-proactive",
    is_flag=True,
    help="Disable proactive greetings",
)
def daemon(model, user_id, config, wake_words, no_proactive):
    """Start always-on JARVIS daemon (real-life JARVIS experience)"""
    from ai_assistant_pro.jarvis.daemon import JARVISDaemon

    console.print("[bold cyan]🚀 Starting JARVIS Daemon...[/bold cyan]\n")
    console.print("[dim]Creating real-life AI assistant experience...[/dim]\n")

    daemon_instance = JARVISDaemon(
        user_id=user_id,
        model_name=model,
        wake_words=list(wake_words),
        enable_proactive=not no_proactive,
        config_path=config,
    )

    daemon_instance.run()


@jarvis.command()
@click.argument("directory", type=click.Path(exists=True))
@click.option("--pattern", default="*.md", help="File pattern to load")
@click.option("--model", "-m", default="gpt2", help="Model name")
def load_knowledge(directory, pattern, model):
    """Load knowledge base from directory"""
    from ai_assistant_pro.jarvis import JARVIS

    console.print(f"[bold]Loading knowledge from:[/bold] {directory}")
    console.print(f"[bold]Pattern:[/bold] {pattern}\n")

    jarvis_instance = JARVIS(
        model_name=model,
        enable_rag=True,
    )

    jarvis_instance.load_knowledge_base(directory, pattern=pattern)

    stats = jarvis_instance.get_stats()
    console.print(f"\n[green]✓[/green] Loaded {stats['rag']['num_documents']} documents")
    console.print(f"  Total chunks: {stats['rag']['num_chunks']}")


@jarvis.command()
@click.argument("knowledge_text")
@click.option("--importance", default=0.7, type=float, help="Importance score (0-1)")
@click.option("--topic", help="Topic/category")
@click.option("--model", "-m", default="gpt2", help="Model name")
def add_knowledge(knowledge_text, importance, topic, model):
    """Add single knowledge item to JARVIS"""
    from ai_assistant_pro.jarvis import JARVIS

    jarvis_instance = JARVIS(
        model_name=model,
        enable_rag=True,
    )

    metadata = {"topic": topic} if topic else {}

    jarvis_instance.add_knowledge(
        content=knowledge_text,
        metadata=metadata,
        importance=importance,
    )

    console.print(f"[green]✓[/green] Added knowledge item")
    console.print(f"  Text: {knowledge_text[:100]}...")
    console.print(f"  Importance: {importance}")


@jarvis.command()
@click.option("--model", "-m", default="gpt2", help="Model name")
@click.option("--user-id", "-u", default="default", help="User identifier")
def stats(model, user_id):
    """Show JARVIS statistics"""
    from ai_assistant_pro.jarvis import JARVIS

    jarvis_instance = JARVIS(
        model_name=model,
        user_id=user_id,
        enable_memory=True,
        enable_tools=True,
        enable_rag=True,
    )

    stats_data = jarvis_instance.get_stats()

    console.print("\n[bold]JARVIS Statistics[/bold]\n")

    console.print(f"User: {stats_data['user_id']}")
    console.print(f"Model: {stats_data['model']}")

    console.print("\n[bold]Components:[/bold]")
    for component, enabled in stats_data["components"].items():
        status = "[green]✓[/green]" if enabled else "[red]✗[/red]"
        console.print(f"  {status} {component}")

    if "memory" in stats_data:
        console.print("\n[bold]Memory:[/bold]")
        for key, value in stats_data["memory"].items():
            console.print(f"  {key}: {value}")

    if "tools" in stats_data:
        console.print("\n[bold]Tools:[/bold]")
        console.print(f"  Count: {stats_data['tools']['count']}")
        console.print(f"  Available: {', '.join(stats_data['tools']['available'])}")

    if "rag" in stats_data:
        console.print("\n[bold]RAG:[/bold]")
        for key, value in stats_data["rag"].items():
            console.print(f"  {key}: {value}")

    console.print()


@cli.command()
def info():
    """Display system information"""
    console.print("[bold]AI Assistant Pro - System Information[/bold]\n")

    # PyTorch version
    console.print(f"PyTorch version: {torch.__version__}")

    # CUDA
    if torch.cuda.is_available():
        console.print(f"CUDA available: [green]Yes[/green]")
        console.print(f"CUDA version: {torch.version.cuda}")
        console.print(f"GPU: {torch.cuda.get_device_name(0)}")
        console.print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

        # Compute capability
        cc = torch.cuda.get_device_capability(0)
        sm = cc[0] * 10 + cc[1]
        console.print(f"Compute capability: {cc[0]}.{cc[1]} (SM{sm})")

        if sm >= 120:
            console.print("  [bold green]✓ Blackwell (SM120+) - All features available[/bold green]")
        elif sm >= 90:
            console.print("  [yellow]⚠ Hopper (SM90+) - Most features available[/yellow]")
        else:
            console.print("  [red]⚠ Older GPU - Limited feature support[/red]")
    else:
        console.print(f"CUDA available: [red]No[/red]")

    # Triton
    try:
        import triton
        console.print(f"\nTriton version: {triton.__version__}")
    except ImportError:
        console.print("\nTriton: [red]Not installed[/red]")


@cli.command()
@click.option("--n-candidates", default=100, help="Number of test candidates")
@click.option("--query", "-q", default="Test query", help="Query text")
def srf_demo(n_candidates, query):
    """Interactive SRF demonstration"""
    from ai_assistant_pro.srf import StoneRetrievalFunction, SRFConfig, MemoryCandidate
    import time

    console.print("[bold]Stone Retrieval Function - Interactive Demo[/bold]\n")

    # Create SRF
    config = SRFConfig(alpha=0.3, beta=0.2, gamma=0.25, delta=0.15)
    srf = StoneRetrievalFunction(config)

    console.print(f"Creating {n_candidates} test candidates...")

    # Add test candidates
    for i in range(n_candidates):
        candidate = MemoryCandidate(
            id=i,
            content=torch.randn(768),
            text=f"Memory {i}: " + ("Important" if i < 10 else "Regular"),
            emotional_score=0.9 if i < 10 else 0.3,
            timestamp=time.time() - (i * 60),  # Older as i increases
        )
        srf.add_candidate(candidate)

    console.print(f"[green]✓[/green] Added {len(srf)} candidates\n")

    # Retrieve
    console.print(f"[bold]Query:[/bold] {query}")
    query_emb = torch.randn(768)
    results = srf.retrieve(query_emb, top_k=10)

    # Display results
    table = Table(title="Top-10 Retrieved Memories", show_header=True)
    table.add_column("Rank", style="cyan", justify="right")
    table.add_column("Memory", style="white", width=40)
    table.add_column("Score", style="green", justify="right")
    table.add_column("Components", style="yellow")

    for i, result in enumerate(results, 1):
        components_str = (
            f"S:{result.components['semantic']:.2f} "
            f"E:{result.components['emotional']:.2f} "
            f"A:{result.components['associative']:.2f} "
            f"R:{result.components['recency']:.2f} "
            f"D:{result.components['decay']:.2f}"
        )

        table.add_row(
            str(i),
            result.candidate.text,
            f"{result.score:.3f}",
            components_str,
        )

    console.print("\n")
    console.print(table)

    # Statistics
    stats = srf.get_statistics()
    console.print(f"\n[bold]Statistics:[/bold]")
    console.print(f"  Total candidates: {stats['total_candidates']}")
    console.print(f"  Retrievals performed: {stats['total_retrievals']}")


if __name__ == "__main__":
    cli()
