"""
Basic usage example for AI Assistant Pro

Demonstrates:
- Loading the engine
- Generating text
- Using custom kernels
- Performance monitoring
"""

from ai_assistant_pro import AssistantEngine


def main():
    print("AI Assistant Pro - Basic Usage Example")
    print("=" * 50)

    # Initialize engine with SM120 optimizations
    print("\n1. Initializing engine...")
    engine = AssistantEngine(
        model_name="gpt2",  # Use a small model for demo
        use_triton=True,  # Enable custom Triton kernels
        use_fp8=False,  # FP8 requires supported hardware
        enable_paged_attention=True,  # Enable paged KV-cache
        max_batch_size=32,
        device="cuda",
    )

    # Generate text
    print("\n2. Generating text...")
    prompt = "The future of AI is"
    response = engine.generate(
        prompt=prompt,
        max_tokens=100,
        temperature=0.8,
        top_p=0.9,
    )

    print(f"\nPrompt: {prompt}")
    print(f"Response: {response}")

    # Get engine statistics
    print("\n3. Engine statistics:")
    stats = engine.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Benchmark performance
    print("\n4. Running benchmark...")
    benchmark_results = engine.benchmark(seq_lengths=[128, 512, 1024])

    print("\nBenchmark Results:")
    for seq_len, metrics in benchmark_results.items():
        print(f"  {seq_len}:")
        print(f"    Avg time: {metrics['avg_time_ms']:.2f} ms")
        print(f"    Tokens/sec: {metrics['tokens_per_sec']:.2f}")

    print("\n✓ Example complete!")


if __name__ == "__main__":
    main()
