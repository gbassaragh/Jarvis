# Examples

This directory contains examples demonstrating various features of AI Assistant Pro.

## Basic Usage

**File:** `basic_usage.py`

Simple example showing:
- Engine initialization
- Text generation
- Performance monitoring

```bash
python examples/basic_usage.py
```

## Advanced Usage

**File:** `advanced_usage.py`

Advanced features including:
- Custom Triton kernels
- FlashAttention-3 demonstration
- Fused operations
- FP8 quantization
- Performance comparison

```bash
python examples/advanced_usage.py
```

Requires: CUDA-enabled GPU

## API Server

**File:** `api_server.py`

FastAPI server with:
- RESTful API
- OpenAI-compatible endpoints
- Health checks
- Statistics

Start server:
```bash
python examples/api_server.py
```

Test client:
```bash
python examples/api_server.py client
```

## Benchmarks

See `benchmarks/benchmark.py` for comprehensive performance benchmarks.

```bash
python benchmarks/benchmark.py
```

## Requirements

All examples require:
- PyTorch 2.5+
- CUDA 12.4+ (for GPU examples)
- Triton 3.0+

Install dependencies:
```bash
pip install -r requirements.txt
```

## Notes

- For SM120 (Blackwell) GPUs, all optimizations will be available
- For SM90 (Hopper) GPUs, most optimizations work (FP8 support varies)
- For older GPUs, the framework falls back to standard implementations

## Performance

Expected speedups on SM120:
- FlashAttention-3: **10x faster** than PyTorch
- FP8 mode: **2x additional speedup**
- Fused operations: **2-3x faster**
- Paged KV-cache: **50% memory reduction**
