# Quick Start Guide

Get started with AI Assistant Pro in 5 minutes!

## Installation

### Requirements

- Python 3.10+
- NVIDIA GPU with CUDA 12.4+
- PyTorch 2.5+
- Triton 3.0+

### Install from source

```bash
git clone https://github.com/ai-assistant-pro/ai-assistant-pro.git
cd ai-assistant-pro
pip install -e .
```

### Install dependencies

```bash
pip install -r requirements.txt
```

## Basic Usage

### 1. Initialize Engine

```python
from ai_assistant_pro import AssistantEngine

engine = AssistantEngine(
    model_name="gpt2",
    use_triton=True,      # Enable custom kernels
    use_fp8=True,         # Enable FP8 (SM120 only)
    enable_paged_attention=True,
)
```

### 2. Generate Text

```python
response = engine.generate(
    prompt="The future of AI is",
    max_tokens=100,
    temperature=0.8,
)

print(response)
```

### 3. Monitor Performance

```python
stats = engine.get_stats()
print(stats)
```

## Using Custom Kernels

### FlashAttention-3

```python
from ai_assistant_pro.kernels.attention import flashattention_v3
import torch

q = torch.randn(2, 8, 1024, 64, device="cuda", dtype=torch.float16)
k = torch.randn(2, 8, 1024, 64, device="cuda", dtype=torch.float16)
v = torch.randn(2, 8, 1024, 64, device="cuda", dtype=torch.float16)

# Standard mode
out = flashattention_v3(q, k, v, use_fp8=False)

# FP8 mode (2x faster on SM120)
out_fp8 = flashattention_v3(q, k, v, use_fp8=True)
```

### Fused Operations

```python
from ai_assistant_pro.kernels.fused_ops import fused_layernorm, fused_gelu
import torch

x = torch.randn(4, 512, 768, device="cuda", dtype=torch.float16)
weight = torch.randn(768, device="cuda", dtype=torch.float16)
bias = torch.randn(768, device="cuda", dtype=torch.float16)

# Fused LayerNorm (2x faster)
out_ln = fused_layernorm(x, weight, bias)

# Fused GELU (3x faster)
out_gelu = fused_gelu(x)
```

### FP8 Quantization

```python
from ai_assistant_pro.kernels.quantization import quantize_fp8, dequantize_fp8
import torch

x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)

# Quantize to FP8
x_fp8, scale = quantize_fp8(x)

# Dequantize back
x_recovered = dequantize_fp8(x_fp8, scale)
```

## Running Benchmarks

```bash
python benchmarks/benchmark.py
```

Expected results on SM120:
- FlashAttention-3: **10x faster** than PyTorch
- FP8 mode: **2x additional speedup**
- Fused LayerNorm: **2x faster**
- Fused GELU: **3x faster**

## Starting API Server

### Start server

```bash
python -m ai_assistant_pro.serving.server
```

Or use the convenience function:

```python
from ai_assistant_pro.serving import serve

serve(
    host="0.0.0.0",
    port=8000,
    model_name="gpt2",
)
```

### Make requests

```python
import requests

response = requests.post(
    "http://localhost:8000/generate",
    json={
        "prompt": "Hello, world!",
        "max_tokens": 50,
    }
)

print(response.json())
```

## Examples

See `examples/` directory for more examples:

- `basic_usage.py` - Simple example
- `advanced_usage.py` - Custom kernels, benchmarking
- `api_server.py` - API server example

## Configuration Options

### Engine Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | "gpt2" | HuggingFace model name |
| `use_triton` | True | Enable custom Triton kernels |
| `use_fp8` | False | Enable FP8 quantization |
| `enable_paged_attention` | True | Enable paged KV-cache |
| `max_batch_size` | 32 | Maximum batch size |
| `max_num_blocks` | 1024 | Maximum KV-cache blocks |
| `block_size` | 16 | KV-cache block size |

### Generation Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_tokens` | 512 | Maximum tokens to generate |
| `temperature` | 1.0 | Sampling temperature |
| `top_p` | 1.0 | Nucleus sampling |
| `top_k` | 50 | Top-k sampling |

## Hardware Compatibility

| GPU Architecture | SM | Support Level |
|-----------------|-----|---------------|
| Blackwell | 120 | ✅ Full (all optimizations) |
| Hopper | 90 | ✅ Most optimizations |
| Ada Lovelace | 89 | ⚠️ Limited FP8 support |
| Ampere | 80 | ⚠️ No FP8, Triton works |
| Older | <80 | ❌ Not recommended |

## Troubleshooting

### Out of Memory

Reduce:
- `max_batch_size`
- `max_num_blocks`
- Model size

### Slow Performance

Ensure:
- CUDA 12.4+ installed
- Latest GPU drivers
- `use_triton=True`
- `use_fp8=True` (on SM120)

### Import Errors

```bash
pip install --upgrade torch triton transformers
```

## Next Steps

1. Read [Architecture](ARCHITECTURE.md) for design details
2. Check [Examples](../examples/README.md) for more use cases
3. See [Contributing](../CONTRIBUTING.md) to contribute
4. Run benchmarks to verify performance

## Support

- GitHub Issues: Report bugs
- GitHub Discussions: Ask questions
- Documentation: Full API reference

---

**Ready to build something incredible with SM120!** 🚀
