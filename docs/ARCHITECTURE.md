# Architecture

AI Assistant Pro is designed for maximum performance on NVIDIA Blackwell (SM120) GPUs.

## Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     User Application                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    AssistantEngine                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Scheduler   │  │ Cache Manager│  │ Model Wrapper│      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Custom Triton Kernels                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │FlashAttn-3   │  │ Fused Ops    │  │ Quantization │      │
│  │  - FP8 mode  │  │ - LayerNorm  │  │  - FP8       │      │
│  │  - Paged     │  │ - GELU       │  │  - INT8      │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   NVIDIA Blackwell GPU                       │
│                      (SM120 / Hopper+)                       │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. AssistantEngine

Main inference engine that orchestrates all components.

**Responsibilities:**
- Model loading and management
- Request handling
- Performance monitoring

**Key Features:**
- Automatic optimization selection
- Memory management
- Statistics tracking

### 2. Custom Triton Kernels

Hand-optimized kernels for SM120.

#### FlashAttention-3
- Warp-specialized design
- FP8 tensor core support
- Online softmax for memory efficiency
- Optimized for Blackwell's cache hierarchy

**Performance:** Up to 10x faster than PyTorch attention

#### Fused Operations
- Fused LayerNorm + Attention
- Fused GELU activation
- Fused Add + LayerNorm
- Rotary embeddings (RoPE)

**Benefits:** Reduced memory bandwidth, faster execution

#### Quantization
- FP8 E4M3 format (native SM120 support)
- INT8 quantization
- Fused quantized matmul

**Performance:** 2x speedup with FP8

### 3. Paged KV-Cache

Efficient memory management for key-value cache.

**Design:**
- Fixed-size blocks (default: 16 tokens)
- Dynamic allocation
- Block table mapping

**Benefits:**
- 50% memory reduction
- Support for variable-length sequences
- Efficient memory utilization

### 4. Continuous Batching Scheduler

Dynamic request scheduling for maximum throughput.

**Features:**
- Dynamic batch composition
- Priority-based scheduling
- Preemption support
- Memory-aware scheduling

**Performance:** Maximizes GPU utilization

## Memory Layout

### KV-Cache Organization

```
Physical Memory:
┌────────────────────────────────────────┐
│ Block 0 │ Block 1 │ ... │ Block N      │
└────────────────────────────────────────┘

Block Table (Logical → Physical):
Sequence 0: [0, 3, 7, ...]
Sequence 1: [1, 2, 5, ...]
```

### Tensor Layout

Tensors use contiguous memory layout optimized for:
- Warp-level access patterns
- Coalesced memory accesses
- Cache line alignment

## Optimization Strategies

### 1. SM120-Specific Optimizations

- **Warp Specialization:** Different warps handle different tasks
- **Async Copy:** Overlapping compute and memory
- **FP8 Tensor Cores:** 2x compute throughput
- **Enhanced L2 Cache:** Better data reuse

### 2. Memory Optimizations

- **Paged Attention:** Dynamic memory allocation
- **Kernel Fusion:** Reduce memory traffic
- **Online Algorithms:** Streaming computation (softmax)

### 3. Compute Optimizations

- **FP8 Quantization:** 2x faster matmul
- **Fused Operations:** Eliminate intermediate results
- **Optimal Tiling:** Maximize cache utilization

## Data Flow

### Inference Pipeline

```
Input Tokens
     │
     ▼
Embedding
     │
     ▼
┌──────────────────┐
│ Transformer Layer│ (x N layers)
│  ├─ Attention    │ ← FlashAttention-3
│  ├─ LayerNorm    │ ← Fused LayerNorm
│  ├─ MLP          │ ← Fused GELU
│  └─ Residual     │ ← Fused Add+LN
└──────────────────┘
     │
     ▼
Output Projection
     │
     ▼
Sampling
     │
     ▼
Generated Tokens
```

### Prefill vs Decode

**Prefill (Processing prompt):**
- Uses FlashAttention-3
- Large batch sizes
- Compute-bound

**Decode (Generating tokens):**
- Uses Paged Attention
- One token at a time
- Memory-bound

## Scalability

### Multi-GPU (Planned)

- **Tensor Parallelism:** Shard model across GPUs
- **Pipeline Parallelism:** Split layers across GPUs
- **Data Parallelism:** Multiple requests in parallel

### Batching

- **Continuous Batching:** Dynamic request composition
- **Iteration-level Batching:** Add/remove requests per iteration
- **Memory-aware:** Respect GPU memory limits

## Performance Characteristics

### Throughput

- **Single request:** Low latency, sub-ms per token
- **Batched requests:** High throughput, hundreds of tokens/sec
- **Continuous batching:** Maximum GPU utilization

### Memory Usage

- **Base model:** According to model size
- **KV-cache:** O(num_sequences * seq_len * hidden_dim)
- **Paged cache:** ~50% reduction vs contiguous

### Latency

- **First token (prefill):** Depends on prompt length
- **Subsequent tokens (decode):** <1ms on SM120
- **Total latency:** Dominated by number of tokens

## Future Directions

1. **Speculative Decoding:** 2-3x speedup
2. **Multi-Query/Grouped-Query Attention:** Better cache efficiency
3. **Sparse Attention:** O(n√n) complexity
4. **INT4 Quantization:** 4x memory reduction
5. **Custom Sampling Kernels:** Faster token selection
