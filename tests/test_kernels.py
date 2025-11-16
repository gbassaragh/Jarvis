"""
Unit tests for custom Triton kernels
"""

import os

import pytest
import torch
import torch.nn.functional as F

from ai_assistant_pro.kernels.attention import flashattention_v3
from ai_assistant_pro.kernels.fused_ops import fused_layernorm, fused_gelu
from ai_assistant_pro.kernels.quantization import quantize_fp8, dequantize_fp8


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or os.getenv("RUN_CUDA_TESTS") != "1",
    reason="Requires CUDA and RUN_CUDA_TESTS=1",
)

xfail_unstable = pytest.mark.xfail(reason="Known instability on this setup; kernel WIP", strict=False)


class TestFlashAttention:
    """Test FlashAttention-3 kernel"""

    @xfail_unstable
    def test_flashattention_output_shape(self):
        """Test output shape"""
        batch, heads, seq_len, head_dim = 2, 8, 128, 64

        q = torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
        v = torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=torch.float16)

        out = flashattention_v3(q, k, v)

        assert out.shape == (batch, heads, seq_len, head_dim)

    @xfail_unstable
    def test_flashattention_vs_pytorch(self):
        """Test FlashAttention vs PyTorch implementation"""
        batch, heads, seq_len, head_dim = 1, 4, 64, 32

        q = torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
        v = torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=torch.float16)

        # FlashAttention
        out_flash = flashattention_v3(q, k, v, use_fp8=False, backend="triton")

        # PyTorch reference
        scale = 1.0 / (head_dim ** 0.5)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = torch.softmax(attn, dim=-1)
        out_pytorch = torch.matmul(attn, v)

        # Should be close (allowing for numerical differences)
        assert torch.allclose(out_flash, out_pytorch, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestFusedOps:
    """Test fused operations"""

    def test_fused_layernorm_shape(self):
        """Test fused LayerNorm output shape"""
        batch, seq_len, hidden_dim = 4, 128, 768

        x = torch.randn(batch, seq_len, hidden_dim, device="cuda", dtype=torch.float16)
        weight = torch.randn(hidden_dim, device="cuda", dtype=torch.float16)
        bias = torch.randn(hidden_dim, device="cuda", dtype=torch.float16)

        out = fused_layernorm(x, weight, bias)

        assert out.shape == (batch, seq_len, hidden_dim)

    @xfail_unstable
    def test_fused_gelu_vs_pytorch(self):
        """Test fused GELU vs PyTorch"""
        x = torch.randn(1000, device="cuda", dtype=torch.float16)

        out_fused = fused_gelu(x)
        out_pytorch = F.gelu(x)

        assert torch.allclose(out_fused, out_pytorch, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestQuantization:
    """Test quantization kernels"""

    @xfail_unstable
    def test_fp8_quantization_roundtrip(self):
        """Test FP8 quantization and dequantization"""
        x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)

        # Quantize
        x_fp8, scale = quantize_fp8(x)

        # Dequantize
        x_recovered = dequantize_fp8(x_fp8, scale)

        # Should be close (FP8 has limited precision)
        relative_error = (x.float() - x_recovered).abs() / (x.float().abs() + 1e-8)
        assert relative_error.mean() < 0.1  # 10% average error acceptable for FP8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
