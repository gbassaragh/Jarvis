"""
AI Assistant Pro - High-performance AI assistant framework for NVIDIA Blackwell (SM120)

Optimized with custom Triton kernels for maximum inference performance.
"""

from ai_assistant_pro.engine.model import AssistantEngine
from ai_assistant_pro.kernels.attention import flashattention_v3
from ai_assistant_pro.engine.cache import PagedKVCache

__version__ = "0.1.0"
__all__ = ["AssistantEngine", "flashattention_v3", "PagedKVCache"]
