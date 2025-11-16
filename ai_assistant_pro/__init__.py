"""
AI Assistant Pro - High-performance AI assistant framework for NVIDIA Blackwell (SM120)

Optimized with custom Triton kernels for maximum inference performance.

Note: heavy modules (kernels/engine) are imported lazily to keep `import ai_assistant_pro`
lightweight for tooling/tests that don't need GPU code.
"""

__version__ = "0.1.0"
__all__ = ["AssistantEngine", "flashattention_v3", "PagedKVCache"]


def __getattr__(name):
    if name == "AssistantEngine":
        from ai_assistant_pro.engine.model import AssistantEngine as _AssistantEngine

        return _AssistantEngine
    if name == "flashattention_v3":
        from ai_assistant_pro.kernels.attention import flashattention_v3 as _flashattention_v3

        return _flashattention_v3
    if name == "PagedKVCache":
        from ai_assistant_pro.engine.cache import PagedKVCache as _PagedKVCache

        return _PagedKVCache
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
