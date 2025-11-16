"""Optimized inference engine for AI Assistant Pro.

Heavy modules are imported lazily to avoid pulling GPU deps on lightweight imports.
"""

from typing import Any

__all__ = ["AssistantEngine", "PagedKVCache", "ContinuousBatchScheduler"]


def __getattr__(name: str) -> Any:
    if name == "AssistantEngine":
        from ai_assistant_pro.engine.model import AssistantEngine as _AssistantEngine

        return _AssistantEngine
    if name == "PagedKVCache":
        from ai_assistant_pro.engine.cache import PagedKVCache as _PagedKVCache

        return _PagedKVCache
    if name == "ContinuousBatchScheduler":
        from ai_assistant_pro.engine.scheduler import (
            ContinuousBatchScheduler as _ContinuousBatchScheduler,
        )

        return _ContinuousBatchScheduler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
