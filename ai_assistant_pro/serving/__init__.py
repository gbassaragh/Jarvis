"""Serving layer for AI Assistant Pro.

Imported lazily so tools/tests without FastAPI/UVicorn available still import safely.
"""

__all__ = ["create_app", "serve"]


def __getattr__(name):
    if name in {"create_app", "serve"}:
        from ai_assistant_pro.serving.server import create_app, serve

        return {"create_app": create_app, "serve": serve}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
