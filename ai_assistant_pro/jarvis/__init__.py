"""
JARVIS - Just A Rather Very Intelligent System

Complete AI assistant with:
- Voice interface
- Conversational memory (SRF-powered)
- Tool/plugin system
- Web interface
- Multi-modal support

Imports are lazy to keep lightweight tooling from pulling heavy deps.
"""

__all__ = [
    "JARVIS",
    "VoiceInterface",
    "VoiceAssistant",
    "ConversationalMemory",
    "MultiUserMemory",
    "UserProfile",
    "ConversationTurn",
    "Tool",
    "ToolResult",
    "ToolRegistry",
    "Calculator",
    "WebSearch",
    "ShellCommand",
    "PythonREPL",
    "FileOperations",
    "ToolUseParser",
    "create_default_tools",
    "RAGSystem",
    "Document",
    "DocumentChunker",
    "JARVISWebInterface",
]


def __getattr__(name):
    if name == "JARVIS":
        from ai_assistant_pro.jarvis.core import JARVIS as _JARVIS

        return _JARVIS
    if name in {"VoiceInterface", "VoiceAssistant"}:
        from ai_assistant_pro.jarvis.voice import VoiceInterface, VoiceAssistant

        return {"VoiceInterface": VoiceInterface, "VoiceAssistant": VoiceAssistant}[name]
    if name in {"ConversationalMemory", "MultiUserMemory", "UserProfile", "ConversationTurn"}:
        from ai_assistant_pro.jarvis.memory import (
            ConversationalMemory,
            MultiUserMemory,
            UserProfile,
            ConversationTurn,
        )

        return {
            "ConversationalMemory": ConversationalMemory,
            "MultiUserMemory": MultiUserMemory,
            "UserProfile": UserProfile,
            "ConversationTurn": ConversationTurn,
        }[name]
    if name in {
        "Tool",
        "ToolResult",
        "ToolRegistry",
        "Calculator",
        "WebSearch",
        "ShellCommand",
        "PythonREPL",
        "FileOperations",
        "ToolUseParser",
        "create_default_tools",
    }:
        from ai_assistant_pro.jarvis.tools import (
            Tool,
            ToolResult,
            ToolRegistry,
            Calculator,
            WebSearch,
            ShellCommand,
            PythonREPL,
            FileOperations,
            ToolUseParser,
            create_default_tools,
        )

        return {
            "Tool": Tool,
            "ToolResult": ToolResult,
            "ToolRegistry": ToolRegistry,
            "Calculator": Calculator,
            "WebSearch": WebSearch,
            "ShellCommand": ShellCommand,
            "PythonREPL": PythonREPL,
            "FileOperations": FileOperations,
            "ToolUseParser": ToolUseParser,
            "create_default_tools": create_default_tools,
        }[name]
    if name in {"RAGSystem", "Document", "DocumentChunker"}:
        from ai_assistant_pro.jarvis.rag import RAGSystem, Document, DocumentChunker

        return {
            "RAGSystem": RAGSystem,
            "Document": Document,
            "DocumentChunker": DocumentChunker,
        }[name]
    if name == "JARVISWebInterface":
        from ai_assistant_pro.jarvis.web_ui import JARVISWebInterface as _JARVISWebInterface

        return _JARVISWebInterface
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
