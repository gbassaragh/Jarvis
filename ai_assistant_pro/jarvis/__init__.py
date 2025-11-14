"""
JARVIS - Just A Rather Very Intelligent System

Complete AI assistant with:
- Voice interface
- Conversational memory (SRF-powered)
- Tool/plugin system
- Web interface
- Multi-modal support
"""

from ai_assistant_pro.jarvis.core import JARVIS
from ai_assistant_pro.jarvis.voice import VoiceInterface, VoiceAssistant
from ai_assistant_pro.jarvis.memory import ConversationalMemory, MultiUserMemory, UserProfile, ConversationTurn
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
from ai_assistant_pro.jarvis.rag import RAGSystem, Document, DocumentChunker
from ai_assistant_pro.jarvis.web_ui import JARVISWebInterface

__all__ = [
    # Main
    "JARVIS",
    # Voice
    "VoiceInterface",
    "VoiceAssistant",
    # Memory
    "ConversationalMemory",
    "MultiUserMemory",
    "UserProfile",
    "ConversationTurn",
    # Tools
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
    # RAG
    "RAGSystem",
    "Document",
    "DocumentChunker",
    # Web
    "JARVISWebInterface",
]
