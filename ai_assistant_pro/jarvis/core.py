"""
JARVIS Core - Main orchestrator

Ties together all JARVIS components:
- Voice interface
- Conversational memory
- Tool system
- RAG for knowledge
- Web interface

This is the "brain" of JARVIS.
"""

from typing import Optional, Dict, Any
import time

from ai_assistant_pro import AssistantEngine
from ai_assistant_pro.jarvis.voice import VoiceInterface
from ai_assistant_pro.jarvis.memory import ConversationalMemory, MultiUserMemory
from ai_assistant_pro.jarvis.tools import ToolRegistry, create_default_tools, ToolUseParser
from ai_assistant_pro.jarvis.rag import RAGSystem
from ai_assistant_pro.utils.logging import get_logger, setup_logging

logger = get_logger("jarvis.core")


class JARVIS:
    """
    JARVIS - Just A Rather Very Intelligent System

    Complete AI assistant with:
    - Natural conversation with long-term memory (SRF-powered)
    - Voice interaction (speech-to-text, text-to-speech)
    - Tool use (web search, calculator, code execution, etc.)
    - Knowledge retrieval (RAG with SRF)
    - Beautiful web interface
    - Multi-user support
    - Personalization and learning

    Example:
        >>> jarvis = JARVIS()
        >>> response = jarvis.chat("Hello, who are you?")
        >>> print(response)

        >>> # With voice
        >>> jarvis = JARVIS(enable_voice=True)
        >>> jarvis.start_voice_assistant()
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        user_id: str = "default",
        enable_voice: bool = False,
        enable_tools: bool = True,
        enable_rag: bool = True,
        enable_memory: bool = True,
        use_triton: bool = True,
        use_fp8: bool = False,
        log_level: str = "INFO",
    ):
        """
        Initialize JARVIS

        Args:
            model_name: LLM model name
            user_id: User identifier
            enable_voice: Enable voice interface
            enable_tools: Enable tool system
            enable_rag: Enable RAG knowledge base
            enable_memory: Enable conversational memory
            use_triton: Use Triton kernels
            use_fp8: Use FP8 quantization
            log_level: Logging level
        """
        # Setup logging
        setup_logging(level=log_level)

        logger.info("🚀 Initializing JARVIS...")

        self.user_id = user_id

        # Initialize core engine
        logger.info("Loading language model...")
        self.engine = AssistantEngine(
            model_name=model_name,
            use_triton=use_triton,
            use_fp8=use_fp8,
            enable_paged_attention=True,
        )

        # Initialize memory
        if enable_memory:
            logger.info("Initializing conversational memory (SRF-powered)...")
            self.memory = ConversationalMemory(user_id=user_id)
        else:
            self.memory = None

        # Initialize tools
        if enable_tools:
            logger.info("Loading tools...")
            self.tools = create_default_tools()
            self.tool_parser = ToolUseParser()
        else:
            self.tools = None
            self.tool_parser = None

        # Initialize RAG
        if enable_rag:
            logger.info("Initializing RAG system...")
            self.rag = RAGSystem()
        else:
            self.rag = None

        # Initialize voice
        if enable_voice:
            logger.info("Initializing voice interface...")
            self.voice = VoiceInterface(wake_word="jarvis")
        else:
            self.voice = None

        # System prompt
        self.system_prompt = self._build_system_prompt()

        logger.info("✅ JARVIS ready!")

    def _build_system_prompt(self) -> str:
        """Build system prompt"""
        parts = [
            "You are JARVIS (Just A Rather Very Intelligent System), "
            "an advanced AI assistant created to help with any task."
        ]

        if self.tools:
            parts.append(
                "\n\nYou have access to tools. To use a tool, write: "
                "[TOOL: tool_name] arguments [/TOOL]"
                "\n\nAvailable tools:"
            )

            for tool in self.tools.list_tools():
                parts.append(f"- {tool['name']}: {tool['description']}")

        parts.append(
            "\n\nYou are helpful, concise, and friendly. "
            "You remember past conversations and learn user preferences."
        )

        return "".join(parts)

    def chat(
        self,
        message: str,
        use_memory: bool = True,
        use_tools: bool = True,
        use_rag: bool = False,
        rag_query: Optional[str] = None,
        max_tokens: int = 300,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Chat with JARVIS

        Args:
            message: User message
            use_memory: Use conversational memory
            use_tools: Enable tool use
            use_rag: Use RAG for knowledge retrieval
            rag_query: Custom RAG query (uses message if None)
            max_tokens: Maximum response tokens
            temperature: Sampling temperature

        Returns:
            Dictionary with response and metadata
        """
        start_time = time.time()

        # Build context
        context_parts = [self.system_prompt]

        # Add RAG context if requested
        rag_sources = []
        if use_rag and self.rag:
            rag_query = rag_query or message
            rag_results = self.rag.retrieve(rag_query, top_k=3)

            if rag_results:
                context_parts.append("\n\nRelevant knowledge:")
                for i, result in enumerate(rag_results, 1):
                    context_parts.append(f"\n[{i}] {result['text']}")
                    rag_sources.append(result['metadata'])

        # Add memory context
        if use_memory and self.memory:
            memory_context = self.memory.get_context_summary(query=message)
            if memory_context:
                context_parts.append(f"\n\n{memory_context}")

        # Add current message
        context_parts.append(f"\n\nUser: {message}\nJARVIS:")

        prompt = "".join(context_parts)

        # Generate response
        response = self.engine.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Extract just the assistant's response
        # (remove system prompt and context)
        if "JARVIS:" in response:
            response = response.split("JARVIS:")[-1].strip()

        # Process tool use
        tool_results = []
        if use_tools and self.tools and self.tool_parser:
            tool_uses = self.tool_parser.parse(response)

            for tool_use in tool_uses:
                result = self.tools.execute(
                    tool_use["tool"],
                    **{"expression" if tool_use["tool"] == "calculator" else "query": tool_use["args"]}
                )
                tool_results.append({
                    "tool": tool_use["tool"],
                    "args": tool_use["args"],
                    "result": result.result if result.success else None,
                    "error": result.error,
                })

            # Remove tool markers from response
            response = self.tool_parser.remove_tool_markers(response)

        # Store in memory
        if use_memory and self.memory:
            self.memory.add_turn(
                user_message=message,
                assistant_response=response,
            )

        elapsed = time.time() - start_time

        return {
            "response": response,
            "tool_results": tool_results,
            "rag_sources": rag_sources,
            "elapsed_time": elapsed,
            "metadata": {
                "model": self.engine.model_name,
                "tokens": max_tokens,
                "memory_used": use_memory,
                "tools_used": len(tool_results) > 0,
                "rag_used": len(rag_sources) > 0,
            },
        }

    def voice_chat(self, audio_path: str) -> str:
        """
        Voice-based chat

        Args:
            audio_path: Path to audio file

        Returns:
            Response text (also speaks it)
        """
        if not self.voice:
            return "Voice interface not enabled"

        # Transcribe
        message = self.voice.transcribe(audio_path=audio_path)

        # Chat
        result = self.chat(message)
        response = result["response"]

        # Speak
        self.voice.speak(response)

        return response

    def start_voice_assistant(self):
        """Start continuous voice listening"""
        if not self.voice:
            logger.error("Voice interface not enabled")
            return

        def callback(command: str) -> str:
            result = self.chat(command)
            return result["response"]

        logger.info("🎤 Starting voice assistant...")
        logger.info("Say 'Jarvis' followed by your command")

        self.voice.listen_continuous(callback=callback)

    def add_knowledge(
        self,
        content: str,
        metadata: Optional[Dict] = None,
        importance: float = 0.7,
    ):
        """
        Add knowledge to RAG system

        Args:
            content: Knowledge content
            metadata: Optional metadata
            importance: Importance score
        """
        if not self.rag:
            logger.warning("RAG not enabled")
            return

        from ai_assistant_pro.jarvis.rag import Document

        doc = Document(content=content, metadata=metadata)
        self.rag.add_document(doc, importance=importance)

        logger.info(f"Added knowledge: {content[:50]}...")

    def load_knowledge_base(self, directory: str, pattern: str = "*.txt"):
        """
        Load knowledge base from directory

        Args:
            directory: Directory path
            pattern: File pattern
        """
        if not self.rag:
            logger.warning("RAG not enabled")
            return

        self.rag.add_documents_from_directory(directory, pattern=pattern)
        logger.info(f"✅ Loaded knowledge base from {directory}")

    def get_stats(self) -> Dict[str, Any]:
        """Get JARVIS statistics"""
        stats = {
            "user_id": self.user_id,
            "model": self.engine.model_name,
            "components": {
                "voice": self.voice is not None,
                "memory": self.memory is not None,
                "tools": self.tools is not None,
                "rag": self.rag is not None,
            },
        }

        if self.memory:
            stats["memory"] = self.memory.get_statistics()

        if self.rag:
            stats["rag"] = self.rag.get_statistics()

        if self.tools:
            stats["tools"] = {
                "count": len(self.tools.list_tools()),
                "available": [t["name"] for t in self.tools.list_tools()],
            }

        return stats

    def save_memory(self, filepath: str):
        """Save conversational memory to file"""
        if self.memory:
            self.memory.export_memory(filepath)
            logger.info(f"✅ Memory saved to {filepath}")

    def load_memory(self, filepath: str):
        """Load conversational memory from file"""
        if self.memory:
            self.memory.import_memory(filepath)
            logger.info(f"✅ Memory loaded from {filepath}")

    def start_web_interface(self, host: str = "0.0.0.0", port: int = 8080):
        """
        Start web interface

        Args:
            host: Host to bind to
            port: Port to bind to
        """
        from ai_assistant_pro.jarvis.web_ui import JARVISWebInterface

        logger.info("🌐 Starting web interface...")

        # Create multi-user memory manager
        memory_manager = MultiUserMemory()
        memory_manager.user_memories[self.user_id] = self.memory

        # Create web interface
        web_ui = JARVISWebInterface(
            engine=self.engine,
            memory_manager=memory_manager,
        )

        web_ui.run(host=host, port=port)

    def __repr__(self):
        return f"JARVIS(user={self.user_id}, model={self.engine.model_name})"


# CLI entry point
if __name__ == "__main__":
    import sys

    # Create JARVIS
    jarvis = JARVIS(
        enable_voice=False,
        enable_tools=True,
        enable_rag=True,
        enable_memory=True,
    )

    # Interactive mode
    print("=" * 60)
    print("JARVIS - Just A Rather Very Intelligent System")
    print("=" * 60)
    print("\nType 'quit' to exit, 'stats' for statistics\n")

    while True:
        try:
            message = input("You: ")

            if message.lower() == "quit":
                break

            if message.lower() == "stats":
                stats = jarvis.get_stats()
                print(f"\nStatistics:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                print()
                continue

            result = jarvis.chat(message)

            print(f"\nJARVIS: {result['response']}")

            if result["tool_results"]:
                print("\n[Tools used:]")
                for tr in result["tool_results"]:
                    print(f"  - {tr['tool']}: {tr['result']}")

            print()

        except KeyboardInterrupt:
            break

    print("\nGoodbye!")
