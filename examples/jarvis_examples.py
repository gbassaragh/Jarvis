"""
JARVIS - Complete Usage Examples

This file demonstrates all JARVIS features:
- Basic chat
- Voice interaction
- Conversational memory
- Tool use
- RAG knowledge base
- Web interface
- Multi-user support
"""

import time
from ai_assistant_pro.jarvis import (
    JARVIS,
    ConversationalMemory,
    RAGSystem,
    ToolRegistry,
    create_default_tools,
    VoiceInterface,
    MultiUserMemory,
    Document,
)


def example_basic_chat():
    """Example 1: Basic chat with JARVIS"""
    print("\n" + "=" * 60)
    print("Example 1: Basic Chat")
    print("=" * 60)

    # Create JARVIS (simple configuration)
    jarvis = JARVIS(
        model_name="gpt2",
        enable_voice=False,
        enable_tools=False,
        enable_memory=False,
    )

    # Chat
    result = jarvis.chat("Hello! What can you do?")
    print(f"\nUser: Hello! What can you do?")
    print(f"JARVIS: {result['response']}")
    print(f"Time: {result['elapsed_time']:.2f}s")


def example_with_memory():
    """Example 2: Chat with conversational memory"""
    print("\n" + "=" * 60)
    print("Example 2: Conversational Memory")
    print("=" * 60)

    jarvis = JARVIS(
        model_name="gpt2",
        user_id="alice",
        enable_memory=True,
        enable_tools=False,
    )

    # First conversation
    print("\n--- First conversation ---")
    result = jarvis.chat("My name is Alice and I love Italian food")
    print(f"User: My name is Alice and I love Italian food")
    print(f"JARVIS: {result['response']}")

    # Second conversation - JARVIS should remember
    print("\n--- Later conversation ---")
    result = jarvis.chat("What should I eat for dinner?")
    print(f"User: What should I eat for dinner?")
    print(f"JARVIS: {result['response']}")

    # Save memory for next session
    jarvis.save_memory("alice_memory.json")
    print("\n✓ Memory saved to alice_memory.json")


def example_with_tools():
    """Example 3: Using tools (calculator, web search)"""
    print("\n" + "=" * 60)
    print("Example 3: Tool Use")
    print("=" * 60)

    jarvis = JARVIS(
        model_name="gpt2",
        enable_tools=True,
        enable_memory=False,
    )

    # Calculator
    print("\n--- Using calculator ---")
    result = jarvis.chat("Calculate 15% tip on $85.50")
    print(f"User: Calculate 15% tip on $85.50")
    print(f"JARVIS: {result['response']}")
    if result["tool_results"]:
        print(f"Tool used: {result['tool_results'][0]['tool']}")
        print(f"Result: {result['tool_results'][0]['result']}")

    # Web search
    print("\n--- Using web search ---")
    result = jarvis.chat("Search for Python tutorials")
    print(f"User: Search for Python tutorials")
    print(f"JARVIS: {result['response']}")
    if result["tool_results"]:
        print(f"Tool used: {result['tool_results'][0]['tool']}")
        print(f"Found {len(result['tool_results'][0]['result'])} results")


def example_with_rag():
    """Example 4: RAG knowledge base"""
    print("\n" + "=" * 60)
    print("Example 4: RAG Knowledge Base")
    print("=" * 60)

    jarvis = JARVIS(
        model_name="gpt2",
        enable_rag=True,
        enable_memory=False,
    )

    # Add knowledge
    print("\n--- Adding knowledge ---")
    jarvis.add_knowledge(
        "Python is a high-level programming language created by Guido van Rossum in 1991.",
        metadata={"topic": "programming", "language": "python"},
        importance=0.9,
    )

    jarvis.add_knowledge(
        "Python is known for its simple, readable syntax and extensive standard library.",
        metadata={"topic": "programming", "language": "python"},
        importance=0.8,
    )

    jarvis.add_knowledge(
        "Machine learning in Python is commonly done with libraries like TensorFlow, PyTorch, and scikit-learn.",
        metadata={"topic": "ml", "language": "python"},
        importance=0.9,
    )

    print("✓ Added 3 knowledge items")

    # Query with RAG
    print("\n--- Querying knowledge base ---")
    result = jarvis.chat("What is Python?", use_rag=True)
    print(f"User: What is Python?")
    print(f"JARVIS: {result['response']}")
    print(f"\nSources used: {len(result['rag_sources'])}")
    for i, source in enumerate(result["rag_sources"], 1):
        print(f"  [{i}] {source}")


def example_load_knowledge_directory():
    """Example 5: Load knowledge from directory"""
    print("\n" + "=" * 60)
    print("Example 5: Load Knowledge from Directory")
    print("=" * 60)

    jarvis = JARVIS(enable_rag=True)

    # Load all markdown files from docs directory
    print("\n--- Loading documentation ---")
    jarvis.load_knowledge_base("./docs", pattern="*.md")

    # Query
    result = jarvis.chat("What is the Stone Retrieval Function?", use_rag=True)
    print(f"\nUser: What is the Stone Retrieval Function?")
    print(f"JARVIS: {result['response']}")


def example_voice_assistant():
    """Example 6: Voice-based assistant"""
    print("\n" + "=" * 60)
    print("Example 6: Voice Assistant")
    print("=" * 60)

    # Note: This requires microphone and speakers
    jarvis = JARVIS(
        model_name="gpt2",
        enable_voice=True,
        enable_memory=True,
    )

    print("\n🎤 Starting voice assistant...")
    print("Say 'Jarvis' followed by your command")
    print("(Press Ctrl+C to stop)")

    try:
        # Start continuous listening
        # This will run until interrupted
        jarvis.start_voice_assistant()
    except KeyboardInterrupt:
        print("\n✓ Voice assistant stopped")


def example_web_interface():
    """Example 7: Web interface"""
    print("\n" + "=" * 60)
    print("Example 7: Web Interface")
    print("=" * 60)

    jarvis = JARVIS(
        model_name="gpt2",
        enable_memory=True,
        enable_tools=True,
        enable_rag=True,
    )

    print("\n🌐 Starting web interface...")
    print("Visit http://localhost:8080 in your browser")
    print("(Press Ctrl+C to stop)")

    try:
        # Start web server
        jarvis.start_web_interface(host="0.0.0.0", port=8080)
    except KeyboardInterrupt:
        print("\n✓ Web server stopped")


def example_multi_user():
    """Example 8: Multi-user memory management"""
    print("\n" + "=" * 60)
    print("Example 8: Multi-User System")
    print("=" * 60)

    # Create multi-user memory manager
    memory_manager = MultiUserMemory()

    # Different users
    print("\n--- User: Alice ---")
    alice_mem = memory_manager.get_memory("alice")
    alice_mem.add_turn(
        user_message="I love Italian food",
        assistant_response="Great! I'll remember that.",
    )
    alice_mem.update_user_preference("favorite_cuisine", "Italian")

    print("\n--- User: Bob ---")
    bob_mem = memory_manager.get_memory("bob")
    bob_mem.add_turn(
        user_message="I'm allergic to peanuts",
        assistant_response="I'll keep that in mind for safety.",
    )
    bob_mem.add_user_fact("allergy", "peanuts")

    # Each user has separate memory
    print("\n--- Checking memories ---")
    print(f"Alice preferences: {alice_mem.profile.preferences}")
    print(f"Bob facts: {bob_mem.profile.facts}")

    # Export all memories
    memory_manager.export_all("./user_memories")
    print("\n✓ Exported all user memories to ./user_memories/")


def example_custom_tools():
    """Example 9: Creating custom tools"""
    print("\n" + "=" * 60)
    print("Example 9: Custom Tools")
    print("=" * 60)

    from ai_assistant_pro.jarvis.tools import Tool, ToolResult

    # Define custom tool
    class WeatherTool(Tool):
        """Custom weather lookup tool"""

        def __init__(self):
            super().__init__(
                name="weather",
                description="Get weather information for a location",
            )

        def execute(self, location: str) -> ToolResult:
            # In real implementation, call weather API
            # For demo, return mock data
            weather_data = {
                "temperature": "72°F",
                "conditions": "Sunny",
                "humidity": "45%",
            }

            return ToolResult(
                success=True,
                result=f"Weather in {location}: {weather_data['temperature']}, {weather_data['conditions']}",
            )

    # Create tool registry and register custom tool
    tools = create_default_tools()
    tools.register(WeatherTool())

    # Create JARVIS with custom tools
    jarvis = JARVIS(enable_tools=True)
    jarvis.tools = tools

    print("\n--- Available tools ---")
    for tool in tools.list_tools():
        print(f"  - {tool['name']}: {tool['description']}")


def example_statistics():
    """Example 10: Getting JARVIS statistics"""
    print("\n" + "=" * 60)
    print("Example 10: Statistics and Monitoring")
    print("=" * 60)

    jarvis = JARVIS(
        enable_memory=True,
        enable_tools=True,
        enable_rag=True,
    )

    # Have some conversations
    jarvis.chat("Hello!")
    jarvis.chat("What's 2 + 2?", use_tools=True)

    # Add some knowledge
    jarvis.add_knowledge("Test knowledge item 1")
    jarvis.add_knowledge("Test knowledge item 2")

    # Get statistics
    print("\n--- JARVIS Statistics ---")
    stats = jarvis.get_stats()

    print(f"\nUser: {stats['user_id']}")
    print(f"Model: {stats['model']}")

    print("\nComponents:")
    for component, enabled in stats["components"].items():
        print(f"  - {component}: {'✓' if enabled else '✗'}")

    if "memory" in stats:
        print(f"\nMemory:")
        print(f"  - Total conversations: {stats['memory']['total_turns']}")
        print(f"  - Session conversations: {stats['memory']['session_turns']}")
        print(f"  - User preferences: {stats['memory']['user_preferences']}")

    if "tools" in stats:
        print(f"\nTools:")
        print(f"  - Available: {stats['tools']['count']}")
        print(f"  - List: {', '.join(stats['tools']['available'])}")

    if "rag" in stats:
        print(f"\nRAG:")
        print(f"  - Documents: {stats['rag']['num_documents']}")
        print(f"  - Chunks: {stats['rag']['num_chunks']}")


def example_full_featured():
    """Example 11: Full-featured JARVIS (all components enabled)"""
    print("\n" + "=" * 60)
    print("Example 11: Full-Featured JARVIS")
    print("=" * 60)

    # Create JARVIS with all features
    jarvis = JARVIS(
        model_name="gpt2",
        user_id="demo_user",
        enable_voice=False,  # Set to True if you have microphone
        enable_tools=True,
        enable_rag=True,
        enable_memory=True,
        use_triton=True,
        use_fp8=False,
    )

    # Add knowledge
    print("\n--- Setting up knowledge base ---")
    jarvis.add_knowledge(
        "JARVIS is an AI assistant with voice, memory, tools, and RAG capabilities.",
        importance=1.0,
    )

    # Interactive chat
    print("\n--- Interactive Chat Demo ---")
    conversations = [
        "Hello, I'm interested in learning Python",
        "What is JARVIS?",
        "Calculate the square root of 144",
        "What was I interested in learning?",  # Tests memory
    ]

    for user_msg in conversations:
        print(f"\nUser: {user_msg}")

        result = jarvis.chat(
            user_msg,
            use_memory=True,
            use_tools=True,
            use_rag=True,
        )

        print(f"JARVIS: {result['response']}")

        if result["tool_results"]:
            print(f"  [Used tool: {result['tool_results'][0]['tool']}]")

        if result["rag_sources"]:
            print(f"  [Retrieved {len(result['rag_sources'])} knowledge sources]")

    # Show final stats
    print("\n--- Final Statistics ---")
    stats = jarvis.get_stats()
    print(f"Total conversations: {stats['memory']['total_turns']}")
    print(f"Knowledge items: {stats['rag']['num_documents']}")


def run_all_examples():
    """Run all examples"""
    print("\n" + "=" * 80)
    print(" " * 20 + "JARVIS - Complete Examples")
    print("=" * 80)

    # Basic examples
    example_basic_chat()
    time.sleep(1)

    example_with_memory()
    time.sleep(1)

    example_with_tools()
    time.sleep(1)

    example_with_rag()
    time.sleep(1)

    example_multi_user()
    time.sleep(1)

    example_custom_tools()
    time.sleep(1)

    example_statistics()
    time.sleep(1)

    example_full_featured()

    print("\n" + "=" * 80)
    print("✓ All examples completed!")
    print("=" * 80)

    print("\nTo run individual examples:")
    print("  - example_basic_chat()")
    print("  - example_with_memory()")
    print("  - example_with_tools()")
    print("  - example_with_rag()")
    print("  - example_load_knowledge_directory()")
    print("  - example_voice_assistant()  # Requires microphone")
    print("  - example_web_interface()  # Starts web server")
    print("  - example_multi_user()")
    print("  - example_custom_tools()")
    print("  - example_statistics()")
    print("  - example_full_featured()")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Run specific example
        example_name = sys.argv[1]
        if example_name in globals():
            globals()[example_name]()
        else:
            print(f"Unknown example: {example_name}")
            print("Available examples:")
            for name in dir():
                if name.startswith("example_"):
                    print(f"  - {name}")
    else:
        # Run all examples
        run_all_examples()
