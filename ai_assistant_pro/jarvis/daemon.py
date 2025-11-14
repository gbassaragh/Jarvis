"""
JARVIS Daemon - Always-On Voice Assistant

Runs in the background, listening for wake words and engaging in natural conversation.
This creates the real-life JARVIS experience.
"""

import time
import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List
import threading
import queue

from ai_assistant_pro.jarvis.core import JARVIS
from ai_assistant_pro.utils.logging import get_logger, setup_logging
from rich.console import Console
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich import box

logger = get_logger("jarvis.daemon")
console = Console()


class JARVISDaemon:
    """
    Always-on JARVIS daemon

    Features:
    - Continuous wake word detection
    - Natural wake words (Hi, Hello, Jarvis, etc.)
    - Proactive greetings based on time of day
    - Remembers when you last talked
    - Background listening with minimal resource usage
    """

    def __init__(
        self,
        user_id: str = "default",
        model_name: str = "gpt2",
        wake_words: Optional[List[str]] = None,
        enable_proactive: bool = True,
        config_path: Optional[str] = None,
    ):
        """
        Initialize JARVIS daemon

        Args:
            user_id: User identifier
            model_name: AI model to use
            wake_words: List of wake words (default: ["hi", "hello", "jarvis"])
            enable_proactive: Enable proactive greetings
            config_path: Path to config file
        """
        self.user_id = user_id
        self.wake_words = wake_words or ["hi", "hello", "jarvis", "hey jarvis"]
        self.enable_proactive = enable_proactive

        # Load config if provided
        if config_path:
            self.config = self._load_config(config_path)
        else:
            self.config = None

        # State
        self.running = False
        self.last_interaction = None
        self.greeted_today = False
        self.conversation_active = False

        # Audio queue for background listening
        self.audio_queue = queue.Queue()

        # Initialize JARVIS
        console.print("\n[bold cyan]🚀 Initializing JARVIS Daemon...[/bold cyan]\n")

        self.jarvis = JARVIS(
            model_name=model_name,
            user_id=user_id,
            enable_voice=True,
            enable_memory=True,
            enable_tools=True,
            enable_rag=True,
            use_triton=True,
            use_fp8=False,
            log_level="INFO",
        )

        # Customize JARVIS personality for voice interaction
        self.jarvis.system_prompt = self._build_voice_personality()

        console.print("[green]✓ JARVIS initialized[/green]\n")

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from file"""
        import yaml

        with open(config_path) as f:
            return yaml.safe_load(f)

    def _build_voice_personality(self) -> str:
        """Build personality for natural voice interaction"""
        return f"""You are JARVIS, a sophisticated AI assistant speaking to {self.user_id}.

Personality traits:
- Warm, friendly, and conversational (like talking to a helpful friend)
- Concise but not robotic (2-3 sentences per response in conversation)
- Proactive and thoughtful (remember context and ask follow-up questions)
- Professional but personable (like Tony Stark's JARVIS - helpful and witty)
- Remember past conversations and preferences

Speaking style:
- Use natural language (contractions, casual phrasing)
- Address the user by name when appropriate
- Show genuine interest in their wellbeing
- Acknowledge time of day ("Good morning", "How was your day?")
- Make relevant connections to past conversations

You have access to tools for web search, calculations, and more.
You remember all past conversations with {self.user_id}.

When greeting:
- Morning (5am-12pm): "Good morning", ask about sleep or plans
- Afternoon (12pm-5pm): "Good afternoon", ask about day so far
- Evening (5pm-9pm): "Good evening", ask about day or dinner
- Night (9pm-5am): "Hello", keep it brief and calm

You are helpful, intelligent, and make the user feel like they have a real AI companion.
"""

    def _get_greeting(self) -> str:
        """Get time-appropriate greeting"""
        hour = datetime.now().hour

        if 5 <= hour < 12:
            greetings = [
                f"Good morning, {self.user_id}! How did you sleep?",
                f"Morning! Ready to tackle the day?",
                f"Good morning! What's on your agenda today?",
            ]
        elif 12 <= hour < 17:
            greetings = [
                f"Good afternoon, {self.user_id}! How's your day going?",
                f"Afternoon! Hope your day is going well.",
                f"Hey there! What have you been up to?",
            ]
        elif 17 <= hour < 21:
            greetings = [
                f"Good evening, {self.user_id}! How was your day?",
                f"Evening! Winding down from the day?",
                f"Hey! Tell me about your day.",
            ]
        else:
            greetings = [
                f"Hello, {self.user_id}. Working late?",
                f"Evening! Still up?",
                f"Hi there. What can I help with?",
            ]

        import random
        return random.choice(greetings)

    def _should_greet_proactively(self) -> bool:
        """Check if should greet proactively"""
        if not self.enable_proactive:
            return False

        # Don't greet if already greeted today
        if self.greeted_today:
            return False

        # Check if it's been a while since last interaction
        if self.last_interaction:
            hours_since = (datetime.now() - self.last_interaction).total_seconds() / 3600
            if hours_since < 4:  # Don't greet if talked recently
                return False

        # Greet during waking hours
        hour = datetime.now().hour
        if 7 <= hour <= 22:
            return True

        return False

    def _detect_wake_word(self, text: str) -> bool:
        """Detect if wake word is present in transcribed text"""
        text_lower = text.lower().strip()

        # Check exact matches
        if text_lower in self.wake_words:
            return True

        # Check if starts with wake word
        for wake_word in self.wake_words:
            if text_lower.startswith(wake_word):
                return True

        # Check if contains wake word
        words = text_lower.split()
        if any(wake_word in words for wake_word in self.wake_words):
            return True

        return False

    def _listen_for_wake_word(self) -> Optional[str]:
        """Listen for wake word (non-blocking)"""
        try:
            # Record short audio snippet
            audio = self.jarvis.voice.record_audio(duration=3, sample_rate=16000)

            if audio is None:
                return None

            # Quick transcription
            text = self.jarvis.voice.transcribe_audio(audio)

            if text and self._detect_wake_word(text):
                # Remove wake word from text
                text_lower = text.lower()
                for wake_word in self.wake_words:
                    text_lower = text_lower.replace(wake_word, "").strip()

                return text_lower if text_lower else None

            return None

        except Exception as e:
            logger.error(f"Wake word detection error: {e}")
            return None

    def _have_conversation(self, initial_message: Optional[str] = None):
        """Have a conversation with the user"""
        self.conversation_active = True
        self.last_interaction = datetime.now()

        try:
            # Initial greeting or response
            if initial_message:
                console.print(f"\n[bold cyan]You:[/bold cyan] {initial_message}\n")

                result = self.jarvis.chat(
                    initial_message,
                    use_memory=True,
                    use_tools=True,
                    use_rag=False,
                )

                response = result["response"]
            else:
                # Proactive greeting
                greeting = self._get_greeting()
                response = greeting
                self.greeted_today = True

            # Speak response
            console.print(f"[bold green]JARVIS:[/bold green] {response}\n")
            self.jarvis.voice.speak(response)

            # Continue conversation
            while True:
                console.print("[dim]Listening...[/dim]")

                # Listen for response (longer duration for full sentence)
                audio = self.jarvis.voice.record_audio(duration=5, sample_rate=16000)

                if audio is None:
                    console.print("[dim]No response detected.[/dim]\n")
                    break

                # Transcribe
                user_message = self.jarvis.voice.transcribe_audio(audio)

                if not user_message or len(user_message.strip()) < 2:
                    console.print("[dim]Nothing heard. Ending conversation.[/dim]\n")
                    break

                # Check for exit phrases
                exit_phrases = ["goodbye", "bye", "that's all", "thank you bye", "stop"]
                if any(phrase in user_message.lower() for phrase in exit_phrases):
                    console.print(f"\n[bold cyan]You:[/bold cyan] {user_message}\n")

                    farewell = f"Goodbye, {self.user_id}! Let me know if you need anything."
                    console.print(f"[bold green]JARVIS:[/bold green] {farewell}\n")
                    self.jarvis.voice.speak(farewell)
                    break

                # Process message
                console.print(f"\n[bold cyan]You:[/bold cyan] {user_message}\n")

                result = self.jarvis.chat(
                    user_message,
                    use_memory=True,
                    use_tools=True,
                    use_rag=False,
                )

                response = result["response"]

                # Show tool use
                if result["tool_results"]:
                    console.print(f"[dim]Using tools: {', '.join(t['tool'] for t in result['tool_results'])}[/dim]")

                # Respond
                console.print(f"[bold green]JARVIS:[/bold green] {response}\n")
                self.jarvis.voice.speak(response)

        except KeyboardInterrupt:
            console.print("\n[yellow]Conversation interrupted[/yellow]\n")

        except Exception as e:
            logger.error(f"Conversation error: {e}")
            console.print(f"\n[red]Error: {e}[/red]\n")

        finally:
            self.conversation_active = False

    def _display_status(self) -> Panel:
        """Create status display"""
        status_text = Text()

        status_text.append("🎤 ", style="bold cyan")
        status_text.append("Listening for: ", style="white")
        status_text.append(", ".join(self.wake_words), style="bold yellow")
        status_text.append("\n\n")

        status_text.append("👤 ", style="bold cyan")
        status_text.append("User: ", style="white")
        status_text.append(self.user_id, style="bold yellow")
        status_text.append("\n\n")

        if self.last_interaction:
            time_str = self.last_interaction.strftime("%I:%M %p")
            status_text.append("💬 ", style="bold cyan")
            status_text.append("Last interaction: ", style="white")
            status_text.append(time_str, style="bold yellow")
        else:
            status_text.append("💬 ", style="bold cyan")
            status_text.append("No interactions yet", style="dim")

        return Panel(
            status_text,
            title="[bold]JARVIS Status[/bold]",
            border_style="cyan",
            box=box.ROUNDED,
        )

    def run(self):
        """Run the daemon"""
        self.running = True

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Display startup message
        console.clear()
        console.print("\n")

        startup_panel = Panel(
            "[bold green]✓ JARVIS is now active![/bold green]\n\n"
            f"[bold]Wake words:[/bold] {', '.join(self.wake_words)}\n\n"
            "[dim]Say any wake word to start a conversation.[/dim]\n"
            "[dim]Press Ctrl+C to stop.[/dim]",
            title="🤖 JARVIS Daemon",
            border_style="green",
            box=box.DOUBLE,
        )
        console.print(startup_panel)
        console.print("\n")

        # Proactive greeting on startup (after a delay)
        if self.enable_proactive:
            console.print("[dim]Waiting for activity...[/dim]\n")
            time.sleep(5)

            if self._should_greet_proactively():
                console.print("[bold cyan]🎤 JARVIS: Initiating greeting...[/bold cyan]\n")
                self._have_conversation(initial_message=None)

        # Main listening loop
        wake_word_check_interval = 0.5  # Check every 500ms
        last_check = time.time()

        try:
            while self.running:
                current_time = time.time()

                # Display status
                if current_time - last_check > 10:  # Update every 10 seconds
                    console.clear()
                    console.print(self._display_status())
                    console.print("\n[dim]Listening for wake word...[/dim]\n")
                    last_check = current_time

                # Listen for wake word
                message = self._listen_for_wake_word()

                if message is not None:
                    console.print("\n[bold green]✓ Wake word detected![/bold green]\n")

                    # Have conversation
                    self._have_conversation(initial_message=message if message else None)

                    # Reset display
                    console.clear()
                    console.print(self._display_status())
                    console.print("\n[dim]Listening for wake word...[/dim]\n")

                # Small sleep to prevent CPU spinning
                time.sleep(wake_word_check_interval)

        except KeyboardInterrupt:
            console.print("\n\n[yellow]Shutting down JARVIS daemon...[/yellow]\n")

        finally:
            self.stop()

    def _signal_handler(self, sig, frame):
        """Handle shutdown signals"""
        console.print("\n\n[yellow]Received shutdown signal...[/yellow]\n")
        self.stop()

    def stop(self):
        """Stop the daemon"""
        self.running = False

        # Save memory
        memory_path = Path.home() / ".jarvis" / f"{self.user_id}_memory.json"
        memory_path.parent.mkdir(exist_ok=True)

        try:
            self.jarvis.save_memory(str(memory_path))
            console.print(f"[green]✓ Memory saved to {memory_path}[/green]\n")
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")

        console.print("[bold]JARVIS daemon stopped.[/bold]\n")
        sys.exit(0)


def main():
    """CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="JARVIS Always-On Daemon")
    parser.add_argument("--user-id", "-u", default="default", help="User identifier")
    parser.add_argument("--model", "-m", default="gpt2", help="AI model name")
    parser.add_argument("--config", "-c", help="Config file path")
    parser.add_argument(
        "--wake-words",
        "-w",
        nargs="+",
        default=["hi", "hello", "jarvis"],
        help="Wake words",
    )
    parser.add_argument(
        "--no-proactive",
        action="store_true",
        help="Disable proactive greetings",
    )

    args = parser.parse_args()

    # Load config from default location if not specified
    config_path = args.config
    if not config_path:
        default_config = Path.home() / ".jarvis" / "config.yaml"
        if default_config.exists():
            config_path = str(default_config)

    # Load config if exists
    config = None
    if config_path and Path(config_path).exists():
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Override with config values
        if config:
            args.user_id = config.get("user", {}).get("user_id", args.user_id)
            args.model = config.get("jarvis", {}).get("model_name", args.model)
            args.wake_words = config.get("voice", {}).get("wake_words", args.wake_words)

    # Setup logging
    setup_logging(level="INFO")

    # Create and run daemon
    daemon = JARVISDaemon(
        user_id=args.user_id,
        model_name=args.model,
        wake_words=args.wake_words,
        enable_proactive=not args.no_proactive,
        config_path=config_path,
    )

    daemon.run()


if __name__ == "__main__":
    main()
