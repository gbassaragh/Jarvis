#!/usr/bin/env python3
"""
JARVIS Installation Script

Beautiful interactive installer for the AI Assistant Pro JARVIS system.
Installs all dependencies, sets up voice, and configures the system.
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.live import Live
from rich.align import Align
from rich.text import Text
from rich import box

console = Console()


def print_banner():
    """Display stunning JARVIS banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║         ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗             ║
    ║         ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝             ║
    ║         ██║███████║██████╔╝██║   ██║██║███████╗             ║
    ║    ██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║             ║
    ║    ╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║             ║
    ║     ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝             ║
    ║                                                               ║
    ║        Just A Rather Very Intelligent System                 ║
    ║                                                               ║
    ║     🎤 Voice  🧠 Memory  🛠️ Tools  📚 Knowledge  🌐 Web      ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """

    console.print(banner, style="bold cyan", justify="center")
    console.print()
    console.print(
        Align.center(
            Text("Welcome to the future of AI assistance", style="italic bright_white")
        )
    )
    console.print()
    time.sleep(1)


def check_system_requirements():
    """Check system requirements"""
    console.print("\n[bold cyan]🔍 Checking System Requirements[/bold cyan]\n")

    requirements = {
        "Python Version": None,
        "CUDA Available": None,
        "GPU Memory": None,
        "Microphone": None,
        "Speaker": None,
    }

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:

        # Python version
        task = progress.add_task("Checking Python version...", total=100)
        time.sleep(0.5)
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        requirements["Python Version"] = f"✓ {python_version}"
        progress.update(task, advance=100)

        # CUDA
        task = progress.add_task("Checking CUDA availability...", total=100)
        time.sleep(0.5)
        try:
            import torch
            if torch.cuda.is_available():
                requirements["CUDA Available"] = f"✓ CUDA {torch.version.cuda}"
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
                requirements["GPU Memory"] = f"✓ {gpu_mem:.1f} GB"
            else:
                requirements["CUDA Available"] = "⚠ Not available (CPU mode)"
                requirements["GPU Memory"] = "N/A"
        except ImportError:
            requirements["CUDA Available"] = "⚠ PyTorch not installed yet"
            requirements["GPU Memory"] = "N/A"
        progress.update(task, advance=100)

        # Audio devices
        task = progress.add_task("Checking audio devices...", total=100)
        time.sleep(0.5)
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            has_mic = any(d['max_input_channels'] > 0 for d in devices)
            has_speaker = any(d['max_output_channels'] > 0 for d in devices)
            requirements["Microphone"] = "✓ Detected" if has_mic else "✗ Not found"
            requirements["Speaker"] = "✓ Detected" if has_speaker else "✗ Not found"
        except:
            requirements["Microphone"] = "? Unknown"
            requirements["Speaker"] = "? Unknown"
        progress.update(task, advance=100)

    # Display results
    table = Table(title="System Requirements", box=box.ROUNDED, show_header=True)
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")

    for component, status in requirements.items():
        table.add_row(component, status)

    console.print("\n")
    console.print(table)
    console.print()

    return requirements


def install_dependencies():
    """Install all dependencies"""
    console.print("\n[bold cyan]📦 Installing Dependencies[/bold cyan]\n")

    console.print("[dim]This may take a few minutes...[/dim]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console,
    ) as progress:

        # Install core dependencies
        task = progress.add_task("Installing core framework...", total=None)
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", ".", "-q"],
            capture_output=True,
        )
        if result.returncode != 0:
            console.print(f"[red]✗ Failed to install core framework[/red]")
            return False
        progress.update(task, completed=True)
        console.print("[green]✓[/green] Core framework installed")

        # Install JARVIS dependencies
        jarvis_deps = [
            ("sentence-transformers>=2.5.0", "Embedding models"),
            ("openai-whisper>=20231117", "Speech recognition"),
            ("duckduckgo-search>=5.0.0", "Web search"),
            ("SpeechRecognition>=3.10.0", "Audio input"),
            ("sounddevice>=0.4.6", "Audio I/O"),
            ("scipy>=1.12.0", "Audio processing"),
            ("websockets>=12.0", "Web interface"),
        ]

        for dep, description in jarvis_deps:
            task = progress.add_task(f"Installing {description}...", total=None)
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", dep, "-q"],
                capture_output=True,
            )
            if result.returncode == 0:
                progress.update(task, completed=True)
                console.print(f"[green]✓[/green] {description} installed")
            else:
                console.print(f"[yellow]⚠[/yellow] {description} installation had issues")

    console.print("\n[bold green]✓ All dependencies installed![/bold green]\n")
    return True


def setup_voice():
    """Set up voice interface"""
    console.print("\n[bold cyan]🎤 Voice Setup Wizard[/bold cyan]\n")

    enable_voice = Confirm.ask(
        "[bold]Enable voice interface?[/bold]\n"
        "This allows you to talk to JARVIS naturally",
        default=True,
    )

    if not enable_voice:
        return {"enabled": False}

    console.print("\n[dim]Configuring voice settings...[/dim]\n")

    # Wake words
    console.print("[bold]Choose wake words:[/bold]")
    console.print("  Default: 'Hi', 'Hello', 'Jarvis'")
    console.print("  These are natural words to activate JARVIS\n")

    use_default_wake_words = Confirm.ask("Use default wake words?", default=True)

    if use_default_wake_words:
        wake_words = ["hi", "hello", "jarvis"]
    else:
        wake_words_input = Prompt.ask("Enter wake words (comma-separated)")
        wake_words = [w.strip().lower() for w in wake_words_input.split(",")]

    # Voice calibration
    console.print("\n[bold]🎙️ Voice Calibration[/bold]")
    console.print("[dim]Testing microphone...[/dim]\n")

    calibrate = Confirm.ask("Test microphone now?", default=True)

    if calibrate:
        console.print("\n[yellow]Say something![/yellow]")
        console.print("[dim]Listening for 3 seconds...[/dim]\n")

        try:
            import sounddevice as sd
            import numpy as np

            duration = 3
            sample_rate = 16000

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Recording...", total=None)
                recording = sd.rec(
                    int(duration * sample_rate),
                    samplerate=sample_rate,
                    channels=1,
                    dtype=np.int16,
                )
                sd.wait()
                progress.update(task, completed=True)

            # Check if audio was captured
            max_amplitude = np.abs(recording).max()

            if max_amplitude > 100:
                console.print("[green]✓ Microphone working! Audio detected.[/green]\n")
            else:
                console.print("[yellow]⚠ Low audio level. Check microphone volume.[/yellow]\n")

        except Exception as e:
            console.print(f"[red]✗ Microphone test failed: {e}[/red]\n")

    return {
        "enabled": True,
        "wake_words": wake_words,
        "calibrated": calibrate,
    }


def setup_user_profile():
    """Set up user profile"""
    console.print("\n[bold cyan]👤 User Profile Setup[/bold cyan]\n")

    console.print("[dim]Let's personalize JARVIS for you[/dim]\n")

    name = Prompt.ask("[bold]What should JARVIS call you?[/bold]", default="there")

    console.print(f"\n[dim]Nice to meet you, {name}![/dim]\n")

    # User ID
    user_id = name.lower().replace(" ", "_")

    return {
        "name": name,
        "user_id": user_id,
    }


def configure_jarvis():
    """Configure JARVIS settings"""
    console.print("\n[bold cyan]⚙️  JARVIS Configuration[/bold cyan]\n")

    # Model selection
    console.print("[bold]Select AI Model:[/bold]")
    console.print("  1. GPT-2 (Small, fast, offline)")
    console.print("  2. GPT-2 Medium (Better quality)")
    console.print("  3. GPT-2 Large (Best quality, slower)\n")

    model_choice = Prompt.ask(
        "Choose model",
        choices=["1", "2", "3"],
        default="1",
    )

    model_map = {
        "1": "gpt2",
        "2": "gpt2-medium",
        "3": "gpt2-large",
    }
    model_name = model_map[model_choice]

    # Features
    console.print("\n[bold]Enable Features:[/bold]\n")

    enable_memory = Confirm.ask("💾 Conversational memory (remembers past conversations)", default=True)
    enable_tools = Confirm.ask("🛠️  Tools (web search, calculator, etc.)", default=True)
    enable_rag = Confirm.ask("📚 Knowledge base (RAG)", default=True)

    # Performance
    console.print("\n[bold]Performance Options:[/bold]\n")

    use_triton = Confirm.ask("⚡ Use Triton kernels (faster on NVIDIA GPUs)", default=True)
    use_fp8 = Confirm.ask("🔢 Use FP8 quantization (Blackwell GPUs only)", default=False)

    return {
        "model_name": model_name,
        "enable_memory": enable_memory,
        "enable_tools": enable_tools,
        "enable_rag": enable_rag,
        "use_triton": use_triton,
        "use_fp8": use_fp8,
    }


def create_config_file(config: dict):
    """Create configuration file"""
    console.print("\n[bold cyan]💾 Saving Configuration[/bold cyan]\n")

    config_dir = Path.home() / ".jarvis"
    config_dir.mkdir(exist_ok=True)

    config_file = config_dir / "config.yaml"

    import yaml

    with open(config_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    console.print(f"[green]✓ Configuration saved to {config_file}[/green]\n")

    return config_file


def create_launcher():
    """Create desktop launcher"""
    console.print("\n[bold cyan]🚀 Creating Launcher[/bold cyan]\n")

    create = Confirm.ask("Create desktop launcher?", default=True)

    if not create:
        return

    # Create launcher script
    launcher_dir = Path.home() / ".local" / "bin"
    launcher_dir.mkdir(parents=True, exist_ok=True)

    launcher_script = launcher_dir / "jarvis"

    script_content = f"""#!/bin/bash
# JARVIS Launcher

cd {Path.cwd()}
{sys.executable} -m ai_assistant_pro.jarvis.daemon "$@"
"""

    with open(launcher_script, "w") as f:
        f.write(script_content)

    launcher_script.chmod(0o755)

    console.print(f"[green]✓ Launcher created: {launcher_script}[/green]")
    console.print("[dim]You can now run 'jarvis' from anywhere![/dim]\n")

    # Add to PATH if not already
    bashrc = Path.home() / ".bashrc"
    if bashrc.exists():
        with open(bashrc, "r") as f:
            content = f.read()

        if str(launcher_dir) not in content:
            with open(bashrc, "a") as f:
                f.write(f'\n# JARVIS\nexport PATH="$PATH:{launcher_dir}"\n')
            console.print(f"[green]✓ Added to PATH in {bashrc}[/green]\n")


def print_completion(config: dict):
    """Print completion message with instructions"""
    console.print("\n")
    console.print("═" * 70, style="cyan")
    console.print()

    success_panel = Panel(
        "[bold green]✓ Installation Complete![/bold green]\n\n"
        f"[bold]Welcome, {config.get('user', {}).get('name', 'User')}![/bold]\n\n"
        "JARVIS is now ready to assist you.",
        title="🎉 Success",
        border_style="green",
        box=box.DOUBLE,
    )

    console.print(Align.center(success_panel))
    console.print()

    # Usage instructions
    usage_panel = Panel(
        "[bold cyan]Quick Start:[/bold cyan]\n\n"
        "🎤 [bold]Voice Assistant (Always-On):[/bold]\n"
        "   $ ai-assistant-pro jarvis daemon\n"
        "   Then just say: 'Hi Jarvis!'\n\n"
        "💬 [bold]Interactive Chat:[/bold]\n"
        "   $ ai-assistant-pro jarvis chat\n\n"
        "🌐 [bold]Web Interface:[/bold]\n"
        "   $ ai-assistant-pro jarvis serve\n"
        "   Visit: http://localhost:8080\n\n"
        "📚 [bold]Load Knowledge:[/bold]\n"
        "   $ ai-assistant-pro jarvis load-knowledge ./docs\n\n"
        "📊 [bold]View Statistics:[/bold]\n"
        "   $ ai-assistant-pro jarvis stats\n",
        title="Usage",
        border_style="cyan",
        box=box.ROUNDED,
    )

    console.print(usage_panel)
    console.print()

    # Next steps
    next_steps = Table(title="Next Steps", box=box.SIMPLE, show_header=False)
    next_steps.add_column("", style="cyan")
    next_steps.add_column("", style="white")

    next_steps.add_row("1.", "Start the voice assistant: ai-assistant-pro jarvis daemon")
    next_steps.add_row("2.", "Say 'Hi' to activate JARVIS")
    next_steps.add_row("3.", "Have a conversation!")
    next_steps.add_row("4.", "JARVIS will remember you for next time")

    console.print(next_steps)
    console.print()

    console.print("═" * 70, style="cyan")
    console.print()

    console.print(
        Align.center(
            Text("Your personal AI assistant is ready! 🤖✨", style="bold bright_white")
        )
    )
    console.print()


def main():
    """Main installation flow"""
    try:
        # Clear screen
        console.clear()

        # Banner
        print_banner()
        time.sleep(1)

        # Welcome
        welcome_panel = Panel(
            "[bold]This installer will set up JARVIS on your system.[/bold]\n\n"
            "JARVIS is your personal AI assistant with:\n"
            "  🎤 Natural voice interaction\n"
            "  🧠 Long-term memory\n"
            "  🛠️ Powerful tools\n"
            "  📚 Knowledge base\n"
            "  🌐 Beautiful web interface\n\n"
            "[dim]Press Ctrl+C at any time to cancel[/dim]",
            title="Welcome",
            border_style="bright_blue",
            box=box.DOUBLE,
        )
        console.print(Align.center(welcome_panel))
        console.print()

        proceed = Confirm.ask("[bold]Ready to begin?[/bold]", default=True)

        if not proceed:
            console.print("\n[yellow]Installation cancelled.[/yellow]\n")
            return

        # Configuration object
        config = {}

        # Check requirements
        requirements = check_system_requirements()
        time.sleep(1)

        # Install dependencies
        if not install_dependencies():
            console.print("\n[red]Installation failed.[/red]\n")
            return
        time.sleep(1)

        # Voice setup
        voice_config = setup_voice()
        config["voice"] = voice_config
        time.sleep(0.5)

        # User profile
        user_config = setup_user_profile()
        config["user"] = user_config
        time.sleep(0.5)

        # JARVIS configuration
        jarvis_config = configure_jarvis()
        config["jarvis"] = jarvis_config
        time.sleep(0.5)

        # Save configuration
        config_file = create_config_file(config)
        time.sleep(0.5)

        # Create launcher
        create_launcher()
        time.sleep(0.5)

        # Completion
        print_completion(config)

        # Auto-start option
        auto_start = Confirm.ask(
            "\n[bold]Start JARVIS voice assistant now?[/bold]",
            default=True,
        )

        if auto_start:
            console.print("\n[bold cyan]🚀 Starting JARVIS...[/bold cyan]\n")
            time.sleep(1)

            # Start daemon
            subprocess.run([
                sys.executable,
                "-m",
                "ai_assistant_pro.jarvis.daemon",
                "--user-id",
                user_config["user_id"],
            ])

    except KeyboardInterrupt:
        console.print("\n\n[yellow]Installation cancelled by user.[/yellow]\n")
        sys.exit(1)

    except Exception as e:
        console.print(f"\n\n[red]Installation error: {e}[/red]\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
