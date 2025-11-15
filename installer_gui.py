#!/usr/bin/env python3
"""
JARVIS Professional Installer
Adobe/Microsoft Flight Simulator style installer

Beautiful GUI with model selection, download tracking, and progress visualization.
"""

import os
import sys
import threading
import queue
import json
from pathlib import Path
from typing import Dict, List, Optional
import tkinter as tk
from tkinter import ttk, messagebox, font
import subprocess

# Try to import required libraries
try:
    from PIL import Image, ImageTk, ImageDraw
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class ModelInfo:
    """Information about a downloadable model"""

    def __init__(
        self,
        name: str,
        display_name: str,
        description: str,
        size_mb: int,
        repo_id: str,
        required: bool = False,
        category: str = "Optional",
    ):
        self.name = name
        self.display_name = display_name
        self.description = description
        self.size_mb = size_mb
        self.repo_id = repo_id
        self.required = required
        self.category = category


# Available models for download
AVAILABLE_MODELS = {
    # Language Models
    "gpt2": ModelInfo(
        name="gpt2",
        display_name="GPT-2 Small (Recommended)",
        description="Fast, lightweight language model. Perfect for quick responses.",
        size_mb=500,
        repo_id="gpt2",
        required=True,
        category="Language Model",
    ),
    "gpt2-medium": ModelInfo(
        name="gpt2-medium",
        display_name="GPT-2 Medium",
        description="Better quality responses, slightly slower.",
        size_mb=1500,
        repo_id="gpt2-medium",
        category="Language Model",
    ),
    "gpt2-large": ModelInfo(
        name="gpt2-large",
        display_name="GPT-2 Large",
        description="Highest quality, requires more memory and time.",
        size_mb=3000,
        repo_id="gpt2-large",
        category="Language Model",
    ),
    # Speech-to-Text
    "whisper-tiny": ModelInfo(
        name="whisper-tiny",
        display_name="Whisper Tiny",
        description="Fastest speech recognition, good for wake word detection.",
        size_mb=150,
        repo_id="openai/whisper-tiny",
        category="Speech-to-Text",
    ),
    "whisper-base": ModelInfo(
        name="whisper-base",
        display_name="Whisper Base (Recommended)",
        description="Balanced speed and accuracy for voice commands.",
        size_mb=300,
        repo_id="openai/whisper-base",
        required=True,
        category="Speech-to-Text",
    ),
    "whisper-small": ModelInfo(
        name="whisper-small",
        display_name="Whisper Small",
        description="Better accuracy, slightly slower transcription.",
        size_mb=950,
        repo_id="openai/whisper-small",
        category="Speech-to-Text",
    ),
    # Text-to-Speech
    "bark": ModelInfo(
        name="bark",
        display_name="Bark TTS (Recommended)",
        description="Natural-sounding text-to-speech with emotional expression.",
        size_mb=1200,
        repo_id="suno/bark",
        required=True,
        category="Text-to-Speech",
    ),
    # Embeddings
    "embeddings-mini": ModelInfo(
        name="embeddings-mini",
        display_name="MiniLM Embeddings (Recommended)",
        description="Fast embeddings for memory and knowledge retrieval.",
        size_mb=80,
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        required=True,
        category="Embeddings",
    ),
    "embeddings-mpnet": ModelInfo(
        name="embeddings-mpnet",
        display_name="MPNet Embeddings",
        description="Higher quality embeddings, better memory retrieval.",
        size_mb=420,
        repo_id="sentence-transformers/all-mpnet-base-v2",
        category="Embeddings",
    ),
}


class JARVISInstaller:
    """Professional GUI installer for JARVIS"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("JARVIS Setup")
        self.root.geometry("900x600")
        self.root.resizable(False, False)

        # State
        self.current_page = 0
        self.selected_models = {}
        self.install_path = Path.home() / ".jarvis"
        self.installation_complete = False
        self.install_queue = queue.Queue()

        # Colors (Adobe-style dark theme)
        self.colors = {
            "bg": "#1e1e1e",
            "fg": "#ffffff",
            "accent": "#0078d4",
            "success": "#4caf50",
            "warning": "#ff9800",
            "panel": "#2d2d30",
            "border": "#3e3e42",
            "text_secondary": "#cccccc",
        }

        # Setup UI
        self.setup_styles()
        self.setup_ui()

        # Initialize with required models selected
        for model_id, model in AVAILABLE_MODELS.items():
            if model.required:
                self.selected_models[model_id] = True
            else:
                self.selected_models[model_id] = False

    def setup_styles(self):
        """Setup ttk styles"""
        style = ttk.Style()
        style.theme_use("clam")

        # Configure colors
        self.root.configure(bg=self.colors["bg"])

        # Button style
        style.configure(
            "Accent.TButton",
            background=self.colors["accent"],
            foreground=self.colors["fg"],
            borderwidth=0,
            focuscolor="none",
            font=("Segoe UI", 10),
            padding=10,
        )
        style.map(
            "Accent.TButton",
            background=[("active", "#005a9e")],
        )

        # Secondary button
        style.configure(
            "Secondary.TButton",
            background=self.colors["panel"],
            foreground=self.colors["fg"],
            borderwidth=1,
            relief="solid",
            font=("Segoe UI", 10),
            padding=10,
        )

        # Progress bar
        style.configure(
            "TProgressbar",
            background=self.colors["accent"],
            troughcolor=self.colors["panel"],
            bordercolor=self.colors["border"],
            lightcolor=self.colors["accent"],
            darkcolor=self.colors["accent"],
        )

    def setup_ui(self):
        """Setup main UI structure"""
        # Main container
        self.main_frame = tk.Frame(self.root, bg=self.colors["bg"])
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Content area (changes per page)
        self.content_frame = tk.Frame(
            self.main_frame,
            bg=self.colors["bg"],
        )
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Bottom navigation bar
        self.nav_frame = tk.Frame(
            self.main_frame,
            bg=self.colors["panel"],
            height=80,
        )
        self.nav_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.nav_frame.pack_propagate(False)

        # Navigation buttons
        btn_frame = tk.Frame(self.nav_frame, bg=self.colors["panel"])
        btn_frame.pack(side=tk.RIGHT, padx=20, pady=20)

        self.back_btn = ttk.Button(
            btn_frame,
            text="← Back",
            style="Secondary.TButton",
            command=self.previous_page,
            width=15,
        )
        self.back_btn.pack(side=tk.LEFT, padx=5)

        self.next_btn = ttk.Button(
            btn_frame,
            text="Next →",
            style="Accent.TButton",
            command=self.next_page,
            width=15,
        )
        self.next_btn.pack(side=tk.LEFT, padx=5)

        # Show welcome page
        self.show_welcome_page()

    def clear_content(self):
        """Clear content frame"""
        for widget in self.content_frame.winfo_children():
            widget.destroy()

    def show_welcome_page(self):
        """Show welcome screen"""
        self.clear_content()
        self.current_page = 0

        # JARVIS logo/title
        title_frame = tk.Frame(self.content_frame, bg=self.colors["bg"])
        title_frame.pack(pady=40)

        title = tk.Label(
            title_frame,
            text="JARVIS",
            font=("Segoe UI", 48, "bold"),
            fg=self.colors["accent"],
            bg=self.colors["bg"],
        )
        title.pack()

        subtitle = tk.Label(
            title_frame,
            text="Just A Rather Very Intelligent System",
            font=("Segoe UI", 14),
            fg=self.colors["text_secondary"],
            bg=self.colors["bg"],
        )
        subtitle.pack(pady=5)

        # Welcome message
        message_frame = tk.Frame(self.content_frame, bg=self.colors["bg"])
        message_frame.pack(pady=30, padx=60)

        welcome_text = """Welcome to the JARVIS Setup Wizard

This wizard will guide you through the installation of JARVIS,
your personal AI assistant with voice, memory, and intelligence.

Features:
  🎤  Natural voice interaction
  🧠  Long-term conversational memory
  🛠️  Powerful tools (web search, calculator, code execution)
  📚  Knowledge base with RAG
  🌐  Beautiful web interface

You will be able to choose which AI models to download
based on your needs and available disk space.
"""

        welcome_label = tk.Label(
            message_frame,
            text=welcome_text,
            font=("Segoe UI", 11),
            fg=self.colors["fg"],
            bg=self.colors["bg"],
            justify=tk.LEFT,
        )
        welcome_label.pack()

        # Update navigation
        self.back_btn.config(state=tk.DISABLED)
        self.next_btn.config(text="Next →", state=tk.NORMAL)

    def show_model_selection_page(self):
        """Show model selection screen"""
        self.clear_content()
        self.current_page = 1

        # Title
        title = tk.Label(
            self.content_frame,
            text="Select Models to Download",
            font=("Segoe UI", 24, "bold"),
            fg=self.colors["fg"],
            bg=self.colors["bg"],
        )
        title.pack(pady=(0, 10))

        subtitle = tk.Label(
            self.content_frame,
            text="Choose which AI models to install. Required models are pre-selected.",
            font=("Segoe UI", 10),
            fg=self.colors["text_secondary"],
            bg=self.colors["bg"],
        )
        subtitle.pack(pady=(0, 20))

        # Scrollable frame for models
        canvas_frame = tk.Frame(self.content_frame, bg=self.colors["bg"])
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(
            canvas_frame,
            bg=self.colors["bg"],
            highlightthickness=0,
        )
        scrollbar = ttk.Scrollbar(
            canvas_frame,
            orient="vertical",
            command=canvas.yview,
        )

        scrollable_frame = tk.Frame(canvas, bg=self.colors["bg"])

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Group models by category
        categories = {}
        for model_id, model in AVAILABLE_MODELS.items():
            if model.category not in categories:
                categories[model.category] = []
            categories[model.category].append((model_id, model))

        # Display each category
        for category, models in categories.items():
            # Category header
            category_frame = tk.Frame(
                scrollable_frame,
                bg=self.colors["panel"],
                highlightbackground=self.colors["border"],
                highlightthickness=1,
            )
            category_frame.pack(fill=tk.X, pady=10, padx=20)

            category_label = tk.Label(
                category_frame,
                text=category,
                font=("Segoe UI", 12, "bold"),
                fg=self.colors["accent"],
                bg=self.colors["panel"],
            )
            category_label.pack(anchor=tk.W, padx=15, pady=10)

            # Models in category
            for model_id, model in models:
                self.create_model_checkbox(scrollable_frame, model_id, model)

        # Total size indicator
        self.size_label = tk.Label(
            self.content_frame,
            text="",
            font=("Segoe UI", 10, "bold"),
            fg=self.colors["accent"],
            bg=self.colors["bg"],
        )
        self.size_label.pack(pady=10)

        self.update_total_size()

        # Update navigation
        self.back_btn.config(state=tk.NORMAL)
        self.next_btn.config(text="Install", state=tk.NORMAL)

    def create_model_checkbox(self, parent, model_id: str, model: ModelInfo):
        """Create a model selection checkbox"""
        model_frame = tk.Frame(
            parent,
            bg=self.colors["bg"],
            highlightbackground=self.colors["border"],
            highlightthickness=1,
        )
        model_frame.pack(fill=tk.X, pady=5, padx=40)

        # Checkbox and name
        checkbox_var = tk.BooleanVar(value=self.selected_models.get(model_id, False))

        def on_toggle():
            if model.required and not checkbox_var.get():
                checkbox_var.set(True)
                messagebox.showinfo(
                    "Required Model",
                    f"{model.display_name} is required for JARVIS to function.",
                )
                return

            self.selected_models[model_id] = checkbox_var.get()
            self.update_total_size()

        checkbox = tk.Checkbutton(
            model_frame,
            text=model.display_name,
            variable=checkbox_var,
            command=on_toggle,
            font=("Segoe UI", 10, "bold"),
            fg=self.colors["fg"],
            bg=self.colors["bg"],
            selectcolor=self.colors["panel"],
            activebackground=self.colors["bg"],
            activeforeground=self.colors["fg"],
        )
        checkbox.pack(anchor=tk.W, padx=15, pady=5)

        # Description
        desc_label = tk.Label(
            model_frame,
            text=model.description,
            font=("Segoe UI", 9),
            fg=self.colors["text_secondary"],
            bg=self.colors["bg"],
            wraplength=700,
            justify=tk.LEFT,
        )
        desc_label.pack(anchor=tk.W, padx=40, pady=(0, 5))

        # Size and required indicator
        info_frame = tk.Frame(model_frame, bg=self.colors["bg"])
        info_frame.pack(anchor=tk.W, padx=40, pady=(0, 10))

        size_label = tk.Label(
            info_frame,
            text=f"Size: {model.size_mb} MB",
            font=("Segoe UI", 8),
            fg=self.colors["text_secondary"],
            bg=self.colors["bg"],
        )
        size_label.pack(side=tk.LEFT, padx=(0, 10))

        if model.required:
            required_label = tk.Label(
                info_frame,
                text="REQUIRED",
                font=("Segoe UI", 8, "bold"),
                fg=self.colors["warning"],
                bg=self.colors["bg"],
            )
            required_label.pack(side=tk.LEFT)

    def update_total_size(self):
        """Update total download size label"""
        total_mb = sum(
            AVAILABLE_MODELS[mid].size_mb
            for mid, selected in self.selected_models.items()
            if selected
        )

        total_gb = total_mb / 1024
        self.size_label.config(
            text=f"Total Download Size: {total_mb:,} MB ({total_gb:.2f} GB)"
        )

    def show_installation_page(self):
        """Show installation progress screen"""
        self.clear_content()
        self.current_page = 2

        # Title
        title = tk.Label(
            self.content_frame,
            text="Installing JARVIS",
            font=("Segoe UI", 24, "bold"),
            fg=self.colors["fg"],
            bg=self.colors["bg"],
        )
        title.pack(pady=(0, 10))

        subtitle = tk.Label(
            self.content_frame,
            text="Please wait while we download and install the selected models...",
            font=("Segoe UI", 10),
            fg=self.colors["text_secondary"],
            bg=self.colors["bg"],
        )
        subtitle.pack(pady=(0, 30))

        # Overall progress
        overall_frame = tk.Frame(self.content_frame, bg=self.colors["bg"])
        overall_frame.pack(fill=tk.X, padx=40, pady=10)

        self.overall_label = tk.Label(
            overall_frame,
            text="Preparing installation...",
            font=("Segoe UI", 11, "bold"),
            fg=self.colors["fg"],
            bg=self.colors["bg"],
        )
        self.overall_label.pack(anchor=tk.W, pady=5)

        self.overall_progress = ttk.Progressbar(
            overall_frame,
            mode="determinate",
            length=800,
        )
        self.overall_progress.pack(fill=tk.X, pady=5)

        # Current task
        task_frame = tk.Frame(self.content_frame, bg=self.colors["bg"])
        task_frame.pack(fill=tk.X, padx=40, pady=20)

        self.task_label = tk.Label(
            task_frame,
            text="",
            font=("Segoe UI", 10),
            fg=self.colors["text_secondary"],
            bg=self.colors["bg"],
        )
        self.task_label.pack(anchor=tk.W, pady=5)

        self.task_progress = ttk.Progressbar(
            task_frame,
            mode="indeterminate",
            length=800,
        )
        self.task_progress.pack(fill=tk.X, pady=5)

        # Log output
        log_frame = tk.Frame(
            self.content_frame,
            bg=self.colors["panel"],
            highlightbackground=self.colors["border"],
            highlightthickness=1,
        )
        log_frame.pack(fill=tk.BOTH, expand=True, padx=40, pady=10)

        self.log_text = tk.Text(
            log_frame,
            font=("Consolas", 9),
            bg=self.colors["panel"],
            fg=self.colors["text_secondary"],
            wrap=tk.WORD,
            height=10,
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Update navigation
        self.back_btn.config(state=tk.DISABLED)
        self.next_btn.config(state=tk.DISABLED)

        # Start installation
        self.start_installation()

    def log(self, message: str):
        """Add message to log"""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update()

    def start_installation(self):
        """Start the installation process"""
        self.task_progress.start(10)

        # Run installation in background thread
        install_thread = threading.Thread(target=self.run_installation, daemon=True)
        install_thread.start()

        # Monitor progress
        self.root.after(100, self.check_installation_progress)

    def run_installation(self):
        """Run installation tasks"""
        selected_model_ids = [
            mid for mid, selected in self.selected_models.items() if selected
        ]

        total_steps = len(selected_model_ids) + 3  # +3 for core, deps, config
        current_step = 0

        try:
            # Step 1: Install core framework
            self.install_queue.put(("status", "Installing core framework..."))
            self.install_queue.put(("task", "Installing AI Assistant Pro"))

            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", ".", "-q"],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                self.install_queue.put(("log", "✓ Core framework installed"))
            else:
                self.install_queue.put(("log", f"✗ Error: {result.stderr}"))
                return

            current_step += 1
            progress = int((current_step / total_steps) * 100)
            self.install_queue.put(("progress", progress))

            # Step 2: Install dependencies
            self.install_queue.put(("status", "Installing dependencies..."))
            self.install_queue.put(("task", "Installing JARVIS dependencies"))

            deps = [
                "sentence-transformers",
                "openai-whisper",
                "duckduckgo-search",
                "SpeechRecognition",
                "sounddevice",
                "scipy",
                "websockets",
            ]

            for dep in deps:
                self.install_queue.put(("log", f"Installing {dep}..."))
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", dep, "-q"],
                    capture_output=True,
                )

            self.install_queue.put(("log", "✓ Dependencies installed"))

            current_step += 1
            progress = int((current_step / total_steps) * 100)
            self.install_queue.put(("progress", progress))

            # Step 3: Download models
            for model_id in selected_model_ids:
                model = AVAILABLE_MODELS[model_id]

                self.install_queue.put(
                    ("status", f"Downloading {model.display_name}...")
                )
                self.install_queue.put(("task", f"Model: {model.repo_id}"))
                self.install_queue.put(
                    ("log", f"Downloading {model.display_name} ({model.size_mb} MB)...")
                )

                # Download model using transformers/sentence-transformers
                if model.category == "Embeddings":
                    code = f"""
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('{model.repo_id}')
print('Downloaded successfully')
"""
                else:
                    code = f"""
from transformers import AutoModel, AutoTokenizer
try:
    model = AutoModel.from_pretrained('{model.repo_id}')
    tokenizer = AutoTokenizer.from_pretrained('{model.repo_id}')
    print('Downloaded successfully')
except:
    # Try other model classes
    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
    try:
        model = AutoModelForCausalLM.from_pretrained('{model.repo_id}')
    except:
        model = AutoModelForSeq2SeqLM.from_pretrained('{model.repo_id}')
    tokenizer = AutoTokenizer.from_pretrained('{model.repo_id}')
    print('Downloaded successfully')
"""

                result = subprocess.run(
                    [sys.executable, "-c", code],
                    capture_output=True,
                    text=True,
                )

                if result.returncode == 0:
                    self.install_queue.put(("log", f"✓ {model.display_name} ready"))
                else:
                    self.install_queue.put(
                        ("log", f"⚠ {model.display_name} download issue (will download on first use)")
                    )

                current_step += 1
                progress = int((current_step / total_steps) * 100)
                self.install_queue.put(("progress", progress))

            # Step 4: Create configuration
            self.install_queue.put(("status", "Creating configuration..."))
            self.install_queue.put(("task", "Setting up JARVIS"))

            config_dir = Path.home() / ".jarvis"
            config_dir.mkdir(exist_ok=True)

            config = {
                "installed_models": selected_model_ids,
                "install_path": str(config_dir),
                "jarvis": {
                    "model_name": "gpt2",
                    "enable_voice": True,
                    "enable_memory": True,
                    "enable_tools": True,
                    "enable_rag": True,
                },
                "voice": {
                    "wake_words": ["hi", "hello", "jarvis"],
                    "stt_model": "openai/whisper-base",
                    "tts_model": "suno/bark",
                },
            }

            import yaml

            with open(config_dir / "config.yaml", "w") as f:
                yaml.dump(config, f)

            self.install_queue.put(("log", f"✓ Configuration saved to {config_dir}"))

            current_step += 1
            self.install_queue.put(("progress", 100))

            # Complete
            self.install_queue.put(("status", "Installation complete!"))
            self.install_queue.put(("task", "JARVIS is ready to use"))
            self.install_queue.put(("complete", True))

        except Exception as e:
            self.install_queue.put(("error", str(e)))
            self.install_queue.put(("log", f"✗ Installation failed: {e}"))

    def check_installation_progress(self):
        """Check installation progress from queue"""
        try:
            while True:
                msg_type, msg_data = self.install_queue.get_nowait()

                if msg_type == "status":
                    self.overall_label.config(text=msg_data)
                elif msg_type == "task":
                    self.task_label.config(text=msg_data)
                elif msg_type == "progress":
                    self.overall_progress["value"] = msg_data
                elif msg_type == "log":
                    self.log(msg_data)
                elif msg_type == "complete":
                    self.task_progress.stop()
                    self.installation_complete = True
                    self.next_btn.config(text="Finish", state=tk.NORMAL)
                elif msg_type == "error":
                    self.task_progress.stop()
                    messagebox.showerror("Installation Error", msg_data)
                    self.back_btn.config(state=tk.NORMAL)

        except queue.Empty:
            pass

        # Continue checking if not complete
        if not self.installation_complete:
            self.root.after(100, self.check_installation_progress)

    def show_completion_page(self):
        """Show installation completion screen"""
        self.clear_content()
        self.current_page = 3

        # Success icon/message
        success_frame = tk.Frame(self.content_frame, bg=self.colors["bg"])
        success_frame.pack(pady=40)

        # Success checkmark
        check_label = tk.Label(
            success_frame,
            text="✓",
            font=("Segoe UI", 72),
            fg=self.colors["success"],
            bg=self.colors["bg"],
        )
        check_label.pack()

        title = tk.Label(
            success_frame,
            text="Installation Complete!",
            font=("Segoe UI", 28, "bold"),
            fg=self.colors["fg"],
            bg=self.colors["bg"],
        )
        title.pack(pady=10)

        subtitle = tk.Label(
            success_frame,
            text="JARVIS is now installed and ready to use.",
            font=("Segoe UI", 12),
            fg=self.colors["text_secondary"],
            bg=self.colors["bg"],
        )
        subtitle.pack()

        # Quick start info
        info_frame = tk.Frame(
            self.content_frame,
            bg=self.colors["panel"],
            highlightbackground=self.colors["border"],
            highlightthickness=1,
        )
        info_frame.pack(fill=tk.X, padx=80, pady=30)

        info_title = tk.Label(
            info_frame,
            text="Quick Start",
            font=("Segoe UI", 14, "bold"),
            fg=self.colors["accent"],
            bg=self.colors["panel"],
        )
        info_title.pack(anchor=tk.W, padx=20, pady=(15, 10))

        quick_start = """To start JARVIS:

  🎤  Voice Assistant (Always-On):
      ai-assistant-pro jarvis daemon

  💬  Interactive Chat:
      ai-assistant-pro jarvis chat

  🌐  Web Interface:
      ai-assistant-pro jarvis serve

Just say "Hi" to activate JARVIS!
"""

        info_label = tk.Label(
            info_frame,
            text=quick_start,
            font=("Segoe UI", 10),
            fg=self.colors["fg"],
            bg=self.colors["panel"],
            justify=tk.LEFT,
        )
        info_label.pack(anchor=tk.W, padx=40, pady=(0, 15))

        # Checkbox for launching
        self.launch_var = tk.BooleanVar(value=True)

        launch_check = tk.Checkbutton(
            self.content_frame,
            text="Launch JARVIS now",
            variable=self.launch_var,
            font=("Segoe UI", 11),
            fg=self.colors["fg"],
            bg=self.colors["bg"],
            selectcolor=self.colors["panel"],
        )
        launch_check.pack(pady=10)

        # Update navigation
        self.back_btn.config(state=tk.DISABLED)
        self.next_btn.config(text="Finish", state=tk.NORMAL)

    def next_page(self):
        """Go to next page"""
        if self.current_page == 0:
            self.show_model_selection_page()
        elif self.current_page == 1:
            self.show_installation_page()
        elif self.current_page == 2:
            self.show_completion_page()
        elif self.current_page == 3:
            # Finish
            if self.launch_var.get():
                self.launch_jarvis()
            self.root.quit()

    def previous_page(self):
        """Go to previous page"""
        if self.current_page == 1:
            self.show_welcome_page()

    def launch_jarvis(self):
        """Launch JARVIS after installation"""
        try:
            subprocess.Popen(
                [sys.executable, "-m", "ai_assistant_pro.jarvis.daemon"],
                start_new_session=True,
            )
        except Exception as e:
            messagebox.showerror("Launch Error", f"Failed to launch JARVIS: {e}")

    def run(self):
        """Run the installer"""
        # Center window
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (900 // 2)
        y = (self.root.winfo_screenheight() // 2) - (600 // 2)
        self.root.geometry(f"900x600+{x}+{y}")

        self.root.mainloop()


def main():
    """Main entry point"""
    # Check if running with GUI support
    try:
        installer = JARVISInstaller()
        installer.run()
    except Exception as e:
        print(f"Error launching installer: {e}")
        print("\nFalling back to CLI installer...")
        print("Run: python install.py")
        sys.exit(1)


if __name__ == "__main__":
    main()
