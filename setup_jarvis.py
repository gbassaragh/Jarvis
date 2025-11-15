#!/usr/bin/env python3
"""
JARVIS Setup Launcher

Small bootstrap script that launches the professional GUI installer.
This is the entry point users double-click or run.
"""

import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Check if Python version is adequate"""
    if sys.version_info < (3, 8):
        print("ERROR: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        input("\nPress Enter to exit...")
        sys.exit(1)


def install_requirements():
    """Install basic requirements for installer"""
    print("Installing installer dependencies...")

    required = ["PyYAML", "rich"]

    for package in required:
        print(f"Installing {package}...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", package, "-q"],
            check=False,
        )

    print("✓ Installer dependencies ready\n")


def launch_installer():
    """Launch the GUI installer"""
    installer_path = Path(__file__).parent / "installer_gui.py"

    if not installer_path.exists():
        print(f"ERROR: Installer not found at {installer_path}")
        input("\nPress Enter to exit...")
        sys.exit(1)

    print("Launching JARVIS Setup...")
    print()

    try:
        subprocess.run([sys.executable, str(installer_path)])
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: Failed to launch installer: {e}")
        print("\nTrying CLI installer instead...")
        print("Run: python install.py")
        input("\nPress Enter to exit...")
        sys.exit(1)


def main():
    """Main setup flow"""
    print("=" * 60)
    print(" " * 15 + "JARVIS Setup")
    print("=" * 60)
    print()

    # Check Python version
    check_python_version()

    # Install requirements
    install_requirements()

    # Launch installer
    launch_installer()


if __name__ == "__main__":
    main()
