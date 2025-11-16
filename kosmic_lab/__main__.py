"""
Kosmic Lab CLI - Version and System Information

Usage:
    python -m kosmic_lab --version
    python -m kosmic_lab --info
    python -m kosmic_lab --help
"""

import argparse
import platform
import sys
from pathlib import Path
from typing import Dict, List

try:
    from kosmic_lab import (
        PROJECT_NAME,
        PROJECT_TAGLINE,
        REPOSITORY,
        __author__,
        __email__,
        __license__,
        __version__,
    )
except ImportError:
    # Fallback if running from outside package
    __version__ = "1.1.0"
    PROJECT_NAME = "Kosmic Lab"
    PROJECT_TAGLINE = "Revolutionary AI-Accelerated Platform for Consciousness Research"
    REPOSITORY = "https://github.com/Luminous-Dynamics/kosmic-lab"
    __author__ = "Kosmic Lab Team"
    __email__ = "kosmic-lab@example.org"
    __license__ = "MIT"


def check_dependencies() -> Dict[str, bool]:
    """Check which dependencies are installed."""
    deps = {}
    required = [
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "networkx",
        "sklearn",
        "gymnasium",
        "yaml",
    ]

    for dep in required:
        try:
            __import__(dep)
            deps[dep] = True
        except ImportError:
            deps[dep] = False

    return deps


def get_git_info() -> Dict[str, str]:
    """Get git repository information."""
    import subprocess

    info = {}
    try:
        # Get current branch
        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        info["branch"] = branch

        # Get commit SHA
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        info["commit"] = sha

        # Check if working directory is clean
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode()
        info["clean"] = len(status.strip()) == 0

    except (subprocess.CalledProcessError, FileNotFoundError):
        info["available"] = False

    return info


def show_version():
    """Display version information."""
    print(f"🌊 {PROJECT_NAME} v{__version__}")


def show_info():
    """Display comprehensive system information."""
    print("=" * 70)
    print(f"🌊 {PROJECT_NAME}")
    print(f"   {PROJECT_TAGLINE}")
    print("=" * 70)
    print()

    # Version information
    print(f"📦 Version:        {__version__}")
    print(f"🐍 Python:         {sys.version.split()[0]} ({platform.python_implementation()})")
    print(f"💻 Platform:       {platform.system()} {platform.release()} ({platform.machine()})")
    print()

    # Installation path
    install_path = Path(__file__).parent.parent
    print(f"📁 Installation:   {install_path}")
    print()

    # Git information
    git_info = get_git_info()
    if git_info.get("available", True):
        print("🔧 Git Information:")
        print(f"   Branch:         {git_info.get('branch', 'unknown')}")
        print(f"   Commit:         {git_info.get('commit', 'unknown')}")
        status = "✅ clean" if git_info.get("clean", False) else "⚠️  modified"
        print(f"   Status:         {status}")
        print()

    # Dependencies
    print("📚 Dependencies:")
    deps = check_dependencies()
    required_ok = []
    required_missing = []

    for dep, installed in deps.items():
        if installed:
            required_ok.append(dep)
        else:
            required_missing.append(dep)

    print(f"   ✅ Installed ({len(required_ok)}/{len(deps)}): {', '.join(required_ok)}")
    if required_missing:
        print(f"   ❌ Missing ({len(required_missing)}): {', '.join(required_missing)}")
    print()

    # Project information
    print("🔗 Project:")
    print(f"   Repository:     {REPOSITORY}")
    print(f"   Author:         {__author__}")
    print(f"   License:        {__license__}")
    print()

    # Quick start
    print("🚀 Quick Start:")
    print("   make health-check      # Validate installation (9 checks)")
    print("   python quick_start.py  # 30-second demo")
    print("   make run-examples      # Run all examples")
    print("   make help              # See all 60+ commands")
    print()

    print("=" * 70)
    print("Welcome to the future of consciousness research! 🌊")
    print("=" * 70)


def show_help():
    """Display help information."""
    print(f"🌊 {PROJECT_NAME} v{__version__}")
    print()
    print("Usage:")
    print("  python -m kosmic_lab --version    Show version number")
    print("  python -m kosmic_lab --info       Show comprehensive system information")
    print("  python -m kosmic_lab --help       Show this help message")
    print()
    print("Quick Start:")
    print("  make health-check                 Validate installation")
    print("  python quick_start.py             30-second demo")
    print("  make run-examples                 Run all examples")
    print()
    print(f"Documentation: {REPOSITORY}")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description=f"{PROJECT_NAME} - {PROJECT_TAGLINE}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--version", "-v", action="store_true", help="Show version number"
    )
    parser.add_argument(
        "--info", "-i", action="store_true", help="Show comprehensive system information"
    )

    # If no arguments provided, show help
    if len(sys.argv) == 1:
        show_help()
        return 0

    args = parser.parse_args()

    if args.version:
        show_version()
    elif args.info:
        show_info()
    else:
        show_help()

    return 0


if __name__ == "__main__":
    sys.exit(main())
