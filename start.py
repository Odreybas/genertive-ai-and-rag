#!/usr/bin/env python3
"""
RAG Dashboard Launcher
Easy Python script to start the dashboard
"""

import subprocess
import sys
import os

def main():
    print("🚀 Starting RAG Dashboard...")
    print("📊 Opening at: http://localhost:7861")
    print()

    # Check if we're in the right directory
    if not os.path.exists("dashboard.py"):
        print("❌ Error: dashboard.py not found. Please run from project root.")
        sys.exit(1)

    # Check and install dependencies
    try:
        import streamlit
        print("✅ Dependencies already installed")
    except ImportError:
        print("📦 Installing dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"], check=True)

    # Start dashboard
    print("🌐 Launching dashboard...")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "dashboard.py",
        "--server.port=7861", "--server.headless=true"
    ])

if __name__ == "__main__":
    main()