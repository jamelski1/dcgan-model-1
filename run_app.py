#!/usr/bin/env python
"""
Launcher script for Jamel's BetaBox Describinator on Render.
This script ensures the app starts correctly regardless of working directory.
"""

import sys
import os
from pathlib import Path

# Add app directory to path
app_dir = Path(__file__).parent / "app"
sys.path.insert(0, str(app_dir))

# Change to app directory
os.chdir(app_dir)

# Import and run the app
from app import create_interface

if __name__ == "__main__":
    print("🎨 Starting Jamel's BetaBox Describinator...")
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"🐍 Python: {sys.version}")

    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
        show_error=True,
    )
