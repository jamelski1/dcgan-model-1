#!/usr/bin/env python
"""
Launcher script for Jamel's BetaBox Describinator on Render.
This script ensures the app starts correctly regardless of working directory.
"""

import sys
import os
from pathlib import Path

# Get project root directory
project_root = Path(__file__).parent.absolute()

# Add project root to path for imports
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "app"))

# Ensure we're in the project root directory
os.chdir(project_root)

# Import and run the app
from app.app import create_interface

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
