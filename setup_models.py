#!/usr/bin/env python3
"""
Setup script for deploying Jamel's BetaBox Describinator to Render.

This script downloads the model checkpoints from cloud storage if they're not
already present. Add your model download URLs below.
"""

import os
import sys
from pathlib import Path
import urllib.request
import shutil

# Configuration
# URLs can be set via environment variables (MODEL_URL, DCGAN_URL) or hardcoded below
CHECKPOINTS = {
    "runs_hybrid/best.pt": {
        "url": os.environ.get("MODEL_URL"),  # Set MODEL_URL env var or hardcode here
        "description": "Hybrid encoder checkpoint (65.37% accuracy)"
    },
    "runs_gan_sn/best_disc.pt": {
        "url": os.environ.get("DCGAN_URL"),  # Set DCGAN_URL env var or hardcode here
        "description": "Frozen DCGAN discriminator"
    }
}


def download_file(url: str, destination: str, description: str):
    """Download a file from URL to destination."""
    print(f"📥 Downloading {description}...")
    print(f"   From: {url}")
    print(f"   To: {destination}")

    # Create parent directory
    Path(destination).parent.mkdir(parents=True, exist_ok=True)

    try:
        # Download with progress
        with urllib.request.urlopen(url) as response:
            total_size = int(response.headers.get('content-length', 0))
            with open(destination, 'wb') as f:
                shutil.copyfileobj(response, f)

        file_size = os.path.getsize(destination)
        print(f"   ✓ Downloaded {file_size / 1024 / 1024:.1f} MB")
        return True
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False


def setup_models():
    """Download all required model checkpoints."""
    print("=" * 60)
    print("🎨 Jamel's BetaBox Describinator - Model Setup")
    print("=" * 60)
    print()

    all_present = True
    all_downloaded = True

    for checkpoint_path, info in CHECKPOINTS.items():
        url = info["url"]
        description = info["description"]

        # Check if file exists
        if Path(checkpoint_path).exists():
            file_size = os.path.getsize(checkpoint_path)
            print(f"✓ {checkpoint_path} already exists ({file_size / 1024 / 1024:.1f} MB)")
            continue

        all_present = False

        # Download if URL is provided
        if url:
            success = download_file(url, checkpoint_path, description)
            if not success:
                all_downloaded = False
        else:
            print(f"⚠ {checkpoint_path} not found and no URL provided")
            print(f"   Please add the download URL to setup_models.py")
            all_downloaded = False

    print()
    print("=" * 60)

    if all_present:
        print("✓ All model checkpoints are present!")
        print("=" * 60)
        return True
    elif all_downloaded:
        print("✓ All model checkpoints downloaded successfully!")
        print("=" * 60)
        return True
    else:
        print("✗ Some model checkpoints are missing")
        print("=" * 60)
        print()
        print("Options to deploy with models:")
        print()
        print("1. Add download URLs to this script (setup_models.py)")
        print("   - Upload checkpoints to Google Drive, Dropbox, or similar")
        print("   - Get public download links")
        print("   - Add URLs to CHECKPOINTS dict above")
        print()
        print("2. Use Git LFS to commit checkpoints:")
        print("   git lfs install")
        print("   git lfs track '*.pt'")
        print("   git add runs_hybrid/best.pt runs_gan_sn/best_disc.pt")
        print("   git commit -m 'Add model checkpoints via Git LFS'")
        print()
        print("3. Manual upload to Render:")
        print("   - Deploy app without models (will run in demo mode)")
        print("   - Use Render shell to upload checkpoints")
        print()
        print("=" * 60)
        return False


if __name__ == "__main__":
    success = setup_models()
    sys.exit(0 if success else 1)
