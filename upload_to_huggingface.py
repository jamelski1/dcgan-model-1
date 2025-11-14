#!/usr/bin/env python3
"""
Upload BetaBox Describinator models to Hugging Face Hub.
Run this script to upload your trained models for Render deployment.

Usage:
    pip install huggingface-hub
    python upload_to_huggingface.py
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo, hf_hub_url

# Configuration
REPO_NAME = "jamels-betabox-describinator"
USERNAME = None  # Will be auto-detected from token

# Model files to upload
MODEL_FILES = {
    "runs_hybrid/best.pt": "best.pt",
    "runs_gan_sn/best_disc.pt": "best_disc.pt"
}

def main():
    print("=" * 70)
    print("🎨 BetaBox Describinator - Upload Models to Hugging Face")
    print("=" * 70)
    print()

    # Initialize API
    api = HfApi()

    # Get username
    try:
        user_info = api.whoami()
        username = user_info['name']
        print(f"✓ Logged in as: {username}")
    except Exception as e:
        print("✗ Not logged in to Hugging Face!")
        print()
        print("Please run: huggingface-cli login")
        print("Then get a token from: https://huggingface.co/settings/tokens")
        return

    print()

    # Create repo
    repo_id = f"{username}/{REPO_NAME}"
    print(f"📦 Creating repository: {repo_id}")
    try:
        create_repo(repo_id, repo_type="model", exist_ok=True)
        print(f"✓ Repository ready: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"✗ Failed to create repo: {e}")
        return

    print()

    # Upload files
    for local_path, remote_path in MODEL_FILES.items():
        if not Path(local_path).exists():
            print(f"⚠ Skipping {local_path} (not found)")
            continue

        file_size = os.path.getsize(local_path) / 1024 / 1024
        print(f"📤 Uploading {local_path} ({file_size:.1f} MB)...")

        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=remote_path,
                repo_id=repo_id,
                repo_type="model",
            )
            print(f"✓ Uploaded to: https://huggingface.co/{repo_id}/blob/main/{remote_path}")
        except Exception as e:
            print(f"✗ Upload failed: {e}")
            continue

    print()
    print("=" * 70)
    print("✓ Upload Complete!")
    print("=" * 70)
    print()
    print("🔗 Your model URLs for Render environment variables:")
    print()

    # Generate download URLs
    model_url = f"https://huggingface.co/{repo_id}/resolve/main/best.pt"
    dcgan_url = f"https://huggingface.co/{repo_id}/resolve/main/best_disc.pt"

    print(f"MODEL_URL={model_url}")
    print(f"DCGAN_URL={dcgan_url}")
    print()
    print("📋 Next Steps:")
    print("1. Copy the URLs above")
    print("2. Go to Render dashboard → Your service → Environment")
    print("3. Add MODEL_URL and DCGAN_URL environment variables")
    print("4. Save and Render will auto-redeploy")
    print()
    print(f"📖 View your models: https://huggingface.co/{repo_id}")
    print()


if __name__ == "__main__":
    main()
