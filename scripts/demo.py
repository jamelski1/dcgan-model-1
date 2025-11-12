#!/usr/bin/env python3
"""
Demo script for generating captions from images.

Supports multiple modes:
- sample: Generate captions for random test images
- grid: Save image grid with captions
- interactive: Generate captions for specific indices

Usage:
    # Generate captions for 16 random images
    python scripts/demo.py --ckpt runs/best.pt --num 16

    # Save image grid
    python scripts/demo.py --ckpt runs/best.pt --num 16 --save_grid demo.png
"""

import os
import sys
import argparse
import random
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data import make_splits, collate_pad, decode_ids
from models.encoders import EncoderCNN, DiscriminatorEncoder
from models.decoders import DecoderGRU
from models.gan import Discriminator


def load_model(ckpt_path: str, device: str):
    """
    Load model from checkpoint.

    Args:
        ckpt_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Tuple of (encoder, decoder, vocab, args_dict)
    """
    print(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    stoi, itos = ckpt["vocab"]
    vocab_size = len(itos)
    args_dict = ckpt.get("args", {})

    # Extract hyperparameters with defaults
    feat_dim = args_dict.get("feat_dim", 256)
    ndf = args_dict.get("ndf", 64)
    hid = args_dict.get("hid", 256)
    emb = args_dict.get("emb", 128)
    encoder_type = args_dict.get("encoder_type", "cnn")

    # Create encoder based on type
    if encoder_type == "cnn" or "enc" in ckpt:
        encoder = EncoderCNN(feat_dim=feat_dim).to(device)
        encoder.load_state_dict(ckpt["enc"])
        print("Loaded baseline CNN encoder")
    elif "enc_disc" in ckpt or encoder_type in ("dcgan", "dcgan_sn"):
        use_spectral_norm = (encoder_type == "dcgan_sn")
        discriminator = Discriminator(ndf=ndf, use_spectral_norm=use_spectral_norm)
        encoder = DiscriminatorEncoder(
            discriminator=discriminator,
            feat_dim=feat_dim,
            ndf=ndf
        ).to(device)
        enc_state = ckpt.get("enc", ckpt.get("enc_disc"))
        encoder.load_state_dict(enc_state)
        print(f"Loaded DCGAN encoder ({'SN' if use_spectral_norm else 'BN'})")
    else:
        raise ValueError("Cannot determine encoder type from checkpoint")

    # Create decoder
    decoder = DecoderGRU(
        feat_dim=feat_dim,
        vocab_size=vocab_size,
        hid=hid,
        emb=emb
    ).to(device)
    decoder.load_state_dict(ckpt["dec"])

    encoder.eval()
    decoder.eval()

    return encoder, decoder, (stoi, itos), args_dict


def main():
    parser = argparse.ArgumentParser(description="Demo caption generation")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to checkpoint file")
    parser.add_argument("--num", type=int, default=16,
                        help="Number of images to sample")
    parser.add_argument("--max_len", type=int, default=8,
                        help="Maximum caption length")
    parser.add_argument("--save_grid", type=str, default=None,
                        help="Path to save image grid (optional)")
    parser.add_argument("--indices", type=int, nargs="+", default=None,
                        help="Specific test set indices to caption")
    parser.add_argument("--data_root", type=str, default="./data",
                        help="Root directory for CIFAR-100 data")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    encoder, decoder, vocab, args_dict = load_model(args.ckpt, device)
    stoi, itos = vocab

    # Load test set
    print("Loading test dataset...")
    _, _, test_set, _ = make_splits(max_len=args.max_len, data_root=args.data_root)
    test_set.dataset.stoi, test_set.dataset.itos = stoi, itos

    # Select images
    if args.indices:
        indices = args.indices
        print(f"Using specified indices: {indices}")
    else:
        indices = random.sample(range(len(test_set)), k=min(args.num, len(test_set)))
        print(f"Randomly selected {len(indices)} images")

    # Collect images and generate captions
    images = []
    for i in indices:
        x, tgt, y = test_set[i]
        images.append(x)

    images_batch = torch.stack(images, dim=0).to(device)

    print("Generating captions...")
    with torch.no_grad():
        features = encoder(images_batch)
        generated_ids = decoder.generate(
            features,
            stoi["<bos>"],
            stoi["<eos>"],
            max_len=args.max_len
        )

    # Print captions
    print("\nGenerated Captions:")
    print("-" * 60)
    for i, idx in enumerate(indices):
        caption = decode_ids(generated_ids[i].tolist(), itos)
        _, _, label = test_set[idx]
        true_label = test_set.dataset.fine_names[label]
        print(f"[{idx:04d}] (true: {true_label:20s}) → {caption}")

    # Save grid if requested
    if args.save_grid:
        images_vis = (images_batch * 0.5 + 0.5).clamp(0, 1).cpu()
        grid = make_grid(images_vis, nrow=int(len(images_vis) ** 0.5))
        save_image(grid, args.save_grid)
        print(f"\nSaved image grid to {args.save_grid}")


if __name__ == "__main__":
    main()
