#!/usr/bin/env python3
"""
Unified training script for image-to-text captioning.

Supports multiple encoder types:
- 'cnn': Simple baseline CNN encoder
- 'dcgan': DCGAN discriminator encoder (with BatchNorm)
- 'dcgan_sn': DCGAN discriminator encoder (with Spectral Normalization)

Usage:
    # Baseline CNN encoder
    python training/train_captioner.py --encoder_type cnn --epochs 8

    # DCGAN encoder (requires pre-trained discriminator)
    python training/train_captioner.py --encoder_type dcgan_sn \
        --dcgan_ckpt runs_gan_sn/disc_epoch_100.pt --epochs 20
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Literal

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data import make_splits, collate_pad
from models.encoders import EncoderCNN, DiscriminatorEncoder
from models.decoders import DecoderGRU
from models.gan import Discriminator


def create_encoder(
    encoder_type: str,
    feat_dim: int,
    ndf: int,
    dcgan_ckpt: str = None,
    freeze: bool = False,
    device: str = "cuda"
) -> nn.Module:
    """
    Factory function to create encoder based on type.

    Args:
        encoder_type: One of 'cnn', 'dcgan', 'dcgan_sn'
        feat_dim: Output feature dimension
        ndf: Base discriminator channel width (for DCGAN encoders)
        dcgan_ckpt: Path to pre-trained discriminator checkpoint
        freeze: If True, freeze encoder weights
        device: Device to load encoder on

    Returns:
        Encoder module
    """
    if encoder_type == "cnn":
        encoder = EncoderCNN(feat_dim=feat_dim)
        print("Created baseline CNN encoder")

    elif encoder_type in ("dcgan", "dcgan_sn"):
        use_spectral_norm = (encoder_type == "dcgan_sn")
        discriminator = Discriminator(ndf=ndf, use_spectral_norm=use_spectral_norm)

        # Load pre-trained weights if provided
        if dcgan_ckpt:
            print(f"Loading discriminator weights from {dcgan_ckpt}")
            state_dict = torch.load(dcgan_ckpt, map_location=device)
            discriminator.load_state_dict(state_dict, strict=False)
        else:
            print("WARNING: No DCGAN checkpoint provided. Using random initialization.")

        encoder = DiscriminatorEncoder(
            discriminator=discriminator,
            feat_dim=feat_dim,
            ndf=ndf,
            freeze=freeze
        )
        freeze_str = " (frozen)" if freeze else ""
        print(f"Created DCGAN encoder with {'Spectral Norm' if use_spectral_norm else 'BatchNorm'}{freeze_str}")

    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

    return encoder


def main():
    parser = argparse.ArgumentParser(description="Train image-to-text captioner")
    # Model architecture
    parser.add_argument("--encoder_type", type=str, default="cnn",
                        choices=["cnn", "dcgan", "dcgan_sn"],
                        help="Type of encoder to use")
    parser.add_argument("--feat_dim", type=int, default=256,
                        help="Encoder output feature dimension")
    parser.add_argument("--hid", type=int, default=256,
                        help="GRU hidden dimension")
    parser.add_argument("--emb", type=int, default=128,
                        help="Token embedding dimension")
    parser.add_argument("--ndf", type=int, default=64,
                        help="Discriminator base channel width (for DCGAN encoders)")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=8,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--max_len", type=int, default=8,
                        help="Maximum caption length")

    # DCGAN-specific options
    parser.add_argument("--dcgan_ckpt", type=str, default=None,
                        help="Path to pre-trained discriminator checkpoint")
    parser.add_argument("--freeze_encoder", action="store_true",
                        help="Freeze encoder weights during training")

    # Data and output
    parser.add_argument("--data_root", type=str, default="./data",
                        help="Root directory for CIFAR-100 data")
    parser.add_argument("--out", type=str, default="runs",
                        help="Output directory for checkpoints")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    # Load data
    print("Loading CIFAR-100 dataset...")
    train_set, val_set, test_set, vocab = make_splits(
        max_len=args.max_len,
        data_root=args.data_root
    )
    stoi, itos = vocab
    vocab_size = len(itos)
    print(f"Vocabulary size: {vocab_size}")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=2, collate_fn=collate_pad
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=2, collate_fn=collate_pad
    )

    # Create models
    encoder = create_encoder(
        encoder_type=args.encoder_type,
        feat_dim=args.feat_dim,
        ndf=args.ndf,
        dcgan_ckpt=args.dcgan_ckpt,
        freeze=args.freeze_encoder,
        device=device
    ).to(device)

    decoder = DecoderGRU(
        feat_dim=args.feat_dim,
        vocab_size=vocab_size,
        hid=args.hid,
        emb=args.emb
    ).to(device)

    # Optimizer and loss
    params = list(decoder.parameters())
    if not args.freeze_encoder:
        params += list(encoder.parameters())

    optimizer = torch.optim.AdamW(params, lr=args.lr)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=stoi["<pad>"])

    # Training loop
    best_val_loss = float("inf")
    print(f"\nStarting training for {args.epochs} epochs...")

    for epoch in range(1, args.epochs + 1):
        # Training phase
        encoder.train()
        decoder.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for images, targets, _ in pbar:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            features = encoder(images)
            logits = decoder(features, targets)  # (B, T-1, V)
            loss = criterion(logits.reshape(-1, vocab_size), targets[:, 1:].reshape(-1))
            loss.backward()

            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        train_loss /= len(train_set)

        # Validation phase
        encoder.eval()
        decoder.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, targets, _ in val_loader:
                images, targets = images.to(device), targets.to(device)
                features = encoder(images)
                logits = decoder(features, targets)
                loss = criterion(logits.reshape(-1, vocab_size), targets[:, 1:].reshape(-1))
                val_loss += loss.item() * images.size(0)

        val_loss /= len(val_set)
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "enc": encoder.state_dict(),  # Standardized key name
                "dec": decoder.state_dict(),
                "vocab": vocab,
                "args": vars(args)
            }
            ckpt_path = os.path.join(args.out, "best.pt")
            torch.save(checkpoint, ckpt_path)
            print(f"Saved best checkpoint to {ckpt_path}")

    print(f"\nTraining complete! Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
