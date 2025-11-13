#!/usr/bin/env python3
"""
Unified training script for image-to-text captioning.

Supports multiple encoder types:
- 'cnn': Simple baseline CNN encoder
- 'dcgan': DCGAN discriminator encoder (with BatchNorm)
- 'dcgan_sn': DCGAN discriminator encoder (with Spectral Normalization)

Decoder options:
- Standard GRU decoder with global pooling (default)
- Multi-head spatial attention decoder (--use_attention)

Usage:
    # Baseline CNN encoder
    python training/train_captioner.py --encoder_type cnn --epochs 8

    # DCGAN encoder (requires pre-trained discriminator)
    python training/train_captioner.py --encoder_type dcgan_sn \
        --dcgan_ckpt runs_gan_sn/disc_epoch_100.pt --epochs 20

    # DCGAN encoder with multi-head spatial attention
    python training/train_captioner.py --encoder_type dcgan_sn \
        --dcgan_ckpt runs_gan_sn/best_disc.pt --use_attention \
        --num_heads 8 --epochs 40 --out runs_attention
"""

import os
import sys
import argparse
import json
import logging
from datetime import datetime
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
from models.attention import SpatialEncoder, AttentionDecoderGRU


def setup_logging(output_dir: str, use_tensorboard: bool = False):
    """
    Setup logging to file and optionally TensorBoard.

    Args:
        output_dir: Directory to save logs
        use_tensorboard: If True, also log to TensorBoard

    Returns:
        Tuple of (logger, tensorboard_writer or None)
    """
    logger = logging.getLogger("train_captioner")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    # File handler
    log_file = os.path.join(output_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # TensorBoard
    writer = None
    if use_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = os.path.join(output_dir, "tensorboard")
            writer = SummaryWriter(tb_dir)
            logger.info(f"TensorBoard logging enabled. Run: tensorboard --logdir {tb_dir}")
        except ImportError:
            logger.warning("TensorBoard not available. Install with: pip install tensorboard")

    logger.info(f"Logging to {log_file}")
    return logger, writer


def create_encoder(
    encoder_type: str,
    feat_dim: int,
    ndf: int,
    dcgan_ckpt: str = None,
    freeze: bool = False,
    use_attention: bool = False,
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
        use_attention: If True, return spatial features (for attention mechanism)
        device: Device to load encoder on

    Returns:
        Encoder module
    """
    if encoder_type == "cnn":
        if use_attention:
            raise ValueError("Attention mechanism not supported with CNN encoder. Use 'dcgan' or 'dcgan_sn'.")
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

        # Choose encoder type based on attention flag
        if use_attention:
            # SpatialEncoder for attention mechanism (returns 4x4 spatial features)
            encoder = SpatialEncoder(
                encoder=DiscriminatorEncoder(
                    discriminator=discriminator,
                    feat_dim=feat_dim,
                    ndf=ndf,
                    freeze=freeze
                ),
                feat_dim=feat_dim,
                ndf=ndf
            )
            freeze_str = " (frozen)" if freeze else ""
            print(f"Created DCGAN spatial encoder with {'Spectral Norm' if use_spectral_norm else 'BatchNorm'}{freeze_str}")
            print(f"  Output: (B, {feat_dim}, 4, 4) spatial features for multi-head attention")
        else:
            # Standard global pooling encoder
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

    # Attention mechanism
    parser.add_argument("--use_attention", action="store_true",
                        help="Use multi-head spatial attention decoder")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="Number of attention heads (only with --use_attention)")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout probability for attention (only with --use_attention)")

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
    parser.add_argument("--use_tensorboard", action="store_true",
                        help="Enable TensorBoard logging")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume training from checkpoint")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    # Setup logging
    logger, tb_writer = setup_logging(args.out, args.use_tensorboard)

    # Save hyperparameters
    config = vars(args).copy()
    config["device"] = device
    config["timestamp"] = datetime.now().isoformat()

    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved configuration to {os.path.join(args.out, 'config.json')}")

    # Load data
    logger.info("Loading CIFAR-100 dataset...")
    train_set, val_set, test_set, vocab = make_splits(
        max_len=args.max_len,
        data_root=args.data_root
    )
    stoi, itos = vocab
    vocab_size = len(itos)
    logger.info(f"Vocabulary size: {vocab_size}")

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
        use_attention=args.use_attention,
        device=device
    ).to(device)

    # Create decoder based on attention flag
    if args.use_attention:
        decoder = AttentionDecoderGRU(
            spatial_feat_dim=args.feat_dim,
            vocab_size=vocab_size,
            hid=args.hid,
            emb=args.emb,
            num_heads=args.num_heads,
            dropout=args.dropout
        ).to(device)
        logger.info(f"Created AttentionDecoderGRU with {args.num_heads} heads")
    else:
        decoder = DecoderGRU(
            feat_dim=args.feat_dim,
            vocab_size=vocab_size,
            hid=args.hid,
            emb=args.emb
        ).to(device)
        logger.info("Created standard DecoderGRU")

    # Optimizer and loss
    params = list(decoder.parameters())
    if not args.freeze_encoder:
        params += list(encoder.parameters())

    optimizer = torch.optim.AdamW(params, lr=args.lr)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=stoi["<pad>"])

    # Resume from checkpoint if specified
    start_epoch = 1
    best_val_loss = float("inf")

    if args.resume:
        logger.info(f"Resuming training from {args.resume}")
        resume_ckpt = torch.load(args.resume, map_location=device)

        # Load model states
        enc_state = resume_ckpt.get("enc", resume_ckpt.get("enc_disc"))
        encoder.load_state_dict(enc_state, strict=False)
        decoder.load_state_dict(resume_ckpt["dec"])

        # Load optimizer state if available
        if "optimizer" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer"])
            logger.info("Loaded optimizer state")

        # Get best validation loss and starting epoch
        if "best_val_loss" in resume_ckpt:
            best_val_loss = resume_ckpt["best_val_loss"]
            logger.info(f"Previous best validation loss: {best_val_loss:.4f}")

        if "epoch" in resume_ckpt:
            start_epoch = resume_ckpt["epoch"] + 1
            logger.info(f"Resuming from epoch {start_epoch}")

        logger.info("Successfully resumed from checkpoint")

    # Training loop
    logger.info(f"Training for {args.epochs} epochs (starting from epoch {start_epoch})...")

    for epoch in range(start_epoch, start_epoch + args.epochs):
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
        total_epochs = start_epoch + args.epochs - 1
        logger.info(f"Epoch {epoch}/{total_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # TensorBoard logging
        if tb_writer is not None:
            tb_writer.add_scalar("Loss/Train", train_loss, epoch)
            tb_writer.add_scalar("Loss/Validation", val_loss, epoch)

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "enc": encoder.state_dict(),  # Standardized key name
                "dec": decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "vocab": vocab,
                "args": vars(args)
            }
            ckpt_path = os.path.join(args.out, "best.pt")
            torch.save(checkpoint, ckpt_path)
            logger.info(f"Saved best checkpoint to {ckpt_path}")

    logger.info(f"Training complete! Best validation loss: {best_val_loss:.4f}")
    if tb_writer is not None:
        tb_writer.close()


if __name__ == "__main__":
    main()
