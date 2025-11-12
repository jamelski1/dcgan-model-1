#!/usr/bin/env python3
"""
Unified DCGAN training script with configurable normalization and loss.

Supports:
- BatchNorm + BCE loss (standard DCGAN)
- Spectral Normalization + Hinge loss (improved stability)
- DiffAugment regularization
- Optional super-resolution image mixing

Usage:
    # Standard DCGAN with BCE loss
    python training/train_gan.py --epochs 50 --batch_size 128

    # DCGAN with Spectral Norm + Hinge loss
    python training/train_gan.py --use_spectral_norm --use_hinge_loss \
        --epochs 150 --lr_d 2e-4 --lr_g 1e-4

    # With DiffAugment
    python training/train_gan.py --use_spectral_norm --use_hinge_loss \
        --use_diff_augment --epochs 150
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gan import Generator, Discriminator, weights_init
from utils.data import CIFARWithSR
from utils.augmentations import diff_augment


def hinge_loss_discriminator(real_scores: torch.Tensor, fake_scores: torch.Tensor) -> torch.Tensor:
    """
    Hinge loss for discriminator.

    Args:
        real_scores: Discriminator scores for real images
        fake_scores: Discriminator scores for fake images

    Returns:
        Hinge loss value
    """
    return torch.relu(1.0 - real_scores).mean() + torch.relu(1.0 + fake_scores).mean()


def hinge_loss_generator(fake_scores: torch.Tensor) -> torch.Tensor:
    """
    Hinge loss for generator.

    Args:
        fake_scores: Discriminator scores for fake images

    Returns:
        Hinge loss value
    """
    return -fake_scores.mean()


def main():
    parser = argparse.ArgumentParser(description="Train DCGAN on CIFAR-100")

    # Model architecture
    parser.add_argument("--z_dim", type=int, default=128,
                        help="Latent vector dimension")
    parser.add_argument("--ngf", type=int, default=64,
                        help="Generator base channel width")
    parser.add_argument("--ndf", type=int, default=64,
                        help="Discriminator base channel width")
    parser.add_argument("--use_spectral_norm", action="store_true",
                        help="Use Spectral Normalization in discriminator")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (overrides lr_d and lr_g)")
    parser.add_argument("--lr_d", type=float, default=2e-4,
                        help="Discriminator learning rate")
    parser.add_argument("--lr_g", type=float, default=1e-4,
                        help="Generator learning rate (TTUR)")
    parser.add_argument("--beta1", type=float, default=0.5,
                        help="Adam beta1 (set to 0.0 for spectral norm)")
    parser.add_argument("--beta2", type=float, default=0.999,
                        help="Adam beta2")

    # Loss function
    parser.add_argument("--use_hinge_loss", action="store_true",
                        help="Use hinge loss instead of BCE")
    parser.add_argument("--label_smoothing", type=float, default=0.9,
                        help="Label smoothing for real labels (BCE only)")

    # Augmentation
    parser.add_argument("--use_diff_augment", action="store_true",
                        help="Use DiffAugment regularization")
    parser.add_argument("--noise_std", type=float, default=0.02,
                        help="DiffAugment noise std")
    parser.add_argument("--translation_range", type=int, default=2,
                        help="DiffAugment translation range")

    # Super-resolution mixing
    parser.add_argument("--sr_root", type=str, default=None,
                        help="Path to super-resolved images")
    parser.add_argument("--p_sr", type=float, default=0.0,
                        help="Probability of using SR image")

    # Data and output
    parser.add_argument("--data_root", type=str, default="./data",
                        help="Root directory for CIFAR-100 data")
    parser.add_argument("--out", type=str, default="runs_gan",
                        help="Output directory for checkpoints")
    parser.add_argument("--save_interval", type=int, default=10,
                        help="Save checkpoint every N epochs")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    # Adjust hyperparameters based on mode
    if args.use_spectral_norm:
        # TTUR: Two Time-Scale Update Rule
        if args.lr is None:
            lr_d, lr_g = args.lr_d, args.lr_g
        else:
            lr_d = lr_g = args.lr
        beta1 = 0.0 if args.beta1 == 0.5 else args.beta1  # Default to 0.0 for SN
        print("Using Spectral Normalization mode with TTUR")
    else:
        lr_d = lr_g = args.lr if args.lr is not None else args.lr_d
        beta1 = args.beta1
        print("Using standard BatchNorm mode")

    # Dataset
    if args.sr_root or args.p_sr > 0:
        print(f"Using super-resolution mixing (p={args.p_sr})")
        dataset = CIFARWithSR(sr_root=args.sr_root, p_sr=args.p_sr, data_root=args.data_root)
    else:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = datasets.CIFAR100(args.data_root, train=True, download=False, transform=transform)

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=2, drop_last=True
    )

    # Create models
    generator = Generator(
        z_dim=args.z_dim,
        ngf=args.ngf,
        use_spectral_norm=False  # SN not typically used for generators
    ).to(device)

    discriminator = Discriminator(
        ndf=args.ndf,
        use_spectral_norm=args.use_spectral_norm
    ).to(device)

    # Initialize weights (not needed for spectral norm)
    if not args.use_spectral_norm:
        generator.apply(weights_init)
        discriminator.apply(weights_init)

    # Optimizers
    optimizer_g = optim.Adam(generator.parameters(), lr=lr_g, betas=(beta1, args.beta2))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(beta1, args.beta2))

    # Loss function
    if args.use_hinge_loss:
        print("Using Hinge Loss")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print(f"Using BCE Loss with label smoothing={args.label_smoothing}")

    # Fixed noise for visualization
    fixed_noise = torch.randn(64, args.z_dim, 1, 1, device=device)

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")

        for real_images in pbar:
            # Handle different dataset return types
            if isinstance(real_images, (tuple, list)):
                real_images = real_images[0]

            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            # ==================== Train Discriminator ====================
            optimizer_d.zero_grad()

            # Generate fake images
            z = torch.randn(batch_size, args.z_dim, 1, 1, device=device)
            with torch.no_grad():
                fake_images = generator(z)

            # Apply DiffAugment if enabled
            if args.use_diff_augment:
                real_aug = diff_augment(real_images, args.noise_std, args.translation_range)
                fake_aug = diff_augment(fake_images, args.noise_std, args.translation_range)
            else:
                real_aug = real_images
                fake_aug = fake_images

            # Get discriminator scores
            real_scores = discriminator(real_aug)
            fake_scores = discriminator(fake_aug)

            # Compute discriminator loss
            if args.use_hinge_loss:
                loss_d = hinge_loss_discriminator(real_scores, fake_scores)
            else:
                real_labels = torch.full((batch_size,), args.label_smoothing, device=device)
                fake_labels = torch.zeros(batch_size, device=device)
                loss_d_real = criterion(real_scores, real_labels)
                loss_d_fake = criterion(fake_scores, fake_labels)
                loss_d = loss_d_real + loss_d_fake

            loss_d.backward()
            optimizer_d.step()

            # ==================== Train Generator ====================
            optimizer_g.zero_grad()

            z = torch.randn(batch_size, args.z_dim, 1, 1, device=device)
            fake_images = generator(z)

            if args.use_diff_augment:
                fake_aug = diff_augment(fake_images, args.noise_std, args.translation_range)
            else:
                fake_aug = fake_images

            fake_scores = discriminator(fake_aug)

            # Compute generator loss
            if args.use_hinge_loss:
                loss_g = hinge_loss_generator(fake_scores)
            else:
                real_labels = torch.ones(batch_size, device=device)
                loss_g = criterion(fake_scores, real_labels)

            loss_g.backward()
            optimizer_g.step()

            pbar.set_postfix(loss_d=f"{loss_d.item():.3f}", loss_g=f"{loss_g.item():.3f}")

        # Save checkpoints and samples
        if epoch % args.save_interval == 0:
            with torch.no_grad():
                fake_samples = generator(fixed_noise).detach().cpu()

            # Save sample grid
            vutils.save_image(
                fake_samples * 0.5 + 0.5,  # Denormalize
                os.path.join(args.out, f"samples_epoch_{epoch:03d}.png"),
                nrow=8
            )

            # Save model checkpoints
            disc_path = os.path.join(args.out, f"disc_epoch_{epoch:03d}.pt")
            gen_path = os.path.join(args.out, f"gen_epoch_{epoch:03d}.pt")
            torch.save(discriminator.state_dict(), disc_path)
            torch.save(generator.state_dict(), gen_path)
            print(f"Saved checkpoints: {disc_path}, {gen_path}")

    print(f"\nTraining complete! Checkpoints saved to {args.out}")


if __name__ == "__main__":
    main()
