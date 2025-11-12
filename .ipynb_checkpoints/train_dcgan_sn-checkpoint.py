#!/usr/bin/env python3
"""
Train a DCGAN on CIFAR‑100 using hinge loss and spectral‑normalised
discriminator. This script introduces minimal changes relative to the
existing codebase: it defines a custom dataset wrapper with optional
super‑resolution augmentation, applies light DiffAugment and trains
with the hinge loss objective. It saves discriminator checkpoints
periodically for later feature extraction.

Usage:
    python train_dcgan_sn.py --epochs 160 --batch_size 128

You can adjust the number of epochs, learning rates and other
hyperparameters with the command line flags below. If you have a
directory of super‑resolved CIFAR images (see README for details),
specify it with ``--sr_root`` to mix them into the training set.
"""

import argparse
import os, sys
# Add the project root and the models directory to Python’s search path
BASE_DIR = os.path.dirname(__file__)
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'models'))
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, utils as vutils

from models.dcgan_sn import GeneratorSN, DiscriminatorSN


class CIFARWithSR(Dataset):
    """
    A thin wrapper around ``torchvision.datasets.CIFAR100`` that, with
    probability ``p_sr``, replaces the standard image with a
    super‑resolved version from ``sr_root``. The super‑resolved files
    should be named ``{index:05d}.png`` and stored in ``sr_root``. If
    ``p_sr=0`` or no matching SR file exists, the original image is
    used. The returned image is transformed with the provided
    transform (flip + normalisation).
    """

    def __init__(self, sr_root: str | None = None, p_sr: float = 0.0):
        self.ds = datasets.CIFAR100("./data", train=True, download=False)
        self.sr_root = sr_root
        self.p_sr = p_sr
        # Basic augmentation: random horizontal flip then normalise
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        img, _ = self.ds[idx]
        # optionally replace with SR image
        if self.sr_root and self.p_sr > 0 and random.random() < self.p_sr:
            from PIL import Image
            path = os.path.join(self.sr_root, f"{idx:05d}.png")
            if os.path.exists(path):
                try:
                    img = Image.open(path).convert("RGB")
                except Exception:
                    pass
        return self.transform(img)


def diff_augment(x: torch.Tensor) -> torch.Tensor:
    """
    Apply simple DiffAugment: add small Gaussian noise and integer
    translations. These augmentations regularise the discriminator
    without requiring additional normalisation layers.
    """
    # additive Gaussian noise
    x = x + 0.02 * torch.randn_like(x)
    # integer translation by up to 2 pixels
    shift = 2
    B, C, H, W = x.shape
    device = x.device
    tx = torch.randint(-shift, shift + 1, (B,), device=device)
    ty = torch.randint(-shift, shift + 1, (B,), device=device)
    # build a sampling grid for translation
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij"
    )
    grid = torch.stack([
        (grid_x.unsqueeze(0) + tx.view(-1, 1, 1)).float(),
        (grid_y.unsqueeze(0) + ty.view(-1, 1, 1)).float(),
    ], dim=-1)
    grid[..., 0] = (grid[..., 0] / (W - 1)) * 2 - 1
    grid[..., 1] = (grid[..., 1] / (H - 1)) * 2 - 1
    x = torch.nn.functional.grid_sample(
        x, grid, mode="bilinear", padding_mode="reflection", align_corners=True
    )
    return x.clamp(-1, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=150, help="number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="minibatch size")
    parser.add_argument("--nz", type=int, default=128, help="latent vector size")
    parser.add_argument("--ndf", type=int, default=64, help="discriminator base channel width")
    parser.add_argument("--ngf", type=int, default=64, help="generator base channel width")
    parser.add_argument("--lr_d", type=float, default=2e-4, help="learning rate for discriminator")
    parser.add_argument("--lr_g", type=float, default=1e-4, help="learning rate for generator")
    parser.add_argument("--sr_root", type=str, default=None, help="path to super‑resolved images")
    parser.add_argument("--p_sr", type=float, default=0.0, help="probability of using an SR image")
    parser.add_argument("--out", type=str, default="runs_gan_sn", help="output directory for checkpoints")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    # Dataset and loader
    train_ds = CIFARWithSR(args.sr_root, args.p_sr)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    # Model initialisation
    G = GeneratorSN(nz=args.nz, ngf=args.ngf).to(device)
    D = DiscriminatorSN(ndf=args.ndf).to(device)

    # Optimisers with TTUR and zero beta1 for stability
    optD = optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.0, 0.9))
    optG = optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.0, 0.9))

    fixed_noise = torch.randn(64, args.nz, 1, 1, device=device)

    # main training loop
    for epoch in range(1, args.epochs + 1):
        for i, x in enumerate(loader, 1):
            x = x.to(device)
            bsize = x.size(0)

            # ==================== Train Discriminator ====================
            # generate fake images
            z = torch.randn(bsize, args.nz, 1, 1, device=device)
            with torch.no_grad():
                fake = G(z)
            # apply DiffAugment to real and fake
            real_aug = diff_augment(x)
            fake_aug = diff_augment(fake)
            # discriminator scores
            real_scores = D(real_aug)
            fake_scores = D(fake_aug)
            # hinge loss for D
            lossD = torch.relu(1.0 - real_scores).mean() + torch.relu(1.0 + fake_scores).mean()
            optD.zero_grad()
            lossD.backward()
            optD.step()

            # ==================== Train Generator ========================
            z = torch.randn(bsize, args.nz, 1, 1, device=device)
            fake = G(z)
            fake_aug = diff_augment(fake)
            fake_scores = D(fake_aug)
            # hinge loss for G (maximize D(fake)) = minimize -D(fake)
            lossG = -fake_scores.mean()
            optG.zero_grad()
            lossG.backward()
            optG.step()

        # end of epoch: save checkpoint and sample images every 10 epochs
        if epoch % 10 == 0:
            with torch.no_grad():
                fake_imgs = G(fixed_noise).detach().cpu()
            # save samples grid
            vutils.save_image(
                fake_imgs * 0.5 + 0.5,
                os.path.join(args.out, f"samples_epoch_{epoch:03d}.png"),
                nrow=8,
            )
            # save discriminator weights
            torch.save(D.state_dict(), os.path.join(args.out, f"disc_epoch_{epoch:03d}.pt"))
            print(
                f"Epoch {epoch}/{args.epochs}  lossD={lossD.item():.3f}  lossG={lossG.item():.3f}  (saved checkpoint)"
            )


if __name__ == "__main__":
    main()
