"""
Encoder modules for image-to-text captioning.

This module provides various encoder architectures:
- EncoderCNN: Simple baseline CNN encoder
- DiscriminatorEncoder: Unified wrapper for DCGAN discriminators
"""

import torch
import torch.nn as nn
from typing import Optional


class EncoderCNN(nn.Module):
    """Simple CNN encoder for baseline captioning."""

    def __init__(self, feat_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),   # 16x16
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(128, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True), # 8x8
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(256, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images of shape (B, 3, H, W)
        Returns:
            Features of shape (B, feat_dim)
        """
        h = self.net(x).view(x.size(0), -1)
        return self.fc(h)


class DiscriminatorEncoder(nn.Module):
    """
    Unified encoder wrapper for DCGAN discriminators.

    This wrapper extracts features from any discriminator that implements
    a `features()` method and projects them to a fixed-length feature vector.
    Works with both standard and spectral-normalized discriminators.

    Args:
        discriminator: A discriminator module with a features() method
        feat_dim: Output feature dimension
        ndf: Base channel width of discriminator (default: 64)
        freeze: If True, freeze discriminator weights (default: False)
    """

    def __init__(
        self,
        discriminator: nn.Module,
        feat_dim: int = 256,
        ndf: int = 64,
        freeze: bool = False
    ):
        super().__init__()
        self.disc = discriminator
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(ndf * 4, feat_dim)

        if freeze:
            for param in self.disc.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images of shape (B, 3, H, W)
        Returns:
            Features of shape (B, feat_dim)
        """
        h = self.disc.features(x)
        v = self.pool(h).view(x.size(0), -1)
        return self.proj(v)
