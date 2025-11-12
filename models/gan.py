"""
Unified DCGAN implementation with configurable normalization.

This module provides Generator and Discriminator classes that can use
either BatchNorm or Spectral Normalization, reducing code duplication.
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from typing import Literal


class Generator(nn.Module):
    """
    DCGAN Generator for 32x32 images.

    Generates images from random latent vectors using transposed convolutions.
    Architecture is based on the original DCGAN paper.

    Args:
        z_dim: Latent vector dimension
        ngf: Base number of generator filters
        nc: Number of output channels (3 for RGB)
        use_spectral_norm: If True, apply spectral normalization (not typically used for generators)
    """

    def __init__(
        self,
        z_dim: int = 128,
        ngf: int = 64,
        nc: int = 3,
        use_spectral_norm: bool = False
    ):
        super().__init__()
        # Note: Spectral norm is rarely used for generators, but included for completeness
        norm_fn = spectral_norm if use_spectral_norm else lambda x: x

        self.net = nn.Sequential(
            # Input: (z_dim, 1, 1) -> (ngf*4, 4, 4)
            norm_fn(nn.ConvTranspose2d(z_dim, ngf * 4, 4, 1, 0, bias=False)),
            nn.BatchNorm2d(ngf * 4), nn.ReLU(True),
            # (ngf*4, 4, 4) -> (ngf*2, 8, 8)
            norm_fn(nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ngf * 2), nn.ReLU(True),
            # (ngf*2, 8, 8) -> (ngf, 16, 16)
            norm_fn(nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(ngf), nn.ReLU(True),
            # (ngf, 16, 16) -> (nc, 32, 32)
            norm_fn(nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False)),
            nn.Tanh()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent vectors of shape (B, z_dim, 1, 1)
        Returns:
            Generated images of shape (B, nc, 32, 32) in range [-1, 1]
        """
        return self.net(z)


class Discriminator(nn.Module):
    """
    DCGAN Discriminator with configurable normalization.

    Classifies images as real or fake. Can use either BatchNorm (standard DCGAN)
    or Spectral Normalization (for improved training stability with hinge loss).

    Args:
        ndf: Base number of discriminator filters
        nc: Number of input channels (3 for RGB)
        use_spectral_norm: If True, use spectral normalization instead of batch normalization
    """

    def __init__(
        self,
        ndf: int = 64,
        nc: int = 3,
        use_spectral_norm: bool = False
    ):
        super().__init__()
        self.use_spectral_norm = use_spectral_norm
        norm_fn = spectral_norm if use_spectral_norm else lambda x: x

        # First layer: no normalization
        self.conv1 = norm_fn(nn.Conv2d(nc, ndf, 4, 2, 1, bias=not use_spectral_norm))
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        # Second layer
        self.conv2 = norm_fn(nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=not use_spectral_norm))
        if not use_spectral_norm:
            self.bn2 = nn.BatchNorm2d(ndf * 2)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

        # Third layer
        self.conv3 = norm_fn(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=not use_spectral_norm))
        if not use_spectral_norm:
            self.bn3 = nn.BatchNorm2d(ndf * 4)
        self.act3 = nn.LeakyReLU(0.2, inplace=True)

        # Output layer
        self.conv_out = norm_fn(nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=not use_spectral_norm))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract intermediate features (useful for encoder).

        Args:
            x: Input images of shape (B, nc, 32, 32)
        Returns:
            Feature maps of shape (B, ndf*4, 4, 4)
        """
        x = self.act1(self.conv1(x))
        if self.use_spectral_norm:
            x = self.act2(self.conv2(x))
            x = self.act3(self.conv3(x))
        else:
            x = self.act2(self.bn2(self.conv2(x)))
            x = self.act3(self.bn3(self.conv3(x)))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images of shape (B, nc, 32, 32)
        Returns:
            Discriminator scores of shape (B,)
        """
        h = self.features(x)
        out = self.conv_out(h).view(x.size(0))
        return out


def weights_init(m: nn.Module) -> None:
    """
    Initialize network weights (DCGAN standard initialization).

    Applies to Conv, ConvTranspose, and BatchNorm layers.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
