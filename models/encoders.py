"""
Encoder modules for image-to-text captioning.

This module provides various encoder architectures:
- EncoderCNN: Simple baseline CNN encoder
- DiscriminatorEncoder: Unified wrapper for DCGAN discriminators
- ResNet18Encoder: ResNet18-based encoder for spatial features
- HybridEncoder: Combines frozen DCGAN with trainable ResNet18
"""

import torch
import torch.nn as nn
from typing import Optional
import torchvision.models as models


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


class ResNet18Encoder(nn.Module):
    """
    ResNet18-based encoder that outputs spatial features.

    Extracts features at 4x4 spatial resolution with 512 channels,
    suitable for attention-based decoders.

    Args:
        out_channels: Number of output channels (default: 512)
        pretrained: If True, use ImageNet pretrained weights (default: True)
    """

    def __init__(self, out_channels: int = 512, pretrained: bool = True):
        super().__init__()
        self.out_channels = out_channels

        # Load pretrained ResNet18
        resnet = models.resnet18(pretrained=pretrained)

        # Extract layers up to layer4 (before avgpool and fc)
        # ResNet18 layers: conv1 -> bn1 -> relu -> maxpool -> layer1 -> layer2 -> layer3 -> layer4
        # Input: (B, 3, 32, 32) -> After conv1/maxpool: (B, 64, 8, 8)
        # After layer1: (B, 64, 8, 8), layer2: (B, 128, 4, 4), layer3: (B, 256, 2, 2), layer4: (B, 512, 1, 1)

        # For CIFAR-100 (32x32), we need to modify the initial layers
        # to avoid too much downsampling
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        # Skip maxpool to preserve spatial dimensions

        self.layer1 = resnet.layer1  # 64 channels, no stride
        self.layer2 = resnet.layer2  # 128 channels, stride=2 -> 16x16
        self.layer3 = resnet.layer3  # 256 channels, stride=2 -> 8x8

        # Modified layer4: stride=2 -> 4x4 with 512 channels
        self.layer4 = resnet.layer4  # 512 channels, stride=2 -> 4x4

        # Initialize conv1 with pretrained weights if available
        if pretrained:
            with torch.no_grad():
                # Copy center region of pretrained 7x7 conv to 3x3
                self.conv1.weight[:, :, :, :] = resnet.conv1.weight[:, :, 2:5, 2:5]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images of shape (B, 3, 32, 32)
        Returns:
            Spatial features of shape (B, 512, 4, 4)
        """
        # Input: (B, 3, 32, 32)
        x = self.conv1(x)      # (B, 64, 32, 32)
        x = self.bn1(x)
        x = self.relu(x)
        # Skip maxpool

        x = self.layer1(x)     # (B, 64, 32, 32)
        x = self.layer2(x)     # (B, 128, 16, 16)
        x = self.layer3(x)     # (B, 256, 8, 8)
        x = self.layer4(x)     # (B, 512, 4, 4)

        return x


class HybridEncoder(nn.Module):
    """
    Hybrid encoder combining frozen DCGAN features with trainable ResNet18.

    This encoder concatenates features from two sources:
    1. Frozen DCGAN discriminator: (B, 256, 4, 4)
    2. Trainable ResNet18: (B, 512, 4, 4)
    Result: (B, 768, 4, 4) concatenated features

    Args:
        dcgan_encoder: DCGAN discriminator encoder (will be frozen)
        resnet_encoder: ResNet18 encoder (trainable)
        dcgan_channels: Number of DCGAN feature channels (default: 256)
        resnet_channels: Number of ResNet feature channels (default: 512)
    """

    def __init__(
        self,
        dcgan_encoder: nn.Module,
        resnet_encoder: ResNet18Encoder,
        dcgan_channels: int = 256,
        resnet_channels: int = 512
    ):
        super().__init__()
        self.dcgan_encoder = dcgan_encoder
        self.resnet_encoder = resnet_encoder
        self.dcgan_channels = dcgan_channels
        self.resnet_channels = resnet_channels
        self.total_channels = dcgan_channels + resnet_channels

        # Freeze DCGAN encoder
        for param in self.dcgan_encoder.parameters():
            param.requires_grad = False

        print(f"HybridEncoder: DCGAN ({dcgan_channels}ch) + ResNet18 ({resnet_channels}ch) = {self.total_channels} channels")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images of shape (B, 3, 32, 32)
        Returns:
            Concatenated features of shape (B, 768, 4, 4)
        """
        # Extract DCGAN features (frozen)
        with torch.no_grad():
            dcgan_feats = self.dcgan_encoder(x)  # (B, 256, 4, 4)

        # Extract ResNet18 features (trainable)
        resnet_feats = self.resnet_encoder(x)   # (B, 512, 4, 4)

        # Concatenate along channel dimension
        combined = torch.cat([dcgan_feats, resnet_feats], dim=1)  # (B, 768, 4, 4)

        return combined
