"""
Data augmentation utilities for GAN training.

Includes DiffAugment and other augmentation strategies.
"""

import torch
import torch.nn.functional as F


def diff_augment(
    x: torch.Tensor,
    noise_std: float = 0.02,
    translation_range: int = 2
) -> torch.Tensor:
    """
    Apply DiffAugment: Gaussian noise + random integer translations.

    DiffAugment regularizes the discriminator without requiring additional
    normalization layers. It's especially useful with Spectral Normalization.

    Args:
        x: Input images of shape (B, C, H, W) in range [-1, 1]
        noise_std: Standard deviation for Gaussian noise
        translation_range: Maximum pixel shift in each direction

    Returns:
        Augmented images of shape (B, C, H, W) clamped to [-1, 1]
    """
    # Additive Gaussian noise
    x = x + noise_std * torch.randn_like(x)

    # Integer translation by up to translation_range pixels
    B, C, H, W = x.shape
    device = x.device
    tx = torch.randint(-translation_range, translation_range + 1, (B,), device=device)
    ty = torch.randint(-translation_range, translation_range + 1, (B,), device=device)

    # Build a sampling grid for translation
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij"
    )
    grid = torch.stack([
        (grid_x.unsqueeze(0) + tx.view(-1, 1, 1)).float(),
        (grid_y.unsqueeze(0) + ty.view(-1, 1, 1)).float(),
    ], dim=-1)

    # Normalize grid to [-1, 1]
    grid[..., 0] = (grid[..., 0] / (W - 1)) * 2 - 1
    grid[..., 1] = (grid[..., 1] / (H - 1)) * 2 - 1

    # Apply grid sampling
    x = F.grid_sample(
        x, grid, mode="bilinear", padding_mode="reflection", align_corners=True
    )

    return x.clamp(-1, 1)


def get_augmentation_fn(augment_type: str = "none", **kwargs):
    """
    Factory function for augmentation strategies.

    Args:
        augment_type: Type of augmentation ("none", "diff")
        **kwargs: Additional arguments passed to augmentation function

    Returns:
        Augmentation function that takes a tensor and returns augmented tensor
    """
    if augment_type == "diff":
        return lambda x: diff_augment(x, **kwargs)
    elif augment_type == "none":
        return lambda x: x
    else:
        raise ValueError(f"Unknown augmentation type: {augment_type}")
