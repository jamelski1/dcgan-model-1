"""
Utility modules for data handling and augmentation.

This package provides:
- Data: Dataset classes, vocabulary building, and data loaders
- Augmentations: DiffAugment and other augmentation strategies
"""

from .data import (
    CIFAR100Captions,
    CIFARWithSR,
    make_splits,
    collate_pad,
    build_vocab,
    encode_caption,
    decode_ids,
    simple_tokenize,
    SPECIAL,
)
from .augmentations import diff_augment, get_augmentation_fn

__all__ = [
    "CIFAR100Captions",
    "CIFARWithSR",
    "make_splits",
    "collate_pad",
    "build_vocab",
    "encode_caption",
    "decode_ids",
    "simple_tokenize",
    "SPECIAL",
    "diff_augment",
    "get_augmentation_fn",
]
