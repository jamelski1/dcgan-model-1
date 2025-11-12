"""
Data utilities for CIFAR-100 captioning.

Provides vocabulary building, tokenization, and dataset classes.
"""

import re
import os
import random
import torch
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import datasets, transforms
from typing import Tuple, List, Dict, Optional
from PIL import Image


# Special tokens
SPECIAL = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}


def simple_tokenize(s: str) -> List[str]:
    """Tokenize string using simple regex (lowercase words)."""
    return re.findall(r"[a-z]+", s.lower())


def build_vocab(captions: List[str], min_freq: int = 1) -> Tuple[Dict[str, int], List[str]]:
    """
    Build vocabulary from captions.

    Args:
        captions: List of caption strings
        min_freq: Minimum frequency for word inclusion

    Returns:
        Tuple of (stoi, itos) dictionaries
    """
    from collections import Counter
    cnt = Counter()
    for c in captions:
        cnt.update(simple_tokenize(c))
    itos = list(SPECIAL.keys()) + sorted([
        w for w, f in cnt.items() if f >= min_freq and w not in SPECIAL
    ])
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi, itos


def encode_caption(text: str, stoi: Dict[str, int], max_len: int = 8) -> List[int]:
    """
    Encode caption text to token IDs.

    Args:
        text: Caption string
        stoi: String-to-index vocabulary
        max_len: Maximum sequence length (including special tokens)

    Returns:
        List of token IDs, padded to max_len
    """
    toks = ["<bos>"] + simple_tokenize(text)[:max_len - 2] + ["<eos>"]
    ids = [stoi.get(t, stoi["<unk>"]) for t in toks]
    if len(ids) < max_len:
        ids += [stoi["<pad>"]] * (max_len - len(ids))
    return ids


def decode_ids(ids: List[int], itos: List[str]) -> str:
    """
    Decode token IDs to caption text.

    Args:
        ids: List of token IDs
        itos: Index-to-string vocabulary

    Returns:
        Decoded caption string
    """
    words = []
    for i in ids:
        w = itos[i]
        if w in ("<pad>", "<bos>"):
            continue
        if w == "<eos>":
            break
        words.append(w)
    return " ".join(words)


class CIFAR100Captions(Dataset):
    """
    CIFAR-100 dataset with template-based captions.

    Generates captions of the form "a photo of a {fine_label}" for each image.

    Args:
        train: If True, use training split; else test split
        vocab: Optional (stoi, itos) tuple for consistent vocabulary
        max_len: Maximum caption length
        data_root: Root directory for CIFAR-100 data
    """

    def __init__(
        self,
        train: bool = True,
        vocab: Optional[Tuple[Dict[str, int], List[str]]] = None,
        max_len: int = 8,
        data_root: str = "./data"
    ):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        try:
            # Try without downloading (works if data already exists)
            self.ds = datasets.CIFAR100(
                data_root, train=train, download=False, transform=self.transform
            )
        except Exception:
            # Fall back to download (requires working SSL/certs)
            self.ds = datasets.CIFAR100(
                data_root, train=train, download=True, transform=self.transform
            )
        self.fine_names = self.ds.classes
        self.max_len = max_len

        # Generate captions from fine labels
        self.captions = [f"a photo of a {self.fine_names[y]}" for _, y in self.ds]
        if vocab is None:
            self.stoi, self.itos = build_vocab(self.captions, min_freq=1)
        else:
            self.stoi, self.itos = vocab

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Returns:
            Tuple of (image, target_ids, label)
        """
        x, y = self.ds[idx]
        cap = self.captions[idx]
        tgt = torch.tensor(encode_caption(cap, self.stoi, self.max_len), dtype=torch.long)
        return x, tgt, y


class CIFARWithSR(Dataset):
    """
    CIFAR-100 dataset with optional super-resolution augmentation.

    With probability p_sr, replaces standard images with super-resolved versions.
    Super-resolved files should be named {index:05d}.png in sr_root.

    Args:
        sr_root: Directory containing super-resolved images
        p_sr: Probability of using super-resolved image
        data_root: Root directory for CIFAR-100 data
    """

    def __init__(
        self,
        sr_root: Optional[str] = None,
        p_sr: float = 0.0,
        data_root: str = "./data"
    ):
        self.ds = datasets.CIFAR100(data_root, train=True, download=False)
        self.sr_root = sr_root
        self.p_sr = p_sr
        # Basic augmentation: random horizontal flip then normalize
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img, _ = self.ds[idx]
        # Optionally replace with SR image
        if self.sr_root and self.p_sr > 0 and random.random() < self.p_sr:
            path = os.path.join(self.sr_root, f"{idx:05d}.png")
            if os.path.exists(path):
                try:
                    img = Image.open(path).convert("RGB")
                except Exception:
                    pass
        return self.transform(img)


def make_splits(
    split: Tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 1337,
    max_len: int = 8,
    data_root: str = "./data"
) -> Tuple[Dataset, Dataset, Dataset, Tuple[Dict[str, int], List[str]]]:
    """
    Create train/val/test splits from CIFAR-100.

    Args:
        split: Tuple of (train_frac, val_frac, test_frac)
        seed: Random seed for reproducible splits
        max_len: Maximum caption length
        data_root: Root directory for CIFAR-100 data

    Returns:
        Tuple of (train_set, val_set, test_set, vocab)
    """
    full = CIFAR100Captions(train=True, vocab=None, max_len=max_len, data_root=data_root)
    n = len(full)
    n_tr, n_va = int(split[0] * n), int(split[1] * n)
    n_te = n - n_tr - n_va
    g = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(full, [n_tr, n_va, n_te], generator=g)
    # For consistent vocab, reuse train vocab for val/test
    vocab = (full.stoi, full.itos)
    val_set.dataset.stoi, val_set.dataset.itos = vocab
    test_set.dataset.stoi, test_set.dataset.itos = vocab
    return train_set, val_set, test_set, vocab


def collate_pad(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom collate function for DataLoader.

    Args:
        batch: List of (image, target_ids, label) tuples

    Returns:
        Tuple of batched (images, targets, labels)
    """
    xs, tgts, ys = zip(*batch)
    xs = torch.stack(xs, dim=0)
    tgts = torch.stack(tgts, dim=0)
    ys = torch.tensor(ys, dtype=torch.long)
    return xs, tgts, ys
