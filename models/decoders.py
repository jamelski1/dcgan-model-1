"""
Decoder modules for image-to-text captioning.

This module provides sequence decoders for generating captions from image features.
"""

import torch
import torch.nn as nn
from typing import Tuple


class DecoderGRU(nn.Module):
    """
    GRU-based sequence decoder for caption generation.

    The decoder uses the image features to initialize the GRU hidden state,
    then generates tokens autoregressively. During training, it uses teacher
    forcing; during inference, it generates greedily.

    Args:
        feat_dim: Input feature dimension from encoder
        vocab_size: Size of vocabulary
        hid: Hidden dimension for GRU
        emb: Embedding dimension for tokens
        num_layers: Number of GRU layers
    """

    def __init__(
        self,
        feat_dim: int,
        vocab_size: int,
        hid: int = 256,
        emb: int = 128,
        num_layers: int = 1
    ):
        super().__init__()
        self.img2hid = nn.Linear(feat_dim, hid)
        self.emb = nn.Embedding(vocab_size, emb)
        self.gru = nn.GRU(emb, hid, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hid, vocab_size)

    def forward(self, img_feat: torch.Tensor, tgt_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with teacher forcing.

        Args:
            img_feat: Image features of shape (B, feat_dim)
            tgt_ids: Target token IDs of shape (B, T)
        Returns:
            Logits of shape (B, T-1, vocab_size)
        """
        # Initialize hidden state from image features
        h0 = torch.tanh(self.img2hid(img_feat)).unsqueeze(0)  # (1, B, H)
        # Embed target tokens (excluding last token for teacher forcing)
        x = self.emb(tgt_ids[:, :-1])                         # (B, T-1, E)
        # Run GRU
        out, _ = self.gru(x, h0)
        # Project to vocabulary
        logits = self.fc(out)                                 # (B, T-1, V)
        return logits

    @torch.no_grad()
    def generate(
        self,
        img_feat: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int = 8
    ) -> torch.Tensor:
        """
        Greedy autoregressive generation.

        Args:
            img_feat: Image features of shape (B, feat_dim)
            bos_id: Beginning-of-sequence token ID
            eos_id: End-of-sequence token ID (currently not used for early stopping)
            max_len: Maximum sequence length to generate
        Returns:
            Generated token IDs of shape (B, max_len-1)
        """
        h = torch.tanh(self.img2hid(img_feat)).unsqueeze(0)   # (1, B, H)
        B = img_feat.size(0)
        cur = torch.full((B, 1), bos_id, dtype=torch.long, device=img_feat.device)
        outs = []
        for _ in range(max_len - 1):
            x = self.emb(cur[:, -1:])
            o, h = self.gru(x, h)
            logits = self.fc(o[:, -1])
            nxt = logits.argmax(-1)                           # (B,)
            outs.append(nxt)
            cur = torch.cat([cur, nxt.unsqueeze(1)], dim=1)
        return torch.stack(outs, dim=1)  # (B, max_len-1)
