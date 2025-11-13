"""
Attention-based decoder with multi-head spatial attention.

Implements Transformer-style multi-head attention for image captioning,
allowing the model to attend to different spatial regions when generating
each word.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism for attending to spatial image features.

    Args:
        embed_dim: Dimension of input embeddings
        num_heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query: (B, T_q, embed_dim) - decoder states
            key: (B, T_k, embed_dim) - encoder features
            value: (B, T_k, embed_dim) - encoder features
            return_attention: If True, return attention weights

        Returns:
            output: (B, T_q, embed_dim)
            attention_weights: (B, num_heads, T_q, T_k) if return_attention else None
        """
        B, T_q, _ = query.shape
        T_k = key.shape[1]

        # Project and reshape to (B, num_heads, T, head_dim)
        Q = self.q_proj(query).view(B, T_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(B, T_k, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        # (B, num_heads, T_q, head_dim) @ (B, num_heads, head_dim, T_k) -> (B, num_heads, T_q, T_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        # (B, num_heads, T_q, T_k) @ (B, num_heads, T_k, head_dim) -> (B, num_heads, T_q, head_dim)
        attn_output = torch.matmul(attn_weights, V)

        # Reshape back to (B, T_q, embed_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T_q, self.embed_dim)
        output = self.out_proj(attn_output)

        if return_attention:
            return output, attn_weights
        return output, None


class SpatialEncoder(nn.Module):
    """
    Wrapper that extracts spatial features from encoder instead of global pooling.

    For DCGAN discriminator, extracts 4x4 spatial feature map (ndf*4 channels)
    before the final classification layer.

    Args:
        encoder: Base encoder (should have a .features() method)
        feat_dim: Feature dimension per spatial location
        ndf: Discriminator base channel width
    """

    def __init__(self, encoder: nn.Module, feat_dim: int = 256, ndf: int = 64):
        super().__init__()
        self.encoder = encoder
        # Project spatial features to desired dimension
        self.proj = nn.Conv2d(ndf * 4, feat_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images (B, 3, 32, 32)

        Returns:
            Spatial features (B, feat_dim, H, W) where H=W=4 for DCGAN
        """
        # Get spatial features from discriminator (before pooling)
        spatial_feats = self.encoder.disc.features(x)  # (B, ndf*4, 4, 4)
        # Project to desired dimension
        spatial_feats = self.proj(spatial_feats)  # (B, feat_dim, 4, 4)
        return spatial_feats


class AttentionDecoderGRU(nn.Module):
    """
    GRU decoder with multi-head spatial attention.

    At each timestep:
    1. Embed current token
    2. Attend to spatial image features using multi-head attention
    3. Combine attended features with GRU hidden state
    4. Predict next token

    Args:
        spatial_feat_dim: Dimension of spatial features from encoder
        vocab_size: Size of vocabulary
        hid: GRU hidden dimension
        emb: Token embedding dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
        num_layers: Number of GRU layers
    """

    def __init__(
        self,
        spatial_feat_dim: int,
        vocab_size: int,
        hid: int = 256,
        emb: int = 128,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_layers: int = 1
    ):
        super().__init__()
        self.spatial_feat_dim = spatial_feat_dim
        self.hid = hid
        self.num_heads = num_heads

        # Token embedding
        self.emb = nn.Embedding(vocab_size, emb)
        self.emb_dropout = nn.Dropout(dropout)

        # Initial state from spatial features (use mean pooling)
        self.init_h = nn.Linear(spatial_feat_dim, hid)

        # Multi-head attention
        self.attention = MultiHeadAttention(hid, num_heads, dropout)

        # GRU takes concatenation of embedding and attended context
        self.gru = nn.GRU(emb + hid, hid, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)

        # Output projection
        self.fc = nn.Linear(hid, vocab_size)

        # Layer norm for stability
        self.ln_feat = nn.LayerNorm(spatial_feat_dim)
        self.ln_attn = nn.LayerNorm(hid)

    def forward(
        self,
        spatial_feats: torch.Tensor,
        tgt_ids: torch.Tensor,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with teacher forcing.

        Args:
            spatial_feats: Spatial features (B, feat_dim, H, W)
            tgt_ids: Target token IDs (B, T)
            return_attention: If True, return tuple (logits, attention_weights)

        Returns:
            If return_attention=False: logits tensor (B, T-1, vocab_size)
            If return_attention=True: tuple of (logits, attention_weights)
        """
        B, C, H, W = spatial_feats.shape

        # Reshape spatial features to sequence: (B, H*W, C)
        spatial_seq = spatial_feats.view(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)
        spatial_seq = self.ln_feat(spatial_seq)

        # Initialize hidden state from mean-pooled spatial features
        h0 = torch.tanh(self.init_h(spatial_seq.mean(dim=1))).unsqueeze(0)  # (1, B, hid)

        # Embed target tokens (excluding last for teacher forcing)
        token_embs = self.emb(tgt_ids[:, :-1])  # (B, T-1, emb)
        token_embs = self.emb_dropout(token_embs)

        T = token_embs.size(1)

        # Process sequence through GRU with attention
        outputs = []
        h = h0
        all_attention_weights = [] if return_attention else None

        for t in range(T):
            # Get current hidden state for attention query
            # Run GRU for one step with previous attended features (or zeros for first step)
            if t == 0:
                # First step: no previous context
                gru_input = torch.cat([token_embs[:, t:t+1], torch.zeros(B, 1, self.hid, device=spatial_feats.device)], dim=-1)
            else:
                # Subsequent steps: use previous attended context
                gru_input = torch.cat([token_embs[:, t:t+1], context], dim=-1)

            _, h = self.gru(gru_input, h)

            # Use current hidden state to attend to spatial features
            query = h[-1].unsqueeze(1)  # (B, 1, hid)
            context, attn = self.attention(query, spatial_seq, spatial_seq, return_attention=return_attention)
            context = self.ln_attn(context)  # (B, 1, hid)

            # Predict next token from current hidden state
            logit = self.fc(h[-1])  # (B, vocab_size)
            outputs.append(logit)

            if return_attention:
                all_attention_weights.append(attn)

        logits = torch.stack(outputs, dim=1)  # (B, T-1, vocab_size)

        if return_attention:
            attention_weights = torch.stack(all_attention_weights, dim=1)  # (B, T-1, num_heads, 1, H*W)
            # Reshape to (B, T-1, num_heads, H, W) for visualization
            attention_weights = attention_weights.squeeze(3).view(B, T, self.num_heads, H, W)
            return logits, attention_weights
        else:
            return logits

    @torch.no_grad()
    def generate(
        self,
        spatial_feats: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int = 8,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Greedy autoregressive generation with attention.

        Args:
            spatial_feats: Spatial features (B, feat_dim, H, W)
            bos_id: Beginning-of-sequence token ID
            eos_id: End-of-sequence token ID
            max_len: Maximum sequence length
            return_attention: If True, return tuple (generated_ids, attention_weights)

        Returns:
            If return_attention=False: generated_ids tensor (B, max_len-1)
            If return_attention=True: tuple of (generated_ids, attention_weights)
        """
        B, C, H, W = spatial_feats.shape
        device = spatial_feats.device

        # Reshape spatial features
        spatial_seq = spatial_feats.view(B, C, H * W).permute(0, 2, 1)
        spatial_seq = self.ln_feat(spatial_seq)

        # Initialize hidden state
        h = torch.tanh(self.init_h(spatial_seq.mean(dim=1))).unsqueeze(0)

        # Start with BOS token
        cur_token = torch.full((B,), bos_id, dtype=torch.long, device=device)

        outputs = []
        all_attention_weights = [] if return_attention else None
        context = torch.zeros(B, 1, self.hid, device=device)

        for _ in range(max_len - 1):
            # Embed current token
            token_emb = self.emb(cur_token).unsqueeze(1)  # (B, 1, emb)

            # GRU step
            gru_input = torch.cat([token_emb, context], dim=-1)
            _, h = self.gru(gru_input, h)

            # Attend to spatial features
            query = h[-1].unsqueeze(1)
            context, attn = self.attention(query, spatial_seq, spatial_seq, return_attention=return_attention)
            context = self.ln_attn(context)

            # Predict next token
            logits = self.fc(h[-1])
            next_token = logits.argmax(dim=-1)

            outputs.append(next_token)
            cur_token = next_token

            if return_attention:
                all_attention_weights.append(attn.squeeze(2))  # (B, num_heads, H*W)

        generated_ids = torch.stack(outputs, dim=1)  # (B, max_len-1)

        if return_attention:
            attention_weights = torch.stack(all_attention_weights, dim=1)  # (B, max_len-1, num_heads, H*W)
            # Reshape to (B, max_len-1, num_heads, H, W)
            attention_weights = attention_weights.view(B, max_len - 1, self.num_heads, H, W)
            return generated_ids, attention_weights
        else:
            return generated_ids
