#!/usr/bin/env python3
"""
Evaluation script for image captioning models.

Computes BLEU-1, ROUGE-L, and label-word accuracy metrics.

Usage:
    python scripts/eval.py --ckpt runs/best.pt --batch_size 256
"""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data import make_splits, collate_pad, decode_ids
from models.encoders import EncoderCNN, DiscriminatorEncoder, ResNet18Encoder, HybridEncoder
from models.decoders import DecoderGRU
from models.gan import Discriminator
from models.attention import SpatialEncoder, AttentionDecoderGRU


def bleu1(ref_tokens: list, hyp_tokens: list) -> float:
    """
    Compute BLEU-1 score (unigram precision).

    Args:
        ref_tokens: Reference tokens
        hyp_tokens: Hypothesis tokens

    Returns:
        BLEU-1 score
    """
    if not hyp_tokens:
        return 0.0
    ref = set(ref_tokens)
    hit = sum(1 for t in hyp_tokens if t in ref)
    return hit / max(1, len(hyp_tokens))


def lcs(a: list, b: list) -> int:
    """
    Compute longest common subsequence length.

    Args:
        a: First sequence
        b: Second sequence

    Returns:
        LCS length
    """
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            if a[i] == b[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
    return dp[n][m]


def rouge_l(ref_tokens: list, hyp_tokens: list) -> float:
    """
    Compute ROUGE-L score (F1-measure based on LCS).

    Args:
        ref_tokens: Reference tokens
        hyp_tokens: Hypothesis tokens

    Returns:
        ROUGE-L score
    """
    if not ref_tokens or not hyp_tokens:
        return 0.0
    L = lcs(ref_tokens, hyp_tokens)
    prec = L / len(hyp_tokens)
    rec = L / len(ref_tokens)
    if prec + rec == 0:
        return 0.0
    return (2 * prec * rec) / (prec + rec)


def load_model(ckpt_path: str, device: str, encoder_type_override: str = None, dcgan_ckpt: str = None):
    """
    Load model from checkpoint.

    Args:
        ckpt_path: Path to checkpoint file
        device: Device to load model on
        encoder_type_override: Override encoder type from checkpoint
        dcgan_ckpt: Path to DCGAN discriminator checkpoint (for hybrid encoder)

    Returns:
        Tuple of (encoder, decoder, vocab, args_dict)
    """
    print(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    stoi, itos = ckpt["vocab"]
    vocab_size = len(itos)
    args_dict = ckpt.get("args", {})

    # Extract hyperparameters with defaults
    feat_dim = args_dict.get("feat_dim", 256)
    ndf = args_dict.get("ndf", 64)
    hid = args_dict.get("hid", 256)
    emb = args_dict.get("emb", 128)
    encoder_type = encoder_type_override or args_dict.get("encoder_type", "cnn")
    use_attention = args_dict.get("use_attention", False)
    num_heads = args_dict.get("num_heads", 8)
    dropout = args_dict.get("dropout", 0.1)

    # Create encoder based on type and checkpoint keys
    # First check what keys exist in checkpoint
    has_enc = "enc" in ckpt
    has_enc_disc = "enc_disc" in ckpt

    if has_enc_disc or encoder_type in ("dcgan", "dcgan_sn"):
        # DCGAN encoder (with or without spectral norm)
        use_spectral_norm = (encoder_type == "dcgan_sn")
        discriminator = Discriminator(ndf=ndf, use_spectral_norm=use_spectral_norm)

        if use_attention:
            # Spatial encoder for attention mechanism
            encoder = SpatialEncoder(
                encoder=DiscriminatorEncoder(
                    discriminator=discriminator,
                    feat_dim=feat_dim,
                    ndf=ndf
                ),
                feat_dim=feat_dim,
                ndf=ndf
            ).to(device)
        else:
            # Standard global pooling encoder
            encoder = DiscriminatorEncoder(
                discriminator=discriminator,
                feat_dim=feat_dim,
                ndf=ndf
            ).to(device)

        enc_state = ckpt.get("enc_disc", ckpt.get("enc"))
        if enc_state is None:
            raise KeyError("Checkpoint missing both 'enc' and 'enc_disc' keys")
        # Use strict=False for old checkpoints (enc_disc) to handle bias parameter differences
        strict = not has_enc_disc
        encoder.load_state_dict(enc_state, strict=strict)
        if not strict:
            print("Note: Loaded with strict=False to handle old checkpoint format")
        encoder_desc = f"DCGAN {'spatial ' if use_attention else ''}encoder ({'SN' if use_spectral_norm else 'BN'})"
        print(f"Loaded {encoder_desc}")
    elif has_enc or encoder_type == "cnn":
        # Baseline CNN encoder
        if use_attention:
            raise ValueError("Attention mechanism not supported with CNN encoder")
        encoder = EncoderCNN(feat_dim=feat_dim).to(device)
        encoder.load_state_dict(ckpt["enc"])
        print("Loaded baseline CNN encoder")
    elif encoder_type == "hybrid":
        # Hybrid encoder: Frozen DCGAN + ResNet18
        if not use_attention:
            raise ValueError("Hybrid encoder requires attention mechanism")
        if not dcgan_ckpt:
            raise ValueError("Hybrid encoder requires --dcgan_ckpt for frozen DCGAN features")

        # Load frozen DCGAN discriminator
        print(f"Loading frozen DCGAN discriminator from {dcgan_ckpt}")
        discriminator = Discriminator(ndf=ndf, use_spectral_norm=True)
        dcgan_state = torch.load(dcgan_ckpt, map_location=device)
        discriminator.load_state_dict(dcgan_state, strict=False)

        # Create DCGAN spatial encoder (frozen)
        dcgan_spatial = SpatialEncoder(
            encoder=DiscriminatorEncoder(
                discriminator=discriminator,
                feat_dim=feat_dim,
                ndf=ndf,
                freeze=True
            ),
            feat_dim=feat_dim,
            ndf=ndf
        )

        # Create ResNet18 encoder
        resnet_encoder = ResNet18Encoder(out_channels=512, pretrained=False)  # Already trained

        # Combine into hybrid encoder
        encoder = HybridEncoder(
            dcgan_encoder=dcgan_spatial,
            resnet_encoder=resnet_encoder,
            dcgan_channels=feat_dim,
            resnet_channels=512
        ).to(device)

        # Load encoder state (this loads the ResNet18 weights)
        enc_state = ckpt.get("enc")
        if enc_state is None:
            raise KeyError("Checkpoint missing 'enc' key")
        encoder.load_state_dict(enc_state, strict=False)
        print(f"Loaded HybridEncoder (DCGAN {feat_dim}ch + ResNet18 512ch = 768ch)")
    else:
        raise ValueError("Cannot determine encoder type from checkpoint")

    # Determine spatial feature dimension for decoder
    spatial_feat_dim = 768 if encoder_type == "hybrid" else feat_dim

    # Create decoder based on attention flag
    if use_attention:
        decoder = AttentionDecoderGRU(
            spatial_feat_dim=spatial_feat_dim,
            vocab_size=vocab_size,
            hid=hid,
            emb=emb,
            num_heads=num_heads,
            dropout=dropout
        ).to(device)
        print(f"Loaded AttentionDecoderGRU with {num_heads} heads")
    else:
        decoder = DecoderGRU(
            feat_dim=feat_dim,
            vocab_size=vocab_size,
            hid=hid,
            emb=emb
        ).to(device)
        print("Loaded standard DecoderGRU")
    decoder.load_state_dict(ckpt["dec"])

    encoder.eval()
    decoder.eval()

    return encoder, decoder, (stoi, itos), args_dict


def main():
    parser = argparse.ArgumentParser(description="Evaluate captioning model")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to checkpoint file")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for evaluation")
    parser.add_argument("--max_len", type=int, default=8,
                        help="Maximum caption length")
    parser.add_argument("--data_root", type=str, default="./data",
                        help="Root directory for CIFAR-100 data")
    parser.add_argument("--encoder_type", type=str, default=None,
                        help="Override encoder type (cnn, dcgan, dcgan_sn, hybrid)")
    parser.add_argument("--dcgan_ckpt", type=str, default=None,
                        help="Path to DCGAN discriminator checkpoint (required for hybrid encoder)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    encoder, decoder, vocab, args_dict = load_model(args.ckpt, device, args.encoder_type, args.dcgan_ckpt)
    stoi, itos = vocab

    # Load test set
    print("Loading test dataset...")
    _, _, test_set, _ = make_splits(max_len=args.max_len, data_root=args.data_root)
    test_set.dataset.stoi, test_set.dataset.itos = stoi, itos

    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=2, collate_fn=collate_pad
    )

    # Evaluation metrics
    bleu1_sum = 0.0
    rouge_l_sum = 0.0
    label_correct = 0
    total = 0

    print("Evaluating...")
    with torch.no_grad():
        for images, targets, labels in tqdm(test_loader, desc="Eval"):
            images, targets = images.to(device), targets.to(device)

            # Generate captions
            features = encoder(images)
            generated_ids = decoder.generate(
                features,
                stoi["<bos>"],
                stoi["<eos>"],
                max_len=args.max_len
            )

            # Compute metrics for each sample
            for i in range(images.size(0)):
                ref = decode_ids(targets[i].tolist(), itos).split()
                hyp = decode_ids(generated_ids[i].tolist(), itos).split()

                # BLEU-1 and ROUGE-L
                bleu1_sum += bleu1(ref, hyp)
                rouge_l_sum += rouge_l(ref, hyp)

                # Label-word accuracy: check if true fine label is in hypothesis
                true_label = test_set.dataset.fine_names[labels[i].item()]
                label_correct += (true_label in hyp)
                total += 1

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"BLEU-1:              {bleu1_sum / total:.4f}")
    print(f"ROUGE-L:             {rouge_l_sum / total:.4f}")
    print(f"Label-word accuracy: {label_correct / total:.4f} ({label_correct}/{total})")
    print("=" * 60)


if __name__ == "__main__":
    main()
