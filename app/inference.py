"""
Inference module for CIFAR-100 image captioning model.
Loads trained model and generates captions for uploaded images.
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.encoders import (
    Discriminator, DiscriminatorEncoder, SpatialEncoder,
    ResNet18Encoder, HybridEncoder
)
from models.attention import AttentionDecoderGRU
from models.decoders import DecoderGRU
import json


class CaptionGenerator:
    """Image caption generator using trained model."""

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        """
        Initialize the caption generator.

        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on ("cpu" or "cuda")
        """
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path

        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}...")
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load config
        config_path = Path(checkpoint_path).parent / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            # Fallback config if not available
            self.config = {
                "encoder_type": "hybrid",
                "use_attention": True,
                "feat_dim": 256,
                "ndf": 64,
                "vocab_size": 112,
                "max_len": 8
            }

        # Build vocabulary
        self._build_vocab()

        # Load model
        self._load_model()

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])

        print("Caption generator ready!")

    def _build_vocab(self):
        """Build vocabulary from checkpoint or create default vocab."""
        if "vocab" in self.checkpoint:
            vocab = self.checkpoint["vocab"]
            self.stoi = vocab.get("stoi", {})
            self.itos = vocab.get("itos", {})
        else:
            # Minimal default vocab (will be loaded from checkpoint in practice)
            self.stoi = {"<pad>": 0, "<bos>": 1, "<eos>": 2}
            self.itos = {0: "<pad>", 1: "<bos>", 2: "<eos>"}

        self.vocab_size = len(self.stoi)
        print(f"Vocabulary size: {self.vocab_size}")

    def _load_model(self):
        """Load encoder and decoder models."""
        encoder_type = self.config.get("encoder_type", "hybrid")
        use_attention = self.config.get("use_attention", True)
        feat_dim = self.config.get("feat_dim", 256)
        ndf = self.config.get("ndf", 64)

        # Load encoder
        if encoder_type == "hybrid":
            self._load_hybrid_encoder(feat_dim, ndf)
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")

        # Load decoder
        if use_attention:
            spatial_feat_dim = 768 if encoder_type == "hybrid" else feat_dim
            self.decoder = AttentionDecoderGRU(
                spatial_feat_dim=spatial_feat_dim,
                vocab_size=self.vocab_size,
                hid=256,
                emb=128,
                num_heads=8,
                dropout=0.1
            ).to(self.device)
        else:
            self.decoder = DecoderGRU(
                feat_dim=feat_dim,
                vocab_size=self.vocab_size,
                hid=256,
                emb=128,
                dropout=0.1
            ).to(self.device)

        # Load weights
        enc_state = self.checkpoint.get("enc", self.checkpoint.get("enc_disc"))
        self.encoder.load_state_dict(enc_state, strict=False)
        self.decoder.load_state_dict(self.checkpoint["dec"])

        # Set to eval mode
        self.encoder.eval()
        self.decoder.eval()

        print(f"Loaded {encoder_type} encoder with {'attention' if use_attention else 'standard'} decoder")

    def _load_hybrid_encoder(self, feat_dim: int, ndf: int):
        """Load hybrid encoder (DCGAN + ResNet18)."""
        dcgan_ckpt = self.config.get("dcgan_ckpt", "runs_gan_sn/best_disc.pt")

        # Load DCGAN discriminator
        discriminator = Discriminator(ndf=ndf, use_spectral_norm=True)
        if Path(dcgan_ckpt).exists():
            dcgan_state = torch.load(dcgan_ckpt, map_location=self.device)
            discriminator.load_state_dict(dcgan_state, strict=False)

        # Create frozen DCGAN spatial encoder
        dcgan_spatial = SpatialEncoder(
            encoder=DiscriminatorEncoder(discriminator, feat_dim, ndf, freeze=True),
            feat_dim=feat_dim,
            ndf=ndf
        )

        # Create trainable ResNet18 encoder
        resnet_encoder = ResNet18Encoder(out_channels=512, pretrained=False)

        # Combine into hybrid encoder
        self.encoder = HybridEncoder(
            dcgan_spatial, resnet_encoder, feat_dim, 512
        ).to(self.device)

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess image for model input.

        Args:
            image: PIL Image (any size, will be resized to 32x32)

        Returns:
            Preprocessed tensor (1, 3, 32, 32)
        """
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Apply transforms
        img_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
        return img_tensor.to(self.device)

    def decode_caption(self, token_ids: torch.Tensor) -> str:
        """
        Decode token IDs to caption string.

        Args:
            token_ids: Tensor of token IDs (seq_len,)

        Returns:
            Caption string
        """
        words = []
        for idx in token_ids.cpu().numpy():
            word = self.itos.get(int(idx), "<unk>")
            if word == "<eos>":
                break
            if word not in ["<bos>", "<pad>"]:
                words.append(word)
        return " ".join(words)

    @torch.no_grad()
    def generate_caption(self, image: Image.Image) -> str:
        """
        Generate caption for an image.

        Args:
            image: PIL Image

        Returns:
            Generated caption string
        """
        # Preprocess image
        img_tensor = self.preprocess_image(image)

        # Extract features
        features = self.encoder(img_tensor)

        # Generate caption
        bos_id = self.stoi.get("<bos>", 1)
        eos_id = self.stoi.get("<eos>", 2)
        max_len = self.config.get("max_len", 8)

        token_ids = self.decoder.generate(
            features, bos_id, eos_id, max_len
        )

        # Decode to text
        caption = self.decode_caption(token_ids[0])
        return caption


def load_caption_generator(checkpoint_path: str, device: str = "cpu") -> CaptionGenerator:
    """
    Convenience function to load caption generator.

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to run on ("cpu" or "cuda")

    Returns:
        CaptionGenerator instance
    """
    return CaptionGenerator(checkpoint_path, device)
