import torch
import torch.nn as nn
from .dcgan_sn import DiscriminatorSN


class DCGANDiscSNEncoder(nn.Module):
    """
    Wraps the spectral‑normalised DCGAN discriminator as a feature
    extractor. The ``forward`` method returns a fixed‑length feature
    vector for each input image by spatially averaging the last
    convolutional block and projecting it to ``feat_dim``. Use this
    encoder in place of the baseline CNN for captioning tasks once you
    have trained the discriminator with ``train_dcgan_sn.py``.
    """

    def __init__(self, feat_dim: int = 256, ndf: int = 64) -> None:
        super().__init__()
        self.disc = DiscriminatorSN(ndf=ndf)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(ndf * 4, feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.disc.features(x)
        v = self.pool(h).view(x.size(0), -1)
        return self.proj(v)
