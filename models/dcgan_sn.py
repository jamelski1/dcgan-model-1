import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm as SN


class GeneratorSN(nn.Module):
    """
    A simple 32×32 generator used for the hinge‑loss GAN with spectral
    normalization. It mirrors the original DCGAN generator but omits batch
    normalisation on the output layer. Input is a latent vector of size
    ``nz`` and output is a 3×32×32 tensor with values in [-1, 1].
    """

    def __init__(self, nz: int = 128, ngf: int = 64, nc: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf), nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class DiscriminatorSN(nn.Module):
    """
    Spectral‑normalised discriminator for hinge‑loss GAN. It contains no
    batch normalisation layers and uses spectral normalisation on all
    convolutional layers. ``ndf`` controls the base channel width. The
    ``features`` method returns the spatial feature map before the final
    output layer, which is useful for downstream feature extraction.
    """

    def __init__(self, ndf: int = 64, nc: int = 3) -> None:
        super().__init__()
        # Convolutional trunk with spectral normalisation on each layer
        self.conv1 = SN(nn.Conv2d(nc, ndf, 4, 2, 1))
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = SN(nn.Conv2d(ndf, ndf * 2, 4, 2, 1))
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = SN(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1))
        self.act3 = nn.LeakyReLU(0.2, inplace=True)
        # Final conv outputs a single score per spatial location
        self.conv_out = SN(nn.Conv2d(ndf * 4, 1, 4, 1, 0))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the feature map after the third convolution."""
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.act3(self.conv3(x))
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        out = self.conv_out(h).view(x.size(0))
        return out
