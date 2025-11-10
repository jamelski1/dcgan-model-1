import torch
import torch.nn as nn
from .dcgan import Discriminator

class DCGANDiscEncoder(nn.Module):
    def __init__(self, feat_dim=256, ndf=64):
        super().__init__()
        self.disc = Discriminator(ndf=ndf)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(ndf*4, feat_dim)
    def forward(self, x):
        h = self.disc.features(x)
        v = self.pool(h).view(x.size(0), -1)
        return self.proj(v)
