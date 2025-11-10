import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=128, ngf=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, ngf*4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*4), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf), nn.ReLU(True),
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, ndf=64):
        super().__init__()
        self.conv1 = nn.Conv2d(3, ndf, 4, 2, 1, bias=False)
        self.act1  = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(ndf*2)
        self.act2  = nn.LeakyReLU(0.2, inplace=True)
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(ndf*4)
        self.act3  = nn.LeakyReLU(0.2, inplace=True)
        self.conv_out = nn.Conv2d(ndf*4, 1, 4, 1, 0, bias=False)
    def features(self, x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.act3(self.bn3(self.conv3(x)))
        return x
    def forward(self, x):
        h = self.features(x)
        out = self.conv_out(h).view(x.size(0))
        return out
