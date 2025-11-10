# visualize_dcgan_feature.py
import os
import torch
from torchvision import datasets, transforms, utils as vutils
from models.dcgan import Discriminator

CKPT = "runs_gan/disc_state.pt"
OUTDIR = "feature_maps"
os.makedirs(OUTDIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1) load discriminator
disc = Discriminator(ndf=64).to(device)
disc.load_state_dict(torch.load(CKPT, map_location=device))
disc.eval()

# 2) load a few CIFAR images
tfm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
ds = datasets.CIFAR100("./data", train=True, download=False, transform=tfm)

# pick a single image to visualize feature channels for
x = ds[0][0].unsqueeze(0).to(device)   # (1,3,32,32)

# 3) forward through conv blocks
with torch.no_grad():
    f1 = disc.act1(disc.conv1(x))            # (1, 64, 16, 16)
    f2 = disc.act2(disc.bn2(disc.conv2(f1))) # (1, 128, 8, 8)
    f3 = disc.act3(disc.bn3(disc.conv3(f2))) # (1, 256, 4, 4)

def save_feature_grid(feat, max_channels, nrow, filename):
    """
    feat: (1, C, H, W) → save first max_channels as grayscale tiles.
    """
    feat = feat[0].detach().cpu()                 # (C,H,W)
    C = min(feat.size(0), max_channels)
    maps = feat[:C].unsqueeze(1)                  # (C,1,H,W)
    # normalize each map for visibility
    maps = (maps - maps.min()) / (maps.max() - maps.min() + 1e-5)
    grid = vutils.make_grid(maps, nrow=nrow, padding=1)
    vutils.save_image(grid, os.path.join(OUTDIR, filename))
    print(f"Saved → {os.path.join(OUTDIR, filename)}")

# 4) save a few channels from each layer
save_feature_grid(f1, max_channels=32, nrow=8,  filename="layer1_16x16.png")
save_feature_grid(f2, max_channels=32, nrow=8,  filename="layer2_8x8.png")
save_feature_grid(f3, max_channels=16, nrow=8,  filename="layer3_4x4.png")
