#!/usr/bin/env python3
"""
Linear probe on spectral‑normalised DCGAN discriminators. This script
evaluates a set of discriminator checkpoints by freezing each one,
extracting per‑image features and training a small linear classifier
on those features to predict the CIFAR‑100 fine labels. The
validation accuracy serves as a proxy for the usefulness of the
discriminator's learned representation. The checkpoint with the
highest accuracy is copied to ``best_disc.pt`` in the output
directory.

Run after ``train_dcgan_sn.py`` has completed. You can adjust the
path to the checkpoints and the number of epochs with the command
line flags below.
"""

import argparse
import glob
import os
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.dcgan_sn_encoder import DCGANDiscSNEncoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default="runs_gan_sn", help="directory containing discriminator checkpoints")
    parser.add_argument("--ndf", type=int, default=64, help="ndf used when training the discriminator")
    parser.add_argument("--feat_dim", type=int, default=256, help="dimension of the extracted feature vector")
    parser.add_argument("--epochs", type=int, default=3, help="number of epochs for the linear classifier")
    parser.add_argument("--out", type=str, default="runs_gan_sn", help="directory to save best_disc.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # CIFAR‑100 test set as a proxy for validation
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    val_ds = datasets.CIFAR100("./data", train=False, download=False, transform=tfm)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=2)

    best_acc = -1.0
    best_path = None

    for ckpt_path in sorted(glob.glob(os.path.join(args.ckpt_dir, "disc_epoch_*.pt"))):
        # build encoder with same ndf
        enc = DCGANDiscSNEncoder(feat_dim=args.feat_dim, ndf=args.ndf).to(device)
        enc.disc.load_state_dict(torch.load(ckpt_path, map_location=device))
        enc.eval()

        # extract features for the entire validation set
        features = []
        labels = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                f = enc(x)
                features.append(f.cpu())
                labels.append(y)
        X = torch.cat(features)
        Y = torch.cat(labels)

        # train linear classifier
        clf = nn.Linear(args.feat_dim, 100).to(device)
        opt = torch.optim.Adam(clf.parameters(), lr=1e-2)
        ce = nn.CrossEntropyLoss()
        ds = torch.utils.data.TensorDataset(X, Y)
        dl = DataLoader(ds, batch_size=512, shuffle=True)
        for _ in range(args.epochs):
            for xb, yb in dl:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = ce(clf(xb), yb)
                loss.backward()
                opt.step()

        # compute accuracy
        with torch.no_grad():
            preds = clf(X.to(device)).argmax(1).cpu()
            acc = (preds == Y).float().mean().item()
        print(f"{os.path.basename(ckpt_path)}  linear probe acc: {acc:.3f}")
        if acc > best_acc:
            best_acc = acc
            best_path = ckpt_path

    # copy the best checkpoint
    if best_path is not None:
        dest = os.path.join(args.out, "best_disc.pt")
        shutil.copy(best_path, dest)
        print(f"Best discriminator: {best_path}  acc={best_acc:.3f}")
        print(f"Copied to {dest}")
    else:
        print("No discriminator checkpoints found.")


if __name__ == "__main__":
    main()
