import os, argparse
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from tqdm import tqdm
from models.dcgan import Generator, Discriminator

def weights_init(m):
    cname = m.__class__.__name__
    if 'Conv' in cname:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
    if 'BatchNorm' in cname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--z_dim', type=int, default=128)
    ap.add_argument('--ngf', type=int, default=64)
    ap.add_argument('--ndf', type=int, default=64)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--beta1', type=float, default=0.5)
    ap.add_argument('--beta2', type=float, default=0.999)
    ap.add_argument('--out', type=str, default='runs_gan')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out, exist_ok=True)

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    ds = datasets.CIFAR100('./data', train=True, download=False, transform=tfm)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)

    netG = Generator(z_dim=args.z_dim, ngf=args.ngf).to(device)
    netD = Discriminator(ndf=args.ndf).to(device)
    netG.apply(weights_init); netD.apply(weights_init)

    optG = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optD = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    crit = nn.BCEWithLogitsLoss()

    fixed_z = torch.randn(64, args.z_dim, 1, 1, device=device)

    for epoch in range(1, args.epochs+1):
        pbar = tqdm(dl, desc=f'GAN Epoch {epoch}/{args.epochs}')
        for real, _ in pbar:
            real = real.to(device)
            b = real.size(0)
            real_lab = torch.full((b,), 0.9, device=device)
            fake_lab = torch.zeros(b, device=device)

            # Train D
            netD.train(); netG.train()
            optD.zero_grad()
            out_real = netD(real)
            lossD_real = crit(out_real, real_lab)

            z = torch.randn(b, args.z_dim, 1, 1, device=device)
            fake = netG(z).detach()
            out_fake = netD(fake)
            lossD_fake = crit(out_fake, fake_lab)
            lossD = lossD_real + lossD_fake
            lossD.backward()
            optD.step()

            # Train G
            optG.zero_grad()
            z = torch.randn(b, args.z_dim, 1, 1, device=device)
            fake = netG(z)
            out_fake = netD(fake)
            lossG = crit(out_fake, real_lab)
            lossG.backward()
            optG.step()

            pbar.set_postfix(lossD=f'{lossD.item():.3f}', lossG=f'{lossG.item():.3f}')

        with torch.no_grad():
            fake = netG(fixed_z).cpu()*0.5+0.5
            utils.save_image(fake, os.path.join(args.out, f'samples_epoch_{epoch:03d}.png'), nrow=8)

        torch.save(netD.state_dict(), os.path.join(args.out, 'disc_state.pt'))
        torch.save(netG.state_dict(), os.path.join(args.out, 'gen_state.pt'))

    print('Saved discriminator to', os.path.join(args.out, 'disc_state.pt'))

if __name__ == '__main__':
    main()
