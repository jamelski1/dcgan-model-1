import os, argparse
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_utils import make_splits, collate_pad, SPECIAL
from models.dcgan_encoder import DCGANDiscEncoder
from models.model import DecoderGRU

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=12)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--max_len', type=int, default=8)
    ap.add_argument('--hid', type=int, default=256)
    ap.add_argument('--emb', type=int, default=128)
    ap.add_argument('--feat_dim', type=int, default=256)
    ap.add_argument('--ndf', type=int, default=64)
    ap.add_argument('--disc_ckpt', type=str, default='runs_gan/disc_state.pt')
    ap.add_argument('--freeze_encoder', action='store_true')
    ap.add_argument('--out', type=str, default='runs_dcgan_enc')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out, exist_ok=True)

    train_set, val_set, test_set, vocab = make_splits(max_len=args.max_len)
    stoi, itos = vocab
    V = len(itos)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate_pad)
    val_loader   = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate_pad)

    enc = DCGANDiscEncoder(feat_dim=args.feat_dim, ndf=args.ndf).to(device)
    sd = torch.load(args.disc_ckpt, map_location=device)
    enc.disc.load_state_dict(sd, strict=False)
    if args.freeze_encoder:
        for p in enc.parameters():
            p.requires_grad = False

    dec = DecoderGRU(feat_dim=args.feat_dim, vocab_size=V, hid=args.hid, emb=args.emb).to(device)

    params = list(dec.parameters()) + ([] if args.freeze_encoder else list(enc.parameters()))
    opt = torch.optim.AdamW(params, lr=args.lr)
    ce  = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=stoi['<pad>'])

    best_val = 1e9
    for epoch in range(1, args.epochs+1):
        enc.train(); dec.train()
        pbar = tqdm(train_loader, desc=f'Caption (DCGAN enc) Epoch {epoch}/{args.epochs}')
        for xs, tgts, _ in pbar:
            xs, tgts = xs.to(device), tgts.to(device)
            opt.zero_grad()
            feat = enc(xs)
            logits = dec(feat, tgts)
            Vsz = logits.size(-1)
            loss = ce(logits.reshape(-1, Vsz), tgts[:,1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            pbar.set_postfix(loss=f'{loss.item():.3f}')

        # val
        enc.eval(); dec.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xs, tgts, _ in val_loader:
                xs, tgts = xs.to(device), tgts.to(device)
                feat = enc(xs)
                logits = dec(feat, tgts)
                Vsz = logits.size(-1)
                loss = ce(logits.reshape(-1, Vsz), tgts[:,1:].reshape(-1))
                val_loss += loss.item() * xs.size(0)
        val_loss /= len(val_set)
        print(f'Val loss: {val_loss:.4f}')
        if val_loss < best_val:
            best_val = val_loss
            ckpt = {'enc_disc': enc.state_dict(), 'dec': dec.state_dict(), 'vocab': vocab, 'args': vars(args)}
            torch.save(ckpt, os.path.join(args.out, 'best_dcgan_enc.pt'))
            print('Saved best checkpoint.')

if __name__ == '__main__':
    main()
