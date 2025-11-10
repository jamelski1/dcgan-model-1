import os, argparse, math
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import make_splits, collate_pad, SPECIAL, decode_ids
from models.model import EncoderCNN, DecoderGRU

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max_len", type=int, default=8)
    p.add_argument("--hid", type=int, default=256)
    p.add_argument("--emb", type=int, default=128)
    p.add_argument("--feat_dim", type=int, default=256)
    p.add_argument("--out", type=str, default="runs")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    train_set, val_set, test_set, vocab = make_splits(max_len=args.max_len)
    stoi, itos = vocab
    V = len(itos)
    bos_id = stoi["<bos>"]; eos_id = stoi["<eos>"]

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate_pad)
    val_loader   = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate_pad)

    enc = EncoderCNN(feat_dim=args.feat_dim).to(device)
    dec = DecoderGRU(feat_dim=args.feat_dim, vocab_size=V, hid=args.hid, emb=args.emb).to(device)

    params = list(enc.parameters()) + list(dec.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr)
    ce  = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=stoi["<pad>"])

    best_val = 1e9
    for epoch in range(1, args.epochs+1):
        enc.train(); dec.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        total = 0.0
        for xs, tgts, _ in pbar:
            xs, tgts = xs.to(device), tgts.to(device)
            opt.zero_grad()
            feat = enc(xs)
            logits = dec(feat, tgts)                 # (B,T-1,V)
            loss = ce(logits.reshape(-1, V), tgts[:,1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            total += loss.item() * xs.size(0)
            pbar.set_postfix(loss=f"{loss.item():.3f}")

        # validation (loss)
        enc.eval(); dec.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xs, tgts, _ in val_loader:
                xs, tgts = xs.to(device), tgts.to(device)
                feat = enc(xs)
                logits = dec(feat, tgts)
                loss = ce(logits.reshape(-1, V), tgts[:,1:].reshape(-1))
                val_loss += loss.item() * xs.size(0)

        val_loss /= len(val_set)
        print(f"Val loss: {val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            ckpt = {"enc": enc.state_dict(), "dec": dec.state_dict(), "vocab": vocab, "args": vars(args)}
            torch.save(ckpt, os.path.join(args.out, "best.pt"))
            print("Saved best checkpoint.")

if __name__ == "__main__":
    main()
