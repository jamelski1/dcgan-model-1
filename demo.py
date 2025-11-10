import argparse, random
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from data_utils import make_splits, collate_pad, decode_ids, SPECIAL

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--num", type=int, default=16)
    p.add_argument("--max_len", type=int, default=8)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    from models.model import EncoderCNN, DecoderGRU
    stoi, itos = ckpt["vocab"]
    V = len(itos)
    enc = EncoderCNN(feat_dim=ckpt["args"]["feat_dim"]).to(device)
    dec = DecoderGRU(feat_dim=ckpt["args"]["feat_dim"], vocab_size=V, hid=ckpt["args"]["hid"], emb=ckpt["args"]["emb"]).to(device)
    enc.load_state_dict(ckpt["enc"]); dec.load_state_dict(ckpt["dec"])
    enc.eval(); dec.eval()

    _, _, test_set, _ = make_splits(max_len=args.max_len)
    test_set.dataset.stoi, test_set.dataset.itos = stoi, itos

    idxs = random.sample(range(len(test_set)), k=min(args.num, len(test_set)))
    imgs = []
    for i in idxs:
        x, tgt, y = test_set[i]
        imgs.append(x)

    xs = torch.stack(imgs, dim=0).to(device)
    with torch.no_grad():
        feat = enc(xs)
        gen = dec.generate(feat, stoi["<bos>"], stoi["<eos>"], max_len=args.max_len)

    # Save a grid and print captions
    xs_vis = (xs * 0.5 + 0.5).clamp(0,1).cpu()  # unnormalize
    grid = make_grid(xs_vis, nrow=int(len(xs_vis)**0.5))
    save_image(grid, "demo_grid.png")
    print("Saved grid as demo_grid.png")
    for i in range(xs.size(0)):
        cap = decode_ids(gen[i].tolist(), itos)
        print(f"[{i}] {cap}")

if __name__ == "__main__":
    main()
