import argparse
import os
import torch
from torchvision import datasets, transforms, utils as vutils
from PIL import Image

# Project imports
from models.dcgan_encoder import DCGANDiscEncoder
from models.model import DecoderGRU

def load_ckpt(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt.get("args", {})
    stoi, itos = ckpt["vocab"]
    # hyperparams with sensible defaults
    feat_dim = args.get("feat_dim", 256)
    ndf      = args.get("ndf", 64)
    hid      = args.get("hid", 256)
    emb      = args.get("emb", 128)
    max_len  = args.get("max_len", 8)
    return ckpt, (stoi, itos), dict(feat_dim=feat_dim, ndf=ndf, hid=hid, emb=emb, max_len=max_len)

def build_models(hps, vocab_size, device):
    enc = DCGANDiscEncoder(feat_dim=hps["feat_dim"], ndf=hps["ndf"]).to(device)
    dec = DecoderGRU(feat_dim=hps["feat_dim"], vocab_size=vocab_size, hid=hps["hid"], emb=hps["emb"]).to(device)
    return enc, dec

@torch.no_grad()
def greedy_caption(enc, dec, img, stoi, itos, max_len, device):
    """
    Constrained decoding:
      - seed with "<bos> a photo of a"
      - greedy argmax for the rest
      - no immediate-repeat
    """
    enc.eval(); dec.eval()
    bos = stoi["<bos>"]; eos = stoi["<eos>"]; pad = stoi["<pad>"]

    # Build fixed prefix ids if words exist in vocab; fallback to just <bos>,pad
    prefix_words = ["<bos>", "a", "photo", "of", "a"]
    prefix_ids = []
    for w in prefix_words:
        if w in stoi:
            prefix_ids.append(stoi[w])
    if len(prefix_ids) == 0:
        prefix_ids = [bos]
    if len(prefix_ids) == 1:  # ensure seq_len >= 2 for this decoder
        prefix_ids.append(pad)

    feat = enc(img.unsqueeze(0).to(device))
    inp  = torch.tensor([prefix_ids], device=device)

    outs = []    # we only store tokens AFTER the fixed prefix
    last_token = None

    # we only allow a few extra tokens (CIFAR captions are short)
    steps = max(1, max_len - (inp.size(1) - 1))
    for _ in range(steps):
        logits = dec(feat, inp)[:, -1, :]     # (1, V)
        # greedy with immediate no-repeat
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        if last_token is not None:
            probs[last_token] = 0.0
        nxt = torch.argmax(probs).item()

        outs.append(nxt)
        if nxt == eos:
            break
        last_token = nxt
        inp = torch.cat([inp, torch.tensor([[nxt]], device=device)], dim=1)

    # decode ids → words (skip special)
    words = []
    for tid in outs:
        w = itos[int(tid)]
        if w in ("<bos>", "<pad>"): 
            continue
        if w == "<eos>":
            break
        words.append(w)

    # Build final sentence with fixed prefix text
    return " ".join(["a", "photo", "of", "a"] + words).strip()

def load_external_image(path):
    img = Image.open(path).convert("RGB").resize((32, 32), Image.BICUBIC)
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    ])
    return tfm(img)
    
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="Path to trained captioner checkpoint (.pt)")
    ap.add_argument("--n", type=int, default=16, help="Number of CIFAR-100 test images to sample")
    ap.add_argument("--out", type=str, default="demo_dcgan_grid.png", help="Output grid image filename")
    ap.add_argument("--external", type=str, default=None, help="Optional path to an external image to caption (resized to 32x32)")
    ap.add_argument("--indices", type=str, default=None, help="Optional comma-separated test indices to caption (e.g., '0,5,12')")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load checkpoint + models
    ckpt, (stoi, itos), hps = load_ckpt(args.ckpt, device)
    V = len(itos)
    enc, dec = build_models(hps, V, device)

    # load encoder/decoder states (support both 'enc' and 'enc_disc')
    state_enc = ckpt.get("enc", ckpt.get("enc_disc"))
    if state_enc is None:
        raise KeyError("Checkpoint missing both 'enc' and 'enc_disc' keys.")
    enc.load_state_dict(state_enc)
    dec.load_state_dict(ckpt["dec"])

    # If an external image is provided, caption it and exit
    if args.external is not None:
        x = load_external_image(args.external)
        cap = greedy_caption(enc, dec, x, stoi, itos, hps["max_len"], device)
        print(f"[external] {os.path.basename(args.external)} -> {cap}")
        return

    # Otherwise, sample CIFAR-100 test images
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    test = datasets.CIFAR100("./data", train=False, download=False, transform=tfm)

    # Select indices
    if args.indices:
        idxs = [int(i) for i in args.indices.split(",") if len(i.strip()) > 0]
    else:
        # first N by default
        idxs = list(range(min(args.n, len(test))))

    # stack images and save grid for visual reference
    imgs = torch.stack([test[i][0] for i in idxs])   # (N,3,32,32)
    vutils.save_image(imgs*0.5+0.5, args.out, nrow=min(8, len(idxs)))
    print(f"Saved image grid to: {args.out}")

    # generate captions
    print("Captions:")
    for k, i in enumerate(idxs):
        cap = greedy_caption(enc, dec, imgs[k].cpu(), stoi, itos, hps["max_len"], device)
        print(f"[{i}] {cap}")

if __name__ == "__main__":
    main()
