import argparse, math
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import make_splits, collate_pad, decode_ids, SPECIAL

# Simple BLEU-1 and ROUGE-L (LCS) and label-word accuracy

def bleu1(ref_tokens, hyp_tokens):
    if not hyp_tokens: return 0.0
    ref = set(ref_tokens)
    hit = sum(1 for t in hyp_tokens if t in ref)
    return hit / max(1, len(hyp_tokens))

def lcs(a, b):
    # classic DP
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n):
        for j in range(m):
            if a[i] == b[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    return dp[n][m]

def rouge_l(ref_tokens, hyp_tokens):
    if not ref_tokens or not hyp_tokens: return 0.0
    L = lcs(ref_tokens, hyp_tokens)
    prec = L / len(hyp_tokens)
    rec  = L / len(ref_tokens)
    if prec+rec == 0: return 0.0
    return (2*prec*rec)/(prec+rec)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=256)
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

    # load test split
    _, _, test_set, vocab2 = make_splits(max_len=args.max_len)
    # ensure same vocab
    test_set.dataset.stoi, test_set.dataset.itos = stoi, itos
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate_pad)

    bos_id, eos_id = stoi["<bos>"], stoi["<eos>"]
    b1_sum = 0.0; rl_sum = 0.0; lab_ok = 0; n = 0

    with torch.no_grad():
        for xs, tgts, ys in tqdm(test_loader, desc="Eval"):
            xs, tgts = xs.to(device), tgts.to(device)
            feat = enc(xs)
            gen_ids = dec.generate(feat, bos_id, eos_id, max_len=args.max_len)  # (B,T-1)
            for i in range(xs.size(0)):
                ref = decode_ids(tgts[i].tolist(), itos).split()
                hyp = decode_ids(gen_ids[i].tolist(), itos).split()
                # metrics
                b1_sum += bleu1(ref, hyp)
                rl_sum += rouge_l(ref, hyp)
                # label word accuracy: check if true fine label is in hyp tokens
                true_label = test_set.dataset.fine_names[ys[i].item()]
                lab_ok += (true_label in hyp)
                n += 1

    print(f"BLEU-1: {b1_sum/n:.4f}")
    print(f"ROUGE-L: {rl_sum/n:.4f}")
    print(f"Label-word accuracy: {lab_ok/n:.4f}")

if __name__ == "__main__":
    main()
