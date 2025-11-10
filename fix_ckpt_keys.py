# fix_ckpt_keys.py
import sys, torch

src = sys.argv[1] if len(sys.argv) > 1 else "runs_dcgan_enc/best_dcgan_enc.pt"
dst = sys.argv[2] if len(sys.argv) > 2 else src.replace(".pt", "_eval.pt")

ck = torch.load(src, map_location="cpu")
if "enc" not in ck and "enc_disc" in ck:
    ck["enc"] = ck.pop("enc_disc")
    torch.save(ck, dst)
    print(f"Saved converted checkpoint → {dst}")
else:
    print("No change needed (already has 'enc').")
