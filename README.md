<<<<<<< HEAD
# CIFAR-100 Image Captioning — Baseline (No Extra Credit)

This is a **minimal, from-scratch baseline** captioning system for CIFAR-100:
- **Encoder:** small CNN
- **Decoder:** GRU
- **Captions:** short template "a photo of a {fine_label}"
- **Metrics:** BLEU-1, ROUGE-L (approx via LCS), and label-word accuracy

> You can later plug in DCGAN and/or Stable Diffusion, but this repo builds the basic working pipeline first.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m nltk.downloader punkt
```

## Train

```bash
python train.py --epochs 8 --batch_size 128 --lr 2e-4 --max_len 8
```

## Evaluate

```bash
python eval.py --ckpt runs/best.pt --batch_size 256 --max_len 8
```

## Demo (sample captions)

```bash
python demo.py --ckpt runs/best.pt --num 16 --max_len 8
```

## Notes
- CIFAR-100 is small (32×32). Expect captions to mostly be **class names**.
- Keep it simple; this baseline gives you a clean scaffold to add DCGAN or SD later.
=======
# dcgan-model-1
First image-to-text attempt
>>>>>>> 1ddbd6dc9f5b97259e63f7726c5bd22cbd649f4f
