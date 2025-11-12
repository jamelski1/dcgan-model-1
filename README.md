# CIFAR-100 Image Captioning with DCGAN Encoder

This project implements an image-to-text captioning system for CIFAR-100 with multiple encoder options:
- **Baseline Encoder:** Simple CNN
- **DCGAN Discriminator Encoder:** Uses trained DCGAN discriminator features
- **Spectral Norm DCGAN Encoder:** Uses DCGAN with Spectral Normalization for improved stability
- **Decoder:** GRU-based sequence decoder
- **Captions:** Template-based "a photo of a {fine_label}"
- **Metrics:** BLEU-1, ROUGE-L (approx via LCS), and label-word accuracy

## Project Structure

```
├── models/             # Neural network architectures
├── training/           # Training scripts
├── utils/              # Data utilities and helpers
├── scripts/            # Evaluation and demo scripts
└── config/             # Configuration files
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m nltk.downloader punkt
```

## Training

### 1. Train Baseline Captioner
```bash
python training/train_captioner.py --encoder_type cnn --epochs 8 --batch_size 128 --lr 2e-4
```

### 2. Train DCGAN (with Spectral Normalization)
```bash
python training/train_gan.py --use_spectral_norm --epochs 100 --batch_size 128
```

### 3. Train Captioner with DCGAN Encoder
```bash
python training/train_captioner.py --encoder_type dcgan_sn \
    --dcgan_ckpt runs_gan_sn/dcgan_sn_ckpt_epoch_100.pt \
    --epochs 20 --batch_size 128
```

## Evaluation

```bash
python scripts/eval.py --ckpt runs_dcgan_enc/best.pt --batch_size 256 --max_len 8
```

## Demo (Generate Captions)

```bash
python scripts/demo.py --ckpt runs_dcgan_enc/best.pt --num 16 --max_len 8 --mode sample
```

## Notes
- CIFAR-100 images are small (32×32). Captions are primarily class names.
- Spectral Normalization improves DCGAN training stability.
- The discriminator's learned features serve as a powerful encoder for captioning.
