# DCGAN Encoder Add-on

Files included:
- `models/dcgan.py` — DCGAN Generator + Discriminator for 32x32 CIFAR
- `models/dcgan_encoder.py` — Wrap the discriminator trunk as an encoder
- `train_dcgan.py` — Train the GAN (save `runs_gan/disc_state.pt`)
- `train_captioner_dcgan.py` — Train the captioner using the DCGAN encoder

## Usage

1) Train DCGAN (uses existing `./data/cifar-100-python/`; no download)
```
python train_dcgan.py --epochs 50 --batch_size 128 --out runs_gan
```

2) Train captioner with DCGAN encoder (start frozen)
```
python train_captioner_dcgan.py --disc_ckpt runs_gan/disc_state.pt --freeze_encoder --epochs 12 --batch_size 128
```

3) Optional fine-tune (unfreeze)
```
python train_captioner_dcgan.py --disc_ckpt runs_gan/disc_state.pt --epochs 8 --batch_size 128 --lr 1e-4
```
