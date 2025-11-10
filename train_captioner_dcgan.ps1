# train_captioner_dcgan.ps1
# -------------------------------------
# Keeps PC awake, runs both training phases, logs output, then restores settings
# -------------------------------------

# Stop on any error
$ErrorActionPreference = "Stop"

Write-Host "🟢 Disabling sleep mode while training..."
# Set sleep timeout (0 = never)
powercfg /change standby-timeout-ac 0

# Ensure logs folder exists
if (-not (Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs" | Out-Null
}

Write-Host "🚀 Starting captioner training (frozen encoder)..."
python train_captioner_dcgan.py `
    --disc_ckpt runs_gan/disc_state.pt `
    --freeze_encoder `
    --epochs 12 `
    --batch_size 128 `
    *> logs\cap_frozen.log

Write-Host "✅ Frozen encoder training complete. Starting fine-tuning..."
python train_captioner_dcgan.py `
    --disc_ckpt runs_gan/disc_state.pt `
    --epochs 8 `
    --batch_size 128 `
    --lr 1e-4 `
    *> logs\cap_ft.log

Write-Host "💾 Training finished! Restoring normal sleep settings..."
# 30 minutes sleep timeout
powercfg /change standby-timeout-ac 30

Write-Host "🎉 All done! Logs saved in /logs."
