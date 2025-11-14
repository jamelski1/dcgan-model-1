# 🚀 Deploying BetaBox Describinator to Render

Congratulations on achieving **65.37% label accuracy**! This guide will help you deploy your trained model to Render.

## 📦 What You Need

Your trained model checkpoints:
- `runs_hybrid/best.pt` (Hybrid encoder - **65.37% accuracy**)
- `runs_gan_sn/best_disc.pt` (Frozen DCGAN discriminator)

## 🎯 Deployment Options

### Option 1: Cloud Storage (Recommended for Free Tier)

This is the best option for Render's free tier since checkpoint files are large.

#### Step 1: Upload Checkpoints to Cloud Storage

**Google Drive:**
1. Upload `runs_hybrid/best.pt` to Google Drive
2. Upload `runs_gan_sn/best_disc.pt` to Google Drive
3. Right-click each file → Share → Get link → Set to "Anyone with the link"
4. Get direct download links:
   - Google Drive link: `https://drive.google.com/file/d/FILE_ID/view`
   - Convert to: `https://drive.google.com/uc?export=download&id=FILE_ID`

**Dropbox:**
1. Upload files to Dropbox
2. Share → Copy link
3. Change `dl=0` to `dl=1` in the URL

**Hugging Face Hub (Best for ML models):**
```bash
# Install huggingface_hub
pip install huggingface-hub

# Upload your models
from huggingface_hub import upload_file

upload_file(
    path_or_fileobj="runs_hybrid/best.pt",
    path_in_repo="best.pt",
    repo_id="your-username/betabox-describinator",
    repo_type="model"
)

upload_file(
    path_or_fileobj="runs_gan_sn/best_disc.pt",
    path_in_repo="best_disc.pt",
    repo_id="your-username/betabox-describinator",
    repo_type="model"
)
```

#### Step 2: Deploy to Render

1. **Push your code to GitHub:**
   ```bash
   git add .
   git commit -m "Update BetaBox Describinator with 65.37% accuracy model"
   git push origin main
   ```

2. **Create Render Web Service:**
   - Go to https://render.com/dashboard
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Render will auto-detect `render.yaml` config

3. **Set Environment Variables:**
   In Render dashboard → Environment:
   - `MODEL_URL`: Your direct download link for `best.pt`
   - `DCGAN_URL`: Your direct download link for `best_disc.pt`

   Example:
   ```
   MODEL_URL=https://drive.google.com/uc?export=download&id=YOUR_FILE_ID
   DCGAN_URL=https://drive.google.com/uc?export=download&id=YOUR_FILE_ID
   ```

4. **Deploy:**
   - Click "Create Web Service"
   - Wait for build to complete (~5-10 minutes)
   - Your app will be live at `https://betabox-describinator.onrender.com`

### Option 2: Git LFS (For Larger Repos)

If you want checkpoints in your repo:

```bash
# Install Git LFS
git lfs install

# Track .pt files
git lfs track "*.pt"
git add .gitattributes

# Add your checkpoints
git add runs_hybrid/best.pt runs_gan_sn/best_disc.pt
git commit -m "Add model checkpoints via Git LFS"
git push origin main
```

**Note:** Free tier GitHub has 1GB storage limit for LFS.

### Option 3: Local Testing First

Before deploying to Render, test locally:

```bash
# Make sure your checkpoints are in the right place
ls runs_hybrid/best.pt
ls runs_gan_sn/best_disc.pt

# Install dependencies
pip install -r requirements.txt

# Run locally
python run_app.py
```

Visit `http://localhost:7860` to test!

## 🔍 Troubleshooting

### Models Not Loading on Render

**Issue:** App starts but shows "Demo mode"

**Solutions:**
1. Check environment variables are set correctly
2. Verify download URLs are publicly accessible
3. Check Render logs for download errors
4. Ensure `setup_models.py` is running (check logs)

### Out of Memory

**Issue:** Render free tier has 512MB RAM

**Solutions:**
1. Use CPU inference (already configured)
2. Consider upgrading to paid tier for more RAM
3. Reduce batch size if needed

### Slow Performance

**Issue:** Cold start or slow inference

**Solutions:**
1. Render free tier sleeps after 15 minutes of inactivity
2. First request after sleep takes longer (model loading)
3. Consider paid tier for always-on service
4. Use model quantization for faster CPU inference

## 📊 Expected Performance

- **Label Accuracy:** 65.37% on CIFAR-100 test set
- **Inference Time:** ~100-500ms per image (CPU)
- **Supported Images:** Any size (resized to 32×32)
- **Format:** RGB images (JPEG, PNG, etc.)

## 🎨 Customization

### Update UI Text

Edit `app/app.py`:
```python
# Line 186-194: Change header text
<h1>🎨 Your Custom Title</h1>
```

### Change Colors

Edit `CUSTOM_CSS` in `app/app.py`:
```python
# Line 17-19: Gradient colors
background: linear-gradient(135deg, #YOUR_COLOR1 0%, #YOUR_COLOR2 100%)
```

### Update Model Path

Edit `app/app.py`:
```python
# Line 130-131
MODEL_PATH = "your/custom/path.pt"
DCGAN_PATH = "your/dcgan/path.pt"
```

## 📝 Next Steps

After deployment:

1. **Test the live app** with various CIFAR-100 images
2. **Share the URL** with your network
3. **Monitor usage** in Render dashboard
4. **Collect feedback** for future improvements
5. **Consider adding:**
   - Example images gallery
   - Batch processing
   - API endpoint for programmatic access
   - Confidence scores
   - Attention visualization

## 🌟 Showcase Your Work

Share your accomplishment:
- LinkedIn post with results
- Twitter thread about the architecture
- Blog post explaining the hybrid approach
- GitHub README with performance metrics

---

**Built with ❤️ by Jamel**
*Hybrid DCGAN + ResNet18 Encoder | 65.37% Accuracy | Multi-Head Attention*
