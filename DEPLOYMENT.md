# 🚀 Deployment Guide: Jamel's BetaBox Describinator

This guide explains how to deploy your image captioning model as a web application.

## Quick Start

### Local Testing

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
cd app
python app.py
```

Visit `http://localhost:7860` to see your app!

## Deployment Options

### Option 1: Render (Recommended)

Render offers free hosting perfect for ML demos.

#### Steps:

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Add BetaBox Describinator web UI"
   git push origin main
   ```

2. **Create Render Account**
   - Go to https://render.com
   - Sign up with GitHub

3. **Create New Web Service**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select the repository

4. **Configure (Auto-detected from render.yaml)**
   - **Name**: betabox-describinator
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app/app.py`
   - **Plan**: Free

5. **Deploy**
   - Click "Create Web Service"
   - Wait for build to complete (~5-10 minutes)
   - Your app will be live at `https://betabox-describinator.onrender.com`

### Option 2: Hugging Face Spaces

Hugging Face Spaces is great for Gradio apps.

#### Steps:

1. **Create Space**
   - Go to https://huggingface.co/spaces
   - Click "Create new Space"
   - Choose "Gradio" as the SDK
   - Name it "betabox-describinator"

2. **Upload Files**
   ```
   app/app.py → app.py (in root)
   app/inference.py → inference.py
   requirements.txt → requirements.txt
   models/ → models/ (folder)
   ```

3. **Add Model Checkpoint**
   - Upload your trained model to the Space
   - Or use Git LFS for large files

4. **Deploy**
   - Space will auto-deploy on file push
   - Visit your Space URL

### Option 3: Local Docker

For containerized deployment:

```dockerfile
# Create Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "app/app.py"]
```

```bash
# Build and run
docker build -t betabox .
docker run -p 7860:7860 betabox
```

## Configuration

### Model Path

Update in `app/app.py`:
```python
MODEL_PATH = "runs_hybrid/best_model.pt"
```

Options:
- Local path (if checkpoint is in repo)
- Download from URL in startup script
- Use environment variable

### Device Selection

For free tier deployments, use CPU:
```python
DEVICE = "cpu"  # or "cuda" if GPU available
```

### Custom Domain (Render)

1. Go to Settings → Custom Domain
2. Add your domain
3. Configure DNS records

## Important Notes

### Model Checkpoint

Your trained model checkpoint needs to be accessible:

**Option A: Include in Repository**
- If < 100MB, commit to repo
- ⚠️ Increases repo size

**Option B: Download on Startup**
- Host checkpoint on Google Drive, Dropbox, or S3
- Add download script before app starts

**Option C: Git LFS**
- Use Git Large File Storage
- `git lfs track "*.pt"`

### Memory Constraints

Free tier has limited RAM (512MB on Render):
- Use CPU inference
- Consider model quantization
- Optimize batch size

### Environment Variables

Set these in your hosting platform:

- `MODEL_PATH`: Path to checkpoint
- `DEVICE`: cpu or cuda
- `PORT`: 7860 (default)

## Troubleshooting

### Build Fails

**Issue**: Dependencies not installing
**Fix**: Check requirements.txt compatibility

**Issue**: Out of memory during build
**Fix**: Reduce dependencies, use lighter PyTorch version

### App Won't Start

**Issue**: Model not loading
**Fix**: Verify MODEL_PATH is correct

**Issue**: Port already in use
**Fix**: Change PORT env variable

### Slow Performance

**Issue**: Inference takes too long
**Fix**:
- Use CPU-optimized builds
- Add model caching
- Consider quantization

## Next Steps

### Enhancements

1. **Add Example Images**
   - Include CIFAR-100 samples
   - Show model capabilities

2. **Analytics**
   - Track usage with Google Analytics
   - Monitor inference times

3. **A/B Testing**
   - Compare different model versions
   - Test UI variations

4. **API Endpoint**
   - Add REST API for programmatic access
   - Use FastAPI alongside Gradio

5. **Batch Processing**
   - Allow multiple image uploads
   - Generate captions in batch

## Resources

- [Gradio Documentation](https://gradio.app/docs)
- [Render Docs](https://render.com/docs)
- [Hugging Face Spaces](https://huggingface.co/docs/hub/spaces)

## Support

For issues, check:
1. Model checkpoint is accessible
2. All dependencies are installed
3. PORT is correctly configured
4. Device (CPU/CUDA) matches availability

---

Built with ❤️ by Jamel
