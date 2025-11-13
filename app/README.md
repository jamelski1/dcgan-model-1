# 🎨 Jamel's BetaBox Describinator

A sleek, minimal web UI for CIFAR-100 image captioning using a hybrid DCGAN + ResNet18 encoder with multi-head attention.

## Features

- 📤 **Easy Image Upload**: Drag-and-drop or click to upload
- 🤖 **AI-Powered Captions**: Hybrid encoder architecture for accurate descriptions
- ✨ **Sleek UI**: Minimal, modern design with smooth animations
- 🚀 **Fast Inference**: Optimized for quick caption generation

## Architecture

The model uses:
- **Frozen DCGAN Discriminator**: Pre-trained features (256 channels)
- **Trainable ResNet18**: Fine-tuned features (512 channels)
- **Multi-Head Attention**: 8-head attention mechanism for spatial understanding
- **Total Features**: 768 channels (256 + 512) at 4×4 spatial resolution

## Local Setup

### Prerequisites

- Python 3.10+
- PyTorch 2.1+
- Trained model checkpoint

### Installation

```bash
# Install dependencies
pip install -r ../requirements.txt

# Update model path in app.py
# MODEL_PATH = "runs_hybrid/best_model.pt"  # Point to your trained model

# Run the app
python app.py
```

The app will be available at `http://localhost:7860`

## Deployment to Render

### Step 1: Prepare Your Repository

1. Ensure your trained model checkpoint is committed to the repo or accessible via URL
2. Update `MODEL_PATH` in `app/app.py` to point to your model location
3. Commit all changes to GitHub

### Step 2: Deploy on Render

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Render will automatically detect the `render.yaml` configuration

Alternatively, configure manually:
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `python app/app.py`
- **Environment**: Python 3
- **Plan**: Free tier works great for demos

### Step 3: Configure Environment Variables

If needed, add these environment variables in Render:
- `MODEL_PATH`: Path to your model checkpoint
- `DEVICE`: `cpu` or `cuda` (use `cpu` for free tier)

### Step 4: Deploy!

Click "Create Web Service" and Render will build and deploy your app.

## Configuration Options

### Model Path

Update in `app/app.py`:
```python
MODEL_PATH = "runs_hybrid/best_model.pt"  # Update this
```

### Device Selection

```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

### UI Customization

Edit the `CUSTOM_CSS` variable in `app/app.py` to customize colors, fonts, and styling.

## File Structure

```
app/
├── app.py           # Main Gradio application
├── inference.py     # Model inference logic
└── README.md        # This file
```

## Usage

1. **Upload an Image**: Click or drag an image to the upload area
2. **Generate Caption**: Click "Generate Caption" or wait for auto-generation
3. **View Result**: See the AI-generated caption below

## Troubleshooting

### Model Not Loading

If the model fails to load:
1. Check the `MODEL_PATH` is correct
2. Ensure the checkpoint includes vocabulary (`vocab` key)
3. Verify DCGAN checkpoint path in config

### Memory Issues on Render

For free tier Render deployments:
- Use CPU inference (`DEVICE = "cpu"`)
- Consider model quantization for smaller memory footprint
- Use the free tier's 512MB RAM efficiently

### UI Not Loading

1. Check that port 7860 is accessible
2. Verify Gradio version compatibility
3. Check browser console for CSS/JS errors

## Credits

- **Architecture**: Hybrid DCGAN + ResNet18 with Multi-Head Attention
- **Dataset**: CIFAR-100
- **Framework**: PyTorch, Gradio
- **Built by**: Jamel

## License

See repository root for license information.
