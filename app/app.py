"""
Jamel's BetaBox Describinator
A sleek, minimal web UI for CIFAR-100 image captioning.
"""

import gradio as gr
import torch
from pathlib import Path
from inference import load_caption_generator

# Custom CSS for sleek, minimal design
CUSTOM_CSS = """
/* Main container styling */
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    max-width: 900px !important;
    margin: auto !important;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    padding: 2rem !important;
}

/* Header styling */
.header-container {
    text-align: center;
    margin-bottom: 2rem;
    color: white;
}

.header-container h1 {
    font-size: 2.5rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
}

.header-container p {
    font-size: 1.1rem;
    opacity: 0.9;
    font-weight: 300;
}

/* Main content card */
#component-0 {
    background: white !important;
    border-radius: 20px !important;
    box-shadow: 0 20px 60px rgba(0,0,0,0.3) !important;
    padding: 2rem !important;
}

/* Image upload area */
.image-container {
    border: 3px dashed #667eea !important;
    border-radius: 15px !important;
    background: #f8f9ff !important;
    transition: all 0.3s ease !important;
}

.image-container:hover {
    border-color: #764ba2 !important;
    background: #f0f2ff !important;
}

/* Upload button styling */
button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem 2rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
}

button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
}

/* Output text styling */
.output-class {
    background: #f8f9ff !important;
    border: 2px solid #667eea !important;
    border-radius: 15px !important;
    padding: 1.5rem !important;
    font-size: 1.2rem !important;
    font-weight: 500 !important;
    color: #333 !important;
    text-align: center !important;
    min-height: 80px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

/* Examples styling */
.examples-container {
    margin-top: 2rem !important;
    padding-top: 2rem !important;
    border-top: 2px solid #e0e0e0 !important;
}

/* Footer styling */
.footer {
    text-align: center;
    color: white;
    margin-top: 2rem;
    font-size: 0.9rem;
    opacity: 0.8;
}

/* Loading animation */
.loading {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 3px solid rgba(255,255,255,.3);
    border-radius: 50%;
    border-top-color: white;
    animation: spin 1s ease-in-out infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}
"""

# Initialize model
MODEL_PATH = "runs_hybrid/best_model.pt"  # Update this path to your best model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")
print("Loading caption generator...")

try:
    generator = load_caption_generator(MODEL_PATH, device=DEVICE)
    model_loaded = True
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"⚠ Warning: Could not load model: {e}")
    print("The app will run in demo mode with placeholder responses.")
    model_loaded = False
    generator = None


def generate_caption(image):
    """
    Generate caption for uploaded image.

    Args:
        image: PIL Image or numpy array

    Returns:
        Caption string
    """
    if image is None:
        return "Please upload an image first! 📸"

    try:
        if model_loaded:
            # Use actual model
            from PIL import Image
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)

            caption = generator.generate_caption(image)
            return f"🎨 {caption.capitalize()}"
        else:
            # Demo mode placeholder
            return "🎨 Demo mode: A colorful scene from CIFAR-100 dataset (model not loaded)"

    except Exception as e:
        return f"❌ Error: {str(e)}"


# Create Gradio interface
def create_interface():
    """Create the Gradio interface."""

    with gr.Blocks(css=CUSTOM_CSS, title="Jamel's BetaBox Describinator") as demo:
        # Header
        gr.HTML("""
            <div class="header-container">
                <h1>🎨 Jamel's BetaBox Describinator</h1>
                <p>Upload an image and let AI describe what it sees</p>
                <p style="font-size: 0.9rem; margin-top: 0.5rem;">
                    Powered by Hybrid DCGAN + ResNet18 Encoder with Multi-Head Attention
                </p>
            </div>
        """)

        with gr.Row():
            with gr.Column(scale=1):
                # Image input
                image_input = gr.Image(
                    label="📤 Upload Image",
                    type="pil",
                    elem_classes=["image-container"]
                )

                # Submit button
                submit_btn = gr.Button("🚀 Generate Caption", variant="primary")

        with gr.Row():
            with gr.Column():
                # Caption output
                caption_output = gr.Textbox(
                    label="📝 Generated Caption",
                    placeholder="Your caption will appear here...",
                    lines=3,
                    elem_classes=["output-class"]
                )

        # Examples section
        gr.HTML("""
            <div class="examples-container">
                <h3 style="text-align: center; color: #667eea; margin-bottom: 1rem;">
                    💡 Try Some Examples
                </h3>
            </div>
        """)

        examples = gr.Examples(
            examples=[
                # You can add example images here
                # ["path/to/example1.jpg"],
                # ["path/to/example2.jpg"],
            ],
            inputs=image_input,
            outputs=caption_output,
            fn=generate_caption,
            cache_examples=False,
        )

        # Footer
        gr.HTML("""
            <div class="footer">
                <p>Built with ❤️ by Jamel | Trained on CIFAR-100 Dataset</p>
                <p style="font-size: 0.8rem; margin-top: 0.5rem;">
                    Model: Frozen DCGAN Discriminator + Trainable ResNet18 + Multi-Head Attention
                </p>
            </div>
        """)

        # Connect button to function
        submit_btn.click(
            fn=generate_caption,
            inputs=image_input,
            outputs=caption_output
        )

        # Also trigger on image upload
        image_input.change(
            fn=generate_caption,
            inputs=image_input,
            outputs=caption_output
        )

    return demo


if __name__ == "__main__":
    # Create and launch interface
    demo = create_interface()

    # Launch settings
    demo.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,
        share=False,  # Set to True for temporary public link
        show_error=True,
    )
