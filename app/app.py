"""
Jamel's BetaBox Describinator
A minimal web UI for CIFAR-100 image captioning.
"""

import gradio as gr
import torch
from pathlib import Path
from inference import load_caption_generator
import time

# Custom CSS for minimal, Google-like design
CUSTOM_CSS = """
/* Clean, minimal design inspired by Google */
.gradio-container {
    font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif !important;
    max-width: 600px !important;
    margin: auto !important;
    background: #ffffff !important;
    padding: 0 !important;
}

/* Minimal header */
.header-container {
    text-align: center;
    margin: 8rem 0 3rem 0;
}

.header-container h1 {
    font-size: 3rem;
    font-weight: 300;
    margin: 0;
    color: #202124;
    letter-spacing: -0.5px;
}

.header-container p {
    font-size: 0.875rem;
    color: #5f6368;
    margin-top: 0.5rem;
    font-weight: 400;
}

/* Remove all borders and boxes */
#component-0 {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

/* Minimal image upload */
.image-container {
    border: none !important;
    background: transparent !important;
    border-radius: 0 !important;
    padding: 0 !important;
    margin: 2rem 0 !important;
}

.image-container:hover {
    background: transparent !important;
}

/* Clean button */
button {
    background: #f8f9fa !important;
    color: #202124 !important;
    border: 1px solid #dadce0 !important;
    border-radius: 4px !important;
    padding: 0.5rem 1.5rem !important;
    font-weight: 400 !important;
    font-size: 0.875rem !important;
    transition: all 0.1s ease !important;
    box-shadow: none !important;
    text-transform: none !important;
}

button:hover {
    background: #f1f3f4 !important;
    border-color: #d2d3d4 !important;
    box-shadow: 0 1px 1px rgba(0,0,0,.1) !important;
}

/* Output text - minimal, centered */
.output-class {
    background: transparent !important;
    border: none !important;
    padding: 2rem 0 !important;
    font-size: 1.25rem !important;
    font-weight: 300 !important;
    color: #202124 !important;
    text-align: center !important;
    min-height: 60px !important;
    letter-spacing: 0.25px !important;
}

/* Hide labels */
label {
    display: none !important;
}

/* Hide examples */
.examples-container {
    display: none !important;
}

/* Clean footer */
.footer {
    text-align: center;
    color: #5f6368;
    margin-top: 4rem;
    font-size: 0.75rem;
    padding: 1rem 0 2rem 0;
}

.footer a {
    color: #5f6368;
    text-decoration: none;
}

.footer a:hover {
    text-decoration: underline;
}

/* Thinking indicator */
.thinking {
    color: #5f6368;
    font-style: italic;
    font-weight: 300;
}

/* Hide Gradio branding */
.contain {
    max-width: 100% !important;
}

footer {
    display: none !important;
}

/* Dark mode */
@media (prefers-color-scheme: dark) {
    .gradio-container {
        background: #202124 !important;
    }

    .header-container h1 {
        color: #e8eaed;
    }

    .header-container p {
        color: #9aa0a6;
    }

    button {
        background: #303134 !important;
        color: #e8eaed !important;
        border-color: #5f6368 !important;
    }

    button:hover {
        background: #3c4043 !important;
        border-color: #9aa0a6 !important;
    }

    .output-class {
        color: #e8eaed !important;
    }

    .footer {
        color: #9aa0a6;
    }

    .footer a {
        color: #9aa0a6;
    }
}
"""

# Initialize model
MODEL_PATH = "runs_hybrid/best.pt"
DCGAN_PATH = "runs_gan_sn/best_disc.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")
print("Loading caption generator...")

try:
    generator = load_caption_generator(MODEL_PATH, device=DEVICE)
    model_loaded = True
    print("Model loaded successfully")
except Exception as e:
    print(f"Warning: Could not load model: {e}")
    print("The app will run in demo mode with placeholder responses.")
    model_loaded = False
    generator = None


def generate_caption_stream(image):
    """
    Generate caption for uploaded image with streaming effect.

    Args:
        image: PIL Image or numpy array

    Yields:
        Partial caption strings for streaming effect
    """
    if image is None:
        yield "Upload an image"
        return

    try:
        if model_loaded:
            # Use actual model
            from PIL import Image
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)

            # Show thinking state
            yield "thinking..."
            time.sleep(0.3)

            # Generate caption
            caption = generator.generate_caption(image)

            # Stream output character by character
            result = ""
            for char in caption:
                result += char
                yield result
                time.sleep(0.05)  # Adjust speed here
        else:
            # Demo mode
            yield "thinking..."
            time.sleep(0.3)
            demo_text = "a colorful scene"
            result = ""
            for char in demo_text:
                result += char
                yield result
                time.sleep(0.05)

    except Exception as e:
        yield f"Error: {str(e)}"


def create_interface():
    """Create the minimal Gradio interface."""

    with gr.Blocks(css=CUSTOM_CSS, title="BetaBox Describinator") as demo:
        # Minimal header
        gr.HTML("""
            <div class="header-container">
                <h1>BetaBox Describinator</h1>
                <p>Image captioning powered by AI</p>
            </div>
        """)

        # Image input
        image_input = gr.Image(
            type="pil",
            elem_classes=["image-container"],
            show_label=False
        )

        # Caption output (streaming)
        caption_output = gr.Textbox(
            show_label=False,
            placeholder="",
            lines=2,
            elem_classes=["output-class"],
            interactive=False
        )

        # Submit button
        submit_btn = gr.Button("Describe", variant="secondary")

        # Footer
        gr.HTML("""
            <div class="footer">
                <a href="https://huggingface.co/jamelski/jamels-betabox-describinator" target="_blank">Model</a>
                •
                Built by Jamel
            </div>
        """)

        # Connect streaming function
        submit_btn.click(
            fn=generate_caption_stream,
            inputs=image_input,
            outputs=caption_output
        )

        # Also trigger on image upload
        image_input.change(
            fn=generate_caption_stream,
            inputs=image_input,
            outputs=caption_output
        )

    return demo


if __name__ == "__main__":
    # Create and launch interface
    demo = create_interface()

    # Launch settings
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
