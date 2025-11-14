"""
Jamel's BetaBox Describinator
A minimal web UI for CIFAR-100 image captioning.
"""

import gradio as gr
import torch
from pathlib import Path
from inference import load_caption_generator
import time

# Custom CSS for minimal design with deep green background
CUSTOM_CSS = """
/* Clean, minimal design with deep green monotone background */
.gradio-container {
    font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif !important;
    max-width: 600px !important;
    margin: auto !important;
    background: #0f4c3a !important;
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
    color: #ffffff !important;
    letter-spacing: -0.5px;
}

.header-container p {
    font-size: 0.875rem;
    color: #e0f2e9 !important;
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
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    background: rgba(255, 255, 255, 0.05) !important;
    border-radius: 4px !important;
    padding: 1rem !important;
    margin: 2rem 0 !important;
}

.image-container:hover {
    background: rgba(255, 255, 255, 0.08) !important;
    border-color: rgba(255, 255, 255, 0.3) !important;
}

/* Clean button */
button {
    background: rgba(255, 255, 255, 0.1) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    border-radius: 4px !important;
    padding: 0.5rem 1.5rem !important;
    font-weight: 400 !important;
    font-size: 0.875rem !important;
    transition: all 0.2s ease !important;
    box-shadow: none !important;
    text-transform: none !important;
}

button:hover {
    background: rgba(255, 255, 255, 0.2) !important;
    border-color: rgba(255, 255, 255, 0.5) !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2) !important;
}

/* Output text - minimal, centered, HIGHLY VISIBLE */
.output-class {
    background: transparent !important;
    border: none !important;
    padding: 2rem 0 !important;
    font-size: 1.5rem !important;
    font-weight: 400 !important;
    color: #ffffff !important;
    text-align: center !important;
    min-height: 60px !important;
    letter-spacing: 0.5px !important;
    line-height: 1.6 !important;
}

/* Output textbox styling - VERY AGGRESSIVE */
.output-class textarea,
.output-class input,
textarea.output-class,
input.output-class,
.output-class textarea.scroll-hide,
.output-class input[type="text"],
div.output-class textarea,
div.output-class input {
    color: #ffffff !important;
    background: transparent !important;
    -webkit-text-fill-color: #ffffff !important;
}

/* Target Gradio's internal textbox structure */
.gradio-container .output-class textarea,
.gradio-container .output-class input,
.gr-textbox.output-class textarea,
.gr-textbox.output-class input {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
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
    color: #c5e3d6 !important;
    margin-top: 4rem;
    font-size: 0.75rem;
    padding: 1rem 0 2rem 0;
}

.footer a {
    color: #e0f2e9 !important;
    text-decoration: none;
}

.footer a:hover {
    text-decoration: underline;
    color: #ffffff !important;
}

/* Thinking indicator */
.thinking {
    color: #c5e3d6 !important;
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

/* Override any Gradio default text colors */
* {
    color: inherit;
}

/* Ensure all text in containers is white */
.gradio-container * {
    color: #ffffff !important;
}

/* Fix textbox specifically - MAXIMUM OVERRIDE */
.gradio-container textarea,
.gradio-container input[type="text"],
textarea,
input[type="text"],
.gr-textbox textarea,
.gr-textbox input,
.gr-box textarea,
.gr-box input {
    color: #ffffff !important;
    caret-color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

/* Additional fallback for any text element */
p, span, div, textarea, input {
    color: #ffffff !important;
}

/* Specific targeting for caption output */
#caption-output,
#caption-output textarea,
#caption-output input,
#caption-output * {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    background: transparent !important;
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

    with gr.Blocks(css=CUSTOM_CSS, title="BetaBox Describinator", js="""
        function() {
            // Force white text color on all textareas and inputs
            const style = document.createElement('style');
            style.textContent = `
                textarea, input {
                    color: #ffffff !important;
                    -webkit-text-fill-color: #ffffff !important;
                }
            `;
            document.head.appendChild(style);

            // Also set inline styles
            setInterval(() => {
                document.querySelectorAll('textarea, input').forEach(el => {
                    el.style.color = '#ffffff';
                    el.style.webkitTextFillColor = '#ffffff';
                });
            }, 100);
        }
    """) as demo:
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
            value="Ready to describe your image",
            lines=2,
            elem_classes=["output-class"],
            elem_id="caption-output",
            interactive=False,
            container=False
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
