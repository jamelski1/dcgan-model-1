#!/bin/bash
# Startup script for Jamel's BetaBox Describinator

echo "🎨 Starting Jamel's BetaBox Describinator..."
echo "----------------------------------------"

# Check if model exists
if [ ! -f "../runs_hybrid/best_model.pt" ]; then
    echo "⚠️  Warning: Model checkpoint not found at runs_hybrid/best_model.pt"
    echo "The app will run in demo mode."
fi

# Start the Gradio app
echo "🚀 Launching Gradio interface..."
python app.py
