#!/bin/bash
# Setup isolated environment for Qwen3-Omni audio captioning
# This is needed because mlx-vlm has conflicting dependencies with the main project

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

echo "Setting up Qwen3-Omni captioner environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate and install dependencies
source "$VENV_DIR/bin/activate"

echo "Installing dependencies..."
pip install --upgrade pip
pip install "mlx-vlm @ git+https://github.com/Blaizzy/mlx-vlm.git"
pip install torchaudio soundfile

echo ""
echo "Setup complete!"
echo ""
echo "To use:"
echo "  source $VENV_DIR/bin/activate"
echo "  python caption_audio.py --audio /path/to/audio.wav"
echo ""
echo "Or run directly:"
echo "  $SCRIPT_DIR/run.sh --audio /path/to/audio.wav"
