# Qwen3-Omni Audio Captioner (MLX)

Rich audio captioning using Qwen3-Omni-30B on Apple Silicon via MLX.

## Requirements

- Apple Silicon Mac (M1/M2/M3/M4)
- ~32GB unified memory recommended
- ~60GB disk space for model download + ~17GB for converted model

## Setup

```bash
cd ~/signal4/core/scripts/qwen3_omni_captioner

# 1. Sync environment
uv sync

# 2. Convert model to MLX format (downloads ~60GB, converts to ~17GB 4-bit)
# This takes 30-60 minutes depending on network speed
uv run python -m mlx_vlm convert \
  --hf-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --mlx-path ./mlx_qwen3_omni_4bit \
  --quantize \
  --q-bits 4 \
  --q-group-size 64
```

## Usage

```bash
# Basic usage
uv run python caption_audio.py --audio /path/to/audio.wav

# Custom prompt
uv run python caption_audio.py --audio /path/to/audio.wav \
  --prompt "What emotions are expressed in this audio?"

# Longer output
uv run python caption_audio.py --audio /path/to/audio.wav --max-tokens 1024
```

## Output

The model generates rich descriptions including:
- Speaker identification and emotional tone
- Background sounds and music
- Overall mood and atmosphere
- What's happening or being discussed

## Troubleshooting

**Download corruption errors:**
```bash
# Clear cache and retry
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen3-Omni-30B-A3B-Instruct
# Then re-run the convert command
```

**Out of memory:**
- Close other applications
- Try shorter audio clips (< 30 seconds recommended)
