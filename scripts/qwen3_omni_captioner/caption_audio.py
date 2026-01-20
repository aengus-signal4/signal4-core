#!/usr/bin/env python3
"""
Qwen3-Omni audio captioning on MLX.

Usage:
    uv run caption_audio.py --audio /path/to/audio.wav
    uv run caption_audio.py --audio /path/to/audio.wav --prompt "What emotions are in this audio?"
"""

import argparse
import time
from pathlib import Path

import mlx.core as mx

# Use locally converted model (run convert first if not exists)
MODEL_PATH = "./mlx_qwen3_omni_4bit"


def caption_audio(audio_path: str, prompt: str = None, max_tokens: int = 512):
    """Generate rich audio description using Qwen3-Omni on MLX."""
    from mlx_vlm import load
    from mlx_vlm.models.qwen3_omni_moe.omni_utils import prepare_omni_inputs

    print(f"\nLoading model: {MODEL_PATH}")

    start = time.time()
    model, processor = load(MODEL_PATH, trust_remote_code=True)
    load_time = time.time() - start
    print(f"Model loaded in {load_time:.1f}s")

    if prompt is None:
        prompt = (
            "Describe this audio in rich detail. Include: "
            "1) Who is speaking and their emotional tone (excited, calm, angry, sad, etc.) "
            "2) Any background sounds, music, or ambient noise "
            "3) The overall mood and atmosphere "
            "4) What seems to be happening or being discussed"
        )

    print(f"\nAudio: {audio_path}")
    print(f"Prompt: {prompt[:100]}...")

    # Build conversation with audio
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    # Prepare inputs using the omni utils
    model_inputs, text = prepare_omni_inputs(processor, conversation)

    # Build generate kwargs
    generate_kwargs = {
        "input_ids": model_inputs["input_ids"],
        "pixel_values": model_inputs.get("pixel_values", None),
        "pixel_values_videos": model_inputs.get("pixel_values_videos", None),
        "image_grid_thw": model_inputs.get("image_grid_thw", None),
        "video_grid_thw": model_inputs.get("video_grid_thw", None),
        "input_features": model_inputs.get("input_features", None),
        "feature_attention_mask": model_inputs.get("feature_attention_mask", None),
        "audio_feature_lengths": model_inputs.get("audio_feature_lengths", None),
        "thinker_max_new_tokens": max_tokens,
        "return_audio": False,  # We only want text output
    }

    # Filter out None values
    generate_kwargs = {k: v for k, v in generate_kwargs.items() if v is not None}

    print("\nGenerating description...")
    start = time.time()

    thinker_result, _ = model.generate(**generate_kwargs)

    gen_time = time.time() - start

    # Decode the output
    output_text = processor.decode(thinker_result.sequences[0].tolist())

    print(f"\nGeneration time: {gen_time:.1f}s")
    print("\n" + "=" * 70)
    print("AUDIO DESCRIPTION:")
    print("=" * 70)
    print(output_text)
    print("=" * 70)

    return output_text


def main():
    parser = argparse.ArgumentParser(
        description="Generate rich audio descriptions using Qwen3-Omni on MLX"
    )
    parser.add_argument(
        "--audio", "-a",
        type=str,
        required=True,
        help="Path to audio file (wav, mp3, etc.)"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        help="Custom prompt (default: detailed description request)"
    )
    parser.add_argument(
        "--max-tokens", "-t",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)"
    )
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        return 1

    try:
        caption_audio(str(audio_path), args.prompt, args.max_tokens)
        return 0
    except Exception as e:
        print(f"\nError: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
