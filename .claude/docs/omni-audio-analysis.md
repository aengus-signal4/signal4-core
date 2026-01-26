# Qwen3-Omni Audio Analysis

The tier_2 model server uses **Qwen3-Omni-30B-A3B** which supports native audio input alongside text.

## Model Location

Local 4-bit MLX conversion at:
```
core/scripts/qwen3_omni_captioner/mlx_qwen3_omni_4bit/
```

## Capabilities

| Feature | Support |
|---------|---------|
| Text input | Yes |
| Audio input | Yes (short clips) |
| Image input | No (disabled) |
| Audio output | No (text only) |

## Audio Limitations

**Short clips only (<30 seconds recommended)**

Memory usage scales with audio duration:
- 15s video: ~79GB
- 30s video: ~89GB
- 60s video: ~108GB
- 120s video: ~145GB

For longer audio, split into segments or use the existing transcription pipeline.

## API Usage

### Text-only request (standard)
```python
request = {
    "model": "tier_2",
    "messages": [{"role": "user", "content": "Analyze this text..."}]
}
```

### Audio + text request
```python
request = {
    "model": "tier_2",
    "messages": [{"role": "user", "content": "What emotions are in this audio?"}],
    "audio_path": "/path/to/audio.wav"  # Local path on server
}
```

## Supported Audio Formats

- WAV (recommended, 16kHz mono)
- MP3
- OPUS

The model processes audio natively - no separate transcription step needed.

## Example: Tone Analysis Prompt

```python
{
    "model": "tier_2",
    "messages": [{
        "role": "user",
        "content": """Listen to this audio and classify the speaker's vocal tone.
Respond with JSON:
{
  "gender": "male|female",
  "primary_emotion": "neutral|angry|happy|sad|fearful|disgusted|surprised",
  "dominance": "submissive|neutral|assertive|commanding",
  "confidence": "uncertain|neutral|confident|overconfident"
}"""
    }],
    "audio_path": "/tmp/segment.wav",
    "max_tokens": 256
}
```

## Performance

- ~22GB model weights (loaded once)
- ~10GB cache limit recommended for processing
- Processing speed: ~7x slower than real-time for audio analysis

## Text Performance Trade-off

Compared to Qwen3-30B-A3B-Instruct-2507 (text-only):

| Benchmark | Text-Only | Omni | Delta |
|-----------|-----------|------|-------|
| MMLU-Redux | 89.3 | 86.6 | -2.7 |
| ZebraLogic | 90.0 | 76.0 | -14.0 |
| Creative Writing | 86.0 | 80.6 | -5.4 |
| AIME25 (math) | 61.3 | 65.0 | +3.7 |

The Omni model trades some text performance for multimodal capabilities. For text-heavy tasks requiring maximum quality, tier_1 (80B) is recommended.
