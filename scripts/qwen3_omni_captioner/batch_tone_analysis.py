#!/usr/bin/env python3
"""
Batch tone analysis on real segments using Qwen3-Omni.

Downloads audio segments from S3 and extracts structured tone signals.

Usage:
    uv run python batch_tone_analysis.py --count 20

Requires: Environment variables from core/.env (POSTGRES_PASSWORD, S3_ACCESS_KEY, S3_SECRET_KEY)
"""

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import boto3
import psycopg2
from dotenv import load_dotenv

# Load environment from core/.env
# batch_tone_analysis.py -> qwen3_omni_captioner -> scripts -> core
CORE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(CORE_DIR / ".env", override=True)

MODEL_PATH = "./mlx_qwen3_omni_4bit"

# Structured tone analysis prompt - categorical labels
TONE_PROMPT = """Listen to this audio and classify the speaker's vocal tone.

Respond with JSON only:

{
  "gender": "male|female",
  "primary_emotion": "neutral|angry|happy|sad|fearful|disgusted|surprised",
  "dominance": "submissive|neutral|assertive|commanding",
  "confidence": "uncertain|neutral|confident|overconfident",
  "aggression": "none|mild|moderate|high",
  "sarcasm": "none|subtle|obvious",
  "condescension": "none|mild|strong",
  "empathy": "cold|neutral|warm|very_warm",
  "victimhood": "none|mild|strong",
  "sincerity": "rehearsed|neutral|genuine|very_genuine"
}"""


def get_db_connection():
    """Get direct PostgreSQL connection."""
    return psycopg2.connect(
        host="10.0.0.4",
        port=5432,
        database="av_content",
        user="signal4",
        password=os.environ["POSTGRES_PASSWORD"]
    )


def get_s3_client():
    """Get S3/MinIO client."""
    return boto3.client(
        's3',
        endpoint_url="http://10.0.0.251:9000",
        aws_access_key_id=os.environ["S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["S3_SECRET_KEY"],
        region_name="us-east-1"
    )


def get_random_segments(count: int):
    """Get random segments with audio from database."""
    # Note: embedding_segments.content_id is an FK to content.id (integer)
    # We need content.content_id (varchar) for S3 paths
    query = """
        SELECT
            es.id,
            c.content_id,
            es.start_time,
            es.end_time,
            es.text,
            c.title
        FROM embedding_segments es
        JOIN content c ON c.id = es.content_id
        WHERE es.start_time IS NOT NULL
          AND es.end_time IS NOT NULL
          AND (es.end_time - es.start_time) BETWEEN 15 AND 60
          AND c.is_transcribed = true
        ORDER BY RANDOM()
        LIMIT %s
    """

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(query, (count,))
            results = cur.fetchall()
            return [
                {
                    "id": r[0],
                    "content_id": r[1],  # This is now the varchar content_id for S3
                    "start_time": float(r[2]),
                    "end_time": float(r[3]),
                    "text": r[4][:200] if r[4] else "",
                    "title": r[5]
                }
                for r in results
            ]
    finally:
        conn.close()


def download_segment_audio(content_id: str, start_time: float, end_time: float, output_path: str):
    """
    Download and extract segment audio from S3.

    Uses the same audio files as the backend media router:
    - content/{content_id}/audio.opus (preferred - compressed)
    - content/{content_id}/audio.mp3 (fallback)
    - content/{content_id}/audio.wav (fallback)
    """
    s3 = get_s3_client()
    bucket = "av-content"

    # Check for audio files in the same order as backend media.py
    audio_key = None
    for filename in ["audio.opus", "audio.mp3", "audio.wav"]:
        key = f"content/{content_id}/{filename}"
        try:
            s3.head_object(Bucket=bucket, Key=key)
            audio_key = key
            break
        except Exception:
            continue

    if not audio_key:
        raise FileNotFoundError(f"No audio found for {content_id}")

    # Generate presigned URL for ffmpeg to stream directly (no need to download full file)
    presigned_url = s3.generate_presigned_url(
        'get_object',
        Params={'Bucket': bucket, 'Key': audio_key},
        ExpiresIn=300
    )

    # Extract segment with ffmpeg using HTTP range requests (like backend does)
    duration = end_time - start_time
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_time),  # Seek BEFORE input for faster processing
        "-i", presigned_url,
        "-t", str(duration),
        "-ar", "16000",  # 16kHz for speech models
        "-ac", "1",      # mono
        "-acodec", "pcm_s16le",  # WAV output
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()}")

    return output_path


def analyze_audio(model, processor, audio_path: str, max_tokens: int = 512):
    """Run tone analysis on audio file."""
    from mlx_vlm.models.qwen3_omni_moe.omni_utils import prepare_omni_inputs

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": TONE_PROMPT},
            ],
        },
    ]

    model_inputs, text = prepare_omni_inputs(processor, conversation)

    generate_kwargs = {
        "input_ids": model_inputs["input_ids"],
        "input_features": model_inputs.get("input_features", None),
        "feature_attention_mask": model_inputs.get("feature_attention_mask", None),
        "audio_feature_lengths": model_inputs.get("audio_feature_lengths", None),
        "thinker_max_new_tokens": max_tokens,
        "return_audio": False,
    }
    generate_kwargs = {k: v for k, v in generate_kwargs.items() if v is not None}

    thinker_result, _ = model.generate(**generate_kwargs)
    output_text = processor.decode(thinker_result.sequences[0].tolist())

    return output_text


def parse_json_response(response: str) -> dict | None:
    """Extract JSON from model response."""
    # Try to find JSON in the response
    try:
        # Look for JSON block
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            json_str = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            json_str = response[start:end].strip()
        elif "{" in response:
            start = response.find("{")
            end = response.rfind("}") + 1
            json_str = response[start:end]
        else:
            return None

        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


def main():
    parser = argparse.ArgumentParser(description="Batch tone analysis on real segments")
    parser.add_argument("--count", "-c", type=int, default=20, help="Number of segments to analyze")
    parser.add_argument("--max-tokens", "-t", type=int, default=800, help="Max tokens per response")
    parser.add_argument("--output", "-o", type=str, default="tone_results.json", help="Output file")
    args = parser.parse_args()

    print(f"Fetching {args.count} random segments from database...")
    segments = get_random_segments(args.count)
    print(f"Found {len(segments)} segments")

    if not segments:
        print("No segments found!")
        return 1

    # Load model once
    print(f"\nLoading model: {MODEL_PATH}")
    from mlx_vlm import load
    start = time.time()
    model, processor = load(MODEL_PATH, trust_remote_code=True)
    print(f"Model loaded in {time.time() - start:.1f}s")

    results = []

    for i, seg in enumerate(segments):
        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(segments)}] {seg['title'][:50]}...")
        print(f"Content: {seg['content_id']} | {seg['start_time']:.1f}s - {seg['end_time']:.1f}s")
        print(f"Text preview: {seg['text'][:100]}...")

        try:
            # Download segment audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                audio_path = tmp.name

            print("Downloading audio segment...")
            download_segment_audio(
                seg['content_id'],
                seg['start_time'],
                seg['end_time'],
                audio_path
            )

            # Analyze
            print("Analyzing tone...")
            start = time.time()
            response = analyze_audio(model, processor, audio_path, args.max_tokens)
            analysis_time = time.time() - start

            # Parse JSON
            tone_data = parse_json_response(response)

            result = {
                "segment_id": seg['id'],
                "content_id": seg['content_id'],
                "title": seg['title'],
                "start_time": seg['start_time'],
                "end_time": seg['end_time'],
                "text_preview": seg['text'][:200],
                "analysis_time_s": round(analysis_time, 1),
                "tone_data": tone_data,
                "raw_response": response if tone_data is None else None
            }
            results.append(result)

            # Print summary
            if tone_data:
                print(f"✓ Analysis complete in {analysis_time:.1f}s")

                # Show non-neutral categorical values
                notable = []
                neutral_values = {"neutral", "none", "relaxed"}
                for key, val in tone_data.items():
                    if isinstance(val, str) and val.lower() not in neutral_values:
                        notable.append(f"{key}={val}")

                if notable:
                    print(f"Signals: {', '.join(notable[:6])}")
            else:
                print(f"⚠ Could not parse JSON response")
                print(f"Raw: {response[:200]}...")

            # Cleanup
            Path(audio_path).unlink(missing_ok=True)

        except Exception as e:
            print(f"✗ Error: {e}")
            results.append({
                "segment_id": seg['id'],
                "content_id": seg['content_id'],
                "error": str(e)
            })

    # Save results
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{'='*70}")
    print(f"Results saved to {output_path}")

    # Summary stats
    successful = [r for r in results if r.get("tone_data")]
    failed_parse = [r for r in results if r.get("raw_response")]
    errors = [r for r in results if r.get("error")]

    print(f"\nSummary:")
    print(f"  Successful: {len(successful)}/{len(results)}")
    print(f"  Parse failures: {len(failed_parse)}")
    print(f"  Errors: {len(errors)}")

    if successful:
        avg_time = sum(r["analysis_time_s"] for r in successful) / len(successful)
        print(f"  Avg analysis time: {avg_time:.1f}s")

    return 0


if __name__ == "__main__":
    exit(main())
