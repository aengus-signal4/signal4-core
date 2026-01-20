#!/usr/bin/env python3
"""
Analyze speaker turns for a single content item using Qwen3-Omni.

Pre-loads audio segments efficiently so the model is always running.

Usage:
    uv run python analyze_speaker_turns.py --content-id <id>
    uv run python analyze_speaker_turns.py --content-id cab2db62-04e8-4d5c-84e2-3e4254831e78
"""

import argparse
import json
import os
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from threading import Thread

import boto3
import psycopg2
from dotenv import load_dotenv

# Load environment from core/.env
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


def get_content_info(content_id: str):
    """Get content metadata and check audio exists."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, title, channel_name, publish_date
                FROM content
                WHERE content_id = %s
            """, (content_id,))
            row = cur.fetchone()
            if not row:
                raise ValueError(f"Content not found: {content_id}")
            return {
                "db_id": row[0],
                "title": row[1],
                "channel": row[2],
                "publish_date": row[3]
            }
    finally:
        conn.close()


def get_speaker_turns(content_id: str, min_duration: float = 5.0):
    """Get speaker turns > min_duration seconds."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    st.id,
                    st.speaker_id,
                    st.start_time,
                    st.end_time,
                    st.text,
                    st.turn_index,
                    s.display_name,
                    si.primary_name
                FROM speaker_transcriptions st
                JOIN content c ON c.id = st.content_id
                JOIN speakers s ON s.id = st.speaker_id
                LEFT JOIN speaker_identities si ON si.id = s.speaker_identity_id
                WHERE c.content_id = %s
                  AND (st.end_time - st.start_time) >= %s
                ORDER BY st.turn_index
            """, (content_id, min_duration))

            results = cur.fetchall()
            return [
                {
                    "id": r[0],
                    "speaker_id": r[1],
                    "start_time": float(r[2]),
                    "end_time": float(r[3]),
                    "text": r[4],
                    "turn_index": r[5],
                    "speaker_name": r[7] or r[6] or f"Speaker_{r[1]}"
                }
                for r in results
            ]
    finally:
        conn.close()


def get_audio_presigned_url(content_id: str):
    """Get presigned URL for content audio."""
    s3 = get_s3_client()
    bucket = "av-content"

    # Check for audio files
    for filename in ["audio.opus", "audio.mp3", "audio.wav"]:
        key = f"content/{content_id}/{filename}"
        try:
            s3.head_object(Bucket=bucket, Key=key)
            return s3.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket, 'Key': key},
                ExpiresIn=3600  # 1 hour
            )
        except Exception:
            continue

    raise FileNotFoundError(f"No audio found for {content_id}")


def extract_audio_segment(presigned_url: str, start_time: float, end_time: float, output_path: str):
    """Extract audio segment using ffmpeg with streaming."""
    duration = end_time - start_time
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_time),
        "-i", presigned_url,
        "-t", str(duration),
        "-ar", "16000",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()[:200]}")

    return output_path


def audio_preloader(turns: list, presigned_url: str, queue: Queue, num_ahead: int = 3):
    """Background thread to pre-load audio segments."""
    temp_dir = tempfile.mkdtemp(prefix="tone_audio_")

    for i, turn in enumerate(turns):
        output_path = os.path.join(temp_dir, f"turn_{turn['id']}.wav")

        try:
            extract_audio_segment(
                presigned_url,
                turn['start_time'],
                turn['end_time'],
                output_path
            )
            queue.put((i, turn, output_path, None))
        except Exception as e:
            queue.put((i, turn, None, str(e)))

    # Signal completion
    queue.put((None, None, None, None))


def analyze_audio(model, processor, audio_path: str, max_tokens: int = 256):
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
    try:
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
    parser = argparse.ArgumentParser(description="Analyze speaker turns for a content item")
    parser.add_argument("--content-id", "-c", type=str, required=True, help="Content ID to analyze")
    parser.add_argument("--min-duration", "-d", type=float, default=5.0, help="Min turn duration in seconds")
    parser.add_argument("--max-tokens", "-t", type=int, default=256, help="Max tokens per response")
    parser.add_argument("--output", "-o", type=str, help="Output file (default: tone_<content_id>.json)")
    args = parser.parse_args()

    content_id = args.content_id

    # Get content info
    print(f"Looking up content: {content_id}")
    content_info = get_content_info(content_id)
    print(f"Title: {content_info['title']}")
    print(f"Channel: {content_info['channel']}")

    # Get speaker turns
    print(f"\nFetching speaker turns (>= {args.min_duration}s)...")
    turns = get_speaker_turns(content_id, args.min_duration)
    print(f"Found {len(turns)} turns to analyze")

    if not turns:
        print("No turns found!")
        return 1

    # Get audio URL
    print("\nGetting audio URL...")
    presigned_url = get_audio_presigned_url(content_id)
    print("Audio URL ready")

    # Load model
    print(f"\nLoading model: {MODEL_PATH}")
    from mlx_vlm import load
    start = time.time()
    model, processor = load(MODEL_PATH, trust_remote_code=True)
    print(f"Model loaded in {time.time() - start:.1f}s")

    # Start audio preloader thread
    audio_queue = Queue(maxsize=5)  # Buffer up to 5 segments ahead
    preloader_thread = Thread(
        target=audio_preloader,
        args=(turns, presigned_url, audio_queue),
        daemon=True
    )
    preloader_thread.start()
    print("\nAudio preloader started")

    # Process turns
    results = []
    total_analysis_time = 0
    processed = 0

    print(f"\n{'='*80}")

    while True:
        # Get next pre-loaded audio
        idx, turn, audio_path, error = audio_queue.get()

        if idx is None:  # End signal
            break

        processed += 1
        duration = turn['end_time'] - turn['start_time']

        print(f"[{processed}/{len(turns)}] Turn {turn['turn_index']}: {turn['speaker_name']} "
              f"({turn['start_time']:.1f}s - {turn['end_time']:.1f}s, {duration:.1f}s)")
        print(f"  Text: {turn['text'][:80]}...")

        if error:
            print(f"  ✗ Audio error: {error}")
            results.append({
                "turn_id": turn['id'],
                "turn_index": turn['turn_index'],
                "speaker_id": turn['speaker_id'],
                "speaker_name": turn['speaker_name'],
                "start_time": turn['start_time'],
                "end_time": turn['end_time'],
                "text": turn['text'],
                "error": error
            })
            continue

        # Analyze
        try:
            start = time.time()
            response = analyze_audio(model, processor, audio_path, args.max_tokens)
            analysis_time = time.time() - start
            total_analysis_time += analysis_time

            tone_data = parse_json_response(response)

            result = {
                "turn_id": turn['id'],
                "turn_index": turn['turn_index'],
                "speaker_id": turn['speaker_id'],
                "speaker_name": turn['speaker_name'],
                "start_time": turn['start_time'],
                "end_time": turn['end_time'],
                "duration": duration,
                "text": turn['text'],
                "analysis_time_s": round(analysis_time, 2),
                "tone": tone_data,
                "raw_response": response if tone_data is None else None
            }
            results.append(result)

            if tone_data:
                # Show non-neutral values
                notable = []
                neutral_values = {"neutral", "none", "relaxed"}
                for key, val in tone_data.items():
                    if isinstance(val, str) and val.lower() not in neutral_values:
                        notable.append(f"{key}={val}")
                print(f"  ✓ {analysis_time:.1f}s | {', '.join(notable[:5]) if notable else 'all neutral'}")
            else:
                print(f"  ⚠ Parse failed: {response[:100]}...")

        except Exception as e:
            print(f"  ✗ Analysis error: {e}")
            results.append({
                "turn_id": turn['id'],
                "turn_index": turn['turn_index'],
                "speaker_id": turn['speaker_id'],
                "speaker_name": turn['speaker_name'],
                "start_time": turn['start_time'],
                "end_time": turn['end_time'],
                "text": turn['text'],
                "error": str(e)
            })

        finally:
            # Cleanup audio file
            if audio_path and os.path.exists(audio_path):
                os.unlink(audio_path)

    # Save results
    output_file = args.output or f"tone_{content_id}.json"
    output_data = {
        "content_id": content_id,
        "title": content_info['title'],
        "channel": content_info['channel'],
        "publish_date": str(content_info['publish_date']) if content_info['publish_date'] else None,
        "total_turns": len(turns),
        "analyzed_turns": len(results),
        "total_analysis_time_s": round(total_analysis_time, 1),
        "avg_analysis_time_s": round(total_analysis_time / len(results), 2) if results else 0,
        "turns": results
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"Total turns analyzed: {len(results)}")
    print(f"Successful: {len([r for r in results if r.get('tone')])}")
    print(f"Parse failures: {len([r for r in results if r.get('raw_response')])}")
    print(f"Errors: {len([r for r in results if r.get('error')])}")
    print(f"Total analysis time: {total_analysis_time:.1f}s")
    print(f"Avg time per turn: {total_analysis_time / len(results):.2f}s" if results else "")

    return 0


if __name__ == "__main__":
    exit(main())
