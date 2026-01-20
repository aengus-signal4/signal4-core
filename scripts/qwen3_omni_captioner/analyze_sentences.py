#!/usr/bin/env python3
"""
Analyze sentences for a single content item using Qwen3-Omni.

Pre-loads audio segments efficiently so the model is always running.
Updates the sentences table with tone analysis results.

Usage:
    uv run python analyze_sentences.py --content-id <id>
    uv run python analyze_sentences.py --content-id cab2db62-04e8-4d5c-84e2-3e4254831e78
"""

import argparse
import json
import os
import subprocess
import tempfile
import time
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

# Map our tone categories to emotion scores (0-1)
EMOTION_MAP = {
    "neutral": 0.0,
    "angry": 1.0,
    "happy": 1.0,
    "sad": 1.0,
    "fearful": 1.0,
    "disgusted": 1.0,
    "surprised": 1.0,
}

# Map categorical values to numeric scores for arousal/valence/dominance
DOMINANCE_MAP = {
    "submissive": 0.2,
    "neutral": 0.5,
    "assertive": 0.7,
    "commanding": 0.9,
}

AROUSAL_MAP = {
    # Based on emotion + aggression
    "neutral": 0.3,
    "angry": 0.9,
    "happy": 0.7,
    "sad": 0.3,
    "fearful": 0.8,
    "disgusted": 0.6,
    "surprised": 0.8,
}

VALENCE_MAP = {
    "neutral": 0.5,
    "angry": 0.2,
    "happy": 0.9,
    "sad": 0.2,
    "fearful": 0.3,
    "disgusted": 0.2,
    "surprised": 0.6,
}


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
    """Get content metadata."""
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


def get_sentences(content_id: str, min_words: int = 3):
    """Get sentences with >= min_words that don't have emotion yet."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    s.id,
                    s.speaker_id,
                    s.sentence_index,
                    s.turn_index,
                    s.start_time,
                    s.end_time,
                    s.text,
                    s.word_count,
                    sp.display_name,
                    si.primary_name
                FROM sentences s
                JOIN content c ON c.id = s.content_id
                JOIN speakers sp ON sp.id = s.speaker_id
                LEFT JOIN speaker_identities si ON si.id = sp.speaker_identity_id
                WHERE c.content_id = %s
                  AND s.word_count >= %s
                  AND s.emotion IS NULL
                ORDER BY s.sentence_index
            """, (content_id, min_words))

            results = cur.fetchall()
            return [
                {
                    "id": r[0],
                    "speaker_id": r[1],
                    "sentence_index": r[2],
                    "turn_index": r[3],
                    "start_time": float(r[4]),
                    "end_time": float(r[5]),
                    "text": r[6],
                    "word_count": r[7],
                    "speaker_name": r[9] or r[8] or f"Speaker_{r[1]}"
                }
                for r in results
            ]
    finally:
        conn.close()


def get_audio_presigned_url(content_id: str):
    """Get presigned URL for content audio."""
    s3 = get_s3_client()
    bucket = "av-content"

    for filename in ["audio.opus", "audio.mp3", "audio.wav"]:
        key = f"content/{content_id}/{filename}"
        try:
            s3.head_object(Bucket=bucket, Key=key)
            return s3.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket, 'Key': key},
                ExpiresIn=3600
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


def audio_preloader(sentences: list, presigned_url: str, queue: Queue):
    """Background thread to pre-load audio segments."""
    temp_dir = tempfile.mkdtemp(prefix="tone_audio_")

    for i, sentence in enumerate(sentences):
        output_path = os.path.join(temp_dir, f"sentence_{sentence['id']}.wav")

        try:
            extract_audio_segment(
                presigned_url,
                sentence['start_time'],
                sentence['end_time'],
                output_path
            )
            queue.put((i, sentence, output_path, None))
        except Exception as e:
            queue.put((i, sentence, None, str(e)))

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


def update_sentence_emotion(sentence_id: int, tone_data: dict):
    """Update sentence with emotion data."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Map tone data to sentence emotion columns
            emotion = tone_data.get('primary_emotion', 'neutral')
            emotion_confidence = 0.8 if emotion != 'neutral' else 0.5  # Model doesn't give confidence

            # Build emotion_scores from all tone data
            emotion_scores = {
                "primary_emotion": emotion,
                "dominance_cat": tone_data.get('dominance', 'neutral'),
                "confidence_cat": tone_data.get('confidence', 'neutral'),
                "aggression": tone_data.get('aggression', 'none'),
                "sarcasm": tone_data.get('sarcasm', 'none'),
                "condescension": tone_data.get('condescension', 'none'),
                "empathy": tone_data.get('empathy', 'neutral'),
                "victimhood": tone_data.get('victimhood', 'none'),
                "sincerity": tone_data.get('sincerity', 'neutral'),
                "gender": tone_data.get('gender', 'unknown'),
            }

            # Calculate dimensional scores
            arousal = AROUSAL_MAP.get(emotion, 0.5)
            valence = VALENCE_MAP.get(emotion, 0.5)
            dominance = DOMINANCE_MAP.get(tone_data.get('dominance', 'neutral'), 0.5)

            # Adjust arousal based on aggression
            aggression = tone_data.get('aggression', 'none')
            if aggression == 'high':
                arousal = min(1.0, arousal + 0.2)
            elif aggression == 'moderate':
                arousal = min(1.0, arousal + 0.1)

            cur.execute("""
                UPDATE sentences
                SET emotion = %s,
                    emotion_confidence = %s,
                    emotion_scores = %s,
                    arousal = %s,
                    valence = %s,
                    dominance = %s
                WHERE id = %s
            """, (
                emotion,
                emotion_confidence,
                json.dumps(emotion_scores),
                arousal,
                valence,
                dominance,
                sentence_id
            ))
            conn.commit()
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze sentences for a content item")
    parser.add_argument("--content-id", "-c", type=str, required=True, help="Content ID to analyze")
    parser.add_argument("--min-words", "-w", type=int, default=3, help="Min words per sentence")
    parser.add_argument("--max-tokens", "-t", type=int, default=256, help="Max tokens per response")
    parser.add_argument("--dry-run", action="store_true", help="Don't update database")
    parser.add_argument("--output", "-o", type=str, help="Also save results to JSON file")
    args = parser.parse_args()

    content_id = args.content_id

    # Get content info
    print(f"Looking up content: {content_id}")
    content_info = get_content_info(content_id)
    print(f"Title: {content_info['title']}")
    print(f"Channel: {content_info['channel']}")

    # Get sentences
    print(f"\nFetching sentences (>= {args.min_words} words, no emotion yet)...")
    sentences = get_sentences(content_id, args.min_words)
    print(f"Found {len(sentences)} sentences to analyze")

    if not sentences:
        print("No sentences to analyze!")
        return 0

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
    audio_queue = Queue(maxsize=5)
    preloader_thread = Thread(
        target=audio_preloader,
        args=(sentences, presigned_url, audio_queue),
        daemon=True
    )
    preloader_thread.start()
    print("\nAudio preloader started")

    # Process sentences
    results = []
    total_analysis_time = 0
    total_audio_duration = 0
    processed = 0
    successful = 0
    db_updated = 0

    print(f"\n{'='*80}")

    while True:
        idx, sentence, audio_path, error = audio_queue.get()

        if idx is None:
            break

        processed += 1
        duration = sentence['end_time'] - sentence['start_time']
        total_audio_duration += duration

        print(f"[{processed}/{len(sentences)}] Sentence {sentence['sentence_index']} "
              f"(turn {sentence['turn_index']}): {sentence['speaker_name']} "
              f"({duration:.1f}s, {sentence['word_count']} words)")
        print(f"  Text: {sentence['text'][:60]}...")

        if error:
            print(f"  X Audio error: {error}")
            results.append({"sentence_id": sentence['id'], "error": error})
            continue

        try:
            start = time.time()
            response = analyze_audio(model, processor, audio_path, args.max_tokens)
            analysis_time = time.time() - start
            total_analysis_time += analysis_time

            tone_data = parse_json_response(response)

            if tone_data:
                successful += 1

                # Update database unless dry-run
                if not args.dry_run:
                    update_sentence_emotion(sentence['id'], tone_data)
                    db_updated += 1

                # Show non-neutral values
                notable = []
                neutral_values = {"neutral", "none", "relaxed"}
                for key, val in tone_data.items():
                    if isinstance(val, str) and val.lower() not in neutral_values:
                        notable.append(f"{key}={val}")
                print(f"  OK {analysis_time:.1f}s | {', '.join(notable[:5]) if notable else 'all neutral'}")

                results.append({
                    "sentence_id": sentence['id'],
                    "sentence_index": sentence['sentence_index'],
                    "turn_index": sentence['turn_index'],
                    "speaker_name": sentence['speaker_name'],
                    "text": sentence['text'],
                    "duration": duration,
                    "tone": tone_data,
                    "analysis_time_s": round(analysis_time, 2)
                })
            else:
                print(f"  ?? Parse failed: {response[:80]}...")
                results.append({
                    "sentence_id": sentence['id'],
                    "raw_response": response
                })

        except Exception as e:
            print(f"  X Analysis error: {e}")
            results.append({"sentence_id": sentence['id'], "error": str(e)})

        finally:
            if audio_path and os.path.exists(audio_path):
                os.unlink(audio_path)

    # Save results if requested
    if args.output:
        output_data = {
            "content_id": content_id,
            "title": content_info['title'],
            "total_sentences": len(sentences),
            "successful": successful,
            "db_updated": db_updated,
            "total_analysis_time_s": round(total_analysis_time, 1),
            "total_audio_duration_s": round(total_audio_duration, 1),
            "rt_factor": round(total_analysis_time / total_audio_duration, 3) if total_audio_duration > 0 else 0,
            "results": results
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    # Summary
    print(f"\n{'='*80}")
    print(f"Total sentences: {len(sentences)}")
    print(f"Successful: {successful}")
    print(f"DB updated: {db_updated}" + (" (dry-run)" if args.dry_run else ""))
    print(f"Total audio: {total_audio_duration:.1f}s ({total_audio_duration/60:.1f} mins)")
    print(f"Total analysis: {total_analysis_time:.1f}s ({total_analysis_time/60:.1f} mins)")
    if total_audio_duration > 0:
        print(f"RT factor: {total_analysis_time / total_audio_duration:.3f}x ({total_audio_duration / total_analysis_time:.1f}x real-time)")

    return 0


if __name__ == "__main__":
    exit(main())
