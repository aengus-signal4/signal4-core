#!/usr/bin/env python3
"""
Run tone analysis on Canadian project content.

Prioritizes recent content, works backwards through history.
At 7x RT, we can process ~168 hours/day (192 hrs with margin).

Usage:
    uv run python run_canadian_tones.py
    uv run python run_canadian_tones.py --limit 100  # Process 100 content items
    uv run python run_canadian_tones.py --dry-run    # Don't update database
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
from dataclasses import dataclass
from typing import List, Dict, Optional

import boto3
import psycopg2
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment from core/.env
CORE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(CORE_DIR / ".env", override=True)

MODEL_PATH = "./mlx_qwen3_omni_4bit"
MIN_WORDS = 5
PROJECT = "Canadian"

# Tone analysis prompt
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

DOMINANCE_MAP = {"submissive": 0.2, "neutral": 0.5, "assertive": 0.7, "commanding": 0.9}
AROUSAL_MAP = {"neutral": 0.3, "angry": 0.9, "happy": 0.7, "sad": 0.3, "fearful": 0.8, "disgusted": 0.6, "surprised": 0.8}
VALENCE_MAP = {"neutral": 0.5, "angry": 0.2, "happy": 0.9, "sad": 0.2, "fearful": 0.3, "disgusted": 0.2, "surprised": 0.6}


@dataclass
class AnalysisChunk:
    chunk_id: str
    content_id: str
    start_time: float
    end_time: float
    sentence_ids: List[int]
    chunk_type: str
    speaker_hash: Optional[str] = None


def get_db_connection():
    return psycopg2.connect(
        host="10.0.0.4",
        port=5432,
        database="av_content",
        user="signal4",
        password=os.environ["POSTGRES_PASSWORD"]
    )


def get_s3_client():
    return boto3.client(
        's3',
        endpoint_url="http://10.0.0.251:9000",
        aws_access_key_id=os.environ["S3_ACCESS_KEY"],
        aws_secret_access_key=os.environ["S3_SECRET_KEY"],
        region_name="us-east-1"
    )


def get_canadian_content_needing_tone(limit: int = None) -> List[Dict]:
    """Get Canadian content with sentences needing tone, ordered by publish_date DESC."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            query = """
                SELECT c.content_id, c.title, c.channel_name, c.publish_date,
                       c.duration,
                       COUNT(s.id) as sentences_needing_tone
                FROM content c
                JOIN sentences s ON s.content_id = c.id
                WHERE %s = ANY(c.projects)
                  AND s.emotion IS NULL
                  AND s.word_count >= %s
                  AND c.is_transcribed = true
                GROUP BY c.content_id, c.title, c.channel_name, c.publish_date, c.duration
                HAVING COUNT(s.id) > 0
                ORDER BY c.publish_date DESC NULLS LAST
            """
            if limit:
                query += f" LIMIT {limit}"

            cur.execute(query, (PROJECT, MIN_WORDS))
            return [
                {
                    "content_id": r[0],
                    "title": r[1],
                    "channel": r[2],
                    "publish_date": r[3],
                    "duration": r[4],
                    "sentences_needing": r[5]
                }
                for r in cur.fetchall()
            ]
    finally:
        conn.close()


def get_analysis_chunks(content_id: str) -> List[AnalysisChunk]:
    """Build analysis chunks - single-speaker segments or speaker turns."""
    conn = get_db_connection()
    chunks = []

    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM content WHERE content_id = %s", (content_id,))
            row = cur.fetchone()
            if not row:
                return []
            db_content_id = row[0]

            cur.execute("""
                SELECT id, segment_index, start_time, end_time,
                       source_speaker_hashes, source_sentence_ids
                FROM embedding_segments
                WHERE content_id = %s
                  AND source_sentence_ids IS NOT NULL
                ORDER BY segment_index
            """, (db_content_id,))
            segments = cur.fetchall()

            for seg_id, seg_idx, start_time, end_time, speaker_hashes, sentence_indices in segments:
                if not sentence_indices:
                    continue

                num_speakers = len(speaker_hashes) if speaker_hashes else 0

                if num_speakers == 1:
                    # source_sentence_ids contains sentence_index values, not IDs
                    # Need to look up actual sentence IDs by sentence_index
                    cur.execute("""
                        SELECT id FROM sentences
                        WHERE content_id = %s
                          AND sentence_index = ANY(%s)
                          AND emotion IS NULL
                          AND word_count >= %s
                    """, (db_content_id, sentence_indices, MIN_WORDS))
                    valid_sentence_ids = [r[0] for r in cur.fetchall()]

                    if valid_sentence_ids:
                        chunks.append(AnalysisChunk(
                            chunk_id=f"seg_{seg_id}",
                            content_id=content_id,
                            start_time=start_time,
                            end_time=end_time,
                            sentence_ids=valid_sentence_ids,
                            chunk_type="segment",
                            speaker_hash=speaker_hashes[0] if speaker_hashes else None
                        ))
                else:
                    cur.execute("""
                        SELECT st.id, st.speaker_id, st.start_time, st.end_time, st.speaker_hash
                        FROM speaker_transcriptions st
                        WHERE st.content_id = %s
                          AND st.start_time >= %s
                          AND st.end_time <= %s
                        ORDER BY st.start_time
                    """, (db_content_id, start_time, end_time))
                    turns = cur.fetchall()

                    for turn_id, speaker_id, turn_start, turn_end, spk_hash in turns:
                        cur.execute("""
                            SELECT id FROM sentences
                            WHERE content_id = %s
                              AND start_time >= %s
                              AND end_time <= %s
                              AND emotion IS NULL
                              AND word_count >= %s
                        """, (db_content_id, turn_start, turn_end, MIN_WORDS))
                        turn_sentence_ids = [r[0] for r in cur.fetchall()]

                        if turn_sentence_ids:
                            chunks.append(AnalysisChunk(
                                chunk_id=f"turn_{turn_id}",
                                content_id=content_id,
                                start_time=turn_start,
                                end_time=turn_end,
                                sentence_ids=turn_sentence_ids,
                                chunk_type="turn",
                                speaker_hash=spk_hash
                            ))

            return chunks
    finally:
        conn.close()


def get_audio_presigned_url(content_id: str) -> str:
    s3 = get_s3_client()
    bucket = "av-content"

    for filename in ["audio.opus", "audio.mp3", "audio.wav"]:
        key = f"content/{content_id}/{filename}"
        try:
            s3.head_object(Bucket=bucket, Key=key)
            return s3.generate_presigned_url(
                'get_object', Params={'Bucket': bucket, 'Key': key}, ExpiresIn=3600
            )
        except Exception:
            continue
    raise FileNotFoundError(f"No audio found for {content_id}")


def extract_audio_segment(presigned_url: str, start_time: float, end_time: float, output_path: str):
    duration = end_time - start_time
    cmd = [
        "ffmpeg", "-y", "-ss", str(start_time), "-i", presigned_url,
        "-t", str(duration), "-ar", "16000", "-ac", "1", "-acodec", "pcm_s16le",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode()[:200]}")
    return output_path


def audio_preloader(chunks: List[AnalysisChunk], presigned_url: str, queue: Queue):
    temp_dir = tempfile.mkdtemp(prefix="tone_audio_")
    for i, chunk in enumerate(chunks):
        output_path = os.path.join(temp_dir, f"{chunk.chunk_id}.wav")
        try:
            extract_audio_segment(presigned_url, chunk.start_time, chunk.end_time, output_path)
            queue.put((i, chunk, output_path, None))
        except Exception as e:
            queue.put((i, chunk, None, str(e)))
    queue.put((None, None, None, None))


def analyze_audio(model, processor, audio_path: str, max_tokens: int = 256) -> str:
    from mlx_vlm.models.qwen3_omni_moe.omni_utils import prepare_omni_inputs

    conversation = [
        {"role": "user", "content": [
            {"type": "audio", "audio": audio_path},
            {"type": "text", "text": TONE_PROMPT},
        ]},
    ]

    model_inputs, _ = prepare_omni_inputs(processor, conversation)
    generate_kwargs = {
        "input_ids": model_inputs["input_ids"],
        "input_features": model_inputs.get("input_features"),
        "feature_attention_mask": model_inputs.get("feature_attention_mask"),
        "audio_feature_lengths": model_inputs.get("audio_feature_lengths"),
        "thinker_max_new_tokens": max_tokens,
        "return_audio": False,
    }
    generate_kwargs = {k: v for k, v in generate_kwargs.items() if v is not None}

    thinker_result, _ = model.generate(**generate_kwargs)
    return processor.decode(thinker_result.sequences[0].tolist())


def parse_json_response(response: str) -> Optional[Dict]:
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


def update_sentences(sentence_ids: List[int], tone_data: Dict):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            emotion = tone_data.get('primary_emotion', 'neutral')
            emotion_confidence = 0.8 if emotion != 'neutral' else 0.5

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

            arousal = AROUSAL_MAP.get(emotion, 0.5)
            valence = VALENCE_MAP.get(emotion, 0.5)
            dominance = DOMINANCE_MAP.get(tone_data.get('dominance', 'neutral'), 0.5)

            aggression = tone_data.get('aggression', 'none')
            if aggression == 'high':
                arousal = min(1.0, arousal + 0.2)
            elif aggression == 'moderate':
                arousal = min(1.0, arousal + 0.1)

            cur.execute("""
                UPDATE sentences
                SET emotion = %s, emotion_confidence = %s, emotion_scores = %s,
                    arousal = %s, valence = %s, dominance = %s
                WHERE id = ANY(%s)
            """, (emotion, emotion_confidence, json.dumps(emotion_scores),
                  arousal, valence, dominance, sentence_ids))
            conn.commit()
            return cur.rowcount
    finally:
        conn.close()


def process_content(content_id: str, model, processor, dry_run: bool = False, pbar_chunks=None) -> Dict:
    """Process a single content item."""
    chunks = get_analysis_chunks(content_id)
    if not chunks:
        return {"status": "no_chunks", "chunks": 0, "sentences": 0, "audio_s": 0, "analysis_s": 0}

    try:
        presigned_url = get_audio_presigned_url(content_id)
    except FileNotFoundError:
        return {"status": "no_audio", "chunks": len(chunks), "sentences": 0, "audio_s": 0, "analysis_s": 0}

    audio_queue = Queue(maxsize=5)
    preloader = Thread(target=audio_preloader, args=(chunks, presigned_url, audio_queue), daemon=True)
    preloader.start()

    total_analysis_time = 0
    total_audio_duration = 0
    sentences_updated = 0
    successful_chunks = 0

    while True:
        idx, chunk, audio_path, error = audio_queue.get()

        if idx is None:
            break

        duration = chunk.end_time - chunk.start_time
        total_audio_duration += duration

        if error:
            if pbar_chunks:
                pbar_chunks.update(1)
            continue

        try:
            t0 = time.time()
            response = analyze_audio(model, processor, audio_path)
            analysis_time = time.time() - t0
            total_analysis_time += analysis_time

            tone_data = parse_json_response(response)

            if tone_data:
                successful_chunks += 1
                if not dry_run:
                    updated = update_sentences(chunk.sentence_ids, tone_data)
                    sentences_updated += updated
                else:
                    sentences_updated += len(chunk.sentence_ids)

        except Exception:
            pass

        finally:
            if audio_path and os.path.exists(audio_path):
                os.unlink(audio_path)
            if pbar_chunks:
                pbar_chunks.update(1)

    return {
        "status": "success",
        "chunks": len(chunks),
        "successful": successful_chunks,
        "sentences": sentences_updated,
        "audio_s": total_audio_duration,
        "analysis_s": total_analysis_time,
    }


def main():
    parser = argparse.ArgumentParser(description=f"Run tone analysis on {PROJECT} content")
    parser.add_argument("--limit", "-l", type=int, help="Max content items to process")
    parser.add_argument("--dry-run", action="store_true", help="Don't update database")
    args = parser.parse_args()

    # Load model
    print(f"Loading model: {MODEL_PATH}")
    from mlx_vlm import load
    model, processor = load(MODEL_PATH, trust_remote_code=True)
    print("Model loaded\n")

    # Get content needing tone
    print(f"Finding {PROJECT} content needing tone analysis...")
    content_list = get_canadian_content_needing_tone(args.limit)

    total_sentences = sum(c['sentences_needing'] for c in content_list)
    total_duration = sum(c['duration'] or 0 for c in content_list)

    print(f"Found {len(content_list)} content items")
    print(f"Total sentences needing tone: {total_sentences:,}")
    print(f"Total audio duration: {total_duration/3600:.1f} hours")
    print(f"Estimated processing time at 7x RT: {total_duration/3600/7:.1f} hours")
    print()

    # Stats tracking
    stats = {
        "processed": 0,
        "sentences_updated": 0,
        "audio_seconds": 0,
        "analysis_seconds": 0,
        "errors": 0,
    }

    # Process with progress bar
    with tqdm(content_list, desc="Content", unit="item") as pbar_content:
        for item in pbar_content:
            title_short = item['title'][:30] if item['title'] else "Unknown"
            pbar_content.set_postfix_str(f"{title_short}...")

            result = process_content(item['content_id'], model, processor, args.dry_run)

            if result['status'] == 'success':
                stats['processed'] += 1
                stats['sentences_updated'] += result['sentences']
                stats['audio_seconds'] += result['audio_s']
                stats['analysis_seconds'] += result['analysis_s']
            else:
                stats['errors'] += 1

            # Update progress bar with running stats
            if stats['audio_seconds'] > 0:
                rt = stats['analysis_seconds'] / stats['audio_seconds']
                pbar_content.set_postfix({
                    'sentences': stats['sentences_updated'],
                    'RT': f"{rt:.2f}x",
                })

    # Final summary
    print("\n" + "="*60)
    print(f"COMPLETED - {PROJECT} Tone Analysis")
    print("="*60)
    print(f"Content processed: {stats['processed']}")
    print(f"Errors/skipped: {stats['errors']}")
    print(f"Sentences updated: {stats['sentences_updated']:,}")
    print(f"Audio processed: {stats['audio_seconds']/3600:.2f} hours")
    print(f"Analysis time: {stats['analysis_seconds']/3600:.2f} hours")
    if stats['audio_seconds'] > 0:
        rt = stats['analysis_seconds'] / stats['audio_seconds']
        speed = stats['audio_seconds'] / stats['analysis_seconds']
        print(f"RT factor: {rt:.3f}x ({speed:.1f}x real-time)")
    if args.dry_run:
        print("\n[DRY RUN - no database updates made]")


if __name__ == "__main__":
    main()
