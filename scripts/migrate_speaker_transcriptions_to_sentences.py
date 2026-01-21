#!/usr/bin/env python3
"""
Migration Script: Generate sentences from speaker_transcriptions with emotion detection

This script handles the migration from speaker_transcriptions to sentences:
1. Load speaker turns from speaker_transcriptions table
2. Split turns into sentences using NLTK with proportional timing
3. Run emotion detection (optional, can be done in batch later)
4. Save to sentences table

This is designed to run on a worker (worker5) that has GPU access for emotion detection.

Usage:
    # Dry run - show what would be processed
    python scripts/migrate_speaker_transcriptions_to_sentences.py --dry-run

    # Process specific project
    python scripts/migrate_speaker_transcriptions_to_sentences.py --project CPRMV

    # Process with emotion detection (requires GPU)
    python scripts/migrate_speaker_transcriptions_to_sentences.py --project CPRMV --with-emotion

    # Process batch of content IDs
    python scripts/migrate_speaker_transcriptions_to_sentences.py --content-ids id1 id2 id3

    # Limit to N items
    python scripts/migrate_speaker_transcriptions_to_sentences.py --limit 10

Author: Signal4 Content Processing Team
Date: 2025-12
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# Suppress noisy warnings from funasr/emotion2vec before importing
import warnings
warnings.filterwarnings('ignore', message='.*miss key in ckpt.*')
warnings.filterwarnings('ignore', message='.*trust_remote_code.*')
import logging
logging.getLogger('funasr').setLevel(logging.ERROR)
logging.getLogger('modelscope').setLevel(logging.ERROR)

import os
os.environ['FUNASR_DISABLE_PROGRESS'] = '1'  # Disable rtf progress bars

import argparse
import gzip
import io
import pickle
import time
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd
import yaml
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

from src.utils.logger import setup_worker_logger
from src.database.session import get_session
from src.database.models import Content, Sentence, Speaker, SpeakerTranscription, EmbeddingSegment
from src.storage.s3_utils import S3Storage, S3StorageConfig
from src.processing_steps.stitch_steps.util_stitch import smart_join_words

logger = setup_worker_logger('migrate_sentences')


def load_config():
    """Load configuration with env var substitution"""
    from src.utils.config import load_config as load_config_with_env
    return load_config_with_env()


def get_stitch_version(config: Dict) -> str:
    """Get current stitch version from config"""
    return config.get('processing', {}).get('stitch', {}).get('current_version', 'stitch_v14.2')


# smart_join_words is imported from src.processing_steps.stitch_steps.util_stitch


def download_word_table(s3_storage: S3Storage, content_id: str) -> Optional[pd.DataFrame]:
    """Download and decompress word_table.pkl.gz from S3

    Args:
        s3_storage: S3Storage instance
        content_id: Platform content ID (e.g. YouTube video ID)

    Returns:
        DataFrame with word-level data or None if not found
    """
    s3_key = f"content/{content_id}/word_table.pkl.gz"

    try:
        # Download from S3
        response = s3_storage._client.get_object(
            Bucket=s3_storage.config.bucket_name,
            Key=s3_key
        )
        compressed_data = response['Body'].read()

        # Decompress
        decompressed_data = gzip.decompress(compressed_data)

        # Unpickle
        word_table_df = pickle.load(io.BytesIO(decompressed_data))

        logger.info(f"[{content_id}] Downloaded word_table with {len(word_table_df)} words")
        return word_table_df

    except s3_storage._client.exceptions.NoSuchKey:
        logger.warning(f"[{content_id}] No word_table.pkl.gz found in S3")
        return None
    except Exception as e:
        # FAIL LOUDLY if file exists but can't be loaded (e.g., numpy version mismatch)
        # This prevents accidentally falling back to inferred timing
        logger.error(f"[{content_id}] FATAL: word_table.pkl.gz exists but failed to load: {e}")
        raise RuntimeError(f"word_table.pkl.gz exists but failed to load: {e}. "
                          f"This may be a numpy version mismatch - ensure numpy >= 2.0 is installed.")


def get_speaker_mappings_from_db(content_id: str, word_table_df: pd.DataFrame) -> Dict[str, Dict]:
    """Get speaker database mappings for all unique speakers in the word table

    Args:
        content_id: Platform content ID
        word_table_df: Word table DataFrame

    Returns:
        Dictionary mapping speaker names (e.g. SPEAKER_00) to database info
    """
    import hashlib

    mappings = {}

    # Get all unique speakers
    unique_speakers = set()
    for speaker in word_table_df['speaker_current'].dropna().unique():
        if speaker and speaker.startswith('SPEAKER_'):
            unique_speakers.add(speaker)

    if not unique_speakers:
        logger.warning(f"[{content_id}] No SPEAKER_* labels found in word table")
        return mappings

    logger.info(f"[{content_id}] Looking up {len(unique_speakers)} speakers in database")

    try:
        with get_session() as session:
            for speaker_name in unique_speakers:
                # Calculate speaker hash (content_id:speaker_name)
                hash_str = f"{content_id}:{speaker_name}"
                speaker_hash = hashlib.sha256(hash_str.encode()).hexdigest()[:8]

                # Look up in database
                speaker_record = session.query(Speaker).filter_by(speaker_hash=speaker_hash).first()
                if speaker_record:
                    mappings[speaker_name] = {
                        'speaker_db_id': speaker_record.id,
                        'global_id': speaker_hash,
                        'universal_name': f"Speaker {speaker_record.id}",
                        'source': 'database_lookup'
                    }
                    logger.debug(f"[{content_id}] Found speaker {speaker_name} -> db_id {speaker_record.id}")
                else:
                    logger.warning(f"[{content_id}] Speaker {speaker_name} (hash: {speaker_hash}) not found in database")

    except Exception as e:
        logger.error(f"[{content_id}] Error looking up speakers: {e}")

    logger.info(f"[{content_id}] Found {len(mappings)}/{len(unique_speakers)} speaker mappings")
    return mappings


def generate_sentences_from_speaker_transcriptions(
    content_db_id: int,
    content_id: str,
    stitch_version: str
) -> List[Dict]:
    """Generate sentences from speaker_transcriptions using proportional timing

    This is a fallback for v14 content that doesn't have word_table.pkl.gz.
    Timing is estimated proportionally based on word count within each turn.

    NOTE: Sentence timings are INFERRED/APPROXIMATE, not word-level precise.
    The original stitch step retranscribed and corrected grammar, so we cannot
    map back to raw Whisper word timestamps.

    Args:
        content_db_id: Database ID of content
        content_id: Platform content ID (for logging)
        stitch_version: Base version (will be suffixed with '_inferred')

    Returns:
        List of sentence dictionaries with proportionally estimated timing
    """
    from nltk.tokenize import word_tokenize

    sentences = []
    global_sentence_index = 0

    try:
        with get_session() as session:
            # Get all speaker_transcriptions for this content, ordered by turn
            turns = session.query(SpeakerTranscription).filter_by(
                content_id=content_db_id
            ).order_by(SpeakerTranscription.turn_index).all()

            if not turns:
                logger.warning(f"[{content_id}] No speaker_transcriptions found")
                return []

            logger.info(f"[{content_id}] Processing {len(turns)} turns from speaker_transcriptions (proportional timing)")

            for turn in turns:
                turn_text = turn.text.strip()
                if not turn_text:
                    continue

                # Split turn into sentences
                nltk_sentences = sent_tokenize(turn_text)
                if not nltk_sentences:
                    continue

                turn_start = turn.start_time
                turn_end = turn.end_time
                turn_duration = turn_end - turn_start

                # Tokenize to get word counts for proportional timing
                sentence_word_counts = []
                for sent in nltk_sentences:
                    words = word_tokenize(sent)
                    sentence_word_counts.append(len(words))

                total_words = sum(sentence_word_counts)
                if total_words == 0:
                    continue

                # Calculate proportional timing for each sentence
                current_time = turn_start
                sentence_in_turn = 0

                for sent_idx, sent in enumerate(nltk_sentences):
                    word_count = sentence_word_counts[sent_idx]

                    # Proportional duration based on word count
                    sent_duration = (word_count / total_words) * turn_duration
                    sent_start = current_time
                    sent_end = current_time + sent_duration

                    sentences.append({
                        'sentence_index': global_sentence_index,
                        'turn_index': turn.turn_index,
                        'sentence_in_turn': sentence_in_turn,
                        'speaker_id': turn.speaker_id,
                        'text': sent,
                        'start_time': sent_start,
                        'end_time': sent_end,
                        'word_count': word_count,
                        'stitch_version': f"{stitch_version}_inferred"  # Mark as inferred timing
                    })

                    global_sentence_index += 1
                    sentence_in_turn += 1
                    current_time = sent_end

            logger.info(f"[{content_id}] Generated {len(sentences)} sentences with proportional timing")

    except Exception as e:
        logger.error(f"[{content_id}] Error generating sentences from speaker_transcriptions: {e}")
        return []

    return sentences


def generate_sentences_from_word_table(
    word_table_df: pd.DataFrame,
    content_id: str,
    speaker_mappings: Dict[str, Dict],
    stitch_version: str
) -> List[Dict]:
    """Generate sentence-level data from word table with precise timing

    Args:
        word_table_df: Word table DataFrame
        content_id: Content ID for logging
        speaker_mappings: Speaker name to database ID mappings
        stitch_version: Version string for metadata

    Returns:
        List of sentence dictionaries ready for database insertion
    """
    sorted_words = word_table_df.sort_values('start')

    if len(sorted_words) == 0:
        logger.error(f"[{content_id}] No words found in word table")
        return []

    sentences = []
    global_sentence_index = 0
    turn_index = 0

    # Group words into speaker turns first
    current_speaker = None
    current_turn_words = []
    all_turns = []

    for _, word in sorted_words.iterrows():
        speaker_name = word['speaker_current']

        if speaker_name != current_speaker:
            # Save previous turn if exists
            if current_turn_words and current_speaker:
                all_turns.append({
                    'speaker_name': current_speaker,
                    'words': current_turn_words,
                    'turn_index': turn_index
                })
                turn_index += 1

            # Start new turn
            current_speaker = speaker_name
            current_turn_words = [word.to_dict()]
        else:
            current_turn_words.append(word.to_dict())

    # Add final turn
    if current_turn_words and current_speaker:
        all_turns.append({
            'speaker_name': current_speaker,
            'words': current_turn_words,
            'turn_index': turn_index
        })

    logger.info(f"[{content_id}] Processing {len(all_turns)} turns into sentences")

    # Process each turn into sentences
    for turn in all_turns:
        speaker_name = turn['speaker_name']
        words = turn['words']
        turn_idx = turn['turn_index']

        # Skip if no speaker mapping
        if speaker_name not in speaker_mappings:
            logger.debug(f"[{content_id}] Skipping turn {turn_idx} - speaker {speaker_name} has no database mapping")
            continue

        speaker_db_id = speaker_mappings[speaker_name]['speaker_db_id']

        # Detect sentence boundaries directly from word stream using punctuation
        # This matches the production approach in stage12_output.py - no NLTK text matching
        SENTENCE_ENDERS = {'.', '!', '?'}

        sentence_in_turn = 0
        current_sentence_start = 0

        for word_idx, word in enumerate(words):
            word_text = word['text'].strip()

            # Check if this word ends a sentence
            is_sentence_end = (
                word_text in SENTENCE_ENDERS or
                word_text.endswith('.') or
                word_text.endswith('!') or
                word_text.endswith('?')
            )

            # Also end sentence on last word of turn
            is_last_word = (word_idx == len(words) - 1)

            if is_sentence_end or is_last_word:
                # Create sentence from word range [current_sentence_start, word_idx]
                sentence_words = words[current_sentence_start:word_idx + 1]

                if sentence_words:
                    # Build sentence text from words
                    sent_text = smart_join_words([w['text'] for w in sentence_words])

                    # Timing is DIRECTLY from word indices - no matching needed!
                    sent_start_time = sentence_words[0]['start']
                    sent_end_time = sentence_words[-1]['end']

                    sentences.append({
                        'sentence_index': global_sentence_index,
                        'turn_index': turn_idx,
                        'sentence_in_turn': sentence_in_turn,
                        'speaker_id': speaker_db_id,
                        'text': sent_text,
                        'start_time': sent_start_time,
                        'end_time': sent_end_time,
                        'word_count': len(sentence_words),
                        'stitch_version': stitch_version
                    })
                    global_sentence_index += 1
                    sentence_in_turn += 1

                # Start next sentence after this word
                current_sentence_start = word_idx + 1

    logger.info(f"[{content_id}] Generated {len(sentences)} sentences from {len(all_turns)} turns")
    return sentences


def run_emotion_detection(
    sentences: List[Dict],
    s3_storage: S3Storage,
    content_id: str,
    model_size: str = 'large',
    backend: str = 'auto'
) -> Tuple[List[Dict], float]:
    """Run emotion detection on sentences

    Args:
        sentences: List of sentence dictionaries
        s3_storage: S3Storage for downloading audio
        content_id: Content ID
        model_size: 'base' or 'large' (for emotion2vec)
        backend: 'auto', 'wav2vec2', or 'emotion2vec'

    Returns:
        Tuple of (updated sentences with emotion fields, audio_duration_seconds)
    """
    try:
        from src.processing_steps.stitch_steps.stage13_emotion import SentenceEmotionDetector
        import librosa
    except ImportError as e:
        logger.error(f"[{content_id}] Failed to import emotion detection modules: {e}")
        return sentences, 0.0

    # Download audio
    temp_dir = Path(tempfile.mkdtemp(prefix="emotion_"))
    temp_audio_path = temp_dir / "audio.wav"
    audio_duration = 0.0

    try:
        if not s3_storage.download_audio_flexible(content_id, str(temp_audio_path)):
            logger.warning(f"[{content_id}] Could not download audio - skipping emotion detection")
            return sentences, 0.0

        # Load audio
        audio_data, sample_rate = librosa.load(str(temp_audio_path), sr=16000, mono=True)
        audio_duration = len(audio_data) / sample_rate
        logger.debug(f"[{content_id}] Loaded audio: {audio_duration:.1f}s")

        # Run emotion detection
        detector = SentenceEmotionDetector(model_size=model_size, backend=backend)
        updated_sentences = detector.process_sentences(sentences, audio_data, sample_rate, content_id)

        # Count results
        with_emotion = sum(1 for s in updated_sentences if s.get('emotion'))
        logger.debug(f"[{content_id}] Emotion detection: {with_emotion}/{len(sentences)} sentences processed")

        return updated_sentences, audio_duration

    except Exception as e:
        logger.error(f"[{content_id}] Error during emotion detection: {e}")
        return sentences, audio_duration
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def save_sentences_to_db(
    sentences: List[Dict],
    content_id: str,
    content_db_id: int,
    stitch_version: str,
    dry_run: bool = False
) -> int:
    """Save sentences to database

    Args:
        sentences: List of sentence dictionaries
        content_id: Platform content ID (for logging)
        content_db_id: Database ID of content record
        stitch_version: Stitch version string
        dry_run: If True, don't actually save

    Returns:
        Number of sentences saved
    """
    if not sentences:
        logger.warning(f"[{content_id}] No sentences to save")
        return 0

    if dry_run:
        logger.info(f"[{content_id}] DRY RUN: Would save {len(sentences)} sentences")
        return 0

    saved_count = 0

    try:
        with get_session() as session:
            # Delete existing sentences for this content
            existing_count = session.query(Sentence).filter_by(content_id=content_db_id).count()
            if existing_count > 0:
                logger.info(f"[{content_id}] Deleting {existing_count} existing sentences")
                session.query(Sentence).filter_by(content_id=content_db_id).delete()

            # Create new sentence records
            for sent in sentences:
                sentence_record = Sentence(
                    content_id=content_db_id,
                    speaker_id=sent['speaker_id'],
                    sentence_index=sent['sentence_index'],
                    turn_index=sent['turn_index'],
                    sentence_in_turn=sent['sentence_in_turn'],
                    text=sent['text'],
                    start_time=sent['start_time'],
                    end_time=sent['end_time'],
                    word_count=sent['word_count'],
                    stitch_version=sent['stitch_version'],
                    # Emotion fields (may be None if emotion detection wasn't run)
                    emotion=sent.get('emotion'),
                    emotion_confidence=sent.get('emotion_confidence'),
                    emotion_scores=sent.get('emotion_scores'),
                    arousal=sent.get('arousal'),
                    valence=sent.get('valence'),
                    dominance=sent.get('dominance')
                )
                session.add(sentence_record)
                saved_count += 1

            # Update content flags
            content = session.query(Content).filter_by(id=content_db_id).first()
            if content:
                content.is_stitched = True
                content.stitch_version = stitch_version
                content.last_updated = datetime.now(timezone.utc)

            session.commit()
            logger.info(f"[{content_id}] Saved {saved_count} sentences to database")

            # Link sentences to segments
            segments_updated = link_sentences_to_segments(content_id, content_db_id, sentences)
            if segments_updated > 0:
                logger.info(f"[{content_id}] Linked sentences to {segments_updated} segments")

    except Exception as e:
        logger.error(f"[{content_id}] Error saving sentences: {e}")
        saved_count = 0

    return saved_count


def link_sentences_to_segments(
    content_id: str,
    content_db_id: int,
    sentences: List[Dict]
) -> int:
    """Link sentences to existing embedding segments by time overlap

    Args:
        content_id: Platform content ID (for logging)
        content_db_id: Database ID of content record
        sentences: List of sentence dictionaries with timing

    Returns:
        Number of segments updated
    """
    if not sentences:
        return 0

    updated_count = 0

    try:
        with get_session() as session:
            # Get all segments for this content
            segments = session.query(EmbeddingSegment).filter_by(
                content_id=content_db_id
            ).order_by(EmbeddingSegment.segment_index).all()

            if not segments:
                logger.debug(f"[{content_id}] No segments found to link")
                return 0

            # Build sentence lookup by time for efficient matching
            # Sort sentences by start time
            sorted_sentences = sorted(sentences, key=lambda s: s['start_time'])

            for segment in segments:
                seg_start = float(segment.start_time)
                seg_end = float(segment.end_time)

                # Find all sentences that overlap with this segment
                # A sentence overlaps if: sent_start < seg_end AND sent_end > seg_start
                overlapping_sentence_ids = []

                for sent in sorted_sentences:
                    sent_start = sent['start_time']
                    sent_end = sent['end_time']

                    # Skip sentences that end before segment starts
                    if sent_end <= seg_start:
                        continue

                    # Stop if sentence starts after segment ends
                    if sent_start >= seg_end:
                        break

                    # This sentence overlaps with the segment
                    overlapping_sentence_ids.append(sent['sentence_index'])

                # Update segment with source_sentence_ids
                if overlapping_sentence_ids:
                    segment.source_sentence_ids = overlapping_sentence_ids
                    updated_count += 1

            session.commit()

    except Exception as e:
        logger.error(f"[{content_id}] Error linking sentences to segments: {e}")

    return updated_count


def get_content_to_process(
    project: Optional[str] = None,
    content_ids: Optional[List[str]] = None,
    limit: Optional[int] = None,
    only_missing_sentences: bool = True,
    skip_recent_hours: Optional[float] = None
) -> List[Tuple[int, str, str]]:
    """Get list of content to process

    Args:
        project: Filter by project name
        content_ids: Specific content IDs to process
        limit: Maximum number of items
        only_missing_sentences: Only return content without sentences
        skip_recent_hours: Skip content with sentences created within this many hours

    Returns:
        List of (db_id, content_id, title) tuples
    """
    from sqlalchemy import func

    content_list = []

    try:
        with get_session() as session:
            query = session.query(Content.id, Content.content_id, Content.title)

            # Filter by is_stitched (must have word_table in S3)
            query = query.filter(Content.is_stitched == True)

            # Filter by project if specified
            if project:
                query = query.filter(Content.projects.contains([project]))

            # Filter by specific content IDs
            if content_ids:
                query = query.filter(Content.content_id.in_(content_ids))

            # Skip content with recently created sentences
            if skip_recent_hours is not None:
                cutoff = datetime.now(timezone.utc) - pd.Timedelta(hours=skip_recent_hours)
                # Subquery: content IDs with sentences created after cutoff
                recent_subq = session.query(Sentence.content_id).filter(
                    Sentence.created_at > cutoff
                ).distinct()
                query = query.filter(~Content.id.in_(recent_subq))
                logger.info(f"Skipping content with sentences created after {cutoff}")
            elif only_missing_sentences:
                # Only get content without sentences (original behavior)
                subq = session.query(Sentence.content_id).distinct()
                query = query.filter(~Content.id.in_(subq))

            # Order by most recent first
            query = query.order_by(Content.last_updated.desc())

            # Apply limit
            if limit:
                query = query.limit(limit)

            results = query.all()
            content_list = [(r[0], r[1], r[2]) for r in results]

            logger.info(f"Found {len(content_list)} content items to process")

    except Exception as e:
        logger.error(f"Error querying content: {e}")

    return content_list


def process_single_content(
    db_id: int,
    content_id: str,
    title: str,
    s3_storage: S3Storage,
    stitch_version: str,
    with_emotion: bool = False,
    emotion_model_size: str = 'large',
    emotion_backend: str = 'auto',
    dry_run: bool = False
) -> Dict[str, Any]:
    """Process a single content item

    Args:
        db_id: Database ID
        content_id: Platform content ID
        title: Content title
        s3_storage: S3Storage instance
        stitch_version: Version string
        with_emotion: Run emotion detection
        emotion_model_size: Model size for emotion
        dry_run: Don't save to database

    Returns:
        Result dictionary with stats
    """
    result = {
        'content_id': content_id,
        'title': title,
        'status': 'pending',
        'sentences': 0,
        'with_emotion': 0,
        'audio_duration': 0.0,
        'error': None,
        'method': None  # 'word_table' or 'inferred'
    }

    start_time = time.time()

    try:
        # Step 1: Try to download word table (precise timing)
        word_table_df = download_word_table(s3_storage, content_id)

        if word_table_df is not None:
            # Have word_table - use precise timing
            result['method'] = 'word_table'

            # Get speaker mappings from word table
            speaker_mappings = get_speaker_mappings_from_db(content_id, word_table_df)

            if not speaker_mappings:
                result['status'] = 'skipped'
                result['error'] = 'No speaker mappings found'
                return result

            # Generate sentences with precise word-level timing
            sentences = generate_sentences_from_word_table(
                word_table_df, content_id, speaker_mappings, stitch_version
            )

            # Estimate audio duration from word table
            if len(word_table_df) > 0 and 'end' in word_table_df.columns:
                result['audio_duration'] = word_table_df['end'].max()

        else:
            # No word_table - fallback to speaker_transcriptions with proportional timing
            result['method'] = 'inferred'
            logger.info(f"[{content_id}] No word_table found, using speaker_transcriptions fallback (proportional timing)")

            sentences = generate_sentences_from_speaker_transcriptions(
                db_id, content_id, stitch_version
            )

            # Estimate audio duration from last sentence end time
            if sentences:
                result['audio_duration'] = max(s['end_time'] for s in sentences)

        if not sentences:
            result['status'] = 'skipped'
            result['error'] = 'No sentences generated'
            return result

        result['sentences'] = len(sentences)

        # Step 2: Run emotion detection (optional)
        if with_emotion:
            sentences, audio_duration = run_emotion_detection(
                sentences, s3_storage, content_id, emotion_model_size, emotion_backend
            )
            result['with_emotion'] = sum(1 for s in sentences if s.get('emotion'))
            if audio_duration > 0:
                result['audio_duration'] = audio_duration

        # Step 3: Save to database
        # Use the stitch_version from sentences (may be suffixed with '_inferred')
        save_version = sentences[0]['stitch_version'] if sentences else stitch_version
        saved = save_sentences_to_db(sentences, content_id, db_id, save_version, dry_run)

        if saved > 0 or dry_run:
            result['status'] = 'success'
        else:
            result['status'] = 'error'
            result['error'] = 'Failed to save to database'

    except Exception as e:
        logger.error(f"[{content_id}] Error processing: {e}", exc_info=True)
        result['status'] = 'error'
        result['error'] = str(e)

    result['duration'] = time.time() - start_time
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Migrate speaker_transcriptions to sentences with optional emotion detection'
    )
    parser.add_argument('--project', type=str, help='Filter by project name')
    parser.add_argument('--content-ids', nargs='+', help='Specific content IDs to process')
    parser.add_argument('--limit', type=int, help='Maximum number of items to process')
    parser.add_argument('--with-emotion', action='store_true', help='Run emotion detection')
    parser.add_argument('--emotion-model', default='large', choices=['base', 'large'],
                        help='Emotion model size (for emotion2vec backend)')
    parser.add_argument('--emotion-backend', default='auto', choices=['auto', 'wav2vec2', 'emotion2vec'],
                        help='Emotion detection backend: auto (use MPS/CUDA if available), wav2vec2 (faster, 4 emotions), emotion2vec (slower, 9 emotions)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done')
    parser.add_argument('--include-existing', action='store_true',
                        help='Process content that already has sentences')
    parser.add_argument('--skip-recent-hours', type=float,
                        help='Skip content with sentences created within this many hours (use with --include-existing)')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers (default: 1)')
    args = parser.parse_args()

    print("=" * 80)
    print("SENTENCE MIGRATION SCRIPT")
    print("=" * 80)
    logger.info("=" * 80)
    logger.info("SENTENCE MIGRATION SCRIPT")
    logger.info("=" * 80)

    if args.dry_run:
        print("*** DRY RUN MODE - NO CHANGES WILL BE MADE ***")
        logger.info("*** DRY RUN MODE - NO CHANGES WILL BE MADE ***")

    # Load config
    config = load_config()
    stitch_version = get_stitch_version(config)
    print(f"Using stitch version: {stitch_version}")
    logger.info(f"Using stitch version: {stitch_version}")

    # Initialize S3
    s3_config = S3StorageConfig.from_dict(config['storage']['s3'])
    s3_storage = S3Storage(s3_config)

    # Get content to process
    content_list = get_content_to_process(
        project=args.project,
        content_ids=args.content_ids,
        limit=args.limit,
        only_missing_sentences=not args.include_existing,
        skip_recent_hours=args.skip_recent_hours
    )

    if not content_list:
        print("No content to process")
        logger.info("No content to process")
        return 0

    print(f"\nWill process {len(content_list)} content items")
    logger.info(f"\nWill process {len(content_list)} content items")
    if args.with_emotion:
        print(f"Emotion detection: ENABLED (backend: {args.emotion_backend}, model: {args.emotion_model})")
    else:
        print("Emotion detection: DISABLED (run with --with-emotion to enable)")
    if args.workers > 1:
        print(f"Parallel workers: {args.workers}")

    # Process each content item
    results = {
        'success': 0,
        'skipped': 0,
        'error': 0,
        'total_sentences': 0,
        'total_with_emotion': 0,
        'word_table_count': 0,  # Precise timing from word_table
        'inferred_count': 0     # Proportional timing from speaker_transcriptions
    }

    def process_item(item):
        """Wrapper for parallel processing"""
        db_id, content_id, title = item
        return process_single_content(
            db_id=db_id,
            content_id=content_id,
            title=title or 'Unknown',
            s3_storage=s3_storage,
            stitch_version=stitch_version,
            with_emotion=args.with_emotion,
            emotion_model_size=args.emotion_model,
            emotion_backend=args.emotion_backend,
            dry_run=args.dry_run
        )

    def print_result(result):
        """Print result summary line"""
        duration = result.get('duration', 0)
        audio_duration = result.get('audio_duration', 0)
        rtfx = audio_duration / duration if duration > 0 else 0
        status_icon = '✓' if result['status'] == 'success' else ('○' if result['status'] == 'skipped' else '✗')
        audio_mins = audio_duration / 60 if audio_duration else 0
        content_id = result.get('content_id', 'unknown')
        method = result.get('method', '')
        method_indicator = '[W]' if method == 'word_table' else ('[I]' if method == 'inferred' else '   ')
        tqdm.write(f"{status_icon} {method_indicator} {content_id[:22]:22} | {audio_mins:5.1f}m | {duration:5.1f}s | RTFx: {rtfx:5.1f}")

    # Use tqdm for progress bar
    pbar = tqdm(total=len(content_list), desc="Migrating content", unit="item")

    if args.workers > 1:
        # Parallel processing with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_item = {executor.submit(process_item, item): item for item in content_list}

            for future in as_completed(future_to_item):
                result = future.result()

                results[result['status']] += 1
                results['total_sentences'] += result.get('sentences', 0)
                results['total_with_emotion'] += result.get('with_emotion', 0)
                if result.get('method') == 'word_table':
                    results['word_table_count'] += 1
                elif result.get('method') == 'inferred':
                    results['inferred_count'] += 1

                print_result(result)
                pbar.update(1)
                pbar.set_description(f"ok:{results['success']} skip:{results['skipped']} err:{results['error']}")
    else:
        # Sequential processing
        for db_id, content_id, title in content_list:
            result = process_item((db_id, content_id, title))

            results[result['status']] += 1
            results['total_sentences'] += result.get('sentences', 0)
            results['total_with_emotion'] += result.get('with_emotion', 0)
            if result.get('method') == 'word_table':
                results['word_table_count'] += 1
            elif result.get('method') == 'inferred':
                results['inferred_count'] += 1

            print_result(result)
            pbar.update(1)
            pbar.set_description(f"ok:{results['success']} skip:{results['skipped']} err:{results['error']}")

    pbar.close()

    # Summary
    print("\n" + "=" * 80)
    print("MIGRATION SUMMARY")
    print("=" * 80)
    print(f"Total processed: {len(content_list)}")
    print(f"  Success: {results['success']}")
    print(f"    - Word table (precise timing) [W]: {results['word_table_count']}")
    print(f"    - Inferred (proportional timing) [I]: {results['inferred_count']}")
    print(f"  Skipped: {results['skipped']}")
    print(f"  Error: {results['error']}")
    print(f"Total sentences: {results['total_sentences']}")
    if args.with_emotion:
        print(f"Sentences with emotion: {results['total_with_emotion']}")
    if results['inferred_count'] > 0:
        print(f"\nNOTE: {results['inferred_count']} items used inferred timing (stitch_version ends with '_inferred').")
        print("      These have proportional sentence timing, not word-level precision.")

    logger.info("=" * 80)
    logger.info("MIGRATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total processed: {len(content_list)}")
    logger.info(f"  Success: {results['success']} (word_table: {results['word_table_count']}, inferred: {results['inferred_count']})")
    logger.info(f"  Skipped: {results['skipped']}")
    logger.info(f"  Error: {results['error']}")
    logger.info(f"Total sentences: {results['total_sentences']}")
    if args.with_emotion:
        logger.info(f"Sentences with emotion: {results['total_with_emotion']}")

    if args.dry_run:
        print("\n*** This was a dry run - no changes were made ***")
        logger.info("*** This was a dry run - no changes were made ***")

    return 0 if results['error'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
