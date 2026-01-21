"""
Database utilities for theme classification.
"""

import logging
from typing import List, Dict, Any, Optional
from sqlalchemy import text

from src.database.session import get_session
from .data_structures import SearchCandidate
from src.classification.speaker_assignment import (
    assign_speakers_to_segment,
    format_speaker_attributed_text,
)

logger = logging.getLogger(__name__)


def reconstruct_speaker_text(text: str, speaker_positions: Optional[Dict]) -> str:
    """
    Reconstruct speaker-attributed text from speaker_positions JSONB.

    The speaker_positions format (from embedding_segments table):
    {"speaker_db_id": [[start_char, end_char], ...], ...}

    Output format:
    "Speaker A: <text slice>

    Speaker B: <text slice>"

    Args:
        text: Full segment text
        speaker_positions: Dict mapping speaker_db_id to [[start, end], ...]

    Returns:
        Formatted text with speaker labels, or original text if no speaker info
    """
    if not speaker_positions or not text:
        return text

    # Build sorted list of (start, end, speaker_id)
    turns = []
    for speaker_id, ranges in speaker_positions.items():
        if not isinstance(ranges, list):
            continue
        for range_pair in ranges:
            if isinstance(range_pair, list) and len(range_pair) >= 2:
                start, end = range_pair[0], range_pair[1]
                turns.append((int(start), int(end), str(speaker_id)))

    if not turns:
        return text

    # Sort by start position
    turns.sort(key=lambda x: x[0])

    # Assign labels (A, B, C) by first appearance
    speaker_labels = {}
    label_idx = 0
    for _, _, spk in turns:
        if spk not in speaker_labels:
            speaker_labels[spk] = chr(65 + label_idx)  # A=65 in ASCII
            label_idx += 1

    # Build formatted text
    parts = []
    for start, end, spk in turns:
        # Handle out-of-bounds gracefully
        chunk_start = max(0, start)
        chunk_end = min(len(text), end)
        chunk = text[chunk_start:chunk_end].strip()
        if chunk:
            parts.append(f"Speaker {speaker_labels[spk]}: {chunk}")

    return "\n\n".join(parts) if parts else text


def enrich_segments_with_metadata(candidates: List[SearchCandidate], batch_size: int = 5000) -> List[SearchCandidate]:
    """
    Enrich segment candidates with speaker attribution and metadata from database.

    Args:
        candidates: List of SearchCandidate objects with segment_ids
        batch_size: Number of segments to process per batch (default 5000)

    Returns:
        List of enriched SearchCandidate objects with speaker attribution
    """
    if not candidates:
        return []

    total = len(candidates)
    logger.info(f"Enriching {total} segments with speaker attribution (batch_size={batch_size})...")

    enriched = []

    # Process in batches
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_candidates = candidates[batch_start:batch_end]
        batch_segment_ids = [c.segment_id for c in batch_candidates]

        batch_num = batch_start // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size
        logger.info(f"  Batch {batch_num}/{total_batches}: segments {batch_start:,}-{batch_end:,} of {total:,}")

        with get_session() as session:
            # Query for segment metadata
            logger.debug(f"    Fetching segment metadata...")
            query = text("""
                SELECT
                    es.id as segment_id,
                    es.content_id,
                    es.text,
                    es.start_time,
                    es.end_time,
                    es.segment_index,
                    c.title as episode_title,
                    c.channel_name as episode_channel,
                    c.content_id as content_id_string
                FROM embedding_segments es
                JOIN content c ON es.content_id = c.id
                WHERE es.id = ANY(:segment_ids)
            """)

            result = session.execute(query, {'segment_ids': batch_segment_ids})
            rows = result.fetchall()
            logger.info(f"    Fetched metadata for {len(rows)} segments")

            # Create lookup by segment_id
            metadata_lookup = {}
            for row in rows:
                metadata_lookup[row.segment_id] = {
                    'content_id': row.content_id,
                    'text': row.text,
                    'start_time': row.start_time,
                    'end_time': row.end_time,
                    'segment_index': row.segment_index,
                    'episode_title': row.episode_title,
                    'episode_channel': row.episode_channel,
                    'content_id_string': row.content_id_string
                }

            # Get unique content_ids in this batch for bulk transcript fetch
            content_ids = list(set(m['content_id'] for m in metadata_lookup.values()))

            # Bulk fetch all sentences for these content_ids
            transcript_query = text("""
                SELECT
                    sent.content_id,
                    COALESCE(s.display_name, s.local_speaker_id) as speaker,
                    sent.text,
                    sent.start_time,
                    sent.end_time
                FROM sentences sent
                JOIN speakers s ON s.id = sent.speaker_id
                WHERE sent.content_id = ANY(:content_ids)
                ORDER BY sent.content_id, sent.start_time
            """)
            transcript_result = session.execute(transcript_query, {'content_ids': content_ids})

            # Group transcripts by content_id
            transcripts_by_content = {}
            for row in transcript_result:
                if row.content_id not in transcripts_by_content:
                    transcripts_by_content[row.content_id] = []
                transcripts_by_content[row.content_id].append({
                    'speaker': row.speaker,
                    'text': row.text,
                    'start_time': row.start_time,
                    'end_time': row.end_time
                })

            # Enrich each candidate in batch
            for candidate in batch_candidates:
                segment_id = candidate.segment_id

                if segment_id not in metadata_lookup:
                    logger.warning(f"Segment {segment_id} not found in database")
                    continue

                metadata = metadata_lookup[segment_id]

                # Update candidate with metadata
                candidate.content_id = metadata['content_id']
                candidate.segment_text = metadata['text']
                candidate.start_time = metadata['start_time']
                candidate.end_time = metadata['end_time']
                candidate.segment_index = metadata['segment_index']
                candidate.episode_title = metadata['episode_title']
                candidate.episode_channel = metadata['episode_channel']
                candidate.content_id_string = metadata['content_id_string']

                # Add speaker attribution
                try:
                    transcripts = transcripts_by_content.get(metadata['content_id'], [])

                    speaker_segments = assign_speakers_to_segment(
                        segment_text=candidate.segment_text,
                        transcripts=transcripts,
                        segment_start=metadata['start_time'],
                        segment_end=metadata['end_time']
                    )

                    if speaker_segments:
                        candidate.speaker_attributed_text = format_speaker_attributed_text(speaker_segments)
                        candidate.speaker_segments = speaker_segments
                    else:
                        candidate.speaker_attributed_text = candidate.segment_text
                        candidate.speaker_segments = []

                except Exception as e:
                    logger.warning(f"Failed to assign speakers for segment {segment_id}: {e}")
                    candidate.speaker_attributed_text = candidate.segment_text
                    candidate.speaker_segments = []

                enriched.append(candidate)

    logger.info(f"Successfully enriched {len(enriched)} segments")

    return enriched


def fetch_segment_metadata(segment_ids: List[int]) -> Dict[int, Dict[str, Any]]:
    """
    Fetch metadata for a list of segment IDs.

    Args:
        segment_ids: List of segment IDs to fetch

    Returns:
        Dictionary mapping segment_id to metadata dict
    """
    if not segment_ids:
        return {}

    with get_session() as session:
        query = text("""
            SELECT
                es.id as segment_id,
                es.content_id,
                es.text,
                es.start_time,
                es.end_time,
                es.segment_index,
                c.title as episode_title,
                c.channel_name as episode_channel,
                c.content_id as content_id_string
            FROM embedding_segments es
            JOIN content c ON es.content_id = c.id
            WHERE es.id = ANY(:segment_ids)
        """)

        result = session.execute(query, {'segment_ids': segment_ids})
        rows = result.fetchall()

        metadata = {}
        for row in rows:
            metadata[row.segment_id] = {
                'content_id': row.content_id,
                'text': row.text,
                'start_time': row.start_time,
                'end_time': row.end_time,
                'segment_index': row.segment_index,
                'episode_title': row.episode_title,
                'episode_channel': row.episode_channel,
                'content_id_string': row.content_id_string
            }

        return metadata
