#!/usr/bin/env python3
"""
Sample CPRMV segments for semantic query evaluation.

This script runs semantic queries from queries_dave.json against the cprmv_analysis
table using 4B embeddings, and generates balanced samples for manual evaluation.

Features:
- Runs 16 semantic queries (en/fr pairs) over cprmv_analysis table
- Generates 50 samples per query (800 total)
- Balances samples by language and channel to ensure diversity
- Avoids duplicate segments across queries
- Outputs CSV with segment metadata for evaluation

Usage:
    # Generate 50 samples per query (800 total)
    python src/classification/sample_cprmv_queries.py

    # Custom sample size per query
    python src/classification/sample_cprmv_queries.py --samples-per-query 30

    # Use French queries
    python src/classification/sample_cprmv_queries.py --language fr

    # Dry run to see query results without sampling
    python src/classification/sample_cprmv_queries.py --dry-run

    # Custom queries file
    python src/classification/sample_cprmv_queries.py --queries-file custom_queries.json
"""

import sys
import os
import json
import argparse
import csv
import numpy as np
import pickle
from pathlib import Path
from src.utils.paths import get_project_root, get_config_path
from typing import List, Dict, Set, Tuple
from datetime import datetime
from collections import defaultdict
import logging
import hashlib

# Add project root to path
sys.path.append(str(get_project_root()))

from src.database.session import get_session
from src.database.models import CprmvAnalysis, EmbeddingSegment, Content
from sentence_transformers import SentenceTransformer
from sqlalchemy import text
from src.utils.logger import setup_worker_logger

logger = setup_worker_logger('sample_cprmv_queries')


def load_queries(queries_file: str) -> List[Dict]:
    """Load semantic queries from JSON file."""
    with open(queries_file, 'r', encoding='utf-8') as f:
        queries = json.load(f)
    logger.info(f"Loaded {len(queries)} queries from {queries_file}")
    return queries


def get_cache_key(query_text: str, model_name: str) -> str:
    """Generate cache key for query embedding."""
    text_hash = hashlib.md5(query_text.encode('utf-8')).hexdigest()
    model_hash = hashlib.md5(model_name.encode('utf-8')).hexdigest()[:8]
    return f"{model_hash}_{text_hash}"


def get_cache_path(queries_file: str, model_name: str) -> Path:
    """Get cache file path for query embeddings."""
    queries_hash = hashlib.md5(Path(queries_file).read_bytes()).hexdigest()[:8]
    model_hash = hashlib.md5(model_name.encode('utf-8')).hexdigest()[:8]
    cache_dir = Path('projects/CPRMV/.cache')
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"query_embeddings_{queries_hash}_{model_hash}.pkl"


def load_or_generate_query_embeddings(
    queries: List[Dict],
    model: SentenceTransformer,
    queries_file: str,
    model_name: str
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Load cached query embeddings or generate and cache them.

    Returns:
        Dict mapping query_id to (embedding_en, embedding_fr) tuple
    """
    cache_path = get_cache_path(queries_file, model_name)

    # Try to load from cache
    if cache_path.exists():
        logger.info(f"Loading cached query embeddings from {cache_path}")
        with open(cache_path, 'rb') as f:
            cached_embeddings = pickle.load(f)

        # Verify all queries are present
        if all(q['id'] in cached_embeddings for q in queries):
            logger.info(f"Loaded {len(cached_embeddings)} cached query embeddings")
            return cached_embeddings
        else:
            logger.warning("Cache incomplete, regenerating embeddings...")

    # Generate embeddings
    logger.info(f"Generating embeddings for {len(queries)} queries...")
    embeddings = {}

    for query in queries:
        query_id = query['id']
        query_en = f"query: {query['en']}"
        query_fr = f"query: {query['fr']}"

        # Generate both embeddings with "query: " prefix
        embedding_en = model.encode(query_en, normalize_embeddings=True)
        embedding_fr = model.encode(query_fr, normalize_embeddings=True)

        embeddings[query_id] = (embedding_en, embedding_fr)
        logger.info(f"  Generated embeddings for {query_id}")

    # Save to cache
    logger.info(f"Caching query embeddings to {cache_path}")
    with open(cache_path, 'wb') as f:
        pickle.dump(embeddings, f)

    return embeddings


def format_speaker_turns(text: str, speaker_positions: dict) -> str:
    """
    Format text with speaker turns as:
    Speaker A: <text>
    Speaker B: <text>

    Groups consecutive turns from the same speaker into a single block.

    Args:
        text: The full segment text
        speaker_positions: JSONB dict mapping speaker_id to [[start, end]] positions

    Returns:
        Formatted text with speaker labels, or original text if no speaker info
    """
    if not speaker_positions or not text:
        return text

    # Build list of (start, end, speaker_label) tuples
    turns = []
    speaker_ids = sorted(speaker_positions.keys())

    # Assign letter labels (A, B, C, etc.)
    speaker_labels = {}
    for i, speaker_id in enumerate(speaker_ids):
        speaker_labels[speaker_id] = chr(65 + i)  # A=65 in ASCII

    # Collect all turns with positions
    for speaker_id, positions in speaker_positions.items():
        for start, end in positions:
            turns.append((start, end, speaker_labels[speaker_id]))

    # Sort by start position
    turns.sort(key=lambda x: x[0])

    # Group consecutive turns from same speaker
    formatted_lines = []
    current_speaker = None
    current_text_parts = []

    for start, end, label in turns:
        speaker_text = text[start:end].strip()
        if not speaker_text:
            continue

        if label == current_speaker:
            # Same speaker, accumulate text
            current_text_parts.append(speaker_text)
        else:
            # Different speaker, flush previous and start new
            if current_speaker is not None and current_text_parts:
                formatted_lines.append(f"Speaker {current_speaker}: {' '.join(current_text_parts)}")
            current_speaker = label
            current_text_parts = [speaker_text]

    # Flush final speaker
    if current_speaker is not None and current_text_parts:
        formatted_lines.append(f"Speaker {current_speaker}: {' '.join(current_text_parts)}")

    return "\n".join(formatted_lines) if formatted_lines else text


def build_youtube_link(content_id_string: str, start_time: float) -> str:
    """
    Build a YouTube link with timestamp.

    Args:
        content_id_string: Content ID (could be YouTube ID or other format)
        start_time: Start time in seconds

    Returns:
        YouTube URL with timestamp, or empty string if not a YouTube video
    """
    if not content_id_string:
        return ""

    # Check if it's a YouTube-style ID (11 chars, alphanumeric with - and _)
    if len(content_id_string) == 11 and all(c.isalnum() or c in '-_' for c in content_id_string):
        timestamp = int(start_time) if start_time else 0
        return f"https://youtu.be/{content_id_string}?t={timestamp}"

    return ""


def run_semantic_search_bilingual(
    session,
    query_embedding_en: np.ndarray,
    query_embedding_fr: np.ndarray,
    limit: int = 100,
    min_confidence: float = 0.75
) -> List[Dict]:
    """
    Run semantic search over cprmv_analysis table using BOTH English and French query embeddings.
    Takes the union of top results from each language query, ranked by best similarity.

    Args:
        session: Database session
        query_embedding_en: English query embedding vector (2000-dim)
        query_embedding_fr: French query embedding vector (2000-dim)
        limit: Maximum results to return (after merging both searches)
        min_confidence: Minimum confidence threshold for themes

    Returns:
        List of dicts with segment_id, distance, themes, etc.
    """
    # Convert embeddings to lists for pgvector
    embedding_en_list = query_embedding_en.tolist()
    embedding_fr_list = query_embedding_fr.tolist()

    # Query: semantic search over segments with high-confidence themes
    # Use cosine similarity (1 - cosine_distance) for 0-1 range, higher = better
    sql = text("""
        SELECT
            ca.segment_id,
            1 - (ca.embedding <=> CAST(:embedding AS vector(2000))) as similarity,
            ca.themes,
            ca.high_confidence_themes,
            ca.confidence_scores,
            es.text,
            es.speaker_positions,
            es.content_id_string,
            es.start_time,
            es.end_time,
            c.channel_name,
            c.title as episode_title,
            c.main_language
        FROM cprmv_analysis ca
        JOIN embedding_segments es ON es.id = ca.segment_id
        JOIN content c ON c.id = es.content_id
        WHERE ca.embedding IS NOT NULL
          AND array_length(ca.high_confidence_themes, 1) > 0
        ORDER BY similarity DESC
        LIMIT :limit
    """)

    # Run English query
    result_en = session.execute(sql, {
        'embedding': embedding_en_list,
        'limit': limit * 2  # Get more to ensure good coverage after merge
    })

    # Run French query
    result_fr = session.execute(sql, {
        'embedding': embedding_fr_list,
        'limit': limit * 2
    })

    # Merge results, keeping best similarity per segment
    results_map = {}

    for row in result_en:
        similarity = float(row.similarity)
        results_map[row.segment_id] = {
            'segment_id': row.segment_id,
            'similarity': similarity,
            'matched_via': 'en',
            'themes': row.themes,
            'high_confidence_themes': row.high_confidence_themes,
            'confidence_scores': row.confidence_scores,
            'text': row.text,
            'text_with_speakers': format_speaker_turns(row.text, row.speaker_positions),
            'content_id_string': row.content_id_string,
            'start_time': row.start_time,
            'end_time': row.end_time,
            'channel_name': row.channel_name,
            'episode_title': row.episode_title,
            'language': row.main_language,
            'youtube_link': build_youtube_link(row.content_id_string, row.start_time)
        }

    for row in result_fr:
        similarity = float(row.similarity)
        segment_id = row.segment_id

        # If segment already seen, keep the better similarity (higher is better)
        if segment_id in results_map:
            if similarity > results_map[segment_id]['similarity']:
                results_map[segment_id]['similarity'] = similarity
                results_map[segment_id]['matched_via'] = 'fr'
            else:
                results_map[segment_id]['matched_via'] = 'both'
        else:
            results_map[segment_id] = {
                'segment_id': segment_id,
                'similarity': similarity,
                'matched_via': 'fr',
                'themes': row.themes,
                'high_confidence_themes': row.high_confidence_themes,
                'confidence_scores': row.confidence_scores,
                'text': row.text,
                'text_with_speakers': format_speaker_turns(row.text, row.speaker_positions),
                'content_id_string': row.content_id_string,
                'start_time': row.start_time,
                'end_time': row.end_time,
                'channel_name': row.channel_name,
                'episode_title': row.episode_title,
                'language': row.main_language,
                'youtube_link': build_youtube_link(row.content_id_string, row.start_time)
            }

    # Sort by similarity (best first) and limit
    results = sorted(results_map.values(), key=lambda x: x['similarity'], reverse=True)[:limit]

    return results


def balanced_sample(
    results: List[Dict],
    sample_size: int,
    used_segment_ids: Set[int],
    balance_by: List[str] = ['language', 'channel_name']
) -> List[Dict]:
    """
    Sample results with balancing by specified fields and avoiding duplicates.

    Args:
        results: List of result dicts from semantic search
        sample_size: Number of samples to select
        used_segment_ids: Set of segment IDs already used (to avoid duplicates)
        balance_by: List of fields to balance by (e.g., ['language', 'channel_name'])

    Returns:
        List of sampled result dicts
    """
    # Filter out already-used segments
    available = [r for r in results if r['segment_id'] not in used_segment_ids]

    if len(available) <= sample_size:
        return available

    # Group by balance_by fields
    groups = defaultdict(list)
    for result in available:
        key = tuple(result.get(field) for field in balance_by)
        groups[key].append(result)

    # Calculate samples per group
    num_groups = len(groups)
    samples_per_group = max(1, sample_size // num_groups)
    remainder = sample_size % num_groups

    # Sample from each group
    sampled = []
    group_items = list(groups.items())

    # First pass: get samples_per_group from each group
    for key, group_results in group_items:
        n_samples = min(samples_per_group, len(group_results))
        # Sample by taking evenly spaced results (better than random for diversity)
        if n_samples < len(group_results):
            indices = np.linspace(0, len(group_results) - 1, n_samples, dtype=int)
            group_sample = [group_results[i] for i in indices]
        else:
            group_sample = group_results
        sampled.extend(group_sample)

    # Second pass: distribute remainder to groups with remaining items
    if len(sampled) < sample_size:
        for key, group_results in group_items:
            already_sampled = len([s for s in sampled if s in group_results])
            remaining = len(group_results) - already_sampled
            if remaining > 0 and len(sampled) < sample_size:
                # Take next unseen items from this group
                unseen = [r for r in group_results if r not in sampled]
                n_more = min(len(unseen), sample_size - len(sampled))
                sampled.extend(unseen[:n_more])

    return sampled[:sample_size]


def main():
    parser = argparse.ArgumentParser(description='Sample CPRMV segments for semantic query evaluation')
    parser.add_argument('--queries-file', type=str,
                       default='projects/CPRMV/queries_dave.json',
                       help='Path to queries JSON file')
    parser.add_argument('--samples-per-query', type=int, default=50,
                       help='Number of samples to generate per query')
    parser.add_argument('--output-csv', type=str,
                       default='projects/CPRMV/query_samples.csv',
                       help='Output CSV path')
    parser.add_argument('--embedding-model', type=str,
                       default='Qwen/Qwen3-Embedding-4B',
                       help='Embedding model to use')
    parser.add_argument('--search-limit', type=int, default=200,
                       help='Number of results to retrieve per query before sampling')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show query results without generating samples')

    args = parser.parse_args()

    # Set random seed for reproducibility
    np.random.seed(42)

    # Load queries
    queries = load_queries(args.queries_file)
    logger.info(f"Will sample {args.samples_per_query} results per query ({len(queries)} queries = {len(queries) * args.samples_per_query} total samples)")

    # Load embedding model with truncate_dim for 4B model
    logger.info(f"Loading embedding model: {args.embedding_model}")
    if 'Qwen3-Embedding-4B' in args.embedding_model:
        # Use truncate_dim=2000 to match cprmv_analysis table embeddings
        model = SentenceTransformer(
            args.embedding_model,
            trust_remote_code=True,
            truncate_dim=2000
        )
        logger.info(f"Model loaded with truncate_dim=2000 (4B model)")
    else:
        model = SentenceTransformer(args.embedding_model, trust_remote_code=True)

    logger.info(f"Model loaded (embedding dim: {model.get_sentence_embedding_dimension()})")

    # Load or generate cached query embeddings
    query_embeddings = load_or_generate_query_embeddings(
        queries,
        model,
        args.queries_file,
        args.embedding_model
    )

    # Track used segment IDs to avoid duplicates
    used_segment_ids = set()

    # Store all samples
    all_samples = []

    with get_session() as session:
        for query in queries:
            query_id = query['id']
            query_text_en = query['en']
            query_text_fr = query['fr']

            logger.info(f"\n{'='*80}")
            logger.info(f"Query {query_id}:")
            logger.info(f"  EN: {query_text_en[:80]}...")
            logger.info(f"  FR: {query_text_fr[:80]}...")

            # Get cached embeddings for BOTH languages
            query_embedding_en, query_embedding_fr = query_embeddings[query_id]

            # Run bilingual semantic search (merges results from both queries)
            results = run_semantic_search_bilingual(
                session,
                query_embedding_en,
                query_embedding_fr,
                limit=args.search_limit
            )

            logger.info(f"Found {len(results)} results (merged from en+fr queries)")

            if args.dry_run:
                # Just show top 5 results
                logger.info("Top 5 results:")
                for i, result in enumerate(results[:5], 1):
                    logger.info(f"  {i}. Similarity: {result['similarity']:.4f} | "
                              f"Matched via: {result['matched_via']} | "
                              f"Lang: {result['language']} | "
                              f"Channel: {result['channel_name'][:30]} | "
                              f"Themes: {','.join(result['high_confidence_themes'][:3])}")
                continue

            # Balanced sampling
            samples = balanced_sample(
                results,
                args.samples_per_query,
                used_segment_ids,
                balance_by=['language', 'channel_name']
            )

            logger.info(f"Sampled {len(samples)} results")

            # Log language/channel distribution
            lang_dist = defaultdict(int)
            channel_dist = defaultdict(int)
            for sample in samples:
                lang_dist[sample['language']] += 1
                channel_dist[sample['channel_name']] += 1

            logger.info(f"Language distribution: {dict(lang_dist)}")
            logger.info(f"Top 5 channels: {dict(list(channel_dist.items())[:5])}")

            # Add to samples and mark as used
            for sample in samples:
                sample['query_id'] = query_id
                sample['query_text_en'] = query_text_en
                sample['query_text_fr'] = query_text_fr
                used_segment_ids.add(sample['segment_id'])
                all_samples.append(sample)

    if args.dry_run:
        logger.info("\nDry run complete - no samples generated")
        return 0

    # Write samples to CSV
    logger.info(f"\n{'='*80}")
    logger.info(f"Writing {len(all_samples)} samples to {args.output_csv}")

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        fieldnames = [
            'query_id', 'query_text_en', 'query_text_fr', 'segment_id',
            'similarity', 'matched_via', 'themes', 'high_confidence_themes',
            'confidence_scores', 'language', 'channel_name', 'episode_title',
            'youtube_link', 'content_id_string', 'start_time', 'end_time', 'text', 'text_with_speakers'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for sample in all_samples:
            writer.writerow({
                'query_id': sample['query_id'],
                'query_text_en': sample['query_text_en'],
                'query_text_fr': sample['query_text_fr'],
                'segment_id': sample['segment_id'],
                'similarity': f"{sample['similarity']:.4f}",
                'matched_via': sample['matched_via'],
                'themes': ','.join(sample['themes']),
                'high_confidence_themes': ','.join(sample['high_confidence_themes']),
                'confidence_scores': json.dumps(sample['confidence_scores']),
                'language': sample['language'],
                'channel_name': sample['channel_name'],
                'episode_title': sample['episode_title'],
                'youtube_link': sample['youtube_link'],
                'content_id_string': sample['content_id_string'],
                'start_time': sample['start_time'],
                'end_time': sample['end_time'],
                'text': sample['text'],
                'text_with_speakers': sample['text_with_speakers']
            })

    logger.info(f"Successfully wrote {len(all_samples)} samples")
    logger.info(f"Unique segments: {len(used_segment_ids)}")
    logger.info(f"Output: {output_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
