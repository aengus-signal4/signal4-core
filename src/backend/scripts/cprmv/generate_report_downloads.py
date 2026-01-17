#!/usr/bin/env python3
"""
Generate download files for the CPRMV Misogyny Report.

Creates three files:
1. episodes.csv.gz - Episode metadata for the full corpus
2. segments.csv.gz - High-confidence flagged segments (>=75% confidence)
3. themes.csv - Theme taxonomy definitions

Usage:
    cd ~/signal4/core
    python -m src.backend.scripts.cprmv.generate_report_downloads

    # Specify output directory
    python -m src.backend.scripts.cprmv.generate_report_downloads \
        --output-dir ../frontend/public/data/cprmv-misogyny/downloads
"""

import sys
import gzip
import argparse
import logging
from pathlib import Path
from typing import Dict
import pandas as pd

# Add core root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.database.session import get_session
from sqlalchemy import text

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('generate_misogyny_downloads')

# Configuration
SCHEMA_NAME = 'CPRMV'
SCHEMA_VERSION = 'v3.0'
MIN_CONFIDENCE = 0.75


def build_youtube_link(content_id_string: str, start_time: float) -> str:
    """Build a YouTube link with timestamp."""
    if not content_id_string:
        return ""

    # Check if it looks like a YouTube ID (11 chars, alphanumeric with - and _)
    if len(content_id_string) == 11 and all(c.isalnum() or c in '-_' for c in content_id_string):
        timestamp = int(start_time) if start_time else 0
        return f"https://youtu.be/{content_id_string}?t={timestamp}"

    return ""


def get_schema_id(session) -> int:
    """Get the schema ID for CPRMV v3.0."""
    result = session.execute(
        text("""
            SELECT id FROM classification_schemas
            WHERE name = :name AND version = :version
        """),
        {'name': SCHEMA_NAME, 'version': SCHEMA_VERSION}
    )
    row = result.fetchone()
    if not row:
        raise ValueError(f"Schema '{SCHEMA_NAME}' version '{SCHEMA_VERSION}' not found")
    return row[0]


def generate_episodes_csv(session, output_path: Path) -> int:
    """Generate episodes.csv.gz with all CPRMV corpus metadata."""
    logger.info("Generating episodes.csv.gz...")

    result = session.execute(
        text("""
            SELECT
                c.id as content_id,
                c.content_id as youtube_id,
                c.channel_name,
                c.title,
                c.publish_date,
                c.duration,
                c.main_language,
                c.platform,
                CASE
                    WHEN c.platform = 'youtube' AND length(c.content_id) = 11
                    THEN 'https://youtu.be/' || c.content_id
                    ELSE ''
                END as url
            FROM content c
            WHERE 'CPRMV' = ANY(c.projects)
              AND c.is_embedded = true
            ORDER BY c.publish_date DESC
        """)
    )

    rows = []
    for row in result:
        rows.append({
            'content_id': row.content_id,
            'youtube_id': row.youtube_id or '',
            'channel_name': row.channel_name or '',
            'title': row.title or '',
            'publish_date': row.publish_date.isoformat() if row.publish_date else '',
            'duration_seconds': round(row.duration, 1) if row.duration else '',
            'duration_formatted': format_duration(row.duration) if row.duration else '',
            'language': row.main_language or '',
            'platform': row.platform or '',
            'url': row.url or ''
        })

    df = pd.DataFrame(rows)

    # Write gzipped CSV
    with gzip.open(output_path, 'wt', encoding='utf-8', newline='') as f:
        df.to_csv(f, index=False)

    logger.info(f"  Written {len(df)} episodes to {output_path}")
    return len(df)


def generate_segments_csv(session, schema_id: int, output_path: Path) -> int:
    """Generate segments.csv.gz with high-confidence flagged segments."""
    logger.info("Generating segments.csv.gz...")

    # Query for high-confidence segments that passed Stage 5 relevance and Stage 6/7 stance filtering
    result = session.execute(
        text("""
            SELECT
                tc.segment_id,
                es.content_id,
                c.content_id as youtube_id,
                c.channel_name,
                c.title as episode_title,
                c.publish_date,
                c.main_language,
                es.start_time,
                es.end_time,
                es.text as segment_text,
                tc.high_confidence_themes,
                tc.final_confidence_scores,
                tc.max_similarity_score,
                tc.matched_via
            FROM theme_classifications tc
            JOIN embedding_segments es ON tc.segment_id = es.id
            JOIN content c ON es.content_id = c.id
            WHERE tc.schema_id = :schema_id
              AND array_length(tc.high_confidence_themes, 1) > 0
              AND (tc.stage5_final_check->>'is_relevant')::boolean = true
              AND (
                  tc.stage6_false_positive_check->>'speaker_stance' IN ('strongly_holds', 'holds', 'leans_holds')
                  OR (tc.stage7_expanded_context->>'is_false_positive')::boolean = false
              )
            ORDER BY c.publish_date DESC, es.start_time
        """),
        {'schema_id': schema_id}
    )

    rows = []
    for row in result:
        # Parse themes and confidence scores
        themes = row.high_confidence_themes or []
        conf_scores = row.final_confidence_scores or {}

        # Filter to Q-prefixed subthemes only
        q_themes = [t for t in themes if t.startswith('Q')]

        # Build confidence string
        conf_str = ', '.join([f"{t}:{conf_scores.get(t, 0.75):.2f}" for t in q_themes])

        # Build URL
        url = build_youtube_link(row.youtube_id, row.start_time)

        rows.append({
            'segment_id': row.segment_id,
            'content_id': row.content_id,
            'youtube_id': row.youtube_id or '',
            'channel_name': row.channel_name or '',
            'episode_title': row.episode_title or '',
            'publish_date': row.publish_date.isoformat() if row.publish_date else '',
            'language': row.main_language or '',
            'start_time': round(row.start_time, 2) if row.start_time else 0,
            'end_time': round(row.end_time, 2) if row.end_time else 0,
            'themes': ','.join(q_themes),
            'confidence_scores': conf_str,
            'max_similarity': round(row.max_similarity_score, 3) if row.max_similarity_score else '',
            'matched_via': row.matched_via or '',
            'segment_text': row.segment_text or '',
            'url': url
        })

    df = pd.DataFrame(rows)

    # Write gzipped CSV
    with gzip.open(output_path, 'wt', encoding='utf-8', newline='') as f:
        df.to_csv(f, index=False)

    logger.info(f"  Written {len(df)} segments to {output_path}")
    return len(df)


def generate_themes_csv(session, output_path: Path) -> int:
    """Generate themes.csv with theme taxonomy definitions from database."""
    logger.info("Generating themes.csv...")

    # Query theme definitions from the database
    result = session.execute(
        text("""
            SELECT
                cs.id as schema_id,
                cs.name as schema_name,
                cs.version,
                cs.theme_hierarchy
            FROM classification_schemas cs
            WHERE cs.name = :name AND cs.version = :version
        """),
        {'name': SCHEMA_NAME, 'version': SCHEMA_VERSION}
    )

    row = result.fetchone()
    if not row:
        raise ValueError(f"Schema '{SCHEMA_NAME}' version '{SCHEMA_VERSION}' not found")

    theme_hierarchy = row.theme_hierarchy or {}

    # Build flat list of themes and subthemes
    rows = []
    for theme_id, theme_data in theme_hierarchy.items():
        theme_name_en = theme_data.get('name', '')
        theme_name_fr = theme_data.get('name_fr', '')
        theme_desc_en = theme_data.get('description', '')
        theme_desc_fr = theme_data.get('description_fr', '')

        subthemes = theme_data.get('subthemes', {})
        for subtheme_id, subtheme_data in subthemes.items():
            rows.append({
                'theme_id': theme_id,
                'theme_name_en': theme_name_en,
                'theme_name_fr': theme_name_fr,
                'theme_description_en': theme_desc_en,
                'theme_description_fr': theme_desc_fr,
                'subtheme_id': subtheme_id,
                'subtheme_name_en': subtheme_data.get('name', ''),
                'subtheme_name_fr': subtheme_data.get('name_fr', ''),
                'subtheme_description_en': subtheme_data.get('description', ''),
                'subtheme_description_fr': subtheme_data.get('description_fr', '')
            })

    df = pd.DataFrame(rows)

    # Write CSV (not gzipped since it's small)
    df.to_csv(output_path, index=False, encoding='utf-8')

    logger.info(f"  Written {len(df)} theme definitions to {output_path}")
    return len(df)


def format_duration(seconds: float) -> str:
    """Format duration in seconds to HH:MM:SS."""
    if not seconds:
        return ""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def main():
    parser = argparse.ArgumentParser(description='Generate CPRMV Misogyny Report download files')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='../frontend/public/data/cprmv-misogyny/downloads',
        help='Output directory for generated files'
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Generating CPRMV Misogyny Report Downloads")
    logger.info("=" * 80)
    logger.info(f"Output directory: {output_dir.absolute()}")

    with get_session() as session:
        schema_id = get_schema_id(session)
        logger.info(f"Using schema_id: {schema_id}")

        # Generate all three files
        stats: Dict[str, int] = {}

        # 1. Episodes metadata
        episodes_path = output_dir / 'episodes.csv.gz'
        stats['episodes'] = generate_episodes_csv(session, episodes_path)

        # 2. Flagged segments
        segments_path = output_dir / 'segments.csv.gz'
        stats['segments'] = generate_segments_csv(session, schema_id, segments_path)

        # 3. Theme taxonomy
        themes_path = output_dir / 'themes.csv'
        stats['themes'] = generate_themes_csv(session, themes_path)

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("Generation Complete")
    logger.info("=" * 80)
    logger.info(f"Episodes:  {stats['episodes']:,} rows")
    logger.info(f"Segments:  {stats['segments']:,} rows")
    logger.info(f"Themes:    {stats['themes']:,} rows")

    # Print file sizes
    for name, path in [('episodes', episodes_path), ('segments', segments_path), ('themes', themes_path)]:
        size = path.stat().st_size
        if size > 1024 * 1024:
            size_str = f"{size / (1024*1024):.1f} MB"
        elif size > 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size} bytes"
        logger.info(f"  {path.name}: {size_str}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
