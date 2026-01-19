#!/usr/bin/env python3
"""
Export CPRMV aggregations from PostgreSQL to JSON for Next.js consumption.

Pulls data directly from theme_classifications table with stage 6/7 filtering:
- Stage 5 relevance check (is_relevant = true)
- Stage 6 false positive detection (is_false_positive = false OR null)
- Stage 7 expanded context re-check (for edge cases)

Usage:
    cd ~/signal4/core
    python -m src.backend.scripts.cprmv.export_aggregations

    # Specify output directory
    python -m src.backend.scripts.cprmv.export_aggregations \
        --output-dir ../frontend/public/data/cprmv-misogyny

Output:
    ../frontend/public/data/cprmv-misogyny/aggregations.json
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Add core root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from sqlalchemy import text
from src.database.session import get_session

try:
    import circlify
except ImportError:
    print("Error: circlify package required. Install with: pip install circlify")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('export_misogyny_aggregations')

# Schema to use (CPRMV v3.0 - has stage 5-7 data)
SCHEMA_ID = 3

# Date range for analysis
DATE_START = '2020-01-01'
DATE_END = '2025-12-31'

# Default output path
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent.parent.parent.parent / "frontend/public/data/cprmv-misogyny"


def fetch_themes(session):
    """Fetch theme definitions from classification schema."""
    result = session.execute(
        text("""
            SELECT themes_json, subthemes_json
            FROM classification_schemas
            WHERE id = :schema_id
        """),
        {'schema_id': SCHEMA_ID}
    )
    row = result.fetchone()
    if not row:
        raise ValueError(f"Schema {SCHEMA_ID} not found")
    return row[0], row[1]


def fetch_true_positives(session):
    """
    Fetch segments that are true positives after all stages.

    A segment is a TRUE POSITIVE if:
    1. Stage 5 passed (is_relevant = true)
    2. Stage 6 stance is 'strongly_holds', 'holds', or 'leans_holds', OR
    3. Stage 7 upgraded back to true positive (stance changed to one of the above)

    Note: 'leans_holds' is included as it indicates the speaker still holds the view,
    just less strongly.
    """
    result = session.execute(
        text("""
            WITH stage_status AS (
                SELECT
                    tc.id as tc_id,
                    tc.segment_id,
                    tc.theme_ids,
                    tc.subtheme_ids,
                    tc.high_confidence_themes,
                    tc.final_confidence_scores,
                    tc.created_at,
                    -- Stage 5: is_relevant
                    COALESCE((tc.stage5_final_check->>'is_relevant')::boolean, false) as stage5_relevant,
                    -- Stage 6 stance
                    tc.stage6_false_positive_check->>'speaker_stance' as stage6_stance,
                    -- Stage 7 stance (overrides stage 6 if present)
                    tc.stage7_expanded_context->>'speaker_stance' as stage7_stance
                FROM theme_classifications tc
                WHERE tc.schema_id = :schema_id
            )
            SELECT
                ss.*,
                es.text as segment_text,
                es.start_time,
                es.end_time,
                es.content_id,
                c.content_id as youtube_id,
                c.title as episode_title,
                c.channel_name,
                c.main_language,
                c.publish_date,
                c.duration
            FROM stage_status ss
            JOIN embedding_segments es ON ss.segment_id = es.id
            JOIN content c ON es.content_id = c.id
            WHERE
                ss.stage5_relevant = true
                AND (
                    -- Use stage 7 stance if available, otherwise stage 6
                    COALESCE(ss.stage7_stance, ss.stage6_stance) IN ('strongly_holds', 'holds', 'leans_holds')
                )
                AND c.publish_date >= :date_start
                AND c.publish_date <= :date_end
            ORDER BY ss.created_at DESC
        """),
        {'schema_id': SCHEMA_ID, 'date_start': DATE_START, 'date_end': DATE_END}
    )
    return [dict(row._mapping) for row in result]


def fetch_content_summary(session):
    """Fetch content volume summary by language and year."""
    result = session.execute(
        text("""
            SELECT
                EXTRACT(YEAR FROM c.publish_date)::int as year,
                c.main_language,
                COUNT(DISTINCT c.id) as episode_count,
                SUM(c.duration) / 3600.0 as hours
            FROM content c
            WHERE 'CPRMV' = ANY(c.projects)
                AND c.publish_date IS NOT NULL
                AND c.main_language IN ('en', 'fr')
                AND c.publish_date >= :date_start
                AND c.publish_date <= :date_end
            GROUP BY EXTRACT(YEAR FROM c.publish_date), c.main_language
            ORDER BY year, main_language
        """),
        {'date_start': DATE_START, 'date_end': DATE_END}
    )
    return [dict(row._mapping) for row in result]


def fetch_channel_summary(session):
    """Fetch channel-level summary."""
    result = session.execute(
        text("""
            SELECT
                c.channel_name,
                c.main_language,
                COUNT(DISTINCT c.id) as episode_count,
                SUM(c.duration) / 3600.0 as total_hours
            FROM content c
            WHERE 'CPRMV' = ANY(c.projects)
                AND c.channel_name IS NOT NULL
                AND c.publish_date >= :date_start
                AND c.publish_date <= :date_end
            GROUP BY c.channel_name, c.main_language
            ORDER BY episode_count DESC
        """),
        {'date_start': DATE_START, 'date_end': DATE_END}
    )
    return [dict(row._mapping) for row in result]


def fetch_total_segments(session):
    """Fetch total number of embedding segments in the corpus."""
    result = session.execute(
        text("""
            SELECT COUNT(*) as total_segments
            FROM embedding_segments es
            JOIN content c ON es.content_id = c.id
            WHERE 'CPRMV' = ANY(c.projects)
                AND c.publish_date >= :date_start
                AND c.publish_date <= :date_end
        """),
        {'date_start': DATE_START, 'date_end': DATE_END}
    )
    row = result.fetchone()
    return row[0] if row else 0


def compute_circle_packing_by_volume(channels):
    """
    Compute circle packing layout for channels by total hours.
    Returns top 50 channels with circle positions.
    """
    # Sort by hours descending and take top 50
    sorted_channels = sorted(
        [c for c in channels if c['total_hours'] and c['total_hours'] > 0],
        key=lambda x: x['total_hours'],
        reverse=True
    )[:50]

    if not sorted_channels:
        return []

    # Prepare data for circlify (needs list of dicts with 'datum' key)
    circle_data = [{'id': c['channel_name'], 'datum': c['total_hours']} for c in sorted_channels]

    # Compute circle packing
    circles = circlify.circlify(
        circle_data,
        show_enclosure=False,
        target_enclosure=circlify.Circle(x=0, y=0, r=1)
    )

    # circlify returns in ascending order, reverse to match our descending sort
    circles_reversed = list(reversed(circles))

    # Combine channel data with circle positions
    result = []
    for channel, circle in zip(sorted_channels, circles_reversed):
        result.append({
            'channel_name': channel['channel_name'],
            'main_language': channel['main_language'],
            'episode_count': channel['episode_count'],
            'total_hours': round(channel['total_hours'], 1) if channel['total_hours'] else 0,
            'circle_x': circle.x,
            'circle_y': circle.y,
            'circle_r': circle.r
        })

    return result


def compute_circle_packing_by_rate(channels, channel_flagged_counts):
    """
    Compute circle packing layout for channels by flagged segments per hour.
    Returns top 50 channels with circle positions.
    """
    # Calculate segments per hour for each channel
    channels_with_rate = []
    for c in channels:
        if c['total_hours'] and c['total_hours'] > 10:  # Minimum 10 hours
            flagged = channel_flagged_counts.get(c['channel_name'], 0)
            if flagged > 0:
                rate = flagged / c['total_hours']
                channels_with_rate.append({
                    **c,
                    'flagged_segment_count': flagged,
                    'segments_per_hour': round(rate, 2)
                })

    # Sort by rate descending and take top 50
    sorted_channels = sorted(
        channels_with_rate,
        key=lambda x: x['segments_per_hour'],
        reverse=True
    )[:50]

    if not sorted_channels:
        return []

    # Prepare data for circlify
    circle_data = [{'id': c['channel_name'], 'datum': c['segments_per_hour']} for c in sorted_channels]

    # Compute circle packing
    circles = circlify.circlify(
        circle_data,
        show_enclosure=False,
        target_enclosure=circlify.Circle(x=0, y=0, r=1)
    )

    # circlify returns in ascending order, reverse to match our descending sort
    circles_reversed = list(reversed(circles))

    # Combine channel data with circle positions
    result = []
    for channel, circle in zip(sorted_channels, circles_reversed):
        result.append({
            'channel_name': channel['channel_name'],
            'main_language': channel['main_language'],
            'total_hours': round(channel['total_hours'], 1) if channel['total_hours'] else 0,
            'flagged_segment_count': channel['flagged_segment_count'],
            'segments_per_hour': channel['segments_per_hour'],
            'circle_x': circle.x,
            'circle_y': circle.y,
            'circle_r': circle.r
        })

    return result


def aggregate_data(segments, themes_json):
    """Aggregate segment data into summary statistics."""

    # Theme distribution
    theme_counts = defaultdict(int)
    theme_by_language = defaultdict(lambda: defaultdict(int))
    theme_by_month = defaultdict(lambda: defaultdict(int))

    # Channel flagged counts
    channel_flagged = defaultdict(int)

    for seg in segments:
        # Count themes
        for theme_id in seg['theme_ids']:
            theme_counts[theme_id] += 1
            theme_by_language[theme_id][seg['main_language']] += 1

            if seg['publish_date']:
                month_key = seg['publish_date'].strftime('%Y-%m')
                theme_by_month[month_key][theme_id] += 1

        # Channel counts
        if seg['channel_name']:
            channel_flagged[seg['channel_name']] += 1

    total_segments = len(segments)

    # Build theme distribution
    theme_distribution = []
    for theme_id, count in sorted(theme_counts.items(), key=lambda x: -x[1]):
        theme_info = themes_json.get(theme_id, {})
        theme_distribution.append({
            'theme_id': theme_id,
            'theme_name': theme_info.get('name', f'Theme {theme_id}'),
            'description_en': theme_info.get('description_en', ''),
            'description_fr': theme_info.get('description_fr', ''),
            'segment_count': count,
            'proportion': count / total_segments if total_segments > 0 else 0,
            'by_language': dict(theme_by_language[theme_id])
        })

    # Build theme trends (monthly)
    theme_trends = []
    for month, theme_data in sorted(theme_by_month.items()):
        month_total = sum(theme_data.values())
        for theme_id, count in theme_data.items():
            theme_info = themes_json.get(theme_id, {})
            theme_trends.append({
                'year_month': month,
                'theme_id': theme_id,
                'theme_name': theme_info.get('name', f'Theme {theme_id}'),
                'segment_count': count,
                'proportion': count / month_total if month_total > 0 else 0
            })

    return {
        'theme_distribution': theme_distribution,
        'theme_trends': theme_trends,
        'channel_flagged_counts': dict(channel_flagged),
        'total_flagged_segments': total_segments
    }


def get_sample_segments(segments, themes_json, subthemes_json, samples_per_theme=3, samples_per_subtheme=2):
    """Get sample segments for each theme and subtheme (for audio playback)."""
    theme_samples = defaultdict(lambda: {'en': [], 'fr': []})
    subtheme_samples = defaultdict(lambda: {'en': [], 'fr': []})
    subtheme_counts = defaultdict(int)
    total_flagged = len(segments)

    # Helper to create sample entry
    def make_sample(seg, lang):
        return {
            'segment_id': seg['segment_id'],
            'content_id': seg['content_id'],
            'youtube_id': seg['youtube_id'],
            'title': seg['episode_title'],
            'channel_name': seg['channel_name'],
            'publish_date': seg['publish_date'].strftime('%Y-%m-%d') if seg.get('publish_date') else None,
            'start_time': float(seg['start_time']) if seg['start_time'] else 0,
            'end_time': float(seg['end_time']) if seg['end_time'] else 0,
            'text': seg['segment_text'][:500] if seg['segment_text'] else '',
            'language': lang
        }

    # Track which segments are used for theme samples (so subthemes use different ones)
    theme_sample_ids = set()

    # First pass: collect theme samples
    for seg in segments:
        for theme_id in seg['theme_ids']:
            lang = seg['main_language']
            if lang in ['en', 'fr'] and len(theme_samples[theme_id][lang]) < samples_per_theme:
                theme_samples[theme_id][lang].append(make_sample(seg, lang))
                theme_sample_ids.add(seg['segment_id'])

    # Second pass: count subthemes and collect subtheme samples
    for seg in segments:
        subtheme_ids = seg.get('subtheme_ids') or []
        for subtheme_id in subtheme_ids:
            subtheme_counts[subtheme_id] += 1

            # Collect samples (preferring segments not used for theme samples)
            lang = seg['main_language']
            if lang in ['en', 'fr'] and len(subtheme_samples[subtheme_id][lang]) < samples_per_subtheme:
                # Prefer segments not already used for theme-level samples
                if seg['segment_id'] not in theme_sample_ids:
                    subtheme_samples[subtheme_id][lang].append(make_sample(seg, lang))

    # Third pass: fill remaining subtheme samples if needed (allow reuse if necessary)
    for seg in segments:
        subtheme_ids = seg.get('subtheme_ids') or []
        for subtheme_id in subtheme_ids:
            lang = seg['main_language']
            if lang in ['en', 'fr'] and len(subtheme_samples[subtheme_id][lang]) < samples_per_subtheme:
                # Check if this segment is already in this subtheme's samples
                existing_ids = {s['segment_id'] for s in subtheme_samples[subtheme_id][lang]}
                if seg['segment_id'] not in existing_ids:
                    subtheme_samples[subtheme_id][lang].append(make_sample(seg, lang))

    # Build subthemes lookup by theme_id with counts and samples
    theme_subthemes = defaultdict(list)
    if subthemes_json:
        for subtheme_id, subtheme_data in subthemes_json.items():
            parent_theme = subtheme_data.get('theme_id')
            if parent_theme:
                count = subtheme_counts.get(subtheme_id, 0)
                theme_subthemes[parent_theme].append({
                    'subtheme_id': subtheme_id,
                    'subtheme_name': {
                        'en': subtheme_data.get('name_en', subtheme_data.get('name', '')),
                        'fr': subtheme_data.get('name_fr', '')
                    },
                    'subtheme_description': {
                        'en': subtheme_data.get('description_en', ''),
                        'fr': subtheme_data.get('description_fr', '')
                    },
                    'proportion': count / total_flagged if total_flagged > 0 else 0,
                    'examples': {
                        'en': subtheme_samples[subtheme_id]['en'],
                        'fr': subtheme_samples[subtheme_id]['fr']
                    }
                })

    # Convert to list format
    result = []
    for theme_id, samples in theme_samples.items():
        theme_info = themes_json.get(theme_id, {})
        theme_entry = {
            'theme_id': theme_id,
            'theme_name': theme_info.get('name', f'Theme {theme_id}'),
            'description': {
                'en': theme_info.get('description_en', ''),
                'fr': theme_info.get('description_fr', '')
            },
            'examples': {
                'en': samples['en'],
                'fr': samples['fr']
            }
        }
        # Add subthemes if available
        if theme_id in theme_subthemes:
            theme_entry['subthemes'] = sorted(theme_subthemes[theme_id], key=lambda x: x['subtheme_id'])
        result.append(theme_entry)

    return sorted(result, key=lambda x: x['theme_id'])


def main():
    parser = argparse.ArgumentParser(description='Export CPRMV Misogyny Report aggregations')
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help='Output directory for aggregations.json'
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_path = output_dir / "aggregations.json"

    logger.info("=" * 80)
    logger.info("Exporting CPRMV Misogyny Report Aggregations")
    logger.info("=" * 80)
    logger.info(f"Date range: {DATE_START} to {DATE_END}")
    logger.info(f"Output: {output_path.absolute()}")

    with get_session() as session:
        # Fetch themes
        logger.info(f"Fetching theme definitions from schema {SCHEMA_ID}...")
        themes_json, subthemes_json = fetch_themes(session)
        logger.info(f"  Found {len(themes_json)} themes")

        # Fetch true positive segments
        logger.info(f"Fetching true positive segments (stage 5-7 filtered)...")
        segments = fetch_true_positives(session)
        logger.info(f"  Found {len(segments)} true positive segments")

        # Fetch content summary
        logger.info(f"Fetching content volume summary...")
        content_summary = fetch_content_summary(session)

        # Fetch channel summary
        logger.info(f"Fetching channel summary...")
        channel_summary = fetch_channel_summary(session)

        # Fetch total segments count
        logger.info(f"Fetching total segment count...")
        total_segments = fetch_total_segments(session)
        logger.info(f"  Found {total_segments:,} total segments")

        # Aggregate
        logger.info(f"Aggregating data...")
        aggregations = aggregate_data(segments, themes_json)

        # Get sample segments
        logger.info(f"Selecting sample segments for each theme...")
        theme_examples = get_sample_segments(segments, themes_json, subthemes_json)

        # Compute circle packing layouts
        logger.info(f"Computing circle packing layouts...")
        circle_pack_volume = compute_circle_packing_by_volume(channel_summary)
        logger.info(f"  By volume: {len(circle_pack_volume)} channels")

        circle_pack_rate = compute_circle_packing_by_rate(
            channel_summary,
            aggregations['channel_flagged_counts']
        )
        logger.info(f"  By rate: {len(circle_pack_rate)} channels")

        # Calculate totals
        total_hours = sum(r['hours'] for r in content_summary if r['hours'])
        total_episodes = sum(r['episode_count'] for r in content_summary)

        # Calculate language breakdowns
        english_hours = sum(r['hours'] for r in content_summary if r['hours'] and r['main_language'] == 'en')
        french_hours = sum(r['hours'] for r in content_summary if r['hours'] and r['main_language'] == 'fr')
        english_episodes = sum(r['episode_count'] for r in content_summary if r['main_language'] == 'en')
        french_episodes = sum(r['episode_count'] for r in content_summary if r['main_language'] == 'fr')

        # Count unique content with flagged segments
        flagged_content_ids = set(seg['content_id'] for seg in segments)
        flagged_videos = len(flagged_content_ids)
        flagged_pct = (flagged_videos / total_episodes * 100) if total_episodes > 0 else 0

        # Format dates for display
        min_date = datetime.strptime(DATE_START, '%Y-%m-%d').strftime('%B %d, %Y')
        max_date = datetime.strptime(DATE_END, '%Y-%m-%d').strftime('%B %d, %Y')

        # Build output
        output = {
            'exportedAt': datetime.now().isoformat(),
            'version': '2.0',
            'schemaId': SCHEMA_ID,

            # Summary stats for MDX interpolation
            'summary_stats': {
                'n_channels': len(set(r['channel_name'] for r in channel_summary)),
                'min_date': min_date,
                'max_date': max_date,
                'english_hours': round(english_hours, 0),
                'french_hours': round(french_hours, 0),
                'english_episodes': english_episodes,
                'french_episodes': french_episodes,
                'flagged_videos': flagged_videos,
                'flagged_pct': round(flagged_pct, 1),
                'segment_pct': round(aggregations['total_flagged_segments'] / total_segments * 100, 2) if total_segments > 0 else 0,
                'total_segments': total_segments,
            },

            # Volume data
            'lang_volume_by_year': [
                {
                    'year': r['year'],
                    'main_language': r['main_language'],
                    'hours': round(r['hours'], 1) if r['hours'] else 0,
                    'episode_count': r['episode_count']
                }
                for r in content_summary
            ],
            'total_hours': round(total_hours, 0),
            'total_episodes': total_episodes,

            # Theme data
            'theme_distribution': aggregations['theme_distribution'],
            'theme_trends': aggregations['theme_trends'],
            'themes': [
                {
                    'theme_id': tid,
                    'theme_name': tdata.get('name', ''),
                    'description_en': tdata.get('description_en', ''),
                    'description_fr': tdata.get('description_fr', '')
                }
                for tid, tdata in sorted(themes_json.items())
            ],
            'theme_examples': theme_examples,

            # Channel data
            'channels_summary': [
                {
                    'channel_name': r['channel_name'],
                    'main_language': r['main_language'],
                    'episode_count': r['episode_count'],
                    'total_hours': round(r['total_hours'], 1) if r['total_hours'] else 0,
                    'flagged_segment_count': aggregations['channel_flagged_counts'].get(r['channel_name'], 0)
                }
                for r in channel_summary
            ],

            # Circle packing data for visualizations
            'circle_pack_by_volume': circle_pack_volume,
            'circle_pack_by_rate': circle_pack_rate,

            # Counts
            'flagged_segment_count': aggregations['total_flagged_segments'],
        }

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write JSON
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        file_size = output_path.stat().st_size / 1024
        logger.info("")
        logger.info("=" * 80)
        logger.info("Export Complete")
        logger.info("=" * 80)
        logger.info(f"Output: {output_path}")
        logger.info(f"File size: {file_size:.1f} KB")
        logger.info("")
        logger.info("Summary:")
        logger.info(f"  Date range: {min_date} - {max_date}")
        logger.info(f"  Channels: {output['summary_stats']['n_channels']}")
        logger.info(f"  Total hours: {output['total_hours']:,.0f}")
        logger.info(f"  Total episodes: {output['total_episodes']:,}")
        logger.info(f"  Flagged segments: {output['flagged_segment_count']:,}")
        logger.info(f"  Themes: {len(output['themes'])}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
