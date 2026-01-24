#!/usr/bin/env python
"""
US-Canada MEO Analysis - Data Extraction

Extracts podcast segments from database for:
1. Timeline analysis (Jan 2025 - Jan 2026 Canada mentions)
2. Davos discourse analysis (Jan 21-23, 2026 reactions)

Output:
  .cache/canada_sentiment_data.json - Daily aggregated sentiment for timeline
  .cache/davos_reaction_segments.json - Individual segments for classification
"""

import sys
import json
from pathlib import Path
from datetime import date, datetime, timedelta
from collections import defaultdict
from typing import List, Dict

# Add paths for imports
_script_dir = Path(__file__).parent
_core_root = _script_dir.parent.parent
if str(_core_root) not in sys.path:
    sys.path.insert(0, str(_core_root))

import psycopg2
from psycopg2.extras import RealDictCursor
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.utils.config import load_config, get_credential

# Directories
CACHE_DIR = _script_dir / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

# Output files
TIMELINE_CACHE = CACHE_DIR / "canada_sentiment_data.json"
DAVOS_SEGMENTS_CACHE = CACHE_DIR / "davos_reaction_segments.json"

# Date ranges
TIMELINE_START = date(2025, 1, 1)
TIMELINE_END = date(2026, 1, 23)
DAVOS_START = datetime(2026, 1, 21, 12, 0, 0)  # After Carney's speech
DAVOS_END = datetime(2026, 1, 24, 0, 0, 0)

# Initialize VADER
_vader = SentimentIntensityAnalyzer()


def get_db_connection():
    """Get database connection using config."""
    config = load_config()
    db_config = config.get("database", {})
    return psycopg2.connect(
        host=db_config.get("host", "10.0.0.4"),
        port=db_config.get("port", 5432),
        database=db_config.get("database", "av_content"),
        user=db_config.get("user", "signal4"),
        password=db_config.get("password") or get_credential("POSTGRES_PASSWORD", "")
    )


def classify_sentiment(text: str) -> str:
    """Classify text sentiment using VADER."""
    scores = _vader.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        return 'positive'
    elif compound <= -0.05:
        return 'negative'
    else:
        return 'neutral'


def extract_timeline_data() -> List[Dict]:
    """Extract daily aggregated sentiment data for timeline visualization."""
    print("\n[Timeline Data]")

    # Get total hours per day for Big_Channels
    total_hours_query = """
    SELECT DATE_TRUNC('day', c.publish_date) as day,
           SUM(c.duration) / 3600.0 as total_hours
    FROM content c
    WHERE c.projects && ARRAY['Big_Channels']::varchar[]
      AND c.publish_date >= %s
      AND c.publish_date <= %s
      AND c.duration IS NOT NULL
    GROUP BY DATE_TRUNC('day', c.publish_date);
    """

    # Get Canada-related segments with their durations
    canada_query = """
    SELECT DATE_TRUNC('day', c.publish_date) as day,
           es.text,
           (es.end_time - es.start_time) / 3600.0 as segment_hours
    FROM embedding_segments es
    JOIN content c ON es.content_id = c.id
    WHERE c.projects && ARRAY['Big_Channels']::varchar[]
      AND c.publish_date >= %s
      AND c.publish_date <= %s
      AND (es.text ILIKE '%%canada%%' OR es.text ILIKE '%%canadian%%'
           OR es.text ILIKE '%%carney%%' OR es.text ILIKE '%%trudeau%%')
    ORDER BY day;
    """

    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            print("  Querying total hours by day...")
            cur.execute(total_hours_query, (TIMELINE_START, TIMELINE_END))
            total_rows = cur.fetchall()
            total_hours_by_day = {row['day']: row['total_hours'] for row in total_rows}

            print("  Querying Canada-related segments...")
            cur.execute(canada_query, (TIMELINE_START, TIMELINE_END))
            canada_rows = cur.fetchall()

    print(f"  Found {len(canada_rows)} Canada-related segments")

    # Aggregate by day with sentiment and hours
    daily_data = defaultdict(lambda: {
        'positive_hours': 0.0, 'negative_hours': 0.0, 'neutral_hours': 0.0
    })

    for row in canada_rows:
        day = row['day']
        hours = row['segment_hours'] or 0.0
        sentiment = classify_sentiment(row['text'])
        daily_data[day][f'{sentiment}_hours'] += hours

    # Convert to sorted list with percentages
    result = []
    for day in sorted(daily_data.keys()):
        data = daily_data[day]
        total_day_hours = total_hours_by_day.get(day, 1.0)

        canada_hours = data['positive_hours'] + data['negative_hours'] + data['neutral_hours']
        pct_of_total = (canada_hours / total_day_hours * 100) if total_day_hours > 0 else 0

        result.append({
            'day': day.isoformat(),
            'positive': data['positive_hours'],
            'negative': data['negative_hours'],
            'neutral': data['neutral_hours'],
            'total': canada_hours,
            'pct_of_total': pct_of_total,
        })

    print(f"  Aggregated to {len(result)} days")
    return result


def extract_davos_segments() -> List[Dict]:
    """Extract individual segments from Davos period for classification."""
    print("\n[Davos Segments]")

    query = """
    SELECT es.id as segment_id,
           es.text,
           c.title as episode_title,
           ch.name as channel_name,
           c.publish_date,
           es.start_time,
           es.end_time
    FROM embedding_segments es
    JOIN content c ON es.content_id = c.id
    JOIN channels ch ON c.channel_id = ch.id
    WHERE c.projects && ARRAY['Big_Channels']::varchar[]
      AND c.publish_date >= %s
      AND c.publish_date < %s
      AND (es.text ILIKE '%%canada%%' OR es.text ILIKE '%%canadian%%'
           OR es.text ILIKE '%%carney%%')
    ORDER BY c.publish_date, es.start_time;
    """

    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            print(f"  Querying segments from {DAVOS_START} to {DAVOS_END}...")
            cur.execute(query, (DAVOS_START, DAVOS_END))
            rows = cur.fetchall()

    print(f"  Found {len(rows)} segments")

    # Convert to JSON-serializable format
    result = []
    for row in rows:
        result.append({
            'segment_id': row['segment_id'],
            'text': row['text'],
            'episode_title': row['episode_title'],
            'channel_name': row['channel_name'],
            'publish_date': row['publish_date'].isoformat() if row['publish_date'] else None,
            'start_time': row['start_time'],
            'end_time': row['end_time'],
        })

    return result


def main():
    print("=" * 60)
    print("US-Canada MEO Analysis - Data Extraction")
    print("=" * 60)

    # Extract timeline data
    timeline_data = extract_timeline_data()
    with open(TIMELINE_CACHE, 'w') as f:
        json.dump(timeline_data, f, indent=2)
    print(f"  Saved: {TIMELINE_CACHE}")

    # Extract Davos segments
    davos_segments = extract_davos_segments()
    with open(DAVOS_SEGMENTS_CACHE, 'w') as f:
        json.dump(davos_segments, f, indent=2)
    print(f"  Saved: {DAVOS_SEGMENTS_CACHE}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
