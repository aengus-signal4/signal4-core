"""
Database query functions for methodology generation.

These functions query the PostgreSQL database to extract statistics
and metadata for the methodology section.
"""

import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import date, datetime
from typing import List, Dict, Optional, Any, Set
from dataclasses import dataclass
from contextlib import contextmanager

try:
    from .config import DB_CONFIG, EXCLUDED_LANGUAGES
except ImportError:
    from config import DB_CONFIG, EXCLUDED_LANGUAGES


@dataclass
class CorpusStats:
    """Statistics for a project or the entire corpus."""
    project: str
    total_episodes: int
    transcribed_episodes: int
    total_hours: float
    avg_duration_minutes: float
    channel_count: int
    earliest_date: Optional[date]
    latest_date: Optional[date]


@dataclass
class ChannelInfo:
    """Information about a podcast channel."""
    id: int
    name: str
    description: Optional[str]
    language: str
    platform: str
    project: str
    episode_count: int
    chart_rank: Optional[int]
    chart_platform: Optional[str]
    chart_country: Optional[str]
    chart_category: Optional[str]


@dataclass
class LanguageStats:
    """Statistics by language."""
    language: str
    episode_count: int
    transcribed_count: int
    hours: float
    channel_count: int


@dataclass
class YearStats:
    """Statistics by year."""
    year: int
    episode_count: int
    transcribed_count: int


@contextmanager
def get_connection():
    """Get a database connection with context manager."""
    conn = psycopg2.connect(
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"],
        database=DB_CONFIG["database"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"]
    )
    try:
        yield conn
    finally:
        conn.close()


def get_corpus_statistics(
    projects: List[str],
    start_date: date,
    end_date: date,
    exclude_languages: Optional[Set[str]] = None
) -> Dict[str, CorpusStats]:
    """
    Get corpus statistics for each project.

    Args:
        projects: List of project names to include
        start_date: Start date for filtering
        end_date: End date for filtering
        exclude_languages: Set of language codes to exclude

    Returns:
        Dictionary mapping project name to CorpusStats
    """
    exclude_languages = exclude_languages or EXCLUDED_LANGUAGES

    query = """
    WITH project_stats AS (
        SELECT
            unnested as project,
            COUNT(*) as total_episodes,
            COUNT(CASE WHEN is_transcribed = true THEN 1 END) as transcribed_episodes,
            COALESCE(SUM(CASE WHEN is_transcribed = true THEN duration END) / 3600.0, 0) as total_hours,
            COALESCE(AVG(CASE WHEN is_transcribed = true THEN duration END) / 60.0, 0) as avg_duration,
            MIN(publish_date) as earliest_date,
            MAX(publish_date) as latest_date
        FROM content, LATERAL unnest(projects) as unnested
        WHERE unnested = ANY(%s)
          AND publish_date >= %s
          AND publish_date <= %s
          AND (main_language IS NULL OR main_language NOT IN %s)
        GROUP BY unnested
    ),
    channel_counts AS (
        SELECT
            project_name as project,
            COUNT(*) as channel_count
        FROM channel_projects
        WHERE project_name = ANY(%s)
        GROUP BY project_name
    )
    SELECT
        ps.project,
        ps.total_episodes,
        ps.transcribed_episodes,
        ps.total_hours,
        ps.avg_duration,
        COALESCE(cc.channel_count, 0) as channel_count,
        ps.earliest_date,
        ps.latest_date
    FROM project_stats ps
    LEFT JOIN channel_counts cc ON ps.project = cc.project
    ORDER BY ps.project;
    """

    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (
                projects,
                start_date,
                end_date,
                tuple(exclude_languages),
                projects
            ))
            rows = cur.fetchall()

    results = {}
    for row in rows:
        results[row['project']] = CorpusStats(
            project=row['project'],
            total_episodes=row['total_episodes'],
            transcribed_episodes=row['transcribed_episodes'],
            total_hours=float(row['total_hours'] or 0),
            avg_duration_minutes=float(row['avg_duration'] or 0),
            channel_count=row['channel_count'],
            earliest_date=row['earliest_date'].date() if row['earliest_date'] else None,
            latest_date=row['latest_date'].date() if row['latest_date'] else None
        )

    return results


def get_channel_list(
    projects: List[str],
    exclude_languages: Optional[Set[str]] = None,
    chart_month: str = "2025-12"
) -> List[ChannelInfo]:
    """
    Get full list of channels with metadata and chart rankings.

    Args:
        projects: List of project names to include
        exclude_languages: Set of language codes to exclude
        chart_month: Month for chart rankings (YYYY-MM format)

    Returns:
        List of ChannelInfo objects
    """
    exclude_languages = exclude_languages or EXCLUDED_LANGUAGES

    query = """
    WITH channel_projects_filtered AS (
        SELECT DISTINCT cp.channel_id, cp.project_name
        FROM channel_projects cp
        JOIN channels c ON cp.channel_id = c.id
        WHERE cp.project_name = ANY(%s)
          AND (c.language IS NULL OR c.language NOT IN %s)
    ),
    channel_episodes AS (
        SELECT
            ch.id as channel_id,
            COUNT(DISTINCT c.id) as episode_count
        FROM channels ch
        JOIN content c ON c.channel_id = ch.id
        WHERE ch.id IN (SELECT channel_id FROM channel_projects_filtered)
        GROUP BY ch.id
    ),
    best_ranks AS (
        SELECT DISTINCT ON (pc.channel_id)
            pc.channel_id,
            pc.rank,
            pc.platform,
            pc.country,
            pc.category
        FROM podcast_charts pc
        WHERE pc.month = %s
          AND pc.channel_id IN (SELECT channel_id FROM channel_projects_filtered)
        ORDER BY pc.channel_id, pc.rank ASC
    )
    SELECT
        ch.id,
        ch.display_name as name,
        ch.description,
        COALESCE(ch.language, 'en') as language,
        ch.platform,
        cpf.project_name as project,
        COALESCE(ce.episode_count, 0) as episode_count,
        br.rank as chart_rank,
        br.platform as chart_platform,
        br.country as chart_country,
        br.category as chart_category
    FROM channels ch
    JOIN channel_projects_filtered cpf ON ch.id = cpf.channel_id
    LEFT JOIN channel_episodes ce ON ch.id = ce.channel_id
    LEFT JOIN best_ranks br ON ch.id = br.channel_id
    ORDER BY
        cpf.project_name,
        COALESCE(br.rank, 9999),
        ch.display_name;
    """

    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (
                projects,
                tuple(exclude_languages),
                chart_month
            ))
            rows = cur.fetchall()

    return [
        ChannelInfo(
            id=row['id'],
            name=row['name'],
            description=row['description'],
            language=row['language'],
            platform=row['platform'],
            project=row['project'],
            episode_count=row['episode_count'],
            chart_rank=row['chart_rank'],
            chart_platform=row['chart_platform'],
            chart_country=row['chart_country'],
            chart_category=row['chart_category']
        )
        for row in rows
    ]


def get_language_distribution(
    projects: List[str],
    start_date: date,
    end_date: date,
    exclude_languages: Optional[Set[str]] = None
) -> List[LanguageStats]:
    """
    Get episode and hour counts by language.

    Args:
        projects: List of project names to include
        start_date: Start date for filtering
        end_date: End date for filtering
        exclude_languages: Set of language codes to exclude

    Returns:
        List of LanguageStats sorted by episode count descending
    """
    exclude_languages = exclude_languages or EXCLUDED_LANGUAGES

    query = """
    WITH language_episodes AS (
        SELECT
            COALESCE(main_language, 'unknown') as language,
            COUNT(*) as episode_count,
            COUNT(CASE WHEN is_transcribed = true THEN 1 END) as transcribed_count,
            COALESCE(SUM(CASE WHEN is_transcribed = true THEN duration END) / 3600.0, 0) as hours
        FROM content, LATERAL unnest(projects) as unnested
        WHERE unnested = ANY(%s)
          AND publish_date >= %s
          AND publish_date <= %s
          AND (main_language IS NULL OR main_language NOT IN %s)
        GROUP BY COALESCE(main_language, 'unknown')
    ),
    language_channels AS (
        SELECT
            COALESCE(c.language, 'unknown') as language,
            COUNT(DISTINCT c.id) as channel_count
        FROM channels c
        JOIN channel_projects cp ON c.id = cp.channel_id
        WHERE cp.project_name = ANY(%s)
          AND (c.language IS NULL OR c.language NOT IN %s)
        GROUP BY COALESCE(c.language, 'unknown')
    )
    SELECT
        le.language,
        le.episode_count,
        le.transcribed_count,
        le.hours,
        COALESCE(lc.channel_count, 0) as channel_count
    FROM language_episodes le
    LEFT JOIN language_channels lc ON le.language = lc.language
    ORDER BY le.episode_count DESC;
    """

    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (
                projects,
                start_date,
                end_date,
                tuple(exclude_languages),
                projects,
                tuple(exclude_languages)
            ))
            rows = cur.fetchall()

    return [
        LanguageStats(
            language=row['language'],
            episode_count=row['episode_count'],
            transcribed_count=row['transcribed_count'],
            hours=float(row['hours'] or 0),
            channel_count=row['channel_count']
        )
        for row in rows
    ]


def get_year_distribution(
    projects: List[str],
    start_date: date,
    end_date: date,
    exclude_languages: Optional[Set[str]] = None
) -> List[YearStats]:
    """
    Get episode counts by year.

    Args:
        projects: List of project names to include
        start_date: Start date for filtering
        end_date: End date for filtering
        exclude_languages: Set of language codes to exclude

    Returns:
        List of YearStats sorted by year ascending
    """
    exclude_languages = exclude_languages or EXCLUDED_LANGUAGES

    query = """
    SELECT
        EXTRACT(YEAR FROM publish_date)::int as year,
        COUNT(*) as episode_count,
        COUNT(CASE WHEN is_transcribed = true THEN 1 END) as transcribed_count
    FROM content, LATERAL unnest(projects) as unnested
    WHERE unnested = ANY(%s)
      AND publish_date >= %s
      AND publish_date <= %s
      AND (main_language IS NULL OR main_language NOT IN %s)
    GROUP BY EXTRACT(YEAR FROM publish_date)
    ORDER BY year;
    """

    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (
                projects,
                start_date,
                end_date,
                tuple(exclude_languages)
            ))
            rows = cur.fetchall()

    return [
        YearStats(
            year=row['year'],
            episode_count=row['episode_count'],
            transcribed_count=row['transcribed_count']
        )
        for row in rows
    ]


def get_processing_statistics(
    projects: List[str],
    exclude_languages: Optional[Set[str]] = None
) -> Dict[str, Any]:
    """
    Get detailed processing statistics (segments, embeddings, etc.).

    Args:
        projects: List of project names to include
        exclude_languages: Set of language codes to exclude

    Returns:
        Dictionary with processing statistics
    """
    exclude_languages = exclude_languages or EXCLUDED_LANGUAGES

    # Query for embedding segments
    segment_query = """
    SELECT
        unnested as project,
        COUNT(*) as segment_count,
        COUNT(DISTINCT es.content_id) as content_count,
        AVG(es.token_count) as avg_tokens_per_segment
    FROM embedding_segments es
    JOIN content c ON es.content_id = c.id
    CROSS JOIN LATERAL unnest(c.projects) as unnested
    WHERE unnested = ANY(%s)
      AND (c.main_language IS NULL OR c.main_language NOT IN %s)
    GROUP BY unnested;
    """

    # Query for speaker transcriptions
    speaker_query = """
    SELECT
        unnested as project,
        COUNT(*) as turn_count,
        COUNT(DISTINCT st.content_id) as content_count,
        COUNT(DISTINCT st.speaker_id) as unique_speakers
    FROM speaker_transcriptions st
    JOIN content c ON st.content_id = c.id
    CROSS JOIN LATERAL unnest(c.projects) as unnested
    WHERE unnested = ANY(%s)
      AND (c.main_language IS NULL OR c.main_language NOT IN %s)
    GROUP BY unnested;
    """

    # Query for chart coverage
    chart_query = """
    SELECT
        COUNT(DISTINCT channel_id) as channels_with_charts,
        COUNT(DISTINCT month) as months_collected,
        COUNT(DISTINCT country) as countries,
        COUNT(DISTINCT platform) as platforms
    FROM podcast_charts
    WHERE month LIKE '2025%';
    """

    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Segment stats
            cur.execute(segment_query, (projects, tuple(exclude_languages)))
            segment_rows = cur.fetchall()

            # Speaker stats
            cur.execute(speaker_query, (projects, tuple(exclude_languages)))
            speaker_rows = cur.fetchall()

            # Chart stats
            cur.execute(chart_query)
            chart_row = cur.fetchone()

    return {
        "segments": {row['project']: {
            "count": row['segment_count'],
            "content_count": row['content_count'],
            "avg_tokens": float(row['avg_tokens_per_segment'] or 0)
        } for row in segment_rows},
        "speakers": {row['project']: {
            "turn_count": row['turn_count'],
            "content_count": row['content_count'],
            "unique_speakers": row['unique_speakers']
        } for row in speaker_rows},
        "charts": chart_row
    }


def get_chart_rankings_summary(chart_month: str = "2025-12") -> Dict[str, Any]:
    """
    Get summary of chart rankings for a given month.

    Args:
        chart_month: Month in YYYY-MM format

    Returns:
        Dictionary with chart ranking statistics
    """
    query = """
    SELECT
        platform,
        COUNT(DISTINCT channel_id) as unique_channels,
        COUNT(*) as total_entries,
        COUNT(DISTINCT country) as countries,
        COUNT(DISTINCT category) as categories
    FROM podcast_charts
    WHERE month = %s
    GROUP BY platform
    ORDER BY platform;
    """

    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (chart_month,))
            rows = cur.fetchall()

    return {
        "month": chart_month,
        "platforms": {
            row['platform']: {
                "unique_channels": row['unique_channels'],
                "total_entries": row['total_entries'],
                "countries": row['countries'],
                "categories": row['categories']
            }
            for row in rows
        }
    }


def get_total_word_count_estimate(
    projects: List[str],
    exclude_languages: Optional[Set[str]] = None
) -> Dict[str, int]:
    """
    Estimate total word count from embedding segments (faster than full calculation).

    Uses average tokens per segment * segment count as estimate.

    Args:
        projects: List of project names to include
        exclude_languages: Set of language codes to exclude

    Returns:
        Dictionary mapping project to estimated word count
    """
    exclude_languages = exclude_languages or EXCLUDED_LANGUAGES

    # Use token count as rough word estimate (tokens ~ 0.75 * words for English)
    query = """
    SELECT
        unnested as project,
        SUM(es.token_count) as total_tokens
    FROM embedding_segments es
    JOIN content c ON es.content_id = c.id
    CROSS JOIN LATERAL unnest(c.projects) as unnested
    WHERE unnested = ANY(%s)
      AND (c.main_language IS NULL OR c.main_language NOT IN %s)
    GROUP BY unnested;
    """

    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (projects, tuple(exclude_languages)))
            rows = cur.fetchall()

    # Convert tokens to words (rough estimate: 1 token ~ 0.75 words)
    return {
        row['project']: int(row['total_tokens'] * 0.75) if row['total_tokens'] else 0
        for row in rows
    }
