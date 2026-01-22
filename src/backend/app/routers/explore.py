"""
Explore Router
==============

API endpoints for exploring content within a project.
Provides statistics, recent episodes, and browse functionality.
"""

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import func, desc, text
from pydantic import BaseModel, Field
from typing import List, Optional
import time

from ..utils.backend_logger import get_logger
logger = get_logger("explore_router")

from src.database.session import get_session
from src.database.models import Content, Channel, EmbeddingSegment

router = APIRouter(prefix="/api/explore", tags=["explore"])


# ============================================================================
# Response Models
# ============================================================================

class ProjectStats(BaseModel):
    """Statistics for a project within a time window"""
    project: str
    time_window_days: int
    total_episodes: int
    total_channels: int
    total_segments: int
    total_duration_hours: float
    date_range: Optional[dict] = None  # {earliest: str, latest: str}
    processing_time_ms: float


class EpisodeCard(BaseModel):
    """Episode card data for display"""
    content_id: str
    content_id_numeric: int
    title: str
    channel_name: str
    channel_id: Optional[int] = None
    publish_date: Optional[str] = None
    duration: Optional[int] = None  # seconds
    platform: str
    description_short: Optional[str] = None  # Truncated to ~200 chars
    thumbnail_url: Optional[str] = None
    importance_score: Optional[float] = None  # Channel importance (for ranking)


class RecentEpisodesResponse(BaseModel):
    """Response for recent episodes listing"""
    success: bool = True
    project: str
    time_window_days: int
    total_available: int
    episodes: List[EpisodeCard]
    processing_time_ms: float


class ProjectStatsResponse(BaseModel):
    """Response for project statistics"""
    success: bool = True
    stats: ProjectStats


class HeatmapDataPoint(BaseModel):
    """Single data point for release heatmap"""
    date: str  # YYYY-MM-DD
    hour: int  # 0-23
    count: int


class HeatmapResponse(BaseModel):
    """Response for release heatmap data"""
    success: bool = True
    project: str
    data: List[HeatmapDataPoint]
    processing_time_ms: float


# ============================================================================
# Helper Functions
# ============================================================================

def _format_date(dt) -> Optional[str]:
    """Format datetime to ISO string or return None"""
    if dt is None:
        return None
    return dt.isoformat() if hasattr(dt, 'isoformat') else str(dt)


def _get_thumbnail_url(content_id: str, platform: str, channel_key: Optional[str] = None, has_channel_image: bool = False) -> Optional[str]:
    """Generate thumbnail URL based on platform.

    For YouTube: Uses img.youtube.com URL pattern.
    For podcasts: Uses frontend proxy to backend images API.
    """
    if platform == 'youtube' and content_id and not content_id.startswith('pod_'):
        return f"https://img.youtube.com/vi/{content_id}/mqdefault.jpg"
    if platform == 'podcast' and channel_key and has_channel_image:
        # URL-encode the channel_key to handle spaces and special characters
        from urllib.parse import quote
        encoded_key = quote(channel_key, safe='')
        # Route through frontend image proxy (no auth required)
        return f"/api/images/channels/{encoded_key}.jpg"
    return None


def _truncate_description(description: Optional[str], max_length: int = 200) -> Optional[str]:
    """Truncate description to max length, ending at word boundary"""
    if not description:
        return None
    if len(description) <= max_length:
        return description
    # Find last space before max_length
    truncated = description[:max_length]
    last_space = truncated.rfind(' ')
    if last_space > max_length * 0.7:  # Only truncate at word if we keep >70%
        truncated = truncated[:last_space]
    return truncated.rstrip() + '...'


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/stats/{project}", response_model=ProjectStatsResponse)
async def get_project_stats(
    project: str,
    time_window_days: int = Query(default=30, ge=1, le=90, description="Time window in days (1-90)")
):
    """
    Get statistics for a project within a time window.

    Returns counts of episodes, channels, segments, and total duration.
    Uses the embedding cache tables for efficient queries.
    """
    start_time = time.time()

    try:
        with get_session() as session:
            # Choose cache table based on time window
            if time_window_days <= 7:
                cache_table = "embedding_cache_7d"
            else:
                cache_table = "embedding_cache_30d"

            # Query stats from cache table
            # Note: Using raw SQL for efficiency with array contains operator
            stats_query = text(f"""
                SELECT
                    COUNT(DISTINCT content_id) as total_episodes,
                    COUNT(DISTINCT channel_name) as total_channels,
                    COUNT(*) as total_segments,
                    MIN(publish_date) as earliest_date,
                    MAX(publish_date) as latest_date
                FROM {cache_table}
                WHERE projects @> ARRAY[:project]::varchar[]
                AND publish_date >= NOW() - INTERVAL ':days days'
            """)

            result = session.execute(
                stats_query,
                {"project": project, "days": time_window_days}
            ).fetchone()

            if not result or result.total_episodes == 0:
                # No data found, return zeros
                processing_time = (time.time() - start_time) * 1000
                return ProjectStatsResponse(
                    success=True,
                    stats=ProjectStats(
                        project=project,
                        time_window_days=time_window_days,
                        total_episodes=0,
                        total_channels=0,
                        total_segments=0,
                        total_duration_hours=0.0,
                        date_range=None,
                        processing_time_ms=processing_time
                    )
                )

            # Get total duration from content table
            duration_query = text("""
                SELECT COALESCE(SUM(c.duration), 0) as total_seconds
                FROM content c
                WHERE c.projects @> ARRAY[:project]::varchar[]
                AND c.publish_date >= NOW() - INTERVAL ':days days'
                AND c.is_embedded = true
            """)

            duration_result = session.execute(
                duration_query,
                {"project": project, "days": time_window_days}
            ).fetchone()

            total_hours = (duration_result.total_seconds or 0) / 3600.0

            # Build date range
            date_range = None
            if result.earliest_date and result.latest_date:
                date_range = {
                    "earliest": _format_date(result.earliest_date),
                    "latest": _format_date(result.latest_date)
                }

            processing_time = (time.time() - start_time) * 1000

            return ProjectStatsResponse(
                success=True,
                stats=ProjectStats(
                    project=project,
                    time_window_days=time_window_days,
                    total_episodes=result.total_episodes,
                    total_channels=result.total_channels,
                    total_segments=result.total_segments,
                    total_duration_hours=round(total_hours, 1),
                    date_range=date_range,
                    processing_time_ms=processing_time
                )
            )

    except Exception as e:
        logger.error(f"Error fetching project stats for {project}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/heatmap/{project}", response_model=HeatmapResponse)
async def get_release_heatmap(
    project: str,
    days: int = Query(default=7, ge=1, le=30, description="Number of days (1-30)")
):
    """
    Get episode release counts by hour for heatmap visualization.

    Returns hourly counts for the specified number of days.
    """
    start_time = time.time()

    try:
        with get_session() as session:
            heatmap_query = text("""
                SELECT
                    DATE(publish_date) as date,
                    EXTRACT(HOUR FROM publish_date)::int as hour,
                    COUNT(*) as count
                FROM content
                WHERE publish_date >= CURRENT_DATE - INTERVAL ':days days'
                    AND publish_date < CURRENT_DATE + INTERVAL '1 day'
                    AND projects @> ARRAY[:project]::varchar[]
                    AND is_embedded = true
                GROUP BY DATE(publish_date), EXTRACT(HOUR FROM publish_date)
                ORDER BY date, hour
            """)

            results = session.execute(
                heatmap_query,
                {"project": project, "days": days}
            ).fetchall()

            data = [
                HeatmapDataPoint(
                    date=str(row.date),
                    hour=row.hour,
                    count=row.count
                )
                for row in results
            ]

            processing_time = (time.time() - start_time) * 1000

            return HeatmapResponse(
                success=True,
                project=project,
                data=data,
                processing_time_ms=processing_time
            )

    except Exception as e:
        logger.error(f"Error fetching heatmap data for {project}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/recent/{project}", response_model=RecentEpisodesResponse)
async def get_recent_episodes(
    project: str,
    time_window_days: int = Query(default=30, ge=1, le=90, description="Time window in days (1-90)"),
    limit: int = Query(default=20, ge=1, le=100, description="Number of episodes to return"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
    channel_id: Optional[int] = Query(default=None, description="Filter by channel ID"),
    start_date: Optional[str] = Query(default=None, description="Filter by start date (ISO format)"),
    end_date: Optional[str] = Query(default=None, description="Filter by end date (ISO format)"),
    search: Optional[str] = Query(default=None, max_length=200, description="Search in title or description")
):
    """
    Get recent episodes for a project, sorted by publish date descending.

    Returns episode cards with title, channel, date, thumbnail, and short description.
    Supports filtering by channel, date range, and keyword search.
    """
    start_time = time.time()

    try:
        with get_session() as session:
            # Build dynamic WHERE clause
            where_clauses = [
                "c.projects @> ARRAY[:project]::varchar[]",
                "c.is_embedded = true"
            ]
            params = {"project": project, "limit": limit, "offset": offset}

            # Date filtering - use time_window_days OR explicit date range
            if start_date and end_date:
                where_clauses.append("c.publish_date >= :start_date::timestamp")
                where_clauses.append("c.publish_date <= :end_date::timestamp")
                params["start_date"] = start_date
                params["end_date"] = end_date
            elif start_date:
                where_clauses.append("c.publish_date >= :start_date::timestamp")
                params["start_date"] = start_date
            elif end_date:
                where_clauses.append("c.publish_date <= :end_date::timestamp")
                params["end_date"] = end_date
            else:
                # Default: use time_window_days
                where_clauses.append("c.publish_date >= NOW() - INTERVAL ':days days'")
                params["days"] = time_window_days

            # Channel filter
            if channel_id is not None:
                where_clauses.append("c.channel_id = :channel_id")
                params["channel_id"] = channel_id

            # Keyword search (title or description)
            if search:
                where_clauses.append("(c.title ILIKE :search OR c.description ILIKE :search)")
                params["search"] = f"%{search}%"

            where_sql = " AND ".join(where_clauses)

            # Optimized query with recency + importance weighting
            # Only shows most recent episode per channel (DISTINCT ON channel_id)
            # Score = importance_score * recency_factor
            # recency_factor decays: 1.0 at day 0, ~0.35 at day 7, ~0.01 at day 30
            episodes_query = text(f"""
                SELECT * FROM (
                    SELECT DISTINCT ON (c.channel_id)
                        c.id,
                        c.content_id,
                        c.title,
                        c.channel_name,
                        c.channel_id,
                        c.publish_date,
                        c.duration,
                        c.platform,
                        c.description,
                        ch.channel_key,
                        ch.importance_score,
                        CASE WHEN ch.platform_metadata->>'image_url' IS NOT NULL
                             AND ch.platform_metadata->>'image_url' <> ''
                             THEN true ELSE false END as has_channel_image,
                        COALESCE(ch.importance_score, 1000) *
                            EXP(-0.15 * EXTRACT(EPOCH FROM (NOW() - c.publish_date)) / 86400) as weighted_score
                    FROM content c
                    LEFT JOIN channels ch ON c.channel_id = ch.id
                    WHERE {where_sql}
                    ORDER BY c.channel_id, c.publish_date DESC
                ) sub
                ORDER BY weighted_score DESC
                LIMIT :limit OFFSET :offset
            """)

            results = session.execute(episodes_query, params).fetchall()

            # For top episodes view, total_available = number returned (not paginated)
            total_available = len(results)

            episodes = []
            for row in results:
                episodes.append(EpisodeCard(
                    content_id=row.content_id,
                    content_id_numeric=row.id,
                    title=row.title or "Untitled",
                    channel_name=row.channel_name or "Unknown Channel",
                    channel_id=row.channel_id,
                    publish_date=_format_date(row.publish_date),
                    duration=int(row.duration) if row.duration else None,
                    platform=row.platform or "unknown",
                    description_short=_truncate_description(row.description),
                    thumbnail_url=_get_thumbnail_url(row.content_id, row.platform or "", row.channel_key, row.has_channel_image),
                    importance_score=float(row.importance_score) if row.importance_score else None
                ))

            processing_time = (time.time() - start_time) * 1000

            return RecentEpisodesResponse(
                success=True,
                project=project,
                time_window_days=time_window_days,
                total_available=total_available,
                episodes=episodes,
                processing_time_ms=processing_time
            )

    except Exception as e:
        logger.error(f"Error fetching recent episodes for {project}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
