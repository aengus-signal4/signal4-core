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
    segment_count: Optional[int] = None


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
    For podcasts: Uses image proxy URL if channel has an image stored.
    """
    if platform == 'youtube' and content_id and not content_id.startswith('pod_'):
        return f"https://img.youtube.com/vi/{content_id}/mqdefault.jpg"
    if platform == 'podcast' and channel_key and has_channel_image:
        return f"/api/images/channels/{channel_key}.jpg"
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


@router.get("/recent/{project}", response_model=RecentEpisodesResponse)
async def get_recent_episodes(
    project: str,
    time_window_days: int = Query(default=30, ge=1, le=90, description="Time window in days (1-90)"),
    limit: int = Query(default=20, ge=1, le=100, description="Number of episodes to return"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination")
):
    """
    Get recent episodes for a project, sorted by publish date descending.

    Returns episode cards with title, channel, date, thumbnail, and short description.
    """
    start_time = time.time()

    try:
        with get_session() as session:
            # Get total count first
            count_query = text("""
                SELECT COUNT(DISTINCT c.id) as total
                FROM content c
                WHERE c.projects @> ARRAY[:project]::varchar[]
                AND c.publish_date >= NOW() - INTERVAL ':days days'
                AND c.is_embedded = true
            """)

            count_result = session.execute(
                count_query,
                {"project": project, "days": time_window_days}
            ).fetchone()

            total_available = count_result.total if count_result else 0

            # Get episodes with segment counts and channel info for thumbnails
            episodes_query = text("""
                SELECT
                    c.id,
                    c.content_id,
                    c.title,
                    c.channel_name,
                    c.channel_id,
                    c.publish_date,
                    c.duration,
                    c.platform,
                    c.description,
                    COUNT(es.id) as segment_count,
                    ch.channel_key,
                    CASE WHEN ch.platform_metadata->>'image_url' IS NOT NULL
                         AND ch.platform_metadata->>'image_url' != ''
                         THEN true ELSE false END as has_channel_image
                FROM content c
                LEFT JOIN embedding_segments es ON c.id = es.content_id
                LEFT JOIN channels ch ON c.channel_id = ch.id
                WHERE c.projects @> ARRAY[:project]::varchar[]
                AND c.publish_date >= NOW() - INTERVAL ':days days'
                AND c.is_embedded = true
                GROUP BY c.id, ch.channel_key, ch.platform_metadata
                ORDER BY c.publish_date DESC
                LIMIT :limit OFFSET :offset
            """)

            results = session.execute(
                episodes_query,
                {"project": project, "days": time_window_days, "limit": limit, "offset": offset}
            ).fetchall()

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
                    segment_count=row.segment_count
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
