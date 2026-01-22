"""
Report Router
=============

Public API endpoints for the daily report feature.
All endpoints are unauthenticated (no API key required).

Endpoints:
- GET /api/report/daily - Combined daily report data
- GET /api/report/stats - Platform statistics by project
- GET /api/report/recent - Recent content/episodes
- GET /api/report/channels/top - Top channels by content volume
- GET /api/report/speakers/top - Top speakers by appearances
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from ..utils.backend_logger import get_logger
logger = get_logger("report_router")

from ..services.report_service import get_report_service

router = APIRouter(prefix="/api/report", tags=["report"])


# ============================================================================
# Response Models
# ============================================================================

class PlatformStats(BaseModel):
    """Statistics for a single platform"""
    episodes: int
    channels: int
    duration_hours: float


class TotalStats(BaseModel):
    """Aggregated totals across all platforms"""
    episodes: int
    channels: int
    speakers: int
    duration_hours: float


class StatsResponse(BaseModel):
    """Response for /stats endpoint"""
    success: bool = True
    time_window_days: int
    projects_filter: Optional[List[str]] = None
    by_platform: Dict[str, PlatformStats]
    totals: TotalStats
    processing_time_ms: float


class EpisodeCard(BaseModel):
    """Episode card for recent content display"""
    content_id: str
    title: str
    channel_name: str
    platform: str
    publish_date: Optional[str] = None
    duration_minutes: Optional[float] = None
    description_short: Optional[str] = None
    thumbnail_url: Optional[str] = None


class RecentContentResponse(BaseModel):
    """Response for /recent endpoint"""
    success: bool = True
    time_window_days: int
    projects_filter: Optional[List[str]] = None
    total_returned: int
    episodes: List[EpisodeCard]
    processing_time_ms: float


class ChannelCard(BaseModel):
    """Channel card for top channels display"""
    channel_id: int
    name: str
    platform: str
    episode_count: int
    total_duration_hours: float
    importance_score: Optional[float] = None


class TopChannelsResponse(BaseModel):
    """Response for /channels/top endpoint"""
    success: bool = True
    time_window_days: int
    projects_filter: Optional[List[str]] = None
    channels: List[ChannelCard]
    processing_time_ms: float


class SpeakerCard(BaseModel):
    """Speaker card for top speakers display"""
    speaker_id: int
    name: str
    occupation: Optional[str] = None
    role: Optional[str] = None
    recent_appearances: int
    total_appearances: Optional[int] = None


class TopSpeakersResponse(BaseModel):
    """Response for /speakers/top endpoint"""
    success: bool = True
    time_window_days: int
    projects_filter: Optional[List[str]] = None
    speakers: List[SpeakerCard]
    processing_time_ms: float


class DailyReportResponse(BaseModel):
    """Response for /daily endpoint - combined report"""
    success: bool = True
    report_date: str
    projects_filter: Optional[List[str]] = None
    stats: Dict[str, Any]
    recent_content: Dict[str, Any]
    top_channels: Dict[str, Any]
    top_speakers: Dict[str, Any]
    total_processing_time_ms: float


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/daily", response_model=DailyReportResponse)
async def get_daily_report(
    projects: Optional[str] = Query(
        default=None,
        description="Comma-separated list of projects to include (e.g., 'podcasts,health')"
    )
):
    """
    Get combined daily report data.

    Returns stats, recent content, top channels, and top speakers
    all in a single response. Useful for rendering the full report page.

    This endpoint is PUBLIC (no API key required).
    """
    try:
        service = get_report_service()

        # Parse projects parameter
        projects_list = None
        if projects:
            projects_list = [p.strip() for p in projects.split(",") if p.strip()]

        result = service.get_daily_report(projects=projects_list)

        return DailyReportResponse(
            success=True,
            report_date=result["report_date"],
            projects_filter=result["projects_filter"],
            stats=result["stats"],
            recent_content=result["recent_content"],
            top_channels=result["top_channels"],
            top_speakers=result["top_speakers"],
            total_processing_time_ms=result["total_processing_time_ms"]
        )

    except Exception as e:
        logger.error(f"Error generating daily report: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/stats", response_model=StatsResponse)
async def get_platform_stats(
    projects: Optional[str] = Query(
        default=None,
        description="Comma-separated list of projects to include"
    ),
    days: int = Query(
        default=30,
        ge=1,
        le=365,
        description="Time window in days (1-365)"
    )
):
    """
    Get platform statistics aggregated by project.

    Returns episode counts, channel counts, and total duration
    broken down by platform (youtube, podcast, rumble, etc.).

    This endpoint is PUBLIC (no API key required).
    """
    try:
        service = get_report_service()

        # Parse projects parameter
        projects_list = None
        if projects:
            projects_list = [p.strip() for p in projects.split(",") if p.strip()]

        result = service.get_platform_stats(projects=projects_list, days=days)

        return StatsResponse(
            success=True,
            time_window_days=result["time_window_days"],
            projects_filter=result["projects_filter"],
            by_platform=result["by_platform"],
            totals=result["totals"],
            processing_time_ms=result["processing_time_ms"]
        )

    except Exception as e:
        logger.error(f"Error fetching platform stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/recent", response_model=RecentContentResponse)
async def get_recent_content(
    projects: Optional[str] = Query(
        default=None,
        description="Comma-separated list of projects to include"
    ),
    days: int = Query(
        default=7,
        ge=1,
        le=30,
        description="Time window in days (1-30)"
    ),
    limit: int = Query(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of episodes to return (1-100)"
    )
):
    """
    Get recently published content/episodes.

    Returns the most recent episodes with title, channel, date,
    duration, and thumbnail URL.

    This endpoint is PUBLIC (no API key required).
    """
    try:
        service = get_report_service()

        # Parse projects parameter
        projects_list = None
        if projects:
            projects_list = [p.strip() for p in projects.split(",") if p.strip()]

        result = service.get_recent_content(
            projects=projects_list,
            days=days,
            limit=limit
        )

        return RecentContentResponse(
            success=True,
            time_window_days=result["time_window_days"],
            projects_filter=result["projects_filter"],
            total_returned=result["total_returned"],
            episodes=result["episodes"],
            processing_time_ms=result["processing_time_ms"]
        )

    except Exception as e:
        logger.error(f"Error fetching recent content: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/channels/top", response_model=TopChannelsResponse)
async def get_top_channels(
    projects: Optional[str] = Query(
        default=None,
        description="Comma-separated list of projects to include"
    ),
    days: int = Query(
        default=30,
        ge=1,
        le=365,
        description="Time window in days (1-365)"
    ),
    limit: int = Query(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of channels to return (1-50)"
    )
):
    """
    Get top channels by content volume.

    Returns channels ranked by episode count within the time window,
    with total duration and importance scores.

    This endpoint is PUBLIC (no API key required).
    """
    try:
        service = get_report_service()

        # Parse projects parameter
        projects_list = None
        if projects:
            projects_list = [p.strip() for p in projects.split(",") if p.strip()]

        result = service.get_top_channels(
            projects=projects_list,
            days=days,
            limit=limit
        )

        return TopChannelsResponse(
            success=True,
            time_window_days=result["time_window_days"],
            projects_filter=result["projects_filter"],
            channels=result["channels"],
            processing_time_ms=result["processing_time_ms"]
        )

    except Exception as e:
        logger.error(f"Error fetching top channels: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/speakers/top", response_model=TopSpeakersResponse)
async def get_top_speakers(
    projects: Optional[str] = Query(
        default=None,
        description="Comma-separated list of projects to include"
    ),
    days: int = Query(
        default=30,
        ge=1,
        le=365,
        description="Time window in days (1-365)"
    ),
    limit: int = Query(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of speakers to return (1-50)"
    )
):
    """
    Get top speakers by appearances.

    Returns speakers ranked by episode appearances within the time window,
    with occupation and role information when available.

    This endpoint is PUBLIC (no API key required).
    """
    try:
        service = get_report_service()

        # Parse projects parameter
        projects_list = None
        if projects:
            projects_list = [p.strip() for p in projects.split(",") if p.strip()]

        result = service.get_top_speakers(
            projects=projects_list,
            days=days,
            limit=limit
        )

        return TopSpeakersResponse(
            success=True,
            time_window_days=result["time_window_days"],
            projects_filter=result["projects_filter"],
            speakers=result["speakers"],
            processing_time_ms=result["processing_time_ms"]
        )

    except Exception as e:
        logger.error(f"Error fetching top speakers: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
