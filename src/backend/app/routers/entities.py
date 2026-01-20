"""
Entity Details Router
=====================

API endpoints for retrieving detailed information about episodes, speakers, and channels.
These endpoints support the site-wide entity modals in the frontend.
"""

from fastapi import APIRouter, HTTPException, Path, Query
from sqlalchemy import func, desc, distinct
from sqlalchemy.orm import joinedload, selectinload
import time
import logging

from ..utils.backend_logger import get_logger
logger = get_logger("entities_router")

from ..models.responses import (
    EpisodeDetailsResponse,
    SpeakerDetailsResponse,
    ChannelDetailsResponse,
    EpisodePreview,
    SpeakerPreview,
    ChannelPreview,
    ChannelWithCount,
    DateRange,
)

from src.database.session import get_session
from src.database.models import (
    Content,
    Channel,
    Speaker,
    SpeakerIdentity,
    EmbeddingSegment,
)

router = APIRouter(prefix="/api/entities", tags=["entities"])


def _format_date(dt) -> str | None:
    """Format datetime to ISO string or return None"""
    if dt is None:
        return None
    return dt.isoformat() if hasattr(dt, 'isoformat') else str(dt)


def _get_youtube_thumbnail(content_id: str) -> str | None:
    """Generate YouTube thumbnail URL from content ID"""
    if content_id and not content_id.startswith('pod_'):
        return f"https://img.youtube.com/vi/{content_id}/mqdefault.jpg"
    return None


def _get_source_url(content: Content) -> str | None:
    """Extract source URL from content metadata"""
    if content.meta_data and isinstance(content.meta_data, dict):
        return content.meta_data.get('episode_url') or content.meta_data.get('source_url')
    return None


@router.get("/episodes/{content_id}", response_model=EpisodeDetailsResponse)
async def get_episode_details(
    content_id: str = Path(..., description="Content ID (e.g., 'dQw4w9WgXcQ' or 'pod_abc123')")
):
    """
    Get detailed information about an episode.

    Returns full episode details including speakers, segment count, and related episodes.
    """
    start_time = time.time()

    try:
        with get_session() as session:
            # Get content with channel relationship
            content = session.query(Content)\
                .options(joinedload(Content.channel))\
                .filter(Content.content_id == content_id)\
                .first()

            if not content:
                raise HTTPException(status_code=404, detail=f"Episode not found: {content_id}")

            # Get segment count
            segment_count = session.query(func.count(EmbeddingSegment.id))\
                .filter(EmbeddingSegment.content_id == content.id)\
                .scalar() or 0

            # Get speakers for this episode
            speakers_query = session.query(Speaker)\
                .options(joinedload(Speaker.speaker_identity))\
                .filter(Speaker.content_id == content_id)\
                .all()

            speakers = []
            for speaker in speakers_query:
                name = speaker.display_name
                if speaker.speaker_identity:
                    name = speaker.speaker_identity.primary_name or name
                if name:
                    speakers.append(SpeakerPreview(
                        speaker_id=speaker.speaker_hash,
                        name=name,
                        appearance_count=speaker.segment_count,
                        image_url=None  # TODO: Add image URL when available
                    ))

            # Get related episodes (same channel, excluding current)
            related_episodes = []
            if content.channel_id:
                related_query = session.query(Content)\
                    .filter(Content.channel_id == content.channel_id)\
                    .filter(Content.id != content.id)\
                    .filter(Content.is_embedded == True)\
                    .order_by(desc(Content.publish_date))\
                    .limit(5)\
                    .all()

                for ep in related_query:
                    related_episodes.append(EpisodePreview(
                        content_id=ep.content_id,
                        content_id_numeric=ep.id,
                        title=ep.title or "Untitled",
                        channel_name=ep.channel_name,
                        channel_id=ep.channel_id,
                        publish_date=_format_date(ep.publish_date),
                        duration=int(ep.duration) if ep.duration else None,
                        platform=ep.platform,
                    ))

            # Build channel info
            channel_key = None
            if content.channel:
                channel_key = content.channel.channel_key

            processing_time = (time.time() - start_time) * 1000

            return EpisodeDetailsResponse(
                success=True,
                content_id=content.id,
                content_id_string=content.content_id,
                title=content.title or "Untitled",
                description=content.description,
                channel_id=content.channel_id,
                channel_name=content.channel_name,
                channel_key=channel_key,
                platform=content.platform,
                publish_date=_format_date(content.publish_date),
                duration=int(content.duration) if content.duration else None,
                main_language=content.main_language,
                has_transcript=content.is_transcribed,
                has_diarization=content.is_diarized,
                has_embeddings=content.is_embedded,
                source_url=_get_source_url(content),
                thumbnail_url=_get_youtube_thumbnail(content.content_id),
                speakers=speakers,
                segment_count=segment_count,
                related_episodes=related_episodes,
                processing_time_ms=processing_time,
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching episode details for {content_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/channels/{channel_id}", response_model=ChannelDetailsResponse)
async def get_channel_details(
    channel_id: int = Path(..., description="Channel ID from database")
):
    """
    Get detailed information about a channel.

    Returns full channel details including episode count, date range,
    regular speakers, and recent episodes.
    """
    start_time = time.time()

    try:
        with get_session() as session:
            # Get channel
            channel = session.query(Channel)\
                .filter(Channel.id == channel_id)\
                .first()

            if not channel:
                raise HTTPException(status_code=404, detail=f"Channel not found: {channel_id}")

            # Get episode stats
            stats = session.query(
                func.count(Content.id).label('episode_count'),
                func.min(Content.publish_date).label('earliest'),
                func.max(Content.publish_date).label('latest')
            ).filter(Content.channel_id == channel_id)\
             .filter(Content.is_embedded == True)\
             .first()

            episode_count = stats.episode_count or 0
            date_range = None
            if stats.earliest and stats.latest:
                date_range = DateRange(
                    earliest=_format_date(stats.earliest),
                    latest=_format_date(stats.latest)
                )

            # Get regular speakers (appear in 2+ episodes)
            regular_speakers_query = session.query(
                SpeakerIdentity.id,
                SpeakerIdentity.primary_name,
                func.count(distinct(Speaker.content_id)).label('episode_count')
            ).select_from(Speaker)\
             .join(Content, Speaker.content_id == Content.content_id)\
             .join(SpeakerIdentity, Speaker.speaker_identity_id == SpeakerIdentity.id)\
             .filter(Content.channel_id == channel_id)\
             .filter(SpeakerIdentity.primary_name.isnot(None))\
             .group_by(SpeakerIdentity.id, SpeakerIdentity.primary_name)\
             .having(func.count(distinct(Speaker.content_id)) >= 2)\
             .order_by(desc('episode_count'))\
             .limit(8)\
             .all()

            regular_speakers = [
                SpeakerPreview(
                    speaker_id=str(s.id),
                    name=s.primary_name,
                    appearance_count=s.episode_count,
                    image_url=None
                )
                for s in regular_speakers_query
            ]

            # Get recent episodes
            recent_query = session.query(Content)\
                .filter(Content.channel_id == channel_id)\
                .filter(Content.is_embedded == True)\
                .order_by(desc(Content.publish_date))\
                .limit(5)\
                .all()

            recent_episodes = [
                EpisodePreview(
                    content_id=ep.content_id,
                    content_id_numeric=ep.id,
                    title=ep.title or "Untitled",
                    channel_name=ep.channel_name,
                    channel_id=ep.channel_id,
                    publish_date=_format_date(ep.publish_date),
                    duration=int(ep.duration) if ep.duration else None,
                    platform=ep.platform,
                )
                for ep in recent_query
            ]

            # Parse tags
            tags = []
            if channel.tags:
                if isinstance(channel.tags, list):
                    tags = [str(t) for t in channel.tags]
                elif isinstance(channel.tags, dict):
                    tags = list(channel.tags.keys())

            processing_time = (time.time() - start_time) * 1000

            return ChannelDetailsResponse(
                success=True,
                channel_id=channel.id,
                channel_key=channel.channel_key,
                name=channel.display_name,
                platform=channel.platform,
                description=channel.description,
                primary_url=channel.primary_url,
                language=channel.language,
                status=channel.status,
                episode_count=episode_count,
                date_range=date_range,
                publishing_frequency=None,  # TODO: Calculate from episode dates
                regular_speakers=regular_speakers,
                recent_episodes=recent_episodes,
                tags=tags,
                processing_time_ms=processing_time,
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching channel details for {channel_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/channels/by-name/{channel_name}", response_model=ChannelDetailsResponse)
async def get_channel_details_by_name(
    channel_name: str = Path(..., description="Channel name")
):
    """
    Get channel details by channel name.

    Useful when only the channel name is known (e.g., from search results).
    """
    start_time = time.time()

    try:
        with get_session() as session:
            # Find channel by name
            channel = session.query(Channel)\
                .filter(Channel.display_name == channel_name)\
                .first()

            if not channel:
                # Try case-insensitive match
                channel = session.query(Channel)\
                    .filter(func.lower(Channel.display_name) == func.lower(channel_name))\
                    .first()

            if not channel:
                raise HTTPException(status_code=404, detail=f"Channel not found: {channel_name}")

            # Reuse the ID-based endpoint logic
            return await get_channel_details(channel.id)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching channel details by name {channel_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/speakers/{speaker_id}", response_model=SpeakerDetailsResponse)
async def get_speaker_details(
    speaker_id: str = Path(..., description="Speaker hash or speaker identity ID")
):
    """
    Get detailed information about a speaker.

    Accepts either a speaker_hash or a speaker_identity_id.
    Returns speaker profile, stats, recent episodes, and top channels.
    """
    start_time = time.time()

    try:
        with get_session() as session:
            # Try to find by speaker identity ID first (if numeric)
            identity = None
            if speaker_id.isdigit():
                identity = session.query(SpeakerIdentity)\
                    .filter(SpeakerIdentity.id == int(speaker_id))\
                    .first()

            # If not found by ID, try to find by speaker hash
            speaker = None
            if not identity:
                speaker = session.query(Speaker)\
                    .options(joinedload(Speaker.speaker_identity))\
                    .filter(Speaker.speaker_hash == speaker_id)\
                    .first()

                if speaker and speaker.speaker_identity:
                    identity = speaker.speaker_identity

            # If we have an identity, get aggregated stats
            if identity:
                # Get stats across all speakers with this identity
                stats = session.query(
                    func.sum(Speaker.duration).label('total_duration'),
                    func.sum(Speaker.segment_count).label('total_segments'),
                    func.count(distinct(Speaker.content_id)).label('total_episodes')
                ).filter(Speaker.speaker_identity_id == identity.id).first()

                # Get recent episodes
                recent_query = session.query(Content)\
                    .join(Speaker, Speaker.content_id == Content.content_id)\
                    .filter(Speaker.speaker_identity_id == identity.id)\
                    .filter(Content.is_embedded == True)\
                    .order_by(desc(Content.publish_date))\
                    .limit(5)\
                    .all()

                recent_episodes = [
                    EpisodePreview(
                        content_id=ep.content_id,
                        content_id_numeric=ep.id,
                        title=ep.title or "Untitled",
                        channel_name=ep.channel_name,
                        channel_id=ep.channel_id,
                        publish_date=_format_date(ep.publish_date),
                        duration=int(ep.duration) if ep.duration else None,
                        platform=ep.platform,
                    )
                    for ep in recent_query
                ]

                # Get top channels
                top_channels_query = session.query(
                    Channel.id,
                    Channel.channel_key,
                    Channel.display_name,
                    Channel.platform,
                    func.count(distinct(Speaker.content_id)).label('appearance_count')
                ).select_from(Speaker)\
                 .join(Content, Speaker.content_id == Content.content_id)\
                 .join(Channel, Content.channel_id == Channel.id)\
                 .filter(Speaker.speaker_identity_id == identity.id)\
                 .group_by(Channel.id, Channel.channel_key, Channel.display_name, Channel.platform)\
                 .order_by(desc('appearance_count'))\
                 .limit(5)\
                 .all()

                top_channels = [
                    ChannelWithCount(
                        channel_id=c.id,
                        channel_key=c.channel_key,
                        name=c.display_name,
                        platform=c.platform,
                        count=c.appearance_count
                    )
                    for c in top_channels_query
                ]

                # Get related speakers (co-appear in same episodes)
                related_query = session.query(
                    SpeakerIdentity.id,
                    SpeakerIdentity.primary_name,
                    func.count(distinct(Speaker.content_id)).label('co_appearances')
                ).select_from(Speaker)\
                 .join(SpeakerIdentity, Speaker.speaker_identity_id == SpeakerIdentity.id)\
                 .filter(
                    Speaker.content_id.in_(
                        session.query(Speaker.content_id)
                        .filter(Speaker.speaker_identity_id == identity.id)
                    )
                 )\
                 .filter(SpeakerIdentity.id != identity.id)\
                 .filter(SpeakerIdentity.primary_name.isnot(None))\
                 .group_by(SpeakerIdentity.id, SpeakerIdentity.primary_name)\
                 .order_by(desc('co_appearances'))\
                 .limit(8)\
                 .all()

                related_speakers = [
                    SpeakerPreview(
                        speaker_id=str(s.id),
                        name=s.primary_name,
                        appearance_count=s.co_appearances,
                        image_url=None
                    )
                    for s in related_query
                ]

                processing_time = (time.time() - start_time) * 1000

                return SpeakerDetailsResponse(
                    success=True,
                    speaker_id=str(identity.id),
                    name=identity.primary_name or "Unknown Speaker",
                    bio=identity.bio,
                    occupation=identity.occupation,
                    organization=identity.organization,
                    role=identity.role,
                    image_url=None,  # TODO: Add when available
                    total_appearances=stats.total_segments or 0,
                    total_duration_seconds=float(stats.total_duration or 0),
                    total_episodes=stats.total_episodes or 0,
                    recent_episodes=recent_episodes,
                    top_channels=top_channels,
                    related_speakers=related_speakers,
                    processing_time_ms=processing_time,
                )

            # If we only have a speaker (no identity), return basic info
            elif speaker:
                # Get content for this speaker
                content = session.query(Content)\
                    .filter(Content.content_id == speaker.content_id)\
                    .first()

                recent_episodes = []
                if content:
                    recent_episodes = [EpisodePreview(
                        content_id=content.content_id,
                        content_id_numeric=content.id,
                        title=content.title or "Untitled",
                        channel_name=content.channel_name,
                        channel_id=content.channel_id,
                        publish_date=_format_date(content.publish_date),
                        duration=int(content.duration) if content.duration else None,
                        platform=content.platform,
                    )]

                processing_time = (time.time() - start_time) * 1000

                return SpeakerDetailsResponse(
                    success=True,
                    speaker_id=speaker.speaker_hash,
                    name=speaker.display_name or f"Speaker {speaker.local_speaker_id}",
                    bio=None,
                    occupation=None,
                    organization=None,
                    role=None,
                    image_url=None,
                    total_appearances=speaker.segment_count or 0,
                    total_duration_seconds=float(speaker.duration or 0),
                    total_episodes=1,
                    recent_episodes=recent_episodes,
                    top_channels=[],
                    related_speakers=[],
                    processing_time_ms=processing_time,
                )

            else:
                raise HTTPException(status_code=404, detail=f"Speaker not found: {speaker_id}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching speaker details for {speaker_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/speakers/by-name/{speaker_name}", response_model=SpeakerDetailsResponse)
async def get_speaker_details_by_name(
    speaker_name: str = Path(..., description="Speaker name to search for")
):
    """
    Get speaker details by name.

    Searches speaker identities by primary name (case-insensitive).
    """
    start_time = time.time()

    try:
        with get_session() as session:
            # Find speaker identity by name
            identity = session.query(SpeakerIdentity)\
                .filter(func.lower(SpeakerIdentity.primary_name) == func.lower(speaker_name))\
                .first()

            if not identity:
                # Try partial match
                identity = session.query(SpeakerIdentity)\
                    .filter(SpeakerIdentity.primary_name.ilike(f"%{speaker_name}%"))\
                    .first()

            if not identity:
                raise HTTPException(status_code=404, detail=f"Speaker not found: {speaker_name}")

            # Reuse the ID-based endpoint
            return await get_speaker_details(str(identity.id))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching speaker details by name {speaker_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
