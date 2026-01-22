"""
YouTube Video Cache model for episode matching.

This table stores video metadata from YouTube channels associated with podcasts,
enabling efficient podcast-to-YouTube episode matching without live API calls.

Key design decisions:
- Completely isolated from download pipeline (no Channel/Content records created)
- Uses youtube_channel_id (UC...) as the channel identifier, not our internal channel_id
- Supports fuzzy title matching via trigram index
"""

from sqlalchemy import (
    Column, Integer, String, DateTime, Text, BigInteger, UniqueConstraint, Index
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from datetime import datetime

from .base import Base


class YouTubeVideoCache(Base):
    """
    Cache table for YouTube video metadata used in episode matching.

    This table stores video metadata from YouTube channels that have been matched
    to podcast channels. It enables fast, API-free episode matching by caching
    all videos from a channel locally.

    Attributes:
        id: Primary key
        youtube_channel_id: YouTube channel ID (UC...) - NOT our internal channel_id
        video_id: YouTube video ID (11 character string)
        title: Video title
        description: Video description (first 5000 chars typically)
        publish_date: When the video was published on YouTube
        duration: Duration in seconds
        view_count: Number of views at time of indexing
        thumbnail_url: URL to video thumbnail
        meta_data: Additional metadata (tags, categories, etc.)
        indexed_at: When this video was cached

    Indexes:
        - youtube_channel_id: Fast lookup by channel
        - publish_date: For date-range queries during matching
        - video_id: Quick lookup of specific video
        - title (gin_trgm): Fuzzy title matching for episode comparison
    """
    __tablename__ = 'youtube_video_cache'

    id = Column(Integer, primary_key=True)
    youtube_channel_id = Column(String(50), nullable=False, index=True)
    video_id = Column(String(20), nullable=False, index=True)
    title = Column(Text)
    description = Column(Text)
    publish_date = Column(DateTime(timezone=True))
    duration = Column(Integer)  # Duration in seconds
    view_count = Column(BigInteger)
    thumbnail_url = Column(Text)
    meta_data = Column(JSONB, default=dict)
    indexed_at = Column(DateTime(timezone=True), default=func.now())

    __table_args__ = (
        UniqueConstraint('youtube_channel_id', 'video_id', name='uq_youtube_video_cache_channel_video'),
        Index('idx_yvc_channel_id', 'youtube_channel_id'),
        Index('idx_yvc_publish_date', 'publish_date'),
        Index('idx_yvc_video_id', 'video_id'),
        # Note: trigram index is created in migration, not here
    )

    def __repr__(self):
        return f"<YouTubeVideoCache(channel={self.youtube_channel_id}, video={self.video_id}, title='{self.title[:50] if self.title else ''}...')>"
