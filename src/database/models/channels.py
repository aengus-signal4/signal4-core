"""
Channel models for content sources across platforms.

Contains:
- Channel: Universal channel table (YouTube, podcasts, Rumble, etc.)
- ChannelSource: Multiple source URLs for channels
- ChannelProject: Channel-project assignments
- PodcastMetadata: Enriched podcast metadata (deprecated, being replaced by Channel)
- PodcastChart: Podcast chart rankings
"""

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, ForeignKey, Text, Boolean,
    Index, UniqueConstraint, PrimaryKeyConstraint
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from datetime import datetime
from typing import Optional

from .base import Base


class Channel(Base):
    """
    Universal channel table supporting all platforms (YouTube, podcasts, Rumble, etc.).

    This table replaces platform-specific tables (like podcast_metadata) with a unified
    approach. Channels can have multiple sources (e.g., Jordan Peterson has YouTube + podcast),
    and be assigned to multiple projects.

    Attributes:
        id: Primary key
        channel_key: Unique canonical identifier (normalized name or ID)
        display_name: Human-readable channel/podcast name
        platform: Primary platform ('youtube', 'podcast', 'rumble', 'odysee')
        primary_url: Main URL/feed for this channel
        language: Primary language code (ISO 639-1)
        description: Channel/podcast description
        status: Channel status ('discovered', 'active', 'archived')
        tags: JSON array for classification/categorization
        platform_metadata: JSONB for platform-specific fields
                          - For podcasts: podcast_index_id, monthly_rankings, episode_count, creator, categories
                          - For YouTube: subscriber_count, video_count, etc.
        created_at: Record creation timestamp
        updated_at: Record update timestamp

    Relationships:
        sources: Multiple URLs across platforms (via channel_sources)
        projects: Project assignments (via channel_projects)
        content: All content from this channel
        chart_entries: Historical chart rankings (for podcasts)
    """
    __tablename__ = 'channels'

    id = Column(Integer, primary_key=True)
    channel_key = Column(String(500), nullable=False, unique=True, index=True)
    display_name = Column(String(500), nullable=False)
    platform = Column(String(50), nullable=False, index=True)
    primary_url = Column(String(1000), nullable=False, index=True)
    language = Column(String(10))
    description = Column(Text)
    status = Column(String(20), nullable=False, default='discovered', index=True)
    tags = Column(JSONB, default=list, nullable=False)
    platform_metadata = Column(JSONB, default=dict, nullable=False)
    importance_score = Column(Float, nullable=True, default=None, index=True)  # Weighted ranking score for RAG sorting
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    sources = relationship("ChannelSource", back_populates="channel", cascade="all, delete-orphan")
    projects = relationship("ChannelProject", back_populates="channel", cascade="all, delete-orphan")
    content = relationship("Content", back_populates="channel")
    chart_entries = relationship("PodcastChart", back_populates="channel")

    __table_args__ = (
        Index('idx_channels_platform', 'platform'),
        Index('idx_channels_status', 'status'),
        Index('idx_channels_platform_status', 'platform', 'status'),
    )

    @classmethod
    def get_channel_key(cls, name: str) -> str:
        """Generate normalized channel key from name"""
        return name.lower().strip()


class ChannelSource(Base):
    """
    Multiple source URLs for channels across platforms.

    Enables tracking multi-platform creators (e.g., Jordan Peterson has
    2 YouTube channels + 1 podcast feed = 3 sources, 1 channel).

    Attributes:
        id: Primary key
        channel_id: Foreign key to channels table
        platform: Platform for this source ('youtube', 'podcast', etc.)
        source_url: The actual URL/feed
        is_primary: Whether this is the primary source
        created_at: When this source was added
    """
    __tablename__ = 'channel_sources'

    id = Column(Integer, primary_key=True)
    channel_id = Column(Integer, ForeignKey('channels.id', ondelete='CASCADE'), nullable=False, index=True)
    platform = Column(String(50), nullable=False)
    source_url = Column(String(1000), nullable=False, index=True)
    is_primary = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    channel = relationship("Channel", back_populates="sources")

    __table_args__ = (
        UniqueConstraint('platform', 'source_url', name='uq_channel_sources_platform_url'),
    )


class ChannelProject(Base):
    """
    Many-to-many relationship between channels and projects.

    Tracks which channels are assigned to which projects for indexing.
    Replaces the sources.csv approach with database-backed management.

    Attributes:
        channel_id: Foreign key to channels table
        project_name: Project identifier (e.g., 'CPRMV', 'Finance')
        added_at: When channel was added to project
        added_by: Who added it (username or 'system')
        notes: Optional notes about why this channel is in the project
    """
    __tablename__ = 'channel_projects'

    channel_id = Column(Integer, ForeignKey('channels.id', ondelete='CASCADE'), nullable=False)
    project_name = Column(String(100), nullable=False, index=True)
    added_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    added_by = Column(String(100))
    notes = Column(Text)

    # Relationships
    channel = relationship("Channel", back_populates="projects")

    __table_args__ = (
        PrimaryKeyConstraint('channel_id', 'project_name'),
    )


class PodcastMetadata(Base):
    """
    DEPRECATED: Being replaced by Channel model.

    Stores enriched podcast metadata from PodcastIndex API.

    This table maintains a master list of all podcasts discovered across
    all chart collections, with their enriched metadata and historical rankings.

    Attributes:
        id: Primary key
        podcast_name: Name of the podcast
        podcast_key: Normalized lowercase key for deduplication
        creator: Podcast creator/author
        description: Podcast description
        rss_url: RSS feed URL
        language: Primary language code
        categories: JSON array of category strings
        episode_count: Number of episodes
        podcast_index_id: ID from PodcastIndex API

        monthly_rankings: JSON object with historical rankings
                         Format: {"2025-10": {"spotify_us_news": 15, "apple_gb_politics": 3}}

        first_seen: When this podcast was first discovered
        last_updated: Last time metadata was updated
        last_enriched: Last time enrichment was performed (for staleness checking)

        created_at: Record creation timestamp
        updated_at: Record update timestamp
    """
    __tablename__ = 'podcast_metadata'

    id = Column(Integer, primary_key=True)

    # Core identification
    podcast_name = Column(String(500), nullable=False)
    podcast_key = Column(String(500), nullable=False, unique=True, index=True)  # Normalized key

    # Metadata from PodcastIndex API
    creator = Column(String(500))
    description = Column(Text)
    rss_url = Column(String(1000))
    language = Column(String(10))
    categories = Column(JSONB, default=list)  # JSON array of categories
    episode_count = Column(Integer, default=0)
    podcast_index_id = Column(String(50))

    # Historical rankings (embedded JSON)
    monthly_rankings = Column(JSONB, default=dict, nullable=False)

    # Additional metadata (e.g., classification results)
    meta_data = Column(JSONB, default=dict)

    # Timestamps
    first_seen = Column(DateTime, nullable=False, default=datetime.utcnow)
    last_updated = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_enriched = Column(DateTime)  # Last API enrichment

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships - REMOVED: chart_entries now points to Channel table
    # chart_entries = relationship("PodcastChart", back_populates="podcast")

    __table_args__ = (
        Index('idx_podcast_metadata_name', 'podcast_name'),
        Index('idx_podcast_metadata_key', 'podcast_key'),
        Index('idx_podcast_metadata_creator', 'creator'),
        Index('idx_podcast_metadata_last_enriched', 'last_enriched'),
        Index('idx_podcast_metadata_rankings', 'monthly_rankings', postgresql_using='gin'),
    )

    @classmethod
    def get_cache_key(cls, podcast_name: str) -> str:
        """Generate normalized cache key from podcast name"""
        return podcast_name.lower().strip()

    @classmethod
    def is_stale(cls, last_enriched: Optional[datetime], days: int = 90) -> bool:
        """Check if metadata needs refreshing (default: 90 days)"""
        if not last_enriched:
            return True
        age = datetime.utcnow() - last_enriched
        return age.days > days


class PodcastChart(Base):
    """
    Stores raw podcast chart data from Podstatus.

    Each record represents a podcast's appearance on a specific chart
    at a specific rank for a specific month.

    Attributes:
        id: Primary key
        channel_id: Foreign key to channels table (renamed from podcast_id)

        month: Month identifier (YYYY-MM format)
        platform: Platform (spotify, apple)
        country: Country code (us, gb, ca, etc.)
        category: Category slug (news, politics, all-podcasts, etc.)
        rank: Rank on this chart (1-200)

        chart_key: Unique identifier for this chart (platform_country_category)

        collected_at: When this chart data was collected
        created_at: Record creation timestamp
    """
    __tablename__ = 'podcast_charts'

    id = Column(Integer, primary_key=True)
    channel_id = Column(Integer, ForeignKey('channels.id'), nullable=False, index=True)

    # Chart identification
    month = Column(String(7), nullable=False, index=True)  # YYYY-MM
    platform = Column(String(20), nullable=False, index=True)  # spotify, apple
    country = Column(String(10), nullable=False, index=True)  # us, gb, ca, etc.
    category = Column(String(100), nullable=False, index=True)  # news, politics, etc.
    rank = Column(Integer, nullable=False)

    # Computed chart identifier
    chart_key = Column(String(150), nullable=False, index=True)  # platform_country_category

    # Timestamps
    collected_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    channel = relationship("Channel", back_populates="chart_entries")

    __table_args__ = (
        # Unique constraint: one channel can only have one rank per chart per month
        UniqueConstraint('channel_id', 'month', 'chart_key', name='uq_podcast_chart_month'),

        # Composite indexes for common queries
        Index('idx_podcast_charts_month_platform', 'month', 'platform'),
        Index('idx_podcast_charts_month_country', 'month', 'country'),
        Index('idx_podcast_charts_chart_key_month', 'chart_key', 'month'),
        Index('idx_podcast_charts_platform_country_category', 'platform', 'country', 'category'),
    )

    @classmethod
    def get_chart_key(cls, platform: str, country: str, category: str) -> str:
        """Generate chart key from components"""
        return f"{platform.lower()}_{country.lower()}_{category.lower()}"
