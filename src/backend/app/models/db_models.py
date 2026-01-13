"""
Database Models
===============

Minimal SQLAlchemy models for backend API access to content_processing database.
Only includes models needed for search and embedding operations.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, Index, ARRAY, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from pgvector.sqlalchemy import Vector

Base = declarative_base()


class Content(Base):
    """
    Media content (videos, podcasts).
    Minimal fields needed for search filtering.
    """
    __tablename__ = 'content'

    id = Column(Integer, primary_key=True)
    projects = Column(ARRAY(String))  # Array of project names
    platform = Column(String(50))
    content_id = Column(String(255), unique=True, index=True)
    channel_name = Column(String(500))
    channel_url = Column(Text)
    title = Column(Text)
    description = Column(Text)
    publish_date = Column(DateTime, index=True)
    duration = Column(Float)
    stitch_version = Column(String, nullable=False, server_default='stitch_v1', index=True)
    main_language = Column(String(10), nullable=True, index=True)  # Primary language (ISO 639-1: en, fr, de, etc.)

    # Relationships
    embedding_segments = relationship("EmbeddingSegment", back_populates="content")


class EmbeddingSegment(Base):
    """
    Embedding-optimized segments for retrieval.
    Core model for semantic search.
    """
    __tablename__ = 'embedding_segments'

    id = Column(Integer, primary_key=True)
    content_id = Column(Integer, ForeignKey('content.id'), nullable=False, index=True)
    segment_index = Column(Integer, nullable=False)

    # Text and temporal boundaries
    text = Column(Text, nullable=False)
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)

    # Token count
    token_count = Column(Integer, nullable=False)

    # Source references
    source_transcription_ids = Column(ARRAY(Integer), nullable=False)
    source_speaker_hashes = Column(ARRAY(String), nullable=True)

    # Embeddings (two dimensions supported)
    embedding = Column(Vector(1024), nullable=True)  # Qwen3-0.6B (1024-dim)
    embedding_alt = Column(Vector(2000), nullable=True)  # Qwen3-4B (2000-dim)

    # Content ID string for easy reference
    content_id_string = Column(String(255), nullable=True)

    # Timestamps
    created_at = Column(DateTime)

    # Relationships
    content = relationship("Content", back_populates="embedding_segments")

    # Indexes for performance
    __table_args__ = (
        Index('idx_embedding_segments_content_id', 'content_id'),
        Index('idx_embedding_segments_publish_date', 'content_id'),
    )


class SpeakerTranscription(Base):
    """
    Speaker-attributed transcript segments (speaker turns).
    Used for getting speaker context and full transcripts.
    """
    __tablename__ = 'speaker_transcriptions'

    id = Column(Integer, primary_key=True)
    content_id = Column(Integer, ForeignKey('content.id'), nullable=False, index=True)
    speaker_id = Column(Integer, nullable=False)
    speaker_hash = Column(String(8), nullable=True, index=True)
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    text = Column(Text, nullable=False)
    turn_index = Column(Integer, nullable=False)
    created_at = Column(DateTime)
    updated_at = Column(DateTime)


class HierarchicalSummary(Base):
    """
    Hierarchical summaries with citations.
    Stores both theme summaries and meta-summaries.

    Queryable columns enable meta-analysis across summaries:
    - groupings: Which groups were analyzed
    - time_window_days: Time range covered
    - num_themes: Target number of themes
    - theme_discovery_method: clustering vs predefined
    - clustering_params: HDBSCAN parameters used
    - synthesis_type: cross_theme, cross_group, temporal
    """
    __tablename__ = 'hierarchical_summaries'

    id = Column(Integer, primary_key=True)
    summary_type = Column(String(50), nullable=False)  # 'theme' or 'meta'
    summary_id = Column(String(255), unique=True, index=True)  # User-friendly ID
    config_hash = Column(String(64), index=True)  # Hash of generation params
    summary_data = Column(JSONB, nullable=False)  # Full summary + citations + metadata

    # Query parameters for filtering and meta-analysis
    time_window_days = Column(Integer, index=True)  # For backward compatibility
    start_date = Column(DateTime(timezone=True), index=True)  # Explicit start date
    end_date = Column(DateTime(timezone=True), index=True)  # Explicit end date
    groupings = Column(JSONB)  # List of group configs
    num_themes = Column(Integer)  # Target number of themes (null = auto)
    theme_discovery_method = Column(String(50))  # 'clustering' or 'predefined'
    theme_selection_strategy = Column(String(50))  # 'top_per_group' or 'aligned'
    clustering_params = Column(JSONB)  # HDBSCAN parameters
    synthesis_type = Column(String(50))  # 'cross_theme', 'cross_group', 'temporal'

    created_at = Column(DateTime, server_default=func.now(), index=True)

    __table_args__ = (
        Index('idx_hierarchical_summary_config', 'config_hash', 'summary_type'),
        Index('idx_hierarchical_summary_id', 'summary_id'),
        Index('idx_hierarchical_summary_created', 'created_at'),
        Index('idx_hierarchical_summary_time_window', 'time_window_days'),
    )


class Speaker(Base):
    """
    Speaker identity records.
    Minimal fields for speaker filtering.
    """
    __tablename__ = 'speakers'

    id = Column(Integer, primary_key=True)
    content_id = Column(String, nullable=False)
    local_speaker_id = Column(String, nullable=False)
    speaker_hash = Column(String, nullable=False, unique=True, index=True)
    display_name = Column(String)
    embedding = Column(Vector(512), nullable=True)
    cluster_id = Column(String(64), nullable=True)
