"""
Embedding models for the audio-visual processing pipeline.

Contains:
- EmbeddingSegment: Retrieval-optimized segments with embeddings
"""

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, ForeignKey, Text, Index, ARRAY,
    UniqueConstraint, JSON
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from datetime import datetime

from .base import Base


class EmbeddingSegment(Base):
    """
    Represents an embedding-optimized segment derived from one or more SpeakerTranscriptions.
    Each segment is designed for optimal retrieval performance.
    """
    __tablename__ = 'embedding_segments'

    id = Column(Integer, primary_key=True)
    content_id = Column(Integer, ForeignKey('content.id'), nullable=False, index=True)
    segment_index = Column(Integer, nullable=False)  # Order within content

    # Text and temporal boundaries
    text = Column(Text, nullable=False)
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)

    # Token count for this segment
    token_count = Column(Integer, nullable=False)

    # Segment type describes how it was created
    segment_type = Column(String(20), nullable=False)  # 'single', 'combined', 'split'

    # References to source SpeakerTranscription records
    # For 'single': [id] - one source
    # For 'combined': [id1, id2, ...] - multiple sources
    # For 'split': [id] - one source (but only part of it)
    source_transcription_ids = Column(ARRAY(Integer), nullable=False)

    # For split segments, store character offsets into source text
    source_start_char = Column(Integer, nullable=True)  # Start character in source text
    source_end_char = Column(Integer, nullable=True)    # End character in source text

    # The embedding vector
    embedding = Column(Vector(1024), nullable=True)

    # Alternative embedding for model migration (2000 dimensions)
    embedding_alt = Column(Vector(2000), nullable=True)
    embedding_alt_model = Column(String(50), nullable=True)  # Model used for alternative embedding

    # Additional metadata and convenience fields
    meta_data = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    stitch_version = Column(String(50), nullable=True)  # Version of stitching algorithm used
    embedding_version = Column(String(50), nullable=True)  # Segmentation algorithm version (e.g., 'stitch_v14_xlmr'). See src/processing_steps/stitch_steps/stage13_segment.py
    segment_hash = Column(String(8), nullable=False)  # Segment identifier (unique per content)
    content_id_string = Column(String, nullable=True, index=True)  # String content ID for easy API lookup
    source_speaker_hashes = Column(ARRAY(String(8)), nullable=True)  # Speaker hashes for quick filtering

    # Speaker position mapping for reconstructing speaker-attributed text
    # Format: {"2668417": [[0, 280]], "2668422": [[281, 315], [650, 750]]}
    # Keys are speakers.id (speaker_db_id) for direct joining to speakers table
    # Each speaker_id maps to list of [start_char, end_char] ranges in segment text
    speaker_positions = Column(JSONB, nullable=True)

    # Sentence-level source tracking (replaces source_transcription_ids for new content)
    # Array of sentence.sentence_index values that comprise this segment
    source_sentence_ids = Column(ARRAY(Integer), nullable=True)

    # Emotion summary for fast emotion-based filtering without joining sentences table
    # Format: {"angry": 3, "happy": 1} - count of each emotion (excludes neutral/unknown, conf > 0.20)
    # NULL if no interesting emotions detected
    #
    # Query examples using GIN index (idx_embedding_segments_emotion_summary):
    #   WHERE emotion_summary ? 'angry'                    -- segments with any anger
    #   WHERE emotion_summary ?| array['angry', 'sad']     -- segments with anger OR sadness
    #   WHERE emotion_summary ?& array['angry', 'happy']   -- segments with anger AND happiness
    #   WHERE (emotion_summary->>'angry')::int > 3         -- segments with more than 3 angry sentences
    emotion_summary = Column(JSONB, nullable=True)

    # Relationships
    content = relationship("Content", back_populates="embedding_segments")

    # We don't use speaker_id directly since segments might combine multiple speakers
    # Instead, we track speaker(s) in metadata

    __table_args__ = (
        Index('idx_embedding_segments_content_index', 'content_id', 'segment_index'),
        Index('idx_embedding_segments_source_ids', 'source_transcription_ids', postgresql_using='gin'),
        Index('idx_embedding_segments_text_search', 'text', postgresql_using='gin', postgresql_ops={'text': 'gin_tsvector_ops'}),
        Index('idx_embedding_segments_text_trigram', 'text', postgresql_using='gin', postgresql_ops={'text': 'gin_trgm_ops'}),
        UniqueConstraint('content_id', 'segment_hash', name='uq_embedding_segments_content_hash'),  # Unique hash per content
        Index('idx_embedding_segments_sentence_ids', 'source_sentence_ids', postgresql_using='gin'),
        Index('idx_embedding_segments_emotion_summary', 'emotion_summary', postgresql_using='gin'),
    )
