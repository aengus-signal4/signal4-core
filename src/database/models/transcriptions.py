"""
Alternative transcription models for external provider support.

Contains:
- AlternativeTranscription: Transcriptions from external services (AssemblyAI, Deepgram, etc.)
"""

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, ForeignKey, Text, Index,
    UniqueConstraint
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from datetime import datetime

from .base import Base


class AlternativeTranscription(Base):
    """
    Stores alternative transcriptions from external services (AssemblyAI, Deepgram, etc.).

    This table tracks re-transcriptions of embedding segments using different providers
    and models, enabling quality comparison and multi-provider fallback strategies.
    Each segment can have multiple alternative transcriptions from different providers/models.

    Attributes:
        id: Primary key
        segment_id: Reference to embedding_segments.id
        content_id: Denormalized content ID for faster queries

        provider: Service provider ('assemblyai', 'deepgram', 'openai_whisper', etc.)
        model: Model identifier ('best', 'nano', 'base', 'large-v3', etc.)
        language: Language code used for transcription ('en', 'fr', etc.)

        transcription_text: The actual transcription text
        confidence: Overall confidence score (0.0-1.0) if provided by service

        word_timings: Optional word-level timestamps as JSON array
                     Format: [{"word": "hello", "start": 0.0, "end": 0.5, "confidence": 0.95}, ...]

        speaker_labels: Optional speaker diarization from provider
                       Format: [{"speaker": "A", "start": 0.0, "end": 5.2}, ...]

        audio_duration: Duration of audio segment transcribed (seconds)
        processing_time: Time taken by API to process (seconds)

        meta_data: Additional provider-specific metadata as JSON
                  Examples: API version, detected language, audio quality metrics, etc.

        created_at: When this transcription was generated
        api_cost: Cost of API call if known (in USD)
    """
    __tablename__ = 'alternative_transcriptions'

    id = Column(Integer, primary_key=True)
    segment_id = Column(Integer, ForeignKey('embedding_segments.id'), nullable=True, index=True)
    content_id = Column(Integer, ForeignKey('content.id'), nullable=False, index=True)

    # Provider information
    provider = Column(String(50), nullable=False, index=True)
    model = Column(String(100), nullable=True)
    language = Column(String(10), nullable=True)

    # Transcription results
    transcription_text = Column(Text, nullable=False)
    confidence = Column(Float, nullable=True)

    # Translations (if requested)
    translation_en = Column(Text, nullable=True)  # English translation
    translation_fr = Column(Text, nullable=True)  # French translation

    # Optional detailed results
    word_timings = Column(JSONB, nullable=True)
    speaker_labels = Column(JSONB, nullable=True)

    # Processing metadata
    audio_duration = Column(Float, nullable=True)
    processing_time = Column(Float, nullable=True)
    meta_data = Column(JSONB, default=dict)

    # Audit and cost tracking
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    api_cost = Column(Float, nullable=True)

    # Relationships
    segment = relationship("EmbeddingSegment", foreign_keys=[segment_id])
    content = relationship("Content", foreign_keys=[content_id])

    __table_args__ = (
        # Unique constraint: one transcription per segment+provider+model combination
        UniqueConstraint('segment_id', 'provider', 'model', name='uq_alt_trans_segment_provider_model'),

        # Indexes for common queries
        Index('idx_alt_trans_segment_provider', 'segment_id', 'provider'),
        Index('idx_alt_trans_content_provider', 'content_id', 'provider'),
        Index('idx_alt_trans_created_at', 'created_at'),
        Index('idx_alt_trans_text_search', 'transcription_text', postgresql_using='gin',
              postgresql_ops={'transcription_text': 'gin_tsvector_ops'}),
    )
