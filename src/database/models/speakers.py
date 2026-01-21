"""
Speaker models for the audio-visual processing pipeline.

Contains:
- Speaker: Unified speaker table with embeddings
- SpeakerIdentity: Permanent speaker profiles
- SpeakerAssignment: Embedding-to-identity mappings
- SpeakerTranscription: Speaker-attributed transcript segments
- Sentence: Sentence-level granularity for transcripts
"""

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, ForeignKey, Text, Boolean,
    Index, ARRAY, text, UniqueConstraint, func, Enum
)
from sqlalchemy import text as sa_text  # Alias to avoid shadowing by column names
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import numpy as np
import hashlib
import logging

from .base import Base, SpeakerProcessingStatus

logger = logging.getLogger(__name__)


class SpeakerIdentity(Base):
    """Permanent speaker profiles that persist across re-clustering.

    This table stores verified speaker identities that can be linked to
    multiple speaker embeddings across different content. It serves as the
    authoritative source for speaker identification.
    """
    __tablename__ = 'speaker_identities'

    id = Column(Integer, primary_key=True)

    # Primary identification
    primary_name = Column(String(255))
    aliases = Column(ARRAY(Text), default=list)

    # Identity confidence and verification
    confidence_score = Column(Float, default=0.5)
    verification_status = Column(String(50), default='unverified')
    verification_metadata = Column(JSONB, default=dict)

    # Rich metadata
    bio = Column(Text)
    occupation = Column(String(255))
    organization = Column(String(255))
    location = Column(String(255))
    country = Column(String(100))  # Increased size for full country names
    gender = Column(String(50))  # Male, Female, Non-binary, etc.
    role = Column(String(100))  # Host, Guest, Regular, Co-host, etc.

    # External profiles and IDs
    social_profiles = Column(JSONB, default=dict)
    external_ids = Column(JSONB, default=dict)
    website = Column(String(500))

    # Categorization
    tags = Column(ARRAY(Text), default=list)
    speaker_type = Column(String(50))

    # Activity statistics (denormalized for performance)
    first_appearance = Column(DateTime)
    last_appearance = Column(DateTime)
    total_episodes = Column(Integer, default=0)
    total_duration = Column(Float, default=0.0)
    primary_channels = Column(ARRAY(Text), default=list)

    # Management
    notes = Column(Text)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String(100))
    updated_by = Column(String(100))

    # Text-evidence-first pipeline tracking
    centroid_source = Column(String(50), default='legacy')  # 'text_verified', 'legacy'
    text_verified_count = Column(Integer, default=0)  # Number of speakers with text evidence

    # Relationships
    speaker_embeddings = relationship("Speaker", back_populates="speaker_identity")
    assignments = relationship("SpeakerAssignment", back_populates="speaker_identity")

    __table_args__ = (
        Index('idx_speaker_identities_active', 'id', postgresql_where=text('is_active = true')),
    )


class SpeakerAssignment(Base):
    """Tracks how speaker embeddings are assigned to identities.

    This table maintains the history of how individual speaker embeddings
    (from specific content) are mapped to speaker identities. It supports
    temporal validity for tracking assignment changes over time.
    """
    __tablename__ = 'speaker_assignments'

    id = Column(Integer, primary_key=True)

    # The assignment
    speaker_embedding_id = Column(Integer, ForeignKey('speakers.id'), nullable=False)
    speaker_identity_id = Column(Integer, ForeignKey('speaker_identities.id'), nullable=False)

    # Assignment details
    method = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    clustering_run_id = Column(String(64))
    assigned_by = Column(String(100))

    # Evidence for the assignment
    evidence = Column(JSONB, default=dict)

    # Temporal validity
    valid_from = Column(DateTime, default=datetime.utcnow, nullable=False)
    valid_to = Column(DateTime)

    # Relationships
    speaker_embedding = relationship("Speaker", back_populates="assignments")
    speaker_identity = relationship("SpeakerIdentity", back_populates="assignments")

    __table_args__ = (
        UniqueConstraint('speaker_embedding_id', 'valid_to', name='unique_current_assignment'),
        Index('idx_speaker_assignments_identity', 'speaker_identity_id'),
        Index('idx_speaker_assignments_embedding', 'speaker_embedding_id'),
        Index('idx_speaker_assignments_run', 'clustering_run_id'),
        Index('idx_speaker_assignments_current', 'speaker_embedding_id', 'valid_to',
              postgresql_where=text('valid_to IS NULL')),
    )


class Speaker(Base):
    """Unified speaker table with embeddings and identity data.

    **Database Sequence Requirement:**
    This model requires a PostgreSQL sequence named 'speaker_number_seq' for
    atomic sequential numbering. Create it with:
    ```sql
    CREATE SEQUENCE speaker_number_seq START 1 INCREMENT 1;
    ```

    **Key Design Changes:**
    - Consolidates Speaker and SpeakerEmbedding into single table
    - Enforces 1:1 constraint on (content_id, local_speaker_id)
    - Each record represents unique content-speaker observation
    - Supports canonical speaker relationships for clustering

    **Creating Speakers:**
    Use `Speaker.create_with_sequential_name()` for new speakers to ensure
    proper sequential numbering without race conditions.
    """
    __tablename__ = 'speakers'

    id = Column(Integer, primary_key=True)

    # Content-Speaker Identity (1:1 constraint)
    content_id = Column(String, nullable=False)  # Content where this speaker was observed
    local_speaker_id = Column(String, nullable=False)  # Local speaker ID from diarization (SPEAKER_00, etc.)

    # Speaker Identity
    speaker_hash = Column(String, nullable=False, unique=True)  # Unique speaker hash identifier
    display_name = Column(String)  # Human-readable name if manually assigned

    # Embedding Data (Enriched from Stitch)
    embedding = Column(Vector(512), nullable=True)  # Enriched speaker centroid from stitch (512-dim, NULL if insufficient data)
    embedding_quality_score = Column(Float, default=1.0)  # Quality/confidence score of enriched embedding
    algorithm_version = Column(String, default='stage6b_tight_clusters')  # Algorithm version for enriched embedding

    # Raw Diarization Embedding (from FluidAudio)
    embedding_diarization = Column(Vector(256), nullable=True)  # Raw embedding from FluidAudio diarization (256-dim)
    embedding_diarization_quality = Column(Float, nullable=True)  # Quality score for raw diarization embedding

    # Content Statistics
    duration = Column(Float, default=0.0)  # Duration of speech this embedding represents
    segment_count = Column(Integer, default=0)  # Number of segments this embedding represents

    # Clustering
    cluster_id = Column(String(64), nullable=True)  # Current cluster assignment

    # Processing Status - tracks progress through 3-phase pipeline
    rebase_status = Column(Enum(SpeakerProcessingStatus), default=SpeakerProcessingStatus.PENDING, nullable=False)
    rebase_batch_id = Column(String, nullable=True)  # Batch ID for tracking processing operations
    rebase_processed_at = Column(DateTime, nullable=True)  # When this speaker was last processed

    # Metadata
    meta_data = Column(JSONB, default=dict)  # Includes 'clustering_timestamp' (ISO-8601) - last time speaker was evaluated for clustering
    notes = Column(Text)  # Additional notes about this speaker
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Identity relationship (simplified schema - Dec 2025 refactor)
    speaker_identity_id = Column(Integer, ForeignKey('speaker_identities.id'), nullable=True)
    assignment_confidence = Column(Float, nullable=True)  # 0-1 confidence score
    assignment_phase = Column(String(20), nullable=True)  # 'phase2' | 'phase3' | 'phase4' | 'manual'

    # JSONB audit trail for all identification phases
    # Structure: {"phase2": {...}, "phase3": {...}, "phase4": {...}}
    identification_details = Column(JSONB, default=dict)

    # Relationships
    sentences = relationship("Sentence", back_populates="speaker")

    # Identity relationships
    speaker_identity = relationship("SpeakerIdentity", back_populates="speaker_embeddings")
    assignments = relationship("SpeakerAssignment", back_populates="speaker_embedding")

    # Indexes and Constraints
    __table_args__ = (
        # Primary 1:1 constraint - each content-speaker pair has exactly one record
        UniqueConstraint('content_id', 'local_speaker_id', name='uq_speaker_content_local'),

        # Standard indexes
        Index('idx_speaker_hash', 'speaker_hash'),
        Index('idx_speaker_content_id', 'content_id'),
        Index('idx_speaker_local_speaker_id', 'local_speaker_id'),
        Index('idx_speaker_duration', 'duration'),
        Index('idx_speaker_quality', 'embedding_quality_score'),
        Index('idx_speaker_rebase_status', 'rebase_status'),
        Index('idx_speaker_rebase_batch', 'rebase_batch_id'),

        # Composite indexes for common query patterns
        Index('idx_speaker_quality_duration', 'embedding_quality_score', 'duration'),
        Index('idx_speaker_rebase_pending', 'rebase_status', 'created_at'),

        # Index for embedding similarity search (HNSW for fast vector similarity)
        Index('idx_speaker_embedding', 'embedding', postgresql_using='hnsw', postgresql_with={'m': 16, 'ef_construction': 64}),

        # Performance optimization indexes for speaker annotation dashboard
        Index('idx_speakers_cluster', 'cluster_id'),

        # Speaker management pipeline indices
        Index('idx_speaker_identity', 'speaker_identity_id'),
        Index('idx_speakers_identity_count', 'speaker_identity_id',
              postgresql_where=text('speaker_identity_id IS NOT NULL')),
        Index('idx_speakers_quality_filter', 'embedding_quality_score',
              postgresql_where=text('embedding IS NOT NULL AND embedding_quality_score >= 0.5')),

        # New schema indexes (Dec 2025 refactor) - defined in migration
        # idx_speakers_assignment_phase - assignment_phase filtering
        # idx_speakers_identification_details - GIN for JSONB
        # idx_speakers_phase2_pending - speakers needing Phase 2
        # idx_speakers_phase2_certain - Phase 2 certain (for centroid building)
        # idx_speakers_unassigned - unassigned with embeddings (for Phase 4)
    )

    def __init__(self, content_id: str, local_speaker_id: str, embedding: np.ndarray = None,
                 duration: float = 0.0, segment_count: int = 0,
                 embedding_quality_score: float = 1.0, algorithm_version: str = "stage6b_tight_clusters",
                 speaker_hash: str = None,
                 rebase_status=None, rebase_batch_id: str = None):
        """Initialize with numpy array embedding (None for speakers with insufficient data)"""
        self.content_id = content_id
        self.local_speaker_id = local_speaker_id
        self.duration = duration
        self.segment_count = segment_count
        self.embedding_quality_score = embedding_quality_score
        self.algorithm_version = algorithm_version
        self.rebase_status = rebase_status or SpeakerProcessingStatus.PENDING
        self.rebase_batch_id = rebase_batch_id

        # Generate speaker hash if not provided
        if speaker_hash is None:
            self.speaker_hash = self.create_deterministic_hash(content_id, local_speaker_id)
        else:
            self.speaker_hash = speaker_hash

        # Handle embedding (None for speakers with insufficient data)
        if embedding is not None:
            # Ensure embedding is float32 and contiguous
            if not isinstance(embedding, np.ndarray):
                raise ValueError("Embedding must be a numpy array")

            # Ensure embedding is normalized
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            else:
                raise ValueError("Embedding has zero norm")

            # Ensure embedding is float32 and contiguous
            embedding = np.ascontiguousarray(embedding, dtype=np.float32)

        # Validate embedding if not None
        if embedding is not None and (np.isnan(embedding).any() or np.isinf(embedding).any()):
            raise ValueError("Embedding contains NaN or Inf values")

        # Store as vector for similarity search - pgvector handles numpy arrays directly
        self.embedding = embedding

    def __repr__(self):
        return f"<Speaker(speaker_hash='{self.speaker_hash}', content_id='{self.content_id}', local_speaker_id='{self.local_speaker_id}')>"

    @property
    def country(self) -> Optional[str]:
        """Get speaker's country from metadata"""
        return self.meta_data.get('country') if self.meta_data else None

    @property
    def role(self) -> Optional[str]:
        """Get speaker's role from metadata"""
        return self.meta_data.get('role') if self.meta_data else None

    def update_metadata(self, **kwargs) -> None:
        """Update speaker metadata"""
        if not self.meta_data:
            self.meta_data = {}
        self.meta_data.update(kwargs)
        self.updated_at = datetime.utcnow()

    @staticmethod
    def create_deterministic_hash(content_id: str, local_speaker_id: str) -> str:
        """
        Create a deterministic 8-character hash from content_id and local_speaker_id.

        Args:
            content_id: Content ID where speaker was observed
            local_speaker_id: Local speaker ID from diarization (e.g., SPEAKER_00)

        Returns:
            str: 8-character hash that uniquely identifies this content-speaker pair
        """
        hash_input = f"{content_id}:{local_speaker_id}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:8]

    @classmethod
    def create_with_sequential_name(cls, session, content_id: str, local_speaker_id: str,
                                  embedding: np.ndarray = None, **kwargs) -> 'Speaker':
        """
        Create a new speaker with deterministic speaker hash.

        Args:
            session: SQLAlchemy session for database operations
            content_id: Content ID where speaker was observed
            local_speaker_id: Local speaker ID from diarization
            embedding: Speaker embedding as numpy array (None for speakers with insufficient data)
            **kwargs: Additional speaker attributes

        Returns:
            Speaker: New speaker instance with speaker_hash

        Example:
            ```python
            with get_session() as session:
                # With embedding
                speaker = Speaker.create_with_sequential_name(session, "content123", "SPEAKER_00", embedding_array)

                # Without embedding (insufficient data)
                speaker = Speaker.create_with_sequential_name(session, "content123", "SPEAKER_01", None)

                session.add(speaker)
                session.commit()
            ```
        """
        # Create speaker using the standard constructor (which will generate speaker_hash)
        return cls(content_id=content_id, local_speaker_id=local_speaker_id,
                  embedding=embedding, **kwargs)

    @classmethod
    def get_speakers_with_stats(cls, session):
        """Get all speakers with their statistics"""
        return session.query(
            cls,
            func.count(cls.transcriptions).label('total_segments'),
            func.count(func.distinct(cls.transcriptions.any(SpeakerTranscription.content_id))).label('total_content')
        ).outerjoin(cls.transcriptions).group_by(cls.id).order_by(text('total_segments DESC')).all()

    @classmethod
    def find_by_content_speaker(cls, session, content_id: str, local_speaker_id: str) -> Optional['Speaker']:
        """Find speaker by content-speaker pair"""
        return session.query(cls).filter(
            cls.content_id == content_id,
            cls.local_speaker_id == local_speaker_id
        ).first()

    @classmethod
    def find_similar_speakers(cls, session, embedding: np.ndarray,
                            similarity_threshold: float = 0.85, limit: int = 10) -> List[Tuple['Speaker', float]]:
        """Find similar speakers using pgvector similarity search"""
        # Determine embedding dimension
        embedding_dim = len(embedding)

        # Convert embedding to list for pgvector
        embedding_list = embedding.tolist()

        # Use pgvector similarity search
        query = text(f"""
            SELECT
                s.*,
                1 - (s.embedding <=> CAST(:embedding AS vector({embedding_dim}))) as similarity
            FROM speakers s
            WHERE 1 - (s.embedding <=> CAST(:embedding AS vector({embedding_dim}))) > :threshold
            ORDER BY similarity DESC
            LIMIT :limit
        """)

        result = session.execute(query, {
            'embedding': embedding_list,
            'threshold': similarity_threshold,
            'limit': limit
        })

        # Convert results to Speaker objects with similarity scores
        similar_speakers = []
        for row in result:
            speaker = session.query(cls).filter_by(id=row.id).first()
            if speaker:
                similarity = float(row.similarity)
                similar_speakers.append((speaker, similarity))

        return similar_speakers


# SpeakerEmbedding table has been consolidated into the unified Speaker table above
# This maintains all embedding functionality while enforcing proper 1:1 constraints


class SpeakerTranscription(Base):
    """
    DEPRECATED: Speaker-attributed transcript segments (speaker turns).

    This model has been superseded by the Sentence model. New content uses
    Sentence records exclusively. This class is retained only for reference
    during the transition period.

    DEPRECATION NOTICE (January 2026):
    - The speaker_transcriptions table will be DROPPED in an upcoming migration
    - All 348,579 legacy content items have been migrated to the sentences table
    - New content (75,992+ items) only has sentence records, not speaker_transcriptions
    - Use Sentence model for all new code

    S3 FILE CLEANUP NOTE:
    Some legacy content may still have speaker_turns.json files in S3:
    - Location: content/{content_id}/speaker_turns.json
    - These are OUTPUT files from the old stitch pipeline, not source data
    - Safe to delete after verifying sentences exist for the content
    - The word_table.pkl.gz files are the authoritative source for re-migration

    REPLACEMENT:
    - Speaker turns can be reconstructed from sentences by grouping on (content_id, turn_index)
    - Use: SELECT turn_index, string_agg(text, ' ' ORDER BY sentence_in_turn) FROM sentences GROUP BY turn_index
    """
    __tablename__ = 'speaker_transcriptions'

    id = Column(Integer, primary_key=True)
    content_id = Column(Integer, ForeignKey('content.id'), nullable=False, index=True)
    speaker_id = Column(Integer, ForeignKey('speakers.id'), nullable=False, index=True)
    speaker_hash = Column(String(8), nullable=True, index=True)  # Speaker hash for quick identification
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    text = Column(Text, nullable=False)
    turn_index = Column(Integer, nullable=False) # Sequential index of this turn within the content
    stitch_version = Column(String(50), nullable=True) # Version of the stitching algorithm used for this turn
    created_at = Column(DateTime(timezone=True), default=func.now()) # Use func.now() for DB time
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())

    # Relationships - DEPRECATED: These relationships are commented out as the model is deprecated
    # content = relationship("Content", back_populates="speaker_transcriptions")
    # speaker = relationship("Speaker", back_populates="transcriptions")

    # Indexes
    __table_args__ = (
        Index('idx_speaker_transcriptions_content_id', 'content_id'),
        Index('idx_speaker_transcriptions_speaker_id', 'speaker_id'),
        Index('idx_speaker_transcriptions_start_time', 'start_time'),
        Index('idx_speaker_transcriptions_end_time', 'end_time'),
        Index('idx_speaker_transcriptions_text', 'text', postgresql_using='gin', postgresql_ops={'text': 'gin_trgm_ops'}), # Optional for text search
        Index('idx_speaker_transcriptions_turn_index', 'turn_index'),
        # Performance optimization indexes for speaker annotation dashboard
        Index('idx_speaker_transcriptions_content_speaker', 'content_id', 'speaker_id'),
        Index('idx_speaker_transcriptions_speaker_content_times', 'speaker_id', 'content_id', 'start_time', 'end_time')
    )

    def __repr__(self):
        has_embedding = "Yes" if hasattr(self, 'embedding') and self.embedding is not None else "No"
        stitch_v_info = f", stitch_version={self.stitch_version}" if self.stitch_version else ""
        return (f"<SpeakerTranscription(id={self.id}, content_id={self.content_id}, speaker_id={self.speaker_id}, "
                f"turn={self.turn_index}, start={self.start_time:.2f}s, end={self.end_time:.2f}s, "
                f"embedding={has_embedding}{stitch_v_info})>")

    @classmethod
    def get_context_turns(cls, session, content_id: int, center_turn_index: int, num_context_turns: int) -> List[Tuple[int, str, str]]:
        """Fetches context turns around a central turn index.

        Args:
            session: The SQLAlchemy session.
            content_id: The integer ID of the Content record.
            center_turn_index: The index of the turn to center the context around.
            num_context_turns: The number of turns to fetch before and after the center.

        Returns:
            A list of tuples: (turn_index, speaker_identifier, text), ordered by turn_index.
            Speaker identifier is speaker_hash if available.
        """
        if num_context_turns <= 0:
            return []

        min_ctx_idx = center_turn_index - num_context_turns
        max_ctx_idx = center_turn_index + num_context_turns

        try:
            context_results = (
                session.query(
                    cls.turn_index,
                    Speaker.speaker_hash,
                    cls.text,
                    cls.speaker_id # Include speaker_id for fallback identifier
                )
                .join(Speaker, cls.speaker_id == Speaker.id)
                .filter(
                    cls.content_id == content_id,
                    cls.turn_index >= min_ctx_idx,
                    cls.turn_index <= max_ctx_idx
                )
                .order_by(cls.turn_index)
                .all()
            )

            # Format the results
            formatted_context = []
            for turn_idx, speaker_hash, txt, spk_id in context_results:
                # Use speaker_hash or fallback to speaker_id
                speaker_name = speaker_hash or f"Speaker_{spk_id}"
                formatted_context.append((turn_idx, speaker_name, txt))

            return formatted_context

        except Exception as e:
            logger.error(f"Error fetching context turns for content_id={content_id}, turn_index={center_turn_index}: {e}", exc_info=True)
            return []

    @classmethod
    def get_context_window(cls, session, content_id: int, turn_index: int, turns_before: int = 2, turns_after: int = 2) -> List['SpeakerTranscription']:
        """Get a window of SpeakerTranscription objects around a specific turn index."""
        min_idx = turn_index - turns_before
        max_idx = turn_index + turns_after
        return session.query(cls).filter(
            cls.content_id == content_id,
            cls.turn_index >= min_idx,
            cls.turn_index <= max_idx
        ).order_by(cls.turn_index).all()

    @classmethod
    def get_time_window(cls, session, content_id: int, center_time: float, seconds_before: float = 30.0, seconds_after: float = 30.0) -> List['SpeakerTranscription']:
        """Get turns within a time window around a specific time"""
        return session.query(cls).filter(
            cls.content_id == content_id,
            cls.start_time >= center_time - seconds_before,
            cls.end_time <= center_time + seconds_after
        ).order_by(cls.start_time).all()

    @classmethod
    def search_with_context(cls, session, content_id: int, search_term: str, turns_before: int = 2, turns_after: int = 2) -> List[Dict]:
        """Search for a term and return matches with context"""
        # Use PostgreSQL full-text search
        matches = session.query(cls).filter(
            cls.content_id == content_id,
            cls.text.ilike(f"%{search_term}%")
        ).order_by(cls.turn_index).all()

        results = []
        for match in matches:
            context = cls.get_context_window(session, content_id, match.turn_index, turns_before, turns_after)
            results.append({
                'match': match,
                'context': context
            })
        return results

    @classmethod
    def search_with_time_context(cls, session, content_id: int, search_term: str, seconds_before: float = 30.0, seconds_after: float = 30.0) -> List[Dict]:
        """Search for a term and return matches with time-based context"""
        matches = session.query(cls).filter(
            cls.content_id == content_id,
            cls.text.ilike(f"%{search_term}%")
        ).order_by(cls.start_time).all()

        results = []
        for match in matches:
            context = cls.get_time_window(session, content_id, match.start_time, seconds_before, seconds_after)
            results.append({
                'match': match,
                'context': context
            })
        return results


class Sentence(Base):
    """
    Atomic unit for transcript data with sentence-level granularity.

    Sentences replace speaker_transcriptions as the primary atomic unit,
    enabling granular emotion queries and efficient segment retrieval.
    Speaker turns can be reconstructed by grouping on (content_id, turn_index).

    Attributes:
        id: Primary key
        content_id: Foreign key to Content
        speaker_id: Foreign key to Speaker

        Position indices:
        sentence_index: Global index within content (0, 1, 2...)
        turn_index: Speaker turn this belongs to
        sentence_in_turn: Position within turn (0, 1, 2...)

        Text & timing:
        text: Sentence text
        start_time: Start time in seconds (word-level precision)
        end_time: End time in seconds (word-level precision)
        word_count: Number of words in sentence

        Emotion (nullable until emotion stage runs):
        emotion: Primary detected emotion (e.g., 'angry', 'happy', 'neutral')
        emotion_confidence: Confidence score for primary emotion
        emotion_scores: Full distribution as JSONB {"angry": 0.1, "happy": 0.8, ...}
        arousal: Dimensional score (0-1)
        valence: Dimensional score (0-1, negative to positive)
        dominance: Dimensional score (0-1)

        Metadata:
        stitch_version: Version of algorithm that created this
        created_at: When this record was created

    Relationships:
        content: The Content this belongs to
        speaker: The global Speaker record

    Usage Patterns:
        - Query by speaker_id + emotion for "what did speaker X say when angry"
        - Group by (content_id, turn_index) to reconstruct speaker turns
        - Use emotion_confidence > threshold for high-confidence emotion filtering
        - Link to embedding_segments via source_sentence_ids array
    """
    __tablename__ = 'sentences'

    id = Column(Integer, primary_key=True)
    content_id = Column(Integer, ForeignKey('content.id', ondelete='CASCADE'), nullable=False, index=True)
    speaker_id = Column(Integer, ForeignKey('speakers.id', ondelete='CASCADE'), nullable=False, index=True)

    # Position indices
    sentence_index = Column(Integer, nullable=False)      # global index within content (0, 1, 2...)
    turn_index = Column(Integer, nullable=False)          # speaker turn this belongs to
    sentence_in_turn = Column(Integer, nullable=False)    # position within turn (0, 1, 2...)

    # Text & timing (word-level precision from stitch)
    text = Column(Text, nullable=False)
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    word_count = Column(Integer, nullable=False)

    # Emotion (nullable until emotion stage runs)
    emotion = Column(String(30), nullable=True)
    emotion_confidence = Column(Float, nullable=True)
    emotion_scores = Column(JSONB, nullable=True)  # {"angry": 0.1, "happy": 0.8, ...}
    arousal = Column(Float, nullable=True)
    valence = Column(Float, nullable=True)
    dominance = Column(Float, nullable=True)

    # Metadata
    stitch_version = Column(String(50), nullable=True)
    created_at = Column(DateTime(timezone=True), default=func.now())

    # Relationships
    content = relationship("Content", back_populates="sentences")
    speaker = relationship("Speaker", back_populates="sentences")

    __table_args__ = (
        Index('idx_sentences_content', 'content_id'),
        Index('idx_sentences_speaker', 'speaker_id'),
        Index('idx_sentences_turn', 'content_id', 'turn_index'),
        Index('idx_sentences_time', 'content_id', 'start_time', 'end_time'),
        Index('idx_sentences_emotion', 'emotion', postgresql_where=sa_text('emotion IS NOT NULL')),
        Index('idx_sentences_speaker_emotion', 'speaker_id', 'emotion', postgresql_where=sa_text('emotion IS NOT NULL')),
        UniqueConstraint('content_id', 'sentence_index', name='uq_sentence_content_index'),
    )

    def __repr__(self):
        emotion_info = f", emotion={self.emotion}({self.emotion_confidence:.2f})" if self.emotion else ""
        return (f"<Sentence(id={self.id}, content_id={self.content_id}, speaker_id={self.speaker_id}, "
                f"sentence_idx={self.sentence_index}, turn={self.turn_index}, "
                f"start={self.start_time:.2f}s, end={self.end_time:.2f}s{emotion_info})>")

    @classmethod
    def get_turn_sentences(cls, session, content_id: int, turn_index: int) -> List['Sentence']:
        """Get all sentences for a specific speaker turn, ordered by position."""
        return session.query(cls).filter(
            cls.content_id == content_id,
            cls.turn_index == turn_index
        ).order_by(cls.sentence_in_turn).all()

    @classmethod
    def reconstruct_turn(cls, session, content_id: int, turn_index: int) -> Dict:
        """Reconstruct a speaker turn from its sentences."""
        sentences = cls.get_turn_sentences(session, content_id, turn_index)
        if not sentences:
            return None
        return {
            'turn_index': turn_index,
            'speaker_id': sentences[0].speaker_id,
            'text': ' '.join(s.text for s in sentences),
            'start_time': sentences[0].start_time,
            'end_time': sentences[-1].end_time,
            'emotions': [s.emotion for s in sentences if s.emotion],
            'sentences': sentences
        }

    @classmethod
    def get_emotional_sentences(cls, session, speaker_id: int, emotion: str,
                                min_confidence: float = 0.25, limit: int = 100) -> List['Sentence']:
        """Find sentences where a speaker expressed a specific emotion."""
        return session.query(cls).filter(
            cls.speaker_id == speaker_id,
            cls.emotion == emotion,
            cls.emotion_confidence >= min_confidence
        ).order_by(cls.emotion_confidence.desc()).limit(limit).all()
