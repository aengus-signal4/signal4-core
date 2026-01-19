"""
Content models for the audio-visual processing pipeline.

Contains:
- Content: Central model representing a piece of media (video/podcast)
- ContentChunk: Audio chunks for parallel processing
- Transcription: Legacy transcription model (being phased out)
- Source: Content sources (YouTube channels, podcast feeds)
"""

from sqlalchemy import (
    Column, Integer, SmallInteger, String, Float, DateTime, JSON, ForeignKey,
    Text, Boolean, Index, ARRAY, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from typing import Dict
from pathlib import Path

from .base import Base


class Content(Base):
    """
    Central model representing a piece of media content (video or podcast).

    This is the main entity that flows through the processing pipeline. Content
    is discovered from sources (channels/feeds), downloaded, converted to audio,
    transcribed, diarized, and analyzed.

    Attributes:
        id: Primary key
        projects: Comma-separated list of projects this content belongs to
        platform: Source platform ('youtube', 'podcast', etc.)
        content_id: Platform-specific content identifier
        channel_name: Name of the source channel/podcast
        channel_url: URL of the source channel/feed
        title: Content title
        description: Content description
        publish_date: When the content was published
        download_date: When we downloaded the content
        duration: Total duration in seconds
        meta_data: Additional metadata as JSON

    Processing State Flags:
        is_downloaded: Media file has been downloaded to S3
        is_converted: Audio extracted and chunked into WAV files
        is_transcribed: All chunks have been transcribed
        is_diarized: Speaker diarization has been completed
        is_identified: Speakers mapped to global IDs (deprecated)
        is_stitched: Final transcript assembled (deprecated)
        is_embedded: Embedding segments have been created
        is_compressed: Storage has been optimized and compressed

    Processing Control:
        blocked_download: Download was blocked (e.g., 403 error)
        is_duplicate: This is a duplicate of another content
        is_short: This is a YouTube short
        processing_priority: Priority for task scheduling
        stitch_version: Version of stitching algorithm used

    Chunk Management:
        total_chunks: Number of chunks after splitting
        chunks_processed: Number of chunks completed
        chunks_status: JSON status of each chunk

    S3 Storage:
        s3_source_key: Key for source media file
        s3_transcript_key: Key for final transcript
        s3_metadata_key: Key for content metadata

    Relationships:
        transcriptions: Legacy transcription records
        chunks: ContentChunk records for parallel processing
        speaker_transcriptions: Final speaker-attributed segments
        embedding_segments: Retrieval-optimized segments

    Standard Paths:
        Audio: content/{content_id}/audio.wav
        Chunks: content/{content_id}/chunks/{index}/
        Transcript: content/{content_id}/transcript.json
        Metadata: content/{content_id}/meta.json
    """
    __tablename__ = 'content'

    id = Column(Integer, primary_key=True)
    projects = Column(ARRAY(String), nullable=False)  # Array of project names
    platform = Column(String, nullable=False)  # youtube, twitch, podcast
    content_id = Column(String, nullable=False)
    channel_id = Column(Integer, ForeignKey('channels.id'), nullable=True, index=True)  # NEW: Link to channels table
    channel_name = Column(String, nullable=False)
    channel_url = Column(String, nullable=False)
    title = Column(String)
    description = Column(Text)
    publish_date = Column(DateTime)
    download_date = Column(DateTime, default=datetime.utcnow)
    duration = Column(Float)  # in seconds
    meta_data = Column(JSON)  # Additional metadata
    main_language = Column(String(10), nullable=True, index=True)  # Primary language of content (ISO 639-1 codes: en, fr, ro, etc.)
    is_downloaded = Column(Boolean, default=False, nullable=False)
    is_converted = Column(Boolean, default=False, nullable=False)  # Whether audio has been converted to WAV
    is_transcribed = Column(Boolean, default=False, nullable=False)  # Whether content has been transcribed
    is_duplicate = Column(Boolean, default=False)  # Whether this is a duplicate of another content
    is_diarized = Column(Boolean, default=False)  # Whether speaker diarization has been completed
    is_identified = Column(Boolean, default=False, nullable=False, index=True) # ADDED: Whether speakers have been identified globally
    is_stitched = Column(Boolean, default=False, nullable=False, index=True)  # Whether diarization has been stitched to transcript
    is_embedded = Column(Boolean, default=False, nullable=False, index=True)  # Future: Whether content/segments have embeddings
    is_compressed = Column(Boolean, default=False, nullable=False, index=True)  # Whether storage has been optimized/compressed
    stitch_version = Column(String, nullable=True, server_default=None, index=True)  # Version of stitching algorithm used (NULL if not yet stitched)

    # Linear processing state (replaces boolean flags above)
    # -1=BLOCKED, 0=NEW, 1=DOWNLOADED, 2=CONVERTED, 3=DIARIZED, 4=TRANSCRIBED,
    # 5=STITCHED, 6=EMBEDDED, 7=IDENTIFIED, 8=COMPLETE, 9=COMPRESSED
    processing_state = Column(SmallInteger, default=0, nullable=False, index=True)
    failed_at_state = Column(SmallInteger, nullable=True)  # State where failure occurred
    state_failure_reason = Column(Text, nullable=True)  # Error message for failures
    diarization_method = Column(String(50), nullable=True, index=True)  # Method used for diarization ('pyannote3.1', 'fluid_audio', etc.)

    # Suggestion for new feature: Consider adding a dedicated diarization_ignored = Column(Boolean, default=False)
    # field to mark content (e.g., music-only) where diarization should not be retried.
    # Immediate fix: This is currently tracked in meta_data['diarization_ignored'] by the pipeline manager.

    blocked_download = Column(Boolean, default=False, nullable=False)  # Whether download was blocked (e.g., 403 error)
    is_short = Column(Boolean, default=False, nullable=False)  # Whether this is a YouTube short
    processing_priority = Column(Integer, default=0)  # Added processing_priority field
    total_chunks = Column(Integer)  # Total number of chunks for this content
    chunks_processed = Column(Integer, default=0)  # Number of chunks processed
    chunks_status = Column(JSON, default=dict)  # Status of each chunk: {chunk_id: status}
    last_updated = Column(DateTime(timezone=True), onupdate=datetime.utcnow)

    # S3 storage fields
    s3_source_key = Column(String)  # S3 key for source file
    s3_transcript_key = Column(String)  # S3 key for final transcript
    s3_metadata_key = Column(String)  # S3 key for content metadata

    # Speaker metadata columns (Phase 1 speaker identification)
    hosts = Column(JSON, default=list)  # [{"name": "...", "confidence": "...", "reasoning": "..."}]
    guests = Column(JSON, default=list)  # [{"name": "...", "confidence": "...", "reasoning": "..."}]
    mentioned = Column(JSON, default=list)  # [{"name": "...", "reasoning": "..."}]
    hosts_consolidated = Column(Boolean, default=False)  # Whether host names have been consolidated to canonical forms

    # Relationships
    channel = relationship("Channel", back_populates="content")
    transcriptions = relationship("Transcription", back_populates="content", cascade="all, delete-orphan", overlaps="transcription")
    chunks = relationship("ContentChunk", back_populates="content", cascade="all, delete-orphan")
    speaker_transcriptions = relationship("SpeakerTranscription", back_populates="content", cascade="all, delete-orphan")
    sentences = relationship("Sentence", back_populates="content", cascade="all, delete-orphan")
    embedding_segments = relationship("EmbeddingSegment", back_populates="content", cascade="all, delete-orphan")

    # Indexes
    __table_args__ = (
        Index('idx_content_projects', 'projects'),
        Index('idx_content_platform', 'platform'),
        Index('idx_content_channel', 'channel_name'),
        Index('idx_content_is_downloaded', 'is_downloaded'),
        Index('idx_content_is_converted', 'is_converted'),
        Index('idx_content_is_transcribed', 'is_transcribed'),
        Index('idx_content_is_diarized', 'is_diarized'),
        Index('idx_content_is_identified', 'is_identified'),
        Index('idx_content_is_stitched', 'is_stitched'),
        Index('idx_content_is_embedded', 'is_embedded'),
        Index('idx_content_is_duplicate', 'is_duplicate'),
        Index('idx_content_blocked_download', 'blocked_download'),
        Index('idx_content_is_short', 'is_short'),
        Index('idx_content_content_id', 'content_id'),
        Index('idx_content_last_updated', 'last_updated'),
        Index('idx_content_s3_source_key', 's3_source_key'),
        Index('idx_content_s3_transcript_key', 's3_transcript_key'),
        Index('idx_content_stitch_version', 'stitch_version'),
        Index('idx_content_main_language', 'main_language'),
        Index('idx_content_diarization_method', 'diarization_method'),
        # Performance optimization index for project filtering in speaker annotation dashboard
        Index('idx_content_projects_gin', 'projects', postgresql_using='gin'),
        # Unique constraint on (platform, content_id) to prevent duplicate content
        UniqueConstraint('platform', 'content_id', name='uq_content_platform_content_id')
    )

    def init_s3_structure(self, s3_storage) -> bool:
        """Initialize S3 structure for this content. Returns True if successful."""
        try:
            paths = self.get_s3_paths()

            # Create content metadata
            metadata = {
                'projects': self.projects,
                'platform': self.platform,
                'channel_name': self.channel_name,
                'channel_url': self.channel_url,
                'title': self.title,
                'description': self.description,
                'publish_date': self.publish_date.isoformat() if self.publish_date else None,
                'duration': self.duration,
                'main_language': self.main_language,
                'meta_data': self.meta_data
            }

            # Upload metadata file
            if s3_storage.upload_json(paths['meta'], metadata):
                self.s3_metadata_key = paths['meta']
                return True
            return False

        except Exception as e:
            return False

    def upload_source_to_s3(self, s3_storage, local_path: str) -> bool:
        """
        Upload source file to S3 and update metadata with extension.
        Returns True if successful.
        """
        try:
            # Get extension from local file
            extension = Path(local_path).suffix
            if not extension:
                extension = self.source_extension  # Use default if no extension found

            # Update metadata with actual extension
            self.update_source_extension(extension)

            # Get S3 paths (this will now use the updated extension)
            paths = self.get_s3_paths()

            # First ensure metadata exists
            if not self.s3_metadata_key:
                if not self.init_s3_structure(s3_storage):
                    return False

            # Upload source file
            if s3_storage.upload_file(local_path, paths['source']):
                self.s3_source_key = paths['source']
                return True
            return False

        except Exception as e:
            return False

    def ensure_s3_structure(self, s3_storage) -> bool:
        """
        Ensure S3 structure exists, creating it if needed.
        Returns True if structure exists or was created successfully.
        """
        if self.s3_metadata_key and s3_storage.file_exists(self.s3_metadata_key):
            return True
        return self.init_s3_structure(s3_storage)

    @property
    def source_extension(self) -> str:
        """Get the source file extension from metadata"""
        # First check if we have a stored extension
        if self.meta_data and 'source_extension' in self.meta_data:
            return self.meta_data['source_extension']

        # If not, use platform defaults
        if self.platform == 'podcast':
            return '.mp3'
        return '.mp4'  # Default for video platforms

    def update_source_extension(self, extension: str):
        """Update the source file extension in metadata"""
        if not self.meta_data:
            self.meta_data = {}
        self.meta_data['source_extension'] = extension
        # Also update s3_source_key if it exists
        if self.s3_source_key:
            paths = self.get_s3_paths()
            self.s3_source_key = paths['source']

    def get_chunk_count(self):
        """Get the total number of chunks for this content"""
        return self.total_chunks if self.total_chunks is not None else 0

    def get_processed_chunk_count(self):
        """Get the number of processed chunks for this content"""
        return self.chunks_processed if self.chunks_processed is not None else 0

    def get_chunk_duration(self):
        """Get the target duration for each chunk in seconds"""
        return 300  # 5 minutes

    def get_chunk_overlap(self):
        """Get the overlap duration between chunks in seconds"""
        return 30  # 30 seconds overlap

    def get_s3_key(self):
        """Get the S3 key for this content's audio file"""
        return f"{self.content_id}/audio.mp3"

    def get_chunk_s3_key(self, chunk_index):
        """Get the S3 key for a specific chunk's audio file"""
        return f"{self.content_id}/chunks/chunk_{chunk_index}.mp3"

    def get_s3_paths(self) -> Dict[str, str]:
        """Get all S3 paths for this content"""
        content_prefix = f"content/{self.content_id}"

        return {
            'meta': f"{content_prefix}/meta.json",
            'source': f"{content_prefix}/source{self.source_extension}",
            'transcript': f"{content_prefix}/transcript.json",
            'chunks_prefix': f"{content_prefix}/chunks",
            'audio': f"{content_prefix}/audio.mp3",
            'transcript_words': f"{content_prefix}/transcript_words.json"
        }

    def get_chunk_s3_paths(self, chunk_index: int) -> Dict[str, str]:
        """Get all S3 paths for a chunk"""
        chunk_prefix = f"{self.get_s3_paths()['chunks_prefix']}/{chunk_index}"
        return {
            'audio': f"{chunk_prefix}/audio.wav",
            'meta': f"{chunk_prefix}/meta.json",
            'transcript': f"{chunk_prefix}/transcript.json",
            'transcript_words': f"{chunk_prefix}/transcript_words.json"
        }

    @property
    def download_path(self) -> str:
        """Get the path where the downloaded file should be stored"""
        ext = '.mp3' if self.platform == 'podcast' else '.mp4'
        return f"data/downloads/{self.content_id}{ext}"

    @property
    def wav_segments_path(self) -> str:
        """Get the path where WAV segments should be stored"""
        return f"data/wav_segments"

    @property
    def partial_transcripts_path(self) -> str:
        """Get the path where partial transcripts should be stored"""
        return f"data/partial_transcripts"

    @property
    def transcription_path(self) -> str:
        """Get the path where the final transcription should be stored"""
        return f"data/transcriptions/{self.content_id}_transcription.json"

    def get_storage_paths(self, storage_type: str = 'local') -> Dict[str, str]:
        """Get all storage paths for this content based on storage type"""
        if storage_type == 'local':
            return {
                'source': self.download_path,
                'wav_segments': self.wav_segments_path,
                'partial_transcripts': self.partial_transcripts_path,
                'final_transcript': self.transcription_path,
                'final_transcript_words': f"final/{self.content_id}/transcript_words.json"
            }
        elif storage_type == 's3':
            ext = 'mp3' if self.platform == 'podcast' else 'mp4'
            return {
                'source': f"raw/{self.content_id}/source.{ext}",
                'wav_segments': f"chunks/{self.content_id}",
                'partial_transcripts': f"transcripts/{self.content_id}",
                'final_transcript': f"final/{self.content_id}/transcript.json",
                'final_transcript_words': f"final/{self.content_id}/transcript_words.json"
            }
        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")

    def get_chunk_path(self, chunk_index: int, start_time: float, end_time: float, storage_type: str = 'local') -> str:
        """Get path for a specific chunk file"""
        if storage_type == 'local':
            return f"{self.wav_segments_path}/{self.content_id}_{chunk_index}_{int(start_time)}_{int(end_time)}.wav"
        elif storage_type == 's3':
            return f"chunks/{self.content_id}/chunk_{chunk_index}_{int(start_time)}_{int(end_time)}.wav"
        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")

    def get_chunk_transcript_path(self, chunk_index: int, storage_type: str = 'local') -> str:
        """Get path for a chunk's transcript file"""
        if storage_type == 'local':
            return f"{self.partial_transcripts_path}/{self.content_id}_{chunk_index}.json"
        elif storage_type == 's3':
            return f"transcripts/{self.content_id}/chunk_{chunk_index}.json"
        else:
            raise ValueError(f"Unsupported storage type: {storage_type}")

    def update_chunk_progress(self, session):
        """Update chunk progress based on chunk statuses"""
        # Get all chunks for this content
        chunks = session.query(ContentChunk).filter_by(content_id=self.id).all()

        if not chunks:
            return

        # Count completed chunks
        completed_chunks = sum(1 for chunk in chunks if chunk.extraction_status == 'completed')

        # Update content status
        self.total_chunks = len(chunks)
        self.chunks_processed = completed_chunks
        self.is_converted = (completed_chunks == len(chunks) and len(chunks) > 0)

        # Update chunks_status
        self.chunks_status = {
            str(chunk.chunk_index): {
                'extraction_status': chunk.extraction_status,
                'transcription_status': chunk.transcription_status
            }
            for chunk in chunks
        }

        self.last_updated = datetime.now(timezone.utc)


class Transcription(Base):
    """
    Legacy model for storing raw transcription output.

    NOTE: This model is being phased out in favor of SpeakerTranscription,
    which provides speaker-attributed segments. New code should use
    SpeakerTranscription instead.

    Attributes:
        id: Primary key
        content_id: Foreign key to Content
        full_text: Complete transcription text
        segments: JSON list of segments with timing:
                 [{"text": "...", "start": 0.0, "end": 5.2}, ...]
        model_version: Whisper model version used
        processing_status: Status ('pending', 'processed', 'failed')
        created_at: When the transcription was created
        updated_at: Last update timestamp

    Relationships:
        content: The Content this transcription belongs to
    """
    __tablename__ = 'transcription'

    id = Column(Integer, primary_key=True)
    content_id = Column(Integer, ForeignKey('content.id'), nullable=False)
    full_text = Column(Text, nullable=False)
    segments = Column(JSON, nullable=False)  # List of {text, start, end} dicts
    model_version = Column(String)
    processing_status = Column(String, default='processed')  # pending, processed, failed
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    content = relationship("Content", back_populates="transcriptions")

    # Indexes
    __table_args__ = (
        Index('idx_transcription_content', 'content_id'),
        Index('idx_transcription_status', 'processing_status'),
    )


class Source(Base):
    """
    Represents a content source (YouTube channel, podcast feed, etc.).

    Sources are the starting point for content discovery. The system
    periodically checks sources for new content to process.

    Attributes:
        id: Primary key
        name: Human-readable source name
        type: Source type ('youtube', 'podcast', etc.)
        url: Source URL (channel URL, RSS feed, etc.)
        projects: Comma-separated list of projects using this source
        description: Optional description
        meta_data: Additional source metadata (e.g., API keys)
        created_at: When the source was added
        updated_at: Last update timestamp
    """
    __tablename__ = 'source'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    type = Column(String, nullable=False)  # youtube, podcast
    url = Column(String, nullable=False)
    projects = Column(ARRAY(String), nullable=False)  # Array of project names
    description = Column(Text)
    meta_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Indexes
    __table_args__ = (
        Index('idx_source_projects', 'projects'),
        Index('idx_source_type', 'type'),
    )


class ContentChunk(Base):
    """
    Represents an audio chunk for parallel processing.

    Long audio files are split into overlapping chunks during the convert
    step to enable parallel transcription. This model tracks each chunk's
    processing status independently.

    Attributes:
        id: Primary key
        content_id: Foreign key to Content
        chunk_index: 0-based index of this chunk
        start_time: Start time in the original audio (seconds)
        end_time: End time in the original audio (seconds)
        duration: Chunk duration (seconds)

    Extraction Status (convert step):
        extraction_status: Status of audio extraction
        extraction_worker_id: Worker handling extraction
        extraction_attempts: Number of extraction attempts
        extraction_completed_at: When extraction finished
        extraction_error: Error message if failed

    Transcription Status (transcribe step):
        transcription_status: Status of transcription
        transcription_worker_id: Worker handling transcription
        transcription_attempts: Number of transcription attempts
        transcription_completed_at: When transcription finished
        transcription_error: Error message if failed

    Timestamps:
        created_at: When the chunk record was created
        last_updated: Last status update

    Relationships:
        content: The Content this chunk belongs to

    S3 Paths:
        Audio: content/{content_id}/chunks/{index}/audio.wav
        Transcript: content/{content_id}/chunks/{index}/transcript.json
    """
    __tablename__ = 'content_chunks'

    id = Column(Integer, primary_key=True)
    content_id = Column(Integer, ForeignKey('content.id'), nullable=False)
    chunk_index = Column(Integer, nullable=False)  # 0-based index of the chunk
    start_time = Column(Float, nullable=False)  # Start time in seconds
    end_time = Column(Float, nullable=False)  # End time in seconds
    duration = Column(Float, nullable=False)  # Duration in seconds

    # Extraction status fields
    extraction_status = Column(String, nullable=False, default='pending')  # pending, processing, completed, failed
    extraction_worker_id = Column(String)  # ID of worker processing this chunk
    extraction_attempts = Column(Integer, default=0)  # Number of extraction attempts
    extraction_completed_at = Column(DateTime(timezone=True))  # When extraction completed
    extraction_error = Column(Text)  # Error message if extraction failed

    # Transcription status fields
    transcription_status = Column(String)  # pending, processing, completed, failed
    transcription_worker_id = Column(String)  # ID of worker processing this chunk
    transcription_attempts = Column(Integer, default=0)  # Number of transcription attempts
    transcription_completed_at = Column(DateTime(timezone=True))  # When transcription completed
    transcription_error = Column(Text)  # Error message if transcription failed
    transcribed_with = Column(String)  # Model/method used for transcription (e.g., "whisper-large-v3", "whisper-medium")

    created_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))
    last_updated = Column(DateTime(timezone=True), onupdate=datetime.now(timezone.utc))  # Track last update

    # Relationships
    content = relationship("Content", back_populates="chunks")

    # Indexes
    __table_args__ = (
        Index('idx_content_chunks_content', 'content_id'),
        Index('idx_content_chunks_index', 'chunk_index'),
        Index('idx_content_chunks_extraction_status', 'extraction_status'),
        Index('idx_content_chunks_transcription_status', 'transcription_status'),
        Index('idx_content_chunks_last_updated', 'last_updated'),
    )

    @property
    def is_extraction_complete(self) -> bool:
        """Check if extraction is complete"""
        return self.extraction_status == 'completed'

    @property
    def is_transcription_complete(self) -> bool:
        """Check if transcription is complete"""
        return self.transcription_status == 'completed'

    @property
    def can_retry_extraction(self) -> bool:
        """Check if extraction can be retried if failed"""
        return self.extraction_status == 'failed' and self.extraction_attempts < 3

    @property
    def can_retry_transcription(self) -> bool:
        """Check if transcription can be retried if failed"""
        return self.transcription_status == 'failed' and self.transcription_attempts < 3

    def mark_extraction_processing(self, worker_id: str):
        """Mark chunk as being extracted"""
        self.extraction_status = 'processing'
        self.extraction_worker_id = worker_id
        self.extraction_attempts += 1
        self.last_updated = datetime.now(timezone.utc)

    def mark_extraction_completed(self):
        """Mark chunk extraction as completed"""
        self.extraction_status = 'completed'
        self.extraction_completed_at = datetime.now(timezone.utc)
        self.extraction_worker_id = None
        self.last_updated = datetime.now(timezone.utc)
        # Initialize transcription status when extraction completes
        if self.transcription_status is None:
            self.transcription_status = 'pending'

    def mark_extraction_failed(self, error: str):
        """Mark chunk extraction as failed"""
        self.extraction_status = 'failed'
        self.extraction_error = error
        self.extraction_worker_id = None
        self.last_updated = datetime.now(timezone.utc)

    def mark_transcription_processing(self, worker_id: str):
        """Mark chunk as being transcribed"""
        self.transcription_status = 'processing'
        self.transcription_worker_id = worker_id
        self.transcription_attempts += 1
        self.last_updated = datetime.now(timezone.utc)

    def mark_transcription_completed(self):
        """Mark chunk transcription as completed"""
        self.transcription_status = 'completed'
        self.transcription_completed_at = datetime.now(timezone.utc)
        self.transcription_worker_id = None
        self.last_updated = datetime.now(timezone.utc)

    def mark_transcription_failed(self, error: str):
        """Mark chunk transcription as failed"""
        self.transcription_status = 'failed'
        self.transcription_error = error
        self.transcription_worker_id = None
        self.last_updated = datetime.now(timezone.utc)

    def get_s3_paths(self) -> Dict[str, str]:
        """Get S3 paths for this chunk"""
        return self.content.get_chunk_s3_paths(self.chunk_index)

    @classmethod
    def validate_chunk_sequence(cls, session, content_id: int) -> Dict:
        """Validate the chunk sequence for a piece of content"""
        chunks = session.query(cls).filter_by(
            content_id=content_id
        ).order_by(cls.chunk_index).all()

        result = {
            'is_valid': False,
            'has_chunks': bool(chunks),
            'total_chunks': len(chunks),
            'missing_indices': [],
            'timing_gaps': [],
            'failed_extraction': [],
            'failed_transcription': [],
            'processing_extraction': [],
            'processing_transcription': [],
            'completed_extraction': [],
            'completed_transcription': [],
            'duration': 0.0 if not chunks else chunks[-1].end_time
        }

        if not chunks:
            return result

        # Verify sequence
        expected_indices = set(range(len(chunks)))
        actual_indices = set(c.chunk_index for c in chunks)
        result['missing_indices'] = sorted(expected_indices - actual_indices)

        # Check chunk status
        for chunk in chunks:
            # Check extraction status
            if chunk.extraction_status == 'failed':
                result['failed_extraction'].append(chunk.chunk_index)
            elif chunk.extraction_status == 'processing':
                result['processing_extraction'].append(chunk.chunk_index)
            elif chunk.extraction_status == 'completed':
                result['completed_extraction'].append(chunk.chunk_index)

            # Check transcription status
            if chunk.transcription_status == 'failed':
                result['failed_transcription'].append(chunk.chunk_index)
            elif chunk.transcription_status == 'processing':
                result['processing_transcription'].append(chunk.chunk_index)
            elif chunk.transcription_status == 'completed':
                result['completed_transcription'].append(chunk.chunk_index)

        # Verify timing
        for i in range(len(chunks)-1):
            if abs(chunks[i].end_time - chunks[i+1].start_time) > 0.1:  # Allow 100ms tolerance
                result['timing_gaps'].append({
                    'chunk_index': chunks[i].chunk_index,
                    'gap': chunks[i+1].start_time - chunks[i].end_time
                })

        # Result is valid if:
        # 1. No missing indices
        # 2. No timing gaps
        # 3. All chunks have valid extraction and transcription status
        result['is_valid'] = (
            not result['missing_indices'] and
            not result['timing_gaps'] and
            all(c.is_extraction_complete or c.can_retry_extraction for c in chunks) and
            all(c.is_transcription_complete or c.can_retry_transcription or c.transcription_status == 'pending' for c in chunks)
        )

        return result
