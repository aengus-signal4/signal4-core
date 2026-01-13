"""
Database Models for Audio-Visual Content Processing Pipeline
===========================================================

This module defines the SQLAlchemy ORM models for the distributed content processing system.
The system downloads, transcribes, diarizes, and analyzes audio-visual content from various
platforms (YouTube, podcasts, etc.) using a distributed worker architecture.

## Core Processing Flow:

1. **Content Discovery & Indexing**
   - Sources are added to the system (YouTube channels, podcast feeds)
   - Content is discovered and indexed from these sources
   - Tasks are created for processing

2. **Download & Conversion** (processing_steps/)
   - download_youtube.py / download_podcast.py: Downloads source media
   - convert.py: Converts to WAV, performs VAD, creates chunks

3. **Transcription** (processing_steps/)
   - transcribe.py: Transcribes individual chunks using Whisper

4. **Speaker Processing** (processing_steps/)
   - diarize.py: Identifies speaker segments and creates speaker turns
   - stitch.py: Combines chunks into final speaker-attributed transcript with LLM post-processing

5. **Embedding & Retrieval** (processing_steps/)
   - segment_embeddings.py: Creates retrieval-optimized segments with embeddings

## Model Relationships:

### Content Models:
- **Content**: Central model representing a piece of media (video/podcast)
  - Has many: ContentChunks, SpeakerTranscriptions, EmbeddingSegments
  - Tracks processing state through boolean flags (is_downloaded, is_transcribed, etc.)
  
- **ContentChunk**: Represents audio chunks created during conversion
  - Belongs to: Content
  - Tracks extraction and transcription status per chunk

- **Source**: Represents content sources (YouTube channels, podcast feeds)
  - Used for content discovery and indexing

### Transcription Models:
- **Transcription**: Stores raw transcription output (legacy, being phased out)
  - Belongs to: Content
  
- **SpeakerTranscription**: Speaker-attributed transcript segments
  - Belongs to: Content, Speaker
  - Created by diarize.py, represents speaker turns
  
- **EmbeddingSegment**: Retrieval-optimized segments with embeddings
  - Belongs to: Content
  - References: SpeakerTranscription records via source_transcription_ids
  - Created by segment_embeddings.py for optimal retrieval

### Speaker Models:
- **Speaker**: Unified speaker table with embeddings
  - Each record represents a unique content-speaker observation
  - Enforces 1:1 relationship between content_id + local_speaker_id
  - Contains both identity and embedding data
  - Supports canonical speaker relationships for clustering

### Task Management:
- **TaskQueue**: Distributed task queue for processing
  - Tracks task status, worker assignment, results
  - Uses unique constraint on (content_id, task_type)
  
- **WorkerConfig**: Worker node configuration
  - Defines which tasks each worker can handle


## Processing State Tracking:

Content processing state is tracked through boolean flags:
- is_downloaded: Source media downloaded
- is_converted: Audio extracted and chunked
- is_transcribed: All chunks transcribed
- is_diarized: Speaker diarization complete
- is_identified: Speakers mapped to global IDs (deprecated)
- is_stitched: Final transcript created (deprecated)
- is_embedded: Embedding segments created

## Storage:

All media files are stored in S3/MinIO with standardized paths:
- content/{content_id}/source.{ext}: Original media
- content/{content_id}/audio.wav: Converted audio
- content/{content_id}/chunks/{index}/: Chunk files
- content/{content_id}/speaker_turns.json: Final transcript
"""

from sqlalchemy import Column, Integer, SmallInteger, String, Float, DateTime, JSON, ForeignKey, Text, Boolean, Index, Date, ARRAY, text, UniqueConstraint, func, LargeBinary, Enum, PrimaryKeyConstraint
from sqlalchemy import text as sa_text  # Alias to avoid shadowing by column names
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from pgvector.sqlalchemy import Vector  # Import the actual Vector type
import numpy as np
import uuid
import enum
import hashlib

Base = declarative_base()

class SpeakerProcessingStatus(enum.Enum):
    """Tracks speaker processing through the 3-phase pipeline"""
    PENDING = "PENDING"           # New from stitch, needs clustering (Phase 1)
    CLUSTERED = "CLUSTERED"       # Phase 1 complete - has speaker_identity_id
    IDENTIFIED = "IDENTIFIED"     # Phase 2 complete - identity has primary_name
    VALIDATED = "VALIDATED"       # Phase 3 complete - checked for merges (optional)


class IdentificationStatus:
    """
    Speaker identification pipeline status tracking.

    Used to prevent re-processing of speakers that have already been evaluated.
    Enables efficient filtering and future Phase 6 retry logic.
    """
    # Initial state
    UNPROCESSED = "unprocessed"                    # Never evaluated (DEFAULT)

    # Success state
    ASSIGNED = "assigned"                          # Linked to speaker_identity

    # Rejection states (detailed for Phase 6 retry logic)
    REJECTED_LOW_SIMILARITY = "rejected_low_similarity"      # Embedding match < 0.40
    REJECTED_NO_CONTEXT = "rejected_no_context"              # No transcript for LLM
    REJECTED_LLM_UNVERIFIED = "rejected_llm_unverified"     # LLM could not verify
    REJECTED_UNKNOWN = "rejected_unknown"                    # LLM returned "unknown"
    REJECTED_SHORT_DURATION = "rejected_short_duration"     # Below duration threshold
    REJECTED_POOR_EMBEDDING = "rejected_poor_embedding"     # Embedding quality too low

    # Review states (future Phase 6)
    PENDING_REVIEW = "pending_review"              # Flagged for human review
    RETRY_ELIGIBLE = "retry_eligible"              # "probably" confidence, worth retry

    @classmethod
    def all_rejected(cls) -> list:
        """Return all rejection status values."""
        return [
            cls.REJECTED_LOW_SIMILARITY,
            cls.REJECTED_NO_CONTEXT,
            cls.REJECTED_LLM_UNVERIFIED,
            cls.REJECTED_UNKNOWN,
            cls.REJECTED_SHORT_DURATION,
            cls.REJECTED_POOR_EMBEDDING,
        ]

    @classmethod
    def all_values(cls) -> list:
        """Return all possible status values."""
        return [
            cls.UNPROCESSED,
            cls.ASSIGNED,
            cls.REJECTED_LOW_SIMILARITY,
            cls.REJECTED_NO_CONTEXT,
            cls.REJECTED_LLM_UNVERIFIED,
            cls.REJECTED_UNKNOWN,
            cls.REJECTED_SHORT_DURATION,
            cls.REJECTED_POOR_EMBEDDING,
            cls.PENDING_REVIEW,
            cls.RETRY_ELIGIBLE,
        ]

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

class TaskQueue(Base):
    """
    Distributed task queue for coordinating processing across workers.
    
    Tasks represent units of work (download, transcribe, etc.) that are
    claimed and executed by worker nodes. The queue ensures each task
    is processed exactly once and tracks execution status.
    
    Attributes:
        id: Primary key
        task_type: Type of task ('download', 'transcribe', 'diarize', etc.)
        content_id: Content ID this task operates on
        status: Task status ('pending', 'processing', 'completed', 'failed')
        worker_id: ID of worker currently processing this task
        processor_task_id: Task ID from the processor API
        input_data: JSON input parameters for the task
        result: JSON result data from task execution
        error: Error message if task failed
        priority: Task priority (higher = more urgent)
        created_at: When the task was created
        started_at: When processing started
        completed_at: When processing completed
        last_heartbeat: Last heartbeat from processing worker
        attempts: Number of execution attempts
        max_attempts: Maximum attempts before giving up
        
    Unique Constraint:
        (content_id, task_type) - Only one task per type per content
        
    Task Lifecycle:
        1. Created as 'pending' with input_data
        2. Worker claims task, status -> 'processing'
        3. Worker sends heartbeats during execution
        4. On success: status -> 'completed', result populated
        5. On failure: status -> 'failed', error populated
    """
    __tablename__ = 'task_queue'
    
    id = Column(Integer, primary_key=True)
    task_type = Column(String(50), nullable=False)
    content_id = Column(String(255), nullable=False)
    status = Column(String(20), default='pending', nullable=False)
    worker_id = Column(String(100))
    processor_task_id = Column(String(255), nullable=True)
    input_data = Column(JSON, nullable=False, default=dict)  # Ensure never NULL
    result = Column(JSON)
    error = Column(Text)
    priority = Column(Integer, default=0)  # Added priority field
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    last_heartbeat = Column(DateTime(timezone=True))
    attempts = Column(Integer, default=0)
    max_attempts = Column(Integer, default=3)
    
    # Indexes
    __table_args__ = (
        Index('idx_task_queue_status', 'status'),
        Index('idx_task_queue_content_id', 'content_id'),
        Index('idx_task_queue_task_type', 'task_type'),
        # Composite index for task claiming with priority ordering
        Index('idx_task_queue_claim',
              'status', 'task_type', 'priority', 'created_at',
              postgresql_where=text("status = 'pending'")),
        # Add unique constraint for content_id and task_type
        UniqueConstraint('content_id', 'task_type', name='uq_task_queue_content_id_task_type'), # Ensure unique constraint name
        {'schema': 'tasks'}  # Use tasks schema
    )

class WorkerConfig(Base):
    """
    Configuration for worker nodes in the distributed system.
    
    Each worker machine registers its capabilities and the orchestrator
    uses this information to assign appropriate tasks.
    
    Attributes:
        id: Primary key  
        hostname: Unique hostname of the worker
        enabled_tasks: List of task types this worker can handle
        max_concurrent_tasks: Maximum parallel tasks
        last_heartbeat: Last time worker checked in
        status: Worker status ('active', 'inactive', 'disabled')
        
    Note: This table uses the 'tasks' schema, not the default schema.
    """
    __tablename__ = 'worker_config'
    __table_args__ = (
        {'schema': 'tasks'}  # Use tasks schema
    )
    
    id = Column(Integer, primary_key=True)
    hostname = Column(String(255), nullable=False, unique=True)
    enabled_tasks = Column(ARRAY(String), nullable=False)
    max_concurrent_tasks = Column(Integer, default=1)
    last_heartbeat = Column(DateTime(timezone=True))
    status = Column(String(20), default='active')
    
    # Indexes
    __table_args__ = (
        Index('idx_worker_config_hostname', 'hostname'),
        Index('idx_worker_config_status', 'status'),
        {'schema': 'tasks'}  # Use tasks schema
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

class ClusteringRun(Base):
    """Audit trail of clustering operations.
    
    Tracks each clustering operation including parameters, statistics,
    and results for debugging and performance monitoring.
    """
    __tablename__ = 'clustering_runs'
    
    id = Column(Integer, primary_key=True)
    run_id = Column(String(64), unique=True, nullable=False)
    
    # Run metadata
    run_type = Column(String(50), nullable=False)
    method = Column(String(50), nullable=False)
    parameters = Column(JSONB, nullable=False)
    
    # Statistics
    embeddings_processed = Column(Integer, default=0)
    clusters_created = Column(Integer, default=0)
    assignments_made = Column(Integer, default=0)
    identities_created = Column(Integer, default=0)
    identities_merged = Column(Integer, default=0)
    
    # Timing
    started_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime)
    processing_time_seconds = Column(Float)
    
    # Status
    status = Column(String(20), default='running')
    error_message = Column(Text)
    
    __table_args__ = (
        Index('idx_clustering_runs_run_id', 'run_id'),
        Index('idx_clustering_runs_status', 'status', 'started_at'),
    )

class IdentityMergeHistory(Base):
    """History of speaker identity merges and splits.
    
    Maintains a complete audit trail of all identity merge and split
    operations for traceability and potential rollback.
    """
    __tablename__ = 'identity_merge_history'
    
    id = Column(Integer, primary_key=True)
    
    # Operation type
    operation = Column(String(20), nullable=False)  # 'merge' or 'split'
    
    # For merges: multiple sources -> one target
    # For splits: one source -> multiple targets
    source_identity_ids = Column(ARRAY(Integer), nullable=False)
    target_identity_ids = Column(ARRAY(Integer), nullable=False)
    
    # Operation details
    confidence = Column(Float)
    reason = Column(Text)
    evidence = Column(JSONB, default=dict)
    
    # Who/what performed it
    performed_by = Column(String(100))
    clustering_run_id = Column(String(64))
    
    # When
    performed_at = Column(DateTime, default=datetime.utcnow)

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
    transcriptions = relationship("SpeakerTranscription", back_populates="speaker")
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
    Speaker-attributed transcript segments (speaker turns).
    
    This is the primary model for storing transcribed content with speaker
    attribution. Created by the diarize step, these records represent
    continuous speech segments by a single speaker.
    
    Attributes:
        id: Primary key
        content_id: Foreign key to Content
        speaker_id: Foreign key to global Speaker
        start_time: Start time in seconds
        end_time: End time in seconds
        text: Transcribed text for this turn
        turn_index: Sequential index within the content
        stitch_version: Version of algorithm that created this
        created_at: When this record was created
        updated_at: Last update timestamp
        
    Relationships:
        content: The Content this belongs to
        speaker: The global Speaker record
        
    Usage Patterns:
        - Query by content_id to get full transcript
        - Query by speaker_id to get all utterances by a speaker
        - Use turn_index for ordered playback
        - Use start_time/end_time for temporal queries
        
    Context Methods:
        - get_context_turns(): Get surrounding turns
        - get_context_window(): Get turns by index range
        - get_time_window(): Get turns by time range
        - search_with_context(): Search text with context
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

    # Relationships
    content = relationship("Content", back_populates="speaker_transcriptions")
    speaker = relationship("Speaker", back_populates="transcriptions")

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
        has_embedding = "Yes" if self.embedding is not None else "No"
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


class ClusteringBatch(Base):
    """
    Tracks speaker clustering batch operations for monitoring and recovery.
    """
    __tablename__ = 'clustering_batches'

    id = Column(Integer, primary_key=True)
    batch_id = Column(String(50), unique=True, nullable=False)
    status = Column(String(20), default='pending', nullable=False)
    speaker_count = Column(Integer, nullable=False)
    merge_candidates_found = Column(Integer, default=0)
    merges_applied = Column(Integer, default=0)
    processing_time = Column(Float, nullable=True)
    config_params = Column(JSONB, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index('idx_clustering_batches_batch_id', 'batch_id'),
        Index('idx_clustering_batches_status', 'status'),
        Index('idx_clustering_batches_created_at', 'created_at'),
    )


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


class CprmvAnalysis(Base):
    """
    CPRMV theme classification results with embeddings for fast semantic search.

    This table stores theme classification results from the CPRMV project,
    enabling fast filtering by themes and direct semantic search within
    theme-filtered segments without requiring joins.

    Attributes:
        id: Primary key
        segment_id: Foreign key to embedding_segments
        themes: Array of all theme IDs (e.g., ['2B', '2C', '9A'])
        confidence_scores: JSONB with theme->confidence mapping (e.g., {"2B": 1.0, "2C": 0.75})
        high_confidence_themes: Array of themes with confidence >= 0.75 for fast filtering
        matched_via: How the segment was matched ('semantic', 'keyword', 'both')
        similarity_score: FAISS similarity score if matched via semantic search
        matched_keywords: Keywords that triggered the match (if applicable)
        embedding: 2000-dim embedding copied from embedding_segments.embedding_alt for fast search
        created_at: When this classification was added
        updated_at: Last update timestamp

    Indexes:
        - GIN on themes for fast array containment queries
        - GIN on high_confidence_themes for filtering >= 0.75 confidence
        - IVFFlat on embedding for fast semantic search within filtered results
        - Standard index on segment_id for joins

    Usage:
        # Find high-confidence anti-trans education segments
        SELECT * FROM cprmv_analysis
        WHERE high_confidence_themes @> ARRAY['2B']
        ORDER BY (confidence_scores->>'2B')::float DESC;

        # Semantic search within theme-filtered results
        SELECT * FROM cprmv_analysis
        WHERE high_confidence_themes @> ARRAY['2B', '2C']
        ORDER BY embedding <-> '[query_vector]'
        LIMIT 20;
    """
    __tablename__ = 'cprmv_analysis'

    id = Column(Integer, primary_key=True)
    segment_id = Column(Integer, ForeignKey('embedding_segments.id'), nullable=False, unique=True, index=True)

    # Theme classifications
    themes = Column(ARRAY(String), nullable=False)
    confidence_scores = Column(JSONB, nullable=False)
    high_confidence_themes = Column(ARRAY(String), nullable=False)

    # Match metadata
    matched_via = Column(String(20), nullable=True)  # 'semantic', 'keyword', 'both'
    similarity_score = Column(Float, nullable=True)
    matched_keywords = Column(Text, nullable=True)

    # Embedding for direct semantic search (copied from embedding_segments.embedding_alt)
    embedding = Column(Vector(2000), nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationship
    segment = relationship("EmbeddingSegment", foreign_keys=[segment_id])

    __table_args__ = (
        # GIN indexes for fast array containment queries
        Index('idx_cprmv_analysis_themes', 'themes', postgresql_using='gin'),
        Index('idx_cprmv_analysis_high_conf_themes', 'high_confidence_themes', postgresql_using='gin'),

        # IVFFlat index for fast semantic search (lists=100 for ~80k segments)
        Index('idx_cprmv_analysis_embedding', 'embedding',
              postgresql_using='ivfflat',
              postgresql_with={'lists': 100},
              postgresql_ops={'embedding': 'vector_cosine_ops'}),

        # Standard indexes
        Index('idx_cprmv_analysis_segment_id', 'segment_id'),
        Index('idx_cprmv_analysis_matched_via', 'matched_via'),
    )


class ClassificationSchema(Base):
    """
    Classification schemas for theme/subtheme definitions.

    Stores versioned theme classification schemas loaded from CSV files.
    Enables multiple classification runs with different schema versions.

    Attributes:
        id: Primary key
        name: Schema name (e.g., 'CPRMV', 'LGBTQ_Education')
        version: Schema version string (e.g., 'v1.0', '2025-01-15')
        description: Optional description of schema
        themes_json: JSONB with theme definitions {theme_id: {name, description_en, description_fr}}
        subthemes_json: JSONB with subtheme definitions {subtheme_id: {theme_id, name, description_en, description_fr}}
        created_at: When schema was loaded

    Indexes:
        - Unique constraint on (name, version)
        - Index on name for lookups
    """
    __tablename__ = 'classification_schemas'

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, index=True)
    version = Column(String(50), nullable=False)
    description = Column(Text, nullable=True)

    # Schema definitions
    themes_json = Column(JSONB, nullable=False)
    subthemes_json = Column(JSONB, nullable=False)

    # Cached query embeddings for semantic search
    # Format: {subtheme_id: {lang: [embedding_vector]}}
    # Example: {"Q1": {"en": [0.1, 0.2, ...], "fr": [0.3, 0.4, ...]}}
    query_embeddings = Column(JSONB, nullable=True)

    # Metadata
    source_file = Column(String(500), nullable=True)  # Path to original CSV
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    classifications = relationship("ThemeClassification", back_populates="schema")

    __table_args__ = (
        UniqueConstraint('name', 'version', name='uq_schema_name_version'),
        Index('idx_schema_name', 'name'),
    )


class ThemeClassification(Base):
    """
    Detailed theme classification results with multi-stage tracking.

    Stores comprehensive classification results from semantic theme classifiers,
    tracking similarity scores, LLM classifications, and validation results
    across all pipeline stages.

    Attributes:
        id: Primary key
        segment_id: Foreign key to embedding_segments
        schema_id: Foreign key to classification_schemas

        # Final results (for fast queries)
        theme_ids: Array of theme IDs (integers or strings depending on schema)
        subtheme_ids: Array of subtheme IDs
        high_confidence_themes: Array of themes with confidence >= 0.75

        # Stage-by-stage results
        stage1_similarities: JSONB with {theme_id: score, subtheme_id: score}
        stage3_results: JSONB with LLM subtheme classifications per theme
        stage4_validations: JSONB with Likert scale validation scores

        # Aggregate confidence
        final_confidence_scores: JSONB with {theme_id: confidence, subtheme_id: confidence}

        # Match metadata
        matched_via: 'semantic', 'keyword', 'both'
        max_similarity_score: Highest similarity score from stage 1

        # Embedding for semantic search
        embedding: 2000-dim embedding copied from embedding_segments

        # Timestamps
        created_at: When classification was created
        updated_at: Last update

    Indexes:
        - GIN on theme_ids, subtheme_ids, high_confidence_themes
        - IVFFlat on embedding for semantic search
        - Unique constraint on (segment_id, schema_id)
    """
    __tablename__ = 'theme_classifications'

    id = Column(Integer, primary_key=True)
    segment_id = Column(Integer, ForeignKey('embedding_segments.id'), nullable=False, index=True)
    schema_id = Column(Integer, ForeignKey('classification_schemas.id'), nullable=False, index=True)

    # Final classification results
    theme_ids = Column(ARRAY(String), nullable=False, server_default='{}')
    subtheme_ids = Column(ARRAY(String), nullable=False, server_default='{}')
    high_confidence_themes = Column(ARRAY(String), nullable=False, server_default='{}')

    # Stage-by-stage results (JSONB for flexibility)
    stage1_similarities = Column(JSONB, nullable=True)  # {theme_id: score, subtheme_id: score}
    stage3_results = Column(JSONB, nullable=True)  # {theme_id: {subtheme_ids, confidence, reasoning}}
    stage4_validations = Column(JSONB, nullable=True)  # {theme_id: {subtheme_id: {confidence, category}}}

    # Final aggregated confidence scores
    final_confidence_scores = Column(JSONB, nullable=False, server_default='{}')  # {theme_id: conf, subtheme_id: conf}

    # Match metadata
    matched_via = Column(String(20), nullable=True)
    max_similarity_score = Column(Float, nullable=True)

    # Embedding for semantic search (copied from embedding_segments.embedding_alt)
    embedding = Column(Vector(2000), nullable=True)

    # Stage 5: Final relevance check (automated LLM)
    # {is_relevant: bool, reasoning: str, relevance: str, model: str, checked_at: str}
    stage5_final_check = Column(JSONB, nullable=True)

    # Stage 6: LLM false positive detection
    # Specialized LLM check to identify content that Stage 5 incorrectly marked as relevant:
    # - pro_progressive: Defends feminist/LGBTQ+ positions (not attacking them)
    # - documenting_harm: Reports/documents prejudice without promoting it
    # - quote_without_endorsement: Quotes someone else's position without endorsing it
    # {
    #   is_false_positive: bool,
    #   false_positive_type: str|null ('pro_progressive'|'documenting_harm'|'quote_without_endorsement'),
    #   reasoning: str,
    #   confidence: str ('definitely'|'probably'|'possibly'|'probably_not'|'definitely_not'),
    #   model: str,
    #   checked_at: str
    # }
    stage6_false_positive_check = Column(JSONB, nullable=True)

    # Stage 7: Expanded context re-check for Stage 6 false positives
    # Re-evaluates segments with ±20 second context window
    # {
    #   is_false_positive: bool,
    #   speaker_stance: str ('strongly_holds'|'holds'|'leans_holds'|'neutral'|'leans_rejects'|'rejects'|'strongly_rejects'),
    #   reasoning: str,
    #   original_stance: str (from stage6),
    #   context_window_seconds: int,
    #   checked_at: str
    # }
    stage7_expanded_context = Column(JSONB, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    segment = relationship("EmbeddingSegment", foreign_keys=[segment_id])
    schema = relationship("ClassificationSchema", back_populates="classifications")

    __table_args__ = (
        # Unique constraint for idempotency
        UniqueConstraint('segment_id', 'schema_id', name='uq_segment_schema'),

        # GIN indexes for fast array containment queries
        Index('idx_theme_class_theme_ids', 'theme_ids', postgresql_using='gin'),
        Index('idx_theme_class_subtheme_ids', 'subtheme_ids', postgresql_using='gin'),
        Index('idx_theme_class_high_conf', 'high_confidence_themes', postgresql_using='gin'),

        # IVFFlat index for semantic search
        Index('idx_theme_class_embedding', 'embedding',
              postgresql_using='ivfflat',
              postgresql_with={'lists': 100},
              postgresql_ops={'embedding': 'vector_cosine_ops'}),

        # Standard indexes
        Index('idx_theme_class_segment_id', 'segment_id'),
        Index('idx_theme_class_schema_id', 'schema_id'),
        Index('idx_theme_class_matched_via', 'matched_via'),
    )