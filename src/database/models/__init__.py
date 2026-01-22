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
   - stitch_steps/stage14_segment.py: Creates retrieval-optimized segments with embeddings

## Model Relationships:

### Content Models:
- **Content**: Central model representing a piece of media (video/podcast)
  - Has many: ContentChunks, Sentences, EmbeddingSegments
  - Tracks processing state through boolean flags (is_downloaded, is_transcribed, etc.)

- **ContentChunk**: Represents audio chunks created during conversion
  - Belongs to: Content
  - Tracks extraction and transcription status per chunk

- **Source**: Represents content sources (YouTube channels, podcast feeds)
  - Used for content discovery and indexing

### Transcription Models:
- **Transcription**: Stores raw transcription output (legacy, being phased out)
  - Belongs to: Content

- **Sentence**: Atomic unit for transcript data (PRIMARY)
  - Belongs to: Content, Speaker
  - Created by stitch pipeline stage 12
  - Speaker turns can be reconstructed by grouping on (content_id, turn_index)

- **EmbeddingSegment**: Retrieval-optimized segments with embeddings
  - Belongs to: Content
  - References: Sentence records via source_sentence_ids
  - Created by stitch_steps/stage14_segment.py for optimal retrieval

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

# Base and utilities
from src.database.models.base import Base, SpeakerProcessingStatus, IdentificationStatus

# Content models
from src.database.models.content import Content, ContentChunk, Transcription, Source

# Speaker models
from src.database.models.speakers import (
    Speaker,
    SpeakerIdentity,
    SpeakerAssignment,
    Sentence,
)

# Embedding models
from src.database.models.embeddings import EmbeddingSegment

# Task management models
from src.database.models.tasks import TaskQueue, WorkerConfig

# Channel models
from src.database.models.channels import (
    Channel,
    ChannelSource,
    ChannelProject,
    PodcastMetadata,
    PodcastChart,
)

# Classification models
from src.database.models.classification import (
    ClassificationSchema,
    ThemeClassification,
    CprmvAnalysis,
)

# Clustering models
from src.database.models.clustering import (
    ClusteringRun,
    ClusteringBatch,
    IdentityMergeHistory,
)

# API key models
from src.database.models.api_keys import ApiKey, ApiKeyUsage

# Alternative transcription models
from src.database.models.transcriptions import AlternativeTranscription

# Query cache models
from src.database.models.query_cache import QueryVariation, QueryExpansion

# Bookmark models
from src.database.models.bookmarks import Bookmark


__all__ = [
    # Base
    "Base",
    "SpeakerProcessingStatus",
    "IdentificationStatus",
    # Content
    "Content",
    "ContentChunk",
    "Transcription",
    "Source",
    # Speakers
    "Speaker",
    "SpeakerIdentity",
    "SpeakerAssignment",
    "Sentence",
    # Embeddings
    "EmbeddingSegment",
    # Tasks
    "TaskQueue",
    "WorkerConfig",
    # Channels
    "Channel",
    "ChannelSource",
    "ChannelProject",
    "PodcastMetadata",
    "PodcastChart",
    # Classification
    "ClassificationSchema",
    "ThemeClassification",
    "CprmvAnalysis",
    # Clustering
    "ClusteringRun",
    "ClusteringBatch",
    "IdentityMergeHistory",
    # API Keys
    "ApiKey",
    "ApiKeyUsage",
    # Transcriptions
    "AlternativeTranscription",
    # Query Cache
    "QueryVariation",
    "QueryExpansion",
    # Bookmarks
    "Bookmark",
]
