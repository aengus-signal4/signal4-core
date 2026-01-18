"""
Content file existence checker for S3 storage.

Provides a single source of truth for determining what files exist
for a given content item, handling both original and compressed formats.
"""

from dataclasses import dataclass, field
from typing import Set, Optional, Dict, Any
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class ContentFiles:
    """Result of checking files for a content item."""
    content_id: str
    prefix: str

    # Core files
    audio_exists: bool = False
    source_exists: bool = False
    storage_manifest_exists: bool = False

    # Processing outputs
    diarization_exists: bool = False
    speaker_embeddings_exist: bool = False
    speaker_mapping_exists: bool = False
    stitched_exists: bool = False
    semantic_segments_exist: bool = False

    # Chunk information
    chunk_indices_with_audio: Set[int] = field(default_factory=set)
    chunk_indices_with_transcript: Set[int] = field(default_factory=set)

    # Raw file list for custom checks
    all_files: Set[str] = field(default_factory=set)

    @property
    def is_downloadable(self) -> bool:
        """Content has source/audio or was compressed (manifest exists)."""
        return self.source_exists or self.audio_exists or self.storage_manifest_exists

    @property
    def has_chunk_audio(self) -> bool:
        """Content has at least one chunk with audio."""
        return len(self.chunk_indices_with_audio) > 0

    @property
    def has_chunk_transcripts(self) -> bool:
        """Content has at least one chunk with transcript."""
        return len(self.chunk_indices_with_transcript) > 0

    def get_chunks_needing_transcription(self) -> Set[int]:
        """Get chunk indices that have audio but no transcript."""
        return self.chunk_indices_with_audio - self.chunk_indices_with_transcript

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            'content_id': self.content_id,
            'audio_exists': self.audio_exists,
            'source_exists': self.source_exists,
            'storage_manifest_exists': self.storage_manifest_exists,
            'diarization_exists': self.diarization_exists,
            'speaker_embeddings_exist': self.speaker_embeddings_exist,
            'speaker_mapping_exists': self.speaker_mapping_exists,
            'stitched_exists': self.stitched_exists,
            'semantic_segments_exist': self.semantic_segments_exist,
            'chunk_count_with_audio': len(self.chunk_indices_with_audio),
            'chunk_count_with_transcript': len(self.chunk_indices_with_transcript),
            'total_files': len(self.all_files),
        }


class ContentFileChecker:
    """
    Checks S3 storage for content files.

    Handles both original and compressed file formats (.gz).
    Thread-safe and stateless - can be shared across calls.
    """

    # File extensions for source files
    SOURCE_EXTENSIONS = ['.mp4', '.mp3', '.wav', '.m4a']
    VIDEO_EXTENSIONS = ['.mp4', '.webm']
    AUDIO_EXTENSIONS = ['.wav', '.opus', '.mp3']

    def __init__(self):
        # Precompile regex for chunk detection
        self._chunk_audio_pattern = re.compile(r'chunks/(\d+)/audio\.wav')
        self._chunk_transcript_pattern = re.compile(r'chunks/(\d+)/transcript_words\.json(\.gz)?')

    def check(self, all_files: Set[str], content_id: str) -> ContentFiles:
        """
        Check what files exist for a content item.

        Args:
            all_files: Set of all file paths from S3 listing
            content_id: The content ID to check

        Returns:
            ContentFiles dataclass with existence flags
        """
        prefix = f"content/{content_id}/"

        # Filter to only this content's files for efficiency
        content_files = {f for f in all_files if f.startswith(prefix)}

        return ContentFiles(
            content_id=content_id,
            prefix=prefix,

            # Core files
            audio_exists=self._check_audio(content_files, prefix),
            source_exists=self._check_source(content_files, prefix),
            storage_manifest_exists=f"{prefix}storage_manifest.json" in content_files,

            # Processing outputs
            diarization_exists=self._check_json_file(content_files, prefix, "diarization.json"),
            speaker_embeddings_exist=self._check_json_file(content_files, prefix, "speaker_embeddings.json"),
            speaker_mapping_exists=self._check_json_file(content_files, prefix, "speaker_mapping.json"),
            stitched_exists=self._check_json_file(content_files, prefix, "transcript_diarized.json"),
            semantic_segments_exist=self._check_json_file(content_files, prefix, "semantic_segments.json"),

            # Chunks
            chunk_indices_with_audio=self._get_chunk_indices(content_files, self._chunk_audio_pattern),
            chunk_indices_with_transcript=self._get_chunk_indices(content_files, self._chunk_transcript_pattern),

            all_files=content_files
        )

    def check_from_prefix(self, s3_storage, content_id: str) -> ContentFiles:
        """
        Check files by listing from S3 storage directly.

        Args:
            s3_storage: S3Storage instance
            content_id: The content ID to check

        Returns:
            ContentFiles dataclass with existence flags
        """
        content_prefix = f"content/{content_id}/"
        try:
            all_files = set(s3_storage.list_files(content_prefix))
            return self.check(all_files, content_id)
        except Exception as e:
            logger.error(f"S3 error listing files for {content_id}: {e}")
            # Return empty result on S3 error
            return ContentFiles(
                content_id=content_id,
                prefix=content_prefix,
                all_files=set()
            )

    def _check_audio(self, files: Set[str], prefix: str) -> bool:
        """Check if any audio file exists (original or compressed)."""
        return any(
            f"{prefix}audio{ext}" in files
            for ext in self.AUDIO_EXTENSIONS
        )

    def _check_source(self, files: Set[str], prefix: str) -> bool:
        """Check if source or video file exists."""
        has_source = any(
            f"{prefix}source{ext}" in files
            for ext in self.SOURCE_EXTENSIONS
        )
        has_video = any(
            f"{prefix}video{ext}" in files
            for ext in self.VIDEO_EXTENSIONS
        )
        return has_source or has_video

    def _check_json_file(self, files: Set[str], prefix: str, filename: str) -> bool:
        """Check if JSON file exists (original or gzipped)."""
        return (
            f"{prefix}{filename}" in files or
            f"{prefix}{filename}.gz" in files
        )

    def _get_chunk_indices(self, files: Set[str], pattern: re.Pattern) -> Set[int]:
        """Extract chunk indices matching a pattern."""
        indices = set()
        for f in files:
            if match := pattern.search(f):
                indices.add(int(match.group(1)))
        return indices
