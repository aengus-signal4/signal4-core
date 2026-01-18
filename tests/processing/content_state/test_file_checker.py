"""
Tests for ContentFileChecker.
"""

import pytest
from src.processing.content_state.file_checker import ContentFileChecker, ContentFiles


class TestContentFiles:
    """Tests for ContentFiles dataclass."""

    def test_is_downloadable_with_source(self):
        """Content with source files is downloadable."""
        files = ContentFiles(
            content_id="test-123",
            prefix="content/test-123/",
            source_exists=True,
            audio_exists=False,
            storage_manifest_exists=False,
        )
        assert files.is_downloadable is True

    def test_is_downloadable_with_audio(self):
        """Content with audio is downloadable."""
        files = ContentFiles(
            content_id="test-123",
            prefix="content/test-123/",
            source_exists=False,
            audio_exists=True,
            storage_manifest_exists=False,
        )
        assert files.is_downloadable is True

    def test_is_downloadable_with_manifest(self):
        """Content with storage manifest is downloadable (compressed)."""
        files = ContentFiles(
            content_id="test-123",
            prefix="content/test-123/",
            source_exists=False,
            audio_exists=False,
            storage_manifest_exists=True,
        )
        assert files.is_downloadable is True

    def test_not_downloadable_when_empty(self):
        """Content with no files is not downloadable."""
        files = ContentFiles(
            content_id="test-123",
            prefix="content/test-123/",
            source_exists=False,
            audio_exists=False,
            storage_manifest_exists=False,
        )
        assert files.is_downloadable is False

    def test_has_chunk_audio(self):
        """Test chunk audio detection."""
        files = ContentFiles(
            content_id="test-123",
            prefix="content/test-123/",
            chunk_indices_with_audio={0, 1, 2},
        )
        assert files.has_chunk_audio is True

        files_empty = ContentFiles(
            content_id="test-123",
            prefix="content/test-123/",
            chunk_indices_with_audio=set(),
        )
        assert files_empty.has_chunk_audio is False

    def test_get_chunks_needing_transcription(self):
        """Test finding chunks that need transcription."""
        files = ContentFiles(
            content_id="test-123",
            prefix="content/test-123/",
            chunk_indices_with_audio={0, 1, 2, 3, 4},
            chunk_indices_with_transcript={0, 2, 4},
        )
        needing = files.get_chunks_needing_transcription()
        assert needing == {1, 3}

    def test_to_dict(self):
        """Test serialization to dict."""
        files = ContentFiles(
            content_id="test-123",
            prefix="content/test-123/",
            audio_exists=True,
            source_exists=False,
            diarization_exists=True,
            chunk_indices_with_audio={0, 1},
            chunk_indices_with_transcript={0},
            all_files={"file1", "file2", "file3"},
        )
        d = files.to_dict()
        assert d['content_id'] == "test-123"
        assert d['audio_exists'] is True
        assert d['source_exists'] is False
        assert d['diarization_exists'] is True
        assert d['chunk_count_with_audio'] == 2
        assert d['chunk_count_with_transcript'] == 1
        assert d['total_files'] == 3


class TestContentFileChecker:
    """Tests for ContentFileChecker."""

    def setup_method(self):
        """Set up test fixtures."""
        self.checker = ContentFileChecker()

    def test_check_empty_files(self):
        """Check with empty file set."""
        files = self.checker.check(set(), "test-123")
        assert files.content_id == "test-123"
        assert files.prefix == "content/test-123/"
        assert files.audio_exists is False
        assert files.source_exists is False
        assert files.diarization_exists is False
        assert len(files.all_files) == 0

    def test_check_audio_wav(self):
        """Detect WAV audio file."""
        all_files = {"content/test-123/audio.wav"}
        files = self.checker.check(all_files, "test-123")
        assert files.audio_exists is True

    def test_check_audio_opus(self):
        """Detect Opus audio file (compressed format)."""
        all_files = {"content/test-123/audio.opus"}
        files = self.checker.check(all_files, "test-123")
        assert files.audio_exists is True

    def test_check_audio_mp3(self):
        """Detect MP3 audio file."""
        all_files = {"content/test-123/audio.mp3"}
        files = self.checker.check(all_files, "test-123")
        assert files.audio_exists is True

    def test_check_source_mp4(self):
        """Detect MP4 source file."""
        all_files = {"content/test-123/source.mp4"}
        files = self.checker.check(all_files, "test-123")
        assert files.source_exists is True

    def test_check_video_mp4(self):
        """Detect video.mp4 file."""
        all_files = {"content/test-123/video.mp4"}
        files = self.checker.check(all_files, "test-123")
        assert files.source_exists is True

    def test_check_video_webm(self):
        """Detect video.webm file."""
        all_files = {"content/test-123/video.webm"}
        files = self.checker.check(all_files, "test-123")
        assert files.source_exists is True

    def test_check_diarization_json(self):
        """Detect diarization.json file."""
        all_files = {"content/test-123/diarization.json"}
        files = self.checker.check(all_files, "test-123")
        assert files.diarization_exists is True

    def test_check_diarization_json_gz(self):
        """Detect compressed diarization.json.gz file."""
        all_files = {"content/test-123/diarization.json.gz"}
        files = self.checker.check(all_files, "test-123")
        assert files.diarization_exists is True

    def test_check_storage_manifest(self):
        """Detect storage manifest."""
        all_files = {"content/test-123/storage_manifest.json"}
        files = self.checker.check(all_files, "test-123")
        assert files.storage_manifest_exists is True

    def test_check_stitched(self):
        """Detect transcript_diarized.json."""
        all_files = {"content/test-123/transcript_diarized.json"}
        files = self.checker.check(all_files, "test-123")
        assert files.stitched_exists is True

    def test_check_stitched_gz(self):
        """Detect compressed transcript_diarized.json.gz."""
        all_files = {"content/test-123/transcript_diarized.json.gz"}
        files = self.checker.check(all_files, "test-123")
        assert files.stitched_exists is True

    def test_check_semantic_segments(self):
        """Detect semantic_segments.json."""
        all_files = {"content/test-123/semantic_segments.json"}
        files = self.checker.check(all_files, "test-123")
        assert files.semantic_segments_exist is True

    def test_check_chunk_audio(self):
        """Detect chunk audio files."""
        all_files = {
            "content/test-123/chunks/0/audio.wav",
            "content/test-123/chunks/1/audio.wav",
            "content/test-123/chunks/2/audio.wav",
        }
        files = self.checker.check(all_files, "test-123")
        assert files.chunk_indices_with_audio == {0, 1, 2}

    def test_check_chunk_transcripts(self):
        """Detect chunk transcript files."""
        all_files = {
            "content/test-123/chunks/0/transcript_words.json",
            "content/test-123/chunks/1/transcript_words.json.gz",
            "content/test-123/chunks/3/transcript_words.json",
        }
        files = self.checker.check(all_files, "test-123")
        assert files.chunk_indices_with_transcript == {0, 1, 3}

    def test_check_filters_by_content_id(self):
        """Only include files for the requested content ID."""
        all_files = {
            "content/test-123/audio.wav",
            "content/test-456/audio.wav",
            "content/test-789/diarization.json",
        }
        files = self.checker.check(all_files, "test-123")
        assert files.audio_exists is True
        assert len(files.all_files) == 1

    def test_check_full_content(self):
        """Check fully processed content."""
        all_files = {
            "content/test-123/audio.opus",
            "content/test-123/diarization.json.gz",
            "content/test-123/speaker_embeddings.json.gz",
            "content/test-123/speaker_mapping.json.gz",
            "content/test-123/transcript_diarized.json.gz",
            "content/test-123/semantic_segments.json.gz",
            "content/test-123/storage_manifest.json",
            "content/test-123/chunks/0/transcript_words.json.gz",
            "content/test-123/chunks/1/transcript_words.json.gz",
        }
        files = self.checker.check(all_files, "test-123")

        assert files.audio_exists is True
        assert files.diarization_exists is True
        assert files.speaker_embeddings_exist is True
        assert files.speaker_mapping_exists is True
        assert files.stitched_exists is True
        assert files.semantic_segments_exist is True
        assert files.storage_manifest_exists is True
        assert files.chunk_indices_with_transcript == {0, 1}
        assert files.is_downloadable is True


class TestContentFileCheckerPerformance:
    """Performance tests for ContentFileChecker."""

    def test_large_file_set(self):
        """Handle large file sets efficiently."""
        checker = ContentFileChecker()

        # Create a large file set with 10000 files
        all_files = set()
        for i in range(1000):
            content_id = f"content-{i:04d}"
            all_files.add(f"content/{content_id}/audio.wav")
            all_files.add(f"content/{content_id}/diarization.json")
            for chunk in range(10):
                all_files.add(f"content/{content_id}/chunks/{chunk}/audio.wav")
                all_files.add(f"content/{content_id}/chunks/{chunk}/transcript_words.json")

        # Check one content item
        files = checker.check(all_files, "content-0500")

        assert files.audio_exists is True
        assert files.diarization_exists is True
        assert files.chunk_indices_with_audio == set(range(10))
        assert files.chunk_indices_with_transcript == set(range(10))
        # Should only contain files for this content
        assert len(files.all_files) == 22  # audio + diarization + 10 chunk audio + 10 chunk transcripts
