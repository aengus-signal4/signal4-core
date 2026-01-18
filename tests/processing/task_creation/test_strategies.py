"""
Tests for task creation strategies.
"""

import pytest
from unittest.mock import MagicMock, patch
from sqlalchemy.orm import Session

from src.processing.task_creation.strategies import (
    DownloadTaskStrategy,
    ConvertTaskStrategy,
    TranscribeTaskStrategy,
    DiarizeTaskStrategy,
    StitchTaskStrategy,
    CleanupTaskStrategy,
)


class MockContent:
    """Mock Content object for testing."""

    def __init__(
        self,
        content_id="test-123",
        platform="youtube",
        blocked_download=False,
        is_transcribed=False,
        is_diarized=False,
        is_stitched=False,
        is_compressed=False,
        stitch_version=None,
        meta_data=None,
    ):
        self.content_id = content_id
        self.platform = platform
        self.blocked_download = blocked_download
        self.is_transcribed = is_transcribed
        self.is_diarized = is_diarized
        self.is_stitched = is_stitched
        self.is_compressed = is_compressed
        self.stitch_version = stitch_version
        self.meta_data = meta_data or {}


class MockTaskQueue:
    """Mock TaskQueue object for testing."""

    def __init__(self, status="pending", error=None, result=None):
        self.status = status
        self.error = error
        self.result = result or {}


class TestDownloadTaskStrategy:
    """Tests for DownloadTaskStrategy."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {}
        self.strategy = DownloadTaskStrategy(self.config, "youtube")
        self.session = MagicMock(spec=Session)

    def test_task_type(self):
        """Task type is correct."""
        assert self.strategy.task_type == "download_youtube"

        rumble_strategy = DownloadTaskStrategy(self.config, "rumble")
        assert rumble_strategy.task_type == "download_rumble"

    def test_should_create_when_not_blocked(self):
        """Should create task when content is not blocked."""
        content = MockContent(blocked_download=False)

        # Mock no existing tasks
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.count.return_value = 0
        mock_query.first.return_value = None
        self.session.query.return_value = mock_query

        should_create, reason = self.strategy.should_create(self.session, content)
        assert should_create is True
        assert reason is None

    def test_should_not_create_when_blocked(self):
        """Should not create task when content is blocked."""
        content = MockContent(blocked_download=True)

        should_create, reason = self.strategy.should_create(self.session, content)
        assert should_create is False
        assert "blocked" in reason.lower()

    def test_should_not_create_with_permanent_error(self):
        """Should not create task when content has permanent error."""
        content = MockContent(
            meta_data={'permanent_error': True, 'error_code': 'MEMBERS_ONLY'}
        )

        should_create, reason = self.strategy.should_create(self.session, content)
        assert should_create is False
        assert "permanent error" in reason.lower()

    def test_should_not_create_when_at_task_limit(self):
        """Should not create task when 3+ tasks exist."""
        content = MockContent()

        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.count.return_value = 3
        self.session.query.return_value = mock_query

        should_create, reason = self.strategy.should_create(self.session, content)
        assert should_create is False
        assert "3" in reason

    def test_should_not_create_with_permanent_failure(self):
        """Should not create task after permanent failure."""
        content = MockContent()

        failed_task = MockTaskQueue(status="failed", error="Video unavailable")

        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.count.return_value = 1
        mock_query.first.return_value = failed_task
        self.session.query.return_value = mock_query

        should_create, reason = self.strategy.should_create(self.session, content)
        assert should_create is False
        assert "permanent" in reason.lower()


class TestConvertTaskStrategy:
    """Tests for ConvertTaskStrategy."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {}
        self.strategy = ConvertTaskStrategy(self.config)
        self.session = MagicMock(spec=Session)

    def test_task_type(self):
        """Task type is correct."""
        assert self.strategy.task_type == "convert"

    def test_should_create_when_no_errors(self):
        """Should create task when no errors."""
        content = MockContent()

        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.count.return_value = 0
        mock_query.first.return_value = None
        self.session.query.return_value = mock_query

        should_create, reason = self.strategy.should_create(self.session, content)
        assert should_create is True

    def test_should_not_create_with_extract_failure(self):
        """Should not create task after extract audio failure."""
        content = MockContent()

        failed_task = MockTaskQueue(
            status="failed",
            error="Failed to extract audio from source file"
        )

        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.count.return_value = 1
        mock_query.first.return_value = failed_task
        self.session.query.return_value = mock_query

        should_create, reason = self.strategy.should_create(self.session, content)
        assert should_create is False
        assert "permanent" in reason.lower()


class TestTranscribeTaskStrategy:
    """Tests for TranscribeTaskStrategy."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {}
        self.strategy = TranscribeTaskStrategy(self.config)
        self.session = MagicMock(spec=Session)

    def test_task_type(self):
        """Task type is correct."""
        assert self.strategy.task_type == "transcribe"

    def test_requires_chunk_index(self):
        """Should require chunk_index."""
        content = MockContent()

        should_create, reason = self.strategy.should_create(self.session, content)
        assert should_create is False
        assert "chunk_index" in reason.lower()

    def test_should_create_with_chunk_index(self):
        """Should create task when chunk_index provided."""
        content = MockContent()

        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.count.return_value = 0
        mock_query.first.return_value = None
        self.session.query.return_value = mock_query

        should_create, reason = self.strategy.should_create(
            self.session, content, chunk_index=0
        )
        assert should_create is True

    def test_input_data_includes_chunk_index(self):
        """Input data should include chunk_index."""
        content = MockContent()
        input_data = self.strategy.get_input_data(content, "test-project", chunk_index=5)
        assert input_data['chunk_index'] == 5
        assert input_data['project'] == "test-project"


class TestDiarizeTaskStrategy:
    """Tests for DiarizeTaskStrategy."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {}
        self.strategy = DiarizeTaskStrategy(self.config)
        self.session = MagicMock(spec=Session)

    def test_task_type(self):
        """Task type is correct."""
        assert self.strategy.task_type == "diarize"

    def test_should_not_create_when_ignored(self):
        """Should not create task when diarization_ignored flag set."""
        content = MockContent(meta_data={'diarization_ignored': True})

        should_create, reason = self.strategy.should_create(self.session, content)
        assert should_create is False
        assert "ignored" in reason.lower()

    def test_should_create_when_not_ignored(self):
        """Should create task when not ignored."""
        content = MockContent()

        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.count.return_value = 0
        mock_query.first.return_value = None
        self.session.query.return_value = mock_query

        should_create, reason = self.strategy.should_create(self.session, content)
        assert should_create is True


class TestStitchTaskStrategy:
    """Tests for StitchTaskStrategy."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'processing': {
                'stitch': {
                    'current_version': 'stitch_v13'
                }
            }
        }
        self.strategy = StitchTaskStrategy(self.config)
        self.session = MagicMock(spec=Session)

    def test_task_type(self):
        """Task type is correct."""
        assert self.strategy.task_type == "stitch"

    def test_get_current_stitch_version(self):
        """Gets version from config."""
        assert self.strategy.get_current_stitch_version() == "stitch_v13"

    def test_should_not_create_when_not_transcribed(self):
        """Should not create task when not transcribed."""
        content = MockContent(is_transcribed=False)

        should_create, reason = self.strategy.should_create(self.session, content)
        assert should_create is False
        assert "transcribed" in reason.lower()

    def test_should_not_create_when_not_diarized(self):
        """Should not create task when not diarized."""
        content = MockContent(is_transcribed=True, is_diarized=False)

        should_create, reason = self.strategy.should_create(self.session, content)
        assert should_create is False
        assert "diarized" in reason.lower()

    def test_should_create_when_diarization_ignored(self):
        """Should create task when diarization_ignored flag set."""
        content = MockContent(
            is_transcribed=True,
            is_diarized=False,
            meta_data={'diarization_ignored': True}
        )

        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.count.return_value = 0
        mock_query.first.return_value = None
        self.session.query.return_value = mock_query

        should_create, reason = self.strategy.should_create(self.session, content)
        assert should_create is True

    def test_should_not_create_when_already_stitched_with_current_version(self):
        """Should not create task when already stitched with current version."""
        content = MockContent(
            is_transcribed=True,
            is_diarized=True,
            is_stitched=True,
            stitch_version='stitch_v13'
        )

        should_create, reason = self.strategy.should_create(self.session, content)
        assert should_create is False
        assert "already stitched" in reason.lower()

    def test_should_create_when_version_outdated(self):
        """Should create task when stitch version is outdated."""
        content = MockContent(
            is_transcribed=True,
            is_diarized=True,
            is_stitched=True,
            stitch_version='stitch_v10'
        )

        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.count.return_value = 0
        mock_query.first.return_value = None
        self.session.query.return_value = mock_query

        should_create, reason = self.strategy.should_create(self.session, content)
        assert should_create is True


class TestCleanupTaskStrategy:
    """Tests for CleanupTaskStrategy."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {}
        self.strategy = CleanupTaskStrategy(self.config)
        self.session = MagicMock(spec=Session)

    def test_task_type(self):
        """Task type is correct."""
        assert self.strategy.task_type == "cleanup"

    def test_should_not_create_when_not_stitched(self):
        """Should not create task when not stitched."""
        content = MockContent(is_stitched=False)

        should_create, reason = self.strategy.should_create(self.session, content)
        assert should_create is False
        assert "stitched" in reason.lower()

    def test_should_not_create_when_already_compressed(self):
        """Should not create task when already compressed."""
        content = MockContent(is_stitched=True, is_compressed=True)

        should_create, reason = self.strategy.should_create(self.session, content)
        assert should_create is False
        assert "compressed" in reason.lower()

    def test_should_create_when_stitched_not_compressed(self):
        """Should create task when stitched but not compressed."""
        content = MockContent(is_stitched=True, is_compressed=False)

        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.count.return_value = 0
        mock_query.first.return_value = None
        self.session.query.return_value = mock_query

        should_create, reason = self.strategy.should_create(self.session, content)
        assert should_create is True
