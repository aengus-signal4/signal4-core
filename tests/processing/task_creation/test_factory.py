"""
Tests for TaskFactory.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timezone
from sqlalchemy.orm import Session

from src.processing.task_creation.factory import TaskFactory


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
        publish_date=None,
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
        self.publish_date = publish_date


class MockTaskQueue:
    """Mock TaskQueue object for testing."""

    def __init__(
        self,
        id=1,
        status="pending",
        error=None,
        result=None,
        attempts=0,
        max_attempts=3
    ):
        self.id = id
        self.status = status
        self.error = error
        self.result = result or {}
        self.attempts = attempts
        self.max_attempts = max_attempts


class TestTaskFactory:
    """Tests for TaskFactory."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = {
            'processing': {
                'stitch': {
                    'current_version': 'stitch_v13'
                }
            },
            'active_projects': {
                'test-project': {
                    'priority': 2,
                    'enabled': True,
                }
            }
        }
        self.factory = TaskFactory(self.config)
        self.session = MagicMock(spec=Session)

    def test_registers_default_strategies(self):
        """Factory registers default strategies on init."""
        assert 'download_youtube' in self.factory._strategies
        assert 'download_rumble' in self.factory._strategies
        assert 'download_podcast' in self.factory._strategies
        assert 'convert' in self.factory._strategies
        assert 'transcribe' in self.factory._strategies
        assert 'diarize' in self.factory._strategies
        assert 'stitch' in self.factory._strategies
        assert 'cleanup' in self.factory._strategies

    @pytest.mark.asyncio
    async def test_create_task_unknown_type(self):
        """Returns error for unknown task type."""
        content = MockContent()

        task_id, reason = await self.factory.create_task(
            self.session, content, "unknown_type", "test-project"
        )

        assert task_id is None
        assert "unknown task type" in reason.lower()

    @pytest.mark.asyncio
    async def test_create_task_blocked_content(self):
        """Returns error for blocked content."""
        content = MockContent(blocked_download=True)

        task_id, reason = await self.factory.create_task(
            self.session, content, "download_youtube", "test-project"
        )

        assert task_id is None
        assert "blocked" in reason.lower()

    @pytest.mark.asyncio
    async def test_create_task_new_task(self):
        """Creates new task when none exists."""
        content = MockContent()

        # Mock no existing tasks
        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.count.return_value = 0
        mock_query.first.return_value = None
        self.session.query.return_value = mock_query

        # Mock flush to set task ID
        def mock_add(task):
            task.id = 42
        self.session.add.side_effect = mock_add

        task_id, reason = await self.factory.create_task(
            self.session, content, "convert", "test-project"
        )

        assert task_id == 42
        assert reason is None
        self.session.add.assert_called_once()
        self.session.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_task_already_exists_pending(self):
        """Returns error when pending task exists."""
        content = MockContent()

        existing_task = MockTaskQueue(id=10, status="pending")

        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.count.return_value = 1
        mock_query.first.return_value = existing_task
        self.session.query.return_value = mock_query

        task_id, reason = await self.factory.create_task(
            self.session, content, "convert", "test-project"
        )

        assert task_id is None
        assert "already exists" in reason.lower()

    @pytest.mark.asyncio
    async def test_create_task_resets_failed_task(self):
        """Resets failed task to pending when appropriate."""
        content = MockContent()

        # First call returns count 1 (for limit check)
        # Second call returns the failed task
        failed_task = MockTaskQueue(id=10, status="failed", error="Transient error")

        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.count.return_value = 1
        mock_query.first.return_value = failed_task
        self.session.query.return_value = mock_query

        task_id, reason = await self.factory.create_task(
            self.session, content, "convert", "test-project"
        )

        assert task_id == 10
        assert reason is None
        assert failed_task.status == "pending"
        assert failed_task.error is None

    @pytest.mark.asyncio
    async def test_create_task_does_not_reset_permanent_failure(self):
        """Does not reset permanently failed task."""
        content = MockContent()

        failed_task = MockTaskQueue(
            id=10,
            status="failed",
            result={'permanent': True, 'error_code': 'MEMBERS_ONLY'}
        )

        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.count.return_value = 1
        mock_query.first.return_value = failed_task
        self.session.query.return_value = mock_query

        task_id, reason = await self.factory.create_task(
            self.session, content, "convert", "test-project"
        )

        assert task_id is None
        assert "permanently failed" in reason.lower()

    @pytest.mark.asyncio
    async def test_create_transcribe_task_with_chunk_index(self):
        """Creates transcribe task with chunk_index in input_data."""
        content = MockContent()

        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.count.return_value = 0
        mock_query.first.return_value = None
        self.session.query.return_value = mock_query

        created_task = None
        def capture_task(task):
            nonlocal created_task
            created_task = task
            task.id = 42
        self.session.add.side_effect = capture_task

        task_id, reason = await self.factory.create_task(
            self.session, content, "transcribe", "test-project",
            input_data={'chunk_index': 5}
        )

        assert task_id == 42
        assert created_task is not None
        assert created_task.input_data['chunk_index'] == 5
        assert created_task.input_data['project'] == "test-project"

    @pytest.mark.asyncio
    async def test_convenience_create_download_task(self):
        """Convenience method creates correct task type."""
        content = MockContent(platform="rumble")

        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.count.return_value = 0
        mock_query.first.return_value = None
        self.session.query.return_value = mock_query

        created_task = None
        def capture_task(task):
            nonlocal created_task
            created_task = task
            task.id = 42
        self.session.add.side_effect = capture_task

        task_id, reason = await self.factory.create_download_task(
            self.session, content, "test-project"
        )

        assert task_id == 42
        assert created_task.task_type == "download_rumble"

    @pytest.mark.asyncio
    async def test_convenience_create_stitch_task(self):
        """Convenience method creates stitch task."""
        content = MockContent(is_transcribed=True, is_diarized=True)

        mock_query = MagicMock()
        mock_query.filter.return_value = mock_query
        mock_query.count.return_value = 0
        mock_query.first.return_value = None
        self.session.query.return_value = mock_query

        created_task = None
        def capture_task(task):
            nonlocal created_task
            created_task = task
            task.id = 42
        self.session.add.side_effect = capture_task

        task_id, reason = await self.factory.create_stitch_task(
            self.session, content, "test-project"
        )

        assert task_id == 42
        assert created_task.task_type == "stitch"

    def test_priority_calculation_with_project(self):
        """Calculates priority with project priority."""
        content = MockContent(publish_date=datetime(2024, 1, 15, tzinfo=timezone.utc))

        priority = self.factory._calculate_priority(
            self.session, content, "test-project", None
        )

        # Priority is calculated as:
        # (days_since_epoch - 20000) * 1000 + (project_priority * 1000000)
        # For 2024-01-15, days_since_epoch is about 19738, so:
        # (19738 - 20000) * 1000 + 2_000_000 = -262000 + 2_000_000 = 1_738_000
        # The actual value depends on exact date calculation
        assert priority > 0
        assert priority < 10000000  # Not recent, so no recency boost

    def test_priority_calculation_without_publish_date(self):
        """Calculates priority with no publish date."""
        content = MockContent(publish_date=None)

        priority = self.factory._calculate_priority(
            self.session, content, "test-project", None
        )

        # With no publish_date, returns project_priority * 1000000
        assert priority == 2000000
