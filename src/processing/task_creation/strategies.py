"""
Task creation strategies for different task types.

Each strategy encapsulates the logic for determining if a task should be created
and what input data it should have.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import logging

from sqlalchemy.orm import Session
from sqlalchemy import text

from src.database.models import Content, TaskQueue, ContentChunk
from src.utils.version_utils import should_recreate_stitch_task

logger = logging.getLogger(__name__)

TASK_STATUS_PENDING = 'pending'
TASK_STATUS_PROCESSING = 'processing'
TASK_STATUS_FAILED = 'failed'


class TaskCreationStrategy(ABC):
    """Base class for task creation strategies."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    @property
    @abstractmethod
    def task_type(self) -> str:
        """Return the task type this strategy handles."""
        pass

    @abstractmethod
    def should_create(
        self,
        session: Session,
        content: Content,
        **kwargs
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if task should be created.

        Returns:
            Tuple of (should_create, reason_if_not)
        """
        pass

    def get_input_data(self, content: Content, project: str, **kwargs) -> Dict[str, Any]:
        """Get input data for the task."""
        return {'project': project}

    def _check_task_limit(
        self,
        session: Session,
        content: Content,
        max_tasks: int = 3,
        **kwargs
    ) -> Tuple[bool, Optional[str]]:
        """Check if task count limit has been reached."""
        query = session.query(TaskQueue).filter(
            TaskQueue.content_id == content.content_id,
            TaskQueue.task_type == self.task_type,
            TaskQueue.status.in_([TASK_STATUS_PENDING, TASK_STATUS_PROCESSING, TASK_STATUS_FAILED])
        )

        # For transcribe tasks, filter by chunk_index
        if self.task_type == 'transcribe':
            chunk_index = kwargs.get('chunk_index')
            if chunk_index is not None:
                query = query.filter(
                    text("input_data->>'chunk_index' = :chunk_index").params(chunk_index=str(chunk_index))
                )

        count = query.count()
        if count >= max_tasks:
            return False, f"Already have {count} {self.task_type} tasks (limit: {max_tasks})"
        return True, None

    def _check_permanent_error(
        self,
        session: Session,
        content: Content
    ) -> Tuple[bool, Optional[str]]:
        """Check if content has a permanent error."""
        if content.meta_data and content.meta_data.get('permanent_error'):
            error_code = content.meta_data.get('error_code', 'UNKNOWN')
            return False, f"Content has permanent error: {error_code}"
        return True, None


class DownloadTaskStrategy(TaskCreationStrategy):
    """Strategy for creating download tasks."""

    def __init__(self, config: Dict[str, Any], platform: str):
        super().__init__(config)
        self._platform = platform

    @property
    def task_type(self) -> str:
        return f"download_{self._platform}"

    def should_create(
        self,
        session: Session,
        content: Content,
        **kwargs
    ) -> Tuple[bool, Optional[str]]:
        # Check blocked status
        if content.blocked_download:
            return False, "Content is blocked for download"

        # Check permanent error
        can_create, reason = self._check_permanent_error(session, content)
        if not can_create:
            return False, reason

        # Check task limit
        can_create, reason = self._check_task_limit(session, content)
        if not can_create:
            return False, reason

        # Check for permanent failure patterns
        failed_task = session.query(TaskQueue).filter(
            TaskQueue.content_id == content.content_id,
            TaskQueue.task_type == self.task_type,
            TaskQueue.status == TASK_STATUS_FAILED
        ).first()

        if failed_task and failed_task.error:
            error_str = str(failed_task.error)
            permanent_errors = [
                'Join this channel to get access to members-only content',
                'Sign in to confirm your age',
                'Bad URL',
                'Video unavailable',
                'This video is private',
                'Private video',
                'HTTP Error 400: Bad Request',
                'HTTP Error 403: Forbidden',
            ]
            if any(pattern in error_str for pattern in permanent_errors):
                return False, f"Permanent failure: {error_str[:100]}"

        return True, None


class ConvertTaskStrategy(TaskCreationStrategy):
    """Strategy for creating convert tasks."""

    @property
    def task_type(self) -> str:
        return "convert"

    def should_create(
        self,
        session: Session,
        content: Content,
        **kwargs
    ) -> Tuple[bool, Optional[str]]:
        # Check permanent error
        can_create, reason = self._check_permanent_error(session, content)
        if not can_create:
            return False, reason

        # Check task limit
        can_create, reason = self._check_task_limit(session, content)
        if not can_create:
            return False, reason

        # Check for permanent failure patterns
        failed_task = session.query(TaskQueue).filter(
            TaskQueue.content_id == content.content_id,
            TaskQueue.task_type == self.task_type,
            TaskQueue.status == TASK_STATUS_FAILED
        ).first()

        if failed_task and failed_task.error:
            error_str = str(failed_task.error)
            permanent_errors = [
                'Failed to extract audio from source file',
                'Corrupt media',
            ]
            if any(pattern in error_str for pattern in permanent_errors):
                return False, f"Permanent failure: {error_str[:100]}"

        return True, None


class TranscribeTaskStrategy(TaskCreationStrategy):
    """Strategy for creating transcribe tasks."""

    @property
    def task_type(self) -> str:
        return "transcribe"

    def should_create(
        self,
        session: Session,
        content: Content,
        **kwargs
    ) -> Tuple[bool, Optional[str]]:
        chunk_index = kwargs.get('chunk_index')
        if chunk_index is None:
            return False, "chunk_index required for transcribe tasks"

        # Check permanent error
        can_create, reason = self._check_permanent_error(session, content)
        if not can_create:
            return False, reason

        # Check task limit (per chunk)
        can_create, reason = self._check_task_limit(session, content, chunk_index=chunk_index)
        if not can_create:
            return False, reason

        # Check for audio download failure
        failed_task = session.query(TaskQueue).filter(
            TaskQueue.content_id == content.content_id,
            TaskQueue.task_type == self.task_type,
            TaskQueue.status == TASK_STATUS_FAILED,
            text("input_data->>'chunk_index' = :chunk_index").params(chunk_index=str(chunk_index))
        ).first()

        if failed_task and failed_task.error:
            if "Failed to download audio chunk" in str(failed_task.error):
                return False, "Audio chunk download failure"

        return True, None

    def get_input_data(self, content: Content, project: str, **kwargs) -> Dict[str, Any]:
        return {
            'project': project,
            'chunk_index': kwargs.get('chunk_index')
        }


class DiarizeTaskStrategy(TaskCreationStrategy):
    """Strategy for creating diarize tasks."""

    @property
    def task_type(self) -> str:
        return "diarize"

    def should_create(
        self,
        session: Session,
        content: Content,
        **kwargs
    ) -> Tuple[bool, Optional[str]]:
        # Check diarization_ignored flag
        if content.meta_data and content.meta_data.get('diarization_ignored'):
            return False, "Diarization ignored for this content"

        # Check permanent error
        can_create, reason = self._check_permanent_error(session, content)
        if not can_create:
            return False, reason

        # Check task limit
        can_create, reason = self._check_task_limit(session, content)
        if not can_create:
            return False, reason

        return True, None


class StitchTaskStrategy(TaskCreationStrategy):
    """Strategy for creating stitch tasks."""

    @property
    def task_type(self) -> str:
        return "stitch"

    def get_current_stitch_version(self) -> str:
        """Get current stitch version from config."""
        return self.config.get('processing', {}).get(
            'stitch', {}
        ).get('current_version', 'stitch_v1')

    def should_create(
        self,
        session: Session,
        content: Content,
        **kwargs
    ) -> Tuple[bool, Optional[str]]:
        current_version = self.get_current_stitch_version()

        # Check if already stitched with compatible version
        if content.is_stitched and not should_recreate_stitch_task(current_version, content.stitch_version):
            return False, f"Already stitched with compatible version {content.stitch_version}"

        # Check prerequisites
        if not content.is_transcribed:
            return False, "Not fully transcribed"

        if not content.is_diarized:
            # Check diarization_ignored
            if not (content.meta_data and content.meta_data.get('diarization_ignored')):
                return False, "Not diarized"

        # Check permanent error
        can_create, reason = self._check_permanent_error(session, content)
        if not can_create:
            return False, reason

        # Check task limit
        can_create, reason = self._check_task_limit(session, content)
        if not can_create:
            return False, reason

        # Check for permanent stitch failures
        failed_task = session.query(TaskQueue).filter(
            TaskQueue.content_id == content.content_id,
            TaskQueue.task_type == self.task_type,
            TaskQueue.status == TASK_STATUS_FAILED
        ).first()

        if failed_task and failed_task.error:
            error_str = str(failed_task.error)
            # Only these specific errors are permanent for stitch
            permanent_errors = [
                'No diarization data',
                'No transcription data',
                'Content is blocked',
            ]
            if any(pattern in error_str for pattern in permanent_errors):
                return False, f"Permanent failure: {error_str[:100]}"

            # Missing audio is NOT permanent - could be transient S3 issue
            if 'No audio file found' in error_str:
                logger.info(f"Stitch failed due to missing audio - allowing retry")

        return True, None


class CleanupTaskStrategy(TaskCreationStrategy):
    """Strategy for creating cleanup tasks."""

    @property
    def task_type(self) -> str:
        return "cleanup"

    def should_create(
        self,
        session: Session,
        content: Content,
        **kwargs
    ) -> Tuple[bool, Optional[str]]:
        # Only create cleanup if stitched but not compressed
        if not content.is_stitched:
            return False, "Not stitched yet"

        if content.is_compressed:
            return False, "Already compressed"

        # Check permanent error
        can_create, reason = self._check_permanent_error(session, content)
        if not can_create:
            return False, reason

        # Check task limit
        can_create, reason = self._check_task_limit(session, content)
        if not can_create:
            return False, reason

        return True, None
