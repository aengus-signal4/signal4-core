"""
Task Creator - Handles creation of processing tasks.

This module contains the logic for:
- Creating tasks if they don't already exist
- Checking for permanent errors before creating tasks
- Handling task recreation for failed tasks
- Priority calculation based on project and date
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Tuple, List

from sqlalchemy.orm import Session
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from src.database.models import TaskQueue, Content, ContentChunk
from src.utils.priority import calculate_priority_by_date
from src.utils.version_utils import should_recreate_stitch_task, format_version_comparison_log
from src.utils.logger import setup_worker_logger

logger = setup_worker_logger('task_creator')

# Task status constants
TASK_STATUS_PENDING = 'pending'
TASK_STATUS_PROCESSING = 'processing'
TASK_STATUS_COMPLETED = 'completed'
TASK_STATUS_FAILED = 'failed'


class TaskCreator:
    """
    Handles creation of processing tasks.

    Responsibilities:
    - Check if task already exists before creating
    - Handle permanent errors that should prevent task creation
    - Calculate task priority
    - Reset failed tasks for retry when appropriate
    """

    # Permanent errors that should block task recreation
    PERMANENT_ERROR_PATTERNS = [
        'Join this channel to get access to members-only content',
        'Sign in to confirm your age',
        'Bad URL',
        'Permanent failure',
        'Video unavailable',
        'This video is private',
        'Content blocked',
        'This live event will begin in',
        'This live stream has ended',
        'Premiere will begin shortly',
        'Private video',
        'HTTP Error 400: Bad Request',
        'HTTP Error 403: Forbidden',
        'Failed to extract audio from source file'
    ]

    # Stitch-specific permanent errors
    STITCH_PERMANENT_ERRORS = [
        'No diarization data',
        'No transcription data',
        'Content is blocked',
    ]

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize TaskCreator.

        Args:
            config: Application configuration dict
        """
        self.config = config

    def get_current_stitch_version(self) -> str:
        """Get the current stitch version from config."""
        try:
            return self.config.get('processing', {}).get('stitch', {}).get('current_version', 'stitch_v1')
        except Exception as e:
            logger.warning(f"Failed to get stitch version from config: {e}")
            return 'stitch_v1'

    async def create_task_if_not_exists(
        self,
        session: Session,
        content_id: str,
        task_type: str,
        input_data: Dict[str, Any],
        priority: int = None,
        content: Content = None
    ) -> Tuple[Optional[int], Optional[str]]:
        """
        Create a new task if a pending/processing one doesn't already exist.

        Args:
            session: Database session
            content_id: Content ID for the task
            task_type: Type of task (download_youtube, convert, transcribe, etc.)
            input_data: Task input data
            priority: Optional priority override
            content: Optional Content object (avoids extra query)

        Returns:
            Tuple of (task_id, block_reason):
            - task_id: New task ID if created, None if not created
            - block_reason: Reason task was not created (None if task was created)
        """
        logger.debug(f"Checking for existing task {task_type} for {content_id}")

        # Check for permanent error in content metadata
        content_obj = content or session.query(Content).filter_by(content_id=content_id).first()
        block_reason = self._check_permanent_error(content_obj)
        if block_reason:
            return None, block_reason

        # Check task limit (max 3 tasks including failed)
        block_reason = self._check_task_limit(session, content_id, task_type, input_data)
        if block_reason:
            return None, block_reason

        # Special check for stitch tasks
        if task_type == 'stitch':
            block_reason = self._check_stitch_already_done(content_obj)
            if block_reason:
                return None, block_reason

        # Check for permanent failed task
        block_reason = self._check_permanent_failure(session, content_id, task_type)
        if block_reason:
            return None, block_reason

        # Check for existing pending/processing/failed task
        existing_task = self._find_existing_task(session, content_id, task_type, input_data)

        if existing_task:
            if existing_task.status == TASK_STATUS_FAILED:
                # Reset failed task for retry
                return self._reset_failed_task(existing_task, session, content_id, task_type, input_data)
            else:
                logger.debug(f"Task {task_type} already exists (pending/processing) for {content_id}")
                return None, "Task already exists (pending/processing)"

        # Create new task
        return self._create_new_task(
            session, content_id, task_type, input_data, priority, content_obj
        )

    def _check_permanent_error(self, content: Content) -> Optional[str]:
        """Check if content has a permanent error in metadata."""
        if content and content.meta_data:
            if content.meta_data.get('permanent_error'):
                error_code = content.meta_data.get('error_code', 'UNKNOWN')
                error_msg = content.meta_data.get('error_message', 'Permanent error')
                logger.warning(f"Content {content.content_id} has permanent error ({error_code}): {error_msg}")
                return f"Content has permanent error: {error_code}"
        return None

    def _check_task_limit(
        self,
        session: Session,
        content_id: str,
        task_type: str,
        input_data: Dict[str, Any]
    ) -> Optional[str]:
        """Check if task limit (3) has been reached."""
        query = session.query(TaskQueue).filter(
            TaskQueue.content_id == content_id,
            TaskQueue.task_type == task_type,
            TaskQueue.status.in_([TASK_STATUS_PENDING, TASK_STATUS_PROCESSING, TASK_STATUS_FAILED])
        )

        # For transcribe tasks, filter by chunk_index
        if task_type == 'transcribe':
            chunk_index = input_data.get('chunk_index')
            if chunk_index is not None:
                query = query.filter(
                    text("input_data->>'chunk_index' = :chunk_index").params(chunk_index=str(chunk_index))
                )

        count = query.count()

        if count >= 3:
            # Get status breakdown for logging
            all_tasks = session.query(TaskQueue).filter(
                TaskQueue.content_id == content_id,
                TaskQueue.task_type == task_type
            )
            if task_type == 'transcribe' and input_data.get('chunk_index') is not None:
                all_tasks = all_tasks.filter(
                    text("input_data->>'chunk_index' = :chunk_index").params(
                        chunk_index=str(input_data['chunk_index'])
                    )
                )

            status_counts = {}
            for status in ['pending', 'processing', 'completed', 'failed']:
                cnt = all_tasks.filter(TaskQueue.status == status).count()
                if cnt > 0:
                    status_counts[status] = cnt

            status_summary = ', '.join([f"{s}: {c}" for s, c in status_counts.items()])

            if task_type == 'transcribe' and input_data.get('chunk_index') is not None:
                block_reason = f"Already have {count} {task_type} tasks for chunk {input_data['chunk_index']} ({status_summary})"
            else:
                block_reason = f"Already have {count} {task_type} tasks ({status_summary})"

            logger.warning(f"Skipping creation of {task_type} task for {content_id} - {block_reason.lower()}")
            return block_reason

        return None

    def _check_stitch_already_done(self, content: Content) -> Optional[str]:
        """Check if content is already stitched with current version."""
        if content and content.is_stitched:
            current_version = self.get_current_stitch_version()
            if not should_recreate_stitch_task(current_version, content.stitch_version):
                logger.debug(f"Content {content.content_id} already stitched with compatible version. "
                           f"{format_version_comparison_log(current_version, content.stitch_version)}")
                return f"Already stitched with compatible version {content.stitch_version}"
        return None

    def _check_permanent_failure(
        self,
        session: Session,
        content_id: str,
        task_type: str
    ) -> Optional[str]:
        """Check for permanently failed tasks."""
        failed_task = session.query(TaskQueue).filter(
            TaskQueue.content_id == content_id,
            TaskQueue.task_type == task_type,
            TaskQueue.status == 'failed'
        ).first()

        if not failed_task:
            return None

        error_str = str(failed_task.error or '')

        # For stitch tasks, only specific errors are permanent
        if task_type == 'stitch':
            if 'No audio file found' in error_str:
                logger.info(f"Stitch failed due to missing audio for {content_id} - recoverable")
                return None
            elif any(kw in error_str for kw in self.STITCH_PERMANENT_ERRORS):
                logger.info(f"Treating stitch failure as permanent for {content_id}")
                return f"Permanent failure: {error_str[:100]}"
            else:
                logger.info(f"Stitch failure for {content_id} is retryable")
                return None

        # For other tasks, check general permanent errors
        if any(kw in error_str for kw in self.PERMANENT_ERROR_PATTERNS):
            logger.info(f"Not creating new {task_type} task for {content_id} due to permanent failure")
            return f"Permanent failure: {error_str[:100]}"

        return None

    def _find_existing_task(
        self,
        session: Session,
        content_id: str,
        task_type: str,
        input_data: Dict[str, Any]
    ) -> Optional[TaskQueue]:
        """Find existing pending/processing/failed task."""
        query = session.query(TaskQueue).filter(
            TaskQueue.content_id == content_id,
            TaskQueue.task_type == task_type,
            TaskQueue.status.in_([TASK_STATUS_PENDING, TASK_STATUS_PROCESSING, TASK_STATUS_FAILED])
        )

        # For transcribe tasks, filter by chunk_index
        if task_type == 'transcribe':
            chunk_index = input_data.get('chunk_index')
            if chunk_index is not None:
                query = query.filter(
                    text("input_data->>'chunk_index' = :chunk_index").params(chunk_index=str(chunk_index))
                )
            else:
                logger.warning(f"Attempted to create transcribe task for {content_id} without chunk_index")

        return query.first()

    def _reset_failed_task(
        self,
        task: TaskQueue,
        session: Session,
        content_id: str,
        task_type: str,
        input_data: Dict[str, Any]
    ) -> Tuple[int, None]:
        """Reset a failed task for retry."""
        task.status = TASK_STATUS_PENDING
        task.error = None
        task.worker_id = None
        task.processor_task_id = None
        task.started_at = None
        task.completed_at = None
        session.flush()

        chunk_info = f"(chunk {input_data.get('chunk_index')}) " if task_type == 'transcribe' else ""
        logger.info(f"Reset failed task {task.id} ({task_type}) for {content_id} {chunk_info}"
                   f"to pending (attempt {task.attempts + 1}/{task.max_attempts})")

        return task.id, None

    def _create_new_task(
        self,
        session: Session,
        content_id: str,
        task_type: str,
        input_data: Dict[str, Any],
        priority: int,
        content: Content
    ) -> Tuple[Optional[int], Optional[str]]:
        """Create a new task."""
        # Calculate priority
        final_priority = self._calculate_priority(session, content_id, task_type, input_data, priority, content)

        new_task = TaskQueue(
            content_id=content_id,
            task_type=task_type,
            status=TASK_STATUS_PENDING,
            priority=final_priority,
            input_data=input_data,
            created_at=datetime.now(timezone.utc)
        )
        session.add(new_task)

        try:
            session.flush()
            task_id = new_task.id
            chunk_info = f"(chunk {input_data.get('chunk_index')})" if task_type == 'transcribe' else ""
            logger.info(f"Created new task: {task_type} (ID: {task_id}) for {content_id} {chunk_info} "
                       f"with priority {final_priority}")
            return task_id, None
        except IntegrityError as e:
            session.rollback()
            chunk_info = f"(chunk {input_data.get('chunk_index')}) " if task_type == 'transcribe' else ""
            logger.warning(f"Task {task_type} for {content_id} {chunk_info}already exists (race condition)")
            return None, "Task already exists (race condition)"

    def _calculate_priority(
        self,
        session: Session,
        content_id: str,
        task_type: str,
        input_data: Dict[str, Any],
        priority: int,
        content: Content
    ) -> int:
        """Calculate task priority based on project and date."""
        project_priority = 1  # Default

        active_projects_config = self.config.get('active_projects', {})

        # Get project from input_data or content
        project_str = input_data.get('project')
        if not project_str and content:
            project_str = content.projects
        elif not project_str:
            content_obj = session.query(Content).filter_by(content_id=content_id).first()
            project_str = content_obj.projects if content_obj else None

        # Handle multiple projects - use highest priority
        if project_str and isinstance(active_projects_config, dict):
            projects = project_str if isinstance(project_str, list) else [project_str]
            for project in projects:
                if project in active_projects_config:
                    project_config = active_projects_config.get(project)
                    if isinstance(project_config, dict):
                        proj_priority = project_config.get('priority', 1)
                        project_priority = max(project_priority, proj_priority)

        # Calculate priority
        if priority is None:
            content_obj = content or session.query(Content).filter_by(content_id=content_id).first()
            publish_date = getattr(content_obj, 'publish_date', None) if content_obj else None
            return calculate_priority_by_date(publish_date, project_priority)
        else:
            return priority + (project_priority * 1000000)

    def is_content_within_project_date_range(self, content: Content, project: str) -> bool:
        """Check if content's publish_date falls within the project's configured date range."""
        if not content.publish_date:
            return True

        try:
            project_config = self.config.get('active_projects', {}).get(project, {})
            if not project_config:
                return True

            if not project_config.get('enabled', True):
                logger.debug(f"Project {project} is disabled")
                return False

            start_date_str = project_config.get('start_date')
            end_date_str = project_config.get('end_date')

            from datetime import datetime, timezone, timedelta

            if start_date_str:
                start_date = datetime.strptime(start_date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
                if content.publish_date < start_date:
                    return False

            if end_date_str:
                end_date = datetime.strptime(end_date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
                end_date_exclusive = end_date + timedelta(days=1)
                if content.publish_date >= end_date_exclusive:
                    return False

            return True

        except Exception as e:
            logger.warning(f"Error checking date range for {content.content_id}: {e}")
            return True
