"""
Task Result Handler - Handles task completion and failure outcomes.

This module processes the results of completed tasks and determines:
- What database updates to make
- Whether to block content
- Whether to pause workers
- What error handling policies to apply
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional

from sqlalchemy.orm import Session

from src.database.models import TaskQueue, Content, ContentChunk, Sentence
from src.utils.error_codes import ErrorCode, get_error_category
from src.processing.error_handler import ErrorHandler
from src.utils.human_behavior import HumanBehaviorManager
from src.utils.logger import setup_worker_logger
from src.utils.chunk_utils import store_chunk_plan

logger = setup_worker_logger('result_handler')

# Task status constants
TASK_STATUS_PENDING = 'pending'
TASK_STATUS_PROCESSING = 'processing'
TASK_STATUS_COMPLETED = 'completed'
TASK_STATUS_FAILED = 'failed'


class TaskResultHandler:
    """
    Handles the outcome of task completion or failure.

    Responsibilities:
    - Process successful task completions
    - Handle task failures with appropriate error policies
    - Update content state based on task results
    - Determine worker availability after task completion
    """

    # Error patterns that should NOT trigger task recreation
    PERMANENT_ERROR_PATTERNS = [
        'Join this channel to get access to members-only content',
        'Sign in to confirm your age',
        'Video unavailable',
        'This video is private',
        'Private video',
        'This live event will begin in',
        'This live stream has ended',
        'Premiere will begin shortly',
        'Bad URL',
        'HTTP Error 400: Bad Request',
        'HTTP Error 403: Forbidden',
        'Failed to extract audio from source file'
    ]

    # Transient errors that SHOULD trigger recreation
    RECREATE_ERROR_PATTERNS = {
        "*": [  # For all task types
            "Could not connect to the endpoint URL",
            "Connect timeout on endpoint URL",
            "S3 connection failed",
            "Failed to connect to S3",
            "endpoint URL"
        ],
        "download_youtube": [
            "Network is unreachable",
            "Connection reset by peer",
            "Connection timed out",
            "Connection refused",
            "No route to host",
            "Name or service not known",
            "Temporary failure in name resolution",
            "Network error",
            "URLError",
            "ConnectionError",
            "TimeoutError",
            "OSError",
            "socket.gaierror"
        ],
        "download_rumble": [
            "Network is unreachable",
            "Connection reset by peer",
            "Connection timed out",
            "Connection refused",
            "No route to host"
        ],
        "download_podcast": [
            "Network is unreachable",
            "Connection reset by peer",
            "Connection timed out",
            "Connection refused",
            "No route to host"
        ]
    }

    def __init__(
        self,
        config: Dict[str, Any],
        error_handler: ErrorHandler,
        behavior_manager: Optional[HumanBehaviorManager] = None
    ):
        """
        Initialize TaskResultHandler.

        Args:
            config: Application configuration dict
            error_handler: ErrorHandler instance for policy-based error handling
            behavior_manager: Optional HumanBehaviorManager for download tasks
        """
        self.config = config
        self.error_handler = error_handler
        self.behavior_manager = behavior_manager

    def get_current_stitch_version(self) -> str:
        """Get the current stitch version from config."""
        try:
            return self.config.get('processing', {}).get('stitch', {}).get('current_version', 'stitch_v1')
        except Exception:
            return 'stitch_v1'

    async def handle_task_result(
        self,
        session: Session,
        content_id: str,
        task_type: str,
        status: str,
        result: Dict[str, Any],
        db_task: TaskQueue,
        content: Content,
        worker_id: str,
        chunk_index: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Handle task completion and determine next steps.

        Args:
            session: Active database session
            content_id: Content ID
            task_type: Type of task that completed
            status: Task status (completed, failed, skipped)
            result: Task result dictionary
            db_task: Database task object
            content: Content object
            worker_id: Worker ID
            chunk_index: Optional chunk index for transcribe tasks

        Returns:
            Dict containing:
                - available_after: When worker should be available again
                - outcome_message: Description of actions taken
                - skip_state_audit: Whether to skip state evaluation
        """
        outcome = {
            'available_after': None,
            'outcome_message': "No outcome message set",
            'skip_state_audit': False
        }

        try:
            # EARLY CHECK: If content is blocked, skip ALL processing
            if content.blocked_download:
                logger.info(f"Content {content_id} is blocked. Skipping pipeline processing.")
                outcome['outcome_message'] = "Content blocked - no further processing"
                outcome['skip_state_audit'] = True
                return outcome

            # Extract error information
            error_code = result.get('error_code') if isinstance(result, dict) else None
            error_message = str(result.get('error', '')) if isinstance(result, dict) else str(result)
            error_details = result.get('error_details', {}) if isinstance(result, dict) else {}

            # Handle failures
            if status in [TASK_STATUS_FAILED, 'failed_permanent']:
                return await self._handle_failure(
                    session, content_id, task_type, result, db_task, content,
                    worker_id, error_code, error_message, error_details, outcome
                )

            # Handle successful completions
            if status == TASK_STATUS_COMPLETED:
                return await self._handle_completion(
                    session, content_id, task_type, result, content, outcome
                )

            # For other statuses (skipped, etc.), just return outcome
            outcome['outcome_message'] = f"Task {task_type} completed with status {status}"
            return outcome

        except Exception as e:
            logger.error(f"Error handling task result: {e}", exc_info=True)
            return {
                'available_after': None,
                'outcome_message': f"Error handling task result: {e}",
                'skip_state_audit': False
            }

    async def _handle_failure(
        self,
        session: Session,
        content_id: str,
        task_type: str,
        result: Dict[str, Any],
        db_task: TaskQueue,
        content: Content,
        worker_id: str,
        error_code: Optional[str],
        error_message: str,
        error_details: Dict[str, Any],
        outcome: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle a failed task."""

        # Use ErrorHandler for structured errors
        if error_code:
            error_result = self.error_handler.handle_error(
                task_type=task_type,
                error_code=error_code,
                error_message=error_message,
                error_details=error_details,
                content=content,
                session=session,
                worker_id=worker_id
            )

            action = error_result.get('action')

            if action == 'recreate_task':
                self._reset_task_to_pending(db_task, session)
                outcome['outcome_message'] = error_result['outcome_message']
                return outcome

            elif action == 'block_content':
                outcome['outcome_message'] = error_result['outcome_message']
                outcome['skip_state_audit'] = True
                return outcome

            elif action == 'pause_worker':
                outcome['outcome_message'] = error_result['outcome_message']
                outcome['available_after'] = error_result.get('available_after')
                outcome['skip_state_audit'] = True
                return outcome

            elif action == 'mark_ignored':
                outcome['outcome_message'] = error_result['outcome_message']
                outcome['skip_state_audit'] = True
                return outcome

            elif error_result.get('skip_state_audit'):
                outcome['outcome_message'] = error_result['outcome_message']
                outcome['skip_state_audit'] = True
                return outcome

            # For 'trigger_prerequisites' and 'continue', fall through to state evaluation

        # Legacy error handling for tasks not using error codes
        else:
            legacy_result = self._handle_legacy_failure(
                session, content_id, task_type, result, db_task, content,
                worker_id, error_message, outcome
            )
            if legacy_result:
                return legacy_result

        # If no specific handling, allow state evaluation to proceed
        return outcome

    def _handle_legacy_failure(
        self,
        session: Session,
        content_id: str,
        task_type: str,
        result: Dict[str, Any],
        db_task: TaskQueue,
        content: Content,
        worker_id: str,
        error_message: str,
        outcome: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Handle legacy failures without error codes."""

        # Check if this is a transient error we should retry
        if self._should_recreate_task(task_type, error_message):
            # Check retry limit
            failed_count = session.query(TaskQueue).filter(
                TaskQueue.content_id == content_id,
                TaskQueue.task_type == task_type,
                TaskQueue.status == TASK_STATUS_FAILED
            ).count()

            if failed_count >= 3:
                logger.warning(f"Task {task_type} for {content_id} has {failed_count} failed attempts. Not recreating.")
                outcome['outcome_message'] = f"Task {task_type} failed permanently after {failed_count} attempts"
                return outcome

            logger.warning(f"Task {task_type} for {content_id} failed with transient error. Recreating (attempt {failed_count + 1}).")
            self._reset_task_to_pending(db_task, session)
            outcome['outcome_message'] = f"Task {task_type} reset for retry (attempt {failed_count + 1})"
            return outcome

        # Handle permanent failures
        if result.get('permanent', False):
            return self._handle_permanent_failure(
                session, content_id, task_type, result, content, outcome
            )

        # No special handling needed
        return None

    def _should_recreate_task(self, task_type: str, error_message: str) -> bool:
        """Check if error should trigger task recreation."""
        # Check for permanent errors first
        if "Failed to download audio chunk" in error_message:
            return False

        if any(pattern in error_message for pattern in self.PERMANENT_ERROR_PATTERNS):
            logger.info(f"Not recreating {task_type} due to permanent error: {error_message[:100]}")
            return False

        # Don't recreate conda run failures
        if "conda run" in error_message and "failed" in error_message:
            return False

        # Don't recreate most stitch failures
        if task_type == "stitch" and "No audio file found" not in error_message:
            # Check for S3 connection errors (should retry)
            s3_errors = ["Could not connect to the endpoint URL", "Connect timeout on endpoint URL"]
            if any(err in error_message for err in s3_errors):
                return True
            return False

        # Check task-specific and wildcard patterns
        patterns = self.RECREATE_ERROR_PATTERNS.get(task_type, []) + self.RECREATE_ERROR_PATTERNS.get("*", [])
        return any(pattern in error_message for pattern in patterns)

    def _handle_permanent_failure(
        self,
        session: Session,
        content_id: str,
        task_type: str,
        result: Dict[str, Any],
        content: Content,
        outcome: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle a permanent failure."""
        if task_type in ["download_youtube", "download_podcast", "download_rumble"]:
            logger.warning(f"BLOCKING content {content_id} due to permanent download failure")
            content.blocked_download = True

            meta_data = dict(content.meta_data) if content.meta_data else {}
            meta_data.update({
                'block_reason': result.get('error', 'Permanent download failure'),
                'permanent_block': 'true',
                'blocked_at': datetime.now(timezone.utc).isoformat(),
                'error_details': {'permanent': True}
            })
            content.meta_data = meta_data
            content.last_updated = datetime.now(timezone.utc)
            session.add(content)
            session.commit()

            outcome['outcome_message'] = f"Content blocked: {result.get('error', 'Permanent download failure')}"
            outcome['skip_state_audit'] = True

        else:
            # Use error handler for non-download permanent failures
            error_result = self.error_handler.handle_error(
                task_type=task_type,
                error_code=ErrorCode.UNKNOWN_ERROR.value,
                error_message=result.get('error', 'Unknown permanent error'),
                error_details={'permanent': True},
                content=content,
                session=session,
                worker_id=None
            )
            outcome['outcome_message'] = error_result['outcome_message']
            if error_result.get('skip_state_audit'):
                outcome['skip_state_audit'] = True

        return outcome

    async def _handle_completion(
        self,
        session: Session,
        content_id: str,
        task_type: str,
        result: Dict[str, Any],
        content: Content,
        outcome: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle successful task completion."""

        if task_type == "convert":
            return self._handle_convert_completion(session, content_id, result, content, outcome)

        elif task_type == "stitch":
            return await self._handle_stitch_completion(session, content_id, content, outcome)

        elif task_type == "cleanup":
            return self._handle_cleanup_completion(session, content, outcome)

        # For other task types, just proceed to state evaluation
        outcome['outcome_message'] = f"Task {task_type} completed successfully"
        return outcome

    def _handle_convert_completion(
        self,
        session: Session,
        content_id: str,
        result: Dict[str, Any],
        content: Content,
        outcome: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle convert task completion."""
        chunk_plan = result.get('chunk_plan', [])
        if chunk_plan:
            logger.info(f"Convert completed for {content_id} with {len(chunk_plan)} chunks")

            formatted_chunks = [
                {
                    'index': c.get('index'),
                    'start_time': c.get('start_time'),
                    'end_time': c.get('end_time'),
                    'duration': c.get('duration')
                }
                for c in chunk_plan
            ]

            if store_chunk_plan(session, content, formatted_chunks):
                logger.info(f"Stored {len(formatted_chunks)} chunks for {content_id}")
                content.last_updated = datetime.now(timezone.utc)
                session.add(content)
                session.commit()
                session.refresh(content)
            else:
                logger.error(f"Failed to store chunk plan for {content_id}")
        else:
            logger.warning(f"Convert completed but no chunk_plan for {content_id}")

        outcome['outcome_message'] = f"Convert completed with {len(chunk_plan)} chunks"
        return outcome

    async def _handle_stitch_completion(
        self,
        session: Session,
        content_id: str,
        content: Content,
        outcome: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle stitch task completion."""
        current_version = self.get_current_stitch_version()

        # Commit and refresh to see changes from stitch process
        session.commit()
        session.expire_all()
        content = session.query(Content).filter_by(content_id=content_id).first()

        if not content:
            outcome['outcome_message'] = f"Content {content_id} not found after refresh"
            return outcome

        # Check for sentences with current version
        sentence_count = session.query(Sentence).filter(
            Sentence.content_id == content.id,
            Sentence.stitch_version == current_version
        ).count()

        logger.info(f"Found {sentence_count} sentences with version {current_version} for {content_id}")

        if sentence_count > 0:
            if not content.is_stitched or content.stitch_version != current_version:
                content.is_stitched = True
                content.stitch_version = current_version
                content.last_updated = datetime.now(timezone.utc)
                session.add(content)
                session.commit()
                session.refresh(content)
                logger.info(f"Updated stitch flags for {content_id}")

            outcome['outcome_message'] = f"Stitch completed with {sentence_count} sentences"
        else:
            logger.warning(f"Stitch completed but no sentences found for {content_id}")
            outcome['outcome_message'] = "Stitch completed but database save failed"
            outcome['skip_state_audit'] = True

        return outcome

    def _handle_cleanup_completion(
        self,
        session: Session,
        content: Content,
        outcome: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle cleanup task completion."""
        logger.info(f"Cleanup completed for {content.content_id}")
        content.is_compressed = True
        content.last_updated = datetime.now(timezone.utc)
        session.add(content)
        session.commit()

        outcome['outcome_message'] = "Cleanup completed - marked as compressed"
        outcome['skip_state_audit'] = True
        return outcome

    def _reset_task_to_pending(self, task: TaskQueue, session: Session):
        """Reset a task back to pending status."""
        task.status = TASK_STATUS_PENDING
        task.worker_id = None
        task.started_at = None
        task.completed_at = None
        task.error = None
        session.add(task)
        session.commit()

    def calculate_worker_available_after(
        self,
        worker_id: str,
        task_type: str,
        skip_behavior_wait: bool = False
    ) -> Optional[datetime]:
        """Calculate when worker should be available for next task."""
        # Only apply behavior management to download tasks
        if task_type not in ["download_youtube", "download_rumble"]:
            return None

        if skip_behavior_wait:
            logger.info(f"Worker {worker_id} skipping behavior wait (download skipped/cached)")
            return None

        if not self.behavior_manager:
            logger.warning("BehaviorManager not available")
            return None

        if worker_id not in self.behavior_manager.worker_states:
            logger.warning(f"Worker {worker_id} not in behavior manager states")
            return None

        now_utc = datetime.now(timezone.utc)

        # Update behavior state
        self.behavior_manager.handle_task_completion(worker_id, task_type, current_time_override=now_utc)

        # Calculate wait times
        behavior_wait, reason = self.behavior_manager.calculate_next_task_wait_time(worker_id, task_type)
        post_task_delay = self.behavior_manager.calculate_post_task_delay(worker_id, task_type)

        wait_duration = max(behavior_wait, post_task_delay)
        logger.info(f"Worker {worker_id} post-completion delays: behavior={behavior_wait:.1f}s ({reason}), "
                   f"post_task={post_task_delay:.1f}s => Max: {wait_duration:.1f}s")

        if wait_duration > 0:
            return now_utc + timedelta(seconds=wait_duration)

        return None
