"""
State-Driven Handler - Uses state machine for cleaner task handling.

This demonstrates how the state machine approach simplifies:
1. Determining what to do next
2. Handling failures consistently
3. Recovery logic

Example usage:

    handler = StateDrivenHandler(config)

    # On task completion
    result = await handler.handle_task_completion(
        session, content, task_type, status, error_message
    )

    # result contains:
    # - current_state: ContentState.CONVERTED
    # - next_tasks: ['diarize', 'transcribe']
    # - recovery_action: None (no failure)
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from src.database.models import Content, TaskQueue, ContentChunk
from src.utils.logger import setup_worker_logger

from .content_state_machine import (
    ContentStateMachine,
    ContentState,
    FailureType,
    get_content_state
)
from .s3_content_checker import S3ContentChecker, ContentFileIndex

logger = setup_worker_logger('state_driven_handler')


class StateDrivenHandler:
    """
    Handles task results using state machine logic.

    This provides a cleaner, more declarative approach compared to
    the procedural if/else chains in the original pipeline_manager.
    """

    def __init__(self, config: Dict[str, Any], s3_checker: S3ContentChecker):
        self.config = config
        self.s3_checker = s3_checker
        self.state_machine = ContentStateMachine()

    async def handle_task_completion(
        self,
        session: Session,
        content: Content,
        task_type: str,
        status: str,
        error_message: Optional[str] = None,
        retry_count: int = 0
    ) -> Dict[str, Any]:
        """
        Handle a task completion using state machine logic.

        Returns:
            Dict with:
            - previous_state: State before handling
            - current_state: State after handling
            - next_tasks: List of tasks to create
            - recovery_action: Recovery action if failed
            - should_block: Whether to block content
        """
        # Get current state from DB flags
        previous_state = get_content_state(content)

        result = {
            'previous_state': previous_state,
            'current_state': previous_state,
            'next_tasks': [],
            'recovery_action': None,
            'should_block': False,
            'message': ''
        }

        # Handle failure
        if status == 'failed':
            return self._handle_failure(
                content, task_type, error_message, previous_state, retry_count, result
            )

        # Handle success - determine new state
        if status == 'completed':
            return await self._handle_success(
                session, content, task_type, previous_state, result
            )

        # Handle skipped - no state change
        if status == 'skipped':
            result['message'] = f"Task {task_type} skipped"
            result['next_tasks'] = self._get_next_tasks(content, previous_state)
            return result

        return result

    def _handle_failure(
        self,
        content: Content,
        task_type: str,
        error_message: str,
        current_state: ContentState,
        retry_count: int,
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle a failed task using state machine recovery logic."""

        # Classify the error
        failure_type = self.state_machine.classify_error(task_type, error_message or '')

        # Get recovery action
        recovery = self.state_machine.get_recovery_action(
            current_state, failure_type, retry_count
        )

        result['recovery_action'] = recovery['action']
        result['should_block'] = recovery['block']
        result['current_state'] = recovery['recovery_state']

        if recovery['should_retry']:
            result['message'] = f"Transient failure, will retry: {error_message[:100] if error_message else 'Unknown'}"
            # The recovery_state tells us what state to return to for retry
            result['next_tasks'] = [task_type]  # Retry same task
        elif recovery['block']:
            result['message'] = f"Permanent failure, blocking content: {error_message[:100] if error_message else 'Unknown'}"
            result['next_tasks'] = []
        else:
            result['message'] = f"Max retries exceeded: {error_message[:100] if error_message else 'Unknown'}"
            result['next_tasks'] = []

        logger.info(f"Failure handling for {content.content_id}: "
                   f"type={failure_type.name}, recovery={recovery['action']}, "
                   f"retry={recovery['should_retry']}, block={recovery['block']}")

        return result

    async def _handle_success(
        self,
        session: Session,
        content: Content,
        task_type: str,
        previous_state: ContentState,
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle successful task completion."""

        # Get the expected completion state for this task
        task_states = self.state_machine.TASK_STATE_MAP.get(task_type)
        if task_states:
            expected_completion_state = task_states[1]
        else:
            expected_completion_state = previous_state

        # Verify actual state by checking S3
        file_index = self.s3_checker.get_content_file_index(content.content_id)
        actual_state = self._determine_actual_state(content, file_index, session)

        result['current_state'] = actual_state

        # Check if state advanced as expected
        if actual_state == expected_completion_state:
            result['message'] = f"Task {task_type} completed, state: {actual_state.name}"
        elif actual_state.value > expected_completion_state.value:
            result['message'] = f"Task {task_type} completed, state advanced to: {actual_state.name}"
        else:
            # State didn't advance - might indicate a problem
            result['message'] = f"Task {task_type} completed but state unchanged: {actual_state.name}"
            logger.warning(f"State didn't advance after {task_type} for {content.content_id}: "
                          f"expected {expected_completion_state.name}, got {actual_state.name}")

        # Get next tasks based on current state
        result['next_tasks'] = self._get_next_tasks(content, actual_state)

        return result

    def _determine_actual_state(
        self,
        content: Content,
        file_index: ContentFileIndex,
        session: Session
    ) -> ContentState:
        """
        Determine actual state by combining DB flags with S3 reality.

        This reconciles what the DB says vs what actually exists.
        """
        # Check what actually exists in S3
        has_source = file_index.has_source_files or file_index.has_audio
        has_audio = file_index.has_audio
        has_diarization = file_index.has_diarization
        has_stitch = file_index.has_stitched_transcript
        has_manifest = file_index.has_storage_manifest

        # Check transcription state from chunks
        chunks = session.query(ContentChunk).filter_by(content_id=content.id).all()
        all_transcribed = bool(chunks) and all(c.transcription_status == 'completed' for c in chunks)

        # Determine state from actual files
        return self.state_machine.determine_state(
            is_downloaded=has_source,
            is_converted=has_audio and bool(file_index.get_chunk_indices_with_audio()),
            is_diarized=has_diarization,
            is_transcribed=all_transcribed,
            is_stitched=has_stitch,
            is_compressed=has_manifest,
            blocked_download=content.blocked_download,
        )

    def _get_next_tasks(self, content: Content, state: ContentState) -> List[str]:
        """Get the list of tasks that should be created for current state."""
        content_flags = {
            'is_downloaded': content.is_downloaded,
            'is_converted': content.is_converted,
            'is_diarized': content.is_diarized,
            'is_transcribed': content.is_transcribed,
            'is_stitched': content.is_stitched,
            'is_compressed': content.is_compressed,
            'diarization_ignored': content.meta_data.get('diarization_ignored') if content.meta_data else False,
        }
        return self.state_machine.get_required_tasks(state, content_flags)

    def get_recovery_recommendation(
        self,
        content: Content,
        failed_task_type: str,
        error_message: str,
        retry_count: int
    ) -> Dict[str, Any]:
        """
        Get a recommendation for how to recover from a failure.

        This can be used by operators to understand what went wrong
        and what the system will do about it.
        """
        current_state = get_content_state(content)
        failure_type = self.state_machine.classify_error(failed_task_type, error_message)
        recovery = self.state_machine.get_recovery_action(current_state, failure_type, retry_count)

        return {
            'content_id': content.content_id,
            'current_state': current_state.name,
            'failed_task': failed_task_type,
            'failure_type': failure_type.name,
            'error_summary': error_message[:200] if error_message else 'Unknown error',
            'retry_count': retry_count,
            'recommendation': {
                'action': recovery['action'],
                'recovery_state': recovery['recovery_state'].name,
                'will_retry': recovery['should_retry'],
                'will_block': recovery['block'],
            },
            'explanation': self._get_recovery_explanation(failure_type, recovery)
        }

    def _get_recovery_explanation(
        self,
        failure_type: FailureType,
        recovery: Dict[str, Any]
    ) -> str:
        """Generate human-readable explanation of recovery action."""
        explanations = {
            (FailureType.TRANSIENT, True):
                "This appears to be a temporary issue (network, timeout). "
                "The task will be retried automatically.",
            (FailureType.TRANSIENT, False):
                "This task has failed too many times. Manual intervention may be needed.",
            (FailureType.PERMANENT, False):
                "This is a permanent error (content unavailable, access denied). "
                "The content will be blocked from further processing.",
            (FailureType.PREREQUISITE, True):
                "A prerequisite file is missing. The system will attempt to "
                "regenerate it by going back to an earlier processing step.",
            (FailureType.PREREQUISITE, False):
                "A prerequisite file is missing and couldn't be regenerated. "
                "Manual investigation is needed.",
            (FailureType.RESOURCE, True):
                "The system ran out of resources (memory/disk). "
                "The task will be retried after resources are freed.",
        }

        key = (failure_type, recovery['should_retry'])
        return explanations.get(key, "Unknown recovery action.")


# Example usage in a simplified handle_task_result
async def simplified_handle_task_result(
    handler: StateDrivenHandler,
    session: Session,
    content: Content,
    task_type: str,
    status: str,
    result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Simplified task result handling using state machine.

    Compare this to the 500+ line handle_task_result in the original!
    """
    error_message = result.get('error') if status == 'failed' else None
    retry_count = result.get('retry_count', 0)

    # Use state machine to determine what to do
    outcome = await handler.handle_task_completion(
        session, content, task_type, status, error_message, retry_count
    )

    # Apply state changes
    if outcome['should_block']:
        content.blocked_download = True
        content.last_updated = datetime.now(timezone.utc)
        session.add(content)
        session.commit()

    # Return outcome for orchestrator
    return {
        'outcome_message': outcome['message'],
        'next_tasks': outcome['next_tasks'],
        'state': outcome['current_state'].name,
        'blocked': outcome['should_block']
    }
