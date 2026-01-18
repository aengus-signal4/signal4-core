"""
Pipeline Manager V3 - Modular Coordinator for Content Processing.

This is a thin coordinator that delegates to specialized modules:
- StateReconciler: Ensures database matches S3 reality
- TaskFactory: Creates pipeline tasks
- ErrorHandler: Handles task failures

This is a drop-in replacement for the legacy PipelineManager, using
the same interface but with modular implementation.
"""

from typing import Dict, Any, Optional
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import logging

from sqlalchemy.orm import Session

from src.database.models import TaskQueue, Content, ContentChunk, Sentence
from src.processing.content_state.file_checker import ContentFileChecker
from src.processing.content_state.flag_updater import FlagUpdater
from src.processing.content_state.reconciler import StateReconciler
from src.processing.task_creation.factory import TaskFactory
from src.processing.error_handler import ErrorHandler
from src.utils.human_behavior import HumanBehaviorManager
from src.utils.error_codes import ErrorCode
from src.utils.chunk_utils import store_chunk_plan
from src.storage.s3_utils import S3Storage, S3StorageConfig

logger = logging.getLogger(__name__)

# Task status constants
TASK_STATUS_PENDING = 'pending'
TASK_STATUS_PROCESSING = 'processing'
TASK_STATUS_COMPLETED = 'completed'
TASK_STATUS_FAILED = 'failed'


class PipelineManagerV3:
    """
    Coordinates content processing workflow after task completion.

    This is the modular version of PipelineManager that delegates to:
    - ContentFileChecker: S3 file existence checks
    - FlagUpdater: Database flag updates
    - StateReconciler: State reconciliation
    - TaskFactory: Task creation
    - ErrorHandler: Error handling

    Responsibilities of this coordinator:
    - Route task results to appropriate handlers
    - Coordinate state reconciliation after task completion
    - Calculate worker availability times based on behavior rules
    """

    VERSION = "3.0.0"

    def __init__(
        self,
        behavior_manager: HumanBehaviorManager,
        config: Dict[str, Any],
        worker_task_failures: defaultdict
    ):
        """Initialize the PipelineManagerV3.

        Args:
            behavior_manager: Instance of HumanBehaviorManager
            config: The main application configuration dictionary
            worker_task_failures: Reference to the orchestrator's failure tracking dict
        """
        self.behavior_manager = behavior_manager
        self.config = config
        self.worker_task_failures = worker_task_failures

        # Extract specific config values
        self.youtube_auth_pause_timeout = self.config.get('processing', {}).get(
            'youtube_auth_pause_duration_seconds', 50400
        )
        self.max_consecutive_failures = self.config.get('processing', {}).get(
            'max_consecutive_failures', 3
        )

        # Initialize S3 storage
        s3_config = S3StorageConfig(
            endpoint_url=config['storage']['s3']['endpoint_url'],
            access_key=config['storage']['s3']['access_key'],
            secret_key=config['storage']['s3']['secret_key'],
            bucket_name=config['storage']['s3']['bucket_name'],
            use_ssl=config['storage']['s3']['use_ssl']
        )
        self.s3_storage = S3Storage(s3_config)

        # Initialize modular components
        self.file_checker = ContentFileChecker()
        self.flag_updater = FlagUpdater(config, self.s3_storage)
        self.task_factory = TaskFactory(config)
        self.error_handler = ErrorHandler(config, worker_task_failures)

        self.state_reconciler = StateReconciler(
            config=config,
            file_checker=self.file_checker,
            flag_updater=self.flag_updater,
            task_factory=self.task_factory,
            s3_storage=self.s3_storage
        )

        logger.info(f"PipelineManagerV3 v{self.VERSION} initialized with modular components")

    def get_current_stitch_version(self) -> str:
        """Get current stitch version from config."""
        return self.flag_updater.get_current_stitch_version()

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

        This method:
        1. Handles special cases (auth errors, permanent failures)
        2. Handles successful task completion with special processing
        3. Evaluates actual content state via StateReconciler
        4. Updates database flags to match reality
        5. Creates appropriate next tasks based on actual state

        Args:
            session: Active database session
            content_id: Content ID
            task_type: Type of task that completed
            status: Task status (completed, failed, skipped)
            result: Task result dictionary
            db_task: Database task object
            content: Content object
            worker_id: Worker ID
            chunk_index: Optional chunk index for chunk-specific tasks

        Returns:
            Dict containing:
                - available_after: When worker should be available again
                - outcome_message: Description of actions taken
        """
        try:
            outcome = {
                'available_after': None,
                'outcome_message': "No outcome message set"
            }

            # EARLY CHECK: If content is blocked, skip ALL processing
            if content.blocked_download:
                logger.info(f"Content {content_id} is blocked. Skipping pipeline processing.")
                outcome['outcome_message'] = "Content blocked - no further processing"
                return outcome

            # Extract error info
            error_code = result.get('error_code') if isinstance(result, dict) else None
            error_message = str(result.get('error', '')) if isinstance(result, dict) else str(result)
            error_details = result.get('error_details', {}) if isinstance(result, dict) else {}

            # Handle failures
            if status in (TASK_STATUS_FAILED, 'failed_permanent'):
                error_result = await self._handle_failure(
                    session=session,
                    content=content,
                    task_type=task_type,
                    error_code=error_code,
                    error_message=error_message,
                    error_details=error_details,
                    result=result,
                    db_task=db_task,
                    worker_id=worker_id,
                    outcome=outcome
                )
                if error_result.get('handled'):
                    return error_result

            # Handle successful completion with special processing
            if status == TASK_STATUS_COMPLETED:
                completion_result = await self._handle_completion(
                    session=session,
                    content=content,
                    content_id=content_id,
                    task_type=task_type,
                    result=result,
                    outcome=outcome
                )
                if completion_result.get('handled'):
                    return completion_result

            # For all other cases, reconcile state
            reconcile_result = await self.state_reconciler.reconcile(session, content)

            # Calculate worker availability
            skip_behavior_wait = self._should_skip_behavior_wait(task_type, status, result)
            outcome['available_after'] = self._calculate_worker_available_after(
                worker_id, task_type, skip_behavior_wait
            )

            # Build outcome message
            outcome['outcome_message'] = self._build_outcome_message(
                reconcile_result, task_type, chunk_index, session, content
            )

            return outcome

        except Exception as e:
            logger.error(f"Error handling task result: {str(e)}", exc_info=True)
            return {
                'available_after': None,
                'outcome_message': f"Error handling task result: {str(e)}"
            }

    async def _handle_failure(
        self,
        session: Session,
        content: Content,
        task_type: str,
        error_code: Optional[str],
        error_message: str,
        error_details: Dict[str, Any],
        result: Dict[str, Any],
        db_task: TaskQueue,
        worker_id: str,
        outcome: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle task failure via ErrorHandler."""
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
                db_task.status = TASK_STATUS_PENDING
                db_task.worker_id = None
                db_task.started_at = None
                db_task.completed_at = None
                db_task.error = None
                session.add(db_task)
                session.commit()
                return {
                    'handled': True,
                    'available_after': None,
                    'outcome_message': error_result['outcome_message']
                }

            elif action == 'block_content':
                return {
                    'handled': True,
                    'available_after': None,
                    'outcome_message': error_result['outcome_message']
                }

            elif action == 'pause_worker':
                return {
                    'handled': True,
                    'available_after': error_result.get('available_after'),
                    'outcome_message': error_result['outcome_message']
                }

            elif action == 'mark_ignored':
                return {
                    'handled': True,
                    'available_after': None,
                    'outcome_message': error_result['outcome_message']
                }

            elif error_result.get('skip_state_audit', False):
                return {
                    'handled': True,
                    'available_after': None,
                    'outcome_message': error_result['outcome_message']
                }

        # Legacy error handling for tasks not using error codes
        return await self._handle_legacy_failure(
            session, content, task_type, error_message, result, db_task, worker_id, outcome
        )

    async def _handle_legacy_failure(
        self,
        session: Session,
        content: Content,
        task_type: str,
        error_message: str,
        result: Dict[str, Any],
        db_task: TaskQueue,
        worker_id: str,
        outcome: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle legacy error patterns (for backward compatibility)."""
        from src.processing.pipeline_manager import PipelineManager

        # Check if should recreate task
        if PipelineManager.should_recreate_task(task_type, error_message):
            failed_count = session.query(TaskQueue).filter(
                TaskQueue.content_id == content.content_id,
                TaskQueue.task_type == task_type,
                TaskQueue.status == TASK_STATUS_FAILED
            ).count()

            if failed_count >= 3:
                logger.warning(
                    f"Task {task_type} for {content.content_id} has {failed_count} failures. "
                    "Not recreating."
                )
                return {'handled': False}

            logger.warning(
                f"Task {task_type} for {content.content_id} failed. "
                f"Recreating (attempt {failed_count + 1})."
            )
            db_task.status = TASK_STATUS_PENDING
            db_task.worker_id = None
            db_task.started_at = None
            db_task.completed_at = None
            db_task.error = None
            session.add(db_task)
            session.commit()
            return {
                'handled': True,
                'available_after': None,
                'outcome_message': f"Task reset to pending for retry (attempt {failed_count + 1})"
            }

        return {'handled': False}

    async def _handle_completion(
        self,
        session: Session,
        content: Content,
        content_id: str,
        task_type: str,
        result: Dict[str, Any],
        outcome: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle successful task completion with special processing."""
        # Convert task: store chunk plan
        if task_type == "convert" and isinstance(result, dict):
            chunk_plan = result.get('chunk_plan', [])
            if chunk_plan:
                logger.debug(f"Convert completed for {content_id} with {len(chunk_plan)} chunks")
                formatted_chunks = [
                    {
                        'index': chunk.get('index'),
                        'start_time': chunk.get('start_time'),
                        'end_time': chunk.get('end_time'),
                        'duration': chunk.get('duration')
                    }
                    for chunk in chunk_plan
                ]
                if store_chunk_plan(session, content, formatted_chunks):
                    logger.debug(f"Stored {len(formatted_chunks)} chunks for {content_id}")
                    content.last_updated = datetime.now(timezone.utc)
                    session.add(content)
                    session.commit()
                    session.refresh(content)

        # Stitch task: verify and update flags
        elif task_type == "stitch":
            current_version = self.get_current_stitch_version()

            # Commit and refresh to see stitch process changes
            session.commit()
            session.expire_all()
            content = session.query(Content).filter_by(content_id=content_id).first()

            if not content:
                return {
                    'handled': True,
                    'available_after': None,
                    'outcome_message': f"Content {content_id} not found after refresh"
                }

            # Check for Sentence records
            sentence_count = session.query(Sentence).filter(
                Sentence.content_id == content.id,
                Sentence.stitch_version == current_version
            ).count()

            if sentence_count > 0:
                if not content.is_stitched or content.stitch_version != current_version:
                    logger.debug(
                        f"Stitch completed - found {sentence_count} sentences, "
                        f"updating flags for {content_id}"
                    )
                    content.is_stitched = True
                    content.stitch_version = current_version
                    content.last_updated = datetime.now(timezone.utc)
                    session.add(content)
                    session.commit()
                    session.refresh(content)
            else:
                logger.warning(
                    f"Stitch completed but no Sentence records for {content_id} "
                    f"with version {current_version}"
                )
                return {
                    'handled': True,
                    'available_after': None,
                    'outcome_message': "Stitch completed but database save failed"
                }

        # Cleanup task: mark compressed and skip further processing
        elif task_type == "cleanup":
            logger.debug(f"Cleanup completed for {content_id}")
            content.is_compressed = True
            content.last_updated = datetime.now(timezone.utc)
            session.add(content)
            session.commit()
            return {
                'handled': True,
                'available_after': None,
                'outcome_message': "Cleanup completed - marked as compressed"
            }

        return {'handled': False}

    def _should_skip_behavior_wait(
        self,
        task_type: str,
        status: str,
        result: Dict[str, Any]
    ) -> bool:
        """Determine if behavior wait should be skipped."""
        if task_type not in ["download_youtube", "download_rumble"]:
            return True

        return (
            (status == "skipped" and result.get('skipped_existing', False)) or
            (status == "skipped" and result.get('reason') == 'already_exists') or
            result.get('skip_wait_time', False)
        )

    def _calculate_worker_available_after(
        self,
        worker_id: str,
        task_type: str,
        skip_behavior_wait: bool
    ) -> Optional[datetime]:
        """Calculate when worker should be available next."""
        if task_type not in ["download_youtube", "download_rumble"]:
            return None

        if skip_behavior_wait:
            logger.debug(f"Worker {worker_id} skipping behavior wait for {task_type}")
            return None

        if not self.behavior_manager:
            logger.warning("BehaviorManager not available")
            return None

        if worker_id not in self.behavior_manager.worker_states:
            logger.warning(f"Worker {worker_id} not in behavior manager states")
            return None

        now_utc = datetime.now(timezone.utc)

        # Update behavior state
        self.behavior_manager.handle_task_completion(
            worker_id, task_type, current_time_override=now_utc
        )

        # Calculate wait times
        behavior_wait, reason = self.behavior_manager.calculate_next_task_wait_time(
            worker_id, task_type
        )
        post_task_delay = self.behavior_manager.calculate_post_task_delay(worker_id, task_type)

        wait_duration = max(behavior_wait, post_task_delay)
        logger.debug(
            f"Worker {worker_id} delays: behavior={behavior_wait:.1f}s ({reason}), "
            f"post_task={post_task_delay:.1f}s => Max: {wait_duration:.1f}s"
        )

        if wait_duration > 0:
            return now_utc + timedelta(seconds=wait_duration)
        return None

    def _build_outcome_message(
        self,
        reconcile_result,
        task_type: str,
        chunk_index: Optional[int],
        session: Session,
        content: Content
    ) -> str:
        """Build human-readable outcome message."""
        if reconcile_result.errors:
            return f"Errors: {'; '.join(reconcile_result.errors)}"

        updates = []
        if reconcile_result.flags_updated:
            updates.append(f"Updated {', '.join(reconcile_result.flags_updated)}")
        if reconcile_result.tasks_created:
            updates.append(f"Created {', '.join(reconcile_result.tasks_created)}")
        if reconcile_result.tasks_blocked and task_type != 'transcribe':
            updates.append(f"Blocked {', '.join(reconcile_result.tasks_blocked)}")

        if not updates:
            if task_type == 'transcribe':
                chunks = session.query(ContentChunk).filter_by(content_id=content.id).all()
                pending = [str(c.chunk_index) for c in chunks if c.transcription_status != 'completed']
                if pending:
                    return f"Verified chunk {chunk_index}, waiting for chunks {','.join(pending)}"
                else:
                    return f"Verified chunk {chunk_index}, all chunks complete"
            else:
                return f"Verified {task_type} completion"

        return '. '.join(updates)

    async def evaluate_content_state(
        self,
        session: Session,
        content: Content,
        db_task: Optional[TaskQueue] = None
    ) -> Dict[str, Any]:
        """
        Backward-compatible wrapper for state reconciliation.

        Delegates to StateReconciler.reconcile().
        """
        result = await self.state_reconciler.reconcile(session, content)
        return result.to_dict()

    async def bulk_reconcile_content_states(
        self,
        session: Session
    ) -> Dict[str, Any]:
        """
        Bulk reconcile all content states.

        This is a simplified version that uses the modular components.
        For full bulk reconciliation, use the legacy PipelineManager method.
        """
        # For now, delegate to individual reconciliation
        # This could be optimized with batch processing
        from src.database.models import Content

        results = {
            'total_content': 0,
            'updated_content': 0,
            'flag_updates': defaultdict(int),
            'errors': [],
            'processing_time': 0
        }

        start_time = datetime.now()

        try:
            all_content = session.query(Content).all()
            results['total_content'] = len(all_content)

            for content in all_content:
                try:
                    result = await self.state_reconciler.reconcile(
                        session, content, skip_task_creation=True
                    )
                    if result.flags_updated:
                        results['updated_content'] += 1
                        for flag in result.flags_updated:
                            results['flag_updates'][flag] += 1
                    if result.errors:
                        results['errors'].extend(result.errors)
                except Exception as e:
                    results['errors'].append(f"Error processing {content.content_id}: {str(e)}")

            results['processing_time'] = (datetime.now() - start_time).total_seconds()

            logger.info(
                f"Bulk reconciliation complete: {results['total_content']} items, "
                f"{results['updated_content']} updated in {results['processing_time']:.2f}s"
            )

        except Exception as e:
            logger.error(f"Error in bulk reconciliation: {e}", exc_info=True)
            results['errors'].append(str(e))

        return results
