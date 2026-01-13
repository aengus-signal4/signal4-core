"""
Pipeline Manager v2 - Refactored, modular pipeline management.

This is the new, refactored version of PipelineManager that composes
smaller, focused modules instead of being a monolith.

The original pipeline_manager.py remains as the production version.
This file (pipeline_manager_v2.py) is the new implementation that
can be tested and validated before replacing the original.

Usage:
    # Option 1: Use the new modular components directly
    from src.processing.state import S3ContentChecker, FlagReconciler
    from src.processing.pipeline import TaskCreator, TaskResultHandler, StateEvaluator

    # Option 2: Use the facade (drop-in replacement for PipelineManager)
    from src.processing.pipeline_manager_v2 import PipelineManagerV2

Architecture:
    PipelineManagerV2
    ├── StateEvaluator (evaluate_content_state, check_and_create_stitch_task)
    │   ├── S3ContentChecker (file existence checks)
    │   ├── FlagReconciler (database flag updates)
    │   └── TaskCreator (task creation logic)
    ├── TaskResultHandler (handle_task_result)
    │   └── ErrorHandler (error policy handling)
    └── HumanBehaviorManager (worker behavior for downloads)
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, DefaultDict
from collections import defaultdict

from sqlalchemy.orm import Session
from minio import Minio

from src.database.models import TaskQueue, Content, ContentChunk
from src.utils.human_behavior import HumanBehaviorManager
from src.utils.logger import setup_worker_logger
from src.storage.s3_utils import S3Storage, S3StorageConfig
from src.processing.error_handler import ErrorHandler

# Import the new modular components
from src.processing.state import S3ContentChecker, FlagReconciler
from src.processing.pipeline import TaskCreator, TaskResultHandler, StateEvaluator

logger = setup_worker_logger('pipeline_manager_v2')


class PipelineManagerV2:
    """
    Refactored Pipeline Manager - drop-in replacement for PipelineManager.

    This class provides the same public interface as the original PipelineManager
    but delegates to focused, modular components internally.

    Key improvements:
    - Smaller, testable components
    - Shared S3 checking logic (no more duplication)
    - Cleaner separation of concerns
    - Easier to understand and maintain
    """

    VERSION = "2.1.0"  # Modular refactored version

    def __init__(
        self,
        behavior_manager: HumanBehaviorManager,
        config: Dict,
        worker_task_failures: DefaultDict[str, Dict[str, Any]]
    ):
        """
        Initialize PipelineManagerV2.

        Args:
            behavior_manager: Instance of HumanBehaviorManager
            config: Main application configuration dictionary
            worker_task_failures: Reference to orchestrator's failure tracking dict
        """
        self.behavior_manager = behavior_manager
        self.config = config
        self.worker_task_failures = worker_task_failures

        # Initialize S3 storage
        s3_config = S3StorageConfig(
            endpoint_url=config['storage']['s3']['endpoint_url'],
            access_key=config['storage']['s3']['access_key'],
            secret_key=config['storage']['s3']['secret_key'],
            bucket_name=config['storage']['s3']['bucket_name'],
            use_ssl=config['storage']['s3']['use_ssl']
        )
        self.s3_storage = S3Storage(s3_config)

        # Initialize MinIO client for bulk operations
        self.minio_client = Minio(
            endpoint=config['storage']['s3']['endpoint_url'].replace('https://', '').replace('http://', ''),
            access_key=config['storage']['s3']['access_key'],
            secret_key=config['storage']['s3']['secret_key'],
            secure=config['storage']['s3']['use_ssl']
        )
        self.bucket_name = config['storage']['s3']['bucket_name']

        # Initialize error handler
        self.error_handler = ErrorHandler(config, worker_task_failures)

        # Initialize modular components
        self.state_evaluator = StateEvaluator(config)
        self.task_creator = TaskCreator(config)
        self.result_handler = TaskResultHandler(config, self.error_handler, behavior_manager)

        # For backward compatibility
        self.s3_checker = self.state_evaluator.s3_checker
        self.flag_reconciler = self.state_evaluator.flag_reconciler

        logger.info(f"PipelineManagerV2 v{self.VERSION} initialized (modular architecture)")

    # =========================================================================
    # Public API - Maintains backward compatibility with original PipelineManager
    # =========================================================================

    def get_current_stitch_version(self) -> str:
        """Get the current stitch version from config."""
        return self.state_evaluator.get_current_stitch_version()

    @classmethod
    def should_recreate_task(cls, task_type: str, error_message: str) -> bool:
        """
        Return True if the error for this task_type should trigger recreation.

        This is a class method for backward compatibility.
        """
        return TaskResultHandler({}, None, None)._should_recreate_task(task_type, error_message)

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

        Delegates to TaskResultHandler for failure/completion handling,
        then to StateEvaluator for state reconciliation and task creation.

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
        """
        try:
            # Step 1: Handle the task result (completions, failures)
            handler_result = await self.result_handler.handle_task_result(
                session, content_id, task_type, status, result,
                db_task, content, worker_id, chunk_index
            )

            # Step 2: If we should skip state audit, return early
            if handler_result.get('skip_state_audit', False):
                # Calculate worker availability for download tasks
                skip_wait = (
                    task_type in ["download_youtube", "download_rumble"] and
                    status == "skipped" and
                    (result.get('skipped_existing', False) or result.get('skip_wait_time', False))
                )
                handler_result['available_after'] = self.result_handler.calculate_worker_available_after(
                    worker_id, task_type, skip_wait
                )
                return handler_result

            # Step 3: Evaluate content state and create tasks
            eval_results = await self.state_evaluator.evaluate_content_state(
                session, content, db_task
            )

            # Step 4: Calculate worker availability
            skip_wait = (
                (task_type in ["download_youtube", "download_rumble"] and status == "skipped" and result.get('skipped_existing', False)) or
                (task_type in ["download_youtube", "download_rumble"] and status == "skipped" and result.get('reason') == 'already_exists') or
                (task_type in ["download_youtube", "download_rumble"] and result.get('skip_wait_time', False))
            )
            handler_result['available_after'] = self.result_handler.calculate_worker_available_after(
                worker_id, task_type, skip_wait
            )

            # Step 5: Build outcome message from evaluation results
            if eval_results['errors']:
                handler_result['outcome_message'] = f"Errors during evaluation: {'; '.join(eval_results['errors'])}"
            else:
                updates = []
                if eval_results['flags_updated']:
                    updates.append(f"Updated {', '.join(eval_results['flags_updated'])}")
                if eval_results['tasks_created']:
                    updates.append(f"Created {', '.join(eval_results['tasks_created'])}")
                if eval_results['tasks_blocked'] and task_type != 'transcribe':
                    updates.append(f"Blocked {', '.join(eval_results['tasks_blocked'])}")
                if not updates:
                    updates.append(f"Verified {task_type} completion")
                handler_result['outcome_message'] = '. '.join(updates)

            return handler_result

        except Exception as e:
            logger.error(f"Error handling task result: {e}", exc_info=True)
            return {
                'available_after': None,
                'outcome_message': f"Error handling task result: {e}"
            }

    async def evaluate_content_state(
        self,
        session: Session,
        content: Content,
        db_task: Optional[TaskQueue] = None
    ) -> Dict[str, Any]:
        """
        Reconcile database state with S3 reality and create any missing tasks.

        Delegates directly to StateEvaluator.

        Args:
            session: Active database session
            content: Content object to evaluate
            db_task: Optional TaskQueue object that triggered evaluation

        Returns:
            Dict containing evaluation results and actions taken
        """
        return await self.state_evaluator.evaluate_content_state(session, content, db_task)

    async def _check_and_create_stitch_task(
        self,
        session: Session,
        content: Content
    ) -> tuple[bool, Optional[int]]:
        """
        Check if content is ready for stitching and create the task.

        Returns (task_created, task_id).
        """
        return await self.state_evaluator.check_and_create_stitch_task(session, content)

    async def _create_task_if_not_exists(
        self,
        session: Session,
        content_id: str,
        task_type: str,
        input_data: Dict[str, Any],
        priority: int = None,
        content: Content = None
    ) -> tuple[Optional[int], Optional[str]]:
        """
        Create a new task if a pending/processing one doesn't already exist.

        Delegates to TaskCreator.

        Returns:
            Tuple of (task_id, block_reason)
        """
        return await self.task_creator.create_task_if_not_exists(
            session, content_id, task_type, input_data, priority, content
        )

    def _is_content_within_project_date_range(self, content: Content, project: str) -> bool:
        """Check if content's publish_date falls within project's date range."""
        return self.task_creator.is_content_within_project_date_range(content, project)

    async def bulk_reconcile_content_states(self, session: Session) -> Dict[str, Any]:
        """
        Efficiently reconcile database state with S3 for all content.

        This method builds a bulk file index and reconciles all content
        without creating tasks (for use as a cleanup/maintenance operation).

        Args:
            session: Active database session

        Returns:
            Dict with reconciliation statistics
        """
        logger.info("Starting bulk content state reconciliation")
        start_time = datetime.now()

        results = {
            'total_content': 0,
            'updated_content': 0,
            'flag_updates': defaultdict(int),
            'errors': [],
            'processing_time': 0
        }

        try:
            # Build bulk file index
            file_indices = self.s3_checker.get_bulk_file_index(
                self.minio_client, self.bucket_name
            )

            # Get all content
            all_content = session.query(Content).all()
            results['total_content'] = len(all_content)

            # Reconcile in batches
            stats = self.flag_reconciler.bulk_reconcile(
                session, all_content, file_indices
            )

            results['updated_content'] = stats['updated']
            results['flag_updates'] = stats['flag_counts']
            results['errors'] = stats['errors']

        except Exception as e:
            error_msg = f"Error during bulk reconciliation: {e}"
            logger.error(error_msg, exc_info=True)
            results['errors'].append(error_msg)

        end_time = datetime.now()
        results['processing_time'] = (end_time - start_time).total_seconds()

        logger.info(f"Bulk reconciliation complete: {results['total_content']} items, "
                   f"{results['updated_content']} updated in {results['processing_time']:.2f}s")

        return results


# Alias for easy migration
PipelineManager = PipelineManagerV2
