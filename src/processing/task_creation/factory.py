"""
Task creation factory with strategy pattern.

Centralizes task creation logic and delegates to per-task-type strategies.
"""

from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timezone
import logging

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from sqlalchemy import text

from src.database.models import Content, TaskQueue
from src.utils.priority import calculate_priority_by_date
from src.processing.task_creation.strategies import (
    TaskCreationStrategy,
    DownloadTaskStrategy,
    ConvertTaskStrategy,
    TranscribeTaskStrategy,
    DiarizeTaskStrategy,
    StitchTaskStrategy,
    CleanupTaskStrategy,
)

logger = logging.getLogger(__name__)

TASK_STATUS_PENDING = 'pending'
TASK_STATUS_PROCESSING = 'processing'
TASK_STATUS_FAILED = 'failed'


class TaskFactory:
    """
    Factory for creating pipeline tasks.

    Uses strategy pattern to encapsulate per-task-type logic.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._strategies: Dict[str, TaskCreationStrategy] = {}
        self._register_default_strategies()

    def _register_default_strategies(self):
        """Register built-in task strategies."""
        # Download strategies for each platform
        for platform in ['youtube', 'rumble', 'podcast']:
            self.register(DownloadTaskStrategy(self.config, platform))

        # Other task strategies
        self.register(ConvertTaskStrategy(self.config))
        self.register(TranscribeTaskStrategy(self.config))
        self.register(DiarizeTaskStrategy(self.config))
        self.register(StitchTaskStrategy(self.config))
        self.register(CleanupTaskStrategy(self.config))

    def register(self, strategy: TaskCreationStrategy):
        """Register a task creation strategy."""
        self._strategies[strategy.task_type] = strategy

    async def create_task(
        self,
        session: Session,
        content: Content,
        task_type: str,
        project: str,
        priority: int = None,
        input_data: Dict[str, Any] = None,
        **kwargs
    ) -> Tuple[Optional[int], Optional[str]]:
        """
        Create a task if appropriate.

        Args:
            session: Database session
            content: Content to create task for
            task_type: Type of task to create
            project: Project name
            priority: Optional explicit priority
            input_data: Additional input data to merge
            **kwargs: Additional arguments passed to strategy

        Returns:
            Tuple of (task_id, block_reason)
        """
        strategy = self._strategies.get(task_type)
        if not strategy:
            return None, f"Unknown task type: {task_type}"

        # Merge input_data with kwargs for strategy
        merged_kwargs = {**kwargs}
        if input_data:
            merged_kwargs.update(input_data)

        # Check if should create
        should_create, reason = strategy.should_create(session, content, **merged_kwargs)
        if not should_create:
            logger.debug(f"Not creating {task_type} for {content.content_id}: {reason}")
            return None, reason

        # Check for existing pending/processing task
        existing = self._find_existing_task(session, content, task_type, merged_kwargs)
        if existing:
            if existing.status == TASK_STATUS_FAILED:
                # Check if we can reset it
                task_result = existing.result or {}
                if task_result.get('permanent', False):
                    error_code = task_result.get('error_code', 'unknown')
                    return None, f"Task permanently failed with {error_code}"

                # Reset failed task to pending
                existing.status = TASK_STATUS_PENDING
                existing.error = None
                existing.worker_id = None
                existing.processor_task_id = None
                existing.started_at = None
                existing.completed_at = None
                session.flush()
                logger.info(
                    f"Reset failed task {existing.id} ({task_type}) for {content.content_id} "
                    f"to pending (attempt {existing.attempts + 1}/{existing.max_attempts})"
                )
                return existing.id, None
            else:
                return None, "Task already exists (pending/processing)"

        # Calculate priority
        final_priority = self._calculate_priority(session, content, project, priority)

        # Build input data
        base_input_data = strategy.get_input_data(content, project, **merged_kwargs)
        if input_data:
            base_input_data.update(input_data)

        # Create the task
        new_task = TaskQueue(
            content_id=content.content_id,
            task_type=task_type,
            status=TASK_STATUS_PENDING,
            priority=final_priority,
            input_data=base_input_data,
            created_at=datetime.now(timezone.utc)
        )

        try:
            session.add(new_task)
            session.flush()
            task_id = new_task.id

            chunk_info = ""
            if task_type == 'transcribe' and 'chunk_index' in base_input_data:
                chunk_info = f" (chunk {base_input_data['chunk_index']})"

            logger.info(
                f"Created new task: {task_type} (ID: {task_id}) for {content.content_id}"
                f"{chunk_info} with priority {final_priority}"
            )
            return task_id, None

        except IntegrityError as e:
            session.rollback()
            logger.warning(
                f"Task {task_type} for {content.content_id} already exists "
                f"(race condition caught): {e}"
            )
            return None, "Task already exists (race condition)"

    def _find_existing_task(
        self,
        session: Session,
        content: Content,
        task_type: str,
        kwargs: Dict[str, Any]
    ) -> Optional[TaskQueue]:
        """Find existing pending/processing/failed task."""
        query = session.query(TaskQueue).filter(
            TaskQueue.content_id == content.content_id,
            TaskQueue.task_type == task_type,
            TaskQueue.status.in_([TASK_STATUS_PENDING, TASK_STATUS_PROCESSING, TASK_STATUS_FAILED])
        )

        # For transcribe tasks, filter by chunk_index
        if task_type == 'transcribe':
            chunk_index = kwargs.get('chunk_index')
            if chunk_index is not None:
                query = query.filter(
                    text("input_data->>'chunk_index' = :chunk_index").params(chunk_index=str(chunk_index))
                )

        return query.first()

    def _calculate_priority(
        self,
        session: Session,
        content: Content,
        project: str,
        explicit_priority: Optional[int]
    ) -> int:
        """Calculate task priority based on date and project."""
        # Get project priority from config
        project_priority = 1  # Default
        active_projects_config = self.config.get('active_projects', {})

        if project and isinstance(active_projects_config, dict):
            projects = project if isinstance(project, list) else [project]
            for proj in projects:
                if proj in active_projects_config:
                    project_config = active_projects_config.get(proj, {})
                    if isinstance(project_config, dict):
                        proj_priority = project_config.get('priority', 1)
                        project_priority = max(project_priority, proj_priority)

        # Calculate priority
        if explicit_priority is None:
            publish_date = getattr(content, 'publish_date', None)
            priority = calculate_priority_by_date(publish_date, project_priority)
        else:
            priority = explicit_priority + (project_priority * 1000000)

        return priority

    # Convenience methods for common task types
    async def create_download_task(
        self,
        session: Session,
        content: Content,
        project: str
    ) -> Tuple[Optional[int], Optional[str]]:
        """Create download task for content's platform."""
        return await self.create_task(
            session, content, f"download_{content.platform}", project
        )

    async def create_convert_task(
        self,
        session: Session,
        content: Content,
        project: str
    ) -> Tuple[Optional[int], Optional[str]]:
        """Create convert task."""
        return await self.create_task(session, content, "convert", project)

    async def create_transcribe_task(
        self,
        session: Session,
        content: Content,
        project: str,
        chunk_index: int
    ) -> Tuple[Optional[int], Optional[str]]:
        """Create transcribe task for a specific chunk."""
        return await self.create_task(
            session, content, "transcribe", project,
            input_data={'chunk_index': chunk_index}
        )

    async def create_diarize_task(
        self,
        session: Session,
        content: Content,
        project: str
    ) -> Tuple[Optional[int], Optional[str]]:
        """Create diarize task."""
        return await self.create_task(session, content, "diarize", project)

    async def create_stitch_task(
        self,
        session: Session,
        content: Content,
        project: str
    ) -> Tuple[Optional[int], Optional[str]]:
        """Create stitch task."""
        return await self.create_task(session, content, "stitch", project)

    async def create_cleanup_task(
        self,
        session: Session,
        content: Content,
        project: str
    ) -> Tuple[Optional[int], Optional[str]]:
        """Create cleanup task."""
        return await self.create_task(session, content, "cleanup", project)
