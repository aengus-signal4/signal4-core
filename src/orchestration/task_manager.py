"""
Task Manager - Handles task assignment, queue management, and task operations
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import json

from sqlalchemy.orm import Session
from sqlalchemy import text, and_, or_

# Add project root to path
import sys
from pathlib import Path
from src.utils.paths import get_project_root, get_config_path
sys.path.append(str(get_project_root()))

from src.database.session import get_session
from src.database.models import TaskQueue, Content, ContentChunk
from src.utils.logger import setup_worker_logger
from src.utils.priority import calculate_priority_by_date

logger = setup_worker_logger('task_manager')

# Task status constants
TASK_STATUS_PENDING = 'pending'
TASK_STATUS_PROCESSING = 'processing'
TASK_STATUS_COMPLETED = 'completed'
TASK_STATUS_FAILED = 'failed'
TASK_STATUS_CANCELLED = 'cancelled'
TASK_STATUS_SKIPPED = 'skipped'

class TaskManager:
    """Manages task operations for the orchestrator"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logger

        # Task assignment tracking
        self.assigned_tasks = set()  # Track assigned task IDs
        self.task_assignment_lock = asyncio.Lock()

        # Task blocking (for embedding hydration, maintenance, etc.)
        self.blocked_task_types = set()  # Globally blocked task types

        # Worker failure tracking
        self.worker_task_failures = defaultdict(lambda: defaultdict(lambda: {
            'count': 0,
            'last_failure_time': None,
            'paused': False,
            'pause_timeout': None
        }))

        self.max_consecutive_failures = config.get('processing', {}).get('max_consecutive_failures', 3)
        self.failure_cooldown_seconds = config.get('processing', {}).get('failure_cooldown_seconds', 300)

        # Task queue cache (lazy initialization)
        self.cache = None
        self.cache_enabled = config.get('processing', {}).get('task_cache_enabled', True)

    async def initialize_cache(self):
        """Initialize the task queue cache"""
        if not self.cache_enabled:
            self.logger.info("Task cache disabled in config")
            return

        try:
            from .task_queue_cache import TaskQueueCache

            prefetch_size = self.config.get('processing', {}).get('task_cache_prefetch_size', 100)
            refresh_threshold = self.config.get('processing', {}).get('task_cache_refresh_threshold', 20)
            ttl_seconds = self.config.get('processing', {}).get('task_cache_ttl_seconds', 60)

            self.cache = TaskQueueCache(
                task_manager=self,
                prefetch_size=prefetch_size,
                refresh_threshold=refresh_threshold,
                ttl_seconds=ttl_seconds
            )

            await self.cache.start()
            self.logger.info("Task queue cache initialized and started")

        except Exception as e:
            self.logger.error(f"Failed to initialize task cache: {str(e)}")
            self.cache = None
            self.cache_enabled = False

    async def shutdown_cache(self):
        """Shutdown the task queue cache"""
        if self.cache:
            await self.cache.stop()
            self.logger.info("Task queue cache stopped")

    async def get_next_task_cached(self, task_types: List[str], exclude_task_ids: Optional[Set[str]] = None) -> Optional[Dict[str, Any]]:
        """
        Get next single task using cache (fast path).

        Args:
            task_types: List of acceptable task types (in priority order)
            exclude_task_ids: Set of task IDs to skip

        Returns:
            Task dict or None
        """
        if not self.cache_enabled or not self.cache:
            # Fallback to direct DB query
            tasks = await self.get_next_tasks(limit=1, task_types=task_types, exclude_task_ids=exclude_task_ids)
            return tasks[0] if tasks else None

        # Try cache first
        task = await self.cache.get_next_task(task_types, exclude_task_ids)

        if task:
            return task

        # Cache miss - fallback to DB
        self.logger.debug(f"Cache miss for task types {task_types}, falling back to DB")
        tasks = await self.get_next_tasks(limit=1, task_types=task_types, exclude_task_ids=exclude_task_ids)

        if tasks:
            return tasks[0]

        return None

    async def get_next_tasks_cached(self, limit: int, task_types: Optional[List[str]] = None,
                                    exclude_task_ids: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
        """
        Get multiple tasks using cache (batch operation).

        Args:
            limit: Maximum number of tasks to return
            task_types: Optional list of acceptable task types
            exclude_task_ids: Set of task IDs to skip

        Returns:
            List of task dicts
        """
        if not self.cache_enabled or not self.cache or not task_types:
            # Fallback to direct DB query
            return await self.get_next_tasks(limit=limit, task_types=task_types, exclude_task_ids=exclude_task_ids)

        # Try cache first
        tasks = await self.cache.get_next_tasks_batch(task_types, limit, exclude_task_ids)

        if tasks:
            return tasks

        # Cache miss - fallback to DB
        self.logger.debug(f"Cache batch miss for task types {task_types}, falling back to DB")
        return await self.get_next_tasks(limit=limit, task_types=task_types, exclude_task_ids=exclude_task_ids)

    async def get_next_tasks(self, limit: int = 100, exclude_task_ids: Optional[Set[str]] = None, task_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Get next tasks from queue for assignment"""
        try:
            with get_session() as session:
                query = session.query(TaskQueue).filter(
                    TaskQueue.status == TASK_STATUS_PENDING
                ).order_by(TaskQueue.priority.desc(), TaskQueue.created_at.asc())

                # Exclude already assigned tasks
                if exclude_task_ids:
                    query = query.filter(~TaskQueue.id.in_(exclude_task_ids))

                # Filter by task types if specified
                if task_types:
                    query = query.filter(TaskQueue.task_type.in_(task_types))

                tasks = query.limit(limit).all()

                # Convert to dict format and filter out globally blocked task types
                task_list = []
                for task in tasks:
                    # Skip blocked task types
                    if task.task_type in self.blocked_task_types:
                        continue

                    task_dict = {
                        'id': task.id,
                        'content_id': task.content_id,
                        'task_type': task.task_type,
                        'priority': task.priority,
                        'input_data': task.input_data or {},
                        'created_at': task.created_at,
                        'attempts': getattr(task, 'attempts', 0)
                    }
                    task_list.append(task_dict)

                return task_list

        except Exception as e:
            self.logger.error(f"Error getting next tasks: {str(e)}")
            return []
    
    async def can_assign_task_to_worker(self, worker_id: str, task_type: str) -> bool:
        """Check if a task type can be assigned to a worker"""
        try:
            # Check if worker is paused for this task type due to failures
            failure_info = self.worker_task_failures[worker_id][task_type]
            
            if failure_info['paused']:
                # Check if cooldown period has passed
                if failure_info['last_failure_time']:
                    now = datetime.now()
                    cooldown_duration = failure_info.get('pause_timeout', self.failure_cooldown_seconds)
                    time_since_failure = (now - failure_info['last_failure_time']).total_seconds()
                    
                    if time_since_failure >= cooldown_duration:
                        # Reset failure state
                        self.worker_task_failures[worker_id][task_type] = {
                            'count': 0,
                            'last_failure_time': None,
                            'paused': False,
                            'pause_timeout': None
                        }
                        self.logger.info(f"Reset failure state for worker {worker_id} task type {task_type}")
                        return True
                    else:
                        remaining = cooldown_duration - time_since_failure
                        self.logger.debug(f"Worker {worker_id} still paused for {task_type} "
                                        f"({remaining:.1f}s remaining)")
                        return False
                
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking task assignment for {worker_id}: {str(e)}")
            return False
    
    async def mark_task_assigned(self, task_id: str, worker_id: str) -> bool:
        """Mark a task as assigned to a worker"""
        try:
            async with self.task_assignment_lock:
                if task_id in self.assigned_tasks:
                    self.logger.warning(f"Task {task_id} already assigned")
                    return False
                
                with get_session() as session:
                    task = session.query(TaskQueue).filter_by(id=task_id).first()
                    if not task:
                        self.logger.error(f"Task {task_id} not found")
                        return False
                    
                    if task.status != TASK_STATUS_PENDING:
                        self.logger.warning(f"Task {task_id} is not pending (status: {task.status})")
                        return False
                    
                    # Update task status
                    task.status = TASK_STATUS_PROCESSING
                    task.worker_id = worker_id
                    task.started_at = datetime.now(timezone.utc)
                    session.commit()

                    # Track assignment
                    self.assigned_tasks.add(task_id)

                    # Invalidate from cache
                    if self.cache:
                        await self.cache.invalidate_task(task_id)
                    
                    self.logger.debug(f"Marked task {task_id} as assigned to {worker_id}")
                    return True
                    
        except Exception as e:
            self.logger.error(f"Error marking task {task_id} as assigned: {str(e)}")
            return False
    
    async def handle_task_failure(self, task_id: str, worker_id: str, task_type: str, error_message: str) -> None:
        """Handle task failure and update failure tracking"""
        try:
            # Update failure tracking
            now = datetime.now()
            failure_data = self.worker_task_failures[worker_id][task_type]
            failure_data['count'] += 1
            failure_data['last_failure_time'] = now
            
            if failure_data['count'] >= self.max_consecutive_failures:
                failure_data['paused'] = True
                failure_data['pause_timeout'] = self.failure_cooldown_seconds
                self.logger.warning(f"Pausing task type {task_type} for worker {worker_id} "
                                  f"after {failure_data['count']} consecutive failures")
            
            # Update task in database
            with get_session() as session:
                task = session.query(TaskQueue).filter_by(id=task_id).first()
                if task:
                    task.status = TASK_STATUS_FAILED
                    task.error = error_message[:1000]  # Limit error message length
                    task.completed_at = datetime.now(timezone.utc)
                    session.commit()
            
            # Remove from assigned tasks
            self.assigned_tasks.discard(task_id)
            
        except Exception as e:
            self.logger.error(f"Error handling task failure for {task_id}: {str(e)}")
    
    async def handle_task_completion(self, task_id: str, worker_id: str, task_type: str, 
                                   status: str, result: Dict[str, Any]) -> None:
        """Handle task completion"""
        try:
            # Reset failure count on successful completion
            if status == TASK_STATUS_COMPLETED:
                if self.worker_task_failures[worker_id][task_type]['count'] > 0:
                    self.logger.debug(f"Reset failure count for worker {worker_id} task type {task_type}")
                    self.worker_task_failures[worker_id][task_type] = {
                        'count': 0,
                        'last_failure_time': None,
                        'paused': False,
                        'pause_timeout': None
                    }
            
            # Update task in database
            with get_session() as session:
                task = session.query(TaskQueue).filter_by(id=task_id).first()
                if task:
                    task.status = status
                    task.completed_at = datetime.now(timezone.utc)
                    
                    # Store result if provided
                    if result:
                        task.result = result
                        if status == TASK_STATUS_FAILED and 'error' in result:
                            task.error = str(result['error'])[:1000]
                    
                    session.commit()
            
            # Remove from assigned tasks
            self.assigned_tasks.discard(task_id)
            
        except Exception as e:
            self.logger.error(f"Error handling task completion for {task_id}: {str(e)}")
    
    async def create_task(self, content_id: str, task_type: str, input_data: Dict[str, Any], 
                         priority: Optional[int] = None) -> Optional[str]:
        """Create a new task"""
        try:
            with get_session() as session:
                # Check for existing pending/processing task
                existing_task = session.query(TaskQueue).filter(
                    TaskQueue.content_id == content_id,
                    TaskQueue.task_type == task_type,
                    TaskQueue.status.in_([TASK_STATUS_PENDING, TASK_STATUS_PROCESSING])
                ).first()
                
                if existing_task:
                    self.logger.debug(f"Task {task_type} already exists for {content_id}")
                    return None
                
                # Calculate priority if not provided
                if priority is None:
                    content = session.query(Content).filter_by(content_id=content_id).first()
                    if content:
                        project = input_data.get('project', 'unknown')
                        project_priority = self._get_project_priority(project)
                        priority = calculate_priority_by_date(content.publish_date, project_priority)
                    else:
                        priority = 1000  # Default priority
                
                # Create new task
                new_task = TaskQueue(
                    content_id=content_id,
                    task_type=task_type,
                    status=TASK_STATUS_PENDING,
                    priority=priority,
                    input_data=input_data,
                    created_at=datetime.now(timezone.utc)
                )
                
                session.add(new_task)
                session.flush()  # Get ID
                task_id = str(new_task.id)
                session.commit()
                
                self.logger.info(f"Created task {task_id}: {task_type} for {content_id}")
                return task_id
                
        except Exception as e:
            self.logger.error(f"Error creating task {task_type} for {content_id}: {str(e)}")
            return None
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task"""
        try:
            with get_session() as session:
                task = session.query(TaskQueue).filter_by(id=task_id).first()
                if not task:
                    return False
                
                if task.status not in [TASK_STATUS_PENDING, TASK_STATUS_PROCESSING]:
                    return False
                
                task.status = TASK_STATUS_CANCELLED
                task.completed_at = datetime.now(timezone.utc)
                session.commit()
                
                # Remove from assigned tasks
                self.assigned_tasks.discard(task_id)
                
                self.logger.info(f"Cancelled task {task_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error cancelling task {task_id}: {str(e)}")
            return False
    
    async def get_task_stats(self) -> Dict[str, Any]:
        """Get task queue statistics"""
        try:
            with get_session() as session:
                # Count tasks by status
                status_counts = {}
                for status in [TASK_STATUS_PENDING, TASK_STATUS_PROCESSING, 
                              TASK_STATUS_COMPLETED, TASK_STATUS_FAILED]:
                    count = session.query(TaskQueue).filter_by(status=status).count()
                    status_counts[status] = count
                
                # Count tasks by type
                type_counts = {}
                task_types = session.query(TaskQueue.task_type).distinct().all()
                for (task_type,) in task_types:
                    count = session.query(TaskQueue).filter(
                        TaskQueue.task_type == task_type,
                        TaskQueue.status == TASK_STATUS_PENDING
                    ).count()
                    type_counts[task_type] = count
                
                return {
                    'status_counts': status_counts,
                    'type_counts': type_counts,
                    'assigned_tasks': len(self.assigned_tasks),
                    'worker_failures': dict(self.worker_task_failures)
                }
                
        except Exception as e:
            self.logger.error(f"Error getting task stats: {str(e)}")
            return {}
    
    def _get_project_priority(self, project: str) -> int:
        """Get priority for a project"""
        active_projects = self.config.get('active_projects', {})
        if project in active_projects:
            return active_projects[project].get('priority', 1)
        return 1
    
    async def cleanup_stale_tasks(self, timeout_minutes: int = 30) -> int:
        """Clean up tasks that have been processing too long"""
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=timeout_minutes)
            
            with get_session() as session:
                stale_tasks = session.query(TaskQueue).filter(
                    TaskQueue.status == TASK_STATUS_PROCESSING,
                    TaskQueue.started_at < cutoff_time
                ).all()
                
                count = 0
                for task in stale_tasks:
                    task.status = TASK_STATUS_PENDING
                    task.worker_id = None
                    task.started_at = None
                    task.error = "Task timed out and was reset"
                    count += 1
                    
                    # Remove from assigned tasks
                    self.assigned_tasks.discard(str(task.id))
                
                if count > 0:
                    session.commit()
                    self.logger.info(f"Reset {count} stale tasks to pending status")
                
                return count
                
        except Exception as e:
            self.logger.error(f"Error cleaning up stale tasks: {str(e)}")
            return 0