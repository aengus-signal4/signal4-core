"""
Worker Info - Worker state and configuration management
"""
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Set

logger = logging.getLogger(__name__)

class WorkerInfo:
    """Enhanced worker information with service tracking"""
    
    def __init__(self, worker_id: str, config: Dict[str, Any]):
        self.worker_id = worker_id
        self.config = config
        self.status = 'stopped'  # stopped, starting, active, unhealthy, failed
        
        # Parse task types and their concurrency limits from config
        enabled_tasks = config.get('task_types', [])
        self.task_types = []
        self.task_type_limits = {}
        
        for task_entry in enabled_tasks:
            if isinstance(task_entry, str):
                # Parse "task_type:limit" format
                parts = task_entry.split(':', 1)
                task_type = parts[0].strip()
                limit = 1  # Default limit
                
                if len(parts) == 2 and parts[1].strip().isdigit():
                    limit = int(parts[1].strip())
                    if limit <= 0:
                        logger.warning(f"Worker {worker_id}: Invalid limit '{parts[1]}' for task type '{task_type}', defaulting to 1")
                        limit = 1
                elif len(parts) == 2:
                    logger.warning(f"Worker {worker_id}: Non-numeric limit '{parts[1]}' for task type '{task_type}', defaulting to 1")
                
                if task_type:
                    self.task_types.append(task_type)
                    self.task_type_limits[task_type] = limit
                    logger.debug(f"Worker {worker_id}: Task type '{task_type}' with limit {limit}")
                else:
                    logger.warning(f"Worker {worker_id}: Skipping empty task type entry")
            else:
                logger.warning(f"Worker {worker_id}: Invalid task type entry format: {task_entry}")
        
        # Calculate total concurrent task limit from sum of task type limits
        self.max_concurrent_tasks = sum(self.task_type_limits.values()) if self.task_type_limits else 5
        
        # Track current tasks by type
        self.current_tasks_by_type = {task_type: set() for task_type in self.task_types}
        self.active_tasks = 0
        self.assigned_task_ids = set()
        
        # Network configuration
        self.eth_ip = config.get('eth')
        self.wifi_ip = config.get('wifi')
        self.is_head_worker = config.get('type') == 'head'
        
        # Service configuration
        self.services_config = config.get('services', {})
        
        # Download mode configuration
        self.download_mode = config.get('download_mode', 'cookie')  # 'cookie' or 'proxy'

        # Restart retry tracking with exponential backoff
        self.restart_attempts = 0
        self.last_restart_attempt = None
        self.max_restart_attempts = 20  # Maximum restart attempts before giving up
        self.restart_backoff_base = 60  # Base backoff time in seconds (1 minute)
        self.restart_backoff_max = 60*60*24  # Maximum backoff time in seconds (24 hours)
        # Add restart lock to prevent concurrent restart attempts
        self.restart_lock = asyncio.Lock()

        # Queue depth tracking for batch task dispatch
        self.queue_depths_by_type: Dict[str, int] = {task_type: 0 for task_type in self.task_types}
        self.supports_batch_api: Optional[bool] = None  # None = unknown, True/False = detected
        self.last_queue_status_check: Optional[datetime] = None

        # Timestamps
        self.last_heartbeat = datetime.now(timezone.utc)
        self.started_at = None
        self.available_after = None
    
    def should_attempt_restart(self) -> bool:
        """Check if we should attempt to restart this worker based on retry logic."""
        if self.restart_attempts >= self.max_restart_attempts:
            return False
            
        if self.last_restart_attempt is None:
            return True
            
        # Calculate exponential backoff
        backoff_time = min(
            self.restart_backoff_base * (2 ** self.restart_attempts),
            self.restart_backoff_max
        )
        
        time_since_last_attempt = (datetime.now(timezone.utc) - self.last_restart_attempt).total_seconds()
        return time_since_last_attempt >= backoff_time

    def record_restart_attempt(self, success: bool):
        """Record a restart attempt and its outcome."""
        self.last_restart_attempt = datetime.now(timezone.utc)
        
        if success:
            # Reset retry tracking on successful restart
            self.restart_attempts = 0
            logger.info(f"Worker {self.worker_id} successfully restarted, reset retry count")
        else:
            self.restart_attempts += 1
            next_backoff = min(
                self.restart_backoff_base * (2 ** self.restart_attempts),
                self.restart_backoff_max
            )
            logger.warning(f"Worker {self.worker_id} restart attempt {self.restart_attempts}/{self.max_restart_attempts} failed. "
                         f"Next retry in {next_backoff}s")

    def get_restart_backoff_remaining(self) -> float:
        """Get remaining time until next restart attempt is allowed."""
        if self.last_restart_attempt is None or self.restart_attempts >= self.max_restart_attempts:
            return 0.0
            
        backoff_time = min(
            self.restart_backoff_base * (2 ** self.restart_attempts),
            self.restart_backoff_max
        )
        
        time_since_last_attempt = (datetime.now(timezone.utc) - self.last_restart_attempt).total_seconds()
        return max(0.0, backoff_time - time_since_last_attempt)
    
    async def attempt_restart_with_lock(self, restart_manager_func):
        """
        Attempt to restart the worker with a lock to prevent concurrent restart attempts.
        Returns (success: bool, attempted: bool) where attempted indicates if a restart was actually tried.
        """
        if self.restart_lock.locked():
            logger.debug(f"Worker {self.worker_id} restart already in progress, skipping concurrent attempt")
            return False, False
            
        async with self.restart_lock:
            # Double-check should_attempt_restart inside the lock
            if not self.should_attempt_restart():
                backoff_remaining = self.get_restart_backoff_remaining()
                logger.debug(f"Worker {self.worker_id} restart not allowed: "
                           f"attempts={self.restart_attempts}/{self.max_restart_attempts}, "
                           f"backoff_remaining={backoff_remaining:.1f}s")
                return False, False
                
            # Perform the restart
            logger.info(f"Attempting to restart worker {self.worker_id} "
                       f"(attempt {self.restart_attempts + 1}/{self.max_restart_attempts})")
            
            restart_success = await restart_manager_func(self.worker_id)
            self.record_restart_attempt(restart_success)
            
            if restart_success:
                self.status = 'active'
                logger.info(f"Successfully restarted worker {self.worker_id}")
            else:
                logger.warning(f"Failed to restart worker {self.worker_id}")
                
            return restart_success, True

    def get_queue_capacity(self, task_type: str) -> int:
        """Calculate maximum queue capacity: limit × 2 + 2"""
        limit = self.task_type_limits.get(task_type, 0)
        return limit * 2 + 2

    def get_backfill_threshold(self, task_type: str) -> int:
        """Calculate backfill threshold: limit + 1"""
        limit = self.task_type_limits.get(task_type, 0)
        return limit + 1

    def needs_backfill(self, task_type: str) -> bool:
        """Check if queue needs backfilling (depth ≤ threshold)"""
        current_depth = self.queue_depths_by_type.get(task_type, 0)
        threshold = self.get_backfill_threshold(task_type)
        return current_depth <= threshold

    def get_backfill_count(self, task_type: str) -> int:
        """Calculate how many tasks to send to reach capacity"""
        current_depth = self.queue_depths_by_type.get(task_type, 0)
        capacity = self.get_queue_capacity(task_type)
        return max(0, capacity - current_depth)

    def can_accept_task(self, task_type: str) -> bool:
        """Check if worker can accept a task of given type"""
        if (self.status != 'active' or 
            task_type not in self.task_types or
            (self.available_after and datetime.now(timezone.utc) < self.available_after)):
            return False
        
        # Check overall capacity
        if self.active_tasks >= self.max_concurrent_tasks:
            return False
        
        # Check task-type-specific capacity
        task_type_limit = self.task_type_limits.get(task_type, 0)
        current_count = len(self.current_tasks_by_type.get(task_type, set()))
        
        return current_count < task_type_limit
    
    def add_task(self, task_id: str, task_type: str = None) -> bool:
        """Add a task to this worker"""
        if self.active_tasks >= self.max_concurrent_tasks:
            return False
        
        # If task_type is provided, also check task-type-specific limits
        if task_type:
            task_type_limit = self.task_type_limits.get(task_type, 0)
            current_count = len(self.current_tasks_by_type.get(task_type, set()))
            if current_count >= task_type_limit:
                return False
            
            # Add to task-type tracking
            if task_type in self.current_tasks_by_type:
                self.current_tasks_by_type[task_type].add(task_id)
        
        self.assigned_task_ids.add(task_id)
        self.active_tasks += 1
        return True
    
    def remove_task(self, task_id: str, task_type: str = None) -> bool:
        """Remove a task from this worker"""
        if task_id in self.assigned_task_ids:
            self.assigned_task_ids.remove(task_id)
            self.active_tasks = max(0, self.active_tasks - 1)
            
            # Remove from task-type tracking if provided
            if task_type and task_type in self.current_tasks_by_type:
                self.current_tasks_by_type[task_type].discard(task_id)
            
            return True
        return False
    
    def get_task_counts_by_type(self) -> Dict[str, int]:
        """Get actual task counts by type"""
        return {task_type: len(tasks) for task_type, tasks in self.current_tasks_by_type.items()}
    
    def update_from_config(self, new_config: Dict[str, Any]):
        """Update worker configuration dynamically"""
        # Update task types and limits
        old_task_types = set(self.task_types)
        
        # Re-parse task types
        enabled_tasks = new_config.get('task_types', [])
        self.task_types = []
        self.task_type_limits = {}
        
        for task_entry in enabled_tasks:
            if isinstance(task_entry, str):
                parts = task_entry.split(':', 1)
                task_type = parts[0].strip()
                limit = 1
                
                if len(parts) == 2 and parts[1].strip().isdigit():
                    limit = int(parts[1].strip())
                    if limit <= 0:
                        limit = 1
                
                if task_type:
                    self.task_types.append(task_type)
                    self.task_type_limits[task_type] = limit
        
        # Recalculate total limit from sum of task type limits
        self.max_concurrent_tasks = sum(self.task_type_limits.values()) if self.task_type_limits else 5
        
        # Update download mode
        self.download_mode = new_config.get('download_mode', 'cookie')
        
        # Update network IPs
        self.eth_ip = new_config.get('eth')
        self.wifi_ip = new_config.get('wifi')
        
        # Update services config
        self.services_config = new_config.get('services', {})
        
        # Store updated config
        self.config = new_config
        
        new_task_types = set(self.task_types)
        if old_task_types != new_task_types:
            logger.info(f"Worker {self.worker_id} task types changed: {old_task_types} -> {new_task_types}")
    
    def get_status_dict(self) -> Dict[str, Any]:
        """Get worker status as dictionary"""
        return {
            'worker_id': self.worker_id,
            'status': self.status,
            'active_tasks': self.active_tasks,
            'max_concurrent_tasks': self.max_concurrent_tasks,
            'task_types': self.task_types,
            'task_counts_by_type': self.get_task_counts_by_type(),
            'last_heartbeat': self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'available_after': self.available_after.isoformat() if self.available_after else None,
            'restart_info': {
                'restart_attempts': self.restart_attempts,
                'max_restart_attempts': self.max_restart_attempts,
                'last_restart_attempt': self.last_restart_attempt.isoformat() if self.last_restart_attempt else None,
                'can_restart': self.should_attempt_restart(),
                'backoff_remaining': self.get_restart_backoff_remaining()
            },
            'network_info': {
                'eth_ip': self.eth_ip,
                'wifi_ip': self.wifi_ip,
                'is_head_worker': self.is_head_worker
            }
        }