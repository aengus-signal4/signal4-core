"""
Task Queue Cache - In-memory cache with background prefetching for fast task assignment
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Set, Deque
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)

class TaskQueueCache:
    """
    In-memory task queue cache with background prefetching.

    Maintains a cache of pending tasks per task type to eliminate DB queries
    in the hot path of task assignment.
    """

    def __init__(self, task_manager, prefetch_size: int = 100, refresh_threshold: int = 20, ttl_seconds: int = 60):
        """
        Initialize task queue cache.

        Args:
            task_manager: TaskManager instance for fetching tasks from DB
            prefetch_size: Number of tasks to prefetch per task type
            refresh_threshold: Trigger refresh when cache drops below this size
            ttl_seconds: Time-to-live for cached tasks before refresh
        """
        self.task_manager = task_manager
        self.prefetch_size = prefetch_size
        self.refresh_threshold = refresh_threshold
        self.ttl_seconds = ttl_seconds

        # Cache structure: task_type -> deque of task dicts
        self.cache: Dict[str, Deque[Dict[str, Any]]] = defaultdict(deque)
        self.cache_lock = asyncio.Lock()

        # Track cache metadata
        self.cache_timestamps: Dict[str, datetime] = {}  # task_type -> last refresh time
        self.refresh_in_progress: Set[str] = set()  # Track ongoing refreshes

        # Statistics
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'refresh_count': 0,
            'total_fetched': 0
        }

        # Background refresh task
        self.should_stop = False
        self.refresh_task: Optional[asyncio.Task] = None

        logger.info(f"TaskQueueCache initialized: prefetch={prefetch_size}, threshold={refresh_threshold}, ttl={ttl_seconds}s")

    async def start(self):
        """Start background refresh monitoring"""
        logger.info("Starting TaskQueueCache background refresh")
        self.refresh_task = asyncio.create_task(self._refresh_monitor_loop())

    async def stop(self):
        """Stop background refresh"""
        logger.info("Stopping TaskQueueCache")
        self.should_stop = True
        if self.refresh_task:
            self.refresh_task.cancel()
            try:
                await self.refresh_task
            except asyncio.CancelledError:
                pass

    async def get_next_task(self, task_types: List[str], exclude_task_ids: Optional[Set[str]] = None) -> Optional[Dict[str, Any]]:
        """
        Get next task from cache (non-blocking).

        Args:
            task_types: List of acceptable task types (in priority order)
            exclude_task_ids: Set of task IDs to skip

        Returns:
            Task dict or None if no suitable task found
        """
        exclude_task_ids = exclude_task_ids or set()

        async with self.cache_lock:
            # Try each task type in priority order
            for task_type in task_types:
                cache_queue = self.cache.get(task_type)

                if not cache_queue:
                    continue

                # Search for suitable task (not excluded)
                found_task = None
                temp_removed = []

                try:
                    while cache_queue:
                        task = cache_queue.popleft()

                        # Check if task is excluded
                        if str(task['id']) in exclude_task_ids:
                            temp_removed.append(task)
                            continue

                        # Found suitable task
                        found_task = task
                        break

                    # Put back any temporarily removed tasks
                    for task in temp_removed:
                        cache_queue.appendleft(task)

                    if found_task:
                        self.stats['cache_hits'] += 1
                        logger.debug(f"Cache hit for task type {task_type}: {found_task['id']} "
                                   f"(cache size: {len(cache_queue)})")

                        # Trigger async refresh if below threshold
                        if len(cache_queue) < self.refresh_threshold:
                            asyncio.create_task(self._refresh_cache_async(task_type))

                        return found_task

                except IndexError:
                    pass

            # Cache miss - no suitable task found
            self.stats['cache_misses'] += 1
            logger.debug(f"Cache miss for task types {task_types}")

            return None

    async def get_next_tasks_batch(self, task_types: List[str], limit: int,
                                   exclude_task_ids: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
        """
        Get multiple tasks from cache (batch operation).

        Args:
            task_types: List of acceptable task types
            limit: Maximum number of tasks to return
            exclude_task_ids: Set of task IDs to skip

        Returns:
            List of task dicts (may be fewer than limit)
        """
        exclude_task_ids = exclude_task_ids or set()
        result_tasks = []

        async with self.cache_lock:
            for task_type in task_types:
                if len(result_tasks) >= limit:
                    break

                cache_queue = self.cache.get(task_type)
                if not cache_queue:
                    continue

                temp_removed = []

                try:
                    while cache_queue and len(result_tasks) < limit:
                        task = cache_queue.popleft()

                        if str(task['id']) in exclude_task_ids:
                            temp_removed.append(task)
                            continue

                        result_tasks.append(task)

                    # Put back excluded tasks
                    for task in temp_removed:
                        cache_queue.appendleft(task)

                    # Trigger refresh if needed
                    if len(cache_queue) < self.refresh_threshold:
                        asyncio.create_task(self._refresh_cache_async(task_type))

                except IndexError:
                    pass

        if result_tasks:
            self.stats['cache_hits'] += len(result_tasks)
            logger.debug(f"Cache batch hit: {len(result_tasks)} tasks")
        else:
            self.stats['cache_misses'] += 1

        return result_tasks

    async def invalidate_task(self, task_id: str):
        """Remove a task from cache (e.g., when assigned or completed)"""
        async with self.cache_lock:
            for task_type, cache_queue in self.cache.items():
                # Search and remove task
                temp_queue = deque()
                removed = False

                while cache_queue:
                    task = cache_queue.popleft()
                    if str(task['id']) != task_id:
                        temp_queue.append(task)
                    else:
                        removed = True

                # Restore queue without the removed task
                self.cache[task_type] = temp_queue

                if removed:
                    logger.debug(f"Invalidated task {task_id} from cache (type: {task_type})")
                    break

    async def refresh_task_type(self, task_type: str, force: bool = False):
        """
        Manually trigger cache refresh for a specific task type.

        Args:
            task_type: Task type to refresh
            force: Force refresh even if recently refreshed
        """
        await self._refresh_cache_async(task_type, force=force)

    async def _refresh_cache_async(self, task_type: str, force: bool = False):
        """Background refresh for a specific task type"""

        # Skip if already refreshing
        if task_type in self.refresh_in_progress and not force:
            logger.debug(f"Refresh already in progress for {task_type}")
            return

        # Check TTL
        if not force:
            last_refresh = self.cache_timestamps.get(task_type)
            if last_refresh:
                age = (datetime.now(timezone.utc) - last_refresh).total_seconds()
                if age < self.ttl_seconds:
                    logger.debug(f"Cache for {task_type} still fresh ({age:.1f}s old)")
                    return

        self.refresh_in_progress.add(task_type)

        try:
            logger.debug(f"Refreshing cache for task type: {task_type}")

            # Fetch tasks from DB via task manager
            tasks = await self.task_manager.get_next_tasks(
                limit=self.prefetch_size,
                task_types=[task_type],
                exclude_task_ids=self.task_manager.assigned_tasks
            )

            if tasks:
                async with self.cache_lock:
                    # Replace cache with fresh tasks
                    self.cache[task_type] = deque(tasks)
                    self.cache_timestamps[task_type] = datetime.now(timezone.utc)

                self.stats['refresh_count'] += 1
                self.stats['total_fetched'] += len(tasks)

                logger.info(f"Cache refreshed for {task_type}: {len(tasks)} tasks fetched")
            else:
                logger.debug(f"No tasks available for {task_type}")

                # Keep existing cache but update timestamp
                async with self.cache_lock:
                    self.cache_timestamps[task_type] = datetime.now(timezone.utc)

        except Exception as e:
            logger.error(f"Error refreshing cache for {task_type}: {str(e)}")

        finally:
            self.refresh_in_progress.discard(task_type)

    async def _refresh_monitor_loop(self):
        """Background loop to monitor and refresh stale caches"""
        logger.info("Starting cache refresh monitor loop")

        while not self.should_stop:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds

                now = datetime.now(timezone.utc)

                async with self.cache_lock:
                    task_types_to_refresh = []

                    for task_type, last_refresh in self.cache_timestamps.items():
                        age = (now - last_refresh).total_seconds()

                        # Refresh if stale
                        if age > self.ttl_seconds:
                            task_types_to_refresh.append(task_type)

                # Refresh stale caches (outside lock)
                for task_type in task_types_to_refresh:
                    await self._refresh_cache_async(task_type)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in refresh monitor loop: {str(e)}")

        logger.info("Cache refresh monitor loop stopped")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.stats['cache_hits'] + self.stats['cache_misses']
        hit_rate = (self.stats['cache_hits'] / total_requests * 100) if total_requests > 0 else 0

        cache_sizes = {task_type: len(queue) for task_type, queue in self.cache.items()}

        return {
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'hit_rate_percent': round(hit_rate, 2),
            'refresh_count': self.stats['refresh_count'],
            'total_fetched': self.stats['total_fetched'],
            'cache_sizes': cache_sizes,
            'active_refreshes': len(self.refresh_in_progress)
        }

    async def warm_cache(self, task_types: List[str]):
        """Pre-populate cache for specified task types"""
        logger.info(f"Warming cache for task types: {task_types}")

        refresh_tasks = []
        for task_type in task_types:
            refresh_tasks.append(self._refresh_cache_async(task_type, force=True))

        await asyncio.gather(*refresh_tasks, return_exceptions=True)

        logger.info("Cache warm-up complete")
