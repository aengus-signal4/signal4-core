"""
Background Loops - Modular background task loops for the orchestrator
"""
import asyncio
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from src.orchestration.orchestrator import TaskOrchestratorV2

logger = logging.getLogger(__name__)

class StatsCollector:
    """Collects and logs hourly statistics"""
    
    def __init__(self, orchestrator: 'TaskOrchestratorV2', run_logger: logging.Logger):
        self.orchestrator = orchestrator
        self.run_logger = run_logger
        self.interval = 3600  # 1 hour
    
    async def run(self):
        """Run the hourly stats collection loop"""
        logger.info("Starting hourly stats loop")
        
        while not self.orchestrator.should_stop:
            try:
                # Wait for an hour
                await asyncio.sleep(self.interval)
                
                # Collect stats
                pool_stats = self.orchestrator.worker_pool.get_stats()
                
                # Log hourly summary
                self.run_logger.info(
                    f"Hourly summary - Active workers: {pool_stats['active_workers']}/{pool_stats['total_workers']}"
                )
                
                # Get task stats from task manager
                if hasattr(self.orchestrator.task_manager, 'get_stats'):
                    task_stats = await self.orchestrator.task_manager.get_stats()
                    if task_stats:
                        self.run_logger.info(
                            f"Tasks - Pending: {task_stats.get('pending', 0)}, "
                            f"Processing: {task_stats.get('processing', 0)}, "
                            f"Completed: {task_stats.get('completed_last_hour', 0)}, "
                            f"Failed: {task_stats.get('failed_last_hour', 0)}"
                        )
                
                # Log task distribution
                if pool_stats['task_distribution']:
                    self.run_logger.info(f"Task distribution: {pool_stats['task_distribution']}")
                
                # Log active worker details
                for worker_id, worker_status in pool_stats['workers'].items():
                    if worker_status['status'] == 'active' and worker_status['active_tasks'] > 0:
                        self.run_logger.info(
                            f"Worker {worker_id} active tasks: {worker_status['task_counts_by_type']}"
                        )
                
            except Exception as e:
                logger.error(f"Error in hourly stats loop: {str(e)}")
                await asyncio.sleep(self.interval)


class ConfigReloader:
    """Handles configuration hot-reloading"""
    
    def __init__(self, orchestrator: 'TaskOrchestratorV2'):
        self.orchestrator = orchestrator
        self.interval = 30  # seconds
    
    async def run(self):
        """Run the config reload checking loop"""
        logger.info("Starting config reload loop")
        
        while not self.orchestrator.should_stop:
            try:
                # Check and reload config if changed
                reloaded = self.orchestrator.config_manager.reload_config()
                
                if reloaded:
                    # Update worker configurations
                    worker_configs = self.orchestrator.config_manager.get_worker_configs()
                    self.orchestrator.worker_pool.update_all_from_config(worker_configs)
                
                await asyncio.sleep(self.interval)
                
            except Exception as e:
                logger.error(f"Error in config reload loop: {str(e)}")
                await asyncio.sleep(self.interval)


class TaskCleaner:
    """Handles cleanup of stale tasks and failed worker tasks"""
    
    def __init__(self, orchestrator: 'TaskOrchestratorV2'):
        self.orchestrator = orchestrator
        self.interval = 300  # 5 minutes
        self.stale_task_timeout = 30  # minutes
    
    async def run(self):
        """Run the cleanup loop"""
        logger.info("Starting cleanup loop")
        
        while not self.orchestrator.should_stop:
            try:
                # Clean up stale tasks
                cleaned_tasks = await self.orchestrator.task_manager.cleanup_stale_tasks(
                    timeout_minutes=self.stale_task_timeout
                )
                if cleaned_tasks > 0:
                    logger.info(f"Cleaned up {cleaned_tasks} stale tasks")
                
                # Clean up failed worker tasks
                tasks_to_cleanup = self.orchestrator.worker_pool.cleanup_failed_worker_tasks()
                if tasks_to_cleanup:
                    logger.info(f"Cleaned up {len(tasks_to_cleanup)} tasks from failed workers")
                
                await asyncio.sleep(self.interval)
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {str(e)}")
                await asyncio.sleep(self.interval)


class HealthChecker:
    """Performs health checks on workers and services"""

    def __init__(self, orchestrator: 'TaskOrchestratorV2'):
        self.orchestrator = orchestrator
        self.interval = 30  # seconds

    async def run(self):
        """Run the health check loop.

        Note: Worker network health is handled by NetworkMonitor which has
        exponential backoff retry logic. This loop only checks service-level
        health (task processor) and head node services.
        """
        logger.info("Starting health check loop (service-level checks only)")

        while not self.orchestrator.should_stop:
            try:
                # Check head node service health
                await self._check_head_node_services()

                await asyncio.sleep(self.interval)

            except Exception as e:
                logger.error(f"Error in health check loop: {str(e)}")
                await asyncio.sleep(self.interval)

    async def _check_head_node_services(self):
        """Check health of head node services and trigger restarts if needed."""
        if not hasattr(self.orchestrator, 'head_node_monitor'):
            return

        try:
            monitor = self.orchestrator.head_node_monitor

            # Check all services
            health_results = await monitor.check_all_services_health()

            # Handle unhealthy services
            for service_name, is_healthy in health_results.items():
                if not is_healthy:
                    service = monitor.services.get(service_name)
                    if service:
                        logger.warning(
                            f"Head node service {service_name} unhealthy "
                            f"({service.consecutive_failures}/{service.failure_threshold} failures)"
                        )

                        # Attempt restart if threshold exceeded
                        if service.needs_restart():
                            logger.info(f"Attempting to restart head node service {service_name}")
                            await monitor.handle_unhealthy_service(service_name)

        except Exception as e:
            logger.error(f"Error checking head node services: {str(e)}")


class TaskAssigner:
    """Handles the main task assignment loop"""

    def __init__(self, orchestrator: 'TaskOrchestratorV2'):
        self.orchestrator = orchestrator
        self.interval = 5  # seconds
        self.idle_interval = 10  # seconds
        self.queue_status_check_interval = 30  # seconds - check queue status every 30s

    async def _check_worker_queue_status(self, worker) -> bool:
        """Check worker queue status and update queue depths"""
        try:
            success, response = await self.orchestrator.network_manager.send_request(
                worker.worker_id,
                'GET',
                '/tasks/queue_status',
                timeout=5.0
            )

            if success and isinstance(response, dict):
                worker.queue_depths_by_type = response.get('queue_depths', {})
                worker.supports_batch_api = True
                worker.last_queue_status_check = datetime.now(timezone.utc)
                logger.debug(f"Worker {worker.worker_id} queue depths: {worker.queue_depths_by_type}")
                return True
            elif not success and '404' in str(response):
                # Worker doesn't support batch API
                worker.supports_batch_api = False
                logger.debug(f"Worker {worker.worker_id} doesn't support batch API")
                return False
            else:
                logger.warning(f"Failed to get queue status from worker {worker.worker_id}: {response}")
                return False

        except Exception as e:
            logger.error(f"Error checking queue status for worker {worker.worker_id}: {str(e)}")
            return False

    async def _send_task_batch_to_processor(self, worker, batch_tasks: list) -> list:
        """Send batch of tasks to worker processor"""
        try:
            success, response = await self.orchestrator.network_manager.send_request(
                worker.worker_id,
                'POST',
                '/tasks/batch',
                {'tasks': batch_tasks}
            )

            if success and isinstance(response, dict):
                task_ids = response.get('task_ids', [])
                logger.info(f"Sent batch of {len(task_ids)} tasks to worker {worker.worker_id}")
                return task_ids
            else:
                logger.error(f"Failed to send batch to worker {worker.worker_id}: {response}")
                return []

        except Exception as e:
            logger.error(f"Error sending batch to worker {worker.worker_id}: {str(e)}")
            return []

    async def run(self):
        """Run the task assignment loop"""
        logger.info("Starting task assignment loop")

        while not self.orchestrator.should_stop:
            try:
                # Check if globally paused
                if self.orchestrator.global_pause_until and datetime.now(timezone.utc) < self.orchestrator.global_pause_until:
                    await asyncio.sleep(self.idle_interval)
                    continue
                elif self.orchestrator.global_pause_until:
                    logger.info("Global pause ended, resuming task assignment")
                    self.orchestrator.global_pause_until = None

                # Get available workers
                available_workers = self.orchestrator.worker_pool.get_available_workers()

                if not available_workers:
                    await asyncio.sleep(self.idle_interval)
                    continue

                # Process each worker (batch or single-task mode)
                assignments_made = 0

                for worker in available_workers:
                    if self.orchestrator.should_stop:
                        break

                    # Check if we should check queue status
                    should_check = (
                        worker.supports_batch_api is None or
                        (worker.last_queue_status_check is None) or
                        (datetime.now(timezone.utc) - worker.last_queue_status_check).total_seconds() >= self.queue_status_check_interval
                    )

                    # Try batch dispatch if worker supports it (or unknown)
                    if worker.supports_batch_api is not False:
                        if should_check:
                            await self._check_worker_queue_status(worker)

                        # If batch API is supported, use batch dispatch
                        if worker.supports_batch_api:
                            batch_tasks_to_send = []

                            # Check each task type for backfill needs
                            for task_type in worker.task_types:
                                # Skip download tasks - they use single-task assignment with human behavior delays
                                if task_type in ['download_youtube', 'download_rumble']:
                                    continue

                                if self.orchestrator.failure_tracker.is_task_type_paused(worker.worker_id, task_type):
                                    continue

                                if worker.needs_backfill(task_type):
                                    backfill_count = worker.get_backfill_count(task_type)

                                    if backfill_count > 0:
                                        # Get pending tasks for this type
                                        potential_tasks = await self.orchestrator.task_manager.get_next_tasks(
                                            limit=backfill_count,
                                            task_types=[task_type],
                                            exclude_task_ids=self.orchestrator.task_manager.assigned_tasks
                                        )

                                        # Build batch request for each task
                                        for task in potential_tasks:
                                            task_id = str(task['id'])

                                            # Mark as assigned
                                            success = await self.orchestrator.task_manager.mark_task_assigned(
                                                task_id, worker.worker_id
                                            )

                                            if not success:
                                                continue

                                            # Prepare input_data
                                            input_data = task.get('input_data', {})
                                            if isinstance(input_data, str):
                                                try:
                                                    import json
                                                    input_data = json.loads(input_data)
                                                except:
                                                    input_data = {}

                                            # Add cookie profile for YouTube downloads
                                            if task_type == 'download_youtube':
                                                cookie_profile = self.orchestrator.behavior_manager.get_cookie_profile(worker.worker_id)
                                                if cookie_profile:
                                                    input_data['cookie_profile'] = cookie_profile

                                            # Add orchestrator callback URL
                                            input_data['orchestrator_callback_url'] = self.orchestrator.callback_url

                                            batch_tasks_to_send.append({
                                                'content_id': task['content_id'],
                                                'task_type': task_type,
                                                'input_data': input_data,
                                                'worker_id': worker.worker_id,
                                                'original_task_id': task_id
                                            })

                                            # Add to worker's task tracking
                                            worker.add_task(task_id, task_type)

                            # Send batch if we have tasks
                            if batch_tasks_to_send:
                                api_task_ids = await self._send_task_batch_to_processor(worker, batch_tasks_to_send)

                                if api_task_ids:
                                    assignments_made += len(api_task_ids)
                                    # Reset failure tracking on success
                                    for task_req in batch_tasks_to_send:
                                        self.orchestrator.failure_tracker.record_success(
                                            worker.worker_id, task_req['task_type']
                                        )
                                else:
                                    # Batch failed - rollback assignments
                                    for task_req in batch_tasks_to_send:
                                        task_id = task_req['original_task_id']
                                        worker.remove_task(task_id, task_req['task_type'])
                                        await self.orchestrator.task_manager.handle_task_failure(
                                            task_id, worker.worker_id, task_req['task_type'],
                                            "Batch dispatch failed"
                                        )

                            continue  # Skip to next worker

                # Handle download tasks separately for ALL workers (single-task with human behavior checks)
                for worker in available_workers:
                    if self.orchestrator.should_stop:
                        break

                    for task_type in ['download_youtube', 'download_rumble']:
                        if task_type not in worker.task_types:
                            continue

                        # Check if paused
                        if self.orchestrator.failure_tracker.is_task_type_paused(worker.worker_id, task_type):
                            continue

                        # Check capacity
                        if not worker.can_accept_task(task_type):
                            logger.debug(f"[DOWNLOAD] Worker {worker.worker_id} at capacity for {task_type}")
                            continue

                        # Check human behavior delay
                        wait_time, reason = self.orchestrator.behavior_manager.calculate_next_task_wait_time(
                            worker.worker_id,
                            task_type=task_type
                        )
                        if wait_time > 0:
                            logger.debug(f"[DOWNLOAD] Skipping {task_type} for worker {worker.worker_id}: {reason} (wait {wait_time:.1f}s)")
                            continue

                        logger.info(f"[DOWNLOAD] Assigning {task_type} to worker {worker.worker_id} (wait_time=0, reason='{reason}')")

                        # Get exactly 1 download task
                        tasks = await self.orchestrator.task_manager.get_next_tasks(
                            limit=1,
                            task_types=[task_type],
                            exclude_task_ids=self.orchestrator.task_manager.assigned_tasks
                        )

                        if tasks:
                            task = tasks[0]
                            success = await self.orchestrator._assign_task_to_worker(task, worker)
                            if success:
                                assignments_made += 1
                                break  # Only assign 1 download per worker per cycle

                    # Fallback to single-task mode (for workers without batch support)
                    if worker.supports_batch_api is False:
                        # Get next task for this worker's capabilities
                        tasks = await self.orchestrator.task_manager.get_next_tasks(
                            limit=1,
                            task_types=worker.task_types,
                            exclude_task_ids=self.orchestrator.task_manager.assigned_tasks
                        )

                        if tasks:
                            task = tasks[0]
                            task_type = task['task_type']

                            # Check if task type is paused
                            if not self.orchestrator.failure_tracker.is_task_type_paused(
                                worker.worker_id, task_type
                            ):
                                # Check if worker can accept
                                if worker.can_accept_task(task_type):
                                    success = await self.orchestrator._assign_task_to_worker(task, worker)
                                    if success:
                                        assignments_made += 1

                if assignments_made > 0:
                    logger.info(f"Assigned {assignments_made} tasks in this cycle")

                await asyncio.sleep(self.interval)

            except Exception as e:
                logger.error(f"Error in task assignment loop: {str(e)}")
                await asyncio.sleep(self.idle_interval)