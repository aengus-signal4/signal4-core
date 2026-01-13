"""
Reactive Assignment - Immediately assigns new tasks when workers complete tasks
"""
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Set

logger = logging.getLogger(__name__)

class ReactiveAssignmentManager:
    """Manages immediate task assignment when workers complete tasks"""
    
    def __init__(self, task_manager, orchestrator):
        self.task_manager = task_manager
        self.orchestrator = orchestrator
        self.pending_assignments: Set[str] = set()  # Track workers with pending assignments
        
    async def assign_next_task(self, worker_id: str, completed_task_type: str) -> bool:
        """
        Trigger immediate queue backfill check when a task completes.
        If worker supports batch API, send batch refill. Otherwise fall back to single-task assignment.
        """
        # Prevent duplicate assignments
        if worker_id in self.pending_assignments:
            logger.debug(f"Backfill already pending for worker {worker_id}")
            return False

        self.pending_assignments.add(worker_id)

        try:
            # Check global pause first
            if self.orchestrator.global_pause_until:
                now = datetime.now(timezone.utc)
                if now < self.orchestrator.global_pause_until:
                    remaining_pause = (self.orchestrator.global_pause_until - now).total_seconds()
                    logger.info(f"Queue backfill for worker {worker_id} delayed due to global pause ({remaining_pause:.1f}s remaining)")

                    # Schedule retry after pause
                    asyncio.create_task(self._retry_assignment_after_pause(worker_id, completed_task_type, remaining_pause))
                    return False
                else:
                    # Pause has expired
                    logger.info("Global pause expired during backfill check")
                    self.orchestrator.global_pause_until = None

            if worker_id not in self.orchestrator.worker_pool.workers:
                logger.warning(f"Worker {worker_id} not found for queue backfill")
                return False

            worker = self.orchestrator.worker_pool.workers[worker_id]

            # Check if worker is ready for new tasks
            if worker.status != 'active':
                logger.debug(f"Worker {worker_id} not active for backfill (status: {worker.status})")
                return False

            # If worker doesn't support batch API, fall back to single-task assignment
            if worker.supports_batch_api is False:
                logger.debug(f"Worker {worker_id} doesn't support batch API, using single-task fallback")
                return await self._single_task_fallback(worker, completed_task_type)

            # Check queue status immediately for batch-enabled workers
            success = await self._check_and_backfill_queues(worker)
            return success

        except Exception as e:
            logger.error(f"Error in queue backfill for worker {worker_id}: {str(e)}")
            return False
        finally:
            self.pending_assignments.discard(worker_id)

    async def _check_and_backfill_queues(self, worker) -> bool:
        """Check worker queue status and send batch refill if needed"""
        try:
            # Get current queue status
            success, response = await self.orchestrator.network_manager.send_request(
                worker.worker_id,
                'GET',
                '/tasks/queue_status',
                timeout=5.0
            )

            if not success:
                logger.warning(f"Failed to get queue status from worker {worker.worker_id}: {response}")
                return False

            if not isinstance(response, dict):
                logger.warning(f"Invalid queue status response from worker {worker.worker_id}")
                return False

            # Update queue depths
            worker.queue_depths_by_type = response.get('queue_depths', {})
            worker.last_queue_status_check = datetime.now(timezone.utc)

            # Check all task types for backfill needs
            batch_tasks_to_send = []

            for task_type in worker.task_types:
                # Skip download tasks - main loop handles them with human behavior delays
                if task_type in ['download_youtube', 'download_rumble']:
                    logger.debug(f"[REACTIVE] Skipping {task_type} for worker {worker.worker_id} (main loop handles with human behavior)")
                    continue

                # Skip paused task types
                if self.orchestrator.failure_tracker.is_task_type_paused(worker.worker_id, task_type):
                    continue

                # Check if this task type needs backfill
                if worker.needs_backfill(task_type):
                    backfill_count = worker.get_backfill_count(task_type)
                    logger.debug(f"Worker {worker.worker_id} needs {backfill_count} tasks for {task_type}")

                    if backfill_count > 0:
                        # Get pending tasks for this type
                        potential_tasks = await self.task_manager.get_next_tasks(
                            limit=backfill_count,
                            task_types=[task_type],
                            exclude_task_ids=self.task_manager.assigned_tasks
                        )

                        # Build batch requests
                        for task in potential_tasks:
                            task_id = str(task['id'])

                            # Mark as assigned
                            if not await self.task_manager.mark_task_assigned(task_id, worker.worker_id):
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
                batch_success, batch_response = await self.orchestrator.network_manager.send_request(
                    worker.worker_id,
                    'POST',
                    '/tasks/batch',
                    {'tasks': batch_tasks_to_send}
                )

                if batch_success:
                    logger.info(f"Backfilled {len(batch_tasks_to_send)} tasks to worker {worker.worker_id}")
                    # Reset failure tracking
                    for task_req in batch_tasks_to_send:
                        self.orchestrator.failure_tracker.record_success(worker.worker_id, task_req['task_type'])
                    return True
                else:
                    # Rollback on failure
                    logger.error(f"Failed to send backfill batch to worker {worker.worker_id}: {batch_response}")
                    for task_req in batch_tasks_to_send:
                        task_id = task_req['original_task_id']
                        worker.remove_task(task_id, task_req['task_type'])
                        await self.task_manager.handle_task_failure(
                            task_id, worker.worker_id, task_req['task_type'],
                            "Batch backfill failed"
                        )
                    return False
            else:
                logger.debug(f"No backfill needed for worker {worker.worker_id}")
                return True

        except Exception as e:
            logger.error(f"Error in queue backfill for worker {worker.worker_id}: {str(e)}")
            return False

    async def _single_task_fallback(self, worker, completed_task_type: str) -> bool:
        """Fallback to single-task assignment for workers without batch support"""
        # Check if worker can accept more tasks
        if worker.active_tasks >= worker.max_concurrent_tasks:
            logger.debug(f"Worker {worker.worker_id} at capacity for single-task assignment")
            return False

        # Get next suitable task
        task = await self._find_suitable_task(worker, completed_task_type)

        if task:
            # Assign the task
            success = await self._assign_task_to_worker(task, worker)
            if success:
                logger.info(f"Single-task fallback assigned task {task['id']} ({task['task_type']}) to worker {worker.worker_id}")
                return True
            else:
                logger.warning(f"Failed to assign fallback task {task['id']} to worker {worker.worker_id}")
                return False
        else:
            logger.debug(f"No suitable tasks found for fallback assignment to worker {worker.worker_id}")
            return False
    
    async def _find_suitable_task(self, worker, completed_task_type: str) -> Optional[Dict[str, Any]]:
        """Find a suitable task for the worker, prioritizing same task type"""

        # Check if worker can handle the completed task type again
        if (completed_task_type in worker.task_types and
            worker.can_accept_task(completed_task_type) and
            await self.task_manager.can_assign_task_to_worker(worker.worker_id, completed_task_type)):

            # Look for same task type first (using cache for speed)
            task = await self.task_manager.get_next_task_cached(
                task_types=[completed_task_type],
                exclude_task_ids=self.task_manager.assigned_tasks
            )

            if task:
                logger.debug(f"Found same-type task {task['id']} for worker {worker.worker_id} (cached)")
                return task

        # If no same-type task, look for any suitable task
        suitable_task_types = []
        for task_type in worker.task_types:
            if (worker.can_accept_task(task_type) and
                await self.task_manager.can_assign_task_to_worker(worker.worker_id, task_type)):
                suitable_task_types.append(task_type)

        if suitable_task_types:
            # Use cached lookup for fast reactive assignment
            task = await self.task_manager.get_next_task_cached(
                task_types=suitable_task_types,
                exclude_task_ids=self.task_manager.assigned_tasks
            )

            if task:
                logger.debug(f"Found alternative task {task['id']} ({task['task_type']}) for worker {worker.worker_id} (cached)")
                return task

        return None
    
    async def _assign_task_to_worker(self, task: Dict[str, Any], worker) -> bool:
        """Assign a specific task to a worker"""
        try:
            task_id = str(task['id'])
            
            # Mark task as assigned
            success = await self.task_manager.mark_task_assigned(task_id, worker.worker_id)
            if not success:
                return False
            
            # Add to worker's task list
            if not worker.add_task(task_id, task['task_type']):
                # Rollback assignment
                await self.task_manager.handle_task_failure(
                    task_id, worker.worker_id, task['task_type'], 
                    "Worker at capacity"
                )
                return False
            
            # Send task to worker via network manager
            api_success, response = await self.orchestrator.network_manager.send_request(
                worker.worker_id,
                'POST',
                '/tasks/process',
                {
                    'content_id': task['content_id'],
                    'task_type': task['task_type'],
                    'input_data': task['input_data'],
                    'worker_id': worker.worker_id,
                    'original_task_id': task_id
                }
            )
            
            if api_success:
                return True
            else:
                # Rollback on API failure
                worker.remove_task(task_id, task['task_type'])
                await self.task_manager.handle_task_failure(
                    task_id, worker.worker_id, task['task_type'], 
                    f"API call failed: {response}"
                )
                return False
                
        except Exception as e:
            logger.error(f"Error assigning task {task['id']} to worker {worker.worker_id}: {str(e)}")
            return False
    
    async def _retry_assignment_after_pause(self, worker_id: str, completed_task_type: str, delay_seconds: float):
        """Retry assignment after global pause ends"""
        try:
            if delay_seconds > 0:
                await asyncio.sleep(delay_seconds + 0.1)  # Small buffer
            
            logger.info(f"Retrying reactive assignment for worker {worker_id} after global pause")
            await self.assign_next_task(worker_id, completed_task_type)
            
        except asyncio.CancelledError:
            logger.debug(f"Retry assignment cancelled for worker {worker_id}")
        except Exception as e:
            logger.error(f"Error retrying assignment for worker {worker_id}: {str(e)}")
    
    async def handle_post_task_delay(self, worker_id: str, completed_task_type: str, delay_seconds: float):
        """Handle post-task delay before attempting assignment"""
        try:
            if worker_id not in self.orchestrator.worker_pool.workers:
                logger.warning(f"Worker {worker_id} not found for post-task delay")
                return
                
            worker = self.orchestrator.worker_pool.workers[worker_id]
            
            if delay_seconds > 0:
                logger.info(f"Applying post-task delay of {delay_seconds:.1f}s for worker {worker_id}")
                
                # Set delay flag
                worker.available_after = datetime.now(timezone.utc) + timedelta(seconds=delay_seconds)
                
                # Wait for delay
                await asyncio.sleep(delay_seconds)
                
                # Clear delay flag
                worker.available_after = None
            
            # Attempt assignment after delay
            await self.assign_next_task(worker_id, completed_task_type)
            
        except asyncio.CancelledError:
            logger.debug(f"Post-task delay cancelled for worker {worker_id}")
        except Exception as e:
            logger.error(f"Error in post-task delay for worker {worker_id}: {str(e)}")
        finally:
            # Ensure delay flag is cleared
            if worker_id in self.orchestrator.worker_pool.workers:
                self.orchestrator.worker_pool.workers[worker_id].available_after = None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get reactive assignment statistics"""
        return {
            'pending_assignments': len(self.pending_assignments),
            'pending_workers': list(self.pending_assignments)
        }