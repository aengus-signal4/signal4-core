"""
Network Monitor - Monitors worker network connectivity and health with exponential backoff retry
"""
import asyncio
import aiohttp
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict

logger = logging.getLogger(__name__)

class NetworkMonitor:
    """Monitors network connectivity and worker health with sophisticated retry logic"""
    
    def __init__(self, network_manager, service_manager=None):
        self.network_manager = network_manager
        self.service_manager = service_manager
        self.should_stop = False
        self._monitor_task = None

        # Initialize workers dict - will be populated by start_monitoring
        self.workers = {}
        self.get_session = None

        # Monitoring settings
        self.health_check_interval = 30  # seconds
        self.retry_check_interval = 60  # seconds - check for retry opportunities more frequently

        # Pause restart attempts during deployment
        self.restart_paused = False
        
    async def start_monitoring(self, workers: Dict[str, Any], get_session_func):
        """Start network monitoring"""
        self.workers = workers
        self.get_session = get_session_func
        self.should_stop = False
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Started network monitoring with exponential backoff retry")
    
    async def stop_monitoring(self):
        """Stop network monitoring"""
        self.should_stop = True
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped network monitoring")
    
    async def _monitor_loop(self):
        """Main monitoring loop with enhanced retry logic"""
        while not self.should_stop:
            try:
                # Check health of all workers and handle failed workers
                await self._check_all_workers_health()
                
                # Check for retry opportunities for failed workers
                await self._check_retry_opportunities()
                
                # Wait before next check
                await asyncio.sleep(self.health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in network monitoring loop: {str(e)}")
                await asyncio.sleep(self.health_check_interval)
    
    async def _check_all_workers_health(self):
        """Check health of all workers and handle issues"""
        health_results = await self.network_manager.perform_bulk_health_check()
        
        for worker_id, is_healthy in health_results.items():
            if worker_id not in self.workers:
                continue
                
            worker = self.workers[worker_id]
            
            # Update worker status based on health
            if is_healthy:
                if worker.status in ['unhealthy', 'failed', 'network_issue']:
                    logger.info(f"Worker {worker_id} recovered and is now healthy")
                    worker.status = 'active'
                    worker.last_heartbeat = datetime.now(timezone.utc)
                    
                    # Reset restart attempts on recovery
                    if hasattr(worker, 'record_restart_attempt'):
                        worker.record_restart_attempt(True)
            else:
                if worker.status == 'active':
                    logger.warning(f"Worker {worker_id} became unhealthy")
                    worker.status = 'unhealthy'
                    
                    # Attempt immediate restart for newly unhealthy worker
                    await self._attempt_worker_restart(worker_id, worker)
                elif worker.status in ['unhealthy', 'failed']:
                    # Worker is still unhealthy, will be handled by retry check
                    pass
    
    async def _check_retry_opportunities(self):
        """Check for workers that can be retried based on backoff logic"""
        for worker_id, worker in self.workers.items():
            if worker.status in ['unhealthy', 'failed'] and hasattr(worker, 'should_attempt_restart'):
                if worker.should_attempt_restart():
                    backoff_remaining = worker.get_restart_backoff_remaining()
                    if backoff_remaining <= 0:
                        logger.info(f"Retry opportunity detected for worker {worker_id} "
                                  f"(attempt {worker.restart_attempts + 1}/{worker.max_restart_attempts})")
                        await self._attempt_worker_restart(worker_id, worker)
                    else:
                        logger.debug(f"Worker {worker_id} in backoff period, {backoff_remaining:.1f}s remaining")
                elif worker.restart_attempts >= worker.max_restart_attempts and worker.status != 'permanently_failed':
                    logger.error(f"Worker {worker_id} has exceeded maximum restart attempts "
                               f"({worker.max_restart_attempts}). Marking as permanently failed.")
                    worker.status = 'permanently_failed'
                    await self._reset_worker_tasks(worker_id)
    
    async def _attempt_worker_restart(self, worker_id: str, worker) -> bool:
        """Attempt to restart a worker with proper retry tracking"""
        # Skip if restart is paused (during deployment)
        if self.restart_paused:
            logger.debug(f"Restart paused - skipping restart for worker {worker_id}")
            return False

        try:
            logger.info(f"Attempting to restart worker {worker_id} "
                       f"(attempt {worker.restart_attempts + 1}/{worker.max_restart_attempts})")
            
            # Mark worker as restarting
            worker.status = 'restarting'
            
            # Use the worker_manager's enhanced restart functionality if available
            # This will kill all sub-processes and reset tasks to pending
            restart_success = False
            
            # Check if we have access to worker_manager through service_manager
            if hasattr(self, 'worker_manager') and self.worker_manager:
                try:
                    logger.debug(f"Using worker_manager enhanced restart for worker {worker_id}")
                    restart_success = await self.worker_manager.restart_unhealthy_worker(worker_id)
                except Exception as e:
                    logger.error(f"Error using worker_manager restart: {str(e)}")
                    restart_success = False
            elif self.service_manager:
                # Fallback to service_manager approach (LLM server management disabled)
                try:
                    logger.debug(f"Stopping services for worker {worker_id} (LLM server management disabled)")
                    await self.service_manager.stop_task_processor(worker_id)
                    # Skip LLM server stop - managed independently
                    
                    # Wait a moment for services to stop
                    await asyncio.sleep(5)
                    
                    # Get worker task types from worker info
                    task_types = getattr(worker, 'task_types', [])
                    logger.debug(f"Starting services for worker {worker_id} with task types: {task_types} (LLM server management disabled)")
                    restart_success = await self.service_manager.ensure_required_services(worker_id, task_types)
                except Exception as e:
                    logger.error(f"Error restarting via service_manager: {str(e)}")
                    restart_success = False
            else:
                logger.warning(f"No service manager or worker manager available to restart worker {worker_id}")
                restart_success = False
            
            # Record the restart attempt
            worker.record_restart_attempt(restart_success)
            
            if restart_success:
                worker.status = 'active'
                worker.last_heartbeat = datetime.now(timezone.utc)
                logger.info(f"Successfully restarted worker {worker_id}")
                return True
            else:
                # Determine new status based on retry availability
                if worker.restart_attempts >= worker.max_restart_attempts:
                    worker.status = 'permanently_failed'
                    logger.error(f"Worker {worker_id} permanently failed after {worker.max_restart_attempts} restart attempts")
                    # Only reset tasks if worker_manager didn't already handle it
                    if not (hasattr(self, 'worker_manager') and self.worker_manager):
                        await self._reset_worker_tasks(worker_id)
                else:
                    worker.status = 'failed'
                    next_backoff = worker.get_restart_backoff_remaining()
                    logger.warning(f"Worker {worker_id} restart failed. Next retry in {next_backoff:.1f}s")
                return False
                
        except Exception as e:
            logger.error(f"Error attempting restart for worker {worker_id}: {str(e)}")
            # Still record the failed attempt
            if hasattr(worker, 'record_restart_attempt'):
                worker.record_restart_attempt(False)
            worker.status = 'failed'
            return False
    
    async def _reset_worker_tasks(self, worker_id: str):
        """Reset tasks assigned to a failed worker"""
        try:
            from sqlalchemy import text
            
            with self.get_session() as session:
                # Find all processing tasks for this worker
                query = text("""
                    SELECT id, task_type, content_id
                    FROM tasks.task_queue 
                    WHERE worker_id = :worker_id AND status = 'processing'
                """)
                
                tasks = session.execute(query, {'worker_id': worker_id}).fetchall()
                
                if tasks:
                    logger.info(f"Resetting {len(tasks)} tasks from failed worker {worker_id}")
                    
                    # Reset tasks to pending
                    update_query = text("""
                        UPDATE tasks.task_queue 
                        SET status = 'pending', 
                            worker_id = NULL,
                            error = 'Worker failed permanently during processing',
                            updated_at = NOW()
                        WHERE worker_id = :worker_id AND status = 'processing'
                    """)
                    
                    session.execute(update_query, {'worker_id': worker_id})
                    session.commit()
                    
                    # Also remove tasks from worker's tracking
                    if worker_id in self.workers:
                        worker = self.workers[worker_id]
                        if hasattr(worker, 'assigned_task_ids'):
                            worker.assigned_task_ids.clear()
                            worker.active_tasks = 0
                            if hasattr(worker, 'current_tasks_by_type'):
                                for task_set in worker.current_tasks_by_type.values():
                                    task_set.clear()
                    
                    logger.info(f"Successfully reset {len(tasks)} tasks from worker {worker_id}")
                    
        except Exception as e:
            logger.error(f"Error resetting tasks for worker {worker_id}: {str(e)}")
    
    async def manual_restart_worker(self, worker_id: str) -> bool:
        """Manually trigger a worker restart, ignoring backoff logic"""
        if worker_id not in self.workers:
            logger.error(f"Worker {worker_id} not found for manual restart")
            return False
        
        worker = self.workers[worker_id]
        logger.info(f"Manual restart requested for worker {worker_id}")
        
        # Reset retry tracking for manual restart
        if hasattr(worker, 'restart_attempts'):
            worker.restart_attempts = 0
            worker.last_restart_attempt = None
        
        return await self._attempt_worker_restart(worker_id, worker)
    
    async def check_worker_health(self, worker_id: str) -> bool:
        """Check health of a specific worker"""
        try:
            if worker_id not in self.workers:
                return False
            
            # Use network manager to check health
            result = await self.network_manager.check_worker_health(worker_id)
            return result
            
        except Exception as e:
            logger.error(f"Error checking health for worker {worker_id}: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get network monitoring statistics including retry info"""
        stats = {
            'total_workers': len(self.workers),
            'active_workers': len([w for w in self.workers.values() if w.status == 'active']),
            'failed_workers': len([w for w in self.workers.values() if w.status in ['failed', 'permanently_failed']]),
            'restarting_workers': len([w for w in self.workers.values() if w.status == 'restarting']),
            'worker_details': {}
        }
        
        for worker_id, worker in self.workers.items():
            worker_info = {
                'status': worker.status,
                'last_heartbeat': worker.last_heartbeat.isoformat() if worker.last_heartbeat else None
            }
            
            # Add retry information if available
            if hasattr(worker, 'restart_attempts'):
                worker_info.update({
                    'restart_attempts': worker.restart_attempts,
                    'max_restart_attempts': worker.max_restart_attempts,
                    'backoff_remaining': worker.get_restart_backoff_remaining(),
                    'can_retry': worker.should_attempt_restart()
                })
            
            stats['worker_details'][worker_id] = worker_info
        
        return stats