"""
Worker Pool - Manages the pool of workers
"""
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

from .info import WorkerInfo

logger = logging.getLogger(__name__)

class WorkerPool:
    """Manages the pool of workers"""
    
    def __init__(self):
        self.workers: Dict[str, WorkerInfo] = {}
    
    async def add_worker(self, worker_id: str, config: Dict[str, Any]) -> bool:
        """Add a worker to the pool"""
        try:
            if worker_id in self.workers:
                logger.warning(f"Worker {worker_id} already exists in pool")
                return False
            
            worker = WorkerInfo(worker_id, config)
            self.workers[worker_id] = worker
            logger.info(f"Added worker {worker_id} to pool")
            return True
        except Exception as e:
            logger.error(f"Error adding worker {worker_id}: {str(e)}")
            return False
    
    def get_worker(self, worker_id: str) -> Optional[WorkerInfo]:
        """Get a worker by ID"""
        return self.workers.get(worker_id)
    
    def get_all_workers(self) -> Dict[str, WorkerInfo]:
        """Get all workers"""
        return self.workers
    
    def get_active_workers(self) -> List[WorkerInfo]:
        """Get all active workers"""
        return [w for w in self.workers.values() if w.status == 'active']
    
    def get_available_workers(self, task_type: Optional[str] = None) -> List[WorkerInfo]:
        """Get available workers, optionally filtered by task type"""
        available = []
        
        for worker in self.workers.values():
            if worker.status != 'active':
                continue
            
            if worker.active_tasks >= worker.max_concurrent_tasks:
                continue
            
            if task_type and not worker.can_accept_task(task_type):
                continue
            
            available.append(worker)
        
        return available
    
    def find_best_worker(self, task_type: str, failure_check_func=None) -> Optional[WorkerInfo]:
        """Find the best available worker for a task type"""
        candidates = []
        
        for worker in self.workers.values():
            if not worker.can_accept_task(task_type):
                continue
            
            # Optional failure check
            if failure_check_func and failure_check_func(worker.worker_id, task_type):
                continue
            
            candidates.append(worker)
        
        if not candidates:
            return None
        
        # Return worker with fewest active tasks (simple load balancing)
        return min(candidates, key=lambda w: w.active_tasks)
    
    def update_worker_config(self, worker_id: str, new_config: Dict[str, Any]) -> bool:
        """Update a worker's configuration"""
        worker = self.get_worker(worker_id)
        if not worker:
            logger.warning(f"Worker {worker_id} not found for config update")
            return False
        
        worker.update_from_config(new_config)
        return True
    
    def update_all_from_config(self, worker_configs: Dict[str, Dict[str, Any]]):
        """Update all workers from config, adding new ones if needed"""
        # Update existing workers
        for worker_id, config in worker_configs.items():
            if worker_id in self.workers:
                self.update_worker_config(worker_id, config)
            else:
                # Optionally add new workers
                if config.get('enabled', False):
                    logger.info(f"New worker {worker_id} found in config")
                    # Note: Adding would require async, so just log for now
        
        # Remove workers not in config
        workers_to_remove = []
        for worker_id in self.workers:
            if worker_id not in worker_configs:
                logger.warning(f"Worker {worker_id} no longer in config")
                workers_to_remove.append(worker_id)
        
        for worker_id in workers_to_remove:
            del self.workers[worker_id]
    
    def remove_task_from_worker(self, worker_id: str, task_id: str, task_type: Optional[str] = None) -> bool:
        """Remove a task from a worker"""
        worker = self.get_worker(worker_id)
        if not worker:
            return False
        
        return worker.remove_task(task_id, task_type)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        active_workers = self.get_active_workers()
        total_active_tasks = sum(w.active_tasks for w in active_workers)
        
        task_distribution = {}
        for worker in active_workers:
            for task_type, tasks in worker.current_tasks_by_type.items():
                if task_type not in task_distribution:
                    task_distribution[task_type] = 0
                task_distribution[task_type] += len(tasks)
        
        return {
            'total_workers': len(self.workers),
            'active_workers': len(active_workers),
            'failed_workers': len([w for w in self.workers.values() if w.status in ['failed', 'permanently_failed']]),
            'total_active_tasks': total_active_tasks,
            'task_distribution': task_distribution,
            'workers': {
                worker_id: worker.get_status_dict()
                for worker_id, worker in self.workers.items()
            }
        }
    
    def cleanup_failed_worker_tasks(self) -> List[str]:
        """Get task IDs from failed workers that need cleanup"""
        tasks_to_cleanup = []
        
        for worker in self.workers.values():
            if worker.status in ['failed', 'unhealthy', 'permanently_failed'] and worker.assigned_task_ids:
                logger.info(f"Found {len(worker.assigned_task_ids)} tasks to cleanup from {worker.status} worker {worker.worker_id}")
                tasks_to_cleanup.extend(worker.assigned_task_ids)
                
                # Clear the worker's task tracking
                for task_id in list(worker.assigned_task_ids):
                    worker.remove_task(task_id)
        
        return tasks_to_cleanup