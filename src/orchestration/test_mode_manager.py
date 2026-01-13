"""
Test Mode Manager - Handles test mode operations for the orchestrator
"""
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class TestModeManager:
    """Manages test mode operations"""
    
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.test_config: Optional[Dict[str, Any]] = None
    
    def configure(self, worker_id: str, task_type: str, iterations: int):
        """Configure test mode parameters"""
        self.test_config = {
            'worker_id': worker_id,
            'task_type': task_type,
            'iterations': iterations,
            'completed': 0
        }
        logger.info(f"Configured test mode: worker={worker_id}, task={task_type}, iterations={iterations}")
    
    def is_configured(self) -> bool:
        """Check if test mode is configured"""
        return self.test_config is not None
    
    async def initialize_test_worker(self) -> bool:
        """Initialize single worker for test mode"""
        if not self.test_config:
            raise ValueError("Test mode not configured")
        
        worker_configs = self.orchestrator.config_manager.get_worker_configs()
        worker_id = self.test_config['worker_id']
        
        if worker_id not in worker_configs:
            raise ValueError(f"Worker {worker_id} not found in configuration")
        
        config = worker_configs[worker_id]
        if not config.get('enabled', False):
            raise ValueError(f"Worker {worker_id} is not enabled")
        
        # Verify worker can handle the test task type
        test_task_type = self.test_config['task_type']
        worker_can_handle = False
        
        for task_entry in config.get('task_types', []):
            parts = task_entry.split(':', 1)
            task_type = parts[0].strip()
            if task_type == test_task_type:
                worker_can_handle = True
                break
        
        if not worker_can_handle:
            raise ValueError(f"Worker {worker_id} cannot handle task type {test_task_type}")
        
        # Use worker pool manager to add worker
        success = await self.orchestrator.worker_pool.add_worker(worker_id, config)
        if not success:
            raise RuntimeError(f"Failed to add test worker {worker_id}")
        
        # Start worker services
        worker = self.orchestrator.worker_pool.get_worker(worker_id)
        success = await self.orchestrator._start_worker_services(worker_id)
        
        if success:
            worker.status = 'active'
            worker.started_at = datetime.now(timezone.utc)
            logger.info(f"Test worker {worker_id} initialized successfully")
        else:
            raise RuntimeError(f"Failed to start services for test worker {worker_id}")
        
        # Register with behavior manager
        self.orchestrator.behavior_manager.register_worker(worker_id, config)
        
        return True
    
    async def process_test_tasks(self):
        """Process tasks in test mode"""
        if not self.test_config:
            raise ValueError("Test mode not configured")
        
        worker_id = self.test_config['worker_id']
        worker = self.orchestrator.worker_pool.get_worker(worker_id)
        
        if not worker:
            raise ValueError(f"Test worker {worker_id} not found")
        
        while not self.orchestrator.should_stop and self.test_config['completed'] < self.test_config['iterations']:
            try:
                # Check global pause - test mode should respect this too
                if self.orchestrator.global_pause_until and datetime.now(timezone.utc) < self.orchestrator.global_pause_until:
                    await asyncio.sleep(self.orchestrator.config_manager.idle_sleep_interval)
                    continue
                elif self.orchestrator.global_pause_until:
                    logger.info("Test mode: Global pause ended, resuming task assignment")
                    self.orchestrator.global_pause_until = None
                
                # Get next test task
                tasks = await self.orchestrator.task_manager.get_next_tasks(
                    limit=1,
                    task_types=[self.test_config['task_type']],
                    exclude_task_ids=self.orchestrator.task_manager.assigned_tasks
                )
                
                if not tasks:
                    logger.info(f"Test mode: No {self.test_config['task_type']} tasks available, waiting...")
                    await asyncio.sleep(self.orchestrator.config_manager.idle_sleep_interval)
                    continue
                
                task = tasks[0]
                success = await self.orchestrator._assign_task_to_worker(task, worker)
                
                if success:
                    self.test_config['completed'] += 1
                    logger.info(f"Test mode: Assigned task {self.test_config['completed']}/{self.test_config['iterations']}")
                else:
                    logger.warning(f"Test mode: Failed to assign task")
                
                # Small delay between assignments
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error in test mode task processing: {str(e)}")
                await asyncio.sleep(self.orchestrator.config_manager.idle_sleep_interval)
        
        logger.info(f"Test mode completed: {self.test_config['completed']} tasks assigned")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get test mode statistics"""
        if not self.test_config:
            return {'configured': False}
        
        return {
            'configured': True,
            'worker_id': self.test_config['worker_id'],
            'task_type': self.test_config['task_type'],
            'iterations': self.test_config['iterations'],
            'completed': self.test_config['completed'],
            'remaining': self.test_config['iterations'] - self.test_config['completed']
        }