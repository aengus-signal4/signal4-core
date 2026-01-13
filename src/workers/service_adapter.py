"""
Service Management Adapter - Fixes the interface between service manager and worker manager
"""
import asyncio
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ServiceManagementAdapter:
    """Adapter to make service manager work with existing worker manager"""
    
    def __init__(self, worker_manager, service_manager):
        self.worker_manager = worker_manager
        self.service_manager = service_manager
        
        # Override service manager methods to use worker manager properly
        self._patch_service_manager()
    
    def _patch_service_manager(self):
        """Patch service manager to use correct worker manager interface"""
        # Replace the problematic methods
        self.service_manager._start_service_on_worker = self._start_service_on_worker
        self.service_manager._stop_service_on_worker = self._stop_service_on_worker
        self.service_manager._check_service_health = self._check_service_health
    
    async def _start_service_on_worker(self, worker_id: str, service_name: str, script_content: str) -> bool:
        """Start service on worker using worker manager"""
        try:
            # Get worker instance from worker manager
            if worker_id not in self.worker_manager.workers:
                logger.error(f"Worker {worker_id} not found in worker manager")
                return False
            
            worker = self.worker_manager.workers[worker_id]
            
            # For task processor, use the worker manager's built-in method
            if service_name == 'task_processor':
                return await self._start_task_processor_via_worker_manager(worker)
            
            # For model server, write and execute script
            script_path = f"/tmp/start_{service_name}_{worker_id}.sh"
            
            # Write startup script using worker manager
            success = await self.worker_manager._write_file(worker, script_path, script_content)
            if not success:
                logger.error(f"Failed to write startup script for {service_name} on {worker_id}")
                return False
            
            # Execute startup script
            screen_name = f"{service_name}_{worker_id}"
            cmd = f"screen -dmS {screen_name} bash {script_path}"
            
            result = await self.worker_manager._run_command(worker, cmd)
            if hasattr(result, 'returncode') and result.returncode != 0:
                logger.error(f"Failed to start {service_name} on {worker_id}: {result.stderr if hasattr(result, 'stderr') else result}")
                return False
            
            logger.info(f"Started {service_name} on worker {worker_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting service {service_name} on {worker_id}: {str(e)}")
            return False
    
    async def _start_task_processor_via_worker_manager(self, worker) -> bool:
        """Start task processor using worker manager's existing logic"""
        try:
            # Kill any existing processor
            kill_success = await self.worker_manager._kill_processor_unified(worker)
            if not kill_success:
                logger.warning(f"Failed to kill existing processor on worker {worker.worker_id}")
            
            # Wait a moment
            await asyncio.sleep(2)
            
            # Start processor
            start_success = await self.worker_manager._start_processor_unified(worker)
            if start_success:
                logger.info(f"Successfully started task processor on worker {worker.worker_id}")
                worker.status = 'running'
                return True
            else:
                logger.error(f"Failed to start task processor on worker {worker.worker_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error starting task processor via worker manager: {str(e)}")
            return False
    
    async def _stop_service_on_worker(self, worker_id: str, service_name: str) -> bool:
        """Stop service on worker"""
        try:
            if service_name == 'llm_server':
                logger.info(f"LLM server management is disabled - skipping stop for worker {worker_id}")
                return True  # Return success since it's managed independently
            
            if worker_id not in self.worker_manager.workers:
                logger.error(f"Worker {worker_id} not found")
                return False
            
            worker = self.worker_manager.workers[worker_id]
            
            # For task processor, use worker manager's kill method
            if service_name == 'task_processor':
                return await self.worker_manager._kill_processor_unified(worker)
            
            # For other services, kill by process name and screen
            script_name = service_name.replace('_', '')  # model_server -> modelserver
            kill_cmd = f"pkill -f '{script_name}.py' || true; screen -S {service_name}_{worker_id} -X quit || true"
            
            result = await self.worker_manager._run_command(worker, kill_cmd)
            logger.info(f"Stopped {service_name} on worker {worker_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping service {service_name} on {worker_id}: {str(e)}")
            return False
    
    async def _check_service_health(self, worker_id: str, service_name: str) -> bool:
        """Check if service is healthy using worker manager's health check"""
        try:
            # For task processor, use existing health check
            if service_name == 'task_processor':
                return await self.worker_manager.check_worker_health(worker_id)
            
            # For model server, check the specific port
            services = self.service_manager._get_worker_services(worker_id)
            if service_name not in services:
                return False
            
            service_info = services[service_name]
            config = self.service_manager.service_configs.get(service_name, {})
            
            # Get worker instance
            if worker_id not in self.worker_manager.workers:
                return False
            
            worker = self.worker_manager.workers[worker_id]
            
            # Determine IP
            worker_ip = getattr(worker, 'current_ip', 'localhost')
            if getattr(worker, 'is_head_worker', False):
                worker_ip = 'localhost'
            
            # Health check URL
            health_url = f"http://{worker_ip}:{service_info.port}{config.get('health_endpoint', '/health')}"
            
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(health_url, timeout=5) as response:
                    is_healthy = response.status == 200
                    if is_healthy:
                        logger.debug(f"Service {service_name} on {worker_id} is healthy")
                    else:
                        logger.warning(f"Service {service_name} on {worker_id} health check failed: {response.status}")
                    return is_healthy
                    
        except Exception as e:
            logger.debug(f"Health check failed for {service_name} on {worker_id}: {str(e)}")
            return False
    
    async def ensure_services_for_worker(self, worker_id: str, task_types: list) -> bool:
        """Ensure all required services are running for a worker"""
        try:
            logger.info(f"Ensuring services for worker {worker_id} with task types {task_types}")
            
            # Always ensure task processor is running
            processor_success = await self.service_manager.start_task_processor(worker_id)
            if not processor_success:
                logger.error(f"Failed to start task processor on {worker_id}")
                return False
            
            # Model servers are now handled by infrastructure services (ServiceStartupManager)
            # Workers only handle task processing, not model serving
            logger.info(f"Worker {worker_id} configured for task processing only (models served by infrastructure)")

            logger.info(f"All services started successfully for worker {worker_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error ensuring services for worker {worker_id}: {str(e)}")
            return False
    
    async def restart_worker_services(self, worker_id: str, task_types: list) -> bool:
        """Restart all services for a worker"""
        try:
            logger.info(f"Restarting services for worker {worker_id}")
            
            # Stop all services
            await self.service_manager.stop_task_processor(worker_id)
            await self.service_manager.stop_model_server(worker_id)
            
            # Wait for services to stop
            await asyncio.sleep(3)
            
            # Start services again
            return await self.ensure_services_for_worker(worker_id, task_types)
            
        except Exception as e:
            logger.error(f"Error restarting services for worker {worker_id}: {str(e)}")
            return False