"""
Service Manager - Manages services running on workers (model servers, processors)
"""
import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Optional, Set, Any
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum

from src.services.llm.model_config import (
    MODEL_REQUIREMENTS,
    get_required_models_for_worker,
    get_model_config,
    estimate_memory_requirements
)

logger = logging.getLogger(__name__)

class ServiceStatus(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    UNHEALTHY = "unhealthy"
    FAILED = "failed"

@dataclass
class ServiceInfo:
    """Information about a service running on a worker"""
    name: str
    port: int
    status: ServiceStatus
    start_time: Optional[datetime] = None
    last_health_check: Optional[datetime] = None
    health_check_url: Optional[str] = None
    models: List[str] = None
    error: Optional[str] = None

class ServiceManager:
    """Manages services running on each worker (model servers, processors)"""

    def __init__(self, worker_manager, config):
        self.worker_manager = worker_manager
        self.config = config
        self.services: Dict[str, Dict[str, ServiceInfo]] = {}  # worker_id -> {service_name -> ServiceInfo}
        self.logger = logger

        # LLM server management disabled - managed independently
        self.llm_server_management_enabled = False

        # Health check failure tracking - require consecutive failures before marking unhealthy
        self.health_check_failures: Dict[str, Dict[str, int]] = {}  # worker_id -> {service_name -> failure_count}
        self.health_check_failure_threshold = 3  # Consecutive failures before marking unhealthy
        self.health_check_timeout = 15  # Increased timeout for busy workers

        # Project directory - use config or derive from storage.local.base_path
        self.project_dir = config.get('storage', {}).get('local', {}).get(
            'base_path', '/Users/signal4/signal4/core'
        )

        # Service configurations
        self.service_configs = {
            # DEPRECATED: model_server is deprecated due to thread safety issues
            # Use llm_server for LLM requests, load other models directly in processes
            'model_server': {
                'port': 8002,
                'health_endpoint': '/health',
                'startup_timeout': 120,  # 2 minutes for model loading
                'script_path': 'src/services/llm/mlx_server.py',
                'deprecated': True,
                'deprecation_message': 'Use llm_server for LLM requests instead'
            },
            'llm_server': {
                'port': 8002,
                'health_endpoint': '/health',
                'startup_timeout': 60,  # 1 minute for LLM loading
                'script_path': 'src/services/llm/server.py'
            },
            'task_processor': {
                'port': 8000,
                'health_endpoint': '/tasks',
                'startup_timeout': 30,
                'script_path': 'src/workers/processor.py'
            }
        }
    
    def _get_worker_services(self, worker_id: str) -> Dict[str, ServiceInfo]:
        """Get services dict for worker, creating if needed"""
        if worker_id not in self.services:
            self.services[worker_id] = {}
        return self.services[worker_id]
    
    async def start_model_server(self, worker_id: str, task_types: List[str]) -> bool:
        """Start model server on worker with models needed for task types
        
        DEPRECATED: This method starts the deprecated model_server.py.
        Consider using start_llm_server() for LLM requests instead.
        """
        # Log deprecation warning
        self.logger.warning("=" * 60)
        self.logger.warning("DEPRECATION WARNING: model_server.py is deprecated!")
        self.logger.warning("Use llm_server.py for LLM requests instead.")
        self.logger.warning("For other models, load directly in each process.")
        self.logger.warning("=" * 60)
        
        try:
            # Determine required models
            required_models = get_required_models_for_worker(task_types)
            
            if not required_models:
                self.logger.info(f"No models required for worker {worker_id} task types {task_types}")
                return True
            
            # Check if model server is already running with correct models
            services = self._get_worker_services(worker_id)
            if 'model_server' in services:
                existing_service = services['model_server']
                if (existing_service.status == ServiceStatus.RUNNING and 
                    set(existing_service.models or []) == required_models):
                    self.logger.info(f"Model server already running on {worker_id} with correct models")
                    return True
                else:
                    # Stop existing service if models don't match
                    await self.stop_model_server(worker_id)
            
            # Create service info
            service_info = ServiceInfo(
                name='model_server',
                port=self.service_configs['model_server']['port'],
                status=ServiceStatus.STARTING,
                start_time=datetime.now(timezone.utc),
                models=list(required_models)
            )
            services['model_server'] = service_info
            
            # Estimate memory requirements
            memory_mb = estimate_memory_requirements(list(required_models))
            self.logger.info(f"Starting model server on {worker_id} with models {required_models} "
                           f"(estimated {memory_mb}MB memory)")
            
            # Create startup script
            models_str = ','.join(required_models)
            script_content = self._create_model_server_script(
                worker_id, 
                service_info.port, 
                models_str
            )
            
            # Write and execute startup script via worker manager
            success = await self._start_service_on_worker(
                worker_id, 
                'model_server', 
                script_content
            )
            
            if success:
                # Wait for service to be healthy
                healthy = await self._wait_for_service_health(
                    worker_id, 
                    'model_server',
                    timeout=self.service_configs['model_server']['startup_timeout']
                )
                
                if healthy:
                    service_info.status = ServiceStatus.RUNNING
                    self.logger.info(f"Model server started successfully on {worker_id}")
                    return True
                else:
                    service_info.status = ServiceStatus.FAILED
                    service_info.error = "Service failed health check"
                    self.logger.error(f"Model server failed to start on {worker_id}")
                    return False
            else:
                service_info.status = ServiceStatus.FAILED
                service_info.error = "Failed to start service"
                return False
                
        except Exception as e:
            self.logger.error(f"Error starting model server on {worker_id}: {str(e)}")
            services = self._get_worker_services(worker_id)
            if 'model_server' in services:
                services['model_server'].status = ServiceStatus.FAILED
                services['model_server'].error = str(e)
            return False
    
    async def start_llm_server(self, worker_id: str) -> bool:
        """Start LLM server on worker for thread-safe Ollama access
        
        This is the recommended replacement for model_server when you need LLM access.
        """
        if not self.llm_server_management_enabled:
            self.logger.info(f"LLM server management is disabled - skipping start for worker {worker_id}")
            return True  # Return success since it's managed independently
        
        try:
            # Check if LLM server is already running
            services = self._get_worker_services(worker_id)
            if 'llm_server' in services:
                existing_service = services['llm_server']
                if existing_service.status == ServiceStatus.RUNNING:
                    self.logger.info(f"LLM server already running on {worker_id}")
                    return True
                else:
                    # Stop existing service
                    await self.stop_service(worker_id, 'llm_server')
            
            # Create service info
            service_info = ServiceInfo(
                name='llm_server',
                port=self.service_configs['llm_server']['port'],
                status=ServiceStatus.STARTING,
                start_time=datetime.now(timezone.utc),
                models=['qwen3:4b-instruct']  # Default LLM model
            )
            services['llm_server'] = service_info
            
            self.logger.info(f"Starting LLM server on {worker_id} (thread-safe Ollama wrapper)")
            
            # Create startup script
            script_content = self._create_llm_server_script(worker_id, service_info.port)
            
            # Write and execute startup script via worker manager
            success = await self._start_service_on_worker(
                worker_id, 
                'llm_server', 
                script_content
            )
            
            if success:
                # Wait for service to be healthy
                healthy = await self._wait_for_service_health(
                    worker_id, 
                    'llm_server',
                    timeout=self.service_configs['llm_server']['startup_timeout']
                )
                
                if healthy:
                    service_info.status = ServiceStatus.RUNNING
                    self.logger.info(f"LLM server started successfully on {worker_id}")
                    return True
                else:
                    service_info.status = ServiceStatus.FAILED
                    service_info.error = "Service failed health check"
                    self.logger.error(f"LLM server failed to start on {worker_id}")
                    return False
            else:
                service_info.status = ServiceStatus.FAILED
                service_info.error = "Failed to start service"
                return False
                
        except Exception as e:
            self.logger.error(f"Error starting LLM server on {worker_id}: {str(e)}")
            services = self._get_worker_services(worker_id)
            if 'llm_server' in services:
                services['llm_server'].status = ServiceStatus.FAILED
                services['llm_server'].error = str(e)
            return False
    
    async def stop_service(self, worker_id: str, service_name: str) -> bool:
        """Generic method to stop any service on worker"""
        if service_name == 'llm_server' and not self.llm_server_management_enabled:
            self.logger.info(f"LLM server management is disabled - skipping stop for worker {worker_id}")
            return True  # Return success since it's managed independently
        
        try:
            services = self._get_worker_services(worker_id)
            if service_name not in services:
                self.logger.info(f"No {service_name} running on {worker_id}")
                return True
            
            self.logger.info(f"Stopping {service_name} on {worker_id}")
            
            # Kill service process
            success = await self._stop_service_on_worker(worker_id, service_name)
            
            if success:
                services[service_name].status = ServiceStatus.STOPPED
                self.logger.info(f"{service_name} stopped on {worker_id}")
            else:
                self.logger.error(f"Failed to stop {service_name} on {worker_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error stopping {service_name} on {worker_id}: {str(e)}")
            return False
    
    async def stop_model_server(self, worker_id: str) -> bool:
        """Stop model server on worker"""
        try:
            services = self._get_worker_services(worker_id)
            if 'model_server' not in services:
                self.logger.info(f"No model server running on {worker_id}")
                return True
            
            self.logger.info(f"Stopping model server on {worker_id}")
            
            # Kill model server process
            success = await self._stop_service_on_worker(worker_id, 'model_server')
            
            if success:
                services['model_server'].status = ServiceStatus.STOPPED
                self.logger.info(f"Model server stopped on {worker_id}")
            else:
                self.logger.error(f"Failed to stop model server on {worker_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error stopping model server on {worker_id}: {str(e)}")
            return False
    
    async def start_task_processor(self, worker_id: str) -> bool:
        """Start task processor on worker"""
        try:
            services = self._get_worker_services(worker_id)

            # First, check if processor is already running and healthy via HTTP
            # This handles the case where processor was started manually or by another process
            if await self._check_service_health(worker_id, 'task_processor'):
                self.logger.info(f"Task processor already running and healthy on {worker_id}")
                # Update internal state to reflect reality
                if 'task_processor' not in services:
                    services['task_processor'] = ServiceInfo(
                        name='task_processor',
                        port=self.service_configs['task_processor']['port'],
                        status=ServiceStatus.RUNNING,
                        start_time=datetime.now(timezone.utc)
                    )
                else:
                    services['task_processor'].status = ServiceStatus.RUNNING
                return True

            # Check internal state - if we think it's running but health check failed, stop it
            if 'task_processor' in services:
                existing_service = services['task_processor']
                if existing_service.status == ServiceStatus.RUNNING:
                    self.logger.info(f"Task processor marked as running but health check failed on {worker_id}")
                    await self.stop_task_processor(worker_id)
                elif existing_service.status != ServiceStatus.STOPPED:
                    # Stop existing service that's in a bad state
                    await self.stop_task_processor(worker_id)
            
            # Create service info
            service_info = ServiceInfo(
                name='task_processor',
                port=self.service_configs['task_processor']['port'],
                status=ServiceStatus.STARTING,
                start_time=datetime.now(timezone.utc)
            )
            services['task_processor'] = service_info
            
            self.logger.info(f"Starting task processor on {worker_id}")
            
            # Create startup script
            script_content = self._create_task_processor_script(worker_id, service_info.port)
            
            # Write and execute startup script
            success = await self._start_service_on_worker(
                worker_id, 
                'task_processor', 
                script_content
            )
            
            if success:
                # Wait for service to be healthy
                healthy = await self._wait_for_service_health(
                    worker_id, 
                    'task_processor',
                    timeout=self.service_configs['task_processor']['startup_timeout']
                )
                
                if healthy:
                    service_info.status = ServiceStatus.RUNNING
                    self.logger.info(f"Task processor started successfully on {worker_id}")
                    return True
                else:
                    service_info.status = ServiceStatus.FAILED
                    service_info.error = "Service failed health check"
                    self.logger.error(f"Task processor failed to start on {worker_id}")
                    return False
            else:
                service_info.status = ServiceStatus.FAILED
                service_info.error = "Failed to start service"
                return False
                
        except Exception as e:
            self.logger.error(f"Error starting task processor on {worker_id}: {str(e)}")
            services = self._get_worker_services(worker_id)
            if 'task_processor' in services:
                services['task_processor'].status = ServiceStatus.FAILED
                services['task_processor'].error = str(e)
            return False
    
    async def stop_task_processor(self, worker_id: str) -> bool:
        """Stop task processor on worker"""
        try:
            services = self._get_worker_services(worker_id)
            if 'task_processor' not in services:
                self.logger.info(f"No task processor running on {worker_id}")
                return True
            
            self.logger.info(f"Stopping task processor on {worker_id}")
            
            # Kill task processor
            success = await self._stop_service_on_worker(worker_id, 'task_processor')
            
            if success:
                services['task_processor'].status = ServiceStatus.STOPPED
                self.logger.info(f"Task processor stopped on {worker_id}")
            else:
                self.logger.error(f"Failed to stop task processor on {worker_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error stopping task processor on {worker_id}: {str(e)}")
            return False
    
    async def ensure_required_services(self, worker_id: str, task_types: List[str]) -> bool:
        """Ensure worker has necessary services for its task types"""
        try:
            self.logger.info(f"Ensuring required services for {worker_id} with task types {task_types}")
            
            # Always start task processor
            processor_success = await self.start_task_processor(worker_id)
            if not processor_success:
                self.logger.error(f"Failed to start task processor on {worker_id}")
                return False
            
            # Start model server if needed
            required_models = get_required_models_for_worker(task_types)
            if required_models:
                model_server_success = await self.start_model_server(worker_id, task_types)
                if not model_server_success:
                    self.logger.error(f"Failed to start model server on {worker_id}")
                    return False
            else:
                self.logger.info(f"No model server needed for {worker_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error ensuring services for {worker_id}: {str(e)}")
            return False
    
    async def get_service_status(self, worker_id: str, service_name: str) -> Optional[ServiceInfo]:
        """Get status of a specific service"""
        services = self._get_worker_services(worker_id)
        return services.get(service_name)
    
    async def get_all_services_status(self, worker_id: str) -> Dict[str, ServiceInfo]:
        """Get status of all services on a worker"""
        return self._get_worker_services(worker_id).copy()
    
    async def health_check_services(self, worker_id: str) -> Dict[str, bool]:
        """Perform health checks on all services for a worker.

        Uses consecutive failure tracking to avoid false positives when workers
        are busy processing tasks. A service is only marked unhealthy after
        `health_check_failure_threshold` consecutive failures.
        """
        services = self._get_worker_services(worker_id)
        results = {}

        for service_name, service_info in services.items():
            if service_info.status == ServiceStatus.RUNNING:
                healthy = await self._check_service_health(worker_id, service_name)

                if healthy:
                    results[service_name] = True
                    service_info.last_health_check = datetime.now(timezone.utc)
                    # Status stays RUNNING, failures already reset in _check_service_health
                else:
                    # Increment failure count and check threshold
                    failure_count = self._increment_health_check_failures(worker_id, service_name)

                    if failure_count >= self.health_check_failure_threshold:
                        # Only mark unhealthy after consecutive failures
                        results[service_name] = False
                        service_info.status = ServiceStatus.UNHEALTHY
                        service_info.error = f"Health check failed {failure_count} consecutive times"
                        self.logger.warning(
                            f"Service {service_name} on {worker_id} marked unhealthy after "
                            f"{failure_count} consecutive failures"
                        )
                    else:
                        # Still within tolerance, report as healthy for now
                        results[service_name] = True
                        self.logger.debug(
                            f"Health check failed for {service_name} on {worker_id} "
                            f"({failure_count}/{self.health_check_failure_threshold})"
                        )
            else:
                results[service_name] = False

        return results
    
    def _create_model_server_script(self, worker_id: str, port: int, models: str) -> str:
        """Create startup script for model server"""
        project_dir = self.project_dir

        return f"""#!/bin/bash
# Model server startup script for {worker_id}

set -e
cd {project_dir}

# Set environment variables
export MallocStackLogging=0
export MallocStackLoggingNoCompact=0

# Create log directory
mkdir -p logs/content_processing

# Kill any existing model server
pkill -f "mlx_server.py" || true
sleep 2

# Start model server
echo "[$(date)] Starting model server with models: {models}"
uv run python src/services/llm/mlx_server.py \\
    --port {port} \\
    --models {models} \\
    2>&1 | tee -a logs/content_processing/model_server_{worker_id}.log &

echo $! > /tmp/model_server_{worker_id}.pid
echo "[$(date)] Model server started with PID $!"
"""
    
    def _create_llm_server_script(self, worker_id: str, port: int) -> str:
        """Create startup script for LLM server (thread-safe Ollama wrapper)"""
        project_dir = self.project_dir

        return f"""#!/bin/bash
# LLM server startup script for {worker_id}

set -e
cd {project_dir}

# Set environment variables
export MallocStackLogging=0
export MallocStackLoggingNoCompact=0
export LLM_MAX_CONCURRENT=5  # Ollama supports concurrent requests

# Create log directory
mkdir -p logs/content_processing

# Kill any existing LLM server
pkill -f "llm/server.py" || true
sleep 2

# Start LLM server
echo "[$(date)] Starting LLM server (thread-safe Ollama wrapper)"
uv run python src/services/llm/server.py \\
    2>&1 | tee -a logs/content_processing/llm_server_{worker_id}.log &

echo $! > /tmp/llm_server_{worker_id}.pid
echo "[$(date)] LLM server started with PID $!"
"""
    
    def _create_task_processor_script(self, worker_id: str, port: int) -> str:
        """Create startup script for task processor"""
        project_dir = self.project_dir

        return f"""#!/bin/bash
# Task processor startup script for {worker_id}

set -e
cd {project_dir}

# Set environment variables
export MallocStackLogging=0
export MallocStackLoggingNoCompact=0

# Create log directory
mkdir -p logs/content_processing

# Kill any process holding port {port} (catches stale processors regardless of how they were started)
PORT_PID=$(lsof -ti :{port} 2>/dev/null || true)
if [ -n "$PORT_PID" ]; then
    echo "[$(date)] Killing existing process(es) on port {port}: $PORT_PID"
    echo "$PORT_PID" | xargs kill -9 2>/dev/null || true
    sleep 2
fi

# Also kill by process name patterns (belt and suspenders)
pkill -9 -f "src.workers.processor" 2>/dev/null || true
pkill -9 -f "src/workers/processor" 2>/dev/null || true
pkill -9 -f "uvicorn.*processor" 2>/dev/null || true
sleep 1

# Start task processor with uvicorn
echo "[$(date)] Starting task processor on port {port}"
uv run python -m uvicorn src.workers.processor:app --host 0.0.0.0 --port {port} \\
    2>&1 | tee -a logs/content_processing/task_processor_{worker_id}.log &

echo $! > /tmp/task_processor_{worker_id}.pid
echo "[$(date)] Task processor started with PID $!"
"""
    
    async def _start_service_on_worker(self, worker_id: str, service_name: str, script_content: str) -> bool:
        """Start service on worker using worker manager"""
        try:
            # Get worker instance
            if worker_id not in self.worker_manager.workers:
                self.logger.error(f"Worker {worker_id} not found in worker manager")
                return False
            
            worker = self.worker_manager.workers[worker_id]
            
            # Write startup script
            script_path = f"/tmp/start_{service_name}_{worker_id}.sh"
            success = await self.worker_manager._write_file(worker, script_path, script_content)
            if not success:
                self.logger.error(f"Failed to write startup script for {service_name} on {worker_id}")
                return False
            
            # Execute startup script in screen
            screen_name = f"{service_name}_{worker_id}"
            cmd = f"screen -dmS {screen_name} bash {script_path}"
            
            result = await self.worker_manager._run_command(worker, cmd)
            if hasattr(result, 'returncode') and result.returncode != 0:
                self.logger.error(f"Failed to start {service_name} on {worker_id}: {result.stderr}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting service {service_name} on {worker_id}: {str(e)}")
            return False
    
    async def _stop_service_on_worker(self, worker_id: str, service_name: str) -> bool:
        """Stop service on worker"""
        try:
            # Get worker instance
            if worker_id not in self.worker_manager.workers:
                self.logger.error(f"Worker {worker_id} not found")
                return False
            
            worker = self.worker_manager.workers[worker_id]
            
            # Kill service process
            script_name = service_name.replace('_', '')  # model_server -> modelserver
            kill_cmd = f"pkill -f '{script_name}.py' || true; screen -S {service_name}_{worker_id} -X quit || true"
            
            result = await self.worker_manager._run_command(worker, kill_cmd)
            # pkill and screen commands can return non-zero if process not found, that's ok
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping service {service_name} on {worker_id}: {str(e)}")
            return False
    
    async def _wait_for_service_health(self, worker_id: str, service_name: str, timeout: int = 60) -> bool:
        """Wait for service to become healthy"""
        start_time = datetime.now()
        
        while (datetime.now() - start_time).seconds < timeout:
            healthy = await self._check_service_health(worker_id, service_name)
            if healthy:
                return True
            
            await asyncio.sleep(5)  # Check every 5 seconds
        
        self.logger.error(f"Service {service_name} on {worker_id} failed to become healthy within {timeout}s")
        return False
    
    async def _check_service_health(self, worker_id: str, service_name: str) -> bool:
        """Check if service is healthy via HTTP, regardless of internal state"""
        try:
            if service_name not in self.service_configs:
                return False

            config = self.service_configs[service_name]
            port = config['port']
            health_endpoint = config['health_endpoint']

            # Get worker IP
            if worker_id not in self.worker_manager.workers:
                return False

            worker = self.worker_manager.workers[worker_id]
            worker_ip = getattr(worker, 'current_ip', 'localhost')

            # For local worker, use localhost
            if getattr(worker, 'is_head_worker', False):
                worker_ip = 'localhost'

            health_url = f"http://{worker_ip}:{port}{health_endpoint}"

            async with aiohttp.ClientSession() as session:
                async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=self.health_check_timeout)) as response:
                    if response.status == 200:
                        # Reset failure count on success
                        self._reset_health_check_failures(worker_id, service_name)
                        return True
                    return False

        except Exception as e:
            self.logger.debug(f"Health check failed for {service_name} on {worker_id}: {str(e)}")
            return False

    def _get_health_check_failures(self, worker_id: str, service_name: str) -> int:
        """Get current failure count for a service"""
        if worker_id not in self.health_check_failures:
            return 0
        return self.health_check_failures[worker_id].get(service_name, 0)

    def _increment_health_check_failures(self, worker_id: str, service_name: str) -> int:
        """Increment failure count and return new count"""
        if worker_id not in self.health_check_failures:
            self.health_check_failures[worker_id] = {}
        current = self.health_check_failures[worker_id].get(service_name, 0)
        self.health_check_failures[worker_id][service_name] = current + 1
        return current + 1

    def _reset_health_check_failures(self, worker_id: str, service_name: str) -> None:
        """Reset failure count for a service"""
        if worker_id in self.health_check_failures:
            self.health_check_failures[worker_id][service_name] = 0