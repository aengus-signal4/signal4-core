#!/usr/bin/env python3
"""
Task Orchestrator V2 - Modular, Service-Aware Content Processing Orchestrator

This is a complete rewrite of the task orchestrator with:
- Modular architecture for maintainability
- Service management (model servers, task processors)
- Enhanced worker management with network failover
- Better task assignment and tracking
- Comprehensive health monitoring
"""

import asyncio
import logging
import sys
import signal
import subprocess
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Set
from pathlib import Path
from src.utils.paths import get_project_root, get_config_path
from collections import defaultdict
import uvicorn

# Add project root to path
sys.path.append(str(get_project_root()))

# Import modular components
from src.orchestration.task_manager import TaskManager
from src.monitoring.network import NetworkManager
from src.orchestration.api_endpoints import create_api_endpoints
from src.orchestration.timeout_manager import TaskTimeoutManager
from src.monitoring.network_monitor import NetworkMonitor
from src.orchestration.reactive_assignment import ReactiveAssignmentManager
from src.orchestration.failure_tracker import FailureTracker
from src.monitoring.s3 import S3Monitor
from src.workers.service_adapter import ServiceManagementAdapter
# Unified scheduled task manager (replaces legacy managers)
from src.automation.scheduled_task_manager import ScheduledTaskManager
# Code deployment manager
from src.workers.code_deployment import CodeDeploymentManager

# OllamaMonitor removed - we use MLX only now
from src.workers.service_startup import ServiceStartupManager
from src.orchestration.logging_setup import setup_orchestrator_logging, TaskLogger, configure_noise_suppression
from src.workers.info import WorkerInfo
from src.orchestration.config_manager import ConfigManager
from src.orchestration.test_mode_manager import TestModeManager
from src.workers.pool import WorkerPool
from src.automation.background_loops import (
    StatsCollector, ConfigReloader, TaskCleaner, HealthChecker, TaskAssigner
)
from src.workers.service_manager import ServiceManager
from src.workers.management import WorkerManager
from src.processing.pipeline_manager import PipelineManager
from src.utils.human_behavior import HumanBehaviorManager
from src.utils.logger import setup_worker_logger
from src.database.session import get_session
from src.database.models import TaskQueue, Content

# Set up logging
logger = setup_worker_logger('orchestrator_v2')

# Set logging level to INFO to ensure component logs are visible
logging.getLogger().setLevel(logging.INFO)

# Add console handler for all component logs
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s [%(name)s] [%(levelname)s] %(message)s')
console_handler.setFormatter(console_formatter)

# Add console handler to root logger to catch all component logs
root_logger = logging.getLogger()
root_logger.addHandler(console_handler)

# Set up dedicated loggers
completion_logger, error_logger, run_logger = setup_orchestrator_logging()

# Suppress noisy loggers
configure_noise_suppression()

# Create task logger instance
task_logger = TaskLogger(completion_logger, error_logger, logger)

class TaskOrchestratorV2:
    """Modern, modular task orchestrator with service management"""
    
    VERSION = "2.0.0"
    
    def __init__(self, config_path: str = "config/config.yaml"):
        # Initialize config manager
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        
        # Core state
        self.should_stop = False
        self.global_pause_until: Optional[datetime] = None
        
        # Initialize worker pool
        self.worker_pool = WorkerPool()
        
        # Initialize test mode manager
        self.test_mode = TestModeManager(self)
        
        # Initialize core managers
        self.task_manager = TaskManager(self.config)
        self.network_manager = NetworkManager(self.config)
        self.worker_manager = WorkerManager(self.config, {})  # Will update workers dict
        self.service_manager = ServiceManager(self.worker_manager, self.config)
        self.behavior_manager = HumanBehaviorManager(self.config_manager.config_path)
        
        # Initialize new components
        self.timeout_manager = TaskTimeoutManager(self.worker_manager)
        self.network_monitor = NetworkMonitor(self.network_manager, self.service_manager)
        # Pass worker_manager to network_monitor for enhanced restart functionality
        self.network_monitor.worker_manager = self.worker_manager
        self.failure_tracker = FailureTracker()
        self.s3_monitor = S3Monitor(self.config)
        self.reactive_assignment = ReactiveAssignmentManager(self.task_manager, self)

        # Initialize unified scheduled task manager
        self.scheduled_task_manager = ScheduledTaskManager(self.config, orchestrator=self)
        logger.info(f"Scheduled task manager initialized with {len(self.scheduled_task_manager.tasks)} tasks")

        # Initialize service startup manager (LLM server + dashboards)
        self.service_startup = ServiceStartupManager(self.config)
        logger.info("Service startup manager initialized")

        # Initialize code deployment manager
        self.code_deployment = CodeDeploymentManager(self.config)
        logger.info("Code deployment manager initialized")

        # Fix service manager integration
        self.service_adapter = ServiceManagementAdapter(self.worker_manager, self.service_manager)
        
        # Initialize pipeline manager with failure tracker
        self.pipeline_manager = PipelineManager(
            self.behavior_manager, 
            self.config, 
            self.failure_tracker.worker_task_failures
        )
        
        # Use config manager for all configuration values
        self.api_host = self.config_manager.api_host
        self.api_port = self.config_manager.api_port
        self.callback_url = self.config_manager.callback_url
        
        # Initialize background loop components
        self.stats_collector = StatsCollector(self, run_logger)
        self.config_reloader = ConfigReloader(self)
        self.task_cleaner = TaskCleaner(self)
        self.health_checker = HealthChecker(self)
        self.task_assigner = TaskAssigner(self)
        
        # Create FastAPI app
        self.app = create_api_endpoints(self)
        
        logger.info(f"Task Orchestrator V{self.VERSION} initialized successfully")
        logger.info(f"Configuration loaded from: {self.config_manager.config_path}")
        logger.info(f"Global pause enabled: {self.global_pause_until is not None}")
    
    async def _reload_config(self) -> bool:
        """Reload configuration and update components"""
        try:
            # Use config manager for reloading
            reloaded = self.config_manager.reload_config()
            if not reloaded:
                return False
            
            # Update local reference
            self.config = self.config_manager.config
            
            # Update component configurations
            if hasattr(self.task_manager, 'update_config'):
                self.task_manager.update_config(new_config)
            
            # Update worker configurations using worker pool
            worker_configs = self.config_manager.get_worker_configs()
            self.worker_pool.update_all_from_config(worker_configs)
            
            # Update behavior manager
            self.behavior_manager = HumanBehaviorManager(self.config_manager.config_path)
            
            # Update pipeline manager
            self.pipeline_manager = PipelineManager(
                self.behavior_manager,
                self.config,
                self.failure_tracker.worker_task_failures
            )
            
            # Update API configuration from config manager
            self.api_host = self.config_manager.api_host
            self.api_port = self.config_manager.api_port
            self.callback_url = self.config_manager.callback_url
            
            logger.info("Configuration reloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error reloading configuration: {str(e)}")
            return False
    
    async def initialize(self) -> bool:
        """Initialize the orchestrator and all components"""
        try:
            logger.info("Starting Task Orchestrator V2 initialization")

            # Start LLM server and dashboards first
            logger.info("Starting LLM server and dashboards")
            await self.service_startup.start_services()

            # Initialize task queue cache
            logger.info("Initializing task queue cache")
            await self.task_manager.initialize_cache()

            # Initialize network manager
            logger.info("Initializing network manager")
            await self.network_manager.initialize()
            logger.info("Network manager initialized successfully")
            
            # Initialize workers from config
            logger.info("Loading worker configurations")
            workers_config = self.config.get('processing', {}).get('workers', {})
            logger.info(f"Found {len(workers_config)} worker configurations in config")
            
            for worker_id, worker_config in workers_config.items():
                if not worker_config.get('enabled', False):
                    logger.info(f"Skipping disabled worker: {worker_id}")
                    continue
                
                # Add worker to pool
                success = await self.worker_pool.add_worker(worker_id, worker_config)
                if not success:
                    logger.warning(f"Failed to add worker {worker_id} to pool")
                    continue
                
                # Register with network manager
                self.network_manager.register_worker(worker_id, worker_config)
                
                logger.info(f"Registered worker {worker_id}")
                logger.info(f"  Type: {worker_config.get('type', 'worker')}")
                logger.info(f"  IPs: eth={worker_config.get('eth')}, wifi={worker_config.get('wifi')}")
                logger.info(f"  Enabled tasks: {worker_config.get('enabled_tasks', [])}")
                worker_info = self.worker_pool.get_worker(worker_id)
                if worker_info:
                    logger.info(f"  Parsed task types: {worker_info.task_types}")
                    logger.info(f"  Task limits: {worker_info.task_type_limits}")
                    logger.info(f"  Total capacity: {worker_info.max_concurrent_tasks}")
            
            logger.info("Setting up worker manager integration")
            # Update worker manager with our workers dict - create compatible worker objects
            self.worker_manager.workers = {}
            for wid, info in self.worker_pool.workers.items():
                # Create a worker object compatible with WorkerManager expectations
                worker_obj = type('Worker', (), {
                    'worker_id': wid,
                    'current_ip': info.eth_ip or info.wifi_ip or 'localhost',
                    'is_head_worker': info.is_head_worker,
                    'type': 'head' if info.is_head_worker else 'worker',
                    'api_url': f"http://{info.eth_ip or info.wifi_ip or 'localhost'}:8000",
                    'status': info.status
                })()
                self.worker_manager.workers[wid] = worker_obj
                logger.info(f"Created worker manager entry for {wid}: {worker_obj.api_url}")
            logger.info(f"Updated WorkerManager with {len(self.worker_manager.workers)} workers")

            # Sync Swift binaries to workers before starting services
            logger.info("Syncing Swift binaries to workers")
            await self._sync_swift_binaries_to_workers()

            # Start worker services
            logger.info("Starting worker services")
            await self._start_all_worker_services()
            
            # Summary
            logger.info("Initialization summary")
            active_workers = [w for w in self.worker_pool.workers.values() if w.status == 'active']
            failed_workers = [w for w in self.worker_pool.workers.values() if w.status == 'failed']

            logger.info(f"Successfully initialized {len(active_workers)}/{len(self.worker_pool.workers)} workers")
            if failed_workers:
                logger.warning(f"{len(failed_workers)} workers failed to start: {[w.worker_id for w in failed_workers]}")
            
            logger.info("Task Orchestrator V2 initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {str(e)}")
            return False
    
    async def _start_all_worker_services(self) -> None:
        """Start required services on all workers"""
        all_workers = self.worker_pool.get_all_workers()
        logger.info(f"Starting services on {len(all_workers)} workers...")
        
        startup_tasks = []
        for worker_id, worker_info in all_workers.items():
            logger.info(f"Queuing service startup for worker {worker_id}")
            task = asyncio.create_task(self._start_worker_services(worker_id))
            startup_tasks.append((worker_id, task))
        
        logger.info(f"Waiting for {len(startup_tasks)} workers to start their services")
        
        # Wait for all services to start
        for worker_id, task in startup_tasks:
            try:
                logger.info(f"Waiting for worker {worker_id} services")
                success = await task
                if success:
                    worker = self.worker_pool.get_worker(worker_id)
                    if worker:
                        worker.status = 'active'
                        worker.started_at = datetime.now(timezone.utc)
                    logger.info(f"Worker {worker_id} services started successfully")
                else:
                    worker = self.worker_pool.get_worker(worker_id)
                    if worker:
                        worker.status = 'failed'
                    logger.error(f"Failed to start services for worker {worker_id}")
            except Exception as e:
                worker = self.worker_pool.get_worker(worker_id)
                if worker:
                    worker.status = 'failed'
                logger.error(f"Error starting services for worker {worker_id}: {str(e)}")

    async def _sync_swift_binaries_to_workers(self) -> None:
        """Sync compiled Swift binaries (FluidDiarization) to all remote workers.

        Workers may not have Xcode CLI tools installed, so we sync the pre-built
        binary from the head node rather than requiring them to compile.
        """
        swift_build_dir = get_project_root() / "src" / "processing_steps" / "swift" / ".build"

        if not swift_build_dir.exists():
            logger.warning(f"Swift build directory not found at {swift_build_dir}, skipping sync")
            return

        # Get remote workers (not head node)
        remote_workers = [
            (wid, w) for wid, w in self.worker_pool.workers.items()
            if not w.is_head_worker and w.eth_ip != '10.0.0.4'
        ]

        if not remote_workers:
            logger.info("No remote workers to sync Swift binaries to")
            return

        logger.info(f"Syncing Swift binaries to {len(remote_workers)} workers")

        # Sync to each worker concurrently
        sync_tasks = []
        for worker_id, worker_info in remote_workers:
            task = asyncio.create_task(self._sync_swift_binary_to_worker(worker_id, worker_info, swift_build_dir))
            sync_tasks.append((worker_id, task))

        for worker_id, task in sync_tasks:
            try:
                success = await task
                if success:
                    logger.info(f"Swift binary synced to {worker_id}")
                else:
                    logger.warning(f"Failed to sync Swift binary to {worker_id}")
            except Exception as e:
                logger.error(f"Error syncing Swift binary to {worker_id}: {e}")

    async def _sync_swift_binary_to_worker(self, worker_id: str, worker_info: WorkerInfo, swift_build_dir: Path) -> bool:
        """Sync Swift binary to a single worker using rsync"""
        try:
            target_ip = worker_info.eth_ip or worker_info.wifi_ip
            if not target_ip:
                logger.warning(f"No IP for worker {worker_id}, skipping Swift sync")
                return False

            ssh_key = self.config.get('processing', {}).get('ssh_key_path', '/Users/signal4/.ssh/id_ed25519')
            ssh_user = self.config.get('processing', {}).get('ssh_username', 'signal4')

            # Create target directory first
            mkdir_cmd = f"ssh -o StrictHostKeyChecking=no -i {ssh_key} {ssh_user}@{target_ip} 'mkdir -p {swift_build_dir}'"
            mkdir_proc = await asyncio.create_subprocess_shell(
                mkdir_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await mkdir_proc.communicate()

            # Rsync the .build directory
            rsync_cmd = [
                'rsync', '-avz',
                '-e', f'ssh -o StrictHostKeyChecking=no -i {ssh_key}',
                f'{swift_build_dir}/',
                f'{ssh_user}@{target_ip}:{swift_build_dir}/'
            ]

            proc = await asyncio.create_subprocess_exec(
                *rsync_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                logger.error(f"Rsync to {worker_id} failed: {stderr.decode()}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error syncing Swift binary to {worker_id}: {e}")
            return False

    async def _start_worker_services(self, worker_id: str) -> bool:
        """Start required services for a specific worker"""
        try:
            worker_info = self.worker_pool.get_worker(worker_id)
            if not worker_info:
                logger.error(f"Worker {worker_id} not found in pool")
                return False
            worker_info.status = 'starting'
            
            logger.info(f"Starting services for worker {worker_id}")
            logger.info(f"  Task types: {worker_info.task_types}")
            logger.info(f"  Worker type: {'head' if worker_info.is_head_worker else 'remote'}")
            logger.info(f"  Target IPs: eth={worker_info.eth_ip}, wifi={worker_info.wifi_ip}")
            
            # Start required services using service adapter
            logger.info(f"Calling service adapter to start services")
            success = await self.service_adapter.ensure_services_for_worker(
                worker_id, 
                worker_info.task_types
            )
            
            if success:
                logger.info(f"All services started successfully for worker {worker_id}")
                return True
            else:
                logger.error(f"Failed to start some services for worker {worker_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error starting services for worker {worker_id}: {str(e)}")
            return False
    
    async def run(self) -> None:
        """Main orchestrator loop"""
        logger.info("Starting Task Orchestrator V2 main operations")

        # Deploy code to all enabled workers on startup
        # Pause network monitor restarts during deployment
        self.network_monitor.restart_paused = True
        logger.info("Deploying code to enabled workers...")
        try:
            deployment_status = await self.code_deployment.deploy_to_all_workers(self.worker_pool)
            if deployment_status.failed_workers:
                logger.warning(f"Code deployment failed for workers: {deployment_status.failed_workers}")
            else:
                logger.info(f"Code deployed successfully to {len(deployment_status.successful_workers)} workers")
        except Exception as e:
            logger.error(f"Code deployment failed: {e}")
        finally:
            # Resume network monitor restarts after deployment
            self.network_monitor.restart_paused = False

        # Start monitoring components
        logger.info("Starting monitoring components")
        logger.info("Starting timeout manager")
        await self.timeout_manager.start_monitoring(get_session)
        logger.info("Starting network monitor")
        await self.network_monitor.start_monitoring(self.worker_pool.get_all_workers(), get_session)

        # Start unified scheduled task manager for background services
        # (indexing, downloading, speaker ID, hydration, podcast collection)
        logger.info("Starting scheduled task manager")
        await self.scheduled_task_manager.start()

        logger.info("All monitoring components started")
        
        # Start background tasks
        logger.info("Starting background task loops")
        background_tasks = [
            asyncio.create_task(self.task_assigner.run()),
            asyncio.create_task(self.health_checker.run()),
            asyncio.create_task(self.task_cleaner.run()),
            asyncio.create_task(self.config_reloader.run()),
            asyncio.create_task(self.stats_collector.run())
        ]
        logger.info(f"Started {len(background_tasks)} background tasks")
        logger.info("Orchestrator is now running and ready for tasks")
        
        try:
            # Wait for all background tasks
            await asyncio.gather(*background_tasks)
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
        finally:
            logger.info("Orchestrator main loop stopped")
    
    
    async def _assign_task_to_worker(self, task: Dict[str, Any], worker: WorkerInfo) -> bool:
        """Assign a specific task to a worker with health check and retry logic"""
        try:
            task_id = str(task['id'])
            task_type = task['task_type']
            
            # First verify worker is actually healthy before attempting assignment
            try:
                if not await self.network_manager.check_worker_health(worker.worker_id):
                    logger.warning(f"Worker {worker.worker_id} failed health check before task assignment. Triggering restart.")
                    worker.status = 'unhealthy'
                    
                    # Use the locked restart method
                    restart_success, attempted = await worker.attempt_restart_with_lock(
                        self.worker_manager.restart_unhealthy_worker
                    )
                    
                    if not attempted:
                        backoff_remaining = worker.get_restart_backoff_remaining()
                        logger.debug(f"Worker {worker.worker_id} unhealthy but cannot restart yet "
                                   f"(backoff: {backoff_remaining:.1f}s, attempts: {worker.restart_attempts}/{worker.max_restart_attempts})")
                        return False
                    
                    if not restart_success:
                        logger.error(f"Failed to restart unhealthy worker {worker.worker_id}. Task will be retried later.")
                        return False
                    
                    # Even if restart succeeded, wait a moment for services to fully initialize
                    await asyncio.sleep(2)
                    
                    # Verify health again after restart
                    if not await self.network_manager.check_worker_health(worker.worker_id):
                        logger.error(f"Worker {worker.worker_id} still unhealthy after restart. Task will be retried later.")
                        return False
                    else:
                        # Health check passed after restart
                        logger.info(f"Worker {worker.worker_id} successfully restarted and healthy")
            except Exception as health_e:
                logger.error(f"Error checking worker {worker.worker_id} health: {str(health_e)}")
                return False
            
            # Mark task as assigned in task manager
            success = await self.task_manager.mark_task_assigned(task_id, worker.worker_id)
            if not success:
                return False
            
            # Add to worker's task list
            if not worker.add_task(task_id, task_type):
                # Rollback assignment
                await self.task_manager.handle_task_failure(
                    task_id, worker.worker_id, task_type, 
                    "Worker at capacity"
                )
                return False
            
            # Prepare task request with additional data
            input_data = task.get('input_data', {})
            if isinstance(input_data, str):
                try:
                    import json
                    input_data = json.loads(input_data)
                except json.JSONDecodeError:
                    input_data = {}
            
            # Add cookie profile for YouTube downloads
            if task_type == 'download_youtube':
                cookie_profile = self.behavior_manager.get_cookie_profile(worker.worker_id)
                if cookie_profile:
                    input_data['cookie_profile'] = cookie_profile
                    logger.debug(f"Adding cookie profile '{cookie_profile}' to input_data for task {task_id}")
                elif getattr(worker, 'download_mode', 'cookie') == 'cookie':
                    logger.warning(f"No cookie profile found for worker {worker.worker_id} but download_mode is 'cookie'")
            
            # Add orchestrator callback URL
            input_data['orchestrator_callback_url'] = self.callback_url
            
            # Send task to worker via network manager
            api_success, response = await self.network_manager.send_request(
                worker.worker_id,
                'POST',
                '/tasks/process',
                {
                    'content_id': task['content_id'],
                    'task_type': task_type,
                    'input_data': input_data,
                    'worker_id': worker.worker_id,
                    'original_task_id': task_id
                }
            )
            
            if api_success:
                logger.info(f"Assigned task {task_id} ({task_type}) to worker {worker.worker_id}")
                
                # Reset failure count on successful assignment
                self.failure_tracker.record_success(worker.worker_id, task_type)
                return True
            else:
                # Handle API failure with retry logic
                logger.error(f"Failed to assign task {task_id} to worker {worker.worker_id}: {response}")
                
                # Record failure for tracking
                should_pause = self.failure_tracker.record_failure(
                    worker.worker_id, task_type, f"API call failed: {response}"
                )
                if should_pause:
                    logger.warning(f"Task type {task_type} paused for worker {worker.worker_id} due to failures")
                
                # Check if this was a connection error and trigger restart
                if "connection" in str(response).lower() or "timeout" in str(response).lower():
                    logger.warning(f"Connection failed to worker {worker.worker_id}. Marking as unhealthy.")
                    worker.status = 'unhealthy'
                    
                    # Trigger background restart with retry tracking
                    async def background_restart():
                        restart_success, attempted = await worker.attempt_restart_with_lock(
                            self.worker_manager.restart_unhealthy_worker
                        )
                        
                        if attempted:
                            if restart_success:
                                logger.info(f"Background restart of worker {worker.worker_id} succeeded")
                            else:
                                logger.warning(f"Background restart of worker {worker.worker_id} failed")
                        else:
                            logger.debug(f"Background restart of worker {worker.worker_id} skipped "
                                       f"(concurrent attempt or retry limit)")
                    
                    asyncio.create_task(background_restart())
                
                # Rollback on API failure
                worker.remove_task(task_id, task_type)
                await self.task_manager.handle_task_failure(
                    task_id, worker.worker_id, task_type, 
                    f"API call failed: {response}"
                )
                return False
                
        except Exception as e:
            logger.error(f"Error assigning task {task['id']} to worker {worker.worker_id}: {str(e)}")
            # Ensure cleanup on unexpected errors
            if 'task_id' in locals():
                if hasattr(worker, 'remove_task'):
                    worker.remove_task(task_id, task.get('task_type'))
                await self.task_manager.handle_task_failure(
                    task_id, worker.worker_id, task.get('task_type', 'unknown'), 
                    f"Assignment error: {str(e)}"
                )
            return False
    
    
    
    
    
    async def handle_task_callback(self, callback_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle task completion callback from worker (V1 compatible)"""
        # Extract fields from callback data
        task_id = callback_data.get('original_task_id', callback_data.get('task_id'))
        worker_id = callback_data['worker_id']
        status = callback_data['status']
        result = callback_data.get('result', {})
        duration = callback_data.get('duration', 0.0)
        chunk_index = callback_data.get('chunk_index')
        content_id = callback_data.get('content_id')
        task_type = callback_data.get('task_type')
        
        # Call the internal method
        await self._handle_task_callback_internal(
            task_id, worker_id, status, result, duration, chunk_index, content_id, task_type
        )
        
        # Return V1 compatible response
        return {
            'message': f'Task {task_id} callback processed',
            'status': 'success',
            'next_task_id': None  # V2 handles reactive assignment internally
        }
    
    async def _handle_task_callback_internal(self, task_id: str, worker_id: str, status: str,
                                 result: Dict[str, Any], duration: float,
                                 chunk_index: Optional[int] = None,
                                 content_id: Optional[str] = None,
                                 task_type: Optional[str] = None) -> None:
        """Handle task completion callback from worker"""
        try:
            logger.info(f"Processing callback for task {task_id} from {worker_id}: {status}")

            # PHASE 1: Quick task status update and worker release (non-blocking)
            with get_session() as session:
                task = session.query(TaskQueue).filter_by(id=task_id).first()
                if not task:
                    logger.error(f"Task {task_id} not found in database")
                    return

                # Use provided content_id if available, otherwise get from task
                if not content_id:
                    content_id = task.content_id

                # Use provided task_type if available, otherwise get from task
                if not task_type:
                    task_type = task.task_type

                # Update task status immediately
                task.status = 'failed' if status == 'failed' else status
                task.completed_at = datetime.now(timezone.utc)
                task.result = result
                if status == 'failed':
                    task.error = result.get('error', 'Unknown error') if isinstance(result, dict) else str(result)
                session.add(task)
                session.commit()
                logger.debug(f"Task {task_id} status committed as {status}")

                # Remove task from worker IMMEDIATELY
                worker = self.worker_pool.get_worker(worker_id)
                if worker:
                    worker.remove_task(task_id, task_type)

                # Handle S3 error detection and global pause
                error_message = result.get('error', '') if isinstance(result, dict) else str(result)
                if status == 'failed' and self.s3_monitor.should_trigger_global_pause(error_message):
                    self.s3_monitor.extend_global_pause(self, error_message)

                # Handle failure tracking and logging
                if status == 'failed':
                    should_pause = self.failure_tracker.record_failure(
                        worker_id, task_type, error_message
                    )
                    if should_pause:
                        logger.warning(f"Task type {task_type} paused for worker {worker_id} due to failures")
                    # Log error to dedicated file
                    task_logger.log_error(worker_id, task_type, content_id, error_message, chunk_index, duration)
                elif status == 'completed':
                    self.failure_tracker.record_success(worker_id, task_type)
                    # Log completion to dedicated file
                    task_logger.log_completion(worker_id, task_type, content_id, chunk_index, duration)

            # Update task manager (quick operation)
            if status == 'failed':
                await self.task_manager.handle_task_failure(
                    task_id, worker_id, task_type,
                    result.get('error', 'Unknown error')
                )
            else:
                await self.task_manager.handle_task_completion(
                    task_id, worker_id, task_type, status, result
                )

            # PHASE 2: Record download completion in behavior manager (BEFORE reactive assignment)
            # This is critical for download tasks to calculate proper wait times in main loop
            if status == 'completed' and task_type in ['download_youtube', 'download_rumble']:
                self.behavior_manager.handle_task_completion(worker_id, task_type)
                logger.info(f"[BEHAVIOR] Recorded {task_type} completion for worker {worker_id}")

            # PHASE 3: Immediate reactive task assignment (don't wait for pipeline processing)
            should_assign_immediately = status == 'completed'

            if should_assign_immediately:
                # Try to assign next task immediately
                await self.reactive_assignment.assign_next_task(worker_id, task_type)
                logger.debug(f"Reactive assignment initiated for {worker_id} before pipeline processing")

            # PHASE 3: Pipeline processing in background (async, non-blocking)
            # This handles state reconciliation, creating next tasks, etc.
            asyncio.create_task(
                self._process_pipeline_result_background(
                    task_id=task_id,
                    content_id=content_id,
                    task_type=task_type,
                    status=status,
                    result=result,
                    worker_id=worker_id,
                    chunk_index=chunk_index
                )
            )

        except Exception as e:
            logger.error(f"Error handling callback for task {task_id}: {str(e)}")

    async def _process_pipeline_result_background(self, task_id: str, content_id: str, task_type: str,
                                                   status: str, result: Dict[str, Any], worker_id: str,
                                                   chunk_index: Optional[int]) -> None:
        """
        Process pipeline manager logic in background (non-blocking).
        This handles state reconciliation, creating next tasks, and worker availability.
        """
        try:
            logger.debug(f"Background pipeline processing started for task {task_id}")

            with get_session() as session:
                # Look up content
                content = session.query(Content).filter_by(content_id=content_id).first()
                if not content:
                    logger.error(f"Content {content_id} not found in background pipeline processing")
                    return

                # Look up task
                task = session.query(TaskQueue).filter_by(id=task_id).first()
                if not task:
                    logger.error(f"Task {task_id} not found in background pipeline processing")
                    return

                # Handle task completion with pipeline manager
                pipeline_result = await self.pipeline_manager.handle_task_result(
                    session=session,
                    content_id=content_id,
                    task_type=task_type,
                    status=status,
                    result=result,
                    db_task=task,
                    content=content,
                    worker_id=worker_id,
                    chunk_index=chunk_index
                )

                # Commit any changes made by pipeline manager
                session.commit()
                logger.info(f"Background pipeline processing completed for task {task_id} ({task_type}): {pipeline_result.get('outcome_message', 'No message')}")

                # Handle post-task delays (like download cooldowns)
                if pipeline_result.get('available_after'):
                    worker = self.worker_pool.get_worker(worker_id)
                    if worker:
                        delay_seconds = (pipeline_result['available_after'] - datetime.now(timezone.utc)).total_seconds()
                        if delay_seconds > 0:
                            logger.info(f"Setting post-task delay of {delay_seconds:.1f}s for worker {worker_id} after {task_type} task")
                            # Note: We don't need to call assign here since we already did reactive assignment

        except Exception as e:
            logger.error(f"Error in background pipeline processing for task {task_id}: {str(e)}", exc_info=True)

    async def restart_worker(self, worker_id: str) -> bool:
        """Restart a specific worker with enhanced retry logic"""
        try:
            worker = self.worker_pool.get_worker(worker_id)
            if not worker:
                logger.error(f"Worker {worker_id} not found")
                return False
            logger.info(f"Manual restart requested for worker {worker_id} "
                       f"(current status: {worker.status}, attempts: {worker.restart_attempts}/{worker.max_restart_attempts})")
            
            # Reset retry tracking for manual restart to allow immediate restart
            original_attempts = worker.restart_attempts
            worker.restart_attempts = 0
            worker.last_restart_attempt = None
            
            # Use the network monitor's restart logic
            success = await self.network_monitor._attempt_worker_restart(worker_id, worker)
            
            if success:
                logger.info(f"Manual restart of worker {worker_id} succeeded")
            else:
                logger.error(f"Manual restart of worker {worker_id} failed")
                # If manual restart failed, restore some of the attempt count
                # but give it another chance
                worker.restart_attempts = min(original_attempts, worker.max_restart_attempts - 1)
            
            return success
            
        except Exception as e:
            logger.error(f"Error manually restarting worker {worker_id}: {str(e)}")
            return False
    
    def get_worker_status(self, worker_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed status of a specific worker including retry information"""
        worker = self.worker_pool.get_worker(worker_id)
        if not worker:
            return None
        status = {
            'worker_id': worker_id,
            'status': worker.status,
            'active_tasks': worker.active_tasks,
            'max_concurrent_tasks': worker.max_concurrent_tasks,
            'task_types': worker.task_types,
            'task_counts_by_type': worker.get_task_counts_by_type(),
            'last_heartbeat': worker.last_heartbeat.isoformat() if worker.last_heartbeat else None,
            'started_at': worker.started_at.isoformat() if worker.started_at else None,
            'available_after': worker.available_after.isoformat() if worker.available_after else None,
            'restart_info': {
                'restart_attempts': worker.restart_attempts,
                'max_restart_attempts': worker.max_restart_attempts,
                'last_restart_attempt': worker.last_restart_attempt.isoformat() if worker.last_restart_attempt else None,
                'can_restart': worker.should_attempt_restart(),
                'backoff_remaining': worker.get_restart_backoff_remaining()
            },
            'network_info': {
                'eth_ip': worker.eth_ip,
                'wifi_ip': worker.wifi_ip,
                'is_head_worker': worker.is_head_worker
            }
        }
        
        return status
    
    def get_all_workers_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all workers"""
        return {
            worker_id: self.get_worker_status(worker_id)
            for worker_id in self.worker_pool.workers.keys()
        }
    
    def set_global_pause(self, duration_seconds: int) -> datetime:
        """Set global pause for task assignment"""
        self.global_pause_until = datetime.now(timezone.utc) + timedelta(seconds=duration_seconds)
        logger.info(f"Global task assignment paused until {self.global_pause_until}")
        return self.global_pause_until
    
    def configure_test_mode(self, worker_id: str, task_type: str, iterations: int):
        """Configure test mode parameters"""
        self.test_config = {
            'worker_id': worker_id,
            'task_type': task_type,
            'iterations': iterations,
            'completed': 0
        }
        logger.info(f"Configured test mode: worker={worker_id}, task={task_type}, iterations={iterations}")
    
    async def initialize_test_worker(self):
        """Initialize single worker for test mode"""
        if not self.test_config:
            raise ValueError("Test mode not configured")
        
        worker_configs = self.config.get('processing', {}).get('workers', {})
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
        
        # Create worker info
        worker_info = WorkerInfo(worker_id, config)
        self.worker_pool.workers[worker_id] = worker_info
        
        # Register with network manager
        self.network_manager.register_worker(worker_id, config)
        
        # Update worker manager
        worker_obj = type('Worker', (), {
            'worker_id': worker_id,
            'current_ip': worker_info.eth_ip or worker_info.wifi_ip or 'localhost',
            'is_head_worker': worker_info.is_head_worker,
            'type': 'head' if worker_info.is_head_worker else 'worker',
            'api_url': f"http://{worker_info.eth_ip or worker_info.wifi_ip or 'localhost'}:8000",
            'status': worker_info.status
        })()
        self.worker_manager.workers[worker_id] = worker_obj
        
        # Start worker services
        success = await self._start_worker_services(worker_id)
        if success:
            worker_info.status = 'active'
            worker_info.started_at = datetime.now(timezone.utc)
            logger.info(f"Test worker {worker_id} initialized successfully")
        else:
            raise RuntimeError(f"Failed to start services for test worker {worker_id}")
        
        # Register with behavior manager
        self.behavior_manager.register_worker(worker_id, config)
    
    async def process_test_tasks(self):
        """Process tasks in test mode"""
        worker = self.worker_pool.get_worker(self.test_config['worker_id'])
        
        while not self.should_stop and self.test_config['completed'] < self.test_config['iterations']:
            try:
                # Check global pause - test mode should respect this too
                if self.global_pause_until and datetime.now(timezone.utc) < self.global_pause_until:
                    await asyncio.sleep(self.config_manager.idle_sleep_interval)
                    continue
                elif self.global_pause_until:
                    logger.info("Test mode: Global pause ended, resuming task assignment")
                    self.global_pause_until = None
                
                # Get next test task
                tasks = await self.task_manager.get_next_tasks(
                    limit=1,
                    task_types=[self.test_config['task_type']],
                    exclude_task_ids=self.task_manager.assigned_tasks
                )
                
                if not tasks:
                    logger.info(f"Test mode: No {self.test_config['task_type']} tasks available, waiting...")
                    await asyncio.sleep(self.config_manager.idle_sleep_interval)
                    continue
                
                task = tasks[0]
                success = await self._assign_task_to_worker(task, worker)
                
                if success:
                    self.test_config['completed'] += 1
                    logger.info(f"Test mode: Assigned task {self.test_config['completed']}/{self.test_config['iterations']}")
                else:
                    logger.warning(f"Test mode: Failed to assign task")
                
                # Small delay between assignments
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error in test mode task processing: {str(e)}")
                await asyncio.sleep(self.idle_sleep_interval)
        
        logger.info(f"Test mode completed: {self.test_config['completed']} tasks assigned")
    
    async def shutdown(self) -> None:
        """Graceful shutdown"""
        logger.info("Shutting down Task Orchestrator V2...")

        self.should_stop = True

        # Stop task queue cache
        await self.task_manager.shutdown_cache()

        # Stop monitoring components
        await self.timeout_manager.stop_monitoring()
        await self.network_monitor.stop_monitoring()
        await self.s3_monitor.stop_health_monitoring()

        # Stop unified scheduled task manager
        await self.scheduled_task_manager.stop()

        # Stop LLM server and dashboards
        await self.service_startup.stop_services()

        # Stop all worker services
        for worker_id in self.worker_pool.workers.keys():
            await self.service_manager.stop_task_processor(worker_id)
            await self.service_manager.stop_model_server(worker_id)

        # Cleanup network manager
        await self.network_manager.cleanup()

        logger.info("Orchestrator shutdown complete")

def kill_existing_orchestrator(port: int = 8001) -> None:
    """Kill any existing process using the orchestrator port"""
    try:
        # Find process using the port
        result = subprocess.run(['lsof', '-ti', f':{port}'],
                               capture_output=True, text=True, check=False)

        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid:
                    try:
                        print(f"Killing existing orchestrator process (PID: {pid})")
                        # Try graceful termination first
                        subprocess.run(['kill', '-TERM', pid], check=False)
                        # Wait a moment for graceful shutdown
                        import time
                        time.sleep(2)
                        # Force kill if still running
                        subprocess.run(['kill', '-9', pid], check=False, stderr=subprocess.DEVNULL)
                        print(f"Successfully terminated process {pid}")
                    except Exception as e:
                        print(f"Warning: Could not kill process {pid}: {e}")
        else:
            print(f"No existing process found on port {port}")
    except Exception as e:
        print(f"Warning: Error checking for existing orchestrator: {e}")

async def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Task Orchestrator V2')
    parser.add_argument('--test-mode', action='store_true', help='Run in test mode')
    parser.add_argument('--worker', type=str, help='Worker ID for test mode')
    parser.add_argument('--task-type', type=str, help='Task type for test mode')
    parser.add_argument('--iterations', type=int, default=10, help='Number of tasks to process in test mode')
    args = parser.parse_args()

    # Kill any existing orchestrator process before starting
    kill_existing_orchestrator()

    logger.info("Task Orchestrator V2 starting up")
    
    orchestrator = TaskOrchestratorV2()
    
    # Set up signal handlers
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        orchestrator.should_stop = True
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    logger.info("Signal handlers configured")
    
    try:
        # Handle test mode
        if args.test_mode:
            if not args.worker or not args.task_type:
                logger.error("Test mode requires --worker and --task-type arguments")
                return
            
            logger.info(f"Running in test mode: worker={args.worker}, task={args.task_type}, iterations={args.iterations}")
            orchestrator.configure_test_mode(args.worker, args.task_type, args.iterations)
            
            # Initialize test worker only
            await orchestrator.test_mode.initialize_test_worker()
            
            # Start API server in background (still needed for callbacks)
            logger.info(f"Starting API server on port {orchestrator.api_port}")
            config = uvicorn.Config(
                orchestrator.app,
                host=orchestrator.api_host,
                port=orchestrator.api_port,
                log_level="warning",
                access_log=False  # Suppress HTTP request logs
            )
            server = uvicorn.Server(config)
            server_task = asyncio.create_task(server.serve())
            
            # Process test tasks
            test_task = asyncio.create_task(orchestrator.test_mode.process_test_tasks())
            
            # Wait for test completion or interruption
            done, pending = await asyncio.wait(
                [server_task, test_task],
                return_when=asyncio.FIRST_COMPLETED
            )
        else:
            # Normal mode
            # Start API server FIRST so it's ready to receive callbacks
            logger.info(f"Starting API server on port {orchestrator.api_port}")
            config = uvicorn.Config(
                orchestrator.app,
                host=orchestrator.api_host,
                port=orchestrator.api_port,
                log_level="warning",
                access_log=False  # Suppress HTTP request logs
            )
            server = uvicorn.Server(config)
            server_task = asyncio.create_task(server.serve())
            logger.info("API server started in background")

            # Brief pause to ensure server is listening
            await asyncio.sleep(0.5)

            # Now initialize workers (they may call back immediately)
            logger.info("Initializing orchestrator")
            success = await orchestrator.initialize()
            if not success:
                logger.error("Failed to initialize orchestrator")
                return

            # Run main orchestrator loop
            logger.info("Starting main orchestrator loop")
            orchestrator_task = asyncio.create_task(orchestrator.run())
            
            # Wait for either to complete
            done, pending = await asyncio.wait(
                [server_task, orchestrator_task],
                return_when=asyncio.FIRST_COMPLETED
            )
        
        # Cancel pending tasks
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
    finally:
        await orchestrator.shutdown()

if __name__ == "__main__":
    asyncio.run(main())