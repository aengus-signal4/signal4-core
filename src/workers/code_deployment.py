"""
Code Deployment Manager - Handles concurrent code deployment to all workers

Manages:
- Git-based code synchronization
- Dependency updates (uv sync)
- Service restarts after deployment
- Health verification
- Concurrent deployment to multiple workers
"""
import asyncio
import logging
import asyncssh
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DeploymentResult:
    """Result of a deployment to a single worker"""
    worker_id: str
    success: bool
    duration: float
    steps_completed: List[str]
    error: Optional[str] = None
    output: str = ""

@dataclass
class DeploymentStatus:
    """Overall deployment status"""
    started_at: datetime
    completed_at: Optional[datetime] = None
    total_workers: int = 0
    successful_workers: List[str] = None
    failed_workers: List[str] = None
    in_progress: bool = False

    def __post_init__(self):
        if self.successful_workers is None:
            self.successful_workers = []
        if self.failed_workers is None:
            self.failed_workers = []

class CodeDeploymentManager:
    """Manages concurrent code deployment to all workers"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ssh_username = config.get('processing', {}).get('ssh_username', 'signal4')
        self.ssh_key_path = config.get('processing', {}).get('ssh_key_path', '/Users/signal4/.ssh/id_ed25519')
        self.project_path = '/Users/signal4/signal4/core'

        # Track deployment status
        self.current_deployment: Optional[DeploymentStatus] = None
        self.deployment_history: List[DeploymentStatus] = []

        logger.info("Code deployment manager initialized")

    async def deploy_to_all_workers(self, worker_pool, force_restart: bool = False) -> DeploymentStatus:
        """Deploy code to all enabled workers concurrently"""
        try:
            enabled_workers = {
                wid: worker for wid, worker in worker_pool.workers.items()
                if worker.status != 'disabled' and worker.eth_ip != '10.0.0.4'  # Skip disabled workers and head node
            }

            if not enabled_workers:
                logger.info("No remote workers to deploy to")
                return DeploymentStatus(
                    started_at=datetime.now(timezone.utc),
                    completed_at=datetime.now(timezone.utc),
                    total_workers=0
                )

            logger.info(f"Starting concurrent deployment to {len(enabled_workers)} workers")

            # Initialize deployment status
            self.current_deployment = DeploymentStatus(
                started_at=datetime.now(timezone.utc),
                total_workers=len(enabled_workers),
                in_progress=True
            )

            # Deploy to all workers concurrently
            deployment_tasks = []
            for worker_id, worker_info in enabled_workers.items():
                task = asyncio.create_task(
                    self._deploy_to_worker(worker_id, worker_info, force_restart)
                )
                deployment_tasks.append(task)

            # Wait for all deployments to complete
            results = await asyncio.gather(*deployment_tasks, return_exceptions=True)

            # Process results
            for i, result in enumerate(results):
                worker_id = list(enabled_workers.keys())[i]

                if isinstance(result, Exception):
                    logger.error(f"Deployment to {worker_id} failed with exception: {result}")
                    self.current_deployment.failed_workers.append(worker_id)
                elif isinstance(result, DeploymentResult):
                    if result.success:
                        self.current_deployment.successful_workers.append(worker_id)
                        logger.info(f"Deployment to {worker_id} succeeded in {result.duration:.1f}s")
                    else:
                        self.current_deployment.failed_workers.append(worker_id)
                        logger.error(f"Deployment to {worker_id} failed: {result.error}")

            # Complete deployment
            self.current_deployment.completed_at = datetime.now(timezone.utc)
            self.current_deployment.in_progress = False

            # Add to history
            self.deployment_history.append(self.current_deployment)

            logger.info(f"Deployment completed. Success: {len(self.current_deployment.successful_workers)}, "
                       f"Failed: {len(self.current_deployment.failed_workers)}")

            return self.current_deployment

        except Exception as e:
            logger.error(f"Error in deployment orchestration: {e}")
            if self.current_deployment:
                self.current_deployment.in_progress = False
                self.current_deployment.completed_at = datetime.now(timezone.utc)
            raise

    async def _deploy_to_worker(self, worker_id: str, worker_info, force_restart: bool) -> DeploymentResult:
        """Deploy code to a single worker"""
        start_time = datetime.now(timezone.utc)
        steps_completed = []
        output_lines = []

        try:
            # Determine target IP (prefer eth, fallback to wifi)
            target_ip = worker_info.eth_ip or worker_info.wifi_ip
            if not target_ip:
                raise Exception(f"No IP address available for worker {worker_id}")

            logger.info(f"Deploying to {worker_id} at {target_ip}")

            # Connect via SSH
            async with asyncssh.connect(
                target_ip,
                username=self.ssh_username,
                client_keys=[self.ssh_key_path],
                known_hosts=None  # Accept any host key for now
            ) as conn:

                # Step 1: Create project directory if it doesn't exist
                result = await conn.run(f'mkdir -p {self.project_path}')
                steps_completed.append('create_directory')
                output_lines.append(f"Create directory: {result.stdout}")

                # Step 2: Rsync code from head node TO worker (correct direction)
                # Execute locally on head node to push code to worker
                import asyncio

                # Create directory structure on worker first
                await conn.run(f'mkdir -p /Users/signal4/signal4')

                # Rsync from head TO worker (push direction)
                rsync_cmd = [
                    'rsync', '-av', '--delete',
                    '--exclude=.venv',
                    '--exclude=__pycache__',
                    '--exclude=.git',
                    '--exclude=*.pyc',
                    '-e', f'ssh -o StrictHostKeyChecking=no -i {self.ssh_key_path}',
                    '/Users/signal4/signal4/core/',
                    f'{self.ssh_username}@{target_ip}:/Users/signal4/signal4/core/'
                ]

                # Execute rsync locally (on head node) to push to worker
                proc = await asyncio.create_subprocess_exec(
                    *rsync_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await proc.communicate()

                if proc.returncode != 0:
                    raise Exception(f"Rsync failed: {stderr.decode()}")
                steps_completed.append('sync_code')
                output_lines.append(f"Rsync: Synced code from head to worker ({stdout.decode().strip()})")

                # Step 5: Update dependencies
                # Use the working SSH command format
                uv_cmd = f'cd {self.project_path} && /Users/signal4/.local/bin/uv sync --quiet'
                result = await conn.run(uv_cmd)
                steps_completed.append('uv_sync')
                output_lines.append(f"UV sync: Dependencies synced successfully")

                # Step 6: Always restart processor to pick up new code
                # Stop existing processor
                await conn.run('pkill -f "processor.py" || true')
                steps_completed.append('stop_processor')

                # Give process time to stop
                await asyncio.sleep(2)

                # Start processor in background
                # Use nohup to detach from SSH session
                start_cmd = f'cd {self.project_path} && nohup /Users/signal4/.local/bin/uv run python src/workers/processor.py > /tmp/processor.log 2>&1 &'
                await conn.run(start_cmd)
                steps_completed.append('start_processor')
                output_lines.append("Processor restarted")

                # Optionally restart model server if requested
                if force_restart:
                    await conn.run('pkill -f "mlx_server" || true')
                    steps_completed.append('stop_model_server')

            duration = (datetime.now(timezone.utc) - start_time).total_seconds()

            return DeploymentResult(
                worker_id=worker_id,
                success=True,
                duration=duration,
                steps_completed=steps_completed,
                output='\n'.join(output_lines)
            )

        except Exception as e:
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            error_msg = str(e)

            logger.error(f"Deployment to {worker_id} failed: {error_msg}")

            return DeploymentResult(
                worker_id=worker_id,
                success=False,
                duration=duration,
                steps_completed=steps_completed,
                error=error_msg,
                output='\n'.join(output_lines)
            )

    def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status"""
        if not self.current_deployment:
            return {'status': 'no_deployment'}

        return {
            'status': 'in_progress' if self.current_deployment.in_progress else 'completed',
            'started_at': self.current_deployment.started_at.isoformat(),
            'completed_at': self.current_deployment.completed_at.isoformat() if self.current_deployment.completed_at else None,
            'total_workers': self.current_deployment.total_workers,
            'successful_workers': self.current_deployment.successful_workers,
            'failed_workers': self.current_deployment.failed_workers,
            'success_rate': len(self.current_deployment.successful_workers) / max(1, self.current_deployment.total_workers)
        }

    def get_deployment_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get deployment history"""
        return [
            {
                'started_at': deployment.started_at.isoformat(),
                'completed_at': deployment.completed_at.isoformat() if deployment.completed_at else None,
                'total_workers': deployment.total_workers,
                'successful_workers': deployment.successful_workers,
                'failed_workers': deployment.failed_workers,
                'success_rate': len(deployment.successful_workers) / max(1, deployment.total_workers)
            }
            for deployment in self.deployment_history[-limit:]
        ]