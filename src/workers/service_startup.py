"""
Service Startup Manager - Manages central services on head node

Manages startup of:
- LLM server
- Monitoring dashboards (worker, project, orchestrator, system)
- Audio server
- Backend API
"""
import asyncio
import subprocess
import logging
import signal
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

class ServiceStartupManager:
    """Manages startup of central services on head node"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logger

        # Service configuration
        self.python_path = config.get('processing', {}).get('head_python_path',
                                                             '/opt/homebrew/Caskroom/miniforge/base/envs/content-processing/bin/python')
        self.base_path = Path(config.get('storage', {}).get('local', {}).get('base_path',
                                                                              '/Users/signal4/signal4/core'))

        # LLM Server configuration
        llm_config = config.get('processing', {}).get('llm_server', {})
        self.llm_enabled = llm_config.get('enabled', False)
        self.llm_port = llm_config.get('port', 8002)
        self.llm_script = self.base_path / 'src' / 'services' / 'llm' / 'server.py'

        # Dashboard configuration (only system monitoring dashboard)
        dashboard_config = config.get('dashboards', {})
        self.dashboards_enabled = dashboard_config.get('enabled', True)
        self.system_dashboard_port = dashboard_config.get('system_monitoring_port', 8501)

        # Audio Server configuration
        services_config = config.get('services', {})
        audio_config = services_config.get('audio_server', {})
        self.audio_server_enabled = audio_config.get('enabled', True)
        self.audio_server_port = audio_config.get('port', 8010)
        self.audio_server_script = self.base_path / 'src' / 'services' / 'audio_server.py'

        # Backend API configuration
        backend_config = services_config.get('backend', {})
        self.backend_enabled = backend_config.get('enabled', True)
        self.backend_port = backend_config.get('port', 7999)
        self.backend_module = 'src.backend.app.main:app'

        # Model Servers configuration
        model_servers_config = services_config.get('model_servers', {})
        self.model_servers_enabled = model_servers_config.get('enabled', False)
        self.model_server_instances = model_servers_config.get('instances', {})
        self.model_server_script = self.base_path / 'src' / 'services' / 'llm' / 'mlx_server.py'

        # Process tracking
        self.processes = {}
        self.should_stop = False

    async def start_services(self):
        """Start all configured services"""
        # Start backend API first (other services may depend on it)
        if self.backend_enabled:
            await self._start_backend()

        # Start audio server
        if self.audio_server_enabled:
            await self._start_audio_server()

        # Start LLM server
        if self.llm_enabled:
            await self._start_llm_server()

        # Start model servers (infrastructure services)
        if self.model_servers_enabled:
            await self._start_model_servers()

        # Start dashboards last (they depend on other services being up)
        if self.dashboards_enabled:
            await self._start_dashboards()

        self.logger.info(
            f"Service startup manager initialized. Services: "
            f"Backend={self.backend_enabled}, Audio={self.audio_server_enabled}, "
            f"LLM={self.llm_enabled}, ModelServers={self.model_servers_enabled}, Dashboards={self.dashboards_enabled}"
        )

    async def stop_services(self):
        """Stop all managed services"""
        self.should_stop = True

        for service_name, process in self.processes.items():
            if process and process.returncode is None:
                self.logger.info(f"Stopping {service_name}...")
                try:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    self.logger.info(f"Stopped {service_name}")
                except Exception as e:
                    self.logger.error(f"Error stopping {service_name}: {e}")

        self.processes.clear()

    async def _start_llm_server(self):
        """Start the LLM server"""
        try:
            self.logger.info(f"Starting LLM server on port {self.llm_port}...")

            # Check if already running
            if await self._check_port_in_use(self.llm_port):
                self.logger.warning(f"LLM server already running on port {self.llm_port}")
                return

            # Start LLM server
            cmd = [
                str(self.python_path),
                str(self.llm_script),
                '--port', str(self.llm_port)
            ]

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.base_path)
            )

            self.processes['llm_server'] = process

            # Wait a moment and check if it's running
            await asyncio.sleep(2)

            if process.poll() is None:
                self.logger.info(f"LLM server started successfully on port {self.llm_port}")
            else:
                stdout, stderr = process.communicate()
                self.logger.error(f"LLM server failed to start. stdout: {stdout.decode()}, stderr: {stderr.decode()}")

        except Exception as e:
            self.logger.error(f"Error starting LLM server: {e}", exc_info=True)

    async def _start_backend(self):
        """Start the Backend API server"""
        try:
            self.logger.info(f"Starting Backend API on port {self.backend_port}...")

            # Check if already running
            if await self._check_port_in_use(self.backend_port):
                self.logger.warning(f"Backend API already running on port {self.backend_port}")
                return

            # Start Backend API using uvicorn
            cmd = [
                str(self.python_path),
                '-m', 'uvicorn',
                self.backend_module,
                '--host', '0.0.0.0',
                '--port', str(self.backend_port)
            ]

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.base_path)
            )

            self.processes['backend'] = process

            # Wait a moment and check if it's running
            await asyncio.sleep(3)

            if process.poll() is None:
                self.logger.info(f"Backend API started successfully on port {self.backend_port}")
            else:
                stdout, stderr = process.communicate()
                self.logger.error(f"Backend API failed to start. stdout: {stdout.decode()}, stderr: {stderr.decode()}")

        except Exception as e:
            self.logger.error(f"Error starting Backend API: {e}", exc_info=True)

    async def _start_audio_server(self):
        """Start the Audio server"""
        try:
            self.logger.info(f"Starting Audio server on port {self.audio_server_port}...")

            # Check if already running
            if await self._check_port_in_use(self.audio_server_port):
                self.logger.warning(f"Audio server already running on port {self.audio_server_port}")
                return

            # Check if script exists
            if not self.audio_server_script.exists():
                self.logger.warning(f"Audio server script not found: {self.audio_server_script}")
                return

            # Start Audio server
            cmd = [
                str(self.python_path),
                str(self.audio_server_script)
            ]

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.base_path)
            )

            self.processes['audio_server'] = process

            # Wait a moment and check if it's running
            await asyncio.sleep(2)

            if process.poll() is None:
                self.logger.info(f"Audio server started successfully on port {self.audio_server_port}")
            else:
                stdout, stderr = process.communicate()
                self.logger.error(f"Audio server failed to start. stdout: {stdout.decode()}, stderr: {stderr.decode()}")

        except Exception as e:
            self.logger.error(f"Error starting Audio server: {e}", exc_info=True)

    async def _start_dashboards(self):
        """Start the system monitoring dashboard"""
        try:
            # Start system monitoring dashboard (unified view with all monitoring)
            await self._start_dashboard(
                'system_monitoring',
                'dashboards/system_monitoring.py',
                self.system_dashboard_port
            )

        except Exception as e:
            self.logger.error(f"Error starting dashboard: {e}", exc_info=True)

    async def _start_dashboard(self, name: str, script_path: str, port: int):
        """Start a specific Streamlit dashboard"""
        try:
            full_script_path = self.base_path / script_path

            if not full_script_path.exists():
                self.logger.warning(f"Dashboard script not found: {full_script_path}")
                return

            self.logger.info(f"Starting {name} dashboard on port {port}...")

            # Check if already running
            if await self._check_port_in_use(port):
                self.logger.warning(f"{name} dashboard already running on port {port}")
                return

            # Start Streamlit dashboard
            cmd = [
                'streamlit', 'run',
                str(full_script_path),
                '--server.port', str(port),
                '--server.headless', 'true',
                '--browser.gatherUsageStats', 'false'
            ]

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.base_path)
            )

            self.processes[name] = process

            # Wait a moment and check if it's running
            await asyncio.sleep(3)

            if process.poll() is None:
                self.logger.info(f"{name} dashboard started successfully on port {port}")
            else:
                stdout, stderr = process.communicate()
                self.logger.error(f"{name} dashboard failed to start. stdout: {stdout.decode()}, stderr: {stderr.decode()}")

        except Exception as e:
            self.logger.error(f"Error starting {name} dashboard: {e}", exc_info=True)

    async def _start_model_servers(self):
        """Start model server instances as infrastructure services"""
        try:
            self.logger.info("Starting model server instances...")

            for instance_name, instance_config in self.model_server_instances.items():
                if not instance_config.get('enabled', True):
                    self.logger.info(f"Skipping disabled model server: {instance_name}")
                    continue

                await self._start_model_server_instance(instance_name, instance_config)

        except Exception as e:
            self.logger.error(f"Error starting model servers: {e}", exc_info=True)

    async def _start_model_server_instance(self, instance_name: str, config: dict):
        """Start a specific model server instance"""
        try:
            host = config.get('host', 'localhost')
            port = config.get('port', 8004)

            self.logger.info(f"Starting model server {instance_name} on {host}:{port}...")

            # For remote hosts, we need to use SSH to start the service
            if host != 'localhost' and host != '10.0.0.4':  # Not local/head node
                await self._start_remote_model_server(instance_name, host, port, config)
            else:
                # Local model server startup
                await self._start_local_model_server(instance_name, port, config)

        except Exception as e:
            self.logger.error(f"Error starting model server {instance_name}: {e}", exc_info=True)

    async def _start_local_model_server(self, instance_name: str, port: int, config: dict):
        """Start a local model server instance"""
        try:
            # Check if already running
            if await self._check_port_in_use(port):
                self.logger.warning(f"Model server {instance_name} already running on port {port}")
                return

            # Prepare environment with model configuration
            env = {}
            env.update({
                'ALLOWED_MODELS': ','.join(config.get('allowed_models', [])),
                'DEFAULT_MODEL': config.get('default_model', ''),
                'PORT': str(port),
                'HOST': '0.0.0.0'
            })

            # Start model server
            cmd = [
                str(self.python_path),
                str(self.model_server_script),
                '--port', str(port),
                '--host', '0.0.0.0'
            ]

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(self.base_path),
                env={**os.environ, **env}
            )

            self.processes[f'model_server_{instance_name}'] = process

            # Wait a moment and check if it's running
            await asyncio.sleep(3)

            if process.poll() is None:
                self.logger.info(f"Model server {instance_name} started successfully on port {port}")
            else:
                stdout, stderr = process.communicate()
                self.logger.error(f"Model server {instance_name} failed to start. stdout: {stdout.decode()}, stderr: {stderr.decode()}")

        except Exception as e:
            self.logger.error(f"Error starting local model server {instance_name}: {e}", exc_info=True)

    async def _start_remote_model_server(self, instance_name: str, host: str, port: int, config: dict):
        """Start a remote model server instance via SSH"""
        try:
            self.logger.info(f"Starting remote model server {instance_name} on {host}:{port}")

            # Import here to avoid dependency issues
            import asyncssh

            # SSH connection parameters
            ssh_username = 'signal4'  # Could be configurable
            ssh_key_path = '/Users/signal4/.ssh/id_ed25519'

            # Connect via SSH and start model server
            async with asyncssh.connect(
                host,
                username=ssh_username,
                client_keys=[ssh_key_path],
                known_hosts=None  # Accept any host key for now
            ) as conn:

                # Check if model server is already running
                result = await conn.run(f'lsof -i :{port} || echo "not_running"')
                if 'not_running' not in result.stdout:
                    self.logger.warning(f"Model server {instance_name} already running on {host}:{port}")
                    return

                # Prepare environment and command
                allowed_models = config.get('allowed_models', [])
                default_model = config.get('default_model', '')

                # Create startup script
                startup_script = f'''#!/bin/bash
set -e

# Change to project directory
cd /Users/signal4/signal4/core

# Kill any existing model server on this port
pkill -f "port {port}" || true
sleep 2

# Set environment variables
export PYTHONPATH=/Users/signal4/signal4/core:$PYTHONPATH

# Start model server
echo "[$(date)] Starting model server {instance_name} on port {port}"
/Users/signal4/.local/bin/uv run python src/services/llm/mlx_server.py \\
    --port {port} \\
    --host 0.0.0.0 \\
    {"--models " + ",".join(allowed_models) if allowed_models else ""} \\
    {"--default-model " + default_model if default_model else ""} \\
    > /tmp/model_server_{instance_name}.log 2>&1 &

echo $! > /tmp/model_server_{instance_name}.pid
echo "[$(date)] Model server {instance_name} started with PID $!"
'''

                # Write startup script to remote host
                script_path = f'/tmp/start_model_server_{instance_name}.sh'
                await conn.run(f'cat > {script_path}', input=startup_script)
                await conn.run(f'chmod +x {script_path}')

                # Execute startup script in background
                result = await conn.run(f'bash {script_path}')

                if result.returncode == 0:
                    self.logger.info(f"Remote model server {instance_name} started successfully on {host}:{port}")
                else:
                    raise Exception(f"Startup script failed: {result.stderr}")

                # Wait a moment and verify it's running
                await asyncio.sleep(3)
                result = await conn.run(f'lsof -i :{port} || echo "failed"')
                if 'failed' in result.stdout:
                    raise Exception(f"Model server failed to start on {host}:{port}")

                self.logger.info(f"Verified model server {instance_name} is listening on {host}:{port}")

        except Exception as e:
            self.logger.error(f"Error starting remote model server {instance_name}: {e}", exc_info=True)

    async def _check_port_in_use(self, port: int) -> bool:
        """Check if a port is already in use"""
        import socket
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('127.0.0.1', port))
            return result == 0
        except Exception:
            return False
        finally:
            if sock:
                try:
                    sock.close()
                except Exception:
                    pass

    def get_status(self) -> Dict[str, Any]:
        """Get status of all managed services"""
        status = {
            'backend': {
                'enabled': self.backend_enabled,
                'port': self.backend_port,
                'running': False
            },
            'audio_server': {
                'enabled': self.audio_server_enabled,
                'port': self.audio_server_port,
                'running': False
            },
            'llm_server': {
                'enabled': self.llm_enabled,
                'port': self.llm_port,
                'running': False
            },
            'dashboards': {
                'enabled': self.dashboards_enabled,
                'system_monitoring': {
                    'port': self.system_dashboard_port,
                    'running': False
                }
            },
            'model_servers': {
                'enabled': self.model_servers_enabled,
                'instances': {}
            }
        }

        # Add model server instances to status
        if self.model_servers_enabled:
            for instance_name, instance_config in self.model_server_instances.items():
                status['model_servers']['instances'][instance_name] = {
                    'enabled': instance_config.get('enabled', True),
                    'host': instance_config.get('host', 'localhost'),
                    'port': instance_config.get('port', 8004),
                    'tiers': instance_config.get('tiers', []),
                    'running': False
                }

        # Check process status
        for service_name, process in self.processes.items():
            if process and process.returncode is None:
                if service_name == 'backend':
                    status['backend']['running'] = True
                elif service_name == 'audio_server':
                    status['audio_server']['running'] = True
                elif service_name == 'llm_server':
                    status['llm_server']['running'] = True
                elif service_name == 'system_monitoring':
                    status['dashboards']['system_monitoring']['running'] = True
                elif service_name.startswith('model_server_'):
                    instance_name = service_name.replace('model_server_', '')
                    if instance_name in status['model_servers']['instances']:
                        status['model_servers']['instances'][instance_name]['running'] = True

        return status

    async def restart_service(self, service_name: str) -> bool:
        """Restart a specific service"""
        try:
            # Stop the service
            if service_name in self.processes:
                process = self.processes[service_name]
                if process and process.returncode is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                del self.processes[service_name]

            # Restart based on service name
            if service_name == 'backend':
                await self._start_backend()
            elif service_name == 'audio_server':
                await self._start_audio_server()
            elif service_name == 'llm_server':
                await self._start_llm_server()
            elif service_name == 'system_monitoring':
                await self._start_dashboard('system_monitoring', 'dashboards/system_monitoring.py', self.system_dashboard_port)
            else:
                self.logger.error(f"Unknown service: {service_name}")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error restarting {service_name}: {e}", exc_info=True)
            return False