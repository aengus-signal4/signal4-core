"""
Service Startup Manager - Manages central services on head node

Manages startup of:
- LLM server
- Monitoring dashboards (system)
- Backend API
- Embedding Server
- Model Servers

All services run in named screen sessions for persistence across orchestrator restarts.
"""
import asyncio
import subprocess
import logging
import signal
import os
import shlex
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


async def check_screen_exists(screen_name: str) -> bool:
    """Check if a screen session with the given name exists."""
    try:
        process = await asyncio.create_subprocess_exec(
            'screen', '-ls',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await process.communicate()
        output = stdout.decode('utf-8', errors='replace')

        for line in output.split('\n'):
            if f'.{screen_name}' in line or f'\t{screen_name}\t' in line:
                return True
        return False
    except Exception as e:
        logger.warning(f"Error checking screen sessions: {e}")
        return False


async def kill_screen_session(screen_name: str) -> bool:
    """Kill a screen session by name."""
    try:
        process = await asyncio.create_subprocess_exec(
            'screen', '-S', screen_name, '-X', 'quit',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await process.communicate()
        return True
    except Exception as e:
        logger.warning(f"Error killing screen session {screen_name}: {e}")
        return False

class ServiceStartupManager:
    """Manages startup of central services on head node"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logger

        # Service configuration
        # Use uv for running Python scripts to ensure correct environment
        self.uv_path = config.get('processing', {}).get('uv_path', '/Users/signal4/.local/bin/uv')
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

        # Embedding Server configuration
        embedding_config = services_config.get('embedding_server', {})
        self.embedding_server_enabled = embedding_config.get('enabled', True)
        self.embedding_server_port = embedding_config.get('port', 8005)
        self.embedding_server_script = self.base_path / 'src' / 'services' / 'embedding_server.py'

        # Memory Monitor configuration
        memory_monitor_config = services_config.get('memory_monitor', {})
        self.memory_monitor_enabled = memory_monitor_config.get('enabled', True)
        self.memory_monitor_script = self.base_path.parent / 'scripts' / 'memory_monitor.sh'
        self.memory_monitor_log = self.base_path / 'logs' / 'memory_log.csv'

        # Process tracking (legacy - kept for compatibility but not used for screen sessions)
        self.processes = {}
        self.should_stop = False

        # Screen session names for each service
        self.screen_names = {
            'backend': 'backend_api',
            'llm_server': 'llm_server',
            'audio_server': 'audio_server',
            'embedding_server': 'embedding_server',
            'system_monitoring': 'system_monitoring',
            'memory_monitor': 'memory_monitor',
        }

    async def start_services(self):
        """Start all configured services"""
        # Start memory monitor first (lightweight, always useful)
        if self.memory_monitor_enabled:
            await self._start_memory_monitor()

        # Start embedding server (backend depends on it)
        if self.embedding_server_enabled:
            await self._start_embedding_server()

        # Start backend API (other services may depend on it)
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
            f"MemoryMonitor={self.memory_monitor_enabled}, Embedding={self.embedding_server_enabled}, "
            f"Backend={self.backend_enabled}, Audio={self.audio_server_enabled}, "
            f"LLM={self.llm_enabled}, ModelServers={self.model_servers_enabled}, Dashboards={self.dashboards_enabled}"
        )

    async def stop_services(self):
        """Stop all managed services running in screen sessions"""
        self.should_stop = True

        # Stop all screen-based services
        for service_name, screen_name in self.screen_names.items():
            await self._stop_screen_service(screen_name, service_name)

        # Stop any model server screens
        if self.model_servers_enabled:
            for instance_name in self.model_server_instances.keys():
                screen_name = f'model_server_{instance_name}'
                await self._stop_screen_service(screen_name, f'Model server {instance_name}')

        # Legacy: stop any subprocess-based services that might still be running
        for service_name, process in self.processes.items():
            if process and process.returncode is None:
                self.logger.info(f"Stopping legacy process {service_name}...")
                try:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    self.logger.info(f"Stopped legacy process {service_name}")
                except Exception as e:
                    self.logger.error(f"Error stopping {service_name}: {e}")

        self.processes.clear()

    async def _start_in_screen(
        self,
        screen_name: str,
        cmd: List[str],
        port: int,
        service_description: str,
        log_file: Optional[str] = None
    ) -> bool:
        """Start a service in a named screen session.

        Args:
            screen_name: Name for the screen session
            cmd: Command to run (list of strings)
            port: Port the service listens on (for health checking)
            service_description: Human-readable description for logging
            log_file: Optional log file path (relative to base_path)

        Returns:
            True if service started successfully
        """
        try:
            # Check if screen session already exists
            if await check_screen_exists(screen_name):
                # Screen exists - check if service is actually running on port
                if await self._check_port_in_use(port):
                    self.logger.info(f"{service_description} already running in screen '{screen_name}' on port {port}")
                    return True
                else:
                    # Screen exists but port not in use - kill stale screen
                    self.logger.warning(f"Found stale screen session '{screen_name}', killing it")
                    await kill_screen_session(screen_name)
                    await asyncio.sleep(1)

            # Check if port is in use by something else (not our screen)
            if await self._check_port_in_use(port):
                self.logger.warning(f"{service_description} port {port} already in use by another process")
                return True  # Consider it running

            self.logger.info(f"Starting {service_description} in screen '{screen_name}' on port {port}...")

            # Build the command string
            cmd_str = ' '.join(shlex.quote(str(arg)) for arg in cmd)

            # Add logging if log file specified
            if log_file:
                log_path = self.base_path / log_file
                log_path.parent.mkdir(parents=True, exist_ok=True)
                cmd_str = f'{cmd_str} 2>&1 | tee -a {shlex.quote(str(log_path))}'

            # Wrap command to run in project directory
            wrapped_cmd = f'cd {shlex.quote(str(self.base_path))} && {cmd_str}'

            # Start screen session
            screen_cmd = ['screen', '-dmS', screen_name, 'bash', '-c', wrapped_cmd]

            process = await asyncio.create_subprocess_exec(
                *screen_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()

            # Wait for service to start and verify
            await asyncio.sleep(3)

            if await self._check_port_in_use(port):
                self.logger.info(f"{service_description} started successfully in screen '{screen_name}' on port {port}")
                return True
            else:
                self.logger.error(f"{service_description} failed to start - port {port} not listening")
                return False

        except Exception as e:
            self.logger.error(f"Error starting {service_description}: {e}", exc_info=True)
            return False

    async def _stop_screen_service(self, screen_name: str, service_description: str) -> bool:
        """Stop a service running in a screen session."""
        try:
            if await check_screen_exists(screen_name):
                self.logger.info(f"Stopping {service_description} (screen: {screen_name})...")
                await kill_screen_session(screen_name)
                await asyncio.sleep(1)
                self.logger.info(f"Stopped {service_description}")
                return True
            else:
                self.logger.info(f"{service_description} not running (no screen session found)")
                return True
        except Exception as e:
            self.logger.error(f"Error stopping {service_description}: {e}")
            return False

    async def _start_memory_monitor(self):
        """Start the memory monitor in a screen session (no port check needed)"""
        screen_name = self.screen_names['memory_monitor']

        # Check if script exists
        if not self.memory_monitor_script.exists():
            self.logger.warning(f"Memory monitor script not found: {self.memory_monitor_script}")
            return

        # Check if already running
        if await check_screen_exists(screen_name):
            self.logger.info(f"Memory monitor already running in screen '{screen_name}'")
            return

        self.logger.info(f"Starting memory monitor in screen '{screen_name}'...")

        # Ensure log directory exists
        self.memory_monitor_log.parent.mkdir(parents=True, exist_ok=True)

        # Build command - memory_monitor.sh takes log file as first argument
        cmd_str = f'{shlex.quote(str(self.memory_monitor_script))} {shlex.quote(str(self.memory_monitor_log))}'

        # Start screen session
        screen_cmd = ['screen', '-dmS', screen_name, 'bash', '-c', cmd_str]

        process = await asyncio.create_subprocess_exec(
            *screen_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await process.communicate()

        # Brief wait and verify screen started
        await asyncio.sleep(1)
        if await check_screen_exists(screen_name):
            self.logger.info(f"Memory monitor started successfully in screen '{screen_name}'")
        else:
            self.logger.error(f"Memory monitor failed to start")

    async def _start_embedding_server(self):
        """Start the Embedding server in a screen session"""
        # Check if script exists
        if not self.embedding_server_script.exists():
            self.logger.warning(f"Embedding server script not found: {self.embedding_server_script}")
            return

        screen_name = self.screen_names['embedding_server']
        cmd = [
            str(self.uv_path), 'run', 'python',
            str(self.embedding_server_script),
            '--port', str(self.embedding_server_port),
            '--host', '0.0.0.0'
        ]

        await self._start_in_screen(
            screen_name=screen_name,
            cmd=cmd,
            port=self.embedding_server_port,
            service_description='Embedding server',
            log_file='logs/services/embedding_server.log'
        )

    async def _start_llm_server(self):
        """Start the LLM server in a screen session"""
        screen_name = self.screen_names['llm_server']
        cmd = [
            str(self.uv_path), 'run', 'python',
            str(self.llm_script),
            '--port', str(self.llm_port)
        ]

        await self._start_in_screen(
            screen_name=screen_name,
            cmd=cmd,
            port=self.llm_port,
            service_description='LLM server',
            log_file='logs/services/llm_server.log'
        )

    async def _start_backend(self):
        """Start the Backend API server in a screen session"""
        screen_name = self.screen_names['backend']
        cmd = [
            str(self.uv_path), 'run', 'python',
            '-m', 'uvicorn',
            self.backend_module,
            '--host', '0.0.0.0',
            '--port', str(self.backend_port)
        ]

        await self._start_in_screen(
            screen_name=screen_name,
            cmd=cmd,
            port=self.backend_port,
            service_description='Backend API',
            log_file='logs/services/backend_api.log'
        )

    async def _start_audio_server(self):
        """Start the Audio server in a screen session"""
        # Check if script exists
        if not self.audio_server_script.exists():
            self.logger.warning(f"Audio server script not found: {self.audio_server_script}")
            return

        screen_name = self.screen_names['audio_server']
        cmd = [
            str(self.uv_path), 'run', 'python',
            str(self.audio_server_script)
        ]

        await self._start_in_screen(
            screen_name=screen_name,
            cmd=cmd,
            port=self.audio_server_port,
            service_description='Audio server',
            log_file='logs/services/audio_server.log'
        )

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
        """Start a specific Streamlit dashboard in a screen session"""
        full_script_path = self.base_path / script_path

        if not full_script_path.exists():
            self.logger.warning(f"Dashboard script not found: {full_script_path}")
            return

        screen_name = self.screen_names.get(name, f'dashboard_{name}')
        cmd = [
            'streamlit', 'run',
            str(full_script_path),
            '--server.port', str(port),
            '--server.headless', 'true',
            '--browser.gatherUsageStats', 'false'
        ]

        await self._start_in_screen(
            screen_name=screen_name,
            cmd=cmd,
            port=port,
            service_description=f'{name} dashboard',
            log_file=f'logs/services/{name}_dashboard.log'
        )

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
        """Start a local model server instance in a screen session"""
        screen_name = f'model_server_{instance_name}'
        cmd = [
            str(self.uv_path), 'run', 'python',
            str(self.model_server_script),
            '--port', str(port),
            '--host', '0.0.0.0'
        ]

        await self._start_in_screen(
            screen_name=screen_name,
            cmd=cmd,
            port=port,
            service_description=f'Model server {instance_name}',
            log_file=f'logs/services/model_server_{instance_name}.log'
        )

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

    def _check_port_in_use_sync(self, port: int) -> bool:
        """Check if a port is already in use (synchronous version)"""
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

    async def _check_port_in_use(self, port: int) -> bool:
        """Check if a port is already in use (async wrapper)"""
        return self._check_port_in_use_sync(port)

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

        # Check screen session status by port (using sync version for non-async method)
        if self._check_port_in_use_sync(self.backend_port):
            status['backend']['running'] = True
        if self._check_port_in_use_sync(self.audio_server_port):
            status['audio_server']['running'] = True
        if self._check_port_in_use_sync(self.llm_port):
            status['llm_server']['running'] = True
        if self._check_port_in_use_sync(self.system_dashboard_port):
            status['dashboards']['system_monitoring']['running'] = True

        # Check model server instances
        for instance_name, instance_config in self.model_server_instances.items():
            port = instance_config.get('port', 8004)
            host = instance_config.get('host', 'localhost')
            if host in ('localhost', '10.0.0.4'):  # Local only
                if self._check_port_in_use_sync(port):
                    if instance_name in status['model_servers']['instances']:
                        status['model_servers']['instances'][instance_name]['running'] = True

        return status

    async def restart_service(self, service_name: str) -> bool:
        """Restart a specific service running in a screen session"""
        try:
            # Stop the screen session if it exists
            screen_name = self.screen_names.get(service_name)
            if screen_name:
                await self._stop_screen_service(screen_name, service_name)
                await asyncio.sleep(1)

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