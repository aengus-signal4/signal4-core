import asyncio
import logging
import subprocess
import json
import shlex
import sys
from pathlib import Path
from src.utils.paths import get_project_root, get_config_path
import asyncssh
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import aiohttp
import os
import yaml


from src.utils.logger import setup_worker_logger, log_task_completion
from src.utils.ip_utils import get_reachable_ip, test_connectivity

# Set up logging using our established system
logger = setup_worker_logger('worker_manager')


class SessionManager:
    """Abstract session manager for different backends (tmux, screen, nohup)."""
    
    def __init__(self, session_type: str = "tmux"):
        self.session_type = session_type.lower()
        if self.session_type not in ["tmux", "screen", "nohup"]:
            raise ValueError(f"Unsupported session type: {session_type}")
    
    def kill_session_cmd(self, session_name: str, worker_id: str = "") -> str:
        """Generate command to kill a session."""
        if self.session_type == "tmux":
            return f"PATH=/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin:$PATH tmux kill-session -t {session_name} || true"
        elif self.session_type == "screen":
            return f"screen -S {session_name} -X quit || true"
        elif self.session_type == "nohup":
            pid_file = f"/tmp/{session_name}_{worker_id}.pid"
            return f"[ -f {pid_file} ] && kill $(cat {pid_file}) || true; rm -f {pid_file}"
    
    def start_session_cmd(self, session_name: str, script_path: str, worker_id: str = "") -> str:
        """Generate command to start a session."""
        if self.session_type == "tmux":
            return f"PATH=/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin:$PATH tmux new-session -d -s {session_name} 'bash -c \"{script_path} 2>&1 | tee -a /tmp/{session_name}_tmux.log\"'"
        elif self.session_type == "screen":
            return f"screen -dmS {session_name} bash -c '{script_path} 2>&1 | tee -a /tmp/{session_name}_screen.log'"
        elif self.session_type == "nohup":
            pid_file = f"/tmp/{session_name}_{worker_id}.pid"
            return f"nohup bash -c '{script_path} 2>&1 | tee -a /tmp/{session_name}_nohup.log' > /dev/null & echo $! > {pid_file}"
    
    def list_sessions_cmd(self) -> str:
        """Generate command to list sessions."""
        if self.session_type == "tmux":
            return "PATH=/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin:$PATH tmux ls || true"
        elif self.session_type == "screen":
            return "screen -ls || true"
        elif self.session_type == "nohup":
            return "ps aux | grep -E '(processor|llm_server)' | grep -v grep || true"
    
    def check_session_exists(self, session_name: str, list_output: str, worker_id: str = "") -> bool:
        """Check if session exists in list output."""
        if self.session_type == "tmux":
            return session_name in list_output
        elif self.session_type == "screen":
            return session_name in list_output
        elif self.session_type == "nohup":
            pid_file = f"/tmp/{session_name}_{worker_id}.pid"
            return session_name in list_output and os.path.exists(pid_file)
    
    def get_process_check_cmd(self, python_path: str, script_name: str) -> str:
        """Generate command to check if process is running."""
        return f"ps aux | grep '{python_path}.*{script_name}' | grep -v grep"
    
    def check_availability_cmd(self) -> str:
        """Generate command to check if session manager is available."""
        if self.session_type == "tmux":
            return "PATH=/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin:$PATH which tmux"
        elif self.session_type == "screen":
            return "which screen"
        elif self.session_type == "nohup":
            return "which nohup"  # nohup is usually built-in, but check anyway

class WorkerManager:
    def __init__(self, config: Dict[str, Any], workers: Dict[str, Any]):
        """
        Initialize the worker manager.
        
        Args:
            config: The application configuration dictionary
            workers: Dictionary of WorkerProcess instances to manage
        """
        self.config = config
        self.workers = workers
        
        # SSH settings from config
        self.ssh_username = config.get('processing', {}).get('ssh_username', 'signal4')
        self.ssh_key_path = config.get('processing', {}).get('ssh_key_path', '/Users/signal4/.ssh/id_rsa')
        self.ssh_port = config.get('processing', {}).get('ssh_port', 22)
        
        # Get Python paths from config
        self.head_python_path = config.get('processing', {}).get('head_python_path')
        self.worker_python_path = config.get('processing', {}).get('worker_python_path')
        
        # Get project directory
        self.project_dir = str(get_project_root())
        
        # Get uv settings (replaces conda)
        self.uv_path = config.get('processing', {}).get('uv_path', '/Users/signal4/.local/bin/uv')
        
        # Initialize session manager
        session_type = config.get('processing', {}).get('session_manager', 'tmux')
        self.session_manager = SessionManager(session_type)
        logger.info(f"Using session manager: {session_type}")
        
        if not self.head_python_path or not self.worker_python_path:
            logger.error("Missing Python paths in config")
            raise ValueError("Both head_python_path and worker_python_path must be configured")
        
        logger.info(f"WorkerManager initialized with {len(workers)} workers")

    def _is_local_worker(self, worker: Any) -> bool:
        """Determine if the worker is the head node (local)."""
        return getattr(worker, 'type', None) == 'head'

    def _get_reachable_ip_for_worker(self, worker: Any) -> str:
        """
        Get a reachable IP address for the worker.

        This is the single source of truth for IP discovery logic.
        Tries primary IP first, then falls back to alternative IPs.

        Args:
            worker: The worker object with IP configuration

        Returns:
            The reachable IP address

        Raises:
            ConnectionError: If no reachable IP is found
        """
        worker_id = getattr(worker, 'worker_id', 'unknown')

        # Try primary IP discovery
        reachable_ip = get_reachable_ip(worker_id, self.ssh_port, timeout=3.0)

        if reachable_ip:
            logger.debug(f"Using reachable IP {reachable_ip} for worker {worker_id}")
            return reachable_ip

        # Try alternative IPs as fallback
        fallback_ips = []
        if hasattr(worker, 'current_ip') and worker.current_ip:
            fallback_ips.append(worker.current_ip)
        if hasattr(worker, 'wifi') and worker.wifi:
            fallback_ips.append(worker.wifi)
        if hasattr(worker, 'eth') and worker.eth:
            fallback_ips.append(worker.eth)

        for fallback_ip in fallback_ips:
            if test_connectivity(fallback_ip, self.ssh_port, timeout=3.0):
                logger.warning(f"Using fallback IP {fallback_ip} for worker {worker_id}")
                return fallback_ip

        raise ConnectionError(f"No reachable IP found for worker {worker_id}")

    def _validate_ssh_key(self) -> None:
        """
        Validate that the SSH key exists.

        Raises:
            FileNotFoundError: If the SSH key is not found
        """
        if not Path(self.ssh_key_path).exists():
            raise FileNotFoundError(f"SSH key not found at {self.ssh_key_path}")

    async def _get_ssh_connection(self, worker: Any):
        """
        Get an SSH connection to the worker.

        This centralizes SSH connection logic and error handling.

        Args:
            worker: The worker to connect to

        Returns:
            asyncssh.SSHClientConnection context manager

        Raises:
            FileNotFoundError: If SSH key is missing
            ConnectionError: If worker is unreachable
            asyncssh.Error: If SSH connection fails
        """
        self._validate_ssh_key()
        reachable_ip = self._get_reachable_ip_for_worker(worker)

        return asyncssh.connect(
            reachable_ip,
            username=self.ssh_username,
            client_keys=[self.ssh_key_path],
            port=self.ssh_port,
            known_hosts=None,  # Skip host key verification for now
            connect_timeout=10  # Add connection timeout
        )

    async def _run_command(self, worker: Any, cmd: str):
        """
        Run a shell command locally or via SSH depending on worker type.

        Args:
            worker: The worker to run the command on
            cmd: The shell command to execute

        Returns:
            Command result (subprocess.CompletedProcess for local, asyncssh.SSHCompletedProcess for remote)

        Raises:
            FileNotFoundError: If SSH key is missing (remote only)
            ConnectionError: If worker is unreachable (remote only)
            Exception: If command execution fails
        """
        if self._is_local_worker(worker):
            return await asyncio.to_thread(
                subprocess.run,
                cmd,
                shell=True,
                capture_output=True,
                text=True
            )

        worker_id = getattr(worker, 'worker_id', 'unknown')
        reachable_ip = None
        try:
            reachable_ip = self._get_reachable_ip_for_worker(worker)
            async with await self._get_ssh_connection(worker) as conn:
                return await conn.run(cmd)
        except (FileNotFoundError, ConnectionError):
            # Re-raise these without modification - they have good error messages
            raise
        except Exception as e:
            logger.error(f"SSH command failed to worker {worker_id} at {reachable_ip or 'unknown'}: {str(e)}")
            raise

    async def _write_file(self, worker: Any, path: str, content: str) -> bool:
        """
        Write a file locally or via SSH depending on worker type.

        Args:
            worker: The worker to write the file on
            path: The file path to write to
            content: The content to write

        Returns:
            True if successful, False otherwise

        Note: This method returns bool for backward compatibility with callers
        that check the return value instead of catching exceptions.
        """
        if self._is_local_worker(worker):
            try:
                with open(path, "w") as f:
                    f.write(content)
                os.chmod(path, 0o755)
                return True
            except Exception as e:
                logger.error(f"Failed to write local file {path}: {e}")
                return False

        worker_id = getattr(worker, 'worker_id', 'unknown')
        reachable_ip = None
        try:
            reachable_ip = self._get_reachable_ip_for_worker(worker)
            async with await self._get_ssh_connection(worker) as conn:
                # Quote path to prevent command injection
                safe_path = shlex.quote(path)
                write_cmd = f"cat > {safe_path} << 'EOL'\n{content}\nEOL"
                result = await conn.run(write_cmd)
                if result.returncode != 0:
                    logger.error(f"Write command failed on {worker_id}: exit code {result.returncode}")
                    return False
                chmod_result = await conn.run(f"chmod +x {safe_path}")
                if chmod_result.returncode != 0:
                    logger.warning(f"chmod failed on {worker_id} for {path}, continuing anyway")
                return True
        except FileNotFoundError as e:
            logger.error(f"SSH write file failed to worker {worker_id}: {e}")
            return False
        except ConnectionError as e:
            logger.error(f"SSH write file failed to worker {worker_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"SSH write file failed to worker {worker_id} at {reachable_ip or 'unknown'}: {str(e)}")
            return False

    async def _kill_processor_unified(self, worker: Any) -> bool:
        """Kill any existing task processors on a worker (local or remote)."""
        try:
            worker_id = worker.worker_id if hasattr(worker, 'worker_id') else 'local'
            
            logger.info(f"Cleaning up processor session and sub-processes on worker {worker_id}")
            
            # First, list existing tmux sessions for debugging
            list_result = await self._run_command(worker, "PATH=/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin:$PATH tmux ls 2>/dev/null || echo 'No tmux sessions'")
            list_output = list_result.stdout if hasattr(list_result, 'stdout') else str(list_result)
            logger.info(f"Current tmux sessions on {worker_id}: {list_output.strip()}")
            
            # Kill processor tmux session
            tmux_kill = "PATH=/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin:$PATH tmux kill-session -t processor 2>&1"
            tmux_result = await self._run_command(worker, tmux_kill)
            tmux_output = tmux_result.stderr if hasattr(tmux_result, 'stderr') else ''
            if tmux_output and 'no server running' not in tmux_output and "session not found" not in tmux_output:
                logger.warning(f"tmux kill-session on {worker_id} returned: {tmux_output}")

            # Kill processor screen session
            screen_kill = "screen -S processor -X quit 2>&1"
            screen_result = await self._run_command(worker, screen_kill)
            screen_output = screen_result.stderr if hasattr(screen_result, 'stderr') else ''
            if screen_output and 'No screen session found' not in screen_output:
                logger.warning(f"screen quit on {worker_id} returned: {screen_output}")

            # Kill any existing task_processor.py processes more aggressively
            # Note: pkill returns 1 if no processes matched, which is not an error
            await self._run_command(worker, "pkill -9 -f task_processor.py 2>/dev/null || true")
            await self._run_command(worker, "pkill -9 -f 'python.*task_processor' 2>/dev/null || true")
            await self._run_command(worker, "pkill -9 -f 'start_processor' 2>/dev/null || true")
            
            # Kill all sub-processes that might have been started by processor
            # This includes download, convert, transcribe, diarize, etc.
            logger.info(f"Killing all sub-processes on worker {worker_id}")
            processing_scripts = [
                "download_youtube.py",
                "download_podcast.py", 
                "download_rumble.py",
                "convert.py",
                "transcribe.py",
                "diarize.py",
                "stitch.py",
                "segment_embeddings.py",
                "cleanup_and_compress.py",
                "yt-dlp",
                "ffmpeg",
                "whisper",
                "pyannote"
            ]
            
            for script in processing_scripts:
                kill_cmd = f"pkill -9 -f '{script}' || true"
                await self._run_command(worker, kill_cmd)
            
            # Kill any python processes in the processing_steps directory
            await self._run_command(worker, "pkill -9 -f 'processing_steps' || true")
            
            # Verify cleanup
            result = await self._run_command(worker, self.session_manager.list_sessions_cmd())
            list_output = result.stdout if hasattr(result, 'stdout') else str(result)
            
            # Log what sessions remain
            if list_output and list_output.strip():
                logger.info(f"Remaining sessions after cleanup on {worker_id}:\n{list_output}")
                if self.session_manager.check_session_exists('processor', list_output, worker_id):
                    logger.warning(f"Processor session still exists after cleanup on {worker_id}")
            else:
                logger.info(f"Successfully cleaned up all sessions on {worker_id}")
            
            logger.info(f"Successfully cleaned up existing processors and sub-processes on worker {worker_id}")
            return True
        except Exception as e:
            logger.error(f"Error killing existing processors on worker {worker.worker_id if hasattr(worker, 'worker_id') else 'local'}: {str(e)}", exc_info=True)
            return False

    async def _kill_llm_server_unified(self, worker: Any) -> bool:
        """Kill any existing LLM servers on a worker (local or remote)."""
        try:
            worker_id = worker.worker_id if hasattr(worker, 'worker_id') else 'local'
            
            # Kill session using session manager
            kill_cmd = self.session_manager.kill_session_cmd('llm_server', worker_id)
            await self._run_command(worker, kill_cmd)
            
            # Also try to kill the specific tmux session directly
            tmux_kill = "PATH=/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin:$PATH tmux kill-session -t llm_server 2>/dev/null || true"
            await self._run_command(worker, tmux_kill)
            
            # Kill any existing llm_server.py processes
            await self._run_command(worker, "pkill -9 -f llm_server.py || true")
            
            # Verify cleanup
            result = await self._run_command(worker, self.session_manager.list_sessions_cmd())
            list_output = result.stdout if hasattr(result, 'stdout') else str(result)
            if self.session_manager.check_session_exists('llm_server', list_output, worker_id):
                logger.warning(f"LLM server session still exists after cleanup on {worker_id}")
            else:
                logger.debug(f"Successfully cleaned up LLM server session on {worker_id}")
            
            logger.info(f"Successfully cleaned up existing LLM servers on worker {worker_id}")
            return True
        except Exception as e:
            logger.error(f"Error killing existing LLM servers on worker {worker.worker_id if hasattr(worker, 'worker_id') else 'local'}: {str(e)}", exc_info=True)
            return False

    async def _start_llm_server_unified(self, worker: Any) -> bool:
        """Start the LLM server on head node only."""
        try:
            # Only start LLM server on head node
            if not self._is_local_worker(worker):
                logger.debug(f"Skipping LLM server start on worker {worker.worker_id} (not head node)")
                return True
            
            # Check if LLM server is enabled
            llm_server_config = self.config.get('processing', {}).get('llm_server', {})
            if not llm_server_config.get('enabled', True):  # Default to enabled
                logger.info("LLM server disabled in config")
                return True
            
            logger.info("Starting LLM server on head node")
            
            # Paths and settings
            worker_id = worker.worker_id if hasattr(worker, 'worker_id') else 'head'
            project_root = self.project_dir
            llm_server_path = f"{project_root}/src/services/llm/server.py"
            conda_source = self.conda_source_path
            conda_env = self.conda_env
            log_dir = f"/Users/signal4/logs/content_processing/{worker_id}"
            port = llm_server_config.get('port', 8002)
            
            # Create startup script for LLM server
            startup_script = f"""#!/bin/bash
set -e
set -x

echo "[$(date)] Starting LLM server startup script" >> /tmp/llm_server_startup.log
cd {project_root} || {{
    echo "Failed to change to project directory" >> /tmp/llm_server_startup.log
    exit 1
}}
source {conda_source} || {{
    echo "Failed to source conda" >> /tmp/llm_server_startup.log
    exit 1
}}
conda activate {conda_env} || {{
    echo "Failed to activate conda environment" >> /tmp/llm_server_startup.log
    exit 1
}}
export PYTHONPATH={project_root}:$PYTHONPATH
export PYTHONWARNINGS='ignore::UserWarning:multiprocessing.resource_tracker'
which python >> /tmp/llm_server_startup.log
python --version >> /tmp/llm_server_startup.log
mkdir -p {log_dir}
echo "[$(date)] Starting LLM server on port {port}" >> /tmp/llm_server_startup.log
exec {self.head_python_path} {llm_server_path} 2>&1 | tee -a {log_dir}/llm_server.log /tmp/llm_server_startup.log
"""
            script_path = f"/tmp/start_llm_server_{worker_id}.sh"
            
            # Write the startup script
            ok = await self._write_file(worker, script_path, startup_script)
            if not ok:
                logger.error(f"Failed to write LLM server startup script on {worker_id}")
                return False
            
            # Kill any existing LLM server
            await self._kill_llm_server_unified(worker)
            await asyncio.sleep(2)
            
            # Check session manager availability
            availability_result = await self._run_command(worker, self.session_manager.check_availability_cmd())
            if hasattr(availability_result, 'returncode') and availability_result.returncode != 0:
                logger.error(f"Session manager {self.session_manager.session_type} not available on {worker_id}")
                return False
            
            # Start session for LLM server
            start_cmd = self.session_manager.start_session_cmd('llm_server', script_path, worker_id)
            result = await self._run_command(worker, start_cmd)
            if hasattr(result, 'returncode') and result.returncode != 0:
                logger.error(f"Error starting LLM server on worker {worker_id}: {result.stderr if hasattr(result, 'stderr') else result}")
                return False
            
            # Wait for LLM server to initialize
            await asyncio.sleep(10)
            
            # Verify the LLM server process is running
            ps_cmd = self.session_manager.get_process_check_cmd(self.head_python_path, 'llm_server.py')
            result = await self._run_command(worker, ps_cmd)
            if not (result.stdout if hasattr(result, 'stdout') else str(result)):
                logger.error(f"LLM server process not found after startup on worker {worker_id}")
                return False
            
            # Check if LLM server is responding
            check_url = f"http://localhost:{port}/health"
            max_retries = 6  # Try for up to 30 seconds (6 * 5s)
            for retry in range(max_retries):
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(check_url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                            if response.status == 200:
                                logger.info(f"LLM server on head node is healthy and responding on port {port}")
                                return True
                            else:
                                logger.warning(f"LLM server returned status {response.status}")
                except Exception as e:
                    logger.warning(f"LLM server not yet responding (attempt {retry+1}/{max_retries}): {str(e)}")
                    if retry < max_retries - 1:
                        await asyncio.sleep(5)
            
            # If we get here, the server started but isn't responding yet
            logger.warning(f"LLM server started but not yet responding. May need more time to initialize.")
            return True
            
        except Exception as e:
            logger.error(f"Error starting LLM server on worker {worker.worker_id if hasattr(worker, 'worker_id') else 'local'}: {str(e)}", exc_info=True)
            return False

    async def _start_processor_unified(self, worker: Any) -> bool:
        """Start the task processor directly without complex bash scripts."""
        try:
            logger.info(f"Attempting to start processor on worker {worker.worker_id if hasattr(worker, 'worker_id') else 'local'}")
            
            # Basic settings
            is_head = self._is_local_worker(worker)
            python_path = self.head_python_path if is_head else self.worker_python_path
            worker_id = worker.worker_id if hasattr(worker, 'worker_id') else 'local'
            project_root = self.project_dir
            processor_path = f"{project_root}/src/workers/processor.py"
            log_dir = f"/Users/signal4/logs/content_processing/{worker_id}"
            
            # Create log directory
            mkdir_cmd = f"mkdir -p {log_dir}"
            await self._run_command(worker, mkdir_cmd)
            
            # Build the command to run task_processor.py directly
            # Set environment variables and run Python in one command
            env_vars = f"cd {project_root} && export PYTHONPATH={project_root}:$PYTHONPATH"
            
            # For tmux, we need a simpler approach - just run the Python command directly
            if self.session_manager.session_type == "tmux":
                # Kill any existing session and verify it's dead
                kill_cmd = f"PATH=/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin:$PATH tmux kill-session -t processor 2>/dev/null || true"
                await self._run_command(worker, kill_cmd)

                # Also kill any orphaned processor processes
                await self._run_command(worker, "pkill -9 -f 'task_processor.py' 2>/dev/null || true")
                await asyncio.sleep(2)  # Increased from 1 to 2 seconds

                # Verify session is truly dead before creating new one
                for retry in range(3):
                    check_cmd = f"PATH=/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin:$PATH tmux has-session -t processor 2>/dev/null && echo 'exists' || echo 'dead'"
                    check_result = await self._run_command(worker, check_cmd)
                    check_output = check_result.stdout if hasattr(check_result, 'stdout') else str(check_result)
                    if 'dead' in check_output:
                        break
                    logger.warning(f"Session 'processor' still exists on {worker_id}, retrying kill (attempt {retry+1}/3)")
                    await self._run_command(worker, kill_cmd)
                    await asyncio.sleep(1)

                # Start new tmux session with task_processor.py using uv
                uv_path = self.uv_path
                # Run uv which automatically uses the project's .venv
                tmux_cmd = f"cd {project_root} && export PYTHONPATH={project_root}:$PYTHONPATH && PATH=/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin:$PATH tmux new-session -d -s processor '{uv_path} run --project {project_root} python {processor_path} 2>&1 | tee -a {log_dir}/processor.log'"
                result = await self._run_command(worker, tmux_cmd)

                if hasattr(result, 'returncode') and result.returncode != 0:
                    logger.error(f"Failed to start tmux session: {result.stderr if hasattr(result, 'stderr') else result}")
                    return False

            elif self.session_manager.session_type == "screen":
                # Kill any existing session
                kill_cmd = "screen -S processor -X quit 2>/dev/null || true"
                await self._run_command(worker, kill_cmd)

                # Also kill any orphaned processor processes
                await self._run_command(worker, "pkill -9 -f 'task_processor.py' 2>/dev/null || true")
                await asyncio.sleep(2)  # Increased from 1 to 2 seconds

                # Verify session is truly dead before creating new one
                for retry in range(3):
                    check_cmd = "screen -ls | grep processor && echo 'exists' || echo 'dead'"
                    check_result = await self._run_command(worker, check_cmd)
                    check_output = check_result.stdout if hasattr(check_result, 'stdout') else str(check_result)
                    if 'dead' in check_output:
                        break
                    logger.warning(f"Screen session 'processor' still exists on {worker_id}, retrying kill (attempt {retry+1}/3)")
                    await self._run_command(worker, kill_cmd)
                    await asyncio.sleep(1)

                # Start new screen session using uv
                uv_path = self.uv_path
                # Run uv which automatically uses the project's .venv
                screen_cmd = f"cd {project_root} && export PYTHONPATH={project_root}:$PYTHONPATH && screen -dmS processor {uv_path} run --project {project_root} python {processor_path}"
                result = await self._run_command(worker, screen_cmd)

                if hasattr(result, 'returncode') and result.returncode != 0:
                    logger.error(f"Failed to start screen session: {result.stderr if hasattr(result, 'stderr') else result}")
                    return False

            else:  # nohup
                # Kill any existing process
                await self._run_command(worker, "pkill -9 -f task_processor.py || true")
                await asyncio.sleep(2)  # Increased from 1 to 2 seconds

                # Verify process is dead
                for retry in range(3):
                    check_cmd = f"pgrep -f 'task_processor.py' && echo 'exists' || echo 'dead'"
                    check_result = await self._run_command(worker, check_cmd)
                    check_output = check_result.stdout if hasattr(check_result, 'stdout') else str(check_result)
                    if 'dead' in check_output:
                        break
                    logger.warning(f"Processor still running on {worker_id}, retrying kill (attempt {retry+1}/3)")
                    await self._run_command(worker, "pkill -9 -f task_processor.py || true")
                    await asyncio.sleep(1)

                # Start with nohup using uv
                uv_path = self.uv_path
                nohup_cmd = f"cd {project_root} && export PYTHONPATH={project_root}:$PYTHONPATH && nohup caffeinate {uv_path} run --project {project_root} python {processor_path} > {log_dir}/processor.log 2>&1 &"
                result = await self._run_command(worker, nohup_cmd)

                if hasattr(result, 'returncode') and result.returncode != 0:
                    logger.error(f"Failed to start with nohup: {result.stderr if hasattr(result, 'stderr') else result}")
                    return False
            
            # Wait for process to start
            await asyncio.sleep(3)

            # Verify the process is running (look for uv run with processor.py)
            ps_cmd = "ps aux | grep 'processor.py' | grep -v grep"
            result = await self._run_command(worker, ps_cmd)
            if not (result.stdout if hasattr(result, 'stdout') else str(result)):
                logger.error(f"Processor process not found after startup on worker {worker_id}")
                return False
            
            logger.info(f"Successfully started processor on worker {worker_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting processor on worker {worker.worker_id if hasattr(worker, 'worker_id') else 'local'}: {str(e)}", exc_info=True)
            return False

    async def manage_workers(self) -> None:
        """Manage task processors and LLM server on all enabled workers including head node."""
        logger.info("Starting worker management...")
        logger.debug(f"SSH key path: {self.ssh_key_path}")
        logger.debug(f"Workers: {list(self.workers.keys())}")
        logger.info(f"Using session manager: {self.session_manager.session_type}")
        
        # Get enabled workers from config
        worker_configs = self.config.get('processing', {}).get('workers', {})
        enabled_workers = [
            worker for worker in self.workers.values()
            if worker_configs.get(worker.worker_id, {}).get('enabled', False)
        ]
        if not enabled_workers:
            logger.warning("No enabled workers found to manage")
            return
        logger.info(f"Found {len(enabled_workers)} enabled workers to manage")
        
        for worker in enabled_workers:
            try:
                logger.debug(f"Worker {worker.worker_id} type: {getattr(worker, 'type', 'unknown')}")
                logger.debug(f"Worker {worker.worker_id} IP: {getattr(worker, 'current_ip', 'unknown')}")
                
                # Skip LLM server management - LLM server is managed independently
                logger.info(f"Skipping LLM server management for head node {worker.worker_id} (disabled)")
                
                # Check if task processor is already healthy
                logger.info(f"Checking task processor health for worker {worker.worker_id}")
                is_healthy = await self.check_worker_health(worker.worker_id)
                
                if is_healthy:
                    logger.info(f"Task processor on worker {worker.worker_id} is already healthy, skipping restart")
                    worker.status = 'running'
                    continue
                
                # Task processor is not healthy, restart it
                logger.info(f"Task processor on worker {worker.worker_id} is unhealthy, restarting...")
                kill_success = await self._kill_processor_unified(worker)
                if not kill_success:
                    logger.warning(f"Failed to kill existing processor on worker {worker.worker_id}")
                    continue
                await asyncio.sleep(2)
                start_success = await self._start_processor_unified(worker)
                if not start_success:
                    logger.error(f"Failed to start processor on worker {worker.worker_id}")
                else:
                    logger.info(f"Successfully started processor on worker {worker.worker_id}")
                    worker.status = 'running'
            except Exception as e:
                logger.error(f"Error managing services on worker {worker.worker_id}: {str(e)}", exc_info=True)
                continue

    async def check_llm_server_health(self, worker_id: str) -> Dict[str, Any]:
        """Check health of LLM server on head node."""
        if worker_id not in self.workers:
            return {"healthy": False, "error": "Worker not found"}
        
        worker = self.workers[worker_id]
        
        # Only check LLM server on head node
        if not self._is_local_worker(worker):
            return {"healthy": False, "error": "LLM server only runs on head node"}
        
        # Check if LLM server is enabled
        llm_server_config = self.config.get('processing', {}).get('llm_server', {})
        if not llm_server_config.get('enabled', True):
            return {"healthy": False, "error": "LLM server not enabled"}
        
        port = llm_server_config.get('port', 8002)
        
        try:
            async with aiohttp.ClientSession() as session:
                # Check health endpoint
                health_url = f"http://localhost:{port}/health"
                async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status != 200:
                        return {"healthy": False, "error": f"Health check returned status {response.status}"}
                    health_data = await response.json()
                
                # Check status endpoint for detailed info
                status_url = f"http://localhost:{port}/status"
                async with session.get(status_url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        status_data = await response.json()
                        return {
                            "healthy": True,
                            "health": health_data,
                            "status": status_data,
                            "url": f"http://localhost:{port}"
                        }
                    else:
                        return {
                            "healthy": True,
                            "health": health_data,
                            "url": f"http://localhost:{port}"
                        }
                        
        except asyncio.TimeoutError:
            return {"healthy": False, "error": "LLM server request timed out"}
        except Exception as e:
            return {"healthy": False, "error": str(e)}
    
    async def get_all_llm_servers(self) -> Dict[str, Dict[str, Any]]:
        """Get status of LLM server (head node only)."""
        llm_servers = {}
        
        for worker_id, worker in self.workers.items():
            if worker.status == 'active' and self._is_local_worker(worker):
                health_status = await self.check_llm_server_health(worker_id)
                llm_servers[worker_id] = health_status
                break  # Only one head node
        
        return llm_servers

    async def cancel_task(self, worker_id: str, task_id: str) -> bool:
        """Cancel a task on a worker."""
        if worker_id not in self.workers:
            logger.warning(f"Worker {worker_id} not found for task cancellation")
            return False
            
        worker = self.workers[worker_id]
        try:
            async with aiohttp.ClientSession() as session:
                cancel_url = f"{worker.api_url}/tasks/{task_id}"
                async with session.delete(cancel_url) as response:
                    if response.status == 200:
                        logger.info(f"Successfully cancelled task {task_id} on worker {worker_id}")
                        return True
                    elif response.status == 404:
                        # Task not found is expected in some cases (e.g., after worker restart)
                        logger.debug(f"Task {task_id} not found on worker {worker_id} during cancellation (expected if worker was restarted)")
                        return True  # Consider this a "success" since the task is effectively cancelled
                    else:
                        error_text = await response.text()
                        logger.error(f"Failed to cancel task {task_id} on worker {worker_id}: {response.status} - {error_text}")
                        return False
        except aiohttp.ClientConnectorError:
            # Connection error is expected if worker is down
            logger.debug(f"Could not connect to worker {worker_id} to cancel task {task_id} (expected if worker is down)")
            return True  # Consider this a "success" since we can't do anything about it
        except Exception as e:
            logger.error(f"Error cancelling task {task_id} on worker {worker_id}: {str(e)}")
            return False

    async def check_worker_health(self, worker_id: str) -> bool:
        """Check if a worker is healthy by verifying its API is responding."""
        if worker_id not in self.workers:
            logger.warning(f"Worker {worker_id} not found for health check")
            return False
            
        worker = self.workers[worker_id]
        try:
            async with aiohttp.ClientSession() as session:
                health_url = f"{worker.api_url}/tasks"
                async with session.get(health_url) as response:
                    if response.status == 200:
                        try:
                            health_data = await response.json()
                            if health_data.get("status") == "healthy":
                                logger.debug(f"Worker {worker_id} health check passed")
                                return True
                            else:
                                logger.warning(f"Worker {worker_id} reported unhealthy status")
                                return False
                        except json.JSONDecodeError:
                            logger.error(f"Invalid JSON response from worker {worker_id} health check")
                            return False
                    else:
                        logger.warning(f"Worker {worker_id} health check failed with status {response.status}")
                        return False
        except Exception as e:
            logger.error(f"Error checking worker {worker_id} health: {str(e)}")
            return False

    async def restart_unhealthy_worker(self, worker_id: str) -> bool:
        """Restart a worker that has been identified as unhealthy."""
        if worker_id not in self.workers:
            logger.warning(f"Worker {worker_id} not found for restart")
            return False
            
        worker = self.workers[worker_id]
        logger.info(f"Attempting to restart unhealthy worker {worker_id}")
        
        # Import database dependencies here to avoid circular imports
        from src.database.models import TaskQueue
        from src.database.session import get_session
        
        try:
            # First, reset all tasks assigned to this worker to pending
            logger.info(f"Resetting all tasks assigned to worker {worker_id} to pending status")
            try:
                with get_session() as session:
                    # Find all processing tasks assigned to this worker
                    processing_tasks = session.query(TaskQueue).filter(
                        TaskQueue.worker_id == worker_id,
                        TaskQueue.status == 'processing'
                    ).all()
                    
                    reset_count = 0
                    for task in processing_tasks:
                        # Try to cancel task on worker first (but don't fail if we can't)
                        try:
                            await self.cancel_task(worker_id, str(task.id))
                        except Exception as cancel_e:
                            logger.debug(f"Could not cancel task {task.id} on worker {worker_id} (expected if worker is down): {str(cancel_e)}")
                        
                        # Reset task status to pending
                        task.status = 'pending'
                        task.worker_id = None
                        task.started_at = None
                        task.error = f"Worker {worker_id} restarted"
                        reset_count += 1
                    
                    session.commit()
                    logger.info(f"Reset {reset_count} tasks from worker {worker_id} to pending status")
                    
                    # Clear the worker's tracked tasks in memory
                    if hasattr(worker, 'current_tasks'):
                        worker.current_tasks.clear()
                    if hasattr(worker, 'current_tasks_by_type'):
                        for task_set in worker.current_tasks_by_type.values():
                            task_set.clear()
                    if hasattr(worker, 'api_task_ids'):
                        worker.api_task_ids.clear()
                    logger.debug(f"Cleared in-memory task tracking for worker {worker_id}")
                    
            except Exception as db_e:
                logger.error(f"Error resetting tasks for worker {worker_id}: {str(db_e)}")
                # Continue with restart even if task reset fails
            
            # Kill existing services
            # Skip LLM server management - LLM server is managed independently
            logger.info(f"Skipping LLM server restart for head node {worker_id} (disabled)")
            
            # Kill existing processor and all sub-processes
            kill_success = await self._kill_processor_unified(worker)
            if not kill_success:
                logger.warning(f"Failed to kill existing processor on worker {worker_id}")
                # Continue anyway since the process might already be dead
            
            # Wait a moment for processes to fully terminate
            await asyncio.sleep(2)
            
            # Start services
            # Skip LLM server restart - LLM server is managed independently
            logger.info(f"Skipping LLM server restart for head node {worker_id} (disabled)")
            
            # Start new processor
            start_success = await self._start_processor_unified(worker)
            if not start_success:
                logger.error(f"Failed to restart processor on worker {worker_id}")
                return False
                
            # Verify worker is healthy after restart
            max_retries = 3
            retry_delay = 5  # seconds between retries
            for attempt in range(max_retries):
                logger.info(f"Checking worker {worker_id} health after restart (attempt {attempt + 1}/{max_retries})")
                if await self.check_worker_health(worker_id):
                    logger.info(f"Successfully restarted worker {worker_id}")
                    worker.status = 'running'  # Update worker status on successful restart
                    return True
                if attempt < max_retries - 1:  # Don't sleep on last attempt
                    logger.info(f"Health check failed, waiting {retry_delay}s before retry...")
                    await asyncio.sleep(retry_delay)
                
            logger.error(f"Worker {worker_id} failed health check after restart")
            return False
            
        except Exception as e:
            logger.error(f"Error restarting worker {worker_id}: {str(e)}", exc_info=True)
            return False

async def main():
    """Main entry point for direct execution of worker management."""
    import argparse
    import json
    import yaml
    import asyncio
    from pathlib import Path
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Manage content processing workers')
    parser.add_argument('--worker', type=str, help='Worker ID to manage (e.g., worker0)')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file (default: config/config.yaml)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Load config
    try:
        config_path = Path(args.config)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path) as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            else:
                config = json.load(f)
                
        logger.debug(f"Loaded config from {config_path}")
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        return
    
    # Create a minimal worker configuration for the specified worker
    if args.worker:
        # Get worker config from the loaded config if it exists
        worker_config = config.get('processing', {}).get('workers', {}).get(args.worker, {})
        if not worker_config:
            # Create default config if worker not found in config
            worker_config = {
                'worker_id': args.worker,
                'type': 'head' if args.worker == 'worker0' else 'worker',
                'status': 'active'
            }
        else:
            # Ensure required fields are present
            worker_config['worker_id'] = args.worker
            worker_config['status'] = 'active'
            if 'type' not in worker_config:
                worker_config['type'] = 'head' if args.worker == 'worker0' else 'worker'
            # --- Ensure current_ip is set ---
            worker_config['current_ip'] = worker_config.get('eth') or worker_config.get('wifi') or '127.0.0.1'
        workers = {args.worker: type('Worker', (), worker_config)()}
        logger.debug(f"Created worker configuration: {worker_config}")
    else:
        logger.error("No worker specified. Use --worker to specify a worker ID")
        return
    
    # Create worker manager
    try:
        manager = WorkerManager(config, workers)
        
        # Start the specified worker
        logger.info(f"Starting management for worker: {args.worker}")
        await manager.manage_workers()
        
        # Keep the process running to maintain the worker
        try:
            while True:
                await asyncio.sleep(60)  # Check worker health every minute
                if not await manager.check_worker_health(args.worker):
                    logger.warning(f"Worker {args.worker} is unhealthy, attempting restart...")
                    await manager.restart_unhealthy_worker(args.worker)
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
            # Clean up the worker - skip LLM server cleanup (managed independently)
            worker = workers[args.worker]
            await manager._kill_processor_unified(worker)
            logger.info("Skipped LLM server cleanup (managed independently)")
            
    except Exception as e:
        logger.error(f"Error in worker management: {str(e)}", exc_info=True)

if __name__ == '__main__':
    asyncio.run(main()) 