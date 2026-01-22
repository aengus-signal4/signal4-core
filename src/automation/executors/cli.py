"""
CLI Executor - Runs shell commands via asyncio subprocess or screen sessions

Supports remote execution via SSH when 'host' is specified in config.
"""
import asyncio
import os
import logging
import shlex
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from pathlib import Path

from .base import BaseExecutor, ExecutionResult

logger = logging.getLogger(__name__)

# SSH configuration defaults
DEFAULT_SSH_USER = "signal4"
DEFAULT_SSH_KEY = "/Users/signal4/.ssh/id_ed25519"
LOCAL_HOSTS = ('localhost', '127.0.0.1', '10.0.0.4')  # Head node IPs


async def check_screen_exists(screen_name: str, host: Optional[str] = None) -> bool:
    """
    Check if a screen session with the given name exists and is running.

    Args:
        screen_name: Name of the screen session to check
        host: Remote host IP (None or local IP for local execution)

    Returns True if screen session exists (regardless of state), False otherwise.
    """
    try:
        is_remote = host and host not in LOCAL_HOSTS

        if is_remote:
            # Check on remote host via SSH
            ssh_cmd = [
                'ssh', '-o', 'StrictHostKeyChecking=no',
                '-o', 'ConnectTimeout=5',
                '-i', DEFAULT_SSH_KEY,
                f'{DEFAULT_SSH_USER}@{host}',
                'screen', '-ls'
            ]
            process = await asyncio.create_subprocess_exec(
                *ssh_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
        else:
            process = await asyncio.create_subprocess_exec(
                'screen', '-ls',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

        stdout, _ = await process.communicate()
        output = stdout.decode('utf-8', errors='replace')

        # screen -ls output format: "12345.screen_name\t(Detached)" or "(Attached)"
        # Look for the screen name in the output
        for line in output.split('\n'):
            if f'.{screen_name}' in line or f'\t{screen_name}\t' in line:
                logger.debug(f"Found existing screen session: {line.strip()}" + (f" on {host}" if is_remote else ""))
                return True

        return False
    except Exception as e:
        logger.warning(f"Error checking screen sessions{' on ' + host if host else ''}: {e}")
        # On error, assume no session exists to avoid blocking
        return False


async def check_remote_file_exists(host: str, filepath: str) -> bool:
    """Check if a file exists on a remote host."""
    try:
        ssh_cmd = [
            'ssh', '-o', 'StrictHostKeyChecking=no',
            '-o', 'ConnectTimeout=5',
            '-i', DEFAULT_SSH_KEY,
            f'{DEFAULT_SSH_USER}@{host}',
            f'test -f {shlex.quote(filepath)} && echo "exists"'
        ]
        process = await asyncio.create_subprocess_exec(
            *ssh_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await process.communicate()
        return 'exists' in stdout.decode('utf-8', errors='replace')
    except Exception as e:
        logger.warning(f"Error checking remote file {filepath} on {host}: {e}")
        return False


async def read_remote_file(host: str, filepath: str) -> Optional[str]:
    """Read contents of a file on a remote host."""
    try:
        ssh_cmd = [
            'ssh', '-o', 'StrictHostKeyChecking=no',
            '-o', 'ConnectTimeout=5',
            '-i', DEFAULT_SSH_KEY,
            f'{DEFAULT_SSH_USER}@{host}',
            f'cat {shlex.quote(filepath)}'
        ]
        process = await asyncio.create_subprocess_exec(
            *ssh_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, _ = await process.communicate()
        if process.returncode == 0:
            return stdout.decode('utf-8', errors='replace').strip()
        return None
    except Exception as e:
        logger.warning(f"Error reading remote file {filepath} on {host}: {e}")
        return None


async def remove_remote_file(host: str, filepath: str) -> bool:
    """Remove a file on a remote host."""
    try:
        ssh_cmd = [
            'ssh', '-o', 'StrictHostKeyChecking=no',
            '-o', 'ConnectTimeout=5',
            '-i', DEFAULT_SSH_KEY,
            f'{DEFAULT_SSH_USER}@{host}',
            f'rm -f {shlex.quote(filepath)}'
        ]
        process = await asyncio.create_subprocess_exec(
            *ssh_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await process.communicate()
        return process.returncode == 0
    except Exception as e:
        logger.warning(f"Error removing remote file {filepath} on {host}: {e}")
        return False


class CLIExecutor(BaseExecutor):
    """
    Executes CLI commands using asyncio subprocess or screen sessions.

    Config options:
    - command: The command to run (e.g., "python")
    - args: List of arguments
    - cwd: Working directory (optional)
    - env: Additional environment variables (optional)
    - use_screen: Run in a named screen session (optional, default False)
    - screen_name: Name for the screen session (optional, defaults to task_id)
    - host: Remote host IP for SSH execution (optional, default None for local)
    """

    def __init__(self, default_cwd: Optional[str] = None, default_uv_path: Optional[str] = None):
        self.default_cwd = default_cwd or "/Users/signal4/signal4/core"
        self.default_uv_path = default_uv_path or "/Users/signal4/.local/bin/uv"

    async def execute(
        self,
        config: Dict[str, Any],
        context: Dict[str, Any],
        timeout_seconds: int = 3600
    ) -> ExecutionResult:
        """Execute a CLI command (locally or on a remote host via SSH)"""
        start_time = datetime.now(timezone.utc)

        command = config.get('command', 'python')
        args = config.get('args', [])
        cwd = config.get('cwd', self.default_cwd)
        env_additions = config.get('env', {})
        use_screen = config.get('use_screen', False)
        screen_name = config.get('screen_name') or context.get('task_id', 'scheduled_task')
        host = config.get('host')  # Remote host IP (None for local execution)

        # Determine if this is a remote execution
        is_remote = host and host not in LOCAL_HOSTS

        # Use uv run for python commands to ensure correct environment
        if command == 'python':
            full_cmd = [self.default_uv_path, 'run', 'python'] + args
        else:
            full_cmd = [command] + args

        # If use_screen is enabled, run in a screen session
        if use_screen:
            return await self._execute_in_screen(
                full_cmd, screen_name, cwd, env_additions, timeout_seconds, start_time, host
            )

        # Prepare environment
        env = os.environ.copy()
        env.update(env_additions)

        stdout_lines = []
        stderr_lines = []

        try:
            process = await asyncio.create_subprocess_exec(
                *full_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env
            )

            # Capture output (stored for API access)
            async def stream_output(stream, lines_list, is_stderr=False):
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    decoded = line.decode('utf-8', errors='replace').rstrip()
                    lines_list.append(decoded)

            # Run both streams concurrently with timeout
            try:
                await asyncio.wait_for(
                    asyncio.gather(
                        stream_output(process.stdout, stdout_lines),
                        stream_output(process.stderr, stderr_lines)
                    ),
                    timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                end_time = datetime.now(timezone.utc)
                return ExecutionResult(
                    success=False,
                    start_time=start_time,
                    end_time=end_time,
                    output={
                        'stdout': '\n'.join(stdout_lines),
                        'stderr': '\n'.join(stderr_lines),
                        'returncode': -1
                    },
                    error=f"Command timed out after {timeout_seconds} seconds"
                )

            returncode = await process.wait()
            end_time = datetime.now(timezone.utc)

            success = returncode == 0
            if not success:
                logger.error(f"Command failed with return code {returncode}")

            # Build error message including stderr details for failures
            error_msg = None
            if not success:
                # Get last 10 lines of stderr for error context
                stderr_tail = stderr_lines[-10:] if stderr_lines else []
                stderr_snippet = '\n'.join(stderr_tail)
                if stderr_snippet:
                    error_msg = f"Command exited with code {returncode}. stderr:\n{stderr_snippet}"
                else:
                    # No stderr - check stdout for error info
                    stdout_tail = stdout_lines[-10:] if stdout_lines else []
                    stdout_snippet = '\n'.join(stdout_tail)
                    if stdout_snippet:
                        error_msg = f"Command exited with code {returncode}. stdout tail:\n{stdout_snippet}"
                    else:
                        error_msg = f"Command exited with code {returncode}"

            return ExecutionResult(
                success=success,
                start_time=start_time,
                end_time=end_time,
                output={
                    'stdout': '\n'.join(stdout_lines),
                    'stderr': '\n'.join(stderr_lines),
                    'returncode': returncode
                },
                error=error_msg
            )

        except FileNotFoundError as e:
            end_time = datetime.now(timezone.utc)
            error_msg = f"Command not found: {command}"
            logger.error(error_msg)
            return ExecutionResult(
                success=False,
                start_time=start_time,
                end_time=end_time,
                error=error_msg
            )

        except Exception as e:
            end_time = datetime.now(timezone.utc)
            error_msg = f"Error executing command: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return ExecutionResult(
                success=False,
                start_time=start_time,
                end_time=end_time,
                output={
                    'stdout': '\n'.join(stdout_lines),
                    'stderr': '\n'.join(stderr_lines)
                },
                error=error_msg
            )

    async def _execute_in_screen(
        self,
        full_cmd: list,
        screen_name: str,
        cwd: str,
        env_additions: Dict[str, Any],
        timeout_seconds: int,
        start_time: datetime,
        host: Optional[str] = None
    ) -> ExecutionResult:
        """Execute command in a screen session and monitor until completion.

        Supports both local and remote (SSH) execution.

        SAFEGUARD: Checks if a screen session with the same name already exists.
        If it does, returns immediately with an error instead of killing the
        existing session. This prevents duplicate task execution when the
        orchestrator restarts while a long-running task is still processing.

        Args:
            full_cmd: Command and arguments to execute
            screen_name: Name for the screen session
            cwd: Working directory
            env_additions: Additional environment variables
            timeout_seconds: Maximum time to wait for completion
            start_time: When execution started
            host: Remote host IP (None for local execution)
        """
        is_remote = host and host not in LOCAL_HOSTS
        host_desc = f" on {host}" if is_remote else ""

        # Check if screen session already exists - FAIL FAST to prevent duplicates
        if await check_screen_exists(screen_name, host):
            end_time = datetime.now(timezone.utc)
            return ExecutionResult(
                success=False,
                start_time=start_time,
                end_time=end_time,
                output={
                    'screen_name': screen_name,
                    'returncode': -2,
                    'skipped_already_running': True,
                    'message': f'Skipped - screen session already exists{host_desc}'
                },
                error=None  # Not an error, just skipped
            )

        # Build the command string for screen
        cmd_str = ' '.join(shlex.quote(arg) for arg in full_cmd)

        # Create a marker file to detect completion
        marker_file = f"/tmp/screen_done_{screen_name}"
        exit_code_file = f"/tmp/screen_exit_{screen_name}"

        # Remove old marker files
        if is_remote:
            await remove_remote_file(host, marker_file)
            await remove_remote_file(host, exit_code_file)
        else:
            for f in [marker_file, exit_code_file]:
                if os.path.exists(f):
                    os.remove(f)

        # Wrap command to write exit code and marker when done
        # Disable MallocStackLogging to suppress noisy warnings
        wrapped_cmd = f'cd {shlex.quote(cwd)} && unset MallocStackLogging && unset MallocStackLoggingNoCompact && {cmd_str}; echo $? > {exit_code_file}; touch {marker_file}'

        logger.info(f"Starting screen session '{screen_name}'{host_desc}: {cmd_str}")

        try:
            if is_remote:
                # Execute via SSH on remote host
                # Build the full screen command as a single string for SSH
                screen_cmd_str = f"screen -dmS {shlex.quote(screen_name)} bash -c {shlex.quote(wrapped_cmd)}"

                ssh_cmd = [
                    'ssh', '-o', 'StrictHostKeyChecking=no',
                    '-o', 'ConnectTimeout=10',
                    '-i', DEFAULT_SSH_KEY,
                    f'{DEFAULT_SSH_USER}@{host}',
                    screen_cmd_str
                ]

                process = await asyncio.create_subprocess_exec(
                    *ssh_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()

                if process.returncode != 0:
                    end_time = datetime.now(timezone.utc)
                    error_msg = f"SSH command failed: {stderr.decode('utf-8', errors='replace')}"
                    logger.error(error_msg)
                    return ExecutionResult(
                        success=False,
                        start_time=start_time,
                        end_time=end_time,
                        error=error_msg
                    )
            else:
                # Local execution
                screen_cmd = ['screen', '-dmS', screen_name, 'bash', '-c', wrapped_cmd]

                env = os.environ.copy()
                env.update(env_additions)

                process = await asyncio.create_subprocess_exec(
                    *screen_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env
                )
                await process.wait()

            # Wait for completion by polling marker file
            poll_interval = 5  # seconds
            elapsed = 0

            while elapsed < timeout_seconds:
                # Check if marker file exists (local or remote)
                if is_remote:
                    marker_exists = await check_remote_file_exists(host, marker_file)
                else:
                    marker_exists = os.path.exists(marker_file)

                if marker_exists:
                    # Command completed - read exit code
                    exit_code = 0

                    if is_remote:
                        exit_code_str = await read_remote_file(host, exit_code_file)
                        if exit_code_str:
                            try:
                                exit_code = int(exit_code_str)
                            except ValueError:
                                pass
                        # Clean up remote marker files
                        await remove_remote_file(host, marker_file)
                        await remove_remote_file(host, exit_code_file)
                    else:
                        if os.path.exists(exit_code_file):
                            try:
                                with open(exit_code_file, 'r') as f:
                                    exit_code = int(f.read().strip())
                            except (ValueError, IOError):
                                pass
                        # Clean up local marker files
                        for f in [marker_file, exit_code_file]:
                            if os.path.exists(f):
                                os.remove(f)

                    end_time = datetime.now(timezone.utc)
                    success = exit_code == 0

                    return ExecutionResult(
                        success=success,
                        start_time=start_time,
                        end_time=end_time,
                        output={
                            'screen_name': screen_name,
                            'returncode': exit_code,
                            'host': host if is_remote else 'localhost',
                            'message': f"Command completed in screen '{screen_name}'{host_desc}"
                        },
                        error=f"Command exited with code {exit_code}" if not success else None
                    )

                await asyncio.sleep(poll_interval)
                elapsed += poll_interval

            # Timeout - but leave screen running
            end_time = datetime.now(timezone.utc)
            attach_hint = f"ssh {DEFAULT_SSH_USER}@{host} 'screen -r {screen_name}'" if is_remote else f"screen -r {screen_name}"
            return ExecutionResult(
                success=False,
                start_time=start_time,
                end_time=end_time,
                output={
                    'screen_name': screen_name,
                    'returncode': -1,
                    'host': host if is_remote else 'localhost',
                    'message': f"Screen session '{screen_name}' still running{host_desc} (timed out waiting)"
                },
                error=f"Timed out after {timeout_seconds}s - screen '{screen_name}' still running, attach with: {attach_hint}"
            )

        except Exception as e:
            end_time = datetime.now(timezone.utc)
            error_msg = f"Error starting screen session{host_desc}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return ExecutionResult(
                success=False,
                start_time=start_time,
                end_time=end_time,
                error=error_msg
            )

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate CLI executor configuration"""
        if 'command' not in config and 'args' not in config:
            return False

        # args should be a list if present
        if 'args' in config and not isinstance(config['args'], list):
            return False

        return True
