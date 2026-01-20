"""
CLI Executor - Runs shell commands via asyncio subprocess or screen sessions
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


async def check_screen_exists(screen_name: str) -> bool:
    """
    Check if a screen session with the given name exists and is running.

    Returns True if screen session exists (regardless of state), False otherwise.
    """
    try:
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
                logger.debug(f"Found existing screen session: {line.strip()}")
                return True

        return False
    except Exception as e:
        logger.warning(f"Error checking screen sessions: {e}")
        # On error, assume no session exists to avoid blocking
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
        """Execute a CLI command"""
        start_time = datetime.now(timezone.utc)

        command = config.get('command', 'python')
        args = config.get('args', [])
        cwd = config.get('cwd', self.default_cwd)
        env_additions = config.get('env', {})
        use_screen = config.get('use_screen', False)
        screen_name = config.get('screen_name') or context.get('task_id', 'scheduled_task')

        # Use uv run for python commands to ensure correct environment
        if command == 'python':
            full_cmd = [self.default_uv_path, 'run', 'python'] + args
        else:
            full_cmd = [command] + args

        # If use_screen is enabled, run in a screen session
        if use_screen:
            return await self._execute_in_screen(
                full_cmd, screen_name, cwd, env_additions, timeout_seconds, start_time
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
        start_time: datetime
    ) -> ExecutionResult:
        """Execute command in a screen session and monitor until completion.

        SAFEGUARD: Checks if a screen session with the same name already exists.
        If it does, returns immediately with an error instead of killing the
        existing session. This prevents duplicate task execution when the
        orchestrator restarts while a long-running task is still processing.
        """
        # Check if screen session already exists - FAIL FAST to prevent duplicates
        if await check_screen_exists(screen_name):
            end_time = datetime.now(timezone.utc)
            error_msg = (
                f"Screen session '{screen_name}' already exists. "
                f"Another instance may still be running. "
                f"To force restart, manually kill it: screen -S {screen_name} -X quit"
            )
            logger.warning(error_msg)
            return ExecutionResult(
                success=False,
                start_time=start_time,
                end_time=end_time,
                output={
                    'screen_name': screen_name,
                    'returncode': -2,
                    'message': 'Skipped - screen session already exists'
                },
                error=error_msg
            )

        # Build the command string for screen
        cmd_str = ' '.join(shlex.quote(arg) for arg in full_cmd)

        # Create a marker file to detect completion
        marker_file = f"/tmp/screen_done_{screen_name}"
        exit_code_file = f"/tmp/screen_exit_{screen_name}"

        # Remove old marker files
        for f in [marker_file, exit_code_file]:
            if os.path.exists(f):
                os.remove(f)

        # Wrap command to write exit code and marker when done
        # Disable MallocStackLogging to suppress noisy warnings
        wrapped_cmd = f'cd {shlex.quote(cwd)} && unset MallocStackLogging && unset MallocStackLoggingNoCompact && {cmd_str}; echo $? > {exit_code_file}; touch {marker_file}'

        # Start screen session
        screen_cmd = ['screen', '-dmS', screen_name, 'bash', '-c', wrapped_cmd]

        logger.info(f"Starting screen session '{screen_name}': {cmd_str}")

        env = os.environ.copy()
        env.update(env_additions)

        try:
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
                if os.path.exists(marker_file):
                    # Command completed
                    exit_code = 0
                    if os.path.exists(exit_code_file):
                        try:
                            with open(exit_code_file, 'r') as f:
                                exit_code = int(f.read().strip())
                        except (ValueError, IOError):
                            pass

                    # Clean up marker files
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
                            'message': f"Command completed in screen '{screen_name}'"
                        },
                        error=f"Command exited with code {exit_code}" if not success else None
                    )

                await asyncio.sleep(poll_interval)
                elapsed += poll_interval

            # Timeout - but leave screen running
            end_time = datetime.now(timezone.utc)
            return ExecutionResult(
                success=False,
                start_time=start_time,
                end_time=end_time,
                output={
                    'screen_name': screen_name,
                    'returncode': -1,
                    'message': f"Screen session '{screen_name}' still running (timed out waiting)"
                },
                error=f"Timed out after {timeout_seconds}s - screen '{screen_name}' still running, attach with: screen -r {screen_name}"
            )

        except Exception as e:
            end_time = datetime.now(timezone.utc)
            error_msg = f"Error starting screen session: {str(e)}"
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
