"""
CLI Executor - Runs shell commands via asyncio subprocess
"""
import asyncio
import os
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from pathlib import Path

from .base import BaseExecutor, ExecutionResult

logger = logging.getLogger(__name__)


class CLIExecutor(BaseExecutor):
    """
    Executes CLI commands using asyncio subprocess.

    Config options:
    - command: The command to run (e.g., "python")
    - args: List of arguments
    - cwd: Working directory (optional)
    - env: Additional environment variables (optional)
    """

    def __init__(self, default_cwd: Optional[str] = None, default_python: Optional[str] = None):
        self.default_cwd = default_cwd or "/Users/signal4/signal4/core"
        self.default_python = default_python or "/opt/homebrew/Caskroom/miniforge/base/envs/content-processing/bin/python"

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

        # Use default python path if command is just "python"
        if command == 'python':
            command = self.default_python

        # Build full command
        full_cmd = [command] + args

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

            # Capture output silently
            async def stream_output(stream, lines_list):
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

            return ExecutionResult(
                success=success,
                start_time=start_time,
                end_time=end_time,
                output={
                    'stdout': '\n'.join(stdout_lines),
                    'stderr': '\n'.join(stderr_lines),
                    'returncode': returncode
                },
                error=None if success else f"Command exited with code {returncode}"
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

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate CLI executor configuration"""
        if 'command' not in config and 'args' not in config:
            return False

        # args should be a list if present
        if 'args' in config and not isinstance(config['args'], list):
            return False

        return True
