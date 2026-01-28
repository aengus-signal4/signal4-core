# Centralized environment setup (must be before other imports)
from src.utils.env_setup import setup_env, get_subprocess_env
setup_env()

import os

# Load .env file for credentials (S3_ACCESS_KEY, POSTGRES_PASSWORD, etc.)
from dotenv import load_dotenv
from src.utils.paths import get_env_path
load_dotenv(get_env_path())

# Standard library imports
import asyncio
import contextlib
import json
import math
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from src.utils.paths import get_project_root, get_config_path
from typing import Dict, Any, Optional

# Third-party imports
import aiohttp
import psutil
import yaml
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import QueuePool
from sqlalchemy.sql import text

# Add project root to Python path
sys.path.append(str(get_project_root()))

from src.utils.logger import setup_worker_logger, log_task_completion
from src.utils.node_utils import is_head_node, get_worker_name
from src.utils.ip_utils import get_reachable_ip_async, build_worker_url_async, get_worker_ip
from src.storage.s3_utils import S3StorageConfig, S3Storage
from src.database.session import get_session, Session
from src.database.models import TaskQueue

# --- Define Concurrency Limit and Semaphore ---
MAX_CONCURRENT_SCRIPTS = 5
script_semaphore = asyncio.Semaphore(MAX_CONCURRENT_SCRIPTS)
logger = setup_worker_logger('task_processor') # Define logger before using semaphore log
logger.info(f"Script execution concurrency limited to: {MAX_CONCURRENT_SCRIPTS}")
# --- End Semaphore Definition ---

# --- Task Queue System ---
# Per-task-type queues for batched task processing
task_queues: Dict[str, asyncio.Queue] = {}
# Maximum queue size per task type to prevent memory explosion (backpressure)
MAX_QUEUE_SIZE = 100  # Per task type
# Track queue depths for monitoring
queue_depths: Dict[str, int] = {}
# Track currently running tasks per task type
running_tasks_by_type: Dict[str, int] = {}
# Track task type limits from config
task_type_limits: Dict[str, int] = {}
# Lock for queue operations
queue_lock = asyncio.Lock()
# Flag to start queue consumer
queue_consumer_started = False
logger.info("Task queue system initialized")
# --- End Task Queue System ---

def cleanup_existing_processes():
    """Kill any existing task processor processes and processes on port 8000"""
    import subprocess
    current_pid = os.getpid()
    script_name = Path(__file__).name

    # First, kill any process listening on port 8000
    try:
        result = subprocess.run(
            ['lsof', '-ti', ':8000'],
            capture_output=True,
            text=True
        )
        if result.stdout.strip():
            for pid_str in result.stdout.strip().split('\n'):
                try:
                    pid = int(pid_str)
                    if pid != current_pid:
                        logger.info(f"Killing process {pid} on port 8000...")
                        os.kill(pid, signal.SIGKILL)
                        logger.info(f"Killed process {pid}")
                except (ValueError, ProcessLookupError):
                    pass
                except Exception as e:
                    logger.warning(f"Error killing process on port 8000: {e}")
    except Exception as e:
        logger.warning(f"Error checking port 8000: {e}")

    # Find and kill other instances of this script
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # Skip our own process
            if proc.pid == current_pid:
                continue

            # Check if this is a Python process running our script
            if (proc.info['name'] == 'Python' and
                proc.info['cmdline'] and
                script_name in ' '.join(proc.info['cmdline'])):

                logger.info(f"Found existing task processor process (PID: {proc.pid}), terminating...")
                try:
                    os.kill(proc.pid, signal.SIGKILL)
                    logger.info(f"Terminated process {proc.pid}")
                except ProcessLookupError:
                    logger.info(f"Process {proc.pid} already terminated")
                except Exception as e:
                    logger.error(f"Error killing process {proc.pid}: {str(e)}")
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    # Give processes time to terminate
    time.sleep(1)

# Set up logging
logger = setup_worker_logger('task_processor')

# Kill any existing processor processes on port 8000 before starting
cleanup_existing_processes()

class TaskProcessor:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
            
        # Use the centralized session management instead of creating our own pool
        self.session_manager = Session()
        self.logger = setup_worker_logger('task_processor')

    @contextlib.contextmanager
    def get_db_session(self):
        """Provide a transactional scope around a series of operations.

        Uses exponential backoff with jitter for retries to prevent thundering herd.
        """
        import random
        max_retries = 3
        base_delay = 2  # seconds

        for attempt in range(max_retries):
            session = None
            try:
                session = self.session_manager.get_session()
                # Test the connection before proceeding
                session.execute(text("SELECT 1"))
                yield session
                session.commit()
                return  # Success, exit
            except Exception as e:
                if session:
                    try:
                        session.rollback()
                    except Exception:
                        pass  # Ignore rollback errors
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter: 2-4s, 4-8s, 8-16s
                    delay = base_delay * (2 ** attempt) + random.uniform(0, base_delay)
                    self.logger.warning(f"Database connection attempt {attempt + 1}/{max_retries} failed: {str(e)}. Retrying in {delay:.1f}s...")
                    time.sleep(delay)  # Synchronous sleep is appropriate here (sync context manager)
                    continue
                self.logger.error(f"All database connection attempts failed: {str(e)}")
                raise
            finally:
                if session:
                    try:
                        session.close()
                    except Exception:
                        pass  # Ignore close errors

    def __del__(self):
        """Cleanup when processor is destroyed."""
        pass  # No need to dispose engine as it's managed by the Session class

# Create global TaskProcessor instance
task_processor = TaskProcessor()

# Load config using ConfigManager for hot-reload support
from src.orchestration.config_manager import ConfigManager
config_manager = ConfigManager(config_path="config/config.yaml")
config = config_manager.config

# Log session manager configuration
session_manager = config.get('processing', {}).get('session_manager', 'tmux')
logger.info(f"Task Processor running under session manager: {session_manager}")

# Get hostname for worker identification
import socket
hostname = socket.gethostname()
current_worker_id = None  # Track which worker we are

def reload_task_type_limits(new_config: dict = None):
    """Reload task type limits from config. Called on config change."""
    global current_worker_id
    cfg = new_config if new_config else config_manager.config
    worker_configs = cfg.get('processing', {}).get('workers', {})

    for worker_id, worker_config in worker_configs.items():
        # Match if worker_id is in hostname (e.g., "worker2" matches "s4-worker2")
        if worker_id in hostname:
            current_worker_id = worker_id
            enabled_tasks = worker_config.get('task_types', [])
            new_limits = {}
            for task_spec in enabled_tasks:
                if isinstance(task_spec, str):
                    parts = task_spec.split(':')
                    task_type = parts[0]
                    limit = int(parts[1]) if len(parts) > 1 else 1
                    new_limits[task_type] = limit
                    # Initialize running count if not exists
                    if task_type not in running_tasks_by_type:
                        running_tasks_by_type[task_type] = 0

            # Check if limits changed
            if new_limits != task_type_limits:
                logger.info(f"Task type limits changed for {worker_id}: {task_type_limits} -> {new_limits}")
                task_type_limits.clear()
                task_type_limits.update(new_limits)
            else:
                logger.debug(f"Task type limits unchanged for {worker_id}: {task_type_limits}")
            return

    logger.warning(f"No worker config found matching hostname '{hostname}'")

# Initial load of task type limits
reload_task_type_limits()
logger.info(f"Loaded task type limits for {current_worker_id}: {task_type_limits}")

# Register reload callback
config_manager.register_reload_callback(reload_task_type_limits)

# Cache orchestrator URL for fast callbacks (avoid DNS lookups on every callback)
_cached_orchestrator_url = None
_orchestrator_url_cache_time = None
_orchestrator_url_cache_ttl = 300  # 5 minutes
_orchestrator_url_cache_lock = asyncio.Lock()  # Lock to prevent concurrent cache rebuilds

# Get orchestrator port and callback path from config (with defaults)
_orchestrator_port = config.get('orchestration', {}).get('port', 8001)
_callback_path = config.get('orchestration', {}).get('callback_path', '/api/task_callback')


async def get_orchestrator_url() -> Optional[str]:
    """Get cached orchestrator URL or build a new one if cache expired."""
    global _cached_orchestrator_url, _orchestrator_url_cache_time

    # Fast path: check cache without lock
    if _cached_orchestrator_url and _orchestrator_url_cache_time:
        cache_age = (datetime.now(timezone.utc) - _orchestrator_url_cache_time).total_seconds()
        if cache_age < _orchestrator_url_cache_ttl:
            return _cached_orchestrator_url

    # Slow path: acquire lock and rebuild cache
    async with _orchestrator_url_cache_lock:
        # Double-check cache after acquiring lock (another task may have refreshed it)
        if _cached_orchestrator_url and _orchestrator_url_cache_time:
            cache_age = (datetime.now(timezone.utc) - _orchestrator_url_cache_time).total_seconds()
            if cache_age < _orchestrator_url_cache_ttl:
                return _cached_orchestrator_url

        # Cache miss or expired - rebuild URL
        # Find the head worker
        head_worker_name = None
        for worker_id, worker_config in config.get('processing', {}).get('workers', {}).items():
            if worker_config.get('type') == 'head':
                head_worker_name = worker_id
                break

        if not head_worker_name:
            logger.error("No worker with type 'head' found in config for callback.")
            return None

        # Build URL with automatic eth->wifi fallback
        orchestrator_url = await build_worker_url_async(head_worker_name, _orchestrator_port, _callback_path)

        if orchestrator_url:
            _cached_orchestrator_url = orchestrator_url
            _orchestrator_url_cache_time = datetime.now(timezone.utc)
            logger.info(f"Cached orchestrator URL: {orchestrator_url}")

        return orchestrator_url

async def send_completion_callback(task_id: str, task_data: Dict[str, Any], result: Dict[str, Any],
                                  status: str, duration: float, task_type: str,
                                  max_retries: int = 3) -> bool:
    """
    Send a callback to the orchestrator to notify about task completion.
    Uses cached orchestrator URL for fast callbacks without DNS lookups.
    Implements exponential backoff retry on failure.

    Args:
        task_id: The task identifier
        task_data: Task data dictionary
        result: Task result dictionary
        status: Task status string
        duration: Task duration in seconds
        task_type: Type of task
        max_retries: Maximum number of retry attempts (default 3)

    Returns:
        True if callback was sent successfully, False otherwise
    """
    # Get cached orchestrator URL (fast path)
    orchestrator_url = await get_orchestrator_url()

    if not orchestrator_url:
        logger.error("Could not get orchestrator URL for callback")
        return False

    logger.debug(f"Sending callback for task {task_id} to {orchestrator_url}")

    # Get original task ID or default to empty string
    original_task_id = task_data.get("original_task_id", "")

    # Debug log the task data and original ID
    logger.debug(f"Task data for callback: {task_data}")
    logger.debug(f"Original task ID for callback: '{original_task_id}'")

    # Prepare callback data (remains the same)
    callback_data = {
        "task_id": task_id,
        "original_task_id": original_task_id,
        "content_id": task_data["content_id"],
        "task_type": task_type,
        "status": status,
        "duration": duration,
        "result": result,
        "worker_id": task_data["worker_id"],
        "chunk_index": task_data.get("input_data", {}).get("chunk_index")
    }

    # Attempt callback with exponential backoff retry
    last_error = None
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    orchestrator_url,
                    json=callback_data,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        resp_data = await response.json()
                        if attempt > 0:
                            logger.info(f"Callback for task {task_id} successful after {attempt + 1} attempts")
                        else:
                            logger.info(f"Callback for task {task_id} successful: {resp_data.get('message', 'OK')}")
                        return True
                    else:
                        error_text = await response.text()
                        last_error = f"HTTP {response.status}: {error_text}"
                        logger.warning(f"Callback attempt {attempt + 1}/{max_retries} failed with status {response.status}: {error_text}")

        except (aiohttp.ClientConnectorError, asyncio.TimeoutError) as conn_err:
            last_error = str(conn_err)
            logger.warning(f"Callback attempt {attempt + 1}/{max_retries} connection error for task {task_id}: {conn_err}")
        except Exception as e:
            last_error = str(e)
            logger.warning(f"Callback attempt {attempt + 1}/{max_retries} unexpected error for task {task_id}: {e}")

        # Exponential backoff: 1s, 2s, 4s (but not after last attempt)
        if attempt < max_retries - 1:
            backoff_time = 2 ** attempt
            logger.debug(f"Retrying callback in {backoff_time}s...")
            await asyncio.sleep(backoff_time)

    # All retries exhausted
    logger.error(f"Callback for task {task_id} failed after {max_retries} attempts. Last error: {last_error}")
    return False


async def send_tracked_callback(task_id: str, task_data: Dict[str, Any], result: Dict[str, Any],
                                status: str, duration: float, task_type: str) -> bool:
    """
    Send a callback with tracking for graceful shutdown.
    This wrapper ensures the callback is tracked so shutdown can wait for it.

    Returns:
        True if callback was sent successfully, False otherwise
    """
    callback_id = f"callback_{task_id}"

    # Register callback as pending
    callback_task = asyncio.current_task()
    async with pending_callbacks_lock:
        pending_callbacks[callback_id] = callback_task

    try:
        success = await send_completion_callback(task_id, task_data, result, status, duration, task_type)
        if not success:
            logger.warning(f"Callback failed for task {task_id} (task completed but orchestrator may not know)")
        return success
    finally:
        # Remove from pending callbacks
        async with pending_callbacks_lock:
            if callback_id in pending_callbacks:
                del pending_callbacks[callback_id]


# Initialize FastAPI app
app = FastAPI(
    title="Task Processor API",
    description="API for processing content analysis tasks",
    version="1.0.0"
)

# Store task information
tasks: Dict[str, Dict[str, Any]] = {}
tasks_lock = asyncio.Lock()  # Lock for thread-safe access to tasks dictionary
task_processes: Dict[str, asyncio.subprocess.Process] = {}  # Track subprocess objects
task_processes_lock = asyncio.Lock()  # Lock for thread-safe access to task_processes

# Background task tracking
background_tasks: Dict[str, asyncio.Task] = {}  # Track background tasks by name
background_tasks_lock = asyncio.Lock()  # Lock for thread-safe access

# Pending callbacks tracking for graceful shutdown
pending_callbacks: Dict[str, asyncio.Task] = {}  # Track in-flight callback tasks
pending_callbacks_lock = asyncio.Lock()  # Lock for thread-safe access

# Graceful shutdown state
shutdown_requested = False
shutdown_event = asyncio.Event()


async def tracked_create_task(
    coro,
    name: str,
    timeout: Optional[float] = None,
    on_error: Optional[str] = "log"  # "log", "raise", or "ignore"
) -> asyncio.Task:
    """
    Create a tracked background task with optional timeout and error handling.

    Args:
        coro: The coroutine to run
        name: A descriptive name for the task (for logging)
        timeout: Optional timeout in seconds. None means no timeout.
        on_error: How to handle exceptions - "log" (default), "raise", or "ignore"

    Returns:
        The created asyncio.Task
    """
    async def wrapper():
        task_id = f"{name}_{id(coro)}"
        try:
            if timeout:
                result = await asyncio.wait_for(coro, timeout=timeout)
            else:
                result = await coro
            return result
        except asyncio.TimeoutError:
            logger.error(f"Background task '{name}' timed out after {timeout}s")
            if on_error == "raise":
                raise
        except asyncio.CancelledError:
            logger.info(f"Background task '{name}' was cancelled")
            raise  # Always re-raise CancelledError
        except Exception as e:
            if on_error == "log":
                logger.error(f"Background task '{name}' failed: {e}", exc_info=True)
            elif on_error == "raise":
                raise
            # "ignore" does nothing
        finally:
            # Remove from tracking
            async with background_tasks_lock:
                if task_id in background_tasks:
                    del background_tasks[task_id]

    task = asyncio.create_task(wrapper())
    task_id = f"{name}_{id(coro)}"

    # Track the task
    async with background_tasks_lock:
        background_tasks[task_id] = task

    return task


def handle_shutdown_signal(signum, frame):
    """Handle SIGTERM/SIGINT for graceful shutdown."""
    global shutdown_requested
    signal_name = signal.Signals(signum).name
    logger.info(f"Received {signal_name} signal, initiating graceful shutdown...")
    shutdown_requested = True
    # Set the event to wake up any waiting coroutines
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.call_soon_threadsafe(shutdown_event.set)
    except RuntimeError:
        pass  # No event loop running


async def graceful_shutdown():
    """Perform graceful shutdown - wait for in-flight tasks and callbacks to complete."""
    global shutdown_requested
    shutdown_requested = True
    logger.info("Starting graceful shutdown...")

    # Wait for in-flight tasks to complete (with timeout)
    max_wait = 30  # seconds
    start_time = time.time()

    while time.time() - start_time < max_wait:
        async with task_processes_lock:
            active_count = len(task_processes)
        if active_count == 0:
            logger.info("All tasks completed, proceeding with shutdown")
            break
        logger.info(f"Waiting for {active_count} in-flight tasks to complete...")
        await asyncio.sleep(2)

    # Force terminate any remaining processes
    async with task_processes_lock:
        if task_processes:
            logger.warning(f"Force terminating {len(task_processes)} remaining tasks")
            for task_id, process in list(task_processes.items()):
                try:
                    if process.returncode is None:
                        process.terminate()
                        try:
                            await asyncio.wait_for(process.wait(), timeout=5.0)
                        except asyncio.TimeoutError:
                            process.kill()
                except Exception as e:
                    logger.error(f"Error terminating task {task_id}: {e}")

    # Wait for pending callbacks to complete (critical for orchestrator notification)
    callback_wait_start = time.time()
    callback_max_wait = 15  # seconds
    while time.time() - callback_wait_start < callback_max_wait:
        async with pending_callbacks_lock:
            pending_count = len(pending_callbacks)
        if pending_count == 0:
            logger.info("All callbacks completed")
            break
        logger.info(f"Waiting for {pending_count} pending callbacks to complete...")
        await asyncio.sleep(1)

    # Log if any callbacks were not completed
    async with pending_callbacks_lock:
        if pending_callbacks:
            logger.warning(f"{len(pending_callbacks)} callbacks did not complete before shutdown timeout")

    # Cancel any remaining background tasks
    async with background_tasks_lock:
        if background_tasks:
            logger.warning(f"Cancelling {len(background_tasks)} remaining background tasks")
            for task_name, task in list(background_tasks.items()):
                try:
                    if not task.done():
                        task.cancel()
                        logger.info(f"Cancelled background task: {task_name}")
                except Exception as e:
                    logger.error(f"Error cancelling background task {task_name}: {e}")

    # Close database connections
    try:
        if task_processor and task_processor.session_manager:
            logger.info("Closing database connections...")
            # The Session class manages the connection pool
    except Exception as e:
        logger.error(f"Error closing database connections: {e}")

    logger.info("Graceful shutdown complete")

def serialize_task_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert datetime objects to ISO format strings for JSON serialization"""
    serialized = {}
    for key, value in data.items():
        if isinstance(value, datetime):
            serialized[key] = value.isoformat()
        elif isinstance(value, float):
            # Handle special float values (inf, -inf, nan) that aren't valid in JSON
            if math.isinf(value) or math.isnan(value):
                serialized[key] = str(value)  # Convert to string
            else:
                serialized[key] = value  # Keep normal float values as numbers
        elif isinstance(value, int):
            serialized[key] = value  # Keep integer values as numbers
        elif isinstance(value, dict):
            serialized[key] = serialize_task_data(value)
        else:
            serialized[key] = value
    return serialized

class TaskRequest(BaseModel):
    content_id: str
    task_type: str
    input_data: Dict[str, Any]
    worker_id: str
    original_task_id: Optional[str] = ""

class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str

class TaskCancelResponse(BaseModel):
    task_id: str
    status: str
    message: str

class TaskBatchRequest(BaseModel):
    tasks: list[TaskRequest]

class TaskBatchResponse(BaseModel):
    accepted_count: int
    task_ids: list[str]
    status: str
    message: str

class QueueStatusResponse(BaseModel):
    queue_depths: Dict[str, int]
    total_queued: int
    consumer_running: bool

# --- Start Refactor run_processing_script ---
async def is_head_node() -> bool:
    """Check if this is the head node by comparing hostname with config."""
    try:
        import socket
        hostname = socket.gethostname()
        head_node_ip = config.get('network', {}).get('head_node_ip', 'localhost')
        return hostname == head_node_ip or hostname == 'localhost' or hostname == '127.0.0.1'
    except Exception as e:
        logger.error(f"Error checking head node status: {e}")
        return False

async def check_visualization_count() -> int:
    """Check the number of existing visualization files in local directory."""
    try:
        # Only check if we're on the head node
        if not await is_head_node():
            logger.info("Not on head node, skipping visualization count check")
            return 0

        # Create visualization directory on Desktop if it doesn't exist
        desktop_path = Path.home() / "Desktop"
        viz_dir = desktop_path / "transcript_visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # Count existing visualization files - only count PNG files
        viz_files = list(viz_dir.glob("*.png"))
        count = len(viz_files)
        logger.info(f"Found {count} existing visualization files")
        return count
    except Exception as e:
        logger.error(f"Error checking visualization count: {e}")
        return 0  # Return 0 on error to be safe

async def run_processing_script(task_type: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Run the appropriate processing script based on task type using asyncio subprocess."""
    # This function now uses the global script_semaphore implicitly via process_task_background
    
    # Convert relative path to absolute path
    scripts_base_path = config['processing']['scripts_base_path']
    if not os.path.isabs(scripts_base_path):
        # If relative, make it absolute relative to the project root
        project_root = get_project_root()
        base_path = project_root / scripts_base_path
    else:
        base_path = Path(scripts_base_path)
    script_paths = {
        'download_youtube': base_path / 'download_youtube.py',
        'download_podcast': base_path / 'download_podcast.py',
        'download_rumble': base_path / 'download_rumble.py',
        'transcribe': base_path / 'transcribe.py',
        'convert': base_path / 'convert.py',
        'diarize': base_path / 'diarize.py',
        'stitch': base_path / 'stitch.py',
        'cleanup': base_path / 'cleanup_and_compress.py'
    }

    if task_type not in script_paths:
        logger.error(f"Unknown task type: {task_type}")
        return {"status": "failed", "error": f"Unknown task type: {task_type}"}

    script_path = script_paths[task_type]
    if not script_path.exists():
        logger.error(f"Processing script not found at {script_path}")
        return {"status": "failed", "error": f"Processing script not found at {script_path}"}

    # Define required arguments per task type
    task_args_config = {
        'transcribe': ['content_id', 'chunk_index'],
        'download_podcast': ['content_id'],
        'download_youtube': ['content_id'],
        'download_rumble': ['content_id'],
        'convert': ['content_id'],
        'diarize': ['content_id'],
        'stitch': ['content_id'],
        'cleanup': ['content_id'],
    }

    # Determine arguments needed for the current task
    required_args_keys = task_args_config.get(task_type, ['content_id'])
    logger.debug(f"Task type '{task_type}' requires args: {required_args_keys}")
    
    # Build command list using uv
    # uv automatically uses the .venv in the project directory
    uv_paths = [
        '/Users/signal4/.local/bin/uv',  # Standard uv install location
        '/opt/homebrew/bin/uv',  # Homebrew install
        '/usr/local/bin/uv',  # System install
        'uv'  # Fallback to PATH
    ]

    uv_cmd = None
    for uv_path in uv_paths:
        if Path(uv_path).exists():
            uv_cmd = uv_path
            break

    if not uv_cmd:
        logger.warning("uv not found in standard locations, trying PATH")
        uv_cmd = 'uv'

    # Get project root for uv to find the .venv
    project_root = get_project_root()

    # Use uv run which automatically uses the project's .venv
    cmd = [uv_cmd, 'run', '--project', str(project_root), 'python', str(script_path)]
    logger.debug(f"Using uv for subprocess from project: {project_root}")
    logger.debug(f"Running command: {' '.join(cmd)}")
    input_data = task_data.get('input_data', {})
    
    # Standard argument processing for ALL task types
    for arg_key in required_args_keys:
        logger.debug(f"Processing arg_key: '{arg_key}'") # Log current key
        value = None
        source = ""
        # Check task_data first (for content_id)
        if arg_key == 'content_id':
            value = task_data.get('content_id')
            source = "task_data"
        # If not in task_data, check input_data (for chunk_index etc.)
        if value is None:
            value = input_data.get(arg_key)
            source = "input_data"

        logger.debug(f"Retrieved value for '{arg_key}' from '{source}': {repr(value)}") # Log value and source

        if value is None:
            err_msg = f"Missing required argument '{arg_key}' in task_data or input_data for task type '{task_type}'"
            logger.error(err_msg)
            return {"status": "failed", "error": err_msg}

        logger.debug(f"Command list before appending '{arg_key}': {cmd}") # Log cmd before append
        
        # Map argument keys to their new format using --key=value
        if arg_key == 'content_id':
            cmd.append(f'--content={str(value)}')
        elif arg_key == 'chunk_index':
            cmd.append(f'--chunk={str(value)}')
        elif arg_key == 'worker_id':
            cmd.append(f'--worker={str(value)}')
        else:
            # For other arguments, use flag format --key=value
            cmd.append(f"--{arg_key.replace('_', '-')}={str(value)}")
            
        logger.debug(f"Command list after appending '{arg_key}': {cmd}") # Log cmd after append
    
    # Need to rewrite stitch types one time because algo change
    if task_type == 'stitch':
        cmd.append("--rewrite")

    # Add --rewrite for transcribe tasks if requested or if existing transcript uses unsafe model
    if task_type == 'transcribe':
        # First check if rewrite is explicitly requested in input_data
        if input_data.get('rewrite'):
            cmd.append("--rewrite")
            logger.info(f"Adding --rewrite flag: explicitly requested in input_data (reason: {input_data.get('reason', 'unspecified')})")
        else:
            try:
                # Check if chunk has existing transcript with unsafe model
                with task_processor.get_db_session() as session:
                    from src.database.models import Content, ContentChunk

                    content_id = task_data.get('content_id')
                    chunk_index = input_data.get('chunk_index')

                    if content_id and chunk_index is not None:
                        content = session.query(Content).filter_by(content_id=content_id).first()
                        if content:
                            chunk = session.query(ContentChunk).filter_by(
                                content_id=content.id,
                                chunk_index=chunk_index
                            ).first()

                            if chunk and chunk.transcribed_with:
                                # Get safe models from config
                                safe_models = config.get('processing', {}).get('transcription', {}).get('safe_models', [])

                                if chunk.transcribed_with not in safe_models:
                                    cmd.append("--rewrite")
                                    logger.info(f"Adding --rewrite flag: chunk {chunk_index} uses unsafe model '{chunk.transcribed_with}'")
                                else:
                                    logger.debug(f"Chunk {chunk_index} already uses safe model '{chunk.transcribed_with}', no rewrite needed")
            except Exception as e:
                logger.warning(f"Error checking transcription model for rewrite decision: {e}")
                # Don't fail the task, just skip the rewrite check

    # Add cookie profile argument for download_youtube tasks
    if task_type == 'download_youtube':
        cookie_profile = input_data.get('cookie_profile')
        if cookie_profile:
            cmd.extend(['--cookies', str(cookie_profile)])
            logger.debug(f"Added --cookies={cookie_profile} argument for download_youtube")
        else:
            logger.debug("No cookie_profile found in input_data for download_youtube")
    # --- END MODIFICATION ---

    # Add debug flag to convert.py if needed
    if task_type == 'convert' and task_data.get('debug', False):
        cmd.append('--debug')
        logger.debug("Added --debug flag to convert.py command")
    
    # Add visualization flag for stitch tasks if under limit and on head node
    if task_type == 'stitch':
        try:
            # Only proceed with visualization if we're on the head node
            if await is_head_node():
                viz_count = await check_visualization_count()
                logger.info(f"Current visualization count: {viz_count}")
                if viz_count < 100:  # Hard limit of 100 visualizations
                    # Create visualization directory on Desktop if it doesn't exist
                    desktop_path = Path.home() / "Desktop"
                    viz_dir = desktop_path / "transcript_visualizations"
                    viz_dir.mkdir(exist_ok=True)
                    
                    # Add both visualize flag and output directory
                    cmd.extend(['--visualize', '--viz-dir', str(viz_dir)])
                    logger.info(f"Adding --visualize flag to stitch command (under 100 limit), output to {viz_dir}")
                else:
                    logger.info("Skipping --visualize flag (reached 100 limit)")
            else:
                logger.info("Not on head node, skipping visualization generation")
        except Exception as e:
            logger.warning(f"Error checking visualization count, skipping --visualize flag: {e}")
    
    cmd_str = ' '.join(cmd)
    logger.debug(f"Running command: {cmd_str}")

    # Run the script using asyncio.create_subprocess_exec
    process = None
    try:
        # Get subprocess environment with all required paths (PATH, DYLD_*, etc.)
        env = get_subprocess_env()
        # Add PYTHONPATH to ensure src imports work
        current_pythonpath = env.get('PYTHONPATH', '')
        if current_pythonpath:
            env['PYTHONPATH'] = f"{project_root}:{current_pythonpath}"
        else:
            env['PYTHONPATH'] = str(project_root)

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(project_root),
            env=env
        )

        # Store process reference for potential cancellation
        task_id = task_data.get("original_task_id", "")
        if task_id:
            async with task_processes_lock:
                task_processes[task_id] = process
            logger.debug(f"Stored process reference for task {task_id}")

        stdout_bytes, stderr_bytes = await process.communicate()

        # Remove process reference after completion
        if task_id:
            async with task_processes_lock:
                if task_id in task_processes:
                    del task_processes[task_id]
            logger.debug(f"Removed process reference for completed task {task_id}")

        # Wait for the process to terminate and get the return code
        return_code = process.returncode # Get return code directly after communicate

        stdout = stdout_bytes.decode('utf-8', errors='replace').strip() if stdout_bytes else ""
        stderr = stderr_bytes.decode('utf-8', errors='replace').strip() if stderr_bytes else ""

        # Filter out non-error noise from stderr (informational messages and warnings)
        if stderr:
            stderr_lines = []
            for line in stderr.splitlines():
                # Skip uv package manager noise
                if line.startswith(('Bytecode compiled', 'Resolved ', 'Prepared ', 'Installed ', 'Uninstalled ')):
                    continue
                # Skip PyTorch Lightning checkpoint upgrade warnings (informational, not errors)
                if 'Lightning automatically upgraded your loaded checkpoint' in line:
                    continue
                if 'pytorch_lightning.utilities.upgrade_checkpoint' in line:
                    continue
                # Skip aiohttp unclosed session warnings (cleanup warnings, not errors)
                if 'Unclosed client session' in line or 'Unclosed connector' in line:
                    continue
                # Skip multiprocessing resource tracker warnings
                if 'resource_tracker:' in line and 'leaked semaphore' in line:
                    continue
                stderr_lines.append(line)
            stderr = '\n'.join(stderr_lines).strip()
        
        # Special handling for convert.py exit code 2 (invalid media file)
        if task_type == "convert" and return_code == 2:
            logger.warning(f"Convert task exited with code 2, indicating invalid media file for {task_data.get('content_id')}")
            # Check if we can parse the stdout for details
            try:
                parsed_result = json.loads(stdout)
                if parsed_result.get('error') == 'invalid_media_file':
                    return {
                        "status": "failed",
                        "error": "invalid_media_file",
                        "error_details": parsed_result.get('error_details', "Invalid or corrupted media file detected"),
                        "source_key": parsed_result.get('source_key'),
                        "exit_code": 2
                    }
                else:
                    # Fall back to generic error with exit code 2
                    return {
                        "status": "failed",
                        "error": "invalid_media_file",
                        "error_details": "Media file appears to be invalid or corrupted",
                        "exit_code": 2
                    }
            except json.JSONDecodeError:
                # If we can't parse the output, still return the specialized error
                return {
                    "status": "failed",
                    "error": "invalid_media_file",
                    "error_details": "Media file appears to be invalid or corrupted (output parsing failed)",
                    "exit_code": 2
                }

        # --- Start: Robust JSON parsing from stdout --- #
        parsed_result = None
        json_parsing_error = None
        if stdout:
            for line in stdout.splitlines():
                line = line.strip()
                if line.startswith('{') and line.endswith('}'):
                    try:
                        parsed_result = json.loads(line)
                        logger.debug(f"Successfully parsed JSON from stdout line: {line[:200]}...")
                        break # Use the first valid JSON line found
                    except json.JSONDecodeError as e:
                        json_parsing_error = e # Store the error if parsing this potential line failed
                        logger.debug(f"Failed to parse potential JSON line: {line[:200]}... Error: {e}")
                        continue # Try next line

        # --- NEW LOGIC: Check parsed_result first, then return_code ---
        if parsed_result:
            # We have JSON from the script, prioritize its status/error info
            script_status = parsed_result.get('status', 'unknown')
            
            # Determine final status based on JSON content and exit code
            if return_code != 0:
                # Script exited non-zero, force status to failed, but keep details from JSON
                final_status = 'failed'
                parsed_result['status'] = final_status
                parsed_result['exit_code'] = return_code
                # Ensure error field exists, prefer script's error message
                if 'error' not in parsed_result:
                    parsed_result['error'] = stderr or f"Script exited {return_code} but provided no JSON error key."
                logger.error(f"Script execution failed (exit code {return_code}). Using error from JSON: {parsed_result['error']}")
                if parsed_result.get('error_type'):
                    logger.error(f"Error type from JSON: {parsed_result.get('error_type')}")
            else:
                # Script exited zero, trust JSON status but normalize
                if script_status in ['error', 'failed']:
                    final_status = 'failed'
                    if 'error' not in parsed_result:
                        parsed_result['error'] = parsed_result.get('message', f"Script reported status '{script_status}' but no error detail.")
                    logger.warning(f"Script exited 0 but reported failure status in JSON: {script_status}")
                elif script_status in ['completed', 'success', 'skipped']:
                    final_status = 'completed' # Treat skipped as completed for processor status
                    if script_status == 'skipped':
                        logger.info(f"Script reported skipped status - treating as completed.")
                else:
                    final_status = 'completed'
                    logger.warning(f"Script reported unexpected status '{script_status}'. Treating as completed.")
                parsed_result['status'] = final_status
                
            return parsed_result # Return the JSON result (potentially modified)

        else:
            # No valid JSON parsed from stdout
            if return_code != 0:
                # Use stderr or stdout or generic message as error
                error_msg = stderr or stdout or f"Script returned non-zero exit code {return_code}"
                logger.error(f"Command '{cmd_str}' failed with exit code {return_code}. No valid JSON found in stdout.")
                if stderr:
                    logger.error(f"  Stderr: {stderr}")
                if stdout:
                     logger.error(f"  Stdout: {stdout}") # Log stdout if no JSON was found
                # Explicitly set error_type to None as we couldn't parse it
                return {"status": "failed", "error": error_msg, "exit_code": return_code, "error_type": None}
            else:
                # Script exited 0 but no valid JSON found
                logger.warning(f"Script exited 0 but stdout was not valid JSON (or empty): {stdout[:200]}...")
                return {"raw_output": stdout, "status": "completed", "message": "Script completed but output was not valid JSON."}
        # --- End: Robust JSON parsing from stdout --- #

    except FileNotFoundError as fnf_error:
        # Specific error for conda/python/script not found
        err_msg = f"Executable or script not found for command: {' '.join(cmd)}. Error: {fnf_error}"
        logger.error(err_msg)
        # Return error structure for FileNotFoundError
        return {"status": "failed", "error": err_msg}
    except Exception as e:
        logger.error(f"Error running processing script via asyncio: {e}", exc_info=True)
        # Return error structure for general exceptions
        return {"status": "failed", "error": f"Async execution error: {str(e)}"}
    finally:
        # Ensure process is terminated if something went wrong before communicate/wait
        # Check returncode is None *before* trying to terminate
        if process and process.returncode is None:
            logger.warning(f"Script process for command '{cmd_str}' did not exit cleanly. Attempting termination.")
            try:
                process.terminate()
                # Wait briefly for termination (non-blocking if possible)
                try:
                    await asyncio.wait_for(process.wait(), timeout=2.0)
                    logger.info(f"Terminated hanging script process {process.pid}")
                except asyncio.TimeoutError:
                    logger.warning(f"Process {process.pid} did not terminate gracefully after 2s. Sending SIGKILL.")
                    process.kill()
                    await process.wait() # Wait for kill
                    logger.info(f"Killed hanging script process {process.pid}")
            except ProcessLookupError:
                logger.debug("Process already finished before termination attempt.") # Process finished between check and terminate
            except Exception as term_err:
                logger.error(f"Error trying to terminate/kill hanging process {getattr(process, 'pid', 'N/A')}: {term_err}")

        # Clean up process reference if it exists
        task_id = task_data.get("original_task_id", "")
        if task_id:
            async with task_processes_lock:
                if task_id in task_processes:
                    del task_processes[task_id]
            logger.debug(f"Cleaned up process reference for task {task_id} in finally block")

# --- End Refactor run_processing_script ---

# --- Queue Management Functions ---
async def ensure_queue_for_task_type(task_type: str):
    """Ensure a queue exists for the given task type."""
    async with queue_lock:
        if task_type not in task_queues:
            # Create queue with maxsize for backpressure (prevents memory explosion)
            task_queues[task_type] = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
            queue_depths[task_type] = 0
            logger.info(f"Created queue for task type: {task_type} (maxsize={MAX_QUEUE_SIZE})")

async def enqueue_task(task_id: str, task_type: str, task_data: Dict[str, Any]):
    """Add a task to the appropriate queue.

    Note: The queue.put() call is done OUTSIDE the lock to prevent potential deadlock.
    If a maxsize is ever added to the queue, put() could block waiting for space,
    while queue_consumer_loop needs the lock to pull items.
    """
    await ensure_queue_for_task_type(task_type)

    # Get queue reference while holding lock
    async with queue_lock:
        queue = task_queues[task_type]

    # Put task OUTSIDE the lock (safe even if queue has maxsize)
    await queue.put((task_id, task_type, task_data))

    # Update depth tracking while holding lock
    async with queue_lock:
        queue_depths[task_type] = task_queues[task_type].qsize()
        logger.debug(f"Enqueued task {task_id} ({task_type}), queue depth: {queue_depths[task_type]}")

async def queue_consumer_loop():
    """
    Consumer loop that pulls tasks from all queues and processes them.
    This runs continuously and feeds tasks into process_task_background.
    Respects per-task-type concurrency limits from config.
    """
    logger.info("Queue consumer loop started")

    while True:
        try:
            # Check all task type queues for work
            tasks_to_process = []

            async with queue_lock:
                for task_type, queue in task_queues.items():
                    if not queue.empty():
                        # Check if we have capacity for this task type
                        current_running = running_tasks_by_type.get(task_type, 0)
                        limit = task_type_limits.get(task_type, 1)  # Default to 1 if not configured

                        # Pull multiple tasks up to the limit in one iteration
                        while current_running < limit and not queue.empty():
                            try:
                                task_id, task_type, task_data = queue.get_nowait()
                                tasks_to_process.append((task_id, task_type, task_data))
                                queue_depths[task_type] = queue.qsize()
                                # Increment running count (will be decremented when task completes)
                                current_running += 1
                                running_tasks_by_type[task_type] = current_running
                                logger.debug(f"Task type {task_type}: {running_tasks_by_type[task_type]}/{limit} running")
                            except asyncio.QueueEmpty:
                                break

                        if current_running >= limit:
                            logger.debug(f"Task type {task_type} at capacity: {current_running}/{limit}, skipping")

            # Process tasks outside the lock
            for task_id, task_type, task_data in tasks_to_process:
                # Launch task processing (it will handle semaphore internally)
                # Use tracked_create_task with timeout based on task type
                # Most tasks should complete within 2 hours; long tasks like transcription may take longer
                task_timeout = 7200  # 2 hours default
                if task_type in ['transcribe', 'stitch', 'download_youtube', 'download_rumble']:
                    task_timeout = 14400  # 4 hours for long-running tasks
                await tracked_create_task(
                    process_task_background(task_id, task_type, task_data),
                    name=f"process_task_{task_type}_{task_id[:20]}",
                    timeout=task_timeout,
                    on_error="log"
                )
                logger.debug(f"Dispatched task {task_id} from queue to background processor")

            # Sleep between queue checks
            await asyncio.sleep(1.0)

        except Exception as e:
            logger.error(f"Error in queue consumer loop: {str(e)}", exc_info=True)
            await asyncio.sleep(1)  # Longer sleep on error

async def start_queue_consumer():
    """Start the queue consumer loop if not already started."""
    global queue_consumer_started
    if not queue_consumer_started:
        queue_consumer_started = True
        # Queue consumer runs indefinitely - no timeout, log errors
        await tracked_create_task(
            queue_consumer_loop(),
            name="queue_consumer_loop",
            timeout=None,  # No timeout - runs forever
            on_error="log"
        )
        logger.info("Queue consumer task created")

# --- Config Hot-Reload ---
config_watcher_started = False

async def config_watcher_loop():
    """Background loop that checks for config changes and reloads task limits."""
    check_interval = 30  # Check every 30 seconds
    logger.info(f"Config watcher started (checking every {check_interval}s)")

    while True:
        try:
            await asyncio.sleep(check_interval)

            if config_manager.has_config_changed():
                logger.info("Config file changed, reloading...")
                config_manager.reload_config()
                # reload_task_type_limits is called via the registered callback
        except Exception as e:
            logger.error(f"Error in config watcher loop: {str(e)}", exc_info=True)
            await asyncio.sleep(check_interval)

async def start_config_watcher():
    """Start the config watcher loop if not already started."""
    global config_watcher_started
    if not config_watcher_started:
        config_watcher_started = True
        await tracked_create_task(
            config_watcher_loop(),
            name="config_watcher_loop",
            timeout=None,  # No timeout - runs forever
            on_error="log"
        )
        logger.info("Config watcher task created")

# --- End Config Hot-Reload ---

# --- End Queue Management Functions ---

@app.post("/tasks/process", response_model=TaskResponse)
async def process_task(task_request: TaskRequest):
    """Process a task request non-blockingly."""
    logger.info(f"Received task request for content_id={task_request.content_id}, task_type={task_request.task_type}")
    logger.debug(f"Task request includes original_task_id: '{task_request.original_task_id}'")
    logger.debug(f"Full task request: {task_request.model_dump()}")
    
    # Generate a unique task ID (remains the same)
    task_id = f"{task_request.worker_id}_{task_request.content_id}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}" # Added microseconds for more uniqueness
    
    # Store task information with serialized datetime (remains the same)
    task_data = {
        "status": "accepted", # Initial status is accepted, will change to processing in background task
        "start_time": datetime.now(timezone.utc),
        "task_type": task_request.task_type,
        "content_id": task_request.content_id,
        "input_data": task_request.input_data,
        "worker_id": task_request.worker_id,
        "original_task_id": task_request.original_task_id  # Store the original task ID
    }
    async with tasks_lock:
        tasks[task_id] = serialize_task_data(task_data)  # Store basic info immediately

    # Log task acceptance
    logger.info(f"Accepted task {task_id} for content {task_request.content_id} of type {task_request.task_type}")

    # --- Start Refactor Background Task Launch ---
    # Start background loops if not already running
    await start_queue_consumer()
    await start_config_watcher()

    # Enqueue the task to use the same queue system as batch tasks
    # This prevents duplicate processing if both single and batch endpoints are used
    await enqueue_task(task_id, task_request.task_type, task_data)
    logger.debug(f"Enqueued single task {task_id} to queue")
    # --- End Refactor Background Task Launch ---
    
    return TaskResponse(
        task_id=task_id,
        status="accepted",
        message="Task accepted for processing"
    )

async def _log_completion_background(task_type: str, task_id: str, content_id: str, duration: float,
                                     chunk_index: Optional[int], session_task_count: Optional[int],
                                     session_task_limit: Optional[int]) -> None:
    """Log task completion in background to avoid blocking callback"""
    try:
        log_task_completion(
            task_type=task_type,
            task_id=task_id,
            content_id=content_id,
            duration=duration,
            chunk_index=chunk_index,
            session_task_count=session_task_count,
            session_task_limit=session_task_limit
        )
    except Exception as e:
        logger.error(f"Error logging task completion in background: {str(e)}")

async def process_task_background(task_id: str, task_type: str, task_data: Dict[str, Any]):
    """Process a task in the background, respecting concurrency limits."""

    # Update status in shared dict to 'processing' (with lock for thread safety)
    async with tasks_lock:
        if task_id in tasks:
            tasks[task_id]["status"] = "processing"
        else:
            logger.warning(f"Task {task_id} not found in tasks dict at start of background processing.")
            # Task vanished? Don't proceed.
            return

    logger.info(f"Starting background processing for task {task_id} (type: {task_type})")
    content_id = task_data["content_id"]
    result = {}
    status = "failed" # Default status
    duration = 0.0

    # --- Acquire Semaphore ---
    logger.debug(f"Task {task_id} waiting to acquire script semaphore...")
    async with script_semaphore:
        logger.debug(f"Task {task_id} acquired script semaphore.")

        # Reset start time to NOW (when actual processing begins, not when queued)
        start_time_obj = datetime.now(timezone.utc)
        async with tasks_lock:
            if task_id in tasks:
                tasks[task_id]["start_time"] = start_time_obj.isoformat()

        # Update started_at in database to reflect actual execution start
        # This ensures timeout is calculated from execution time, not assignment time
        original_task_id = task_data.get("original_task_id")
        if original_task_id:
            try:
                with task_processor.get_db_session() as db_session:
                    db_task = db_session.query(TaskQueue).filter_by(id=original_task_id).first()
                    if db_task:
                        db_task.started_at = start_time_obj
                        logger.debug(f"Updated started_at in DB for task {original_task_id}")
            except Exception as e:
                logger.warning(f"Failed to update started_at for task {original_task_id}: {e}")
                # Non-fatal - continue with execution

        try:
            # Run the processing script (now async)
            logger.info(f"Executing processing script for task {task_id}")
            result = await run_processing_script(task_type, task_data) # Await the async function
            
            # Extract status from script result, default to 'failed' if missing
            status = result.get('status', 'failed')
            
            # Check for explicit failure from script
            if status == 'failed':
                 error_msg = result.get('error', 'Script indicated failure')
                 logger.error(f"Task {task_id} script execution failed: {error_msg}")
            elif status == 'completed':
                 logger.info(f"Task {task_id} script execution completed successfully.")
            else:
                 logger.warning(f"Task {task_id} script execution finished with unexpected status: {status}")
                 # Treat unexpected status as failure? Or completed? Let's default to completed if exit code was 0.
                 if result.get("exit_code", 1) == 0: # Check if we stored exit code
                      status = "completed"
                      logger.warning(f"Task {task_id} treating status '{result.get('status')}' as completed due to exit code 0.")
                 else:
                      status = "failed"
                      result["error"] = result.get("error", f"Script finished with unhandled status: {result.get('status')}")
                      logger.error(f"Task {task_id} treating status '{result.get('status')}' as failed.")
                      
            
        except Exception as e:
            # Catch errors during the script execution itself (e.g., setup issues before subprocess call)
            logger.error(f"Error processing task {task_id} in background: {str(e)}", exc_info=True)
            status = "failed"
            result = {"error": f"Background processing error: {str(e)}"}
        finally:
             logger.debug(f"Task {task_id} released script semaphore.")
             # --- End Semaphore Release ---

             # Decrement running task count for this task type
             async with queue_lock:
                 if task_type in running_tasks_by_type:
                     running_tasks_by_type[task_type] = max(0, running_tasks_by_type[task_type] - 1)
                     limit = task_type_limits.get(task_type, 1)
                     logger.debug(f"Task {task_id} completed. Task type {task_type}: {running_tasks_by_type[task_type]}/{limit} running")

             # Calculate duration
             end_time = datetime.now(timezone.utc)
             duration = (end_time - start_time_obj).total_seconds()

             # Update task information in shared dict (with lock for thread safety)
             async with tasks_lock:
                 if task_id in tasks:
                      update_data = {
                          "status": status,
                          "end_time": end_time,
                          "result": result, # Store the actual result from the script
                          "duration": duration
                      }
                      # Add error field specifically if status is failed
                      if status == "failed":
                           update_data["error"] = result.get("error", "Unknown failure")

                      tasks[task_id].update(serialize_task_data(update_data))
                      logger.debug(f"Updated task {task_id} final status to {status}.")
                 else:
                      logger.warning(f"Task {task_id} not found in tasks dict for final update.")

             # Send callback to orchestrator IMMEDIATELY (don't wait for logging)
             # Use tracked callback to ensure graceful shutdown waits for it
             try:
                  # Pass the final determined status and result
                  await send_tracked_callback(task_id, task_data, result, status, duration, task_type)
             except Exception as callback_error:
                  logger.error(f"Failed to send completion callback for task {task_id}: {str(callback_error)}")

             # Log completion in background (non-blocking)
             # Short timeout of 30s for logging - should complete quickly
             await tracked_create_task(
                 _log_completion_background(
                     task_type, task_id, content_id, duration,
                     task_data.get("input_data", {}).get("chunk_index"),
                     task_data.get("input_data", {}).get("session_task_count"),
                     task_data.get("input_data", {}).get("session_task_limit")
                 ),
                 name=f"log_completion_{task_id[:20]}",
                 timeout=30,  # 30 seconds should be plenty for logging
                 on_error="log"  # Just log errors, don't fail the task
             )

             # Clean up task entry immediately (no need for sleep)
             async with tasks_lock:
                 if task_id in tasks:
                      del tasks[task_id]
                      logger.debug(f"Removed task {task_id} from task dictionary.")

# Note: The finally block encompassing semaphore release and task cleanup is crucial.

@app.get("/tasks/{task_id}/status")
async def get_task_status(task_id: str):
    """Get the status of a task."""
    logger.debug(f"Checking status for task {task_id}")

    # Get task info with lock for thread safety
    async with tasks_lock:
        if task_id not in tasks:
            logger.warning(f"Task {task_id} not found")
            raise HTTPException(status_code=404, detail="Task not found")
        # Copy task info to avoid holding lock during response generation
        task_info = dict(tasks[task_id])
    
    try:
        # Create response with basic info
        response = {
            "status": task_info["status"],
            "task_type": task_info["task_type"],
            "content_id": task_info["content_id"]
        }
        
        # Add duration if available
        if "duration" in task_info:
            duration = task_info["duration"]
            # Check for invalid JSON values
            if isinstance(duration, float) and (math.isinf(duration) or math.isnan(duration)):
                response["duration"] = str(duration)
            else:
                response["duration"] = duration
        
        # Add result or error if available
        if "result" in task_info:
            response["result"] = task_info["result"]
        if "error" in task_info:
            response["error"] = task_info["error"]
            
        # Test serialize to JSON to catch any issues
        try:
            json.dumps(response)
        except Exception as e:
            logger.error(f"JSON serialization error for task {task_id}: {str(e)}")
            # Remove problematic fields if necessary
            if "result" in response:
                response["result"] = str(response["result"])
            if "error" in response:
                response["error"] = str(response["error"])
        
        logger.debug(f"Returning status for task {task_id}: {task_info['status']}")
        return response
        
    except Exception as e:
        logger.error(f"Error generating response for task {task_id}: {str(e)}")
        # Return simple status as fallback
        return {"status": "error", "message": f"Error generating task status: {str(e)}"}

@app.get("/tasks")
async def list_tasks():
    """List all tasks for health check."""
    async with tasks_lock:
        task_count = len(tasks)
    return {
        "status": "healthy",
        "tasks": task_count,
        "session_manager": session_manager
    }

@app.delete("/tasks/{task_id}", response_model=TaskCancelResponse)
async def cancel_task(task_id: str):
    """Cancel a task if it exists and is still processing."""
    logger.info(f"Received cancellation request for task {task_id}")

    # Find task with matching original_task_id
    async with tasks_lock:
        task_api_id = None
        for api_id, info in tasks.items():
            if info.get("original_task_id") == task_id:
                task_api_id = api_id
                break

        if not task_api_id:
            raise HTTPException(status_code=404, detail="Task not found")

        task_info = tasks[task_api_id]
    
    # Only allow cancellation of processing tasks
    if task_info["status"] != "processing":
        return TaskCancelResponse(
            task_id=task_id,
            status=task_info["status"],
            message=f"Task is not processing (current status: {task_info['status']})"
        )
    
    # Attempt to terminate the running process
    async with task_processes_lock:
        process = task_processes.get(task_id)

    if process:
        try:
            logger.info(f"Attempting to terminate process for task {task_id}")
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=2.0)
                logger.info(f"Successfully terminated process for task {task_id}")
            except asyncio.TimeoutError:
                logger.warning(f"Process for task {task_id} did not terminate gracefully. Sending SIGKILL.")
                process.kill()
                await process.wait()
                logger.info(f"Force killed process for task {task_id}")
        except ProcessLookupError:
            logger.info(f"Process for task {task_id} already finished")
        except Exception as e:
            logger.error(f"Error terminating process for task {task_id}: {str(e)}")
        finally:
            async with task_processes_lock:
                if task_id in task_processes:
                    del task_processes[task_id]
    
    # Update task status
    task_info["status"] = "cancelled"
    task_info["end_time"] = datetime.now(timezone.utc)
    task_info["error"] = "Task cancelled by orchestrator"
    
    # Update task in database using connection pool
    try:
        with task_processor.get_db_session() as session:
            db_task = session.query(TaskQueue).filter_by(id=task_id).first()
            if db_task:
                db_task.status = "pending"  # Reset to pending for retry
                db_task.started_at = None
                db_task.last_heartbeat = None
                db_task.error = "Task cancelled by orchestrator"
                logger.info(f"Reset task {task_id} in database to pending status")
            else:
                logger.warning(f"Task {task_id} not found in database for status update")
    except Exception as e:
        logger.error(f"Error updating task {task_id} in database: {str(e)}")

    # Send cancellation callback to orchestrator so it knows the task was cancelled
    try:
        task_type = task_info.get("task_type", "unknown")
        start_time = task_info.get("start_time")
        duration = 0.0
        if start_time:
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()

        # Build minimal task_data for callback
        callback_task_data = {
            "content_id": task_info.get("content_id", ""),
            "worker_id": task_info.get("worker_id", ""),
            "original_task_id": task_id,
            "input_data": task_info.get("input_data", {})
        }

        await send_tracked_callback(
            task_id=task_api_id,
            task_data=callback_task_data,
            result={"status": "cancelled", "error": "Task cancelled by orchestrator"},
            status="cancelled",
            duration=duration,
            task_type=task_type
        )
        logger.info(f"Sent cancellation callback for task {task_id}")
    except Exception as e:
        logger.error(f"Error sending cancellation callback for task {task_id}: {str(e)}")

    logger.info(f"Cancelled task {task_id}")

    return TaskCancelResponse(
        task_id=task_id,
        status="cancelled",
        message="Task cancelled successfully"
    )

@app.post("/tasks/batch", response_model=TaskBatchResponse)
async def process_task_batch(batch_request: TaskBatchRequest):
    """
    Process a batch of tasks by adding them to the appropriate queues.
    This endpoint allows the orchestrator to send multiple tasks at once.
    """
    logger.info(f"Received batch request with {len(batch_request.tasks)} tasks")

    # Start background loops if not already running
    await start_queue_consumer()
    await start_config_watcher()

    accepted_task_ids = []

    for task_request in batch_request.tasks:
        # Generate unique task ID (same logic as single task endpoint)
        task_id = f"{task_request.worker_id}_{task_request.content_id}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"

        # Store task information
        task_data = {
            "status": "queued",  # Mark as queued instead of accepted
            "start_time": datetime.now(timezone.utc),
            "task_type": task_request.task_type,
            "content_id": task_request.content_id,
            "input_data": task_request.input_data,
            "worker_id": task_request.worker_id,
            "original_task_id": task_request.original_task_id
        }
        async with tasks_lock:
            tasks[task_id] = serialize_task_data(task_data)

        # Enqueue the task
        await enqueue_task(task_id, task_request.task_type, task_data)
        accepted_task_ids.append(task_id)

        logger.debug(f"Queued task {task_id} for content {task_request.content_id} of type {task_request.task_type}")

    logger.info(f"Queued {len(accepted_task_ids)} tasks from batch")

    return TaskBatchResponse(
        accepted_count=len(accepted_task_ids),
        task_ids=accepted_task_ids,
        status="queued",
        message=f"Batch of {len(accepted_task_ids)} tasks queued for processing"
    )

@app.get("/tasks/queue_status", response_model=QueueStatusResponse)
async def get_queue_status():
    """
    Get the current queue depth for all task types.
    This allows the orchestrator to monitor queue levels and decide when to send more tasks.
    """
    async with queue_lock:
        # Get current queue depths
        current_depths = dict(queue_depths)
        total = sum(current_depths.values())

    return QueueStatusResponse(
        queue_depths=current_depths,
        total_queued=total,
        consumer_running=queue_consumer_started
    )


@app.get("/status")
async def get_processor_status():
    """
    Get comprehensive processor status including background tasks.
    Useful for debugging and monitoring.
    """
    async with task_processes_lock:
        active_processes = len(task_processes)
        process_ids = list(task_processes.keys())

    async with background_tasks_lock:
        active_background = len(background_tasks)
        background_task_names = list(background_tasks.keys())

    async with queue_lock:
        current_depths = dict(queue_depths)
        running_by_type = dict(running_tasks_by_type)

    return {
        "status": "running",
        "shutdown_requested": shutdown_requested,
        "active_processes": active_processes,
        "process_ids": process_ids[:10],  # Limit to first 10 for brevity
        "active_background_tasks": active_background,
        "background_task_names": background_task_names[:10],  # Limit to first 10
        "queue_consumer_running": queue_consumer_started,
        "queue_depths": current_depths,
        "running_tasks_by_type": running_by_type,
        "task_type_limits": task_type_limits,
        "tasks_in_memory": len(tasks),
        "config_watcher_running": config_watcher_started
    }


@app.post("/reload-config")
async def reload_config():
    """
    Manually trigger a config reload.
    Useful for immediately applying config changes without waiting for the watcher.
    """
    if config_manager.reload_config():
        return {
            "status": "reloaded",
            "task_type_limits": dict(task_type_limits),
            "message": "Configuration reloaded successfully"
        }
    else:
        return {
            "status": "unchanged",
            "task_type_limits": dict(task_type_limits),
            "message": "Configuration has not changed"
        }


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting task processor API")
    logger.info(f"Session Manager: {session_manager}")

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, handle_shutdown_signal)
    signal.signal(signal.SIGINT, handle_shutdown_signal)
    logger.info("Registered signal handlers for graceful shutdown (SIGTERM, SIGINT)")

    # Clean up any existing processes before starting
    cleanup_existing_processes()

    # Configure uvicorn with shutdown handling
    uvicorn_config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
    server = uvicorn.Server(uvicorn_config)

    # Run with graceful shutdown support
    async def run_with_shutdown():
        """Run server with graceful shutdown handling."""
        try:
            await server.serve()
        except Exception as e:
            logger.error(f"Server error: {e}")
        finally:
            await graceful_shutdown()

    asyncio.run(run_with_shutdown()) 