"""
Unified Scheduled Task Manager

Config-driven system for all periodic/scheduled tasks including:
- Content indexing (podcast, YouTube, Rumble)
- Task creation and embedding hydration
- Speaker identification phases
- Podcast chart collection
- Cache refresh and clustering (replaces pg_cron jobs)

See config/config.yaml under 'scheduled_tasks' for task definitions.
"""
import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List

from .schedule_types import (
    ScheduleType,
    ScheduleState,
    BaseSchedule,
    create_schedule
)
from .executors import CLIExecutor, SQLExecutor, ExecutionResult
from .executors.cli import check_screen_exists

logger = logging.getLogger(__name__)


@dataclass
class ScheduledTask:
    """Definition and state of a scheduled task"""
    task_id: str
    name: str
    description: str
    enabled: bool

    # Schedule configuration
    schedule: BaseSchedule
    schedule_config: Dict[str, Any]

    # Executor configuration
    executor_type: str  # 'cli' or 'sql'
    executor_config: Dict[str, Any]

    # Behavior
    timeout_seconds: int = 3600

    # Runtime state
    state: ScheduleState = field(default_factory=ScheduleState)


class ScheduledTaskManager:
    """
    Unified manager for all scheduled tasks.

    Features:
    - Config-driven task definitions
    - Multiple schedule types (time_of_day, interval, run_then_wait)
    - Multiple executor types (cli, sql)
    - State persistence across restarts
    - Manual trigger support
    - Status/monitoring API
    """

    def __init__(self, config: Dict[str, Any], orchestrator=None):
        self.config = config
        self.orchestrator = orchestrator

        # Task registry
        self.tasks: Dict[str, ScheduledTask] = {}

        # Executors
        self.executors: Dict[str, Any] = {}

        # State
        self.should_stop = False
        self._scheduler_task: Optional[asyncio.Task] = None
        self._running_tasks: Dict[str, asyncio.Task] = {}

        # Configuration
        scheduled_config = config.get('scheduled_tasks', {})
        self.enabled = scheduled_config.get('enabled', True)

        # State persistence
        self.state_file = Path(scheduled_config.get(
            'state_file',
            '/Users/signal4/logs/content_processing/scheduled_tasks_state.json'
        ))

        # Get default paths from config
        processing_config = config.get('processing', {})
        default_uv_path = processing_config.get(
            'uv_path',
            '/Users/signal4/.local/bin/uv'
        )
        storage_config = config.get('storage', {}).get('local', {})
        default_cwd = storage_config.get('base_path', '/Users/signal4/signal4/core')

        # Initialize executors
        self._init_executors(default_cwd, default_uv_path)

        # Load task definitions from config
        self._load_tasks()

        # Load persisted state
        self._load_state()

        logger.info(f"ScheduledTaskManager initialized with {len(self.tasks)} tasks")

    def _init_executors(self, default_cwd: str, default_uv_path: str):
        """Initialize task executors"""
        self.executors['cli'] = CLIExecutor(
            default_cwd=default_cwd,
            default_uv_path=default_uv_path
        )
        self.executors['sql'] = SQLExecutor()

    def _load_tasks(self):
        """Load task definitions from config"""
        scheduled_config = self.config.get('scheduled_tasks', {})
        tasks_config = scheduled_config.get('tasks', {})

        for task_id, task_config in tasks_config.items():
            try:
                # Create schedule
                schedule_config = task_config.get('schedule', {})
                schedule = create_schedule(schedule_config)

                # Create task
                task = ScheduledTask(
                    task_id=task_id,
                    name=task_config.get('name', task_id),
                    description=task_config.get('description', ''),
                    enabled=task_config.get('enabled', True),
                    schedule=schedule,
                    schedule_config=schedule_config,
                    executor_type=task_config.get('executor', {}).get('type', 'cli'),
                    executor_config=task_config.get('executor', {}),
                    timeout_seconds=task_config.get('timeout_seconds', 3600)
                )

                self.tasks[task_id] = task
                logger.info(f"Loaded task: {task_id} ({task.schedule_config.get('type', 'interval')})")

            except Exception as e:
                logger.error(f"Error loading task {task_id}: {e}")

    def _load_state(self):
        """Load persisted state from file.

        Note: is_running state is loaded but will be verified against actual
        screen sessions in _verify_running_tasks() called during start().
        """
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    state_data = json.load(f)

                tasks_state = state_data.get('tasks', {})
                for task_id, task_state in tasks_state.items():
                    if task_id in self.tasks:
                        task = self.tasks[task_id]
                        if task_state.get('last_run_time'):
                            task.state.last_run_time = datetime.fromisoformat(
                                task_state['last_run_time']
                            )
                        task.state.last_run_result = task_state.get('last_run_result')
                        task.state.last_duration_seconds = task_state.get('last_duration_seconds')
                        task.state.last_run_date = task_state.get('last_run_date')
                        task.state.last_error = task_state.get('last_error')
                        task.state.last_summary = task_state.get('last_summary')
                        # Load running state for restart recovery
                        task.state.is_running = task_state.get('is_running', False)
                        task.state.screen_session_name = task_state.get('screen_session_name')
                        if task_state.get('execution_start_time'):
                            task.state.execution_start_time = datetime.fromisoformat(
                                task_state['execution_start_time']
                            )

                logger.info(f"Loaded state for {len(tasks_state)} tasks")

        except Exception as e:
            logger.error(f"Error loading state: {e}")

    def _save_state(self):
        """Save current state to file, including running state for restart recovery"""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)

            state_data = {
                'version': '1.1.0',
                'updated_at': datetime.now(timezone.utc).isoformat(),
                'tasks': {}
            }

            for task_id, task in self.tasks.items():
                state_data['tasks'][task_id] = {
                    'last_run_time': task.state.last_run_time.isoformat() if task.state.last_run_time else None,
                    'last_run_result': task.state.last_run_result,
                    'last_duration_seconds': task.state.last_duration_seconds,
                    'last_run_date': task.state.last_run_date,
                    'last_error': task.state.last_error,
                    'last_summary': task.state.last_summary,
                    # Persist running state for restart recovery
                    'is_running': task.state.is_running,
                    'screen_session_name': task.state.screen_session_name,
                    'execution_start_time': task.state.execution_start_time.isoformat() if task.state.execution_start_time else None,
                }

            with open(self.state_file, 'w') as f:
                json.dump(state_data, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving state: {e}")

    async def _verify_running_tasks(self):
        """Verify tasks marked as running actually have screen sessions.

        Called on startup to reconcile persisted state with actual system state.
        This handles the case where orchestrator restarted while tasks were running.
        """
        verified_count = 0
        cleared_count = 0

        for task_id, task in self.tasks.items():
            if not task.state.is_running:
                continue

            screen_name = task.state.screen_session_name
            if not screen_name:
                # Task marked running but no screen session name - clear the flag
                logger.warning(
                    f"Task {task_id} marked as running but no screen session name, clearing flag"
                )
                task.state.is_running = False
                cleared_count += 1
                continue

            # Check if screen session actually exists
            if await check_screen_exists(screen_name):
                # Screen session exists - task is genuinely still running
                logger.info(
                    f"Task {task_id} has running screen session '{screen_name}', "
                    f"resuming monitoring (started: {task.state.execution_start_time})"
                )
                verified_count += 1
            else:
                # Screen session doesn't exist - task completed or crashed while we were down
                logger.warning(
                    f"Task {task_id} was marked running but screen session '{screen_name}' "
                    f"no longer exists, clearing flag"
                )
                task.state.is_running = False
                task.state.screen_session_name = None
                task.state.execution_start_time = None
                # Mark as failed since we don't know the outcome
                if task.state.last_run_result != 'success':
                    task.state.last_run_result = 'unknown'
                    task.state.last_error = 'Orchestrator restarted and screen session was lost'
                cleared_count += 1

        if verified_count > 0 or cleared_count > 0:
            logger.info(
                f"Running task verification: {verified_count} verified, {cleared_count} cleared"
            )
            self._save_state()

    async def start(self):
        """Start the scheduler"""
        if not self.enabled:
            logger.info("Scheduled task manager is disabled")
            return

        logger.info("Starting scheduled task manager")

        # Verify tasks marked as running actually have screen sessions
        await self._verify_running_tasks()

        self.should_stop = False
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())

    async def stop(self):
        """Stop the scheduler gracefully"""
        logger.info("Stopping scheduled task manager")
        self.should_stop = True

        # Cancel scheduler task
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass

        # Wait for running tasks to complete (with timeout)
        if self._running_tasks:
            logger.info(f"Waiting for {len(self._running_tasks)} running tasks to complete")
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._running_tasks.values(), return_exceptions=True),
                    timeout=60
                )
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for tasks, cancelling")
                for task in self._running_tasks.values():
                    task.cancel()

        # Save final state
        self._save_state()
        logger.info("Scheduled task manager stopped")

    async def _scheduler_loop(self):
        """Main scheduling loop - checks every minute"""
        # Wait a bit before first run to let system stabilize
        await asyncio.sleep(60)

        while not self.should_stop:
            try:
                now = datetime.now(timezone.utc)

                for task_id, task in self.tasks.items():
                    # Skip disabled tasks
                    if not task.enabled:
                        continue

                    # Skip already running tasks, but verify screen sessions still exist
                    if task.state.is_running:
                        # For screen-based tasks, verify the session still exists
                        if task.state.screen_session_name:
                            if not await check_screen_exists(task.state.screen_session_name):
                                # Screen session disappeared - task must have completed/crashed
                                logger.warning(
                                    f"Screen session '{task.state.screen_session_name}' for task "
                                    f"{task_id} no longer exists, clearing running state"
                                )
                                task.state.is_running = False
                                task.state.screen_session_name = None
                                task.state.execution_start_time = None
                                task.state.last_run_result = 'unknown'
                                task.state.last_error = 'Screen session disappeared unexpectedly'
                                self._save_state()
                                # Don't skip - let it be scheduled if needed
                            else:
                                continue  # Screen still running, skip
                        else:
                            continue  # Non-screen task, trust the flag

                    # Check if task should run
                    if task.schedule.should_run(task.state):
                        logger.info(f"Scheduling task: {task_id}")
                        self._running_tasks[task_id] = asyncio.create_task(
                            self._execute_task(task_id)
                        )

                # Clean up completed task references
                completed = [
                    tid for tid, t in self._running_tasks.items()
                    if t.done()
                ]
                for tid in completed:
                    del self._running_tasks[tid]

                # Sleep for 60 seconds before next check
                await asyncio.sleep(60)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}", exc_info=True)
                await asyncio.sleep(60)

    def _parse_task_summary(self, stdout: str) -> Optional[Dict[str, Any]]:
        """Parse TASK_SUMMARY JSON from script output.

        Scripts should print a line like:
        TASK_SUMMARY: {"items_processed": 100, "duration_minutes": 5.2, ...}
        """
        if not stdout:
            return None

        for line in stdout.split('\n'):
            line = line.strip()
            if line.startswith('TASK_SUMMARY:'):
                try:
                    json_str = line[len('TASK_SUMMARY:'):].strip()
                    return json.loads(json_str)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse TASK_SUMMARY: {e}")
                    return None
        return None

    async def _execute_task(self, task_id: str):
        """Execute a single task"""
        task = self.tasks.get(task_id)
        if not task:
            logger.error(f"Task {task_id} not found")
            return

        task.state.is_running = True
        task.state.execution_start_time = datetime.now(timezone.utc)
        logger.info(f"INITIALIZING TASK {task_id}")

        # Track screen session info for restart recovery
        use_screen = task.executor_config.get('use_screen', False)
        if use_screen:
            screen_name = task.executor_config.get('screen_name') or task_id
            task.state.screen_session_name = screen_name
            # Save state immediately so we can recover if orchestrator crashes
            self._save_state()
            logger.info(f"Task {task_id} will use screen session '{screen_name}'")

        try:
            # Get executor
            executor = self.executors.get(task.executor_type)
            if not executor:
                raise ValueError(f"Unknown executor type: {task.executor_type}")

            # Execute
            result = await executor.execute(
                config=task.executor_config,
                context={'orchestrator': self.orchestrator, 'task_id': task_id},
                timeout_seconds=task.timeout_seconds
            )

            # Update state
            task.state.last_run_time = result.end_time
            task.state.last_run_result = 'success' if result.success else 'failed'
            task.state.last_duration_seconds = result.duration_seconds
            task.state.last_run_date = result.end_time.strftime('%Y-%m-%d')

            # Capture error message
            task.state.last_error = result.error if not result.success else None

            # Parse summary from stdout
            stdout = result.output.get('stdout', '') if result.output else ''
            task.state.last_summary = self._parse_task_summary(stdout)

            if result.success:
                summary_info = f" - {task.state.last_summary}" if task.state.last_summary else ""
                logger.info(f"COMPLETED TASK {task_id}: success in {result.duration_seconds:.1f}s{summary_info}")
            elif result.output and result.output.get('skipped_already_running'):
                # Not a real failure - task was scheduled but already running
                logger.info(f"Task {task_id} scheduled but already running, no action taken")
                task.state.last_run_result = 'skipped'  # Override 'failed' status
            else:
                logger.error(f"COMPLETED TASK {task_id}: failed - {result.error}")

            # Save state
            self._save_state()

        except Exception as e:
            logger.error(f"COMPLETED TASK {task_id}: error - {e}", exc_info=True)
            task.state.last_run_time = datetime.now(timezone.utc)
            task.state.last_run_result = 'error'
            task.state.last_error = str(e)
            task.state.last_summary = None
            self._save_state()

        finally:
            task.state.is_running = False
            task.state.screen_session_name = None
            task.state.execution_start_time = None
            self._save_state()

    async def trigger_task(self, task_id: str) -> Dict[str, Any]:
        """Manually trigger a task"""
        task = self.tasks.get(task_id)
        if not task:
            return {
                'status': 'error',
                'message': f'Task {task_id} not found'
            }

        if task.state.is_running:
            return {
                'status': 'error',
                'message': f'Task {task_id} is already running'
            }

        logger.info(f"Manual trigger requested for task: {task_id}")

        # Run in background
        self._running_tasks[task_id] = asyncio.create_task(
            self._execute_task(task_id)
        )

        return {
            'status': 'success',
            'message': f'Task {task_id} triggered successfully'
        }

    def set_enabled(self, task_id: str, enabled: bool) -> Dict[str, Any]:
        """Enable or disable a task"""
        task = self.tasks.get(task_id)
        if not task:
            return {
                'status': 'error',
                'message': f'Task {task_id} not found'
            }

        task.enabled = enabled
        logger.info(f"Task {task_id} {'enabled' if enabled else 'disabled'}")

        return {
            'status': 'success',
            'message': f'Task {task_id} {"enabled" if enabled else "disabled"}'
        }

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task"""
        task = self.tasks.get(task_id)
        if not task:
            return None

        next_run = task.schedule.next_run_time(task.state)

        return {
            'task_id': task_id,
            'name': task.name,
            'description': task.description,
            'enabled': task.enabled,
            'is_running': task.state.is_running,
            'screen_session': task.state.screen_session_name,
            'execution_start_time': task.state.execution_start_time.isoformat() if task.state.execution_start_time else None,
            'schedule': {
                'type': task.schedule_config.get('type'),
                'config': task.schedule_config
            },
            'executor': {
                'type': task.executor_type,
                'config': task.executor_config
            },
            'last_run': {
                'time': task.state.last_run_time.isoformat() if task.state.last_run_time else None,
                'result': task.state.last_run_result,
                'duration_seconds': task.state.last_duration_seconds,
                'error': task.state.last_error,
                'summary': task.state.last_summary
            },
            'next_run_time': next_run.isoformat() if next_run else None,
            'timeout_seconds': task.timeout_seconds
        }

    def get_status(self) -> Dict[str, Any]:
        """Get status of all tasks"""
        return {
            'enabled': self.enabled,
            'task_count': len(self.tasks),
            'running_count': sum(1 for t in self.tasks.values() if t.state.is_running),
            'tasks': {
                task_id: self.get_task_status(task_id)
                for task_id in self.tasks.keys()
            }
        }

    def reload_config(self, new_config: Dict[str, Any]):
        """Reload task configuration (hot reload)"""
        logger.info("Reloading scheduled task configuration")

        # Update config reference
        self.config = new_config
        scheduled_config = new_config.get('scheduled_tasks', {})

        # Update enabled state
        self.enabled = scheduled_config.get('enabled', True)

        # Reload tasks (preserve running state)
        running_states = {
            tid: task.state.is_running
            for tid, task in self.tasks.items()
        }

        # Save current state
        old_states = {}
        for tid, task in self.tasks.items():
            old_states[tid] = task.state

        # Reload task definitions
        self._load_tasks()

        # Restore state for existing tasks
        for tid, state in old_states.items():
            if tid in self.tasks:
                self.tasks[tid].state = state

        logger.info(f"Reloaded configuration with {len(self.tasks)} tasks")
