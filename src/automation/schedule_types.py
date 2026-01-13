"""
Schedule Types for Unified Scheduled Task Manager

Defines the different scheduling strategies:
- TIME_OF_DAY: Run at specific hours (e.g., 00:00, 08:00, 16:00)
- INTERVAL: Run every N seconds
- RUN_THEN_WAIT: Run task, then wait N seconds after completion before next run
"""
from enum import Enum
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class ScheduleType(Enum):
    """Types of scheduling strategies"""
    TIME_OF_DAY = "time_of_day"      # Run at specific hours
    INTERVAL = "interval"            # Run every N seconds
    RUN_THEN_WAIT = "run_then_wait"  # Run, complete, wait N seconds


@dataclass
class ScheduleState:
    """Runtime state for a scheduled task"""
    last_run_time: Optional[datetime] = None
    last_run_result: Optional[str] = None  # 'success', 'failed', 'error'
    last_duration_seconds: Optional[float] = None
    is_running: bool = False
    # For time_of_day with days_interval
    last_run_date: Optional[str] = None  # YYYY-MM-DD format


class BaseSchedule:
    """Base class for schedule implementations"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def should_run(self, state: ScheduleState) -> bool:
        """Check if the task should run now"""
        raise NotImplementedError

    def next_run_time(self, state: ScheduleState) -> Optional[datetime]:
        """Calculate the next scheduled run time"""
        raise NotImplementedError


class TimeOfDaySchedule(BaseSchedule):
    """
    Run at specific hours of the day.

    Config options:
    - hours: List of hours to run (0-23), e.g., [0, 8, 16]
    - minutes: List of minutes (default [0])
    - days_interval: Run every N days (default 1 = daily)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.hours: List[int] = config.get('hours', [0])
        self.minutes: List[int] = config.get('minutes', [0])
        self.days_interval: int = config.get('days_interval', 1)

    def should_run(self, state: ScheduleState) -> bool:
        if state.is_running:
            return False

        now = datetime.now(timezone.utc)
        current_hour = now.hour
        current_minute = now.minute
        current_date = now.strftime('%Y-%m-%d')

        # Check if current time matches any scheduled time (within 1 minute window)
        time_matches = False
        for hour in self.hours:
            for minute in self.minutes:
                if current_hour == hour and abs(current_minute - minute) <= 1:
                    time_matches = True
                    break
            if time_matches:
                break

        if not time_matches:
            return False

        # Check days_interval
        if self.days_interval > 1 and state.last_run_date:
            try:
                last_date = datetime.strptime(state.last_run_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
                days_since = (now.replace(hour=0, minute=0, second=0, microsecond=0) -
                             last_date.replace(hour=0, minute=0, second=0, microsecond=0)).days
                if days_since < self.days_interval:
                    return False
            except ValueError:
                pass  # Invalid date format, allow run

        # Check if already ran this hour (prevent duplicate runs)
        if state.last_run_time:
            # Allow re-run if it's been at least 50 minutes since last run
            elapsed = (now - state.last_run_time).total_seconds()
            if elapsed < 3000:  # 50 minutes
                return False

        return True

    def next_run_time(self, state: ScheduleState) -> Optional[datetime]:
        now = datetime.now(timezone.utc)

        # Find next scheduled time
        candidates = []
        for day_offset in range(self.days_interval + 1):
            check_date = now + timedelta(days=day_offset)
            for hour in sorted(self.hours):
                for minute in sorted(self.minutes):
                    candidate = check_date.replace(
                        hour=hour, minute=minute, second=0, microsecond=0
                    )
                    if candidate > now:
                        candidates.append(candidate)

        if candidates:
            next_time = min(candidates)

            # Apply days_interval if needed
            if self.days_interval > 1 and state.last_run_date:
                try:
                    last_date = datetime.strptime(state.last_run_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
                    min_next_date = last_date + timedelta(days=self.days_interval)
                    if next_time.date() < min_next_date.date():
                        # Find first scheduled time on or after min_next_date
                        for hour in sorted(self.hours):
                            for minute in sorted(self.minutes):
                                candidate = min_next_date.replace(
                                    hour=hour, minute=minute, second=0, microsecond=0
                                )
                                return candidate
                except ValueError:
                    pass

            return next_time

        return None


class IntervalSchedule(BaseSchedule):
    """
    Run every N seconds.

    Config options:
    - interval_seconds: Seconds between runs
    - offset_seconds: Offset from interval start (e.g., 600 to run 10 min after another task)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.interval_seconds: int = config.get('interval_seconds', 3600)
        self.offset_seconds: int = config.get('offset_seconds', 0)

    def should_run(self, state: ScheduleState) -> bool:
        if state.is_running:
            return False

        if not state.last_run_time:
            # First run - check offset
            if self.offset_seconds > 0:
                # For offset tasks, wait for offset time to pass since system start
                # This is a simplified approach - could be enhanced with a start timestamp
                return False
            return True

        now = datetime.now(timezone.utc)
        elapsed = (now - state.last_run_time).total_seconds()
        return elapsed >= self.interval_seconds

    def next_run_time(self, state: ScheduleState) -> Optional[datetime]:
        if not state.last_run_time:
            now = datetime.now(timezone.utc)
            return now + timedelta(seconds=self.offset_seconds)

        return state.last_run_time + timedelta(seconds=self.interval_seconds)


class RunThenWaitSchedule(BaseSchedule):
    """
    Run task, then wait N seconds after completion before next run.
    Unlike interval, the wait starts AFTER the task completes.

    Config options:
    - wait_seconds: Seconds to wait after task completion
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.wait_seconds: int = config.get('wait_seconds', 3600)

    def should_run(self, state: ScheduleState) -> bool:
        if state.is_running:
            return False

        if not state.last_run_time:
            return True  # First run

        now = datetime.now(timezone.utc)
        elapsed = (now - state.last_run_time).total_seconds()
        return elapsed >= self.wait_seconds

    def next_run_time(self, state: ScheduleState) -> Optional[datetime]:
        if not state.last_run_time:
            return datetime.now(timezone.utc)

        return state.last_run_time + timedelta(seconds=self.wait_seconds)


def create_schedule(schedule_config: Dict[str, Any]) -> BaseSchedule:
    """Factory function to create schedule instances from config"""
    schedule_type = schedule_config.get('type', 'interval')

    if schedule_type == 'time_of_day':
        return TimeOfDaySchedule(schedule_config)
    elif schedule_type == 'interval':
        return IntervalSchedule(schedule_config)
    elif schedule_type == 'run_then_wait':
        return RunThenWaitSchedule(schedule_config)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")
