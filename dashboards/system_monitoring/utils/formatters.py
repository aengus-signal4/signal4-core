"""
Formatting utility functions for the system monitoring dashboard.
"""

from datetime import datetime, timezone


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string"""
    if seconds is None:
        return "N/A"
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_time_until(target_time_str: str) -> str:
    """Format time until a future datetime"""
    if not target_time_str:
        return "N/A"
    try:
        target = datetime.fromisoformat(target_time_str.replace('Z', '+00:00'))
        now = datetime.now(timezone.utc)
        delta = target - now

        if delta.total_seconds() < 0:
            return "overdue"

        total_seconds = delta.total_seconds()
        if total_seconds < 60:
            return f"in {int(total_seconds)}s"
        elif total_seconds < 3600:
            return f"in {int(total_seconds / 60)}m"
        elif total_seconds < 86400:
            hours = total_seconds / 3600
            return f"in {hours:.1f}h"
        else:
            days = total_seconds / 86400
            return f"in {days:.1f}d"
    except Exception:
        return "N/A"


def format_time_ago(timestamp_str: str) -> str:
    """Format time since a past datetime"""
    if not timestamp_str:
        return "Never"
    try:
        ts = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        now = datetime.now(timezone.utc)
        delta = now - ts

        total_seconds = delta.total_seconds()
        if total_seconds < 60:
            return f"{int(total_seconds)}s ago"
        elif total_seconds < 3600:
            return f"{int(total_seconds / 60)}m ago"
        elif total_seconds < 86400:
            hours = total_seconds / 3600
            return f"{hours:.1f}h ago"
        else:
            days = total_seconds / 86400
            return f"{days:.1f}d ago"
    except Exception:
        return timestamp_str[:16] if timestamp_str else "N/A"


def format_schedule_description(schedule_config: dict) -> str:
    """Format schedule configuration to human-readable description"""
    schedule_type = schedule_config.get('type', 'interval')

    if schedule_type == 'time_of_day':
        hours = schedule_config.get('hours', [0])
        minutes = schedule_config.get('minutes', [0])
        days_interval = schedule_config.get('days_interval', 1)

        times = []
        for h in hours:
            for m in minutes:
                times.append(f"{h:02d}:{m:02d}")

        times_str = ', '.join(times)

        if days_interval == 1:
            return f"Daily at {times_str}"
        else:
            return f"Every {days_interval}d at {times_str}"

    elif schedule_type == 'interval':
        interval = schedule_config.get('interval_seconds', 3600)
        if interval < 3600:
            return f"Every {interval // 60}m"
        else:
            return f"Every {interval // 3600}h"

    elif schedule_type == 'run_then_wait':
        wait = schedule_config.get('wait_seconds', 3600)
        if wait < 3600:
            return f"Continuous (wait {wait // 60}m)"
        else:
            return f"Continuous (wait {wait // 3600}h)"

    return schedule_type


def format_hours(hours: float) -> str:
    """Format hours value to string with appropriate units"""
    if hours >= 1:
        return f"{int(round(hours)):,}h"
    elif hours > 0:
        return f"{int(round(hours * 60))}m"
    else:
        return "0h"
