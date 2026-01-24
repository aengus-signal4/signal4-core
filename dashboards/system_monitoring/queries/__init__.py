"""Database query functions for the system monitoring dashboard."""

from .tasks import (
    get_task_stats,
    get_recent_throughput,
    get_task_queue_status,
    get_hourly_throughput,
    get_completed_tasks_with_duration,
    get_worker_performance_stats,
)

from .content import (
    get_global_content_status,
    get_pipeline_progress_from_db,
)

from .cache import (
    get_cache_table_status,
)

__all__ = [
    # Task queries
    'get_task_stats',
    'get_recent_throughput',
    'get_task_queue_status',
    'get_hourly_throughput',
    'get_completed_tasks_with_duration',
    'get_worker_performance_stats',
    # Content queries
    'get_global_content_status',
    'get_pipeline_progress_from_db',
    # Cache queries
    'get_cache_table_status',
]
