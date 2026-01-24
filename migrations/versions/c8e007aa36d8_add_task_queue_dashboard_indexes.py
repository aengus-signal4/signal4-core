"""add_task_queue_dashboard_indexes

Add composite indexes to tasks.task_queue for dashboard query performance.
These indexes optimize the most common dashboard queries:
- Completed tasks with time filtering (throughput charts)
- Worker performance queries
- Task queue status grouped by type/status

Revision ID: c8e007aa36d8
Revises: add_description_embedding
Create Date: 2026-01-24 07:00:37.059174

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'c8e007aa36d8'
down_revision: Union[str, None] = 'add_description_embedding'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Partial index for completed tasks queries with time filtering
    # Used by: get_completed_tasks_with_duration, throughput charts
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_task_queue_completed_at_desc
        ON tasks.task_queue(completed_at DESC)
        WHERE status = 'completed' AND completed_at IS NOT NULL
    """)

    # Composite index for worker performance queries
    # Used by: get_worker_performance_stats, worker throughput tables
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_task_queue_worker_completed
        ON tasks.task_queue(worker_id, task_type, completed_at DESC)
        WHERE status = 'completed'
    """)

    # Composite index for task queue status grouped queries
    # Used by: get_task_queue_status, task type aggregations
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_task_queue_type_status
        ON tasks.task_queue(task_type, status)
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS tasks.idx_task_queue_completed_at_desc")
    op.execute("DROP INDEX IF EXISTS tasks.idx_task_queue_worker_completed")
    op.execute("DROP INDEX IF EXISTS tasks.idx_task_queue_type_status")
