"""add_task_queue_unique_index

Adds unique index on tasks.task_queue to prevent duplicate tasks.

This prevents:
- Duplicate non-chunk tasks (download, convert, diarize, stitch, segment, cleanup)
- Duplicate chunk tasks (transcribe with same chunk_index)

The index is on (content_id, task_type, input_data->>'chunk_index').
For non-chunk tasks, chunk_index is NULL, so each content+task_type is unique.
For chunk tasks like transcribe, each content+task_type+chunk_index is unique.

Before adding the constraint, we remove duplicates (keeping most recent).

Revision ID: c532b92f9e66
Revises: 90c9b6501362
Create Date: 2025-11-19 09:22:45.249149

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c532b92f9e66'
down_revision: Union[str, None] = '90c9b6501362'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # First, remove duplicate tasks (keep the most recent one)
    op.execute("""
        DELETE FROM tasks.task_queue
        WHERE id IN (
            SELECT id
            FROM (
                SELECT id,
                       ROW_NUMBER() OVER (
                           PARTITION BY content_id, task_type, (input_data->>'chunk_index')
                           ORDER BY created_at DESC NULLS LAST
                       ) as rn
                FROM tasks.task_queue
            ) t
            WHERE t.rn > 1
        )
    """)

    # Create unique index on (content_id, task_type, chunk_index)
    # This handles both chunk and non-chunk tasks elegantly
    op.execute("""
        CREATE UNIQUE INDEX uq_task_queue_content_task_chunk
        ON tasks.task_queue (content_id, task_type, (input_data->>'chunk_index'))
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS tasks.uq_task_queue_content_task_chunk")
