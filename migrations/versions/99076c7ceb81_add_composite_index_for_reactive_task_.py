"""add composite index for reactive task assignment

Revision ID: 99076c7ceb81
Revises: 97787729bcbe
Create Date: 2025-10-02 05:43:55.690030

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '99076c7ceb81'
down_revision: Union[str, None] = '97787729bcbe'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create composite index for reactive task assignment query optimization
    # This index covers: status filter, task_type filter (optional), and ordering by priority/created_at
    op.create_index(
        'idx_task_queue_pending_assignment',
        'task_queue',
        ['status', 'task_type', sa.text('priority DESC'), sa.text('created_at ASC')],
        schema='tasks',
        postgresql_where=sa.text("status = 'pending'")
    )


def downgrade() -> None:
    op.drop_index('idx_task_queue_pending_assignment', table_name='task_queue', schema='tasks')
