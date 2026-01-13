"""add processor_task_id to task_queue

Revision ID: add_processor_task_id
Revises: add_speaker_turn_segments
Create Date: 2025-05-11

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'add_processor_task_id'
down_revision = 'add_speaker_turn_segments'
branch_labels = None
depends_on = None

def upgrade():
    op.add_column('task_queue', sa.Column('processor_task_id', sa.String(length=255), nullable=True))
    op.create_index(op.f('ix_task_queue_processor_task_id'), 'task_queue', ['processor_task_id'], unique=False)

def downgrade():
    op.drop_index(op.f('ix_task_queue_processor_task_id'), table_name='task_queue')
    op.drop_column('task_queue', 'processor_task_id')
