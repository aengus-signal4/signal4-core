"""add embedding metrics

Revision ID: add_embedding_metrics
Revises: add_universal_names
Create Date: 2024-03-30 15:30:00.000000

"""
from alembic import op
import sqlalchemy as sa
from datetime import datetime

# revision identifiers, used by Alembic.
revision = 'add_embedding_metrics'
down_revision = 'add_universal_names'
branch_labels = None
depends_on = None

def upgrade():
    # Add new columns to speaker_embeddings table
    op.add_column('speaker_embeddings', sa.Column('segment_count', sa.Integer(), nullable=False, server_default='1'))
    op.add_column('speaker_embeddings', sa.Column('total_duration', sa.Float(), nullable=False, server_default='0.0'))

def downgrade():
    # Remove columns
    op.drop_column('speaker_embeddings', 'total_duration')
    op.drop_column('speaker_embeddings', 'segment_count') 