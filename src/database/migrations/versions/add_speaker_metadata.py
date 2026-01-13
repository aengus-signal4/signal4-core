"""add speaker metadata

Revision ID: add_speaker_metadata
Revises: add_universal_names
Create Date: 2024-03-30 16:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision = 'add_speaker_metadata'
down_revision = 'add_universal_names'
branch_labels = None
depends_on = None

def upgrade():
    # Add metadata column to speakers table
    op.add_column('speakers', sa.Column('meta_data', JSONB, server_default='{}', nullable=False))

def downgrade():
    # Remove metadata column
    op.drop_column('speakers', 'meta_data') 