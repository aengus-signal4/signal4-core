"""add_is_diarized_column

Revision ID: add_is_diarized_column
Revises: 099a4c99bb10
Create Date: 2025-03-29 21:23:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'add_is_diarized_column'
down_revision = '099a4c99bb10'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Check if the column exists before adding it
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    columns = [col['name'] for col in inspector.get_columns('content')]
    
    if 'is_diarized' not in columns:
        op.add_column('content', sa.Column('is_diarized', sa.Boolean(), server_default='false', nullable=False))


def downgrade() -> None:
    # Check if the column exists before dropping it
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    columns = [col['name'] for col in inspector.get_columns('content')]
    
    if 'is_diarized' in columns:
        op.drop_column('content', 'is_diarized') 