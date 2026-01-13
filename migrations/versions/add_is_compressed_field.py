"""Add is_compressed field to content table

Revision ID: add_is_compressed_field
Revises: ae737bb69988
Create Date: 2025-06-04 09:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'add_is_compressed_field'
down_revision = 'ae737bb69988'
branch_labels = None
depends_on = None


def upgrade():
    """Add is_compressed field to content table."""
    # Add the new column with default value False
    op.add_column('content', sa.Column('is_compressed', sa.Boolean(), nullable=False, server_default='false'))
    
    # Remove the server default after adding the column
    op.alter_column('content', 'is_compressed', server_default=None)


def downgrade():
    """Remove is_compressed field from content table."""
    op.drop_column('content', 'is_compressed')