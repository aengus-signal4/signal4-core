"""Add main_language field to content table

Revision ID: 337b8fb68502
Revises: ff4d02fdde82
Create Date: 2025-07-03 07:33:40.024864

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '337b8fb68502'
down_revision: Union[str, None] = 'ff4d02fdde82'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add main_language column to content table
    op.add_column('content', sa.Column('main_language', sa.String(length=10), nullable=True))
    
    # Create index on main_language
    op.create_index('idx_content_main_language', 'content', ['main_language'])


def downgrade() -> None:
    # Drop index and column
    op.drop_index('idx_content_main_language', table_name='content')
    op.drop_column('content', 'main_language')
