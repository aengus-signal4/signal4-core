"""Add is_stitched and is_embedded flags to Content

Revision ID: a24611f36b04
Revises: ccd024fe31a8
Create Date: 2025-04-17 06:04:48.851730

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
# Remove unused import: from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'a24611f36b04'
down_revision: Union[str, None] = 'ccd024fe31a8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add the new columns to the 'content' table
    op.add_column('content', sa.Column('is_stitched', sa.Boolean(), nullable=False, server_default=sa.text('false')))
    op.add_column('content', sa.Column('is_embedded', sa.Boolean(), nullable=False, server_default=sa.text('false')))
    
    # Create indexes for the new columns (using explicit index names)
    op.create_index('idx_content_is_stitched', 'content', ['is_stitched'], unique=False)
    op.create_index('idx_content_is_embedded', 'content', ['is_embedded'], unique=False)
    
    # Ensure the index for is_diarized exists (might have been missed or removed)
    op.create_index('idx_content_is_diarized', 'content', ['is_diarized'], unique=False, if_not_exists=True)


def downgrade() -> None:
    # Drop the indexes first
    op.drop_index('idx_content_is_embedded', table_name='content')
    op.drop_index('idx_content_is_stitched', table_name='content')
    op.drop_index('idx_content_is_diarized', table_name='content') # Also drop the potentially added diarized index

    # Drop the columns
    op.drop_column('content', 'is_embedded')
    op.drop_column('content', 'is_stitched')
