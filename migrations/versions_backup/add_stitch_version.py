"""add stitch version

Revision ID: add_stitch_version
Revises: a24611f36b04
Create Date: 2024-04-17 06:04:48.851730

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'add_stitch_version'
down_revision: Union[str, None] = 'a24611f36b04'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add stitch_version column to content table
    op.add_column('content', sa.Column('stitch_version', sa.String(), nullable=True))
    
    # Update existing records that are stitched to use stitch_v1
    op.execute("UPDATE content SET stitch_version = 'stitch_v1' WHERE is_stitched = true")
    
    # Make the column not nullable after setting default value
    op.alter_column('content', 'stitch_version',
                    existing_type=sa.String(),
                    nullable=False,
                    server_default='stitch_v1')
    
    # Create index for stitch_version
    op.create_index('idx_content_stitch_version', 'content', ['stitch_version'])


def downgrade() -> None:
    # Drop the index
    op.drop_index('idx_content_stitch_version', table_name='content')
    
    # Drop the column
    op.drop_column('content', 'stitch_version') 