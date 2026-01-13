"""Add transcribed_with field to content_chunks

Revision ID: ff4d02fdde82
Revises: 828b05515b2c
Create Date: 2025-07-03 06:17:34.210069

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ff4d02fdde82'
down_revision: Union[str, None] = 'add_is_compressed_field'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add transcribed_with field to content_chunks table
    op.add_column('content_chunks', sa.Column('transcribed_with', sa.String(), nullable=True))


def downgrade() -> None:
    # Remove transcribed_with field from content_chunks table
    op.drop_column('content_chunks', 'transcribed_with')
