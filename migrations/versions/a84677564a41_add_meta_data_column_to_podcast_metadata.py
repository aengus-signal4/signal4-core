"""Add meta_data column to podcast_metadata

Revision ID: a84677564a41
Revises: e91c18c53674
Create Date: 2025-10-05 06:10:16.572734

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'a84677564a41'
down_revision: Union[str, None] = 'e91c18c53674'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add meta_data column to podcast_metadata table
    op.add_column('podcast_metadata',
        sa.Column('meta_data', postgresql.JSONB(astext_type=sa.Text()), server_default='{}', nullable=True)
    )


def downgrade() -> None:
    # Remove meta_data column
    op.drop_column('podcast_metadata', 'meta_data')
