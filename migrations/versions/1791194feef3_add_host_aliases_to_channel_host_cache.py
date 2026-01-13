"""add host aliases to channel_host_cache

Revision ID: 1791194feef3
Revises: 91d90492b963
Create Date: 2025-11-30

Adds an 'aliases' column to channel_host_cache to store name variations
that should map to this canonical host name. This enables automatic
name normalization for new episodes without re-running LLM consolidation.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '1791194feef3'
down_revision: Union[str, None] = '91d90492b963'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add aliases column as TEXT[] for storing name variations
    op.add_column('channel_host_cache',
                  sa.Column('aliases', sa.ARRAY(sa.Text()), nullable=True, server_default='{}'))

    # Create GIN index for efficient alias lookups
    op.execute("""
        CREATE INDEX idx_channel_host_cache_aliases
        ON channel_host_cache USING GIN (aliases)
    """)


def downgrade() -> None:
    op.drop_index('idx_channel_host_cache_aliases', table_name='channel_host_cache')
    op.drop_column('channel_host_cache', 'aliases')
