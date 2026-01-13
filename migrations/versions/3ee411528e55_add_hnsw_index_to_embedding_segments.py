"""add_hnsw_index_to_embedding_segments

Revision ID: 3ee411528e55
Revises: 0fe3e6c128fd
Create Date: 2025-09-17 06:01:38.866886

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3ee411528e55'
down_revision: Union[str, None] = '0fe3e6c128fd'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # HNSW index creation removed - not needed for this migration
    pass


def downgrade() -> None:
    # No index to drop
    pass
