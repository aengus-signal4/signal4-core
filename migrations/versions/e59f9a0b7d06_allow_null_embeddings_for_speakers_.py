"""Allow NULL embeddings for speakers without centroids

Revision ID: e59f9a0b7d06
Revises: 634f20795101
Create Date: 2025-07-14 08:51:26.342256

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e59f9a0b7d06'
down_revision: Union[str, None] = '634f20795101'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Allow NULL embeddings for speakers without centroids
    op.alter_column('speakers', 'embedding', nullable=True)


def downgrade() -> None:
    # Revert to NOT NULL constraint
    # Note: This will fail if there are any NULL embeddings in the database
    op.alter_column('speakers', 'embedding', nullable=False)
