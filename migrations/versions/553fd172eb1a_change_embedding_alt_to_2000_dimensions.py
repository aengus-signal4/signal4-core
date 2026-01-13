"""Change embedding_alt to 2000 dimensions

Revision ID: 553fd172eb1a
Revises: 3ee411528e55
Create Date: 2025-09-25 06:08:51.297700

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision: str = '553fd172eb1a'
down_revision: Union[str, None] = '3ee411528e55'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop existing embedding_alt column (2560 dimensions)
    op.drop_column('embedding_segments', 'embedding_alt')
    
    # Add new embedding_alt column with 2000 dimensions
    op.add_column('embedding_segments', sa.Column('embedding_alt', Vector(2000), nullable=True))


def downgrade() -> None:
    # Drop 2000-dimension embedding_alt column
    op.drop_column('embedding_segments', 'embedding_alt')
    
    # Restore original embedding_alt column with 2560 dimensions
    op.add_column('embedding_segments', sa.Column('embedding_alt', Vector(2560), nullable=True))
