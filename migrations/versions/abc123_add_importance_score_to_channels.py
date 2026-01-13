"""add importance_score to channels

Revision ID: abc123_importance
Revises: 4da7589f1316
Create Date: 2025-12-31

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'abc123_importance'
down_revision: Union[str, None] = '4da7589f1316'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add importance_score column to channels table
    op.add_column('channels', sa.Column('importance_score', sa.Float(), nullable=True))

    # Create index for efficient sorting by importance_score (DESC, NULLS LAST)
    op.create_index(
        'idx_channels_importance_score',
        'channels',
        ['importance_score'],
        unique=False,
        postgresql_using='btree'
    )


def downgrade() -> None:
    op.drop_index('idx_channels_importance_score', table_name='channels')
    op.drop_column('channels', 'importance_score')
