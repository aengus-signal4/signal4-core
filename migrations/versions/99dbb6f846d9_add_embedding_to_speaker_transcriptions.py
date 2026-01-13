"""add embedding to speaker transcriptions

Revision ID: 99dbb6f846d9
Revises: a24611f36b04
Create Date: 2024-03-21 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = '99dbb6f846d9'
down_revision: Union[str, None] = 'a24611f36b04'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add embedding column to speaker_transcriptions table
    op.add_column('speaker_transcriptions', sa.Column('embedding', Vector(1024), nullable=True))


def downgrade() -> None:
    # Remove embedding column from speaker_transcriptions table
    op.drop_column('speaker_transcriptions', 'embedding')
