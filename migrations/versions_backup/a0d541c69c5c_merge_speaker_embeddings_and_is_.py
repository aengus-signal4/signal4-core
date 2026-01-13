"""Merge speaker embeddings and is_diarized branches

Revision ID: a0d541c69c5c
Revises: add_speaker_embedding_fields, add_is_diarized_column
Create Date: 2025-04-17 06:03:26.318252

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a0d541c69c5c'
down_revision: Union[str, None] = ('add_speaker_embedding_fields', 'add_is_diarized_column')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
