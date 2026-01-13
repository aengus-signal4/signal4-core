"""merge_heads

Revision ID: a1abbc61329a
Revises: add_processor_task_id, add_speaker_embeddings
Create Date: 2025-05-23 12:22:33.875834

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1abbc61329a'
down_revision: Union[str, None] = ('add_processor_task_id', 'add_speaker_embeddings')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
