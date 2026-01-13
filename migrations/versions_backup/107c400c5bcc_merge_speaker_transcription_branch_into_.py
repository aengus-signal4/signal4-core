"""Merge speaker transcription branch into main line

Revision ID: 107c400c5bcc
Revises: a0d541c69c5c, fix_speaker_transcriptions
Create Date: 2025-04-17 06:03:55.909488

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '107c400c5bcc'
down_revision: Union[str, None] = ('a0d541c69c5c', 'fix_speaker_transcriptions')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
