"""add_translation_fields_to_alternative_transcriptions

Revision ID: 745fb16ce098
Revises: 3f1d051325c7
Create Date: 2025-11-13 05:57:07.467841

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '745fb16ce098'
down_revision: Union[str, None] = '3f1d051325c7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add translation_en and translation_fr columns to alternative_transcriptions table
    op.add_column('alternative_transcriptions', sa.Column('translation_en', sa.Text(), nullable=True))
    op.add_column('alternative_transcriptions', sa.Column('translation_fr', sa.Text(), nullable=True))


def downgrade() -> None:
    # Remove translation columns
    op.drop_column('alternative_transcriptions', 'translation_fr')
    op.drop_column('alternative_transcriptions', 'translation_en')
