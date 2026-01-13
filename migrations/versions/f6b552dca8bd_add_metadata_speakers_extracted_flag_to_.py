"""add metadata_speakers_extracted flag to content table

Revision ID: f6b552dca8bd
Revises: cd547589272a
Create Date: 2025-11-29 04:52:10.062847

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f6b552dca8bd'
down_revision: Union[str, None] = 'cd547589272a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add boolean flag to track if Phase 1 metadata speaker extraction has been run
    op.add_column('content', sa.Column('metadata_speakers_extracted', sa.Boolean(), nullable=True, server_default='false'))

    # Backfill: mark episodes that already have hosts/guests/mentioned as extracted
    op.execute("""
        UPDATE content
        SET metadata_speakers_extracted = true
        WHERE (hosts IS NOT NULL AND hosts != '[]'::jsonb)
           OR (guests IS NOT NULL AND guests != '[]'::jsonb)
           OR (mentioned IS NOT NULL AND mentioned != '[]'::jsonb)
    """)


def downgrade() -> None:
    op.drop_column('content', 'metadata_speakers_extracted')
