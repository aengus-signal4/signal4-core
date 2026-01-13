"""add_speaker_positions_to_embedding_segments

Revision ID: add_speaker_positions
Revises: 745fb16ce098
Create Date: 2025-11-16 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'add_speaker_positions'
down_revision: Union[str, None] = '745fb16ce098'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Add speaker_positions JSONB column to embedding_segments.

    This column stores compact speaker position information as:
    {
        "SPEAKER_00": [[0, 280]],
        "SPEAKER_01": [[281, 315], [650, 750]],
        "SPEAKER_02": [[316, 649]]
    }

    Where each speaker maps to a list of [start_char, end_char] ranges.
    This allows reconstruction of speaker-attributed text without storing duplicate text.
    """
    op.add_column('embedding_segments',
                  sa.Column('speaker_positions', postgresql.JSONB(astext_type=sa.Text()), nullable=True))

    # Create GIN index for fast speaker lookup (e.g., "which segments has SPEAKER_00?")
    op.create_index('idx_embedding_segments_speaker_positions',
                    'embedding_segments',
                    ['speaker_positions'],
                    unique=False,
                    postgresql_using='gin')


def downgrade() -> None:
    """Remove speaker_positions column and index."""
    op.drop_index('idx_embedding_segments_speaker_positions', table_name='embedding_segments', postgresql_using='gin')
    op.drop_column('embedding_segments', 'speaker_positions')
