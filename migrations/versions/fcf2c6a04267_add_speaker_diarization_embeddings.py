"""add_speaker_diarization_embeddings

Revision ID: fcf2c6a04267
Revises: add_speaker_positions
Create Date: 2025-11-18 05:02:27.677540

Adds raw diarization embeddings to the speakers table to store embeddings
from FluidAudio (256-dim) alongside enriched stitch centroids (512-dim).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision: str = 'fcf2c6a04267'
down_revision: Union[str, None] = 'add_speaker_positions'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add raw diarization embedding fields
    op.add_column('speakers', sa.Column('embedding_diarization', Vector(256), nullable=True))
    op.add_column('speakers', sa.Column('embedding_diarization_quality', sa.Float(), nullable=True))

    # Note: Index will be created manually later to avoid overhead during population


def downgrade() -> None:
    # Drop columns
    op.drop_column('speakers', 'embedding_diarization_quality')
    op.drop_column('speakers', 'embedding_diarization')
