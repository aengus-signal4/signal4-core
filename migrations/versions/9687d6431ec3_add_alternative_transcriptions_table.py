"""add_alternative_transcriptions_table

Revision ID: 9687d6431ec3
Revises: a84677564a41
Create Date: 2025-11-04 05:05:00

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '9687d6431ec3'
down_revision: Union[str, None] = 'a84677564a41'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create alternative_transcriptions table
    op.create_table(
        'alternative_transcriptions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('segment_id', sa.Integer(), nullable=False),
        sa.Column('content_id', sa.Integer(), nullable=False),
        sa.Column('provider', sa.String(length=50), nullable=False),
        sa.Column('model', sa.String(length=100), nullable=True),
        sa.Column('language', sa.String(length=10), nullable=True),
        sa.Column('transcription_text', sa.Text(), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('word_timings', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('speaker_labels', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('audio_duration', sa.Float(), nullable=True),
        sa.Column('processing_time', sa.Float(), nullable=True),
        sa.Column('meta_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('api_cost', sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(['content_id'], ['content.id'], ),
        sa.ForeignKeyConstraint(['segment_id'], ['embedding_segments.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('segment_id', 'provider', 'model', name='uq_alt_trans_segment_provider_model')
    )

    # Create indexes
    op.create_index('idx_alt_trans_segment_provider', 'alternative_transcriptions', ['segment_id', 'provider'], unique=False)
    op.create_index('idx_alt_trans_content_provider', 'alternative_transcriptions', ['content_id', 'provider'], unique=False)
    op.create_index('idx_alt_trans_created_at', 'alternative_transcriptions', ['created_at'], unique=False)
    op.create_index('ix_alternative_transcriptions_content_id', 'alternative_transcriptions', ['content_id'], unique=False)
    op.create_index('ix_alternative_transcriptions_provider', 'alternative_transcriptions', ['provider'], unique=False)
    op.create_index('ix_alternative_transcriptions_segment_id', 'alternative_transcriptions', ['segment_id'], unique=False)

    # Create GIN index for full-text search on transcription_text
    op.execute("""
        CREATE INDEX idx_alt_trans_text_search
        ON alternative_transcriptions
        USING gin(to_tsvector('simple', transcription_text))
    """)


def downgrade() -> None:
    # Drop indexes
    op.drop_index('idx_alt_trans_text_search', table_name='alternative_transcriptions')
    op.drop_index('ix_alternative_transcriptions_segment_id', table_name='alternative_transcriptions')
    op.drop_index('ix_alternative_transcriptions_provider', table_name='alternative_transcriptions')
    op.drop_index('ix_alternative_transcriptions_content_id', table_name='alternative_transcriptions')
    op.drop_index('idx_alt_trans_created_at', table_name='alternative_transcriptions')
    op.drop_index('idx_alt_trans_content_provider', table_name='alternative_transcriptions')
    op.drop_index('idx_alt_trans_segment_provider', table_name='alternative_transcriptions')

    # Drop table
    op.drop_table('alternative_transcriptions')
