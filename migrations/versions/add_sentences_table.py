"""Add sentences table and emotion support

Creates the sentences table as the new atomic unit for transcript data,
replacing speaker_transcriptions with sentence-level granularity.
Also adds emotion support columns to embedding_segments.

Sentences enable:
- Granular emotion queries ("what did speaker X say when angry")
- Efficient segment retrieval with emotion filtering
- Clean data model where sentences are the atomic unit

Revision ID: add_sentences_table
Revises: refactor_speaker_id_schema
Create Date: 2025-12-16
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, ARRAY

# revision identifiers
revision = 'add_sentences_table'
down_revision = 'refactor_speaker_id_schema'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create sentences table
    op.create_table(
        'sentences',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('content_id', sa.Integer(), nullable=False),
        sa.Column('speaker_id', sa.Integer(), nullable=False),

        # Position indices
        sa.Column('sentence_index', sa.Integer(), nullable=False),      # global index within content
        sa.Column('turn_index', sa.Integer(), nullable=False),          # speaker turn this belongs to
        sa.Column('sentence_in_turn', sa.Integer(), nullable=False),    # position within turn

        # Text & timing
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('start_time', sa.Float(), nullable=False),
        sa.Column('end_time', sa.Float(), nullable=False),
        sa.Column('word_count', sa.Integer(), nullable=False),

        # Emotion (nullable until emotion stage runs)
        sa.Column('emotion', sa.String(30), nullable=True),
        sa.Column('emotion_confidence', sa.Float(), nullable=True),
        sa.Column('emotion_scores', JSONB(), nullable=True),
        sa.Column('arousal', sa.Float(), nullable=True),
        sa.Column('valence', sa.Float(), nullable=True),
        sa.Column('dominance', sa.Float(), nullable=True),

        # Metadata
        sa.Column('stitch_version', sa.String(50), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('NOW()'), nullable=False),

        # Constraints
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['content_id'], ['content.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['speaker_id'], ['speakers.id'], ondelete='CASCADE'),
        sa.UniqueConstraint('content_id', 'sentence_index', name='uq_sentence_content_index')
    )

    # Create indexes for sentences table
    op.create_index('idx_sentences_content', 'sentences', ['content_id'])
    op.create_index('idx_sentences_speaker', 'sentences', ['speaker_id'])
    op.create_index('idx_sentences_turn', 'sentences', ['content_id', 'turn_index'])
    op.create_index('idx_sentences_time', 'sentences', ['content_id', 'start_time', 'end_time'])

    # Partial index for emotion queries (only sentences with detected emotions)
    op.create_index(
        'idx_sentences_emotion',
        'sentences',
        ['emotion'],
        postgresql_where=sa.text('emotion IS NOT NULL')
    )

    # Composite index for common query pattern: find emotional sentences by speaker
    op.create_index(
        'idx_sentences_speaker_emotion',
        'sentences',
        ['speaker_id', 'emotion'],
        postgresql_where=sa.text('emotion IS NOT NULL')
    )

    # Add source_sentence_ids to embedding_segments
    op.add_column(
        'embedding_segments',
        sa.Column('source_sentence_ids', ARRAY(sa.Integer()), nullable=True)
    )

    # Add emotion_summary to embedding_segments
    op.add_column(
        'embedding_segments',
        sa.Column('emotion_summary', JSONB(), nullable=True)
    )

    # Create GIN indexes for the new array and JSONB columns
    op.create_index(
        'idx_segments_sentence_ids',
        'embedding_segments',
        ['source_sentence_ids'],
        postgresql_using='gin'
    )

    op.create_index(
        'idx_segments_emotion_summary',
        'embedding_segments',
        ['emotion_summary'],
        postgresql_using='gin'
    )


def downgrade() -> None:
    # Drop indexes first
    op.drop_index('idx_segments_emotion_summary', table_name='embedding_segments')
    op.drop_index('idx_segments_sentence_ids', table_name='embedding_segments')

    # Drop columns from embedding_segments
    op.drop_column('embedding_segments', 'emotion_summary')
    op.drop_column('embedding_segments', 'source_sentence_ids')

    # Drop sentences table indexes
    op.drop_index('idx_sentences_speaker_emotion', table_name='sentences')
    op.drop_index('idx_sentences_emotion', table_name='sentences')
    op.drop_index('idx_sentences_time', table_name='sentences')
    op.drop_index('idx_sentences_turn', table_name='sentences')
    op.drop_index('idx_sentences_speaker', table_name='sentences')
    op.drop_index('idx_sentences_content', table_name='sentences')

    # Drop sentences table
    op.drop_table('sentences')
