"""Drop speaker_transcriptions table

This migration removes the deprecated speaker_transcriptions table.
The table has been superseded by the sentences table.

Migration rationale (January 2026):
- All 424,570 stitched content items now have sentences
- The sentences table has 240M rows (vs 26M in speaker_transcriptions)
- Speaker turns can be reconstructed from sentences by grouping on (content_id, turn_index)
- The source_transcription_ids column in embedding_segments is deprecated
  (replaced by source_sentence_ids)

Legacy data note:
- 348,579 content items had speaker_transcriptions (all migrated to sentences)
- 75,992+ newer content items only have sentences (never had speaker_transcriptions)

S3 cleanup note:
- Some content may still have speaker_turns.json files in S3 at:
  content/{content_id}/speaker_turns.json
- These are safe to delete after confirming sentences exist
- The word_table.pkl.gz files are the authoritative source for re-migration

Revision ID: drop_speaker_transcriptions
Revises: (will be set by alembic)
Create Date: 2026-01-21
"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'drop_speaker_transcriptions'
down_revision = 'upgrade_api_key_hash_to_bcrypt'  # After the API key bcrypt upgrade
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Drop indexes first using IF EXISTS to avoid errors
    indexes_to_drop = [
        'idx_speaker_transcriptions_content_id',
        'idx_speaker_transcriptions_speaker_id',
        'idx_speaker_transcriptions_start_time',
        'idx_speaker_transcriptions_end_time',
        'idx_speaker_transcriptions_text',
        'idx_speaker_transcriptions_turn_index',
        'idx_speaker_transcriptions_content_speaker',
        'idx_speaker_transcriptions_speaker_content_times',
        # Legacy index names from older migrations
        'idx_speaker_transcription_content',
        'idx_speaker_transcription_speaker',
        'idx_speaker_transcription_time',
        'idx_speaker_transcription_turn_index',
        'idx_speaker_transcription_text',
    ]

    for index_name in indexes_to_drop:
        op.execute(f'DROP INDEX IF EXISTS {index_name}')

    # Drop the table
    op.drop_table('speaker_transcriptions')

    # Drop orphaned functions that referenced speaker_transcriptions
    op.execute('DROP FUNCTION IF EXISTS update_speaker_stats() CASCADE')
    op.execute('DROP FUNCTION IF EXISTS recalculate_all_speaker_stats() CASCADE')

    # Also drop the source_transcription_ids column from embedding_segments
    # since it's no longer used (source_sentence_ids is the replacement)
    op.drop_index('idx_embedding_segments_source_ids', table_name='embedding_segments')
    op.drop_column('embedding_segments', 'source_transcription_ids')


def downgrade() -> None:
    # Re-create the source_transcription_ids column
    op.add_column('embedding_segments', sa.Column('source_transcription_ids', sa.ARRAY(sa.Integer()), nullable=True))
    op.create_index('idx_embedding_segments_source_ids', 'embedding_segments', ['source_transcription_ids'], postgresql_using='gin')

    # Re-create speaker_transcriptions table
    op.create_table(
        'speaker_transcriptions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('content_id', sa.Integer(), nullable=False),
        sa.Column('speaker_id', sa.Integer(), nullable=False),
        sa.Column('speaker_hash', sa.String(8), nullable=True),
        sa.Column('start_time', sa.Float(), nullable=False),
        sa.Column('end_time', sa.Float(), nullable=False),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('turn_index', sa.Integer(), nullable=False),
        sa.Column('stitch_version', sa.String(50), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['content_id'], ['content.id']),
        sa.ForeignKeyConstraint(['speaker_id'], ['speakers.id']),
    )

    # Re-create indexes
    op.create_index('idx_speaker_transcriptions_content_id', 'speaker_transcriptions', ['content_id'])
    op.create_index('idx_speaker_transcriptions_speaker_id', 'speaker_transcriptions', ['speaker_id'])
    op.create_index('idx_speaker_transcriptions_start_time', 'speaker_transcriptions', ['start_time'])
    op.create_index('idx_speaker_transcriptions_end_time', 'speaker_transcriptions', ['end_time'])
    op.create_index('idx_speaker_transcriptions_turn_index', 'speaker_transcriptions', ['turn_index'])
    op.create_index('idx_speaker_transcriptions_content_speaker', 'speaker_transcriptions', ['content_id', 'speaker_id'])
    op.create_index('idx_speaker_transcriptions_speaker_content_times', 'speaker_transcriptions', ['speaker_id', 'content_id', 'start_time', 'end_time'])
