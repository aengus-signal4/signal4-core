"""create speaker tables

Revision ID: create_speaker_tables
Revises: add_speaker_metadata
Create Date: 2024-03-30 16:30:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision = 'create_speaker_tables'
down_revision = 'add_speaker_metadata'
branch_labels = None
depends_on = None

def upgrade():
    # Create speakers table
    op.create_table(
        'speakers',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('global_id', sa.String(), nullable=False),
        sa.Column('universal_name', sa.String(), nullable=False),
        sa.Column('display_name', sa.String(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.Column('total_segments', sa.Integer(), nullable=True),
        sa.Column('total_duration', sa.Float(), nullable=True),
        sa.Column('last_seen', sa.DateTime(), nullable=True),
        sa.Column('last_content_id', sa.String(), nullable=True),
        sa.Column('appearance_count', sa.Integer(), nullable=True),
        sa.Column('meta_data', JSONB, server_default='{}', nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('global_id'),
        sa.UniqueConstraint('universal_name')
    )

    # Create speaker_embeddings table
    op.create_table(
        'speaker_embeddings',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('speaker_id', sa.Integer(), nullable=False),
        sa.Column('content_id', sa.String(), nullable=False),
        sa.Column('embedding', sa.LargeBinary(), nullable=False),
        sa.Column('segment_count', sa.Integer(), nullable=False),
        sa.Column('total_duration', sa.Float(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['speaker_id'], ['speakers.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    # Create speaker_transcriptions table
    op.create_table(
        'speaker_transcriptions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('content_id', sa.Integer(), nullable=False),
        sa.Column('speaker_id', sa.Integer(), nullable=False),
        sa.Column('start_time', sa.Float(), nullable=False),
        sa.Column('end_time', sa.Float(), nullable=False),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('turn_index', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['content_id'], ['content.id'], ),
        sa.ForeignKeyConstraint(['speaker_id'], ['speakers.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    # Create indexes
    op.create_index('idx_speaker_transcription_content', 'speaker_transcriptions', ['content_id'])
    op.create_index('idx_speaker_transcription_speaker', 'speaker_transcriptions', ['speaker_id'])
    op.create_index('idx_speaker_transcription_time', 'speaker_transcriptions', ['start_time', 'end_time'])
    op.create_index('idx_speaker_transcription_turn_index', 'speaker_transcriptions', ['content_id', 'turn_index'])
    op.create_index('idx_speaker_transcription_text', 'speaker_transcriptions', ['text'], postgresql_using='gin', postgresql_ops={'text': 'gin_trgm_ops'})

def downgrade():
    # Drop indexes first
    op.drop_index('idx_speaker_transcription_text')
    op.drop_index('idx_speaker_transcription_turn_index')
    op.drop_index('idx_speaker_transcription_time')
    op.drop_index('idx_speaker_transcription_speaker')
    op.drop_index('idx_speaker_transcription_content')

    # Drop tables
    op.drop_table('speaker_transcriptions')
    op.drop_table('speaker_embeddings')
    op.drop_table('speakers') 