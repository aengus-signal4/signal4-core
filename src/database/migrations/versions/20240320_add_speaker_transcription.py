"""Add speaker transcription model and flag

Revision ID: 20240320_add_speaker_transcription
Revises: 20240319_add_speaker_models
Create Date: 2024-03-20 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '20240320_add_speaker_transcription'
down_revision = '20240319_add_speaker_models'
branch_labels = None
depends_on = None

def upgrade():
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
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['content_id'], ['content.id'], ),
        sa.ForeignKeyConstraint(['speaker_id'], ['speakers.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes
    op.create_index('idx_speaker_transcription_content', 'speaker_transcriptions', ['content_id'])
    op.create_index('idx_speaker_transcription_speaker', 'speaker_transcriptions', ['speaker_id'])
    op.create_index('idx_speaker_transcription_time', 'speaker_transcriptions', ['start_time', 'end_time'])
    op.create_index('idx_speaker_transcription_turn_index', 'speaker_transcriptions', ['content_id', 'turn_index'])
    
    # Create GIN index for text search
    op.execute('CREATE INDEX idx_speaker_transcription_text ON speaker_transcriptions USING gin (to_tsvector(\'english\', text))')

def downgrade():
    # Drop indexes
    op.drop_index('idx_speaker_transcription_text')
    op.drop_index('idx_speaker_transcription_turn_index')
    op.drop_index('idx_speaker_transcription_time')
    op.drop_index('idx_speaker_transcription_speaker')
    op.drop_index('idx_speaker_transcription_content')
    
    # Drop speaker_transcriptions table
    op.drop_table('speaker_transcriptions') 