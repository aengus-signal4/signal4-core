"""add speaker transcriptions table

Revision ID: add_speaker_transcriptions
Revises: add_speaker_metadata
Create Date: 2024-03-30 16:30:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision = 'add_speaker_transcriptions'
down_revision = 'add_speaker_metadata'
branch_labels = None
depends_on = None

def upgrade():
    # Create speaker_transcriptions table
    op.create_table('speaker_transcriptions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('content_id', sa.String(length=255), nullable=False),
        sa.Column('speaker_id', sa.Integer(), nullable=False),
        sa.Column('start_time', sa.Float(), nullable=False),
        sa.Column('end_time', sa.Float(), nullable=False),
        sa.Column('duration', sa.Float(), nullable=False),
        sa.Column('text', sa.Text(), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('turn_index', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['speaker_id'], ['public.speakers.id'], ),
        sa.PrimaryKeyConstraint('id'),
        schema='public'
    )

def downgrade():
    # Drop speaker_transcriptions table
    op.drop_table('speaker_transcriptions', schema='public') 