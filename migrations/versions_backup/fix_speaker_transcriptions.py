"""fix speaker transcriptions table

Revision ID: fix_speaker_transcriptions
Revises: add_speaker_transcriptions
Create Date: 2024-03-30 16:35:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision = 'fix_speaker_transcriptions'
down_revision = 'add_speaker_transcriptions'
branch_labels = None
depends_on = None

def upgrade():
    # Drop existing table
    op.drop_table('speaker_transcriptions', schema='public')
    
    # Recreate table to match model exactly
    op.create_table('speaker_transcriptions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('content_id', sa.Integer(), nullable=False),
        sa.Column('speaker_id', sa.Integer(), nullable=False),
        sa.Column('start_time', sa.Float(), nullable=False),
        sa.Column('end_time', sa.Float(), nullable=False),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('turn_index', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['content_id'], ['public.content.id'], ),
        sa.ForeignKeyConstraint(['speaker_id'], ['public.speakers.id'], ),
        sa.PrimaryKeyConstraint('id'),
        schema='public'
    )

def downgrade():
    # Drop table
    op.drop_table('speaker_transcriptions', schema='public') 