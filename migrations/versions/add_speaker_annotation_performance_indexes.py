"""Add performance indexes for speaker annotation dashboard

Revision ID: add_speaker_annotation_perf
Revises: e59f9a0b7d06
Create Date: 2025-07-22 09:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'add_speaker_annotation_perf'
down_revision = 'e59f9a0b7d06'
branch_labels = None
depends_on = None


def upgrade():
    # Add composite index for speaker quality filtering
    op.create_index(
        'idx_speakers_quality_canonical',
        'speakers',
        ['is_canonical', 'embedding_quality_score', 'duration'],
        postgresql_where=sa.text("is_canonical = true AND embedding IS NOT NULL")
    )
    
    # Add index for canonical speaker relationship queries
    op.create_index(
        'idx_speakers_canonical_relationships',
        'speakers',
        ['id', 'canonical_speaker_id'],
        postgresql_where=sa.text("canonical_speaker_id IS NOT NULL")
    )


def downgrade():
    op.drop_index('idx_speakers_canonical_relationships', table_name='speakers')
    op.drop_index('idx_speakers_quality_canonical', table_name='speakers')