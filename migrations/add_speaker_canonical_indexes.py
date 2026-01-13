"""Add indexes for canonical speaker lookups

Revision ID: add_speaker_canonical_indexes
Revises: 
Create Date: 2025-08-21

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = 'add_speaker_canonical_indexes'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    """Add indexes to improve speaker assignment performance"""
    
    # Add index on canonical_speaker_id for fast lookups
    op.create_index(
        'idx_speakers_canonical_speaker_id',
        'speakers',
        ['canonical_speaker_id'],
        if_not_exists=True
    )
    
    # Add composite index for canonical speaker lookups with display_name
    # Note: idx_speakers_canonical_is_canonical already exists in models.py
    op.create_index(
        'idx_speakers_canonical_display',
        'speakers',
        ['canonical_speaker_id', 'is_canonical', 'display_name'],
        if_not_exists=True
    )
    
    # Add index on display_name for filtering speakers without names
    op.create_index(
        'idx_speakers_display_name',
        'speakers',
        ['display_name'],
        if_not_exists=True
    )
    
    # Add composite index for the speaker assignment query pattern
    # This helps when joining speakers -> speaker_transcriptions -> content
    op.create_index(
        'idx_speakers_for_assignment',
        'speakers',
        ['id', 'display_name', 'canonical_speaker_id'],
        if_not_exists=True
    )
    
    # Add index on speaker_transcriptions for the join
    op.create_index(
        'idx_speaker_transcriptions_speaker_id_content',
        'speaker_transcriptions',
        ['speaker_id', 'content_id'],
        if_not_exists=True
    )
    
    print("✓ Added canonical speaker indexes for improved query performance")


def downgrade():
    """Remove the indexes"""
    op.drop_index('idx_speakers_canonical_speaker_id', 'speakers', if_exists=True)
    op.drop_index('idx_speakers_canonical_display', 'speakers', if_exists=True)
    op.drop_index('idx_speakers_display_name', 'speakers', if_exists=True)
    op.drop_index('idx_speakers_for_assignment', 'speakers', if_exists=True)
    op.drop_index('idx_speaker_transcriptions_speaker_id_content', 'speaker_transcriptions', if_exists=True)