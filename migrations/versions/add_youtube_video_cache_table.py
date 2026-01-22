"""Add youtube_video_cache table for episode matching

Revision ID: add_youtube_video_cache
Revises: a1b2c3d4e5f6, drop_speaker_transcriptions
Create Date: 2026-01-22

This migration adds a cache table for YouTube video metadata, used for
efficient podcast-to-YouTube episode matching. The table stores video
metadata from YouTube channels associated with podcasts, enabling
matching without live API calls.

Key features:
- Stores video metadata (title, description, duration, publish_date)
- Indexed for fast lookup by channel_id and fuzzy title search
- Completely isolated from download pipeline (no Content records created)
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'add_youtube_video_cache'
down_revision: Union[str, Sequence[str]] = ('a1b2c3d4e5f6', 'drop_speaker_transcriptions')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create youtube_video_cache table
    op.create_table('youtube_video_cache',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('youtube_channel_id', sa.String(length=50), nullable=False),
        sa.Column('video_id', sa.String(length=20), nullable=False),
        sa.Column('title', sa.Text(), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('publish_date', sa.DateTime(timezone=True), nullable=True),
        sa.Column('duration', sa.Integer(), nullable=True),
        sa.Column('view_count', sa.BigInteger(), nullable=True),
        sa.Column('thumbnail_url', sa.Text(), nullable=True),
        sa.Column('meta_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('indexed_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()'), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('youtube_channel_id', 'video_id', name='uq_youtube_video_cache_channel_video')
    )

    # Create indexes for efficient querying
    op.create_index('idx_yvc_channel_id', 'youtube_video_cache', ['youtube_channel_id'], unique=False)
    op.create_index('idx_yvc_publish_date', 'youtube_video_cache', ['publish_date'], unique=False)
    op.create_index('idx_yvc_video_id', 'youtube_video_cache', ['video_id'], unique=False)

    # Create trigram index for fuzzy title matching
    # First ensure the pg_trgm extension exists
    op.execute('CREATE EXTENSION IF NOT EXISTS pg_trgm')
    op.create_index('idx_yvc_title_trgm', 'youtube_video_cache', ['title'],
                    unique=False, postgresql_using='gin',
                    postgresql_ops={'title': 'gin_trgm_ops'})


def downgrade() -> None:
    op.drop_index('idx_yvc_title_trgm', table_name='youtube_video_cache')
    op.drop_index('idx_yvc_video_id', table_name='youtube_video_cache')
    op.drop_index('idx_yvc_publish_date', table_name='youtube_video_cache')
    op.drop_index('idx_yvc_channel_id', table_name='youtube_video_cache')
    op.drop_table('youtube_video_cache')
