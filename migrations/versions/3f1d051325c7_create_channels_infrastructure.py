"""create_channels_infrastructure

Revision ID: 3f1d051325c7
Revises: 7ed65f02fcbd
Create Date: 2025-11-11 11:36:00.000000

This migration creates the universal channels infrastructure to replace
platform-specific tables (starting with podcast_metadata). The channels
table supports all platforms (YouTube, podcasts, Rumble, etc.) with:
- Unified channel metadata
- Multi-platform source URLs (channel_sources)
- Project assignments (channel_projects)
- Platform-specific metadata in JSONB

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '3f1d051325c7'
down_revision: Union[str, None] = '7ed65f02fcbd'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create channels table
    op.create_table(
        'channels',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('channel_key', sa.String(length=500), nullable=False),
        sa.Column('display_name', sa.String(length=500), nullable=False),
        sa.Column('platform', sa.String(length=50), nullable=False),
        sa.Column('primary_url', sa.String(length=1000), nullable=False),
        sa.Column('language', sa.String(length=10), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=False, server_default='discovered'),
        sa.Column('tags', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='[]'),
        sa.Column('platform_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='{}'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('NOW()')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('NOW()')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('channel_key')
    )

    # Create indexes for channels
    op.create_index('idx_channels_platform', 'channels', ['platform'])
    op.create_index('idx_channels_status', 'channels', ['status'])
    op.create_index('idx_channels_platform_status', 'channels', ['platform', 'status'])
    op.create_index('idx_channels_primary_url', 'channels', ['primary_url'])

    # Create channel_sources table (for multi-platform creators)
    op.create_table(
        'channel_sources',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('channel_id', sa.Integer(), nullable=False),
        sa.Column('platform', sa.String(length=50), nullable=False),
        sa.Column('source_url', sa.String(length=1000), nullable=False),
        sa.Column('is_primary', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('NOW()')),
        sa.ForeignKeyConstraint(['channel_id'], ['channels.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('platform', 'source_url', name='uq_channel_sources_platform_url')
    )

    # Create indexes for channel_sources
    op.create_index('idx_channel_sources_channel_id', 'channel_sources', ['channel_id'])
    op.create_index('idx_channel_sources_url', 'channel_sources', ['source_url'])

    # Create channel_projects table (many-to-many)
    op.create_table(
        'channel_projects',
        sa.Column('channel_id', sa.Integer(), nullable=False),
        sa.Column('project_name', sa.String(length=100), nullable=False),
        sa.Column('added_at', sa.DateTime(), nullable=False, server_default=sa.text('NOW()')),
        sa.Column('added_by', sa.String(length=100), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['channel_id'], ['channels.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('channel_id', 'project_name')
    )

    # Create indexes for channel_projects
    op.create_index('idx_channel_projects_project', 'channel_projects', ['project_name'])

    # Add channel_id to content table
    op.add_column('content', sa.Column('channel_id', sa.Integer(), nullable=True))
    op.create_foreign_key('fk_content_channel_id', 'content', 'channels', ['channel_id'], ['id'])
    op.create_index('idx_content_channel_id', 'content', ['channel_id'])

    # Rename podcast_charts.podcast_id to channel_id for clarity
    # (We'll update the FK constraint after migration)
    op.alter_column('podcast_charts', 'podcast_id', new_column_name='channel_id')


def downgrade() -> None:
    # Revert podcast_charts column name
    op.alter_column('podcast_charts', 'channel_id', new_column_name='podcast_id')

    # Drop content.channel_id
    op.drop_index('idx_content_channel_id', 'content')
    op.drop_constraint('fk_content_channel_id', 'content', type_='foreignkey')
    op.drop_column('content', 'channel_id')

    # Drop channel_projects
    op.drop_index('idx_channel_projects_project', 'channel_projects')
    op.drop_table('channel_projects')

    # Drop channel_sources
    op.drop_index('idx_channel_sources_url', 'channel_sources')
    op.drop_index('idx_channel_sources_channel_id', 'channel_sources')
    op.drop_table('channel_sources')

    # Drop channels
    op.drop_index('idx_channels_primary_url', 'channels')
    op.drop_index('idx_channels_platform_status', 'channels')
    op.drop_index('idx_channels_status', 'channels')
    op.drop_index('idx_channels_platform', 'channels')
    op.drop_table('channels')
