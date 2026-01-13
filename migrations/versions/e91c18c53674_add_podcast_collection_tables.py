"""Add podcast collection tables

Revision ID: e91c18c53674
Revises: 99076c7ceb81
Create Date: 2025-10-04 07:05:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'e91c18c53674'
down_revision: Union[str, None] = '99076c7ceb81'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create podcast_metadata table
    op.create_table('podcast_metadata',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('podcast_name', sa.String(length=500), nullable=False),
        sa.Column('podcast_key', sa.String(length=500), nullable=False),
        sa.Column('creator', sa.String(length=500), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('rss_url', sa.String(length=1000), nullable=True),
        sa.Column('language', sa.String(length=10), nullable=True),
        sa.Column('categories', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('episode_count', sa.Integer(), nullable=True),
        sa.Column('podcast_index_id', sa.String(length=50), nullable=True),
        sa.Column('monthly_rankings', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('first_seen', sa.DateTime(), nullable=False),
        sa.Column('last_updated', sa.DateTime(), nullable=False),
        sa.Column('last_enriched', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('podcast_key')
    )
    op.create_index('idx_podcast_metadata_creator', 'podcast_metadata', ['creator'], unique=False)
    op.create_index('idx_podcast_metadata_key', 'podcast_metadata', ['podcast_key'], unique=False)
    op.create_index('idx_podcast_metadata_last_enriched', 'podcast_metadata', ['last_enriched'], unique=False)
    op.create_index('idx_podcast_metadata_name', 'podcast_metadata', ['podcast_name'], unique=False)
    op.create_index('idx_podcast_metadata_rankings', 'podcast_metadata', ['monthly_rankings'], unique=False, postgresql_using='gin')

    # Create podcast_charts table
    op.create_table('podcast_charts',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('podcast_id', sa.Integer(), nullable=False),
        sa.Column('month', sa.String(length=7), nullable=False),
        sa.Column('platform', sa.String(length=20), nullable=False),
        sa.Column('country', sa.String(length=10), nullable=False),
        sa.Column('category', sa.String(length=100), nullable=False),
        sa.Column('rank', sa.Integer(), nullable=False),
        sa.Column('chart_key', sa.String(length=150), nullable=False),
        sa.Column('collected_at', sa.DateTime(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['podcast_id'], ['podcast_metadata.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('podcast_id', 'month', 'chart_key', name='uq_podcast_chart_month')
    )
    op.create_index('idx_podcast_charts_chart_key_month', 'podcast_charts', ['chart_key', 'month'], unique=False)
    op.create_index('idx_podcast_charts_month_country', 'podcast_charts', ['month', 'country'], unique=False)
    op.create_index('idx_podcast_charts_month_platform', 'podcast_charts', ['month', 'platform'], unique=False)
    op.create_index('idx_podcast_charts_platform_country_category', 'podcast_charts', ['platform', 'country', 'category'], unique=False)
    op.create_index(op.f('ix_podcast_charts_category'), 'podcast_charts', ['category'], unique=False)
    op.create_index(op.f('ix_podcast_charts_chart_key'), 'podcast_charts', ['chart_key'], unique=False)
    op.create_index(op.f('ix_podcast_charts_country'), 'podcast_charts', ['country'], unique=False)
    op.create_index(op.f('ix_podcast_charts_month'), 'podcast_charts', ['month'], unique=False)
    op.create_index(op.f('ix_podcast_charts_platform'), 'podcast_charts', ['platform'], unique=False)
    op.create_index(op.f('ix_podcast_charts_podcast_id'), 'podcast_charts', ['podcast_id'], unique=False)


def downgrade() -> None:
    # Drop podcast_charts table
    op.drop_index(op.f('ix_podcast_charts_podcast_id'), table_name='podcast_charts')
    op.drop_index(op.f('ix_podcast_charts_platform'), table_name='podcast_charts')
    op.drop_index(op.f('ix_podcast_charts_month'), table_name='podcast_charts')
    op.drop_index(op.f('ix_podcast_charts_country'), table_name='podcast_charts')
    op.drop_index(op.f('ix_podcast_charts_chart_key'), table_name='podcast_charts')
    op.drop_index(op.f('ix_podcast_charts_category'), table_name='podcast_charts')
    op.drop_index('idx_podcast_charts_platform_country_category', table_name='podcast_charts')
    op.drop_index('idx_podcast_charts_month_platform', table_name='podcast_charts')
    op.drop_index('idx_podcast_charts_month_country', table_name='podcast_charts')
    op.drop_index('idx_podcast_charts_chart_key_month', table_name='podcast_charts')
    op.drop_table('podcast_charts')

    # Drop podcast_metadata table
    op.drop_index('idx_podcast_metadata_rankings', table_name='podcast_metadata')
    op.drop_index('idx_podcast_metadata_name', table_name='podcast_metadata')
    op.drop_index('idx_podcast_metadata_last_enriched', table_name='podcast_metadata')
    op.drop_index('idx_podcast_metadata_key', table_name='podcast_metadata')
    op.drop_index('idx_podcast_metadata_creator', table_name='podcast_metadata')
    op.drop_table('podcast_metadata')
