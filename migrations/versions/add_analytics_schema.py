"""Add analytics schema for user query logging

Creates a separate 'analytics' schema to logically isolate user behavior
tracking from content processing data.

Tables:
- analytics.users: Minimal user table synced from frontend config
- analytics.user_queries: Log of all queries sent to the backend

Design decisions:
- Uses separate schema to avoid cluttering the content namespace
- Users identified by client_id (matches frontend auth system)
- Queries logged asynchronously to avoid blocking responses
- JSONB for flexible metadata storage (filters, config, etc.)
- Indexes optimized for analytics queries (by user, by time, by dashboard)

Expected scale: ~500 users, ~25K queries/day

Revision ID: add_analytics_schema
Revises: add_processing_state
Create Date: 2026-01-19
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'add_analytics_schema'
down_revision: Union[str, None] = 'add_processing_state'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create analytics schema
    op.execute('CREATE SCHEMA IF NOT EXISTS analytics')

    # Create users table in analytics schema
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('client_id', sa.String(100), nullable=False),
        sa.Column('email', sa.String(255), nullable=False),
        sa.Column('name', sa.String(255), nullable=True),
        sa.Column('first_seen', sa.DateTime(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        sa.Column('last_seen', sa.DateTime(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        sa.Column('query_count', sa.Integer(), server_default='0', nullable=False),
        sa.Column('meta_data', postgresql.JSONB(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('client_id', name='uq_analytics_users_client_id'),
        sa.UniqueConstraint('email', name='uq_analytics_users_email'),
        schema='analytics'
    )

    # Index for user lookups
    op.create_index(
        'idx_analytics_users_client_id',
        'users',
        ['client_id'],
        unique=True,
        schema='analytics'
    )

    # Create user_queries table in analytics schema
    op.create_table(
        'user_queries',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=True),  # Nullable for anonymous/failed lookups
        sa.Column('client_id', sa.String(100), nullable=True),  # Store even if user lookup fails
        sa.Column('dashboard_id', sa.String(100), nullable=False),
        sa.Column('query_text', sa.Text(), nullable=True),  # Null for discovery workflows
        sa.Column('workflow', sa.String(100), nullable=True),
        sa.Column('time_window_days', sa.Integer(), nullable=True),
        sa.Column('filters', postgresql.JSONB(), nullable=True),  # projects, languages, channels
        sa.Column('cache_hit', sa.Boolean(), nullable=True),
        sa.Column('cache_level', sa.String(50), nullable=True),  # 'full', 'partial', 'miss'
        sa.Column('response_time_ms', sa.Integer(), nullable=True),
        sa.Column('segment_count', sa.Integer(), nullable=True),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        sa.Column('meta_data', postgresql.JSONB(), nullable=True),  # Additional context
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(
            ['user_id'],
            ['analytics.users.id'],
            name='fk_user_queries_user_id',
            ondelete='SET NULL'
        ),
        schema='analytics'
    )

    # Index for time-based queries (most common analytics pattern)
    op.create_index(
        'idx_analytics_queries_created_at',
        'user_queries',
        ['created_at'],
        unique=False,
        schema='analytics'
    )

    # Index for user-specific queries
    op.create_index(
        'idx_analytics_queries_user_id',
        'user_queries',
        ['user_id'],
        unique=False,
        schema='analytics'
    )

    # Index for dashboard analytics
    op.create_index(
        'idx_analytics_queries_dashboard',
        'user_queries',
        ['dashboard_id', 'created_at'],
        unique=False,
        schema='analytics'
    )

    # Composite index for user + time queries
    op.create_index(
        'idx_analytics_queries_user_time',
        'user_queries',
        ['user_id', 'created_at'],
        unique=False,
        schema='analytics'
    )

    # Index for cache analytics
    op.create_index(
        'idx_analytics_queries_cache',
        'user_queries',
        ['cache_hit', 'created_at'],
        unique=False,
        schema='analytics'
    )


def downgrade() -> None:
    # Drop indexes
    op.drop_index('idx_analytics_queries_cache', table_name='user_queries', schema='analytics')
    op.drop_index('idx_analytics_queries_user_time', table_name='user_queries', schema='analytics')
    op.drop_index('idx_analytics_queries_dashboard', table_name='user_queries', schema='analytics')
    op.drop_index('idx_analytics_queries_user_id', table_name='user_queries', schema='analytics')
    op.drop_index('idx_analytics_queries_created_at', table_name='user_queries', schema='analytics')
    op.drop_index('idx_analytics_users_client_id', table_name='users', schema='analytics')

    # Drop tables
    op.drop_table('user_queries', schema='analytics')
    op.drop_table('users', schema='analytics')

    # Drop schema
    op.execute('DROP SCHEMA IF EXISTS analytics')
