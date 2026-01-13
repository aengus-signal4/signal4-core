"""add queryable columns to hierarchical_summaries

Revision ID: 4312c63998ca
Revises: 96c62f34c42d
Create Date: 2025-11-06 15:59:49.809613

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '4312c63998ca'
down_revision: Union[str, None] = '96c62f34c42d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create hierarchical_summaries table
    op.create_table(
        'hierarchical_summaries',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('summary_type', sa.String(50), nullable=False),
        sa.Column('summary_id', sa.String(255), nullable=False),
        sa.Column('config_hash', sa.String(64), nullable=False),
        sa.Column('summary_data', sa.dialects.postgresql.JSONB(), nullable=False),
        sa.Column('time_window_days', sa.Integer(), nullable=True),
        sa.Column('groupings', sa.dialects.postgresql.JSONB(), nullable=True),
        sa.Column('num_themes', sa.Integer(), nullable=True),
        sa.Column('theme_discovery_method', sa.String(50), nullable=True),
        sa.Column('clustering_params', sa.dialects.postgresql.JSONB(), nullable=True),
        sa.Column('synthesis_type', sa.String(50), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=False)
    )

    # Create indexes
    op.create_index('idx_hierarchical_summary_config', 'hierarchical_summaries', ['config_hash', 'summary_type'])
    op.create_index('idx_hierarchical_summary_id', 'hierarchical_summaries', ['summary_id'], unique=True)
    op.create_index('idx_hierarchical_summary_created', 'hierarchical_summaries', ['created_at'])
    op.create_index('idx_hierarchical_summary_time_window', 'hierarchical_summaries', ['time_window_days'])


def downgrade() -> None:
    # Drop table (will also drop indexes)
    op.drop_table('hierarchical_summaries')
