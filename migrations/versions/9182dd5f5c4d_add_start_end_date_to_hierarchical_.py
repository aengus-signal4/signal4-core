"""add_start_end_date_to_hierarchical_summary

Revision ID: 9182dd5f5c4d
Revises: 1089d5b73562
Create Date: 2025-11-07 05:16:17.712929

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9182dd5f5c4d'
down_revision: Union[str, None] = '1089d5b73562'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add start_date and end_date columns to hierarchical_summaries table
    op.add_column('hierarchical_summaries', sa.Column('start_date', sa.DateTime(timezone=True), nullable=True))
    op.add_column('hierarchical_summaries', sa.Column('end_date', sa.DateTime(timezone=True), nullable=True))

    # Add indexes for the new columns
    op.create_index('idx_hierarchical_summary_start_date', 'hierarchical_summaries', ['start_date'])
    op.create_index('idx_hierarchical_summary_end_date', 'hierarchical_summaries', ['end_date'])


def downgrade() -> None:
    # Remove indexes
    op.drop_index('idx_hierarchical_summary_end_date', table_name='hierarchical_summaries')
    op.drop_index('idx_hierarchical_summary_start_date', table_name='hierarchical_summaries')

    # Remove columns
    op.drop_column('hierarchical_summaries', 'end_date')
    op.drop_column('hierarchical_summaries', 'start_date')
