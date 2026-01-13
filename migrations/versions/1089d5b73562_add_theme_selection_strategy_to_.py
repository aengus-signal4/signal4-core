"""add_theme_selection_strategy_to_hierarchical_summaries

Revision ID: 1089d5b73562
Revises: 4312c63998ca
Create Date: 2025-11-07 05:02:28.866150

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '1089d5b73562'
down_revision: Union[str, None] = '4312c63998ca'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add theme_selection_strategy column to hierarchical_summaries
    op.add_column('hierarchical_summaries',
                  sa.Column('theme_selection_strategy', sa.String(50), nullable=True))


def downgrade() -> None:
    # Remove theme_selection_strategy column
    op.drop_column('hierarchical_summaries', 'theme_selection_strategy')
