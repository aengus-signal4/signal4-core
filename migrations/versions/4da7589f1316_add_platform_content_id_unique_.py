"""add_platform_content_id_unique_constraint

Revision ID: 4da7589f1316
Revises: add_sentences_table
Create Date: 2025-12-30 07:35:49.314347

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '4da7589f1316'
down_revision: Union[str, None] = 'add_sentences_table'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add unique constraint on (platform, content_id) to prevent duplicate content
    # This is safe because there are no existing duplicates (verified before migration)
    op.create_unique_constraint(
        'uq_content_platform_content_id',
        'content',
        ['platform', 'content_id']
    )


def downgrade() -> None:
    op.drop_constraint('uq_content_platform_content_id', 'content', type_='unique')
