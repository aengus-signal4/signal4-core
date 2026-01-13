"""make_stitch_version_nullable_and_clear_default

Revision ID: 900e8fbd8e27
Revises: fcf2c6a04267
Create Date: 2025-11-18 16:20:21.677598

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '900e8fbd8e27'
down_revision: Union[str, None] = 'fcf2c6a04267'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Make stitch_version nullable and remove default
    op.alter_column('content', 'stitch_version',
                    existing_type=sa.String(),
                    nullable=True,
                    server_default=None)

    # Update existing 'stitch_v1' values to NULL (these haven't actually been stitched)
    op.execute("UPDATE content SET stitch_version = NULL WHERE stitch_version = 'stitch_v1'")


def downgrade() -> None:
    # Restore 'stitch_v1' for NULL values
    op.execute("UPDATE content SET stitch_version = 'stitch_v1' WHERE stitch_version IS NULL")

    # Make stitch_version non-nullable with default
    op.alter_column('content', 'stitch_version',
                    existing_type=sa.String(),
                    nullable=False,
                    server_default='stitch_v1')
