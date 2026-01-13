"""migrate_speakers_meta_data_to_jsonb

Revision ID: dfa31bbbe156
Revises: 9182dd5f5c4d
Create Date: 2025-11-07 06:01:31.819581

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'dfa31bbbe156'
down_revision: Union[str, None] = '9182dd5f5c4d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Migrate speakers.meta_data from JSON to JSONB
    # This allows using the || operator for merging JSON objects
    op.execute("""
        ALTER TABLE speakers
        ALTER COLUMN meta_data
        TYPE jsonb
        USING meta_data::jsonb
    """)


def downgrade() -> None:
    # Revert speakers.meta_data from JSONB back to JSON
    op.execute("""
        ALTER TABLE speakers
        ALTER COLUMN meta_data
        TYPE json
        USING meta_data::json
    """)
