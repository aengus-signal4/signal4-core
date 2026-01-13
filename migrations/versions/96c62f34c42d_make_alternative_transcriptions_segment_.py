"""make_alternative_transcriptions_segment_id_nullable

Revision ID: 96c62f34c42d
Revises: 9687d6431ec3
Create Date: 2025-11-04 13:13:24.083749

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '96c62f34c42d'
down_revision: Union[str, None] = '9687d6431ec3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop the unique constraint first (it depends on segment_id being NOT NULL)
    op.drop_constraint('uq_alt_trans_segment_provider_model', 'alternative_transcriptions', type_='unique')

    # Make segment_id nullable
    op.alter_column('alternative_transcriptions', 'segment_id',
                    existing_type=sa.Integer(),
                    nullable=True)

    # Recreate the unique constraint (now allows NULL segment_id)
    # Note: In PostgreSQL, NULL values are considered distinct, so multiple rows with
    # segment_id=NULL are allowed, which is what we want for arbitrary time ranges
    op.create_unique_constraint('uq_alt_trans_segment_provider_model',
                                'alternative_transcriptions',
                                ['segment_id', 'provider', 'model'])


def downgrade() -> None:
    # Drop the constraint
    op.drop_constraint('uq_alt_trans_segment_provider_model', 'alternative_transcriptions', type_='unique')

    # Delete any rows with NULL segment_id (since we're making it NOT NULL again)
    op.execute('DELETE FROM alternative_transcriptions WHERE segment_id IS NULL')

    # Make segment_id NOT NULL again
    op.alter_column('alternative_transcriptions', 'segment_id',
                    existing_type=sa.Integer(),
                    nullable=False)

    # Recreate the constraint
    op.create_unique_constraint('uq_alt_trans_segment_provider_model',
                                'alternative_transcriptions',
                                ['segment_id', 'provider', 'model'])
