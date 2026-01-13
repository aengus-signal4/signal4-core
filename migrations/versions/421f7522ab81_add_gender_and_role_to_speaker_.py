"""add_gender_and_role_to_speaker_identities

Revision ID: 421f7522ab81
Revises: drop_redundant_speaker_indexes
Create Date: 2025-09-14 12:14:14.812193

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '421f7522ab81'
down_revision: Union[str, None] = 'drop_redundant_speaker_indexes'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add gender and role columns to speaker_identities table
    op.add_column('speaker_identities', sa.Column('gender', sa.String(50), nullable=True))
    op.add_column('speaker_identities', sa.Column('role', sa.String(100), nullable=True))
    
    # Also increase country column size for full country names
    op.alter_column('speaker_identities', 'country',
                    type_=sa.String(100),
                    existing_type=sa.String(10),
                    existing_nullable=True)


def downgrade() -> None:
    # Remove the added columns
    op.drop_column('speaker_identities', 'role')
    op.drop_column('speaker_identities', 'gender')
    
    # Revert country column size
    op.alter_column('speaker_identities', 'country',
                    type_=sa.String(10),
                    existing_type=sa.String(100),
                    existing_nullable=True)
