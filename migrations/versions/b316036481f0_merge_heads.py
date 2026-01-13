"""merge heads

Revision ID: b316036481f0
Revises: 99dbb6f846d9, a1abbc61329a
Create Date: 2025-05-23 14:07:50.542620

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b316036481f0'
down_revision: Union[str, None] = ('99dbb6f846d9', 'a1abbc61329a')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
