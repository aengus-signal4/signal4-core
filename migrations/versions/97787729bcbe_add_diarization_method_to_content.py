"""add_diarization_method_to_content

Revision ID: 97787729bcbe
Revises: 553fd172eb1a
Create Date: 2025-09-29 06:32:20.410235

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '97787729bcbe'
down_revision: Union[str, None] = '553fd172eb1a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add the diarization_method column
    op.add_column('content', sa.Column('diarization_method', sa.String(length=50), nullable=True))
    
    # Create index for the new column
    op.create_index('idx_content_diarization_method', 'content', ['diarization_method'])
    
    # Populate existing records with 'pyannote3.1' where diarization is complete
    op.execute("UPDATE content SET diarization_method = 'pyannote3.1' WHERE is_diarized = true")


def downgrade() -> None:
    # Drop the index first
    op.drop_index('idx_content_diarization_method', table_name='content')
    
    # Drop the column
    op.drop_column('content', 'diarization_method')
