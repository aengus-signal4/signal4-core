"""add_gin_text_index_embedding_segments

Revision ID: 0fe3e6c128fd
Revises: 421f7522ab81
Create Date: 2025-09-16 17:31:16.063510

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0fe3e6c128fd'
down_revision: Union[str, None] = '421f7522ab81'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add GIN index for multilingual full-text search on embedding_segments.text
    op.execute("CREATE INDEX IF NOT EXISTS idx_embedding_segments_text_search ON embedding_segments USING gin(to_tsvector('simple', text))")
    
    # Add trigram index for similarity search on embedding_segments.text (like speaker_transcriptions has)
    op.execute("CREATE INDEX IF NOT EXISTS idx_embedding_segments_text_trigram ON embedding_segments USING gin(text gin_trgm_ops)")


def downgrade() -> None:
    # Drop the indexes
    op.execute("DROP INDEX IF EXISTS idx_embedding_segments_text_search")
    op.execute("DROP INDEX IF EXISTS idx_embedding_segments_text_trigram")
