"""add_embedding_vector_to_speakers

Revision ID: c04e112f84f0
Revises: a24611f36b04
Create Date: 2024-04-27 11:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, FLOAT


# revision identifiers, used by Alembic.
revision: str = 'c04e112f84f0'
down_revision: Union[str, None] = 'a24611f36b04'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create the pgvector extension if it doesn't exist
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')
    
    # Add the embedding column as a vector type
    op.execute('ALTER TABLE speakers ADD COLUMN embedding vector(256)')
    
    # Create an index for similarity search
    op.execute('CREATE INDEX idx_speakers_embedding ON speakers USING ivfflat (embedding vector_cosine_ops)')


def downgrade() -> None:
    # Drop the index
    op.execute('DROP INDEX IF EXISTS idx_speakers_embedding')
    
    # Drop the column
    op.execute('ALTER TABLE speakers DROP COLUMN IF EXISTS embedding')
