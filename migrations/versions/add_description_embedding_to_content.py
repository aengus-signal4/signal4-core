"""Add description_embedding column to content table

Revision ID: add_description_embedding
Revises: add_youtube_video_cache
Create Date: 2026-01-22

Adds a 1024-dimensional vector column to store embeddings of episode
descriptions. This enables semantic similarity-based "Related Episodes"
recommendations instead of same-channel-only approach.

Uses IVF index (not HNSW) because:
- Descriptions are static (don't change after creation)
- Query pattern is predictable (k=5 nearest neighbors)
- Faster build time and lower memory usage
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = 'add_description_embedding'
down_revision: Union[str, Sequence[str], None] = 'add_youtube_video_cache'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add description_embedding column (1024 dimensions, same as segment embeddings)
    op.add_column('content',
        sa.Column('description_embedding', Vector(1024), nullable=True)
    )

    # Create IVF index for k-NN queries
    # lists=100 is appropriate for ~50k content items (sqrt(n) rule)
    # Using cosine distance (vector_cosine_ops) for semantic similarity
    op.execute("""
        CREATE INDEX content_description_ivf_idx
        ON content USING ivfflat (description_embedding vector_cosine_ops)
        WITH (lists = 100)
    """)


def downgrade() -> None:
    # Drop index first
    op.execute("DROP INDEX IF EXISTS content_description_ivf_idx")

    # Drop column
    op.drop_column('content', 'description_embedding')
