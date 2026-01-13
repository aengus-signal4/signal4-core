"""Add alternative embedding column for model migration

Revision ID: 72ff30f1b825
Revises: 
Create Date: 2025-09-11 18:47:35.615853

"""
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision = '72ff30f1b825'
down_revision = 'add_speaker_annotation_perf'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add new column for alternative embeddings 
    # Using 2560 dimensions to support Qwen3-Embedding-4B
    # - Qwen3-Embedding-4B: 2560 dims (excellent quality/performance balance)
    # - Can also store smaller models (768, 1024, 1536 dims)
    op.add_column('embedding_segments', 
        sa.Column('embedding_alt', Vector(2560), nullable=True)
    )
    
    # Add column to track which model was used for alternative embedding
    op.add_column('embedding_segments',
        sa.Column('embedding_alt_model', sa.String(50), nullable=True)
    )
    
    # Skip vector index for now - ivfflat has 2000 dimension limit
    # For 2560 dimensions, we would need to use different indexing strategy
    # or wait for pgvector updates that support higher dimensions
    # The column itself will work fine, just without optimized similarity search
    
    # Add index for tracking which segments have alternative embeddings
    op.create_index(
        'idx_embedding_segments_alt_model',
        'embedding_segments',
        ['embedding_alt_model'],
        postgresql_where=sa.text('embedding_alt_model IS NOT NULL')
    )


def downgrade() -> None:
    # Drop indexes first
    op.drop_index('idx_embedding_segments_alt_model', table_name='embedding_segments')
    
    # Drop columns
    op.drop_column('embedding_segments', 'embedding_alt_model')
    op.drop_column('embedding_segments', 'embedding_alt')