"""Optimize speaker clustering indexes for scalability

Revision ID: optimize_clustering_indexes
Revises: 
Create Date: 2025-01-11 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'optimize_clustering_indexes'
down_revision = '337b8fb68502'
branch_labels = None
depends_on = None


def upgrade():
    """
    Optimize indexes for scalable speaker clustering:
    1. Enhance HNSW index on speakers.embedding with better parameters
    2. Add optimized HNSW index on speaker_embeddings.embedding
    3. Add covering indexes for common clustering queries
    4. Add merge tracking fields for speaker name preservation
    """
    
    # Drop existing HNSW index on speakers if it exists
    op.execute("DROP INDEX IF EXISTS idx_speaker_embedding")
    
    # Create optimized HNSW index on speakers.embedding
    # Higher m parameter (32) for better recall at cost of more memory
    # Higher ef_construction (128) for better build quality
    op.execute("""
        CREATE INDEX idx_speaker_embedding_optimized ON speakers 
        USING hnsw (embedding vector_cosine_ops) 
        WITH (m = 32, ef_construction = 128)
    """)
    
    # Create HNSW index on speaker_embeddings.embedding for fast KNN queries
    op.execute("""
        CREATE INDEX idx_speaker_embeddings_embedding_hnsw ON speaker_embeddings 
        USING hnsw (embedding vector_cosine_ops) 
        WITH (m = 24, ef_construction = 100)
    """)
    
    # Add covering indexes for clustering queries
    op.create_index(
        'idx_speakers_clustering_query',
        'speakers',
        ['id', 'universal_name', 'display_name'],
        postgresql_include=['embedding'],
        postgresql_where=sa.text('embedding IS NOT NULL')
    )
    
    op.create_index(
        'idx_speaker_embeddings_clustering_query',
        'speaker_embeddings',
        ['speaker_id', 'quality_score', 'duration'],
        postgresql_include=['embedding', 'content_id', 'local_speaker_id'],
        postgresql_where=sa.text('quality_score >= 0.5')
    )
    
    # Add merge tracking fields for speaker name preservation
    op.add_column('speakers', sa.Column('canonical_speaker_id', sa.Integer, nullable=True))
    op.add_column('speakers', sa.Column('merge_history', sa.JSON, default=list))
    op.add_column('speakers', sa.Column('is_canonical', sa.Boolean, nullable=True, default=True))
    op.add_column('speakers', sa.Column('merge_confidence', sa.Float, nullable=True))
    
    # Update existing speakers to be canonical
    op.execute("UPDATE speakers SET is_canonical = true WHERE is_canonical IS NULL")
    
    # Now make is_canonical non-nullable
    op.alter_column('speakers', 'is_canonical', nullable=False)
    
    # Add foreign key for canonical speaker
    op.create_foreign_key(
        'fk_speakers_canonical_speaker',
        'speakers', 'speakers',
        ['canonical_speaker_id'], ['id'],
        ondelete='SET NULL'
    )
    
    # Add indexes for merge tracking
    op.create_index('idx_speakers_canonical', 'speakers', ['canonical_speaker_id'])
    op.create_index('idx_speakers_is_canonical', 'speakers', ['is_canonical'])
    
    # Create clustering batch tracking table for incremental processing
    op.create_table(
        'clustering_batches',
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('batch_id', sa.String(50), unique=True, nullable=False),
        sa.Column('status', sa.String(20), nullable=False, default='pending'),
        sa.Column('speaker_count', sa.Integer, nullable=False),
        sa.Column('merge_candidates_found', sa.Integer, default=0),
        sa.Column('merges_applied', sa.Integer, default=0),
        sa.Column('processing_time', sa.Float, nullable=True),
        sa.Column('config_params', sa.JSON, default=dict),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
    )
    
    op.create_index('idx_clustering_batches_status', 'clustering_batches', ['status'])
    op.create_index('idx_clustering_batches_created_at', 'clustering_batches', ['created_at'])


def downgrade():
    """Reverse the optimizations"""
    
    # Drop the new table
    op.drop_table('clustering_batches')
    
    # Drop new indexes
    op.drop_index('idx_speakers_is_canonical')
    op.drop_index('idx_speakers_canonical')
    op.drop_index('idx_speaker_embeddings_clustering_query')
    op.drop_index('idx_speakers_clustering_query')
    op.drop_index('idx_speaker_embeddings_embedding_hnsw')
    op.drop_index('idx_speaker_embedding_optimized')
    
    # Drop foreign key
    op.drop_constraint('fk_speakers_canonical_speaker', 'speakers', type_='foreignkey')
    
    # Drop new columns
    op.drop_column('speakers', 'merge_confidence')
    op.drop_column('speakers', 'is_canonical')
    op.drop_column('speakers', 'merge_history')
    op.drop_column('speakers', 'canonical_speaker_id')
    
    # Recreate original index with old parameters
    op.execute("""
        CREATE INDEX idx_speaker_embedding ON speakers 
        USING hnsw (embedding vector_cosine_ops) 
        WITH (m = 16, ef_construction = 64)
    """)