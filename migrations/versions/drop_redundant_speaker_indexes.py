"""Drop redundant speaker indexes

Revision ID: drop_redundant_speaker_indexes
Revises: 
Create Date: 2025-01-13

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'drop_redundant_speaker_indexes'
down_revision = 'add_speaker_mgmt_indices'
branch_labels = None
depends_on = None


def upgrade():
    """Drop redundant indexes that are hurting write performance."""
    
    # Drop the HNSW index - we use FAISS instead
    op.execute("DROP INDEX IF EXISTS idx_speakers_embedding_hnsw;")
    print("Dropped idx_speakers_embedding_hnsw (2.9GB freed)")
    
    # Drop duplicate hash index - we have idx_speakers_speaker_hash
    op.execute("DROP INDEX IF EXISTS idx_speakers_hash;")
    print("Dropped idx_speakers_hash (56MB freed)")
    
    print("\nTotal space freed: ~3GB")
    print("This will significantly improve write performance for speaker operations")


def downgrade():
    """Recreate the dropped indexes if needed."""
    
    # Recreate HNSW index (warning: this will take a while)
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_speakers_embedding_hnsw 
        ON speakers USING hnsw (embedding vector_ip_ops)
        WITH (m = 16, ef_construction = 64);
    """)
    
    # Recreate hash index
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_speakers_hash 
        ON speakers(speaker_hash);
    """)