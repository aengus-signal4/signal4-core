"""Fix speaker embedding dimensions to match model output

Revision ID: fix_embedding_dimensions
Revises: b316036481f0  # Changed to point to the merge point
Create Date: 2024-05-23 13:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision = 'fix_embedding_dimensions'
down_revision = 'b316036481f0'  # Changed to point to the merge point
branch_labels = None
depends_on = None


def upgrade():
    # First, create temporary columns with the correct dimension
    op.execute("""
        ALTER TABLE speakers 
        ADD COLUMN embedding_new vector(256);
    """)
    
    op.execute("""
        ALTER TABLE speaker_embeddings 
        ADD COLUMN embedding_new vector(256);
    """)
    
    # Copy data, taking first 256 dimensions if source is larger
    op.execute("""
        UPDATE speakers 
        SET embedding_new = embedding[1:256];
    """)
    
    op.execute("""
        UPDATE speaker_embeddings 
        SET embedding_new = embedding[1:256];
    """)
    
    # Drop old columns
    op.execute("""
        ALTER TABLE speakers 
        DROP COLUMN embedding;
    """)
    
    op.execute("""
        ALTER TABLE speaker_embeddings 
        DROP COLUMN embedding;
    """)
    
    # Rename new columns
    op.execute("""
        ALTER TABLE speakers 
        RENAME COLUMN embedding_new TO embedding;
    """)
    
    op.execute("""
        ALTER TABLE speaker_embeddings 
        RENAME COLUMN embedding_new TO embedding;
    """)
    
    # Add comments
    op.execute("""
        COMMENT ON COLUMN speakers.embedding IS '256-dimensional speaker embedding vector from pyannote/wespeaker-voxceleb-resnet34-LM model';
    """)
    
    op.execute("""
        COMMENT ON COLUMN speaker_embeddings.embedding IS '256-dimensional speaker embedding vector from pyannote/wespeaker-voxceleb-resnet34-LM model';
    """)


def downgrade():
    # First, create temporary columns with 512 dimensions
    op.execute("""
        ALTER TABLE speakers 
        ADD COLUMN embedding_old vector(512);
    """)
    
    op.execute("""
        ALTER TABLE speaker_embeddings 
        ADD COLUMN embedding_old vector(512);
    """)
    
    # Copy data, padding with zeros if source is smaller
    op.execute("""
        UPDATE speakers 
        SET embedding_old = embedding || array_fill(0::float, ARRAY[256]);
    """)
    
    op.execute("""
        UPDATE speaker_embeddings 
        SET embedding_old = embedding || array_fill(0::float, ARRAY[256]);
    """)
    
    # Drop old columns
    op.execute("""
        ALTER TABLE speakers 
        DROP COLUMN embedding;
    """)
    
    op.execute("""
        ALTER TABLE speaker_embeddings 
        DROP COLUMN embedding;
    """)
    
    # Rename old columns back
    op.execute("""
        ALTER TABLE speakers 
        RENAME COLUMN embedding_old TO embedding;
    """)
    
    op.execute("""
        ALTER TABLE speaker_embeddings 
        RENAME COLUMN embedding_old TO embedding;
    """) 