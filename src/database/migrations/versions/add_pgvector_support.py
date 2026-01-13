"""Add pgvector support for speaker embeddings

Revision ID: add_pgvector_support
Revises: truncate_speaker_tables
Create Date: 2024-04-02 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'add_pgvector_support'
down_revision = 'truncate_speaker_tables'
branch_labels = None
depends_on = None

def upgrade():
    # Enable the vector extension
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')
    
    # Add vector column to speaker_embeddings
    op.execute("""
        ALTER TABLE speaker_embeddings 
        ADD COLUMN IF NOT EXISTS embedding_vector vector(256)
    """)
    
    # Create a function to convert binary embedding to vector
    op.execute("""
        CREATE OR REPLACE FUNCTION binary_to_vector(bytes bytea) 
        RETURNS vector 
        AS $$
        DECLARE
            float_array float4[];
        BEGIN
            -- Convert bytea to float4[]
            SELECT array_agg(value::float4)
            INTO float_array
            FROM unnest(float4(bytes)) AS value;
            
            -- Convert float4[] to vector
            RETURN float_array::vector;
        END;
        $$ LANGUAGE plpgsql IMMUTABLE;
    """)
    
    # Migrate existing embeddings
    op.execute("""
        UPDATE speaker_embeddings
        SET embedding_vector = binary_to_vector(embedding)
        WHERE embedding_vector IS NULL
    """)
    
    # Create index for similarity search
    op.execute("""
        CREATE INDEX IF NOT EXISTS speaker_embeddings_vector_idx 
        ON speaker_embeddings 
        USING ivfflat (embedding_vector vector_cosine_ops)
        WITH (lists = 100)
    """)

def downgrade():
    # Drop the index
    op.execute('DROP INDEX IF EXISTS speaker_embeddings_vector_idx')
    
    # Drop the vector column
    op.execute('ALTER TABLE speaker_embeddings DROP COLUMN IF EXISTS embedding_vector')
    
    # Drop the conversion function
    op.execute('DROP FUNCTION IF EXISTS binary_to_vector')
    
    # Drop the vector extension if no other tables are using it
    op.execute('DROP EXTENSION IF EXISTS vector') 