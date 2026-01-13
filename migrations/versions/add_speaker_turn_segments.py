"""add speaker turn segments table with vector support

Revision ID: add_speaker_turn_segments
Revises: ccd024fe31a8
Create Date: 2024-03-26

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'add_speaker_turn_segments'
down_revision = 'ccd024fe31a8'
branch_labels = None
depends_on = None

def upgrade():
    # Enable the vector extension if not already enabled
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')
    
    # Add turn_id to speaker_transcriptions
    op.add_column('speaker_transcriptions', 
        sa.Column('turn_id', sa.String(100), nullable=True)
    )
    
    # Create an index on turn_id
    op.create_index('idx_speaker_transcriptions_turn_id', 'speaker_transcriptions', ['turn_id'])
    
    # Update existing records to have a turn_id
    op.execute("""
        UPDATE speaker_transcriptions 
        SET turn_id = content_id || '_' || turn_index
    """)
    
    # Make turn_id not nullable after populating
    op.alter_column('speaker_transcriptions', 'turn_id',
        nullable=False
    )
    
    # Create unique constraint on turn_id
    op.create_unique_constraint('uq_speaker_transcriptions_turn_id', 'speaker_transcriptions', ['turn_id'])
    
    # Create speaker_turn_segments_hot table for new inserts (no index)
    op.create_table('speaker_turn_segments_hot',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('turn_id', sa.String(100), nullable=False),
        sa.Column('segment_index', sa.Integer(), nullable=False),
        sa.Column('start_time', sa.Float(), nullable=False),
        sa.Column('end_time', sa.Float(), nullable=False),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('embedding', postgresql.VECTOR(dimensions=1024), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )

    # Create speaker_turn_segments_cold table for indexed searches
    op.create_table('speaker_turn_segments_cold',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('turn_id', sa.String(100), nullable=False),
        sa.Column('segment_index', sa.Integer(), nullable=False),
        sa.Column('start_time', sa.Float(), nullable=False),
        sa.Column('end_time', sa.Float(), nullable=False),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('embedding', postgresql.VECTOR(dimensions=1024), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Add foreign key constraints
    op.create_foreign_key(
        'fk_speaker_turn_segments_hot_turn_id', 
        'speaker_turn_segments_hot', 'speaker_transcriptions',
        ['turn_id'], ['turn_id']
    )
    
    op.create_foreign_key(
        'fk_speaker_turn_segments_cold_turn_id', 
        'speaker_turn_segments_cold', 'speaker_transcriptions',
        ['turn_id'], ['turn_id']
    )
    
    # Create indexes for hot table (minimal indexes for write performance)
    op.create_index('idx_speaker_turn_segments_hot_turn_id', 'speaker_turn_segments_hot', ['turn_id'])
    op.create_unique_constraint('uq_speaker_turn_segments_hot_turn_segment', 'speaker_turn_segments_hot', ['turn_id', 'segment_index'])
    
    # Create indexes for cold table (including HNSW)
    op.create_index('idx_speaker_turn_segments_cold_turn_id', 'speaker_turn_segments_cold', ['turn_id'])
    op.create_index('idx_speaker_turn_segments_cold_time', 'speaker_turn_segments_cold', ['turn_id', 'start_time', 'end_time'])
    op.create_index('idx_speaker_turn_segments_cold_text', 'speaker_turn_segments_cold', ['text'], postgresql_using='gin', postgresql_ops={'text': 'gin_trgm_ops'})
    op.create_unique_constraint('uq_speaker_turn_segments_cold_turn_segment', 'speaker_turn_segments_cold', ['turn_id', 'segment_index'])

    # Create function to move data from hot to cold table
    op.execute("""
    CREATE OR REPLACE FUNCTION move_segments_to_cold()
    RETURNS void AS $$
    BEGIN
        -- Insert new records from hot to cold
        INSERT INTO speaker_turn_segments_cold (
            turn_id, segment_index, start_time, end_time, text, embedding, 
            created_at, updated_at
        )
        SELECT 
            turn_id, segment_index, start_time, end_time, text, embedding,
            created_at, updated_at
        FROM speaker_turn_segments_hot
        ON CONFLICT (turn_id, segment_index) DO UPDATE
        SET 
            start_time = EXCLUDED.start_time,
            end_time = EXCLUDED.end_time,
            text = EXCLUDED.text,
            embedding = EXCLUDED.embedding,
            updated_at = EXCLUDED.updated_at;
            
        -- Clear the hot table
        TRUNCATE speaker_turn_segments_hot;
    END;
    $$ LANGUAGE plpgsql;
    """)

def downgrade():
    # Drop function first
    op.execute("DROP FUNCTION IF EXISTS move_segments_to_cold()")
    
    # Drop tables and their constraints/indexes
    op.drop_table('speaker_turn_segments_hot')
    op.drop_table('speaker_turn_segments_cold')
    
    # Remove turn_id from speaker_transcriptions
    op.drop_constraint('uq_speaker_transcriptions_turn_id', 'speaker_transcriptions')
    op.drop_index('idx_speaker_transcriptions_turn_id', 'speaker_transcriptions')
    op.drop_column('speaker_transcriptions', 'turn_id')
    
    # We don't remove the vector extension as it might be used by other tables 