"""Add embedding segments table

Revision ID: add_embedding_segments
Revises: b316036481f0
Create Date: 2024-01-23
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision = 'add_embedding_segments'
down_revision = 'b316036481f0'
branch_labels = None
depends_on = None

def upgrade():
    # Create embedding_segments table
    op.create_table('embedding_segments',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('content_id', sa.Integer(), nullable=False),
        sa.Column('segment_index', sa.Integer(), nullable=False),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('start_time', sa.Float(), nullable=False),
        sa.Column('end_time', sa.Float(), nullable=False),
        sa.Column('token_count', sa.Integer(), nullable=False),
        sa.Column('segment_type', sa.String(20), nullable=False),
        sa.Column('source_transcription_ids', postgresql.ARRAY(sa.Integer()), nullable=False),
        sa.Column('source_start_char', sa.Integer(), nullable=True),
        sa.Column('source_end_char', sa.Integer(), nullable=True),
        sa.Column('embedding', Vector(1024), nullable=True),
        sa.Column('meta_data', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['content_id'], ['content.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Create indexes
    op.create_index('idx_embedding_segments_content_index', 
                    'embedding_segments', ['content_id', 'segment_index'])
    op.create_index('idx_embedding_segments_source_ids', 
                    'embedding_segments', ['source_transcription_ids'], 
                    postgresql_using='gin')
    
    # Note: is_embedded column already exists in Content model, no need to add it

def downgrade():
    # Note: is_embedded column was not added by this migration, so don't remove it
    
    # Drop indexes
    op.drop_index('idx_embedding_segments_source_ids')
    op.drop_index('idx_embedding_segments_content_index')
    
    # Drop table
    op.drop_table('embedding_segments')