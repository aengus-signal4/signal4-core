"""add chunk support

Revision ID: add_chunk_support
Revises: clean_content_ids
Create Date: 2024-02-26 22:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSON

# revision identifiers, used by Alembic.
revision = 'add_chunk_support'
down_revision = 'clean_content_ids'
branch_labels = None
depends_on = None

def upgrade():
    # Add chunk-related fields to Content table
    op.add_column('content', sa.Column('total_chunks', sa.Integer(), nullable=True))
    op.add_column('content', sa.Column('chunks_processed', sa.Integer(), server_default='0', nullable=False))
    op.add_column('content', sa.Column('chunks_status', JSON(), server_default='{}', nullable=False))
    
    # Create ContentChunk table
    op.create_table('content_chunks',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('content_id', sa.Integer(), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('start_time', sa.Float(), nullable=False),
        sa.Column('end_time', sa.Float(), nullable=False),
        sa.Column('duration', sa.Float(), nullable=False),
        sa.Column('status', sa.String(), server_default='pending', nullable=False),
        sa.Column('error', sa.Text()),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('processed_at', sa.DateTime(timezone=True)),
        sa.ForeignKeyConstraint(['content_id'], ['content.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # Add indexes
    op.create_index('idx_content_chunks_content', 'content_chunks', ['content_id'])
    op.create_index('idx_content_chunks_status', 'content_chunks', ['status'])
    op.create_index('idx_content_chunks_index', 'content_chunks', ['chunk_index'])

def downgrade():
    # Drop indexes
    op.drop_index('idx_content_chunks_index')
    op.drop_index('idx_content_chunks_status')
    op.drop_index('idx_content_chunks_content')
    
    # Drop ContentChunk table
    op.drop_table('content_chunks')
    
    # Remove chunk-related columns from Content table
    op.drop_column('content', 'chunks_status')
    op.drop_column('content', 'chunks_processed')
    op.drop_column('content', 'total_chunks') 