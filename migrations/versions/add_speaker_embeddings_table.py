"""Add SpeakerEmbedding table for robust centroid management

Revision ID: add_speaker_embeddings
Revises: a24611f36b04
Create Date: 2025-05-23 12:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision = 'add_speaker_embeddings'
down_revision = 'a24611f36b04'
branch_labels = None
depends_on = None


def upgrade():
    # Create speaker_embeddings table
    op.create_table('speaker_embeddings',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('speaker_id', sa.Integer(), nullable=False),
        sa.Column('content_id', sa.String(), nullable=False),
        sa.Column('local_speaker_id', sa.String(), nullable=False),
        sa.Column('embedding', Vector(256), nullable=False),
        sa.Column('duration', sa.Float(), nullable=False),
        sa.Column('segment_count', sa.Integer(), nullable=False),
        sa.Column('quality_score', sa.Float(), nullable=True, default=1.0),
        sa.Column('algorithm_version', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['speaker_id'], ['speakers.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('speaker_id', 'content_id', 'local_speaker_id', name='uq_speaker_embedding_content')
    )
    
    # Create indexes
    op.create_index('idx_speaker_embeddings_speaker_id', 'speaker_embeddings', ['speaker_id'])
    op.create_index('idx_speaker_embeddings_content_id', 'speaker_embeddings', ['content_id'])
    op.create_index('idx_speaker_embeddings_created_at', 'speaker_embeddings', ['created_at'])
    op.create_index('idx_speaker_embeddings_quality', 'speaker_embeddings', ['quality_score'])
    op.create_index('idx_speaker_embeddings_duration', 'speaker_embeddings', ['duration'])
    op.create_index('idx_speaker_embeddings_speaker_quality', 'speaker_embeddings', ['speaker_id', 'quality_score', 'duration'])


def downgrade():
    # Drop indexes if they exist
    for index_name in [
        'idx_speaker_embeddings_speaker_quality',
        'idx_speaker_embeddings_duration',
        'idx_speaker_embeddings_quality',
        'idx_speaker_embeddings_created_at',
        'idx_speaker_embeddings_content_id',
        'idx_speaker_embeddings_speaker_id'
    ]:
        op.execute(f"DROP INDEX IF EXISTS {index_name};")
    # Drop the table if it exists
    op.execute("DROP TABLE IF EXISTS speaker_embeddings;")