"""Add index to speaker_embeddings content_id

Revision ID: add_speaker_embedding_fields
Revises: 099a4c99bb10
Create Date: 2024-04-02 04:45:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'add_speaker_embedding_fields'
down_revision = '099a4c99bb10'
branch_labels = None
depends_on = None

def upgrade() -> None:
    # Add index for content_id
    op.create_index('idx_speaker_embeddings_content', 'speaker_embeddings', ['content_id'])

def downgrade() -> None:
    # Remove index
    op.drop_index('idx_speaker_embeddings_content') 