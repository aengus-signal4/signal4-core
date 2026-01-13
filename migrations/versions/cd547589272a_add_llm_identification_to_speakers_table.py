"""add llm_identification to speakers table

Revision ID: cd547589272a
Revises: ff5653eb37c3
Create Date: 2025-11-28

Stores LLM identification results for speakers to avoid re-running expensive LLM calls.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = 'cd547589272a'
down_revision = 'd0f7104f4855'
branch_labels = None
depends_on = None


def upgrade():
    # Add llm_identification JSONB column to store LLM results
    # Structure: {
    #   "identified_name": "Andrew Huberman",
    #   "role": "host",
    #   "confidence": "certain",
    #   "reasoning": "...",
    #   "method": "guest_propagation",  # or "guest_identification", "host_verification"
    #   "embedding_similarity": 0.85,  # if embedding-based
    #   "candidate_name": "Andrew Huberman",  # who we asked about
    #   "timestamp": "2025-11-28T..."
    # }
    op.add_column('speakers', sa.Column('llm_identification', postgresql.JSONB, nullable=True))

    # Index for finding speakers with/without LLM identification
    op.create_index(
        'idx_speakers_llm_identification',
        'speakers',
        ['llm_identification'],
        postgresql_using='gin'
    )


def downgrade():
    op.drop_index('idx_speakers_llm_identification', table_name='speakers')
    op.drop_column('speakers', 'llm_identification')
