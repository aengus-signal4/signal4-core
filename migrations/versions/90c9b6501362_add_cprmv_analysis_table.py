"""add_cprmv_analysis_table

Revision ID: 90c9b6501362
Revises: 900e8fbd8e27
Create Date: 2025-11-19 05:15:16.662261

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision: str = '90c9b6501362'
down_revision: Union[str, None] = '900e8fbd8e27'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create cprmv_analysis table
    op.create_table(
        'cprmv_analysis',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('segment_id', sa.Integer(), nullable=False),
        sa.Column('themes', postgresql.ARRAY(sa.String()), nullable=False),
        sa.Column('confidence_scores', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('high_confidence_themes', postgresql.ARRAY(sa.String()), nullable=False),
        sa.Column('matched_via', sa.String(length=20), nullable=True),
        sa.Column('similarity_score', sa.Float(), nullable=True),
        sa.Column('matched_keywords', sa.Text(), nullable=True),
        sa.Column('embedding', Vector(2000), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['segment_id'], ['embedding_segments.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('segment_id')
    )

    # Create indexes
    op.create_index('idx_cprmv_analysis_segment_id', 'cprmv_analysis', ['segment_id'])
    op.create_index('idx_cprmv_analysis_matched_via', 'cprmv_analysis', ['matched_via'])
    op.create_index('idx_cprmv_analysis_themes', 'cprmv_analysis', ['themes'], postgresql_using='gin')
    op.create_index('idx_cprmv_analysis_high_conf_themes', 'cprmv_analysis', ['high_confidence_themes'], postgresql_using='gin')

    # Create IVFFlat index on embedding (must be done after data is loaded)
    # Note: This index requires data to exist, so it's commented out here
    # Run manually after data import: CREATE INDEX idx_cprmv_analysis_embedding ON cprmv_analysis USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_cprmv_analysis_embedding
        ON cprmv_analysis
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
    """)


def downgrade() -> None:
    op.drop_index('idx_cprmv_analysis_embedding', table_name='cprmv_analysis')
    op.drop_index('idx_cprmv_analysis_high_conf_themes', table_name='cprmv_analysis', postgresql_using='gin')
    op.drop_index('idx_cprmv_analysis_themes', table_name='cprmv_analysis', postgresql_using='gin')
    op.drop_index('idx_cprmv_analysis_matched_via', table_name='cprmv_analysis')
    op.drop_index('idx_cprmv_analysis_segment_id', table_name='cprmv_analysis')
    op.drop_table('cprmv_analysis')
