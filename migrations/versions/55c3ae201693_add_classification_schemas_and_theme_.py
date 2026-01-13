"""add_classification_schemas_and_theme_classifications

Revision ID: 55c3ae201693
Revises: c532b92f9e66
Create Date: 2025-11-23 05:27:01.444619

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision: str = '55c3ae201693'
down_revision: Union[str, None] = 'c532b92f9e66'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create classification_schemas table
    op.create_table('classification_schemas',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('version', sa.String(length=50), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('themes_json', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('subthemes_json', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('source_file', sa.String(length=500), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name', 'version', name='uq_schema_name_version')
    )
    op.create_index('idx_schema_name', 'classification_schemas', ['name'], unique=False)

    # Create theme_classifications table
    op.create_table('theme_classifications',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('segment_id', sa.Integer(), nullable=False),
        sa.Column('schema_id', sa.Integer(), nullable=False),
        sa.Column('theme_ids', postgresql.ARRAY(sa.String()), nullable=False, server_default='{}'),
        sa.Column('subtheme_ids', postgresql.ARRAY(sa.String()), nullable=False, server_default='{}'),
        sa.Column('high_confidence_themes', postgresql.ARRAY(sa.String()), nullable=False, server_default='{}'),
        sa.Column('stage1_similarities', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('stage3_results', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('stage4_validations', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('final_confidence_scores', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='{}'),
        sa.Column('matched_via', sa.String(length=20), nullable=True),
        sa.Column('max_similarity_score', sa.Float(), nullable=True),
        sa.Column('embedding', Vector(2000), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.ForeignKeyConstraint(['segment_id'], ['embedding_segments.id'], ),
        sa.ForeignKeyConstraint(['schema_id'], ['classification_schemas.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('segment_id', 'schema_id', name='uq_segment_schema')
    )

    # Create indexes
    op.create_index('idx_theme_class_segment_id', 'theme_classifications', ['segment_id'], unique=False)
    op.create_index('idx_theme_class_schema_id', 'theme_classifications', ['schema_id'], unique=False)
    op.create_index('idx_theme_class_matched_via', 'theme_classifications', ['matched_via'], unique=False)

    # GIN indexes for array queries
    op.create_index('idx_theme_class_theme_ids', 'theme_classifications', ['theme_ids'],
                    unique=False, postgresql_using='gin')
    op.create_index('idx_theme_class_subtheme_ids', 'theme_classifications', ['subtheme_ids'],
                    unique=False, postgresql_using='gin')
    op.create_index('idx_theme_class_high_conf', 'theme_classifications', ['high_confidence_themes'],
                    unique=False, postgresql_using='gin')

    # IVFFlat index for embedding (will be created after data is loaded)
    # Note: IVFFlat index requires data to train, so it should be created manually after initial data load:
    # CREATE INDEX idx_theme_class_embedding ON theme_classifications
    # USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);


def downgrade() -> None:
    op.drop_table('theme_classifications')
    op.drop_table('classification_schemas')
