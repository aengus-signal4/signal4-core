"""add_query_embeddings_to_classification_schemas

Revision ID: cb83ead5fde1
Revises: 55c3ae201693
Create Date: 2025-11-23 06:08:19.417459

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'cb83ead5fde1'
down_revision: Union[str, None] = '55c3ae201693'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add query_embeddings column to store pre-computed embeddings for subtheme queries
    # Format: {subtheme_id: {lang: [embedding_vector]}}
    # Example: {"Q1": {"en": [0.1, 0.2, ...], "fr": [0.3, 0.4, ...]}}
    op.add_column('classification_schemas',
                  sa.Column('query_embeddings', postgresql.JSONB(astext_type=sa.Text()), nullable=True))


def downgrade() -> None:
    op.drop_column('classification_schemas', 'query_embeddings')
