"""Add partial HNSW indexes per project for fast filtered queries.

Revision ID: add_partial_hnsw_indexes
Revises: b6187fea062d
Create Date: 2026-01-19

This migration creates partial HNSW indexes on embedding_cache_30d for each project.
This allows fast nearest-neighbor search with project filters (~8ms vs ~500ms).

The search service will query each relevant project's index separately,
then deduplicate and rerank the combined results.
"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'add_partial_hnsw_indexes'
down_revision = 'b6187fea062d'
branch_labels = None
depends_on = None

# Projects to create indexes for (CPRMV already exists)
PROJECTS = [
    'Canadian',
    'Big_Channels',
    'Health',
    'Finance',
    'Europe',
    'Anglosphere',
]


def upgrade():
    """Create partial HNSW indexes for each project."""
    conn = op.get_bind()

    for project in PROJECTS:
        index_name = f'embedding_cache_30d_hnsw_{project.lower()}'

        # Check if index already exists
        result = conn.execute(sa.text(
            "SELECT 1 FROM pg_indexes WHERE indexname = :index_name"
        ), {"index_name": index_name})

        if result.fetchone() is None:
            # Create index (without CONCURRENTLY to work inside transaction)
            op.execute(f"""
                CREATE INDEX {index_name}
                ON embedding_cache_30d
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
                WHERE (projects && ARRAY['{project}']::varchar[])
            """)
            print(f"Created index: {index_name}")
        else:
            print(f"Index already exists: {index_name}")


def downgrade():
    """Drop partial HNSW indexes."""
    for project in PROJECTS:
        index_name = f'embedding_cache_30d_hnsw_{project.lower()}'
        op.execute(f"DROP INDEX IF EXISTS {index_name};")
        print(f"Dropped index: {index_name}")
