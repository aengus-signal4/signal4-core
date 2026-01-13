"""migrate_projects_to_array

Migrates the 'projects' column from comma-separated String to PostgreSQL ARRAY(String).

This migration:
1. Adds new projects_array column (ARRAY type)
2. Backfills data from projects column (CSV → array, with whitespace trimming)
3. Drops the old projects column
4. Renames projects_array to projects
5. Recreates indexes on the new array column

Revision ID: 7ed65f02fcbd
Revises: dfa31bbbe156
Create Date: 2025-11-11 05:01:05.876540

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY


# revision identifiers, used by Alembic.
revision: str = '7ed65f02fcbd'
down_revision: Union[str, None] = 'dfa31bbbe156'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Migrate projects from comma-separated string to array type.
    """
    # Step 1: Add new projects_array column to content table
    op.add_column('content', sa.Column('projects_array', ARRAY(sa.String()), nullable=True))

    # Step 2: Backfill projects_array from projects column
    # Convert CSV to array and trim whitespace from each element
    op.execute("""
        UPDATE content
        SET projects_array = (
            SELECT array_agg(TRIM(elem))
            FROM unnest(string_to_array(projects, ',')) AS elem
            WHERE TRIM(elem) != ''
        )
        WHERE projects IS NOT NULL
    """)

    # Step 3: Make projects_array NOT NULL (after backfill)
    op.alter_column('content', 'projects_array', nullable=False)

    # Step 4: Drop materialized views that depend on projects column
    op.execute('DROP MATERIALIZED VIEW IF EXISTS embedding_cache_7d CASCADE')
    op.execute('DROP MATERIALIZED VIEW IF EXISTS embedding_cache_180d CASCADE')
    op.execute('DROP MATERIALIZED VIEW IF EXISTS embedding_cache_7d_alt CASCADE')
    op.execute('DROP MATERIALIZED VIEW IF EXISTS embedding_cache_180d_alt CASCADE')

    # Step 5: Drop old indexes on projects column
    op.drop_index('idx_content_projects', table_name='content')
    op.drop_index('idx_content_projects_gin', table_name='content')

    # Step 6: Drop old projects column
    op.drop_column('content', 'projects')

    # Step 7: Rename projects_array to projects
    op.alter_column('content', 'projects_array', new_column_name='projects')

    # Step 8: Recreate indexes on new array column
    # Standard index for array lookups
    op.create_index('idx_content_projects', 'content', ['projects'])

    # GIN index for fast array containment queries (ANY, &&, @> operators)
    op.execute('CREATE INDEX idx_content_projects_gin ON content USING gin(projects)')

    # Step 9: Recreate materialized views with updated projects column
    # Note: These views will need to be refreshed after migration
    op.execute("""
        CREATE MATERIALIZED VIEW embedding_cache_180d AS
        SELECT
            es.id,
            es.segment_hash,
            es.content_id,
            es.content_id_string,
            es.segment_index,
            es.text,
            es.start_time,
            es.end_time,
            es.source_speaker_hashes,
            es.embedding,
            es.embedding_alt,
            es.stitch_version,
            c.publish_date,
            c.projects,
            c.channel_name,
            c.title,
            c.main_language
        FROM embedding_segments es
        JOIN content c ON es.content_id = c.id
        WHERE c.publish_date >= CURRENT_DATE - INTERVAL '180 days'
          AND es.embedding IS NOT NULL;
    """)

    op.execute("""
        CREATE MATERIALIZED VIEW embedding_cache_7d AS
        SELECT * FROM embedding_cache_180d
        WHERE publish_date >= CURRENT_DATE - INTERVAL '7 days';
    """)

    op.execute("""
        CREATE MATERIALIZED VIEW embedding_cache_180d_alt AS
        SELECT
            es.id,
            es.segment_hash,
            es.content_id,
            es.content_id_string,
            es.segment_index,
            es.text,
            es.start_time,
            es.end_time,
            es.source_speaker_hashes,
            es.embedding_alt as embedding,
            es.stitch_version,
            c.publish_date,
            c.projects,
            c.channel_name,
            c.title,
            c.main_language
        FROM embedding_segments es
        JOIN content c ON es.content_id = c.id
        WHERE c.publish_date >= CURRENT_DATE - INTERVAL '180 days'
          AND es.embedding_alt IS NOT NULL;
    """)

    op.execute("""
        CREATE MATERIALIZED VIEW embedding_cache_7d_alt AS
        SELECT * FROM embedding_cache_180d_alt
        WHERE publish_date >= CURRENT_DATE - INTERVAL '7 days';
    """)

    # Create GIN indexes on projects column for fast array queries
    # Note: Vector indexes (ivfflat) can be created manually later with higher maintenance_work_mem
    op.execute('CREATE INDEX idx_cache_180d_projects ON embedding_cache_180d USING gin(projects)')
    op.execute('CREATE INDEX idx_cache_7d_projects ON embedding_cache_7d USING gin(projects)')
    op.execute('CREATE INDEX idx_cache_180d_alt_projects ON embedding_cache_180d_alt USING gin(projects)')
    op.execute('CREATE INDEX idx_cache_7d_alt_projects ON embedding_cache_7d_alt USING gin(projects)')


def downgrade() -> None:
    """
    Rollback: Convert projects array back to comma-separated string.
    """
    # Step 1: Add back projects column as String
    op.add_column('content', sa.Column('projects_str', sa.String(), nullable=True))

    # Step 2: Convert array back to CSV
    op.execute("""
        UPDATE content
        SET projects_str = array_to_string(projects, ',')
        WHERE projects IS NOT NULL
    """)

    # Step 3: Make projects_str NOT NULL
    op.alter_column('content', 'projects_str', nullable=False)

    # Step 4: Drop array indexes
    op.drop_index('idx_content_projects', table_name='content')
    op.drop_index('idx_content_projects_gin', table_name='content')

    # Step 5: Drop array column
    op.drop_column('content', 'projects')

    # Step 6: Rename projects_str back to projects
    op.alter_column('content', 'projects_str', new_column_name='projects')

    # Step 7: Recreate old indexes
    op.create_index('idx_content_projects', 'content', ['projects'])
    op.execute('CREATE INDEX idx_content_projects_gin ON content USING gin(projects gin_trgm_ops)')
