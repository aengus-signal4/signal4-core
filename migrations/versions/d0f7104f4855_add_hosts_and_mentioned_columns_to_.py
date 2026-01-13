"""add hosts and mentioned columns to content table and migrate speakers data

Revision ID: d0f7104f4855
Revises: ff5653eb37c3
Create Date: 2025-11-26

This migration:
1. Adds 'hosts' and 'mentioned' JSONB columns to content table
2. Migrates existing data from content.speakers (nested format) to the new columns
3. Clears the old speakers column after migration

The old format was:
    speakers = {"speakers": [...], "mentioned": [...]}

The new format splits into three columns:
    hosts = [{"name": "...", "confidence": "...", "reasoning": "..."}]
    guests = [{"name": "...", "confidence": "...", "reasoning": "..."}]
    mentioned = [{"name": "...", "reasoning": "..."}]
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd0f7104f4855'
down_revision: Union[str, None] = 'ff5653eb37c3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Step 1: Add hosts and mentioned columns to content table
    op.execute("""
        ALTER TABLE content
        ADD COLUMN IF NOT EXISTS hosts JSONB DEFAULT '[]'::jsonb;

        ALTER TABLE content
        ADD COLUMN IF NOT EXISTS mentioned JSONB DEFAULT '[]'::jsonb;

        CREATE INDEX IF NOT EXISTS idx_content_hosts ON content USING gin(hosts);
        CREATE INDEX IF NOT EXISTS idx_content_mentioned ON content USING gin(mentioned);

        COMMENT ON COLUMN content.hosts IS 'Array of host objects: [{"name": "...", "confidence": "certain", "reasoning": "..."}]';
        COMMENT ON COLUMN content.mentioned IS 'Array of mentioned people: [{"name": "...", "reasoning": "..."}]';
    """)

    # Step 2: Migrate existing data from speakers column
    # The old format: {"speakers": [...], "mentioned": [...]}
    # speakers array has objects with role = "host" or "guest"
    op.execute("""
        UPDATE content
        SET
            hosts = COALESCE(
                (SELECT jsonb_agg(s)
                 FROM jsonb_array_elements(speakers->'speakers') AS s
                 WHERE s->>'role' = 'host'),
                '[]'::jsonb
            ),
            guests = COALESCE(
                (SELECT jsonb_agg(s)
                 FROM jsonb_array_elements(speakers->'speakers') AS s
                 WHERE s->>'role' = 'guest'),
                '[]'::jsonb
            ),
            mentioned = COALESCE(speakers->'mentioned', '[]'::jsonb)
        WHERE speakers IS NOT NULL
          AND jsonb_typeof(speakers) = 'object'
          AND speakers ? 'speakers';
    """)

    # Step 3: Clear the old speakers column (set to empty array)
    # This keeps the column for now but clears the redundant data
    op.execute("""
        UPDATE content
        SET speakers = '[]'::jsonb
        WHERE speakers IS NOT NULL
          AND speakers::text <> '[]';
    """)


def downgrade() -> None:
    # Reconstruct the old nested format from the separate columns
    op.execute("""
        UPDATE content
        SET speakers = jsonb_build_object(
            'speakers',
            COALESCE(hosts, '[]'::jsonb) || COALESCE(guests, '[]'::jsonb),
            'mentioned',
            COALESCE(mentioned, '[]'::jsonb)
        )
        WHERE (hosts IS NOT NULL AND hosts::text <> '[]')
           OR (guests IS NOT NULL AND guests::text <> '[]')
           OR (mentioned IS NOT NULL AND mentioned::text <> '[]');
    """)

    # Drop the new columns
    op.execute("""
        DROP INDEX IF EXISTS idx_content_mentioned;
        DROP INDEX IF EXISTS idx_content_hosts;

        ALTER TABLE content DROP COLUMN IF EXISTS mentioned;
        ALTER TABLE content DROP COLUMN IF EXISTS hosts;
    """)
