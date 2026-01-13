"""add hosts and guests fields to channels and content

Revision ID: ff5653eb37c3
Revises: 7fa146fb537f
Create Date: 2025-11-24 05:51:25.944864

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ff5653eb37c3'
down_revision: Union[str, None] = '7fa146fb537f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add hosts field to channels table (regular hosts)
    op.execute("""
        ALTER TABLE channels
        ADD COLUMN IF NOT EXISTS hosts JSONB DEFAULT '[]'::jsonb;

        CREATE INDEX IF NOT EXISTS idx_channels_hosts ON channels USING gin(hosts);

        COMMENT ON COLUMN channels.hosts IS 'Array of regular host objects: [{"name": "Andrew Huberman", "confidence": "certain", "reasoning": "..."}]';
    """)

    # Add speakers field to content table (hosts and guests for specific episode)
    op.execute("""
        ALTER TABLE content
        ADD COLUMN IF NOT EXISTS speakers JSONB DEFAULT '[]'::jsonb;

        CREATE INDEX IF NOT EXISTS idx_content_speakers ON content USING gin(speakers);

        COMMENT ON COLUMN content.speakers IS 'Array of speaker objects for this episode: [{"name": "Andrew Huberman", "role": "host", "confidence": "certain", "reasoning": "..."}]';
    """)


def downgrade() -> None:
    op.execute("""
        DROP INDEX IF EXISTS idx_content_speakers;
        DROP INDEX IF EXISTS idx_channels_hosts;

        ALTER TABLE content DROP COLUMN IF EXISTS speakers;
        ALTER TABLE channels DROP COLUMN IF EXISTS hosts;
    """)
