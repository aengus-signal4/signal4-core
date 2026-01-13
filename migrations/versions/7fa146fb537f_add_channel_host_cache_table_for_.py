"""add channel_host_cache table for speaker identification

Revision ID: 7fa146fb537f
Revises: 1986110edb9d
Create Date: 2025-11-24 05:42:48.559153

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '7fa146fb537f'
down_revision: Union[str, None] = '1986110edb9d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create channel_host_cache table for Phase 1A host identification
    op.execute("""
        CREATE TABLE IF NOT EXISTS channel_host_cache (
            id SERIAL PRIMARY KEY,
            channel_id INTEGER NOT NULL REFERENCES channels(id) ON DELETE CASCADE,
            host_name VARCHAR(255) NOT NULL,
            confidence FLOAT NOT NULL,
            reasoning TEXT,
            method VARCHAR(50) DEFAULT 'llm_channel_description',
            identified_at TIMESTAMP DEFAULT NOW(),
            metadata JSONB DEFAULT '{}',

            UNIQUE(channel_id, host_name)
        );

        CREATE INDEX idx_channel_host_cache_channel ON channel_host_cache(channel_id);
        CREATE INDEX idx_channel_host_cache_confidence ON channel_host_cache(confidence);
        CREATE INDEX idx_channel_host_cache_identified_at ON channel_host_cache(identified_at);

        COMMENT ON TABLE channel_host_cache IS 'Caches LLM-identified hosts per channel for speaker identification';
        COMMENT ON COLUMN channel_host_cache.host_name IS 'Full name of identified host';
        COMMENT ON COLUMN channel_host_cache.confidence IS 'LLM confidence score (0.0-1.0)';
        COMMENT ON COLUMN channel_host_cache.reasoning IS 'LLM reasoning for identification';
        COMMENT ON COLUMN channel_host_cache.method IS 'Identification method (e.g., llm_channel_description)';
        COMMENT ON COLUMN channel_host_cache.metadata IS 'Additional metadata from LLM response';
    """)


def downgrade() -> None:
    op.execute("""
        DROP INDEX IF EXISTS idx_channel_host_cache_identified_at;
        DROP INDEX IF EXISTS idx_channel_host_cache_confidence;
        DROP INDEX IF EXISTS idx_channel_host_cache_channel;
        DROP TABLE IF EXISTS channel_host_cache;
    """)
