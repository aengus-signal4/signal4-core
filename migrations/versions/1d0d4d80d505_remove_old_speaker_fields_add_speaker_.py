"""remove_old_speaker_fields_add_speaker_hash

Revision ID: 1d0d4d80d505
Revises: optimize_clustering_indexes
Create Date: 2025-07-12 07:52:50.783227

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '1d0d4d80d505'
down_revision: Union[str, None] = 'optimize_clustering_indexes'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enable pgcrypto extension for digest function
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")
    
    # Add speaker_hash column if it doesn't exist
    op.execute("ALTER TABLE speakers ADD COLUMN IF NOT EXISTS speaker_hash VARCHAR(64) UNIQUE;")
    
    # Populate speaker_hash for existing records (deterministic: content_id:local_speaker_id)
    op.execute("""
        UPDATE speakers 
        SET speaker_hash = encode(digest(content_id || ':' || local_speaker_id, 'sha256'), 'hex')
        WHERE speaker_hash IS NULL;
    """)
    
    # Make speaker_hash NOT NULL
    op.execute("ALTER TABLE speakers ALTER COLUMN speaker_hash SET NOT NULL;")
    
    # Create index on speaker_hash
    op.execute("CREATE INDEX IF NOT EXISTS idx_speakers_speaker_hash ON speakers (speaker_hash);")
    
    # Remove old fields that are no longer used
    op.execute("ALTER TABLE speakers DROP COLUMN IF EXISTS global_id;")
    op.execute("ALTER TABLE speakers DROP COLUMN IF EXISTS universal_name;")
    
    # Drop old indexes that may exist on removed columns
    op.execute("DROP INDEX IF EXISTS idx_speakers_global_id;")
    op.execute("DROP INDEX IF EXISTS idx_speakers_universal_name;")


def downgrade() -> None:
    # Re-add the removed columns
    op.execute("ALTER TABLE speakers ADD COLUMN IF NOT EXISTS global_id VARCHAR(255);")
    op.execute("ALTER TABLE speakers ADD COLUMN IF NOT EXISTS universal_name VARCHAR(255);")
    
    # Re-create indexes
    op.execute("CREATE INDEX IF NOT EXISTS idx_speakers_global_id ON speakers (global_id);")
    op.execute("CREATE INDEX IF NOT EXISTS idx_speakers_universal_name ON speakers (universal_name);")
    
    # Remove speaker_hash column and index
    op.execute("DROP INDEX IF EXISTS idx_speakers_speaker_hash;")
    op.execute("ALTER TABLE speakers DROP COLUMN IF EXISTS speaker_hash;")
