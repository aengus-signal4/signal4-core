"""cleanup_speaker_transcriptions_remove_legacy_fields

Revision ID: 634f20795101
Revises: 1d0d4d80d505
Create Date: 2025-07-12 08:05:51.975368

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '634f20795101'
down_revision: Union[str, None] = '1d0d4d80d505'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Remove legacy speaker identification fields that are no longer needed
    # We now use speaker_hash as the primary identifier and speaker_id as FK
    
    # Drop old indexes first
    op.execute("DROP INDEX IF EXISTS idx_speaker_transcriptions_diarization;")
    
    # Remove legacy columns
    op.execute("ALTER TABLE speaker_transcriptions DROP COLUMN IF EXISTS speaker_global_id;")
    op.execute("ALTER TABLE speaker_transcriptions DROP COLUMN IF EXISTS diarization_speaker_id;")


def downgrade() -> None:
    # Re-add the removed columns for rollback
    op.execute("ALTER TABLE speaker_transcriptions ADD COLUMN IF NOT EXISTS speaker_global_id VARCHAR(255);")
    op.execute("ALTER TABLE speaker_transcriptions ADD COLUMN IF NOT EXISTS diarization_speaker_id VARCHAR(20);")
    
    # Re-create indexes
    op.execute("CREATE INDEX IF NOT EXISTS idx_speaker_transcriptions_diarization ON speaker_transcriptions (diarization_speaker_id);")
