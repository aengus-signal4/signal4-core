"""Update speaker processing status enum

Revision ID: update_speaker_processing_status
Revises: add_rebase_tracking_fields
Create Date: 2025-01-10

This migration updates the rebase_status enum to include new values
that track the 3-phase speaker management pipeline:
- PENDING: New from stitch, needs clustering (Phase 1)
- CLUSTERED: Phase 1 complete - has speaker_identity_id
- IDENTIFIED: Phase 2 complete - identity has primary_name
- VALIDATED: Phase 3 complete - checked for merges (optional)

Old values PROCESSING, COMPLETED, FAILED are deprecated but kept for compatibility.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import text


def upgrade():
    """Add new enum values for speaker processing status"""

    # Get a connection to execute raw SQL
    connection = op.get_bind()

    # Add new enum values if they don't exist
    # PostgreSQL doesn't have IF NOT EXISTS for ALTER TYPE ADD VALUE in older versions
    # So we try and catch the error if it already exists
    try:
        connection.execute(text("""
            ALTER TYPE rebasestatus ADD VALUE IF NOT EXISTS 'CLUSTERED';
        """))
        connection.commit()
    except Exception as e:
        if "already exists" not in str(e):
            raise
        connection.rollback()

    try:
        connection.execute(text("""
            ALTER TYPE rebasestatus ADD VALUE IF NOT EXISTS 'IDENTIFIED';
        """))
        connection.commit()
    except Exception as e:
        if "already exists" not in str(e):
            raise
        connection.rollback()

    try:
        connection.execute(text("""
            ALTER TYPE rebasestatus ADD VALUE IF NOT EXISTS 'VALIDATED';
        """))
        connection.commit()
    except Exception as e:
        if "already exists" not in str(e):
            raise
        connection.rollback()


def downgrade():
    """
    Note: PostgreSQL does not support removing enum values.
    To downgrade, you would need to:
    1. Create a new enum type without the values
    2. Update all columns to use the new type
    3. Drop the old type
    4. Rename the new type

    This is not implemented as it's destructive and rarely needed.
    """
    pass
