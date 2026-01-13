"""Add identification_status column to speakers table

Revision ID: add_identification_status
Revises: 1791194feef3
Create Date: 2025-12-06

Adds identification_status column to track speaker evaluation state across
the identification pipeline. This prevents re-processing of speakers that
have already been evaluated and rejected.

Status values:
- unprocessed: Never evaluated (DEFAULT)
- assigned: Successfully linked to speaker_identity
- rejected_low_similarity: Embedding match < 0.40
- rejected_no_context: No transcript for LLM
- rejected_llm_unverified: LLM could not verify
- rejected_unknown: LLM returned "unknown"
- rejected_short_duration: Below duration threshold
- rejected_poor_embedding: Embedding quality too low
- pending_review: Flagged for human review
- retry_eligible: "probably" confidence, worth retry
"""

from alembic import op
import sqlalchemy as sa

revision = 'add_identification_status'
down_revision = '1791194feef3'
branch_labels = None
depends_on = None


def upgrade():
    # Step 1: Add column with default 'unprocessed'
    op.add_column(
        'speakers',
        sa.Column('identification_status', sa.String(50), nullable=False, server_default='unprocessed')
    )

    # Step 2: Backfill existing data based on current state
    # Speakers with speaker_identity_id are "assigned"
    op.execute("""
        UPDATE speakers
        SET identification_status = 'assigned'
        WHERE speaker_identity_id IS NOT NULL
          AND identification_status = 'unprocessed'
    """)

    # Speakers with llm_identification "probably" confidence are retry_eligible
    op.execute("""
        UPDATE speakers
        SET identification_status = 'retry_eligible'
        WHERE speaker_identity_id IS NULL
          AND llm_identification IS NOT NULL
          AND llm_identification->>'confidence' = 'probably'
          AND identification_status = 'unprocessed'
    """)

    # Speakers with llm_identification "unlikely" confidence are rejected
    op.execute("""
        UPDATE speakers
        SET identification_status = 'rejected_llm_unverified'
        WHERE speaker_identity_id IS NULL
          AND llm_identification IS NOT NULL
          AND llm_identification->>'confidence' = 'unlikely'
          AND identification_status = 'unprocessed'
    """)

    # Speakers with llm_identification returning "unknown" name
    op.execute("""
        UPDATE speakers
        SET identification_status = 'rejected_unknown'
        WHERE speaker_identity_id IS NULL
          AND llm_identification IS NOT NULL
          AND (
              llm_identification->>'identified_name' = 'unknown'
              OR llm_identification->>'speaker_name' = 'unknown'
          )
          AND identification_status = 'unprocessed'
    """)

    # Remaining speakers with llm_identification but no identity are rejected
    op.execute("""
        UPDATE speakers
        SET identification_status = 'rejected_llm_unverified'
        WHERE speaker_identity_id IS NULL
          AND llm_identification IS NOT NULL
          AND identification_status = 'unprocessed'
    """)

    # Step 3: Create indexes
    # Primary status index
    op.create_index(
        'idx_speakers_identification_status',
        'speakers',
        ['identification_status']
    )

    # Partial index for unprocessed speakers (most common query pattern)
    op.execute("""
        CREATE INDEX idx_speakers_unprocessed
        ON speakers (embedding_quality_score DESC, duration DESC, content_id)
        WHERE identification_status = 'unprocessed' AND embedding IS NOT NULL
    """)

    # Partial index for retry-eligible speakers (Phase 6)
    op.execute("""
        CREATE INDEX idx_speakers_retry_eligible
        ON speakers (content_id, updated_at)
        WHERE identification_status IN ('retry_eligible', 'rejected_llm_unverified')
    """)

    # Update statistics
    op.execute('ANALYZE speakers')


def downgrade():
    op.drop_index('idx_speakers_retry_eligible', table_name='speakers')
    op.drop_index('idx_speakers_unprocessed', table_name='speakers')
    op.drop_index('idx_speakers_identification_status', table_name='speakers')
    op.drop_column('speakers', 'identification_status')
