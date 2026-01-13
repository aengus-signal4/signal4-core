"""Refactor speaker identification schema - clean slate

Simplifies speaker identification to:
- speaker_identity_id: Final assigned identity
- assignment_confidence: 0-1 confidence score
- assignment_phase: Which phase made the assignment ('phase2'|'phase3'|'phase4'|'manual')
- identification_details: JSONB audit trail for all phases

Changes:
1. Drop indexes FIRST for fast UPDATE
2. Add assignment_phase column
3. Rename llm_identification -> identification_details
4. Clear all existing speaker assignments
5. Clear all existing speaker_identities (fresh start)
6. Remove redundant columns
7. Recreate indexes

Revision ID: refactor_speaker_id_schema
Revises: add_text_evidence_columns
Create Date: 2025-12-12
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers
revision = 'refactor_speaker_id_schema'
down_revision = 'add_text_evidence_columns'
branch_labels = None
depends_on = None


def upgrade():
    # 1. DROP ALL INDEXES FIRST - makes UPDATE much faster
    op.execute("DROP INDEX IF EXISTS idx_speakers_identity")
    op.execute("DROP INDEX IF EXISTS idx_speakers_identity_count")
    op.execute("DROP INDEX IF EXISTS idx_speakers_llm_identification")
    op.execute("DROP INDEX IF EXISTS idx_speakers_identification_status")
    op.execute("DROP INDEX IF EXISTS idx_speakers_unprocessed")
    op.execute("DROP INDEX IF EXISTS idx_speakers_unassigned_with_embedding")
    op.execute("DROP INDEX IF EXISTS idx_speakers_retry_eligible")
    op.execute("DROP INDEX IF EXISTS idx_speakers_text_evidence_status")
    op.execute("DROP INDEX IF EXISTS idx_speakers_text_evidence_certain")
    op.execute("DROP INDEX IF EXISTS idx_speakers_assignment_source")

    # 2. Add new column
    op.add_column('speakers', sa.Column('assignment_phase', sa.String(20), nullable=True))

    # 3. Rename llm_identification -> identification_details
    op.alter_column('speakers', 'llm_identification', new_column_name='identification_details')

    # 4. Clear all existing assignments (clean slate) - FAST now without indexes
    op.execute("""
        UPDATE speakers SET
            speaker_identity_id = NULL,
            assignment_confidence = NULL,
            assignment_phase = NULL,
            identification_details = '{}'::jsonb
    """)

    # 5. Clear speaker_identities (fresh start - centroids will be rebuilt)
    op.execute("DELETE FROM speaker_identities")

    # 6. Drop redundant columns
    op.drop_column('speakers', 'text_evidence_status')
    op.drop_column('speakers', 'evidence_type')
    op.drop_column('speakers', 'evidence_quote')
    op.drop_column('speakers', 'assignment_method')
    op.drop_column('speakers', 'assignment_source')
    op.drop_column('speakers', 'identification_status')

    # 7. Recreate essential indexes
    op.execute("CREATE INDEX idx_speakers_identity ON speakers(speaker_identity_id)")
    op.execute("CREATE INDEX idx_speakers_identity_count ON speakers(speaker_identity_id) WHERE speaker_identity_id IS NOT NULL")
    op.create_index('idx_speakers_assignment_phase', 'speakers', ['assignment_phase'])
    op.create_index('idx_speakers_identification_details', 'speakers', ['identification_details'], postgresql_using='gin')

    # 8. Index for finding unprocessed speakers (no phase2 yet)
    op.execute("""
        CREATE INDEX idx_speakers_phase2_pending ON speakers (id)
        WHERE identification_details->'phase2' IS NULL
          AND embedding IS NOT NULL
    """)

    # 9. Index for phase2 certain (for phase3 centroid building)
    op.execute("""
        CREATE INDEX idx_speakers_phase2_certain ON speakers (id)
        WHERE identification_details->'phase2'->>'status' = 'certain'
    """)

    # 10. Index for unassigned speakers (for phase4)
    op.execute("""
        CREATE INDEX idx_speakers_unassigned ON speakers (id)
        WHERE speaker_identity_id IS NULL
          AND embedding IS NOT NULL
    """)


def downgrade():
    # Re-add dropped columns
    op.add_column('speakers', sa.Column('text_evidence_status', sa.String(30), nullable=True))
    op.add_column('speakers', sa.Column('evidence_type', sa.String(50), nullable=True))
    op.add_column('speakers', sa.Column('evidence_quote', sa.Text, nullable=True))
    op.add_column('speakers', sa.Column('assignment_method', sa.String(50), nullable=True))
    op.add_column('speakers', sa.Column('assignment_source', sa.String(50), nullable=True))
    op.add_column('speakers', sa.Column('identification_status', sa.String(50), nullable=False, server_default='unprocessed'))

    # Rename back
    op.alter_column('speakers', 'identification_details', new_column_name='llm_identification')

    # Drop new column
    op.drop_column('speakers', 'assignment_phase')

    # Drop new indexes
    op.execute("DROP INDEX IF EXISTS idx_speakers_assignment_phase")
    op.execute("DROP INDEX IF EXISTS idx_speakers_identification_details")
    op.execute("DROP INDEX IF EXISTS idx_speakers_phase2_pending")
    op.execute("DROP INDEX IF EXISTS idx_speakers_phase2_certain")
    op.execute("DROP INDEX IF EXISTS idx_speakers_unassigned")

    # Recreate old indexes
    op.create_index('idx_speakers_text_evidence_status', 'speakers', ['text_evidence_status'])
    op.create_index('idx_speakers_assignment_source', 'speakers', ['assignment_source'])
    op.create_index('idx_speakers_identification_status', 'speakers', ['identification_status'])
