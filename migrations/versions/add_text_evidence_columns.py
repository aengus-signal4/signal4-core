"""add text evidence columns for speaker identification refactor

Revision ID: add_text_evidence_columns
Revises: add_identification_status_to_speakers
Create Date: 2025-12-11

Adds columns to track text-based evidence for speaker identification:
- speakers.text_evidence_status: Status of text evidence search ('certain', 'none', 'not_processed')
- speakers.evidence_type: Type of evidence found ('self_intro', 'addressed', 'introduced', 'none')
- speakers.evidence_quote: Actual transcript quote that proves identity
- speakers.assignment_source: How the assignment was made ('text_verified', 'embedding_propagation_auto', etc.)
- speaker_identities.centroid_source: Where the centroid came from ('text_verified', 'legacy')
- speaker_identities.text_verified_count: Number of speakers with text evidence
"""
from alembic import op
import sqlalchemy as sa

revision = 'add_text_evidence_columns'
down_revision = 'add_identification_status'
branch_labels = None
depends_on = None


def upgrade():
    # =====================
    # SPEAKERS TABLE
    # =====================

    # Text evidence tracking
    op.add_column('speakers', sa.Column(
        'text_evidence_status',
        sa.String(30),
        nullable=True,
        comment="Status of text evidence search: 'certain', 'none', 'not_processed', 'short_utterance'"
    ))

    op.add_column('speakers', sa.Column(
        'evidence_type',
        sa.String(50),
        nullable=True,
        comment="Type of evidence: 'self_intro', 'addressed', 'introduced', 'none'"
    ))

    op.add_column('speakers', sa.Column(
        'evidence_quote',
        sa.Text,
        nullable=True,
        comment="Exact transcript quote that proves speaker identity"
    ))

    # Assignment source tracking (how the assignment was made)
    op.add_column('speakers', sa.Column(
        'assignment_source',
        sa.String(50),
        nullable=True,
        comment="Source of assignment: 'text_verified', 'embedding_propagation_auto', 'embedding_propagation_llm', 'legacy'"
    ))

    # Indexes for efficient querying
    op.create_index(
        'idx_speakers_text_evidence_status',
        'speakers',
        ['text_evidence_status']
    )

    # Partial index for finding speakers with certain evidence (most common query)
    op.execute("""
        CREATE INDEX idx_speakers_text_evidence_certain
        ON speakers(text_evidence_status, id)
        WHERE text_evidence_status = 'certain'
    """)

    # Index for assignment source filtering
    op.create_index(
        'idx_speakers_assignment_source',
        'speakers',
        ['assignment_source']
    )

    # =====================
    # SPEAKER_IDENTITIES TABLE
    # =====================

    # Centroid source tracking (where did the centroid come from)
    op.add_column('speaker_identities', sa.Column(
        'centroid_source',
        sa.String(50),
        server_default='legacy',
        nullable=True,
        comment="Source of centroid: 'text_verified', 'legacy'"
    ))

    # Count of text-verified speakers contributing to this identity
    op.add_column('speaker_identities', sa.Column(
        'text_verified_count',
        sa.Integer,
        server_default='0',
        nullable=True,
        comment="Number of speakers with text evidence for this identity"
    ))

    # Index for filtering verified centroids
    op.create_index(
        'idx_identities_centroid_source',
        'speaker_identities',
        ['centroid_source']
    )

    # Partial index for finding text-verified centroids
    op.execute("""
        CREATE INDEX idx_identities_text_verified
        ON speaker_identities(id, centroid_source)
        WHERE centroid_source = 'text_verified'
    """)

    # Mark all existing data as legacy
    op.execute("UPDATE speakers SET assignment_source = 'legacy' WHERE speaker_identity_id IS NOT NULL AND assignment_source IS NULL")
    op.execute("UPDATE speaker_identities SET centroid_source = 'legacy' WHERE centroid_source IS NULL OR centroid_source = 'legacy'")


def downgrade():
    # Drop indexes first
    op.execute("DROP INDEX IF EXISTS idx_identities_text_verified")
    op.drop_index('idx_identities_centroid_source', table_name='speaker_identities')

    op.execute("DROP INDEX IF EXISTS idx_speakers_text_evidence_certain")
    op.drop_index('idx_speakers_assignment_source', table_name='speakers')
    op.drop_index('idx_speakers_text_evidence_status', table_name='speakers')

    # Drop columns
    op.drop_column('speaker_identities', 'text_verified_count')
    op.drop_column('speaker_identities', 'centroid_source')

    op.drop_column('speakers', 'assignment_source')
    op.drop_column('speakers', 'evidence_quote')
    op.drop_column('speakers', 'evidence_type')
    op.drop_column('speakers', 'text_evidence_status')
