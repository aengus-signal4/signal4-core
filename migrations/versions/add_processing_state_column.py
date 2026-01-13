"""Add processing_state column to content table

Replaces multiple boolean flags with a single linear state column.

The pipeline is fundamentally linear:
    NEW(0) → DOWNLOADED(1) → CONVERTED(2) → DIARIZED(3) → TRANSCRIBED(4)
    → STITCHED(5) → EMBEDDED(6) → IDENTIFIED(7) → COMPLETE(8) → COMPRESSED(9)

    BLOCKED(-1) for permanent failures

Benefits:
- Single source of truth (no conflicting boolean flags)
- Easy to query ("give me all content in TRANSCRIBED state")
- Clear progression (state N means all states < N are complete)
- Simple failure handling (track which state failed)

The boolean flags are kept for backwards compatibility but should be
considered deprecated. New code should use processing_state.

Revision ID: add_processing_state
Revises: abc123_importance
Create Date: 2026-01-03
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'add_processing_state'
down_revision: Union[str, None] = 'abc123_importance'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add processing_state column
    # -1 = BLOCKED, 0 = NEW, 1 = DOWNLOADED, 2 = CONVERTED, 3 = DIARIZED,
    # 4 = TRANSCRIBED, 5 = STITCHED, 6 = EMBEDDED, 7 = IDENTIFIED,
    # 8 = COMPLETE, 9 = COMPRESSED
    op.add_column(
        'content',
        sa.Column('processing_state', sa.SmallInteger(), nullable=True, default=0)
    )

    # Add failed_at_state to track where failures occurred
    op.add_column(
        'content',
        sa.Column('failed_at_state', sa.SmallInteger(), nullable=True)
    )

    # Add failure_reason for error tracking
    op.add_column(
        'content',
        sa.Column('state_failure_reason', sa.Text(), nullable=True)
    )

    # Migrate existing data from boolean flags to processing_state
    # This uses a CASE statement to determine the highest completed state
    op.execute("""
        UPDATE content SET processing_state =
            CASE
                WHEN blocked_download = true THEN -1
                WHEN is_compressed = true THEN 9
                WHEN is_embedded = true THEN 6
                WHEN is_stitched = true THEN 5
                WHEN is_transcribed = true THEN 4
                WHEN is_diarized = true THEN 3
                WHEN is_converted = true THEN 2
                WHEN is_downloaded = true THEN 1
                ELSE 0
            END
    """)

    # Now make the column non-nullable with default 0
    op.alter_column(
        'content',
        'processing_state',
        nullable=False,
        server_default='0'
    )

    # Create index for efficient state-based queries
    op.create_index(
        'idx_content_processing_state',
        'content',
        ['processing_state'],
        unique=False
    )

    # Create composite index for common query: find processable content by project
    op.create_index(
        'idx_content_state_projects',
        'content',
        ['processing_state', 'projects'],
        unique=False,
        postgresql_using='btree'
    )

    # Create partial index for content that needs processing (not blocked, not complete)
    op.create_index(
        'idx_content_needs_processing',
        'content',
        ['processing_state', 'last_updated'],
        unique=False,
        postgresql_where=sa.text('processing_state >= 0 AND processing_state < 9')
    )

    # Add a check constraint to ensure valid state values
    op.create_check_constraint(
        'ck_content_processing_state_valid',
        'content',
        'processing_state >= -1 AND processing_state <= 9'
    )


def downgrade() -> None:
    # Drop check constraint
    op.drop_constraint('ck_content_processing_state_valid', 'content', type_='check')

    # Drop indexes
    op.drop_index('idx_content_needs_processing', table_name='content')
    op.drop_index('idx_content_state_projects', table_name='content')
    op.drop_index('idx_content_processing_state', table_name='content')

    # Drop columns
    op.drop_column('content', 'state_failure_reason')
    op.drop_column('content', 'failed_at_state')
    op.drop_column('content', 'processing_state')
