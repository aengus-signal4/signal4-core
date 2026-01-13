"""drop_cprmv_analysis_table_after_migration

Revision ID: 1986110edb9d
Revises: cb83ead5fde1
Create Date: 2025-11-23 06:20:13.288014

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '1986110edb9d'
down_revision: Union[str, None] = 'cb83ead5fde1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop old cprmv_analysis table after migrating data to theme_classifications
    # Data has been migrated using scripts/migrate_cprmv_legacy_data.py
    # All 165,306 records migrated to theme_classifications with schema_id=2 (CPRMV v1.0_legacy)
    op.drop_table('cprmv_analysis')


def downgrade() -> None:
    # Recreate cprmv_analysis table structure if needed to rollback
    # NOTE: This will NOT restore the data - data recovery would require backup restore
    op.execute("""
        CREATE TABLE cprmv_analysis (
            id SERIAL PRIMARY KEY,
            segment_id INTEGER NOT NULL UNIQUE REFERENCES embedding_segments(id),
            themes VARCHAR[] NOT NULL,
            confidence_scores JSONB NOT NULL,
            high_confidence_themes VARCHAR[] NOT NULL,
            matched_via VARCHAR(20),
            similarity_score DOUBLE PRECISION,
            matched_keywords TEXT,
            embedding vector(2000),
            created_at TIMESTAMP NOT NULL,
            updated_at TIMESTAMP NOT NULL
        )
    """)

    # Recreate indexes
    op.execute("CREATE INDEX idx_cprmv_analysis_segment_id ON cprmv_analysis(segment_id)")
    op.execute("CREATE INDEX idx_cprmv_analysis_matched_via ON cprmv_analysis(matched_via)")
    op.execute("CREATE INDEX idx_cprmv_analysis_themes ON cprmv_analysis USING gin(themes)")
    op.execute("CREATE INDEX idx_cprmv_analysis_high_conf_themes ON cprmv_analysis USING gin(high_confidence_themes)")
