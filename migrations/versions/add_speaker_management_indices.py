"""Add speaker management performance indices

Revision ID: add_speaker_mgmt_indices
Revises: 72ff30f1b825
Create Date: 2025-09-13

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = 'add_speaker_mgmt_indices'
down_revision = '72ff30f1b825'
branch_labels = None
depends_on = None


def upgrade():
    """Add indices for speaker management pipeline performance"""
    
    connection = op.get_bind()
    
    # Helper function to check if index exists
    def index_exists(index_name):
        result = connection.execute(
            sa.text(f"SELECT 1 FROM pg_indexes WHERE indexname = '{index_name}'")
        ).fetchone()
        return result is not None
    
    # Phase 1 indices - for pool assembly queries
    # Composite index for new speakers query (WHERE speaker_identity_id IS NULL AND embedding IS NOT NULL)
    if not index_exists('idx_speakers_unassigned_with_embedding'):
        op.create_index(
            'idx_speakers_unassigned_with_embedding',
            'speakers',
            ['speaker_identity_id', 'embedding_quality_score', 'duration'],
            postgresql_where=sa.text('speaker_identity_id IS NULL AND embedding IS NOT NULL')
        )
    
    # Phase 2 indices - for clustering operations (skip primary key index)
    # speakers.id already has primary key index
    
    # Index for clustering runs
    if not index_exists('idx_clustering_runs_run_id'):
        op.create_index(
            'idx_clustering_runs_run_id',
            'clustering_runs',
            ['run_id']
        )
    
    if not index_exists('idx_clustering_runs_status'):
        op.create_index(
            'idx_clustering_runs_status',
            'clustering_runs',
            ['status', 'started_at']
        )
    
    # Phase 3/4 indices - for identity operations
    if not index_exists('idx_speaker_identities_active'):
        op.create_index(
            'idx_speaker_identities_active',
            'speaker_identities',
            ['id'],
            postgresql_where=sa.text('is_active = true')
        )
    
    # Index for speaker assignments
    if not index_exists('idx_speaker_assignments_identity'):
        op.create_index(
            'idx_speaker_assignments_identity',
            'speaker_assignments',
            ['speaker_identity_id']
        )
    
    if not index_exists('idx_speaker_assignments_embedding'):
        op.create_index(
            'idx_speaker_assignments_embedding',
            'speaker_assignments',
            ['speaker_embedding_id']
        )
    
    if not index_exists('idx_speaker_assignments_run'):
        op.create_index(
            'idx_speaker_assignments_run',
            'speaker_assignments',
            ['clustering_run_id']
        )
    
    # Composite index for finding current assignments
    if not index_exists('idx_speaker_assignments_current'):
        op.create_index(
            'idx_speaker_assignments_current',
            'speaker_assignments',
            ['speaker_embedding_id', 'valid_to'],
            postgresql_where=sa.text('valid_to IS NULL')
        )
    
    # Performance index for counting speakers per identity
    if not index_exists('idx_speakers_identity_count'):
        op.create_index(
            'idx_speakers_identity_count',
            'speakers',
            ['speaker_identity_id'],
            postgresql_where=sa.text('speaker_identity_id IS NOT NULL')
        )
    
    # Index for quality filtering
    if not index_exists('idx_speakers_quality_filter'):
        op.create_index(
            'idx_speakers_quality_filter',
            'speakers',
            ['embedding_quality_score'],
            postgresql_where=sa.text('embedding IS NOT NULL AND embedding_quality_score >= 0.5')
        )
    
    # Analyze tables to update statistics
    op.execute('ANALYZE speakers')
    op.execute('ANALYZE speaker_transcriptions')
    op.execute('ANALYZE speaker_identities')
    op.execute('ANALYZE speaker_assignments')
    op.execute('ANALYZE clustering_runs')


def downgrade():
    """Remove speaker management indices"""
    
    # Drop all created indices
    op.drop_index('idx_speakers_unassigned_with_embedding', 'speakers')
    op.drop_index('idx_speaker_transcriptions_speaker_id', 'speaker_transcriptions')
    op.drop_index('idx_speaker_transcriptions_speaker_content', 'speaker_transcriptions')
    op.drop_index('idx_speakers_id_btree', 'speakers')
    op.drop_index('idx_clustering_runs_run_id', 'clustering_runs')
    op.drop_index('idx_clustering_runs_status', 'clustering_runs')
    op.drop_index('idx_speaker_identities_active', 'speaker_identities')
    op.drop_index('idx_speaker_assignments_identity', 'speaker_assignments')
    op.drop_index('idx_speaker_assignments_embedding', 'speaker_assignments')
    op.drop_index('idx_speaker_assignments_run', 'speaker_assignments')
    op.drop_index('idx_speaker_assignments_current', 'speaker_assignments')
    op.drop_index('idx_speakers_identity_count', 'speakers')
    op.drop_index('idx_speakers_quality_filter', 'speakers')