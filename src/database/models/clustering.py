"""
Clustering models for speaker identification pipeline.

Contains:
- ClusteringRun: Audit trail of clustering operations
- ClusteringBatch: Batch operation tracking
- IdentityMergeHistory: Merge/split audit trail
"""

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Text, Index, ARRAY
)
from sqlalchemy.dialects.postgresql import JSONB
from datetime import datetime

from .base import Base


class ClusteringRun(Base):
    """Audit trail of clustering operations.

    Tracks each clustering operation including parameters, statistics,
    and results for debugging and performance monitoring.
    """
    __tablename__ = 'clustering_runs'

    id = Column(Integer, primary_key=True)
    run_id = Column(String(64), unique=True, nullable=False)

    # Run metadata
    run_type = Column(String(50), nullable=False)
    method = Column(String(50), nullable=False)
    parameters = Column(JSONB, nullable=False)

    # Statistics
    embeddings_processed = Column(Integer, default=0)
    clusters_created = Column(Integer, default=0)
    assignments_made = Column(Integer, default=0)
    identities_created = Column(Integer, default=0)
    identities_merged = Column(Integer, default=0)

    # Timing
    started_at = Column(DateTime, nullable=False)
    completed_at = Column(DateTime)
    processing_time_seconds = Column(Float)

    # Status
    status = Column(String(20), default='running')
    error_message = Column(Text)

    __table_args__ = (
        Index('idx_clustering_runs_run_id', 'run_id'),
        Index('idx_clustering_runs_status', 'status', 'started_at'),
    )


class IdentityMergeHistory(Base):
    """History of speaker identity merges and splits.

    Maintains a complete audit trail of all identity merge and split
    operations for traceability and potential rollback.
    """
    __tablename__ = 'identity_merge_history'

    id = Column(Integer, primary_key=True)

    # Operation type
    operation = Column(String(20), nullable=False)  # 'merge' or 'split'

    # For merges: multiple sources -> one target
    # For splits: one source -> multiple targets
    source_identity_ids = Column(ARRAY(Integer), nullable=False)
    target_identity_ids = Column(ARRAY(Integer), nullable=False)

    # Operation details
    confidence = Column(Float)
    reason = Column(Text)
    evidence = Column(JSONB, default=dict)

    # Who/what performed it
    performed_by = Column(String(100))
    clustering_run_id = Column(String(64))

    # When
    performed_at = Column(DateTime, default=datetime.utcnow)


class ClusteringBatch(Base):
    """
    Tracks speaker clustering batch operations for monitoring and recovery.
    """
    __tablename__ = 'clustering_batches'

    id = Column(Integer, primary_key=True)
    batch_id = Column(String(50), unique=True, nullable=False)
    status = Column(String(20), default='pending', nullable=False)
    speaker_count = Column(Integer, nullable=False)
    merge_candidates_found = Column(Integer, default=0)
    merges_applied = Column(Integer, default=0)
    processing_time = Column(Float, nullable=True)
    config_params = Column(JSONB, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index('idx_clustering_batches_batch_id', 'batch_id'),
        Index('idx_clustering_batches_status', 'status'),
        Index('idx_clustering_batches_created_at', 'created_at'),
    )
