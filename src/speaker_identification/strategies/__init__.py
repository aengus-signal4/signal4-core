"""
Speaker Identification Strategies
==================================

Each strategy is independent, idempotent, and returns confidence scores.
Strategies run in priority order to maximize coverage.

Phase 1: Metadata Identification
- metadata_identification.py: Extract speakers from channel/episode metadata

Phase 2: Host Embedding Identification
- host_embedding_identification.py: Match speakers to hosts using embeddings
- hosts/: Host verification strategies
  - single_host.py: SingleHostStrategy for single-host channels
  - multi_host.py: MultiHostStrategy for multi-host channels

See SPEAKER_IDENTIFICATION.md for full documentation.
"""

from .base import (
    HostStrategyContext,
    ClusterVerificationResult,
    HostVerificationStrategy
)
from .hosts import SingleHostStrategy, MultiHostStrategy

__all__ = [
    'HostStrategyContext',
    'ClusterVerificationResult',
    'HostVerificationStrategy',
    'SingleHostStrategy',
    'MultiHostStrategy'
]
