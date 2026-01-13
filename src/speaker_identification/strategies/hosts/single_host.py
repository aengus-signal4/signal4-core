"""
Single Host Strategy
====================

Strategy for channels with ONE primary host (e.g., Huberman Lab, Joe Rogan).

Logic:
1. First cluster: LLM verify with behavioral inference allowed
2. Subsequent clusters: Embedding similarity to verified centroid
   - >= 0.70: Same person (merge)
   - < 0.70: Different person (skip as guest)

This is more efficient than LLM-verifying every cluster since single-host
channels only have one host voice to find.
"""

import numpy as np
from typing import Dict, List, Set

from ..base import (
    HostVerificationStrategy,
    HostStrategyContext,
    ClusterVerificationResult
)
from src.utils.logger import setup_worker_logger

logger = setup_worker_logger('speaker_identification.single_host_strategy')


class SingleHostStrategy(HostVerificationStrategy):
    """
    Strategy for single-host channels.

    Uses embedding similarity after first LLM verification to avoid
    redundant LLM calls. Once we know what the host sounds like,
    we can match additional clusters by voice.
    """

    # Threshold for merging clusters via embedding similarity
    SIMILARITY_THRESHOLD = 0.70

    # Maximum clusters to check (single host = only need to find one)
    MAX_CLUSTERS_TO_CHECK = 5

    async def verify_cluster(
        self,
        ctx: HostStrategyContext
    ) -> ClusterVerificationResult:
        """
        Verify if cluster belongs to the target host.

        Strategy:
        - If host already has centroid: use embedding similarity
        - If no centroid yet: use LLM with behavioral inference

        Args:
            ctx: HostStrategyContext

        Returns:
            ClusterVerificationResult
        """
        host_name = ctx.target_host_name

        # Check if we already have a centroid for this host
        if host_name in ctx.existing_centroids:
            return self._verify_by_embedding_similarity(ctx)

        # No centroid yet - use LLM verification
        ctx.stats['behavioral_inference_used'] = ctx.stats.get('behavioral_inference_used', 0) + 1

        result = await self._verify_cluster_with_llm(ctx, metadata_matches=None)

        # Handle 'probably' confidence
        if result.needs_retry():
            result = await self.handle_probably_retry(result, ctx, metadata_matches=None)

        return result

    def _verify_by_embedding_similarity(
        self,
        ctx: HostStrategyContext
    ) -> ClusterVerificationResult:
        """
        Verify cluster by comparing to existing host centroid.

        If similarity >= 0.70, this is likely the same person (merge).
        If similarity < 0.70, this is a different speaker (guest).

        Args:
            ctx: HostStrategyContext

        Returns:
            ClusterVerificationResult
        """
        host_name = ctx.target_host_name
        host_data = ctx.existing_centroids[host_name]
        host_centroid = host_data['centroid']

        # Compute centroid for this cluster
        cluster_centroid, _ = self.compute_cluster_centroid(
            ctx.cluster_indices, ctx.speakers, ctx.embeddings
        )

        # Compute similarity
        similarity = float(np.dot(cluster_centroid, host_centroid))

        if similarity >= self.SIMILARITY_THRESHOLD:
            ctx.stats['embedding_similarity_merged'] = ctx.stats.get('embedding_similarity_merged', 0) + 1

            logger.info(
                f"    Embedding similarity to {host_name}: {similarity:.3f} >= {self.SIMILARITY_THRESHOLD} - merging"
            )

            return ClusterVerificationResult(
                identified_host=host_name,
                confidence='very_likely',
                reasoning=f'Embedding similarity {similarity:.3f} to verified centroid',
                speaker_name=host_name,
                role='host',
                is_expected_host=True
            )
        else:
            ctx.stats['embedding_similarity_rejected'] = ctx.stats.get('embedding_similarity_rejected', 0) + 1

            logger.info(
                f"    Embedding similarity to {host_name}: {similarity:.3f} < {self.SIMILARITY_THRESHOLD} - skipping (different speaker)"
            )

            return ClusterVerificationResult(
                identified_host='unknown',
                confidence='unlikely',
                reasoning=f'Embedding similarity {similarity:.3f} too low - different speaker',
                speaker_name='unknown',
                role='unknown',
                is_expected_host=False
            )

    def should_stop_search(
        self,
        clusters_checked: int,
        hosts_found: int,
        total_clusters: int,
        target_found: bool
    ) -> bool:
        """
        Stop after checking MAX_CLUSTERS_TO_CHECK clusters.

        For single-host channels, we only need to find one host,
        so we don't need to exhaustively search all clusters.

        Args:
            clusters_checked: Number checked so far
            hosts_found: Number of hosts found (should be 0 or 1)
            total_clusters: Total available
            target_found: Whether target host was found

        Returns:
            True if should stop
        """
        return clusters_checked >= self.MAX_CLUSTERS_TO_CHECK

    def get_max_clusters_to_check(self, total_clusters: int) -> int:
        """Return maximum clusters to check."""
        return min(self.MAX_CLUSTERS_TO_CHECK, total_clusters)
