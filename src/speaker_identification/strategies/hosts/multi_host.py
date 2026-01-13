"""
Multi Host Strategy
===================

Strategy for channels with MULTIPLE co-hosts (e.g., CANADALAND, Breakfast Club).

Logic:
1. Compute metadata overlap (which host's episodes does this cluster appear in?)
2. Always use LLM with metadata context
3. LLM must find transcript evidence (self-ID, addressed by name)

Unlike single-host, we cannot rely on embedding similarity because there are
multiple host voices to distinguish between. Metadata overlap provides strong
context for the LLM.
"""

import numpy as np
from typing import Dict, List, Set

from ..base import (
    HostVerificationStrategy,
    HostStrategyContext,
    ClusterVerificationResult
)
from src.utils.logger import setup_worker_logger

logger = setup_worker_logger('speaker_identification.multi_host_strategy')


class MultiHostStrategy(HostVerificationStrategy):
    """
    Strategy for multi-host channels.

    Always uses LLM verification with metadata context. Metadata overlap
    (which host's episodes this cluster appears in) provides strong context
    but is not sufficient for identification - we need transcript evidence.
    """

    # For multi-host, we want to find ALL hosts, so check many clusters
    MAX_CLUSTERS_TO_CHECK = 50  # Check up to 50 clusters to find all hosts

    async def verify_cluster(
        self,
        ctx: HostStrategyContext
    ) -> ClusterVerificationResult:
        """
        Verify cluster identity using LLM with metadata context.

        Args:
            ctx: HostStrategyContext

        Returns:
            ClusterVerificationResult
        """
        # Compute metadata overlap
        metadata_matches = self._compute_metadata_match(ctx)

        # Log metadata context
        if metadata_matches:
            best_match = metadata_matches[0]
            overlap_pct = best_match['overlap_ratio'] * 100
            logger.info(f"    Metadata suggests: {best_match['host_name']} ({overlap_pct:.0f}% overlap)")

        # Always use LLM with metadata context
        ctx.stats['llm_with_metadata_context'] = ctx.stats.get('llm_with_metadata_context', 0) + 1

        result = await self._verify_cluster_with_llm(
            ctx,
            metadata_matches=metadata_matches
        )

        if result.identified_host != 'unknown':
            ctx.stats['metadata_llm_verified'] = ctx.stats.get('metadata_llm_verified', 0) + 1

        # Handle 'probably' confidence
        if result.needs_retry():
            result = await self.handle_probably_retry(
                result, ctx, metadata_matches=metadata_matches
            )

        return result

    def _compute_metadata_match(
        self,
        ctx: HostStrategyContext
    ) -> List[Dict]:
        """
        Match cluster to hosts using episode metadata overlap.

        For multi-host channels, this is the primary context signal.
        Computes overlap ratio (what % of cluster episodes have this host)
        and coverage ratio (what % of host's episodes have this cluster).

        Args:
            ctx: HostStrategyContext

        Returns:
            List of matches sorted by overlap_ratio descending:
            [
                {
                    'host_name': str,
                    'overlap_ratio': float,
                    'coverage_ratio': float,
                    'overlap_count': int,
                    'cluster_episode_count': int,
                    'host_episode_count': int
                }
            ]
        """
        # Get unique episodes in this cluster
        cluster_episodes = set(
            ctx.speakers[i]['content_id'] for i in ctx.cluster_indices
        )
        cluster_episode_count = len(cluster_episodes)

        if cluster_episode_count == 0:
            return []

        matches = []
        for host in ctx.qualified_hosts:
            host_episodes = set(host['episode_ids'])
            overlap = cluster_episodes & host_episodes

            if not overlap:
                continue

            overlap_ratio = len(overlap) / cluster_episode_count
            coverage_ratio = len(overlap) / len(host_episodes) if host_episodes else 0

            matches.append({
                'host_name': host['name'],
                'overlap_ratio': overlap_ratio,
                'coverage_ratio': coverage_ratio,
                'overlap_count': len(overlap),
                'cluster_episode_count': cluster_episode_count,
                'host_episode_count': len(host_episodes)
            })

        # Sort by overlap ratio (what % of this cluster's episodes have this host)
        matches.sort(key=lambda x: x['overlap_ratio'], reverse=True)

        return matches

    def should_stop_search(
        self,
        clusters_checked: int,
        hosts_found: int,
        total_clusters: int,
        target_found: bool
    ) -> bool:
        """
        For multi-host channels, keep searching until we've checked enough clusters.

        We want to find ALL hosts, so don't stop early based on hosts found.
        Only stop after checking MAX_CLUSTERS_TO_CHECK clusters.

        Args:
            clusters_checked: Number checked so far
            hosts_found: Total hosts found
            total_clusters: Total available
            target_found: Whether target host was found

        Returns:
            True if should stop
        """
        # For multi-host, keep going until we've checked enough clusters
        # Don't stop early just because we found some hosts
        return clusters_checked >= self.MAX_CLUSTERS_TO_CHECK

    def get_max_clusters_to_check(self, total_clusters: int) -> int:
        """
        Return maximum clusters to check.

        For multi-host, check all clusters (up to reasonable limit).
        """
        return min(total_clusters, self.MAX_CLUSTERS_TO_CHECK)

    def get_metadata_confidence(self, metadata_matches: List[Dict]) -> str:
        """
        Determine confidence level based on metadata overlap.

        Args:
            metadata_matches: Sorted matches from _compute_metadata_match

        Returns:
            'high', 'medium', 'low', or 'none'
        """
        if not metadata_matches:
            return 'none'

        best = metadata_matches[0]

        # High: 90%+ overlap AND decent coverage (40%+)
        if best['overlap_ratio'] >= 0.90 and best['coverage_ratio'] >= 0.40:
            return 'high'
        # Medium: 70-90% overlap
        elif best['overlap_ratio'] >= 0.70:
            return 'medium'
        # Low: <70% overlap
        else:
            return 'low'

    def is_ambiguous(self, metadata_matches: List[Dict]) -> bool:
        """
        Check if multiple hosts have significant overlap (ambiguous).

        Args:
            metadata_matches: Sorted matches

        Returns:
            True if ambiguous (multiple strong matches)
        """
        if len(metadata_matches) < 2:
            return False

        # Ambiguous if second-best also has >50% overlap
        return metadata_matches[1]['overlap_ratio'] > 0.50
