"""
Base Strategy Classes for Speaker Identification
=================================================

Abstract base classes for the speaker identification strategy pattern.
See SPEAKER_IDENTIFICATION.md for full documentation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Callable, Tuple
import numpy as np


@dataclass
class ClusterVerificationResult:
    """Result from verifying a speaker cluster."""

    # Identification (backwards compat - use speaker_name if available)
    identified_host: str  # Name if host/co_host role, otherwise 'unknown'
    confidence: str  # 'certain', 'very_likely', 'probably', 'unlikely'
    reasoning: str

    # New fields for discovered hosts
    speaker_name: str = 'unknown'  # Actual identified name (may differ from identified_host)
    role: str = 'unknown'  # 'host', 'co_host', 'guest', 'unknown'
    is_expected_host: bool = False  # True if speaker_name matches expected hosts from metadata

    # For retry logic
    sampled_content_ids: List[str] = field(default_factory=list)

    # Control flow
    skip_remaining: bool = False  # True = stop checking more clusters

    def is_positive(self) -> bool:
        """Return True if identification was successful with high confidence."""
        return (
            self.identified_host != 'unknown' and
            self.confidence in ['certain', 'very_likely']
        )

    def is_discovered_host(self) -> bool:
        """Return True if speaker was identified as host but not in expected hosts."""
        return (
            self.speaker_name != 'unknown' and
            self.role in ['host', 'co_host'] and
            not self.is_expected_host and
            self.confidence in ['certain', 'very_likely']
        )

    def needs_retry(self) -> bool:
        """Return True if 'probably' confidence should trigger retry."""
        return (
            self.identified_host != 'unknown' and
            self.confidence == 'probably'
        )


@dataclass
class HostStrategyContext:
    """Context passed to host verification strategies."""

    # Channel info
    channel_id: int
    all_host_names: List[str]
    qualified_hosts: List[Dict]  # Full host data with episode_ids

    # Current target
    target_host_name: str

    # Cluster being verified
    cluster_indices: Set[int]
    cluster_result: Dict  # {'members': List[int], 'episode_count': int}

    # Speaker data
    speakers: List[Dict]
    embeddings: np.ndarray

    # State from orchestrator
    existing_centroids: Dict[str, Dict]  # host_name -> centroid data
    stats: Dict[str, Any]  # Shared stats dict to update


class HostVerificationStrategy(ABC):
    """
    Abstract base class for host cluster verification strategies.

    Subclasses implement different verification approaches:
    - SingleHostStrategy: Embedding similarity after first LLM verify
    - MultiHostStrategy: Always LLM with metadata context
    """

    def __init__(
        self,
        llm_client,
        context_builder,
        select_best_for_centroid_fn: Callable
    ):
        """
        Initialize strategy.

        Args:
            llm_client: LLM client for verification calls
            context_builder: Builder for transcript context
            select_best_for_centroid_fn: Function(cluster_indices, speakers, embeddings)
                                         -> (centroid, qualities, avg_quality)
        """
        self.llm_client = llm_client
        self.context_builder = context_builder
        self._select_best_for_centroid = select_best_for_centroid_fn

    @abstractmethod
    async def verify_cluster(
        self,
        ctx: HostStrategyContext
    ) -> ClusterVerificationResult:
        """
        Verify if a cluster belongs to the target host.

        Args:
            ctx: HostStrategyContext with all needed data

        Returns:
            ClusterVerificationResult with verdict
        """
        pass

    @abstractmethod
    def should_stop_search(
        self,
        clusters_checked: int,
        hosts_found: int,
        total_clusters: int,
        target_found: bool
    ) -> bool:
        """
        Return True if we should stop checking more clusters.

        Args:
            clusters_checked: Number of clusters already checked
            hosts_found: Number of hosts found so far (total)
            total_clusters: Total clusters available
            target_found: Whether we found the target host

        Returns:
            True if search should stop
        """
        pass

    @abstractmethod
    def get_max_clusters_to_check(self, total_clusters: int) -> int:
        """
        Return maximum number of clusters to check.

        Args:
            total_clusters: Total clusters available

        Returns:
            Maximum clusters to check
        """
        pass

    def compute_cluster_centroid(
        self,
        cluster_indices: Set[int],
        speakers: List[Dict],
        embeddings: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Compute centroid for a cluster using best embeddings.

        Args:
            cluster_indices: Indices of speakers in cluster
            speakers: Full speaker list with quality scores
            embeddings: Full embedding matrix

        Returns:
            (centroid, avg_quality)
        """
        centroid, _, avg_quality = self._select_best_for_centroid(
            cluster_indices, speakers, embeddings
        )
        return centroid, avg_quality

    async def _verify_cluster_with_llm(
        self,
        ctx: HostStrategyContext,
        metadata_matches: List[Dict] = None,
        exclude_content_ids: List[str] = None
    ) -> ClusterVerificationResult:
        """
        Call LLM to verify cluster identity.

        Samples transcript excerpts and asks LLM to identify the speaker.

        Args:
            ctx: Strategy context
            metadata_matches: Optional metadata overlap data (for multi-host)
            exclude_content_ids: Content IDs to exclude (for retry)

        Returns:
            ClusterVerificationResult
        """
        import random

        cluster_indices = list(ctx.cluster_indices)

        # Filter out excluded content_ids if doing a retry
        if exclude_content_ids:
            available_indices = [
                idx for idx in cluster_indices
                if ctx.speakers[idx]['content_id'] not in exclude_content_ids
            ]
            if len(available_indices) < 2:
                return ClusterVerificationResult(
                    identified_host='unknown',
                    confidence='unlikely',
                    reasoning='Not enough different episodes for retry',
                    sampled_content_ids=[]
                )
        else:
            available_indices = cluster_indices

        # Sample up to 5 speakers from the cluster for better name evidence coverage
        sample_size = min(5, len(available_indices))
        sampled_indices = random.sample(available_indices, sample_size)

        # Get transcript context for sampled speakers
        transcript_samples = []
        for idx in sampled_indices:
            speaker = ctx.speakers[idx]
            context = self.context_builder.get_speaker_transcript_context(speaker['id'])
            if context:
                transcript_samples.append({
                    'speaker_id': speaker['id'],
                    'content_id': speaker['content_id'],
                    'episode_title': context.get('episode_title', ''),
                    'duration_pct': context.get('duration_pct', 0),
                    'total_turns': context.get('total_turns', 0),
                    'first_utterance': context.get('first_utterance', ''),
                    'last_utterance': context.get('last_utterance', '')
                })

        sampled_content_ids = [s['content_id'] for s in transcript_samples]

        if not transcript_samples:
            return ClusterVerificationResult(
                identified_host='unknown',
                confidence='unlikely',
                reasoning='Could not get transcript samples',
                sampled_content_ids=[]
            )

        # Calculate total unique episodes across all qualified hosts
        # This is used for dominance calculation (host episodes / total episodes)
        all_episode_ids = set()
        for host in ctx.qualified_hosts:
            all_episode_ids.update(host.get('episode_ids', []))
        total_channel_episodes = len(all_episode_ids)

        # Build verification context
        verification_context = {
            'all_host_names': ctx.all_host_names,
            'cluster_size': len(ctx.cluster_indices),
            'transcript_samples': transcript_samples,
            'qualified_hosts': ctx.qualified_hosts,
            'target_host': ctx.target_host_name,
            'total_channel_episodes': total_channel_episodes
        }

        # Add metadata matches for multi-host
        if metadata_matches:
            verification_context['metadata_matches'] = metadata_matches

        # Call LLM
        result = await self.llm_client.verify_cluster_is_host(verification_context)

        return ClusterVerificationResult(
            identified_host=result.get('identified_host', 'unknown'),
            confidence=result.get('confidence', 'unlikely'),
            reasoning=result.get('reasoning', ''),
            speaker_name=result.get('speaker_name', 'unknown'),
            role=result.get('role', 'unknown'),
            is_expected_host=result.get('is_expected_host', False),
            sampled_content_ids=sampled_content_ids
        )

    async def handle_probably_retry(
        self,
        initial_result: ClusterVerificationResult,
        ctx: HostStrategyContext,
        metadata_matches: List[Dict] = None
    ) -> ClusterVerificationResult:
        """
        Handle 'probably' confidence by retrying with different samples.

        Args:
            initial_result: Result with 'probably' confidence
            ctx: Strategy context
            metadata_matches: Optional metadata (for multi-host)

        Returns:
            Updated result (upgraded or downgraded)
        """
        ctx.stats['probably_retries'] = ctx.stats.get('probably_retries', 0) + 1

        retry_result = await self._verify_cluster_with_llm(
            ctx,
            metadata_matches=metadata_matches,
            exclude_content_ids=initial_result.sampled_content_ids
        )

        if retry_result.confidence in ['certain', 'very_likely']:
            ctx.stats['probably_upgraded'] = ctx.stats.get('probably_upgraded', 0) + 1
            return retry_result
        else:
            ctx.stats['probably_downgraded'] = ctx.stats.get('probably_downgraded', 0) + 1
            return ClusterVerificationResult(
                identified_host='unknown',
                confidence='unlikely',
                reasoning='Retry did not upgrade confidence'
            )
