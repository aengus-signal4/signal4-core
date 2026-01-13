#!/usr/bin/env python3
"""
Host Embedding Identification (Phase 2) - Named Host Centroid Approach
========================================================================

Enhanced Phase 2 using content.hosts frequency data to seed high-confidence
host clusters, then expand to full channel matching.

Three-phase approach:
- Phase 2A: Analyze content.hosts frequency to identify qualified hosts (≥10 occurrences)
- Phase 2B: Bootstrap centroid for each named host from their labeled episodes
- Phase 2C: Expand matching to all unassigned speakers in the channel

Uses strategy pattern for different channel types:
- SingleHostStrategy: For channels with one primary host (embedding similarity after first LLM verify)
- MultiHostStrategy: For channels with multiple co-hosts (always LLM with metadata context)

See SPEAKER_IDENTIFICATION.md for full documentation.

Usage:
    # Test on single channel
    python -m src.speaker_identification.strategies.host_embedding_identification \
        --channel-id 6109 --apply

    # Run on all channels in project
    python -m src.speaker_identification.strategies.host_embedding_identification \
        --project CPRMV --apply

    # Dry run
    python -m src.speaker_identification.strategies.host_embedding_identification \
        --channel-id 6109
"""

import argparse
import asyncio
import sys
from pathlib import Path
from src.utils.paths import get_project_root, get_config_path
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime
from collections import defaultdict, deque

import numpy as np
import faiss

# Add project root to path
project_root = str(get_project_root())
sys.path.append(project_root)

from src.speaker_identification.core.llm_client import MLXLLMClient
from src.speaker_identification.core.context_builder import ContextBuilder
from src.speaker_identification.core.identity_manager import IdentityManager
from src.speaker_identification.core.phase_tracking import PhaseTracker
from src.speaker_identification.strategies.base import HostStrategyContext
from src.speaker_identification.strategies.hosts import SingleHostStrategy, MultiHostStrategy
from src.utils.logger import setup_worker_logger
from src.database.session import get_session
from src.database.models import IdentificationStatus
from sqlalchemy import text

logger = setup_worker_logger('speaker_identification.host_embedding_strategy')


class HostEmbeddingIdentificationStrategy:
    """
    Phase 2: Named Host Centroid Identification.

    Process:
    1. Phase 2A: Analyze content.hosts frequency to find qualified hosts (≥10 occurrences)
    2. Phase 2B: Bootstrap centroid for each named host:
       - Load TOP-3 speakers (by duration) from labeled episodes
       - Cluster at 0.78 threshold
       - LLM verify largest cluster is host voice
       - Compute centroid, store in speaker_identity.verification_metadata
    3. Phase 2C: Host assignment (episodes without host only):
       - Skip episodes that already have a host assigned
       - Load long-speaking (≥15%) unassigned speakers
       - Match to host centroids: ≥0.65 auto-assign, 0.40-0.65 LLM verify, <0.40 skip
       - 1 speaker per host per episode
    """

    def __init__(
        self,
        high_confidence_threshold: float = 0.65,
        medium_confidence_threshold: float = 0.50,
        cluster_threshold: float = 0.78,
        min_quality: float = 0.30,
        min_host_occurrences: int = 10,
        min_duration_pct: float = 10.0,
        dry_run: bool = True,
        max_episodes: int = None,
        episode_progress_callback = None,
        fresh: bool = False
    ):
        """
        Initialize strategy.

        Args:
            high_confidence_threshold: Threshold for auto-assignment (default 0.65)
            medium_confidence_threshold: Minimum threshold for LLM verification (default 0.50)
            cluster_threshold: Threshold for core cluster formation
            min_quality: Minimum embedding quality for speaker consideration (default 0.30)
            min_host_occurrences: Minimum times host must appear in content.hosts
            min_duration_pct: Minimum duration percentage to be considered for host (default 10%)
            dry_run: If True, don't make DB changes
            max_episodes: Maximum episodes to process per channel
            episode_progress_callback: Optional callback(episode_num, total)
            fresh: If True, ignore existing centroids and re-bootstrap all hosts
        """
        self.high_threshold = high_confidence_threshold
        self.medium_threshold = medium_confidence_threshold
        self.cluster_threshold = cluster_threshold
        self.min_quality = min_quality
        self.min_host_occurrences = min_host_occurrences
        self.min_duration_pct = min_duration_pct
        self.dry_run = dry_run
        self.max_episodes = max_episodes
        self.episode_progress_callback = episode_progress_callback
        self.fresh = fresh

        self.llm_client = MLXLLMClient()
        self.context_builder = ContextBuilder()
        self.identity_manager = IdentityManager()

        # Named host centroids for current channel
        self.host_centroids: Dict[str, Dict] = {}

        # Track (episode_id, identity_id) pairs to allow multiple hosts per episode
        # This supports multi-host channels where episodes have co-hosts
        self.episode_host_pairs_assigned: Set[Tuple[str, int]] = set()

        # Centroid generation settings
        self.centroid_min_quality = 0.65  # Prefer embeddings above this quality
        self.centroid_min_samples = 5     # Minimum samples for centroid
        self.centroid_max_samples = 50    # Cap to avoid diminishing returns

        # Tracking
        self.stats = {
            'channels_processed': 0,
            'qualified_hosts_found': 0,
            'centroids_bootstrapped': 0,
            'cluster_llm_verified': 0,
            'cluster_llm_rejected': 0,
            'bootstrap_speakers_assigned': 0,
            'episodes_processed': 0,
            'episodes_skipped_has_host': 0,
            'high_confidence_assigned': 0,
            'medium_verified_assigned': 0,
            'medium_rejected': 0,
            'low_confidence_skipped': 0,
            'llm_verification_calls': 0,
            'identities_created': 0,
            'fallback_to_anonymous': 0,
            'errors': [],
            # Dual-strategy stats
            'single_host_channels': 0,
            'multi_host_channels': 0,
            'llm_with_metadata_context': 0,  # Multi-host LLM calls with metadata
            'metadata_llm_verified': 0,       # Multi-host LLM confirmations
            'behavioral_inference_used': 0,   # Single-host LLM calls (first cluster only)
            'embedding_similarity_merged': 0, # Single-host clusters merged via embedding
            'embedding_similarity_rejected': 0, # Single-host clusters rejected (different speaker)
            # "Probably" retry stats
            'probably_retries': 0,            # "probably" confidence retries
            'probably_upgraded': 0,           # Retries that upgraded to certain/very_likely
            'probably_downgraded': 0,         # Retries that stayed ambiguous → skip
        }

    async def run_single_channel(self, channel_id: int) -> Dict:
        """
        Run Phase 2A + 2B + 2C on a single channel.

        Args:
            channel_id: Channel ID to process

        Returns:
            Stats dict for this channel
        """
        logger.info("=" * 80)
        logger.info(f"Phase 2: Named Host Centroid Identification - Channel ID: {channel_id}")
        logger.info("=" * 80)

        # Phase 2A: Analyze content.hosts frequency
        qualified_hosts = await self._phase_2a_analyze_host_frequency(channel_id)

        if not qualified_hosts:
            logger.warning(f"No qualified hosts (≥{self.min_host_occurrences} occurrences) - falling back to anonymous clustering")
            self.stats['fallback_to_anonymous'] += 1
            return await self._fallback_anonymous_clustering(channel_id)

        # Phase 2B: Bootstrap centroid for each qualified host
        await self._phase_2b_bootstrap_host_centroids(channel_id, qualified_hosts)

        if not self.host_centroids:
            logger.warning("No centroids bootstrapped - falling back to anonymous clustering")
            self.stats['fallback_to_anonymous'] += 1
            return await self._fallback_anonymous_clustering(channel_id)

        # Phase 2C: Expand to full channel
        await self._phase_2c_expand_to_full_channel(channel_id)

        self.stats['channels_processed'] += 1
        return self.stats

    async def _phase_2a_analyze_host_frequency(self, channel_id: int) -> List[Dict]:
        """
        Phase 2A: Analyze content.hosts to find qualified hosts.

        Args:
            channel_id: Channel ID

        Returns:
            List of qualified hosts with episode_ids
        """
        logger.info("\n--- Phase 2A: Host Name Frequency Analysis ---")

        qualified_hosts = self.context_builder.get_host_frequency_for_channel(
            channel_id=channel_id,
            min_count=self.min_host_occurrences,
            merge_names=True
        )

        if not qualified_hosts:
            logger.info(f"No hosts found with ≥{self.min_host_occurrences} occurrences in content.hosts")
            return []

        self.stats['qualified_hosts_found'] = len(qualified_hosts)
        logger.info(f"Found {len(qualified_hosts)} qualified host(s):")
        for host in qualified_hosts:
            logger.info(f"  - {host['name']}: {host['count']} episodes")

        return qualified_hosts

    async def _phase_2b_bootstrap_host_centroids(
        self,
        channel_id: int,
        qualified_hosts: List[Dict]
    ):
        """
        Phase 2B: Bootstrap centroid for each qualified host.

        Idempotent: Loads existing centroids from DB first, only bootstraps
        hosts that don't already have a centroid.

        For each host without centroid:
        1. Load speakers from only their labeled episodes
        2. Take highest-duration speaker per episode
        3. Cluster at 0.78 threshold
        4. Compute centroid from largest cluster
        5. Store centroid and assign cluster members

        Args:
            channel_id: Channel ID
            qualified_hosts: List from Phase 2A
        """
        logger.info("\n--- Phase 2B: Bootstrap Host Centroids ---")

        # Load existing centroids from DB
        self.host_centroids = {}
        self._db_centroids = {}  # Store DB centroids for comparison at end
        existing_centroids = self.identity_manager.get_centroids_for_channel(
            channel_id=channel_id,
            roles=['host', 'co_host']
        )

        for ec in existing_centroids:
            centroid_data = {
                'identity_id': ec['identity_id'],
                'centroid': ec['centroid'],
                'quality': ec.get('centroid_quality', 0.8),
                'sample_count': ec.get('centroid_sample_count', 10)
            }
            # Always store in _db_centroids for later comparison
            self._db_centroids[ec['name']] = centroid_data

            if self.fresh:
                # Fresh mode: don't load into host_centroids, will re-bootstrap
                logger.info(f"  [fresh] Ignoring existing centroid for '{ec['name']}' (quality={ec.get('centroid_quality', 0):.3f})")
            else:
                # Normal mode: use existing centroids
                self.host_centroids[ec['name']] = centroid_data
                logger.info(f"  Loaded existing centroid for '{ec['name']}' (quality={ec.get('centroid_quality', 0):.3f})")

        if self.fresh:
            logger.info(f"  [fresh] Will re-bootstrap all hosts (ignoring {len(self._db_centroids)} existing centroids)")
        elif self.host_centroids:
            logger.info(f"  Loaded {len(self.host_centroids)} existing centroid(s) from DB")

        # Collect all host names for LLM context
        all_host_names = [h['name'] for h in qualified_hosts]

        for host in qualified_hosts:
            host_name = host['name']
            episode_ids = host['episode_ids']

            # Skip if already has centroid (from DB or opportunistic assignment)
            if host_name in self.host_centroids:
                logger.info(f"\n'{host_name}' already has centroid, skipping bootstrap")
                continue

            logger.info(f"\nBootstrapping centroid for '{host_name}' ({len(episode_ids)} labeled episodes)")

            centroid_result = await self._bootstrap_single_host_centroid(
                channel_id=channel_id,
                host_name=host_name,
                episode_ids=episode_ids,
                all_host_names=all_host_names,
                qualified_hosts=qualified_hosts
            )

            if centroid_result:
                self.host_centroids[host_name] = centroid_result
                self.stats['centroids_bootstrapped'] += 1
                logger.info(f"  ✓ Centroid bootstrapped: {centroid_result['sample_count']} samples, quality={centroid_result['quality']:.3f}")
            else:
                logger.warning(f"  ✗ Could not bootstrap centroid for '{host_name}'")

    async def _bootstrap_single_host_centroid(
        self,
        channel_id: int,
        host_name: str,
        episode_ids: List[str],
        all_host_names: List[str] = None,
        qualified_hosts: List[Dict] = None
    ) -> Optional[Dict]:
        """
        Bootstrap centroid for a single named host.

        Process:
        1. Load TOP-3 speakers (by duration) from each labeled episode
        2. Cluster to find the host voice cluster
        3. LLM verify the largest cluster is the host voice
        4. Compute centroid from verified cluster
        5. Assign cluster members (1 per episode)

        Args:
            channel_id: Channel ID
            host_name: Host name
            episode_ids: Episodes where this host is labeled
            all_host_names: All known host names (for LLM context)
            qualified_hosts: Full host data with episode_ids (for metadata matching)

        Returns:
            {
                'identity_id': int,
                'centroid': np.array,
                'quality': float,
                'sample_count': int,
                'member_speaker_ids': List[int]
            }
        """
        if all_host_names is None:
            all_host_names = [host_name]
        # Get or create identity for this host
        if self.dry_run:
            identity_id = None
        else:
            identity_id = self.identity_manager.get_or_create_host_identity(
                name=host_name,
                channel_id=channel_id,
                confidence=0.85
            )

        # Load TOP-3 speakers (by duration) from each labeled episode
        # Host may not be the longest speaker (e.g., 40% host, 50% guest)
        #
        # When fresh=True, we also include speakers already assigned to THIS host
        # so we can re-evaluate the centroid using all available data.
        with get_session() as session:
            # Get the identity ID for this host if it exists (for fresh mode)
            target_identity_id = None
            if self.fresh and host_name in self._db_centroids:
                target_identity_id = self._db_centroids[host_name].get('identity_id')

            if self.fresh and target_identity_id:
                # Fresh mode: include speakers already assigned to this host
                query = text("""
                    WITH ranked_speakers AS (
                        SELECT
                            s.id,
                            s.content_id,
                            s.embedding,
                            s.embedding_quality_score,
                            s.duration,
                            ROW_NUMBER() OVER (PARTITION BY s.content_id ORDER BY s.duration DESC) as rank
                        FROM speakers s
                        WHERE s.content_id = ANY(:episode_ids)
                          AND s.embedding IS NOT NULL
                          AND s.embedding_quality_score >= :min_quality
                          AND (s.speaker_identity_id IS NULL OR s.speaker_identity_id = :target_identity_id)
                    )
                    SELECT id, content_id, embedding, embedding_quality_score, duration
                    FROM ranked_speakers
                    WHERE rank <= 3
                    ORDER BY content_id, duration DESC
                """)
                results = session.execute(query, {
                    'episode_ids': episode_ids,
                    'min_quality': self.min_quality,
                    'target_identity_id': target_identity_id
                }).fetchall()
            else:
                # Normal mode: speakers not yet processed by Phase 2
                # Uses per-phase tracking in llm_identification JSONB
                query = text(f"""
                    WITH ranked_speakers AS (
                        SELECT
                            s.id,
                            s.content_id,
                            s.embedding,
                            s.embedding_quality_score,
                            s.duration,
                            ROW_NUMBER() OVER (PARTITION BY s.content_id ORDER BY s.duration DESC) as rank
                        FROM speakers s
                        WHERE s.content_id = ANY(:episode_ids)
                          AND s.embedding IS NOT NULL
                          AND s.embedding_quality_score >= :min_quality
                          AND s.speaker_identity_id IS NULL
                          AND {PhaseTracker.get_phase_filter_sql(2, 's')}
                    )
                    SELECT id, content_id, embedding, embedding_quality_score, duration
                    FROM ranked_speakers
                    WHERE rank <= 3
                    ORDER BY content_id, duration DESC
                """)
                results = session.execute(query, {
                    'episode_ids': episode_ids,
                    'min_quality': self.min_quality
                }).fetchall()

        if not results:
            logger.warning(f"  No valid speakers found in labeled episodes for '{host_name}'")
            return None

        logger.info(f"  Loaded {len(results)} top-3 speakers from {len(episode_ids)} labeled episodes")

        # Parse embeddings
        speakers = []
        embeddings_list = []

        for row in results:
            try:
                embedding_str = str(row.embedding)
                embedding_values = embedding_str.strip('[]').split(',')
                embedding = np.array([float(v) for v in embedding_values], dtype=np.float32)

                # Normalize
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm

                speakers.append({
                    'id': row.id,
                    'content_id': row.content_id,
                    'quality': row.embedding_quality_score,
                    'duration': row.duration
                })
                embeddings_list.append(embedding)

            except Exception as e:
                logger.error(f"  Error parsing embedding for speaker {row.id}: {e}")
                continue

        if len(embeddings_list) < 5:
            logger.warning(f"  Too few valid embeddings ({len(embeddings_list)}) for clustering")
            return None

        embeddings = np.array(embeddings_list, dtype=np.float32)

        # Cluster to find the actual host cluster (handles mislabeled episodes)
        # Returns clusters sorted by episode count (largest first)
        all_clusters = self._cluster_speakers_all(speakers, embeddings)

        if not all_clusters:
            logger.warning(f"  No clusters found for '{host_name}'")
            return None

        # Select strategy based on channel type
        is_single_host = len(all_host_names) == 1
        target_already_done = host_name in self.host_centroids

        if is_single_host:
            strategy = SingleHostStrategy(
                llm_client=self.llm_client,
                context_builder=self.context_builder,
                select_best_for_centroid_fn=self._select_best_for_centroid
            )
            self.stats['single_host_channels'] = self.stats.get('single_host_channels', 0) + 1
            logger.info(f"  Single-host channel: using SingleHostStrategy")
        else:
            strategy = MultiHostStrategy(
                llm_client=self.llm_client,
                context_builder=self.context_builder,
                select_best_for_centroid_fn=self._select_best_for_centroid
            )
            self.stats['multi_host_channels'] = self.stats.get('multi_host_channels', 0) + 1
            logger.info(f"  Multi-host channel ({len(all_host_names)} hosts): using MultiHostStrategy")

        max_clusters_to_check = strategy.get_max_clusters_to_check(len(all_clusters))
        hosts_already_assigned = len(self.host_centroids)

        if target_already_done:
            logger.info(f"  Target '{host_name}' already has centroid")

        verified_cluster = None
        clusters_checked = 0
        hosts_found_this_pass = 0
        total_hosts_for_limit = hosts_already_assigned

        for cluster_idx, cluster_result in enumerate(all_clusters):
            # Check if strategy says to stop
            if strategy.should_stop_search(
                clusters_checked=cluster_idx,
                hosts_found=total_hosts_for_limit,
                total_clusters=len(all_clusters),
                target_found=host_name in self.host_centroids
            ):
                logger.info(f"  Strategy says stop search after {cluster_idx} clusters")
                break

            cluster_indices = set(cluster_result['members'])

            logger.info(f"  Checking cluster {cluster_idx + 1}/{len(all_clusters)}: {len(cluster_indices)} speakers, {cluster_result['episode_count']} episodes")

            # Build strategy context
            ctx = HostStrategyContext(
                channel_id=channel_id,
                all_host_names=all_host_names,
                qualified_hosts=qualified_hosts or [],
                target_host_name=host_name,
                cluster_indices=cluster_indices,
                cluster_result=cluster_result,
                speakers=speakers,
                embeddings=embeddings,
                existing_centroids=self.host_centroids,
                stats=self.stats
            )

            # Use strategy to verify cluster
            result = await strategy.verify_cluster(ctx)

            # Process identification result
            identified_host = result.identified_host
            confidence = result.confidence
            clusters_checked += 1

            logger.info(f"    Strategy verdict: identified_host='{identified_host}', confidence={confidence}")
            logger.info(f"    Reasoning: {result.reasoning}")

            if identified_host == host_name and confidence in ['certain', 'very_likely']:
                if target_already_done or host_name in self.host_centroids:
                    # Target already has centroid - merge this cluster into it
                    logger.info(f"  → Cluster {cluster_idx + 1} is '{host_name}' (already has centroid) - merging")
                    await self._assign_cluster_to_host(
                        channel_id=channel_id,
                        host_name=identified_host,
                        speakers=speakers,
                        embeddings=embeddings,
                        cluster_result=cluster_result
                    )
                    hosts_found_this_pass += 1
                else:
                    # First time finding target - create centroid immediately
                    verified_cluster = cluster_result
                    self.stats['cluster_llm_verified'] += 1
                    logger.info(f"  ✓ Cluster {cluster_idx + 1} verified as '{host_name}' voice")
                    hosts_found_this_pass += 1
                    total_hosts_for_limit += 1

                    # For single-host: create centroid NOW so we can use embedding similarity for remaining clusters
                    if isinstance(strategy, SingleHostStrategy):
                        # Use best embeddings only for centroid
                        centroid, _, avg_quality = self._select_best_for_centroid(
                            cluster_indices, speakers, embeddings
                        )

                        self.host_centroids[host_name] = {
                            'centroid': centroid,
                            'quality': avg_quality,
                            'speaker_ids': [speakers[i]['id'] for i in cluster_indices]
                        }
                        logger.info(f"    Created initial centroid for '{host_name}' (best of {len(cluster_indices)} speakers, quality={avg_quality:.3f})")
                        # Continue checking more clusters via embedding similarity
                    else:
                        # Multi-host: break after finding target (strategy handles stop condition too)
                        break
            elif identified_host != 'unknown' and identified_host in all_host_names and confidence in ['certain', 'very_likely']:
                # It's a different known host - assign them now!
                host_is_new = identified_host not in self.host_centroids
                logger.info(f"  → Cluster {cluster_idx + 1} identified as '{identified_host}' - assigning now")
                await self._assign_cluster_to_host(
                    channel_id=channel_id,
                    host_name=identified_host,
                    speakers=speakers,
                    embeddings=embeddings,
                    cluster_result=cluster_result
                )
                hosts_found_this_pass += 1
                if host_is_new:
                    total_hosts_for_limit += 1
            elif result.is_discovered_host():
                # NEW: Discovered host not in expected hosts (substitute host, co-host, etc.)
                discovered_name = result.speaker_name
                discovered_role = result.role
                logger.info(f"  ★ Cluster {cluster_idx + 1} discovered new host: '{discovered_name}' (role={discovered_role})")
                logger.info(f"    This speaker was not in expected hosts but shows host behavior")

                # Create identity and centroid for discovered host
                await self._assign_cluster_to_host(
                    channel_id=channel_id,
                    host_name=discovered_name,
                    speakers=speakers,
                    embeddings=embeddings,
                    cluster_result=cluster_result,
                    role=discovered_role,
                    is_discovered=True
                )
                hosts_found_this_pass += 1
                total_hosts_for_limit += 1
                self.stats['hosts_discovered'] = self.stats.get('hosts_discovered', 0) + 1
            else:
                self.stats['cluster_llm_rejected'] += 1
                logger.info(f"  ✗ Cluster {cluster_idx + 1} rejected (unknown speaker)")

        logger.info(f"  Checked {clusters_checked} clusters, found {hosts_found_this_pass} hosts this pass, {total_hosts_for_limit} total hosts")

        # If target was already done, we don't need to return a verified cluster
        if target_already_done:
            return self.host_centroids.get(host_name)

        if not verified_cluster:
            logger.warning(f"  No cluster verified as '{host_name}' voice")
            return None

        cluster_indices = set(verified_cluster['members'])

        # Compute centroid using only the best embeddings
        centroid, _, avg_quality = self._select_best_for_centroid(
            cluster_indices, speakers, embeddings
        )

        logger.info(f"  Core cluster: {len(cluster_indices)} speakers, centroid from best (quality={avg_quality:.3f})")

        # Track which (episode, host) pairs have been assigned
        # This allows multiple hosts per episode for multi-host channels
        assigned_pairs_this_host = set()  # (content_id,) - track for this specific host
        assigned_speaker_ids = []
        bulk_assignments = []  # Collect for bulk update

        # Assign cluster members to identity (these are high confidence)
        for idx in cluster_indices:
            speaker = speakers[idx]
            content_id = speaker['content_id']

            # Only assign THIS host once per episode - take the first (highest duration due to ORDER BY)
            if content_id in assigned_pairs_this_host:
                continue

            # Check if this (episode, host) pair is already assigned
            if (content_id, identity_id) in self.episode_host_pairs_assigned:
                continue

            assigned_pairs_this_host.add(content_id)
            assigned_speaker_ids.append(speaker['id'])

            # Track for Phase 2C - allows other hosts to still be assigned to this episode
            self.episode_host_pairs_assigned.add((content_id, identity_id))
            self.stats['bootstrap_speakers_assigned'] += 1

            if not self.dry_run and identity_id:
                bulk_assignments.append({
                    'speaker_id': speaker['id'],
                    'identity_id': identity_id,
                    'confidence': 0.95,
                    'metadata': {'host_name': host_name, 'cluster_member': True}
                })

        logger.info(f"  Assigned {len(assigned_speaker_ids)} cluster members (1 per episode for host '{host_name}')")

        # Now handle orphaned speakers - those not in the cluster
        # Compare to centroid, assign best match per episode if ≥0.55
        orphan_indices = [i for i in range(len(speakers)) if i not in cluster_indices]

        if orphan_indices:
            # Group orphans by episode (only for episodes where THIS host isn't assigned yet)
            orphans_by_episode = defaultdict(list)
            for idx in orphan_indices:
                speaker = speakers[idx]
                content_id = speaker['content_id']
                # Check if THIS specific host is already assigned to this episode
                if content_id not in assigned_pairs_this_host and (content_id, identity_id) not in self.episode_host_pairs_assigned:
                    similarity = float(np.dot(embeddings[idx], centroid))
                    orphans_by_episode[content_id].append({
                        'idx': idx,
                        'speaker': speaker,
                        'similarity': similarity
                    })

            # For each episode without THIS host assigned, pick the best matching orphan
            orphan_assigned = 0
            for content_id, candidates in orphans_by_episode.items():
                # Sort by similarity descending
                candidates.sort(key=lambda x: x['similarity'], reverse=True)
                best = candidates[0]

                if best['similarity'] >= self.high_threshold:
                    assigned_speaker_ids.append(best['speaker']['id'])

                    # Track for Phase 2C - allows other hosts to still be assigned to this episode
                    self.episode_host_pairs_assigned.add((content_id, identity_id))
                    self.stats['bootstrap_speakers_assigned'] += 1
                    orphan_assigned += 1

                    if not self.dry_run and identity_id:
                        bulk_assignments.append({
                            'speaker_id': best['speaker']['id'],
                            'identity_id': identity_id,
                            'confidence': best['similarity'],
                            'metadata': {
                                'host_name': host_name,
                                'similarity': best['similarity']
                            }
                        })

            if orphan_assigned > 0:
                logger.info(f"  Assigned {orphan_assigned} orphan speakers (similarity >= {self.high_threshold})")

        # Bulk assign all speakers in one DB call
        if bulk_assignments:
            self.identity_manager.bulk_assign_speakers(
                assignments=bulk_assignments,
                method='phase_2b_bootstrap_cluster'
            )

        # Store centroid
        if not self.dry_run and identity_id:
            self.identity_manager.store_host_centroid(
                identity_id=identity_id,
                centroid=centroid,
                quality=avg_quality,
                sample_count=len(cluster_indices),
                episode_ids=list(assigned_pairs_this_host)[:100],
                channel_id=channel_id
            )

        return {
            'identity_id': identity_id,
            'name': host_name,
            'centroid': centroid,
            'quality': avg_quality,
            'sample_count': len(cluster_indices),
            'member_speaker_ids': assigned_speaker_ids
        }

    def _select_best_for_centroid(
        self,
        cluster_indices: set,
        speakers: List[Dict],
        embeddings: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Select the best embeddings from a cluster for centroid generation.

        Strategy:
        1. Take embeddings with quality >= centroid_min_quality (0.65)
        2. Ensure at least centroid_min_samples (5) - if not enough, take top by quality
        3. Cap at centroid_max_samples (50) to avoid diminishing returns
        4. Return quality-weighted centroid

        Args:
            cluster_indices: Set of indices into speakers/embeddings
            speakers: List of speaker dicts with 'quality' field
            embeddings: Full embedding matrix

        Returns:
            (centroid, selected_qualities, avg_quality)
        """
        indices_list = list(cluster_indices)

        # Get qualities and sort by quality descending
        idx_quality = [(idx, speakers[idx]['quality']) for idx in indices_list]
        idx_quality.sort(key=lambda x: x[1], reverse=True)

        # Select high-quality embeddings
        selected = []
        for idx, quality in idx_quality:
            if quality >= self.centroid_min_quality:
                selected.append((idx, quality))
            elif len(selected) < self.centroid_min_samples:
                # Not enough high quality - take this one anyway
                selected.append((idx, quality))

            if len(selected) >= self.centroid_max_samples:
                break

        # Ensure we have minimum samples
        if len(selected) < self.centroid_min_samples and len(idx_quality) >= self.centroid_min_samples:
            selected = idx_quality[:self.centroid_min_samples]

        # Extract embeddings and qualities
        selected_indices = [s[0] for s in selected]
        selected_qualities = np.array([s[1] for s in selected])
        selected_embeddings = embeddings[selected_indices]

        # Compute quality-weighted centroid
        weights = selected_qualities / np.sum(selected_qualities)
        centroid = np.average(selected_embeddings, axis=0, weights=weights)
        centroid = centroid / np.linalg.norm(centroid)
        avg_quality = float(np.mean(selected_qualities))

        logger.debug(
            f"Centroid selection: {len(selected)}/{len(cluster_indices)} speakers "
            f"(quality range: {selected_qualities.min():.3f}-{selected_qualities.max():.3f}, "
            f"avg: {avg_quality:.3f})"
        )

        return centroid, selected_qualities, avg_quality

    def _cluster_speakers_all(
        self,
        speakers: List[Dict],
        embeddings: np.ndarray
    ) -> List[Dict]:
        """
        Cluster speakers and return all clusters sorted by episode count.

        Args:
            speakers: List of speaker dicts
            embeddings: Normalized embedding matrix

        Returns:
            List of {'members': List[int], 'episode_count': int} sorted by episode_count desc
        """
        n_speakers = len(embeddings)
        if n_speakers < 5:
            return []

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        K = min(50, n_speakers - 1)
        similarities, indices = index.search(embeddings, K + 1)

        # Build adjacency graph
        adjacency = defaultdict(set)

        for i in range(n_speakers):
            for j in range(1, K + 1):
                neighbor_idx = indices[i][j]
                similarity = similarities[i][j]

                # Skip same-episode speakers (shouldn't happen with our query but be safe)
                if speakers[i]['content_id'] == speakers[neighbor_idx]['content_id']:
                    continue

                if similarity >= self.cluster_threshold:
                    adjacency[i].add(neighbor_idx)
                    adjacency[neighbor_idx].add(i)

        # BFS to find connected components
        visited = set()
        clusters = []

        for start_node in range(n_speakers):
            if start_node in visited:
                continue

            queue = deque([start_node])
            component = []

            while queue:
                node = queue.popleft()
                if node in visited:
                    continue

                visited.add(node)
                component.append(node)

                for neighbor in adjacency[node]:
                    if neighbor not in visited:
                        queue.append(neighbor)

            if len(component) >= 5:
                cluster_episodes = set(speakers[i]['content_id'] for i in component)
                clusters.append({
                    'members': component,
                    'episode_count': len(cluster_episodes)
                })

        if not clusters:
            return []

        # Return all clusters sorted by episode count (largest first)
        clusters.sort(key=lambda x: x['episode_count'], reverse=True)
        return clusters

    def _compute_cluster_metadata_match(
        self,
        speakers: List[Dict],
        cluster_indices: List[int],
        qualified_hosts: List[Dict]
    ) -> Dict:
        """
        Match cluster to hosts using episode metadata overlap.

        For multi-host channels, this is the primary identification signal.
        Computes overlap ratio (what % of cluster episodes have this host)
        and coverage ratio (what % of host's episodes have this cluster).

        Args:
            speakers: List of speaker dicts with content_id
            cluster_indices: Indices of speakers in this cluster
            qualified_hosts: Hosts from Phase 2A with episode_ids

        Returns:
            {
                'best_match': {
                    'host_name': str,
                    'overlap_ratio': float,
                    'coverage_ratio': float,
                    'overlap_count': int,
                    'cluster_episode_count': int,
                    'host_episode_count': int
                } or None,
                'all_matches': List[Dict],
                'confidence': 'high' | 'medium' | 'low' | 'none',
                'needs_llm': bool
            }
        """
        # Get unique episodes in this cluster
        cluster_episodes = set(speakers[i]['content_id'] for i in cluster_indices)
        cluster_episode_count = len(cluster_episodes)

        if cluster_episode_count == 0:
            return {
                'best_match': None,
                'all_matches': [],
                'confidence': 'none',
                'needs_llm': True
            }

        matches = []
        for host in qualified_hosts:
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

        if not matches:
            return {
                'best_match': None,
                'all_matches': [],
                'confidence': 'none',
                'needs_llm': True
            }

        best = matches[0]

        # Determine confidence and whether LLM verification is needed
        # High: 90%+ overlap AND decent coverage (40%+)
        # Medium: 70-90% overlap
        # Low: <70% overlap
        if best['overlap_ratio'] >= 0.90 and best['coverage_ratio'] >= 0.40:
            confidence = 'high'
            needs_llm = False
        elif best['overlap_ratio'] >= 0.70:
            confidence = 'medium'
            needs_llm = True
        else:
            confidence = 'low'
            needs_llm = True

        # Check for conflicting matches (multiple hosts with significant overlap)
        if len(matches) > 1 and matches[1]['overlap_ratio'] > 0.50:
            # Ambiguous - multiple hosts have strong presence
            needs_llm = True

        return {
            'best_match': best,
            'all_matches': matches,
            'confidence': confidence,
            'needs_llm': needs_llm
        }

    async def _assign_cluster_to_host(
        self,
        channel_id: int,
        host_name: str,
        speakers: List[Dict],
        embeddings: np.ndarray,
        cluster_result: Dict,
        role: str = 'host',
        is_discovered: bool = False
    ):
        """
        Assign a cluster to a host and build their centroid.

        Called when we identify a cluster as a DIFFERENT host than the one
        we're currently processing. This lets us opportunistically assign
        hosts we discover along the way.

        Also handles DISCOVERED hosts - speakers who show host behavior but
        weren't in the expected host list (substitute hosts, co-hosts, etc.)

        Args:
            channel_id: Channel ID
            host_name: Identified host name
            speakers: Full speaker list
            embeddings: Speaker embeddings
            cluster_result: Cluster dict with 'members' and 'episode_count'
            role: Speaker role ('host', 'co_host', etc.) - used for discovered hosts
            is_discovered: True if this host was not in expected hosts
        """
        if is_discovered:
            logger.info(f"    Creating identity for discovered {role}: '{host_name}'")
        cluster_indices = set(cluster_result['members'])

        # If host already has a centroid, check similarity and potentially merge
        if host_name in self.host_centroids:
            existing = self.host_centroids[host_name]
            existing_centroid = existing['centroid']

            # Check if these speakers are already in the existing centroid's member list
            existing_speaker_ids = set(existing.get('member_speaker_ids', []))
            new_speaker_ids = {speakers[idx]['id'] for idx in cluster_indices}

            # If all speakers already assigned, skip entirely
            if new_speaker_ids.issubset(existing_speaker_ids):
                logger.info(f"    Skipping '{host_name}' cluster - all {len(new_speaker_ids)} speakers already assigned")
                return

            # Filter to only truly new speakers
            truly_new_indices = {idx for idx in cluster_indices if speakers[idx]['id'] not in existing_speaker_ids}

            if not truly_new_indices:
                logger.info(f"    Skipping '{host_name}' cluster - no new speakers to assign")
                return

            # Compute new centroid for this cluster (using best embeddings)
            new_centroid, _, new_quality = self._select_best_for_centroid(
                cluster_indices, speakers, embeddings
            )

            # Check similarity between existing and new centroid
            similarity = float(np.dot(existing_centroid, new_centroid))

            if similarity >= 0.95:
                # Very high similarity - same voice, just assign new speakers without re-merging centroid
                logger.info(f"    High similarity ({similarity:.3f}) to existing '{host_name}' centroid - assigning {len(truly_new_indices)} new speakers")
            elif similarity >= 0.80:
                # Similar enough - merge by weighted average
                existing_samples = existing['sample_count']
                new_samples = len(truly_new_indices)
                total_samples = existing_samples + new_samples

                merged_centroid = (
                    existing_centroid * existing_samples +
                    new_centroid * new_samples
                ) / total_samples
                merged_centroid = merged_centroid / np.linalg.norm(merged_centroid)

                existing['centroid'] = merged_centroid
                existing['sample_count'] = total_samples
                existing['quality'] = (existing['quality'] * existing_samples + new_quality * new_samples) / total_samples

                logger.info(f"    Merged centroid for '{host_name}' (sim={similarity:.3f}, now {total_samples} samples)")
            else:
                # Different vocal profile - store as additional centroid
                profile_num = 2
                profile_key = f"{host_name}_profile_{profile_num}"
                while profile_key in self.host_centroids:
                    profile_num += 1
                    profile_key = f"{host_name}_profile_{profile_num}"

                logger.info(f"    New vocal profile for '{host_name}' (sim={similarity:.3f} to existing) - storing as profile {profile_num}")
                # Continue to create new profile below
                host_name = profile_key  # Use profile key for storage

            # If merged or high similarity, just assign the NEW speakers and return
            if similarity >= 0.80:
                identity_id = existing['identity_id']
                assigned_episodes = set()
                bulk_assignments = []

                # Only iterate over truly new speakers (not already assigned)
                for idx in truly_new_indices:
                    speaker = speakers[idx]
                    content_id = speaker['content_id']
                    if content_id in assigned_episodes:
                        continue
                    assigned_episodes.add(content_id)

                    if not self.dry_run and identity_id:
                        bulk_assignments.append({
                            'speaker_id': speaker['id'],
                            'identity_id': identity_id,
                            'confidence': 0.95,
                            'metadata': {'host_name': host_name, 'cluster_member': True, 'merged': True}
                        })
                        self.stats['bootstrap_speakers_assigned'] += 1

                # Bulk assign all speakers in one DB call
                if bulk_assignments:
                    self.identity_manager.bulk_assign_speakers(
                        assignments=bulk_assignments,
                        method='phase_2b_bootstrap_cluster'
                    )

                # Update member list to include new speakers
                existing['member_speaker_ids'] = list(existing_speaker_ids | new_speaker_ids)

                logger.info(f"    Assigned {len(bulk_assignments)} new speakers to '{host_name}'")
                return

        # Get or create identity for this host
        if self.dry_run:
            identity_id = None
        else:
            identity_id = self.identity_manager.get_or_create_host_identity(
                name=host_name,
                channel_id=channel_id,
                confidence=0.85,
                role=role
            )

        # Compute centroid using best embeddings
        centroid, _, avg_quality = self._select_best_for_centroid(
            cluster_indices, speakers, embeddings
        )

        # Track assigned episodes
        assigned_episodes = set()
        bulk_assignments = []

        # Assign cluster members (1 per episode)
        for idx in cluster_indices:
            speaker = speakers[idx]
            content_id = speaker['content_id']

            if content_id in assigned_episodes:
                continue

            assigned_episodes.add(content_id)

            if not self.dry_run and identity_id:
                bulk_assignments.append({
                    'speaker_id': speaker['id'],
                    'identity_id': identity_id,
                    'confidence': 0.95,
                    'metadata': {'host_name': host_name, 'cluster_member': True}
                })
                self.stats['bootstrap_speakers_assigned'] += 1

        # Bulk assign all speakers in one DB call
        if bulk_assignments:
            self.identity_manager.bulk_assign_speakers(
                assignments=bulk_assignments,
                method='phase_2b_bootstrap_cluster'
            )

        # Store centroid
        if not self.dry_run and identity_id:
            episode_ids = list(assigned_episodes)
            self.identity_manager.store_host_centroid(
                identity_id=identity_id,
                centroid=centroid,
                quality=avg_quality,
                sample_count=len(cluster_indices),
                episode_ids=episode_ids,
                channel_id=channel_id
            )

        # Store in host_centroids dict for Phase 2C
        self.host_centroids[host_name] = {
            'identity_id': identity_id,
            'centroid': centroid,
            'quality': avg_quality,
            'sample_count': len(cluster_indices),
            'member_speaker_ids': [speakers[i]['id'] for i in cluster_indices]
        }

        self.stats['centroids_bootstrapped'] += 1
        logger.info(f"    Assigned {len(bulk_assignments)} speakers to '{host_name}', centroid quality={avg_quality:.3f}")

    # NOTE: _verify_cluster_is_host() removed - use base class _verify_cluster_with_llm() instead
    # The base class in strategies/base.py provides the shared implementation

    async def _phase_2c_expand_to_full_channel(self, channel_id: int):
        """
        Phase 2C: Match long-speaking unassigned speakers to host centroids.

        Two-pass approach:
        1. Fast pass: Auto-assign all high-confidence (≥0.65) matches
        2. LLM pass: Verify medium-confidence (0.40-0.65) matches

        Only processes episodes that don't already have a host assigned.
        Only considers speakers with significant duration (>= min_duration_pct).

        Args:
            channel_id: Channel ID
        """
        logger.info("\n--- Phase 2C: Host Assignment (Episodes Without Host) ---")

        if not self.host_centroids:
            logger.warning("No host centroids available for expansion")
            return

        # Get all known host names for LLM verification
        known_host_names = list(self.host_centroids.keys())
        logger.info(f"Matching against {len(known_host_names)} host centroid(s): {known_host_names}")

        # First, find (episode, host_identity) pairs that already exist
        with get_session() as session:
            # Get (episode, host_identity) pairs - allows multiple hosts per episode
            existing_assignments_query = text("""
                SELECT DISTINCT s.content_id, s.speaker_identity_id
                FROM speakers s
                JOIN speaker_identities si ON s.speaker_identity_id = si.id
                JOIN content co ON s.content_id = co.content_id
                WHERE co.channel_id = :channel_id
                  AND si.role = 'host'
            """)

            existing_assignments = session.execute(
                existing_assignments_query,
                {'channel_id': channel_id}
            ).fetchall()

            # Combine DB results with in-memory tracking (for dry_run mode)
            # Track (episode_id, identity_id) pairs to allow multiple hosts per episode
            existing_pairs: Set[Tuple[str, int]] = {
                (row.content_id, row.speaker_identity_id) for row in existing_assignments
            }
            existing_pairs.update(self.episode_host_pairs_assigned)
            logger.info(f"Existing (episode, host) pairs: {len(existing_pairs)}")

            # Load long-speaking speakers not yet processed by Phase 2 from episodes where content.hosts
            # contains one of our known hosts (from Phase 1 metadata identification)
            # This ensures we only process episodes that have been confirmed to have these hosts
            query = text(f"""
                SELECT
                    s.id,
                    s.content_id,
                    s.embedding,
                    s.embedding_quality_score,
                    s.duration,
                    co.duration as episode_duration,
                    CASE
                        WHEN co.duration > 0 THEN (s.duration / co.duration * 100)
                        ELSE 0
                    END as duration_pct
                FROM speakers s
                JOIN content co ON s.content_id = co.content_id
                WHERE co.channel_id = :channel_id
                  AND s.embedding IS NOT NULL
                  AND s.embedding_quality_score >= :min_quality
                  AND s.speaker_identity_id IS NULL
                  AND {PhaseTracker.get_phase_filter_sql(2, 's')}
                  AND CASE
                        WHEN co.duration > 0 THEN (s.duration / co.duration * 100)
                        ELSE 0
                      END >= :min_duration_pct
                  AND EXISTS (
                      SELECT 1 FROM jsonb_array_elements(co.hosts) h
                      WHERE h->>'name' = ANY(:host_names)
                  )
                ORDER BY s.content_id, s.duration DESC
            """)

            results = session.execute(query, {
                'channel_id': channel_id,
                'min_quality': self.min_quality,
                'min_duration_pct': self.min_duration_pct,
                'host_names': known_host_names
            }).fetchall()

        if not results:
            logger.info("No long-speaking unassigned speakers to process")
            return

        logger.info(f"Found {len(results)} candidate speakers before per-host filtering")

        # Build centroid matrix for efficient matching
        centroid_names = list(self.host_centroids.keys())
        centroid_matrix = np.array([
            self.host_centroids[name]['centroid']
            for name in centroid_names
        ], dtype=np.float32)

        # Group speakers by (episode, host) and compute similarities
        # Key: (content_id, host_name), Value: list of speakers matching that host
        speakers_by_episode_host: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
        skipped_count = 0

        for row in results:
            try:
                # Parse embedding
                embedding_str = str(row.embedding)
                embedding_values = embedding_str.strip('[]').split(',')
                embedding = np.array([float(v) for v in embedding_values], dtype=np.float32)
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm

                # Find best matching centroid
                similarities = np.dot(centroid_matrix, embedding)
                best_idx = np.argmax(similarities)
                best_similarity = float(similarities[best_idx])
                best_host_name = centroid_names[best_idx]

                # Check if this specific host is already assigned to this episode
                host_identity_id = self.host_centroids[best_host_name]['identity_id']
                if (row.content_id, host_identity_id) in existing_pairs:
                    skipped_count += 1
                    continue

                speakers_by_episode_host[(row.content_id, best_host_name)].append({
                    'speaker_id': row.id,
                    'content_id': row.content_id,
                    'embedding': embedding,
                    'duration': row.duration,
                    'duration_pct': row.duration_pct,
                    'best_host_name': best_host_name,
                    'best_similarity': best_similarity,
                    'host_identity_id': host_identity_id,
                    'host_similarities': {
                        name: float(similarities[i])
                        for i, name in enumerate(centroid_names)
                    }
                })

            except Exception as e:
                logger.error(f"Error parsing embedding for speaker {row.id}: {e}")
                continue

        if skipped_count > 0:
            self.stats['episodes_skipped_has_host'] += skipped_count
            logger.info(f"Skipped {skipped_count} speakers (host already assigned to their episode)")

        total_episode_host_pairs = len(speakers_by_episode_host)
        if total_episode_host_pairs == 0:
            logger.info("No (episode, host) pairs to process after filtering")
            return

        logger.info(f"Found {total_episode_host_pairs} (episode, host) pairs to process")

        # ============================================================
        # PASS 1: Fast high-confidence assignments (≥0.65)
        # ============================================================
        logger.info(f"\n--- Pass 1: High-confidence assignments (≥{self.high_threshold}) ---")

        high_confidence_pairs = []
        medium_confidence_pairs = []
        low_confidence_pairs = []

        for (content_id, host_name), speakers in speakers_by_episode_host.items():
            # Sort by similarity (pick best speaker for this episode+host combination)
            speakers.sort(key=lambda x: x['best_similarity'], reverse=True)
            best_speaker = speakers[0]

            if best_speaker['best_similarity'] >= self.high_threshold:
                high_confidence_pairs.append((content_id, best_speaker))
            elif best_speaker['best_similarity'] >= self.medium_threshold:
                medium_confidence_pairs.append((content_id, best_speaker))
            else:
                low_confidence_pairs.append((content_id, best_speaker))

        logger.info(f"  High confidence (≥{self.high_threshold}): {len(high_confidence_pairs)} (episode, host) pairs")
        logger.info(f"  Medium confidence ({self.medium_threshold}-{self.high_threshold}): {len(medium_confidence_pairs)} (episode, host) pairs")
        logger.info(f"  Low confidence (<{self.medium_threshold}): {len(low_confidence_pairs)} (episode, host) pairs (will skip)")

        # Process high-confidence assignments (fast, no LLM) - collect for bulk update
        bulk_high_confidence = []
        for content_id, best_speaker in high_confidence_pairs:
            if self.max_episodes and self.stats['high_confidence_assigned'] >= self.max_episodes:
                break

            host_data = self.host_centroids[best_speaker['best_host_name']]
            identity_id = host_data['identity_id']

            if not self.dry_run:
                bulk_high_confidence.append({
                    'speaker_id': best_speaker['speaker_id'],
                    'identity_id': identity_id,
                    'confidence': best_speaker['best_similarity'],
                    'metadata': {
                        'host_name': best_speaker['best_host_name'],
                        'similarity': best_speaker['best_similarity']
                    }
                })
            # Track this (episode, host) pair as assigned
            self.episode_host_pairs_assigned.add((content_id, identity_id))
            self.stats['high_confidence_assigned'] += 1

        # Bulk assign all high-confidence speakers in one DB call
        if bulk_high_confidence:
            self.identity_manager.bulk_assign_speakers(
                assignments=bulk_high_confidence,
                method='phase_2c_high_confidence'
            )

        logger.info(f"  ✓ Assigned {self.stats['high_confidence_assigned']} high-confidence speakers")

        # ============================================================
        # PASS 2: LLM verification for medium-confidence (0.40-0.65)
        # ============================================================
        if medium_confidence_pairs:
            logger.info(f"\n--- Pass 2: LLM verification for {len(medium_confidence_pairs)} (episode, host) pairs ---")

            from tqdm import tqdm

            for content_id, best_speaker in tqdm(medium_confidence_pairs, desc="LLM verification"):
                if self.max_episodes and (self.stats['high_confidence_assigned'] + self.stats['medium_verified_assigned']) >= self.max_episodes:
                    break

                try:
                    # Get LLM verification
                    verified = await self._verify_with_llm(
                        speaker_id=best_speaker['speaker_id'],
                        best_host_name=best_speaker['best_host_name'],
                        host_similarities=best_speaker['host_similarities'],
                        known_host_names=known_host_names
                    )

                    # Log the reasoning and evidence
                    reasoning = verified.get('reasoning', 'No reasoning')
                    confidence = verified.get('confidence', 'unknown')
                    evidence_type = verified.get('evidence_type', 'none')
                    evidence_source = verified.get('evidence_source', 'unknown')
                    evidence_quote = verified.get('evidence_quote', '')

                    if verified['is_host'] and verified['confidence'] in ['certain', 'very_likely']:
                        # LLM confirmed
                        identified_name = verified.get('name', '').strip()

                        if identified_name and identified_name in self.host_centroids:
                            target_host = identified_name
                            target_data = self.host_centroids[identified_name]
                        else:
                            target_host = best_speaker['best_host_name']
                            target_data = self.host_centroids[target_host]

                        tqdm.write(f"  ✓ ASSIGNED speaker {best_speaker['speaker_id']} → '{target_host}' (sim={best_speaker['best_similarity']:.3f}, conf={confidence})")
                        tqdm.write(f"    Evidence: type={evidence_type}, source={evidence_source}")
                        tqdm.write(f"    Quote: {evidence_quote[:100]}..." if evidence_quote else "    Quote: N/A")
                        tqdm.write(f"    Reasoning: {reasoning}")
                        logger.info(f"  ✓ ASSIGNED speaker {best_speaker['speaker_id']} → '{target_host}' (sim={best_speaker['best_similarity']:.3f}, conf={confidence})")
                        logger.info(f"    Reasoning: {reasoning}")

                        target_identity_id = target_data['identity_id']
                        if not self.dry_run:
                            self.identity_manager.assign_speaker_to_identity(
                                speaker_id=best_speaker['speaker_id'],
                                identity_id=target_identity_id,
                                confidence=best_speaker['best_similarity'],
                                method='phase_2c_llm_verified',
                                metadata={
                                    'host_name': target_host,
                                    'similarity': best_speaker['best_similarity'],
                                    'llm_reasoning': verified['reasoning'],
                                    # New categorical evidence fields
                                    'evidence_type': verified.get('evidence_type', 'none'),
                                    'evidence_source': verified.get('evidence_source', 'unknown'),
                                    'evidence_quote': verified.get('evidence_quote', '')
                                }
                            )
                        # Track this (episode, host) pair as assigned
                        self.episode_host_pairs_assigned.add((content_id, target_identity_id))
                        self.stats['medium_verified_assigned'] += 1
                    else:
                        tqdm.write(f"  ✗ REJECTED speaker {best_speaker['speaker_id']} (sim={best_speaker['best_similarity']:.3f} to '{best_speaker['best_host_name']}', conf={confidence})")
                        tqdm.write(f"    Evidence: type={evidence_type}, source={evidence_source}")
                        tqdm.write(f"    Quote: {evidence_quote[:100]}..." if evidence_quote else "    Quote: N/A")
                        tqdm.write(f"    Reasoning: {reasoning}")
                        logger.info(f"  ✗ REJECTED speaker {best_speaker['speaker_id']} (sim={best_speaker['best_similarity']:.3f} to '{best_speaker['best_host_name']}', conf={confidence})")
                        logger.info(f"    Evidence: type={evidence_type}, source={evidence_source}")
                        logger.info(f"    Reasoning: {reasoning}")

                        # Persist rejection to DB so we don't re-process this speaker
                        PhaseTracker.record_result(
                            speaker_id=best_speaker['speaker_id'],
                            phase=2,
                            status=PhaseTracker.STATUS_REJECTED,
                            result={
                                'candidate_name': best_speaker['best_host_name'],
                                'similarity': round(best_speaker['best_similarity'], 4),
                                'confidence': confidence,
                                'reasoning': reasoning,
                                # New categorical evidence fields
                                'evidence_type': verified.get('evidence_type', 'none'),
                                'evidence_source': verified.get('evidence_source', 'unknown'),
                                'evidence_quote': verified.get('evidence_quote', '')
                            },
                            method='phase_2c_llm_rejected',
                            dry_run=self.dry_run
                        )
                        self.stats['medium_rejected'] += 1

                except Exception as e:
                    logger.error(f"Error verifying speaker {best_speaker['speaker_id']}: {e}")
                    self.stats['errors'].append(f"Speaker {best_speaker['speaker_id']}: {str(e)}")
                    continue

            logger.info(f"  ✓ LLM verified: {self.stats['medium_verified_assigned']} assigned, {self.stats['medium_rejected']} rejected")

        # Track low confidence skipped
        self.stats['low_confidence_skipped'] = len(low_confidence_pairs)

        self.stats['episodes_processed'] = total_episode_host_pairs
        logger.info(f"\nPhase 2C complete: processed {total_episode_host_pairs} (episode, host) pairs")

    # NOTE: _process_speaker_with_centroids() removed - logic is inline in _phase_2c_expand_to_full_channel()

    async def _verify_with_llm(
        self,
        speaker_id: int,
        best_host_name: str,
        host_similarities: Dict[str, float],
        known_host_names: List[str]
    ) -> Dict:
        """
        Verify speaker is host using LLM with transcript context.

        Args:
            speaker_id: Speaker ID to verify
            best_host_name: Best-matching host name
            host_similarities: Similarities to all hosts
            known_host_names: All known host names

        Returns:
            {is_host, confidence, reasoning, name}
        """
        self.stats['llm_verification_calls'] += 1

        # Get transcript context
        context = self.context_builder.get_speaker_transcript_context(speaker_id)

        if not context:
            return {
                'is_host': False,
                'confidence': 'unlikely',
                'reasoning': 'Failed to get transcript context',
                'name': ''
            }

        # Add host context for LLM
        context['known_host_names'] = known_host_names
        context['best_host_name'] = best_host_name
        context['host_similarities'] = host_similarities

        # Call LLM
        result = await self.llm_client.verify_host_identity(context)
        return result

    async def _fallback_anonymous_clustering(self, channel_id: int) -> Dict:
        """
        Fallback to anonymous clustering when no content.hosts data.

        This is the original Phase 2 approach - finds largest cluster
        without using host names.

        Args:
            channel_id: Channel ID

        Returns:
            Stats dict
        """
        logger.info("\n--- Fallback: Anonymous Clustering ---")

        # Load all speakers not yet processed by Phase 2
        with get_session() as session:
            query = text(f"""
                SELECT
                    s.id,
                    s.content_id,
                    s.embedding,
                    s.embedding_quality_score,
                    s.duration
                FROM speakers s
                JOIN content co ON s.content_id = co.content_id
                WHERE co.channel_id = :channel_id
                  AND s.embedding IS NOT NULL
                  AND s.embedding_quality_score >= :min_quality
                  AND s.speaker_identity_id IS NULL
                  AND {PhaseTracker.get_phase_filter_sql(2, 's')}
                ORDER BY co.publish_date
            """)

            results = session.execute(query, {
                'channel_id': channel_id,
                'min_quality': self.min_quality
            }).fetchall()

        if not results:
            logger.warning("No unassigned speakers found")
            return self.stats

        logger.info(f"Loaded {len(results)} unassigned speakers")

        # Parse embeddings
        speakers = []
        embeddings_list = []

        for row in results:
            try:
                embedding_str = str(row.embedding)
                embedding_values = embedding_str.strip('[]').split(',')
                embedding = np.array([float(v) for v in embedding_values], dtype=np.float32)

                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm

                speakers.append({
                    'id': row.id,
                    'content_id': row.content_id,
                    'quality': row.embedding_quality_score,
                    'duration': row.duration
                })
                embeddings_list.append(embedding)

            except Exception as e:
                logger.error(f"Error parsing embedding for speaker {row.id}: {e}")
                continue

        if len(embeddings_list) < 5:
            logger.warning("Too few valid embeddings for clustering")
            return self.stats

        embeddings = np.array(embeddings_list, dtype=np.float32)

        # Find largest cluster
        all_clusters = self._cluster_speakers_all(speakers, embeddings)
        cluster_result = all_clusters[0] if all_clusters else None

        if not cluster_result:
            logger.warning("No cluster found")
            return self.stats

        # Compute centroid using best embeddings
        cluster_indices = cluster_result['members']
        centroid, _, avg_quality = self._select_best_for_centroid(
            set(cluster_indices), speakers, embeddings
        )

        logger.info(f"Found cluster with {len(cluster_indices)} members across {cluster_result['episode_count']} episodes (centroid quality={avg_quality:.3f})")

        # Create anonymous host identity
        identity_id = None
        if not self.dry_run:
            identity_id = self.identity_manager.create_or_match_identity(
                name=f"Host (Channel {channel_id})",
                role='host',
                confidence=0.80,
                method='phase_2_anonymous_cluster',
                metadata={'channel_id': channel_id}
            )

            # Collect cluster member assignments for bulk update
            bulk_cluster_assignments = [
                {
                    'speaker_id': speakers[idx]['id'],
                    'identity_id': identity_id,
                    'confidence': 0.90,
                    'metadata': {'cluster_member': True}
                }
                for idx in cluster_indices
            ]
            if bulk_cluster_assignments:
                self.identity_manager.bulk_assign_speakers(
                    assignments=bulk_cluster_assignments,
                    method='phase_2_anonymous_cluster'
                )
            self.stats['bootstrap_speakers_assigned'] += len(cluster_indices)

        # Match remaining speakers to centroid
        cluster_speaker_ids = set(speakers[idx]['id'] for idx in cluster_indices)
        bulk_high_confidence = []

        for i, speaker in enumerate(speakers):
            if speaker['id'] in cluster_speaker_ids:
                continue

            similarity = float(np.dot(embeddings[i], centroid))

            if similarity >= self.high_threshold:
                if not self.dry_run and identity_id:
                    bulk_high_confidence.append({
                        'speaker_id': speaker['id'],
                        'identity_id': identity_id,
                        'confidence': similarity,
                        'metadata': {'similarity': similarity}
                    })
                self.stats['high_confidence_assigned'] += 1

            elif similarity >= self.medium_threshold:
                # Skip LLM in fallback mode - not enough context
                self.stats['low_confidence_skipped'] += 1

            else:
                self.stats['low_confidence_skipped'] += 1

        # Bulk assign high-confidence matches
        if bulk_high_confidence:
            self.identity_manager.bulk_assign_speakers(
                assignments=bulk_high_confidence,
                method='phase_2_anonymous_high_confidence'
            )

        return self.stats

    def _print_summary(self):
        """Print summary statistics."""
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 2 SUMMARY - Named Host Centroid Approach")
        logger.info("=" * 80)
        logger.info(f"Channels processed: {self.stats['channels_processed']}")
        logger.info(f"Qualified hosts found: {self.stats['qualified_hosts_found']}")
        logger.info(f"Centroids bootstrapped: {self.stats['centroids_bootstrapped']}")
        logger.info(f"  - Clusters LLM verified: {self.stats['cluster_llm_verified']}")
        logger.info(f"  - Clusters LLM rejected: {self.stats['cluster_llm_rejected']}")
        logger.info(f"Bootstrap speakers assigned: {self.stats['bootstrap_speakers_assigned']}")
        logger.info(f"Episodes processed (expansion): {self.stats['episodes_processed']}")
        logger.info(f"  - Episodes skipped (has host): {self.stats['episodes_skipped_has_host']}")
        logger.info(f"High confidence assigned (≥0.65): {self.stats['high_confidence_assigned']}")
        logger.info(f"Medium (LLM verified) assigned: {self.stats['medium_verified_assigned']}")
        logger.info(f"Medium rejected: {self.stats['medium_rejected']}")
        logger.info(f"Low confidence skipped (<0.40): {self.stats['low_confidence_skipped']}")
        logger.info(f"Total LLM verification calls: {self.stats['llm_verification_calls']}")
        logger.info(f"Fallback to anonymous clustering: {self.stats['fallback_to_anonymous']}")
        if self.stats['errors']:
            logger.warning(f"Errors: {len(self.stats['errors'])}")
        logger.info("=" * 80)


async def main():
    parser = argparse.ArgumentParser(
        description='Phase 2: Named Host Centroid Identification',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--channel-id', type=int, help='Process single channel')
    parser.add_argument('--project', type=str, help='Filter to project (e.g., CPRMV)')
    parser.add_argument('--max-channels', type=int, help='Max channels to process')
    parser.add_argument('--max-episodes', type=int, help='Max episodes per channel')
    parser.add_argument('--apply', action='store_true', help='Apply changes (default: dry run)')
    parser.add_argument('--high-threshold', type=float, default=0.65,
                       help='High confidence threshold (default: 0.65)')
    parser.add_argument('--medium-threshold', type=float, default=0.40,
                       help='Medium confidence threshold (default: 0.40)')
    parser.add_argument('--min-duration-pct', type=float, default=15.0,
                       help='Minimum speaker duration %% for host consideration (default: 15.0)')
    parser.add_argument('--min-host-occurrences', type=int, default=10,
                       help='Minimum host occurrences in content.hosts (default: 10)')
    parser.add_argument('--reset', action='store_true',
                       help='Reset: clear existing speaker_identity assignments for this channel before processing')
    parser.add_argument('--fresh', action='store_true',
                       help='Fresh analysis: ignore existing centroids, re-bootstrap all hosts. Updates DB if new centroid is better.')

    args = parser.parse_args()

    strategy = HostEmbeddingIdentificationStrategy(
        high_confidence_threshold=args.high_threshold,
        medium_confidence_threshold=args.medium_threshold,
        min_host_occurrences=args.min_host_occurrences,
        min_duration_pct=args.min_duration_pct,
        dry_run=not args.apply,
        max_episodes=args.max_episodes,
        fresh=args.fresh
    )

    try:
        # Reset speaker assignments if requested
        if args.reset and args.channel_id:
            logger.info(f"Resetting speaker_identity assignments for channel {args.channel_id}...")
            with get_session() as session:
                # Clear speaker assignments (but keep the identities - they're global/shared)
                result = session.execute(text("""
                    UPDATE speakers s
                    SET speaker_identity_id = NULL,
                        assignment_confidence = NULL,
                        assignment_method = NULL
                    FROM content c
                    WHERE s.content_id = c.content_id
                      AND c.channel_id = :channel_id
                      AND s.speaker_identity_id IS NOT NULL
                """), {'channel_id': args.channel_id})
                speaker_count = result.rowcount

                # Clear channel-specific role data (but NOT the centroid - it's the best voice profile)
                result2 = session.execute(text("""
                    UPDATE speaker_identities
                    SET verification_metadata = verification_metadata #- ARRAY['channel_roles', :channel_id_str]
                    WHERE verification_metadata->'channel_roles' ? :channel_id_str
                """), {'channel_id_str': str(args.channel_id)})
                role_count = result2.rowcount

                session.commit()
                logger.info(f"  Cleared {speaker_count} speaker assignments")
                logger.info(f"  Cleared {role_count} channel role entries (centroids preserved)")

        if args.channel_id:
            await strategy.run_single_channel(args.channel_id)
        elif args.project:
            # Get all channels with unassigned speakers in this project
            context_builder = ContextBuilder()

            channels = context_builder.get_channels_with_unassigned_speakers(
                project=args.project,
                min_unassigned=10,
                limit=args.max_channels
            )

            logger.info(f"Found {len(channels)} channels in project '{args.project}' with unassigned speakers")

            for i, channel in enumerate(channels, 1):
                logger.info(f"\n{'='*80}")
                logger.info(f"[{i}/{len(channels)}] Processing channel: {channel['name']}")
                logger.info(f"{'='*80}")
                logger.info(f"  Channel ID: {channel['id']}")
                logger.info(f"  Platform: {channel['platform']}")
                logger.info(f"  Unassigned speakers: {channel['unassigned_count']:,}")
                logger.info(f"  Total episodes: {channel['episode_count']:,}")

                try:
                    # Reset stats for each channel
                    strategy.stats = {
                        'channels_processed': 0,
                        'qualified_hosts_found': 0,
                        'centroids_bootstrapped': 0,
                        'cluster_llm_verified': 0,
                        'cluster_llm_rejected': 0,
                        'bootstrap_speakers_assigned': 0,
                        'episodes_processed': 0,
                        'episodes_skipped_has_host': 0,
                        'high_confidence_assigned': 0,
                        'medium_verified_assigned': 0,
                        'medium_rejected': 0,
                        'low_confidence_skipped': 0,
                        'llm_verification_calls': 0,
                        'identities_created': 0,
                        'fallback_to_anonymous': 0,
                        'errors': []
                    }
                    await strategy.run_single_channel(channel['id'])
                except Exception as e:
                    logger.error(f"Error processing channel {channel['id']}: {e}")
                    continue
        else:
            logger.error("Either --channel-id or --project is required")
            sys.exit(1)

        strategy._print_summary()

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Clean up aiohttp session
        if strategy and strategy.llm_client:
            await strategy.llm_client.close()


if __name__ == '__main__':
    asyncio.run(main())
