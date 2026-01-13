#!/usr/bin/env python3
"""
Anchor-Canopy Clustering with Text-Verified Anchors (Phase 3)
================================================================

NEW unified Phase 3: Uses text-verified speakers from Phase 2 as trusted anchors,
then expands clusters using multi-gate validation to assign all remaining speakers.

This replaces both old Phase 3 (verified_centroid_generation.py) and
Phase 4 (embedding_propagation.py).

Key Features:
- Text-verified speakers (~39K with 'certain' status) become anchors
- Multi-gate validation prevents transitive chain errors
- Name collision detection (same name, different people)
- Unnamed identity creation for clusters without text evidence
- Pure embedding-based after Phase 2 (no LLM calls)

Algorithm:
1. Load text-verified speakers as anchors (Phase 2 certain)
2. Build FAISS index for ALL speakers
3. Pre-compute k-NN neighborhoods
4. For each name group:
   a. Detect name collisions via clustering
   b. Select best anchor per distinct person
   c. Grow cluster using multi-gate validation
   d. Build quality-weighted centroid
   e. Create/update identity + assign all members
5. Handle unnamed clusters (speakers not claimed by any anchor)

Usage:
    # Dry run
    python -m src.speaker_identification.strategies.anchor_verified_clustering

    # Run on project
    python -m src.speaker_identification.strategies.anchor_verified_clustering \\
        --project CPRMV --apply
"""

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from src.utils.paths import get_project_root, get_config_path
from typing import Dict, List, Optional, Set, Tuple

import faiss
import numpy as np
from tqdm import tqdm

project_root = str(get_project_root())
sys.path.append(project_root)

from sqlalchemy import text

from src.database.session import get_session
from src.utils.logger import setup_worker_logger
from src.utils.config import get_project_date_range

logger = setup_worker_logger('speaker_identification.anchor_verified_clustering')

# Console logging
import logging
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

# Phase key for identification_details JSONB
PHASE_KEY = "phase3"


@dataclass
class Phase3ClusterConfig:
    """Configuration for anchor-canopy clustering in Phase 3.

    These thresholds are adapted from cluster_speakers.py but tuned
    for the text-verified anchor scenario where we have high-confidence
    starting points.

    Key insight: LLM labels are NOT infallible. Multiple verified speakers
    agreeing on a name is much more reliable than a single label. We adjust
    gate strictness based on this confidence.
    """
    # Multi-gate thresholds
    tau_anchor: float = 0.68      # Gate 1: Similarity to anchor/prototype
    beta: float = 0.82            # Gate 2: Relative density check
    tau_sent: float = 0.65        # Gate 3: Sentinel similarity
    tau_frontier: float = 0.65   # Frontier expansion threshold

    # Confidence-based expansion control
    # If 2+ verified speakers have the same name, we have HIGH confidence
    # (multiple independent LLM calls agreed) - only these get expanded
    min_verified_for_high_confidence: int = 2
    expand_single_verified: bool = False  # Don't expand clusters with only 1 verified speaker

    # Name collision detection
    collision_threshold: float = 0.65  # Cluster same-name speakers (lowered from 0.75)
    collision_skip_if_high_consensus: int = 50  # Skip collision detection if this many verified speakers agree

    # Centroid building
    centroid_min_quality: float = 0.65
    centroid_max_samples: int = 50

    # Resource limits
    Q_max: int = 3000    # Max candidates per anchor
    S_max: int = 500     # Max cluster size

    # k-NN parameters
    k0: int = 128        # Initial k-NN neighbors
    k_expand: int = 256  # Candidates for expansion

    # Prototype parameters
    coreset_M: int = 32   # Max coreset members for prototype
    refresh_R: int = 16   # Prototype refresh interval

    # Minimum cluster size for unnamed identities
    min_unnamed_cluster_size: int = 3


class CanopyCluster:
    """Represents a single canopy cluster during growth.

    Adapted from cluster_speakers.py for use with text-verified anchors.
    Supports confidence-based gate relaxation when multiple verified speakers agree.
    """

    def __init__(self, anchor_id: int, anchor_idx: int, anchor_embedding: np.ndarray,
                 anchor_name: str, config: Phase3ClusterConfig,
                 is_high_confidence: bool = False, n_verified: int = 1):
        self.anchor_id = anchor_id
        self.anchor_idx = anchor_idx  # Array index for k-NN lookups
        self.anchor_embedding = anchor_embedding
        self.anchor_name = anchor_name
        self.config = config

        # Confidence level (True if multiple verified speakers agree on this name)
        self.is_high_confidence = is_high_confidence
        self.n_verified = n_verified

        # Cluster members
        self.members: Set[int] = {anchor_id}
        self.member_embeddings: Dict[int, np.ndarray] = {anchor_id: anchor_embedding}
        self.member_names: Dict[int, Optional[str]] = {anchor_id: anchor_name}

        # Prototype (running mean)
        self.prototype = anchor_embedding.copy()
        self.additions_since_refresh = 0

        # Sentinels (diverse representatives for validation)
        self.sentinels: List[Tuple[int, np.ndarray]] = [(anchor_id, anchor_embedding)]

        # Gate validation results for members
        self.gate_results: Dict[int, str] = {anchor_id: "anchor"}

        # Cache for anchor's neighborhood
        self.anchor_neighbors_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self.anchor_density_cache: Optional[float] = None

    def update_prototype(self, force: bool = False):
        """Update cluster prototype using robust mean of coreset."""
        if not force and self.additions_since_refresh < self.config.refresh_R:
            return

        # Select coreset: top M members most similar to current prototype
        if len(self.members) <= self.config.coreset_M:
            coreset_ids = list(self.members)
        else:
            sims = []
            for mid, emb in self.member_embeddings.items():
                sim = np.dot(emb, self.prototype)
                sims.append((mid, sim))
            sims.sort(key=lambda x: x[1], reverse=True)
            coreset_ids = [s[0] for s in sims[:self.config.coreset_M]]

        # Compute robust mean
        coreset_embeddings = [self.member_embeddings[mid] for mid in coreset_ids]
        self.prototype = np.mean(coreset_embeddings, axis=0)
        self.prototype = self.prototype / np.linalg.norm(self.prototype)

        self.additions_since_refresh = 0
        self._update_sentinels()

    def _update_sentinels(self, n_sentinels: int = 5):
        """Select diverse sentinel members for validation."""
        if len(self.members) <= n_sentinels:
            self.sentinels = [(mid, self.member_embeddings[mid]) for mid in self.members]
            return

        # Start with anchor
        sentinel_ids = [self.anchor_id]
        sentinel_embeddings = [self.anchor_embedding]

        # Greedily add diverse members
        while len(sentinel_ids) < n_sentinels:
            best_id = None
            best_min_sim = -1

            for mid, emb in self.member_embeddings.items():
                if mid in sentinel_ids:
                    continue
                min_sim = min(np.dot(emb, s_emb) for s_emb in sentinel_embeddings)
                if min_sim > best_min_sim:
                    best_min_sim = min_sim
                    best_id = mid

            if best_id:
                sentinel_ids.append(best_id)
                sentinel_embeddings.append(self.member_embeddings[best_id])

        self.sentinels = list(zip(sentinel_ids, sentinel_embeddings))

    def validate_candidate(
        self,
        candidate_emb: np.ndarray,
        candidate_name: Optional[str],
        anchor_knn_sims: np.ndarray
    ) -> Tuple[bool, str]:
        """
        Multi-gate validation for candidate addition.

        Gates:
        1. Prototype similarity threshold (tau_anchor)
        2. Relative density - compare to anchor's local density (beta)
        3. Sentinel check - similarity to diverse cluster representatives
        4. Name consistency - don't mix different verified names

        Args:
            candidate_emb: Candidate embedding vector
            candidate_name: Phase2 identified name if any
            anchor_knn_sims: Similarity scores of anchor's k-NN neighbors (for density)

        Returns:
            Tuple of (passed, rejection_reason_or_gate_info)
        """
        # Gate 1: Prototype similarity
        proto_sim = np.dot(candidate_emb, self.prototype)
        if proto_sim < self.config.tau_anchor:
            return False, f"proto_sim={proto_sim:.3f}<{self.config.tau_anchor}"

        # Gate 2: Relative density check
        # Cache anchor's local density
        if self.anchor_density_cache is None:
            # Use first 10 neighbors (excluding self) for density
            self.anchor_density_cache = float(np.mean(anchor_knn_sims[1:11]))

        if proto_sim < self.config.beta * self.anchor_density_cache:
            return False, f"density={proto_sim:.3f}<{self.config.beta}*{self.anchor_density_cache:.3f}"

        # Gate 3: Sentinel check (must be similar to some sentinels)
        if len(self.sentinels) >= 3:
            sentinel_sims = [np.dot(candidate_emb, s_emb) for _, s_emb in self.sentinels]
            passing_sentinels = sum(1 for sim in sentinel_sims if sim >= self.config.tau_sent)
            min_required = min(2, max(1, len(self.sentinels) // 2))
            if passing_sentinels < min_required:
                return False, f"sentinel_{passing_sentinels}<{min_required}"

        # Gate 5: Name consistency
        # If candidate has a verified name from Phase 2, it must match cluster's anchor name
        if candidate_name and candidate_name != self.anchor_name:
            return False, f"name_conflict:{candidate_name}!={self.anchor_name}"

        return True, f"passed:sim={proto_sim:.3f}"

    def add_member(self, member_id: int, member_emb: np.ndarray,
                   member_name: Optional[str], gate_result: str):
        """Add validated member to cluster."""
        self.members.add(member_id)
        self.member_embeddings[member_id] = member_emb
        self.member_names[member_id] = member_name
        self.gate_results[member_id] = gate_result
        self.additions_since_refresh += 1

        # Update prototype periodically
        if self.additions_since_refresh >= self.config.refresh_R:
            self.update_prototype()


class AnchorVerifiedClusteringStrategy:
    """
    Phase 3: Anchor-Canopy Clustering with Text-Verified Anchors.

    Uses text-verified speakers from Phase 2 as trusted anchors,
    then expands clusters to all remaining speakers using multi-gate
    validation. Creates speaker identities and assigns all cluster members.
    """

    def __init__(
        self,
        config: Phase3ClusterConfig = None,
        dry_run: bool = True,
        force: bool = False,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        name_filter: Optional[str] = None
    ):
        self.config = config or Phase3ClusterConfig()
        self.dry_run = dry_run
        self.force = force  # If True, reprocess already-assigned speakers
        self.start_date = start_date
        self.end_date = end_date
        self.name_filter = name_filter  # Filter to specific speaker name (for testing)

        # Tracking
        self.claimed_speakers: Set[int] = set()
        self.clusters: List[CanopyCluster] = []

        self.stats = {
            'unique_names_found': 0,
            'names_processed': 0,
            'name_collisions_detected': 0,
            'clusters_created': 0,
            'high_confidence_clusters': 0,
            'low_confidence_clusters': 0,
            'identities_created': 0,
            'identities_updated': 0,
            'speakers_assigned': 0,
            'speakers_skipped_already_assigned': 0,
            'unnamed_clusters_created': 0,
            'speakers_in_unnamed': 0,
            'speakers_unclaimed': 0,
            'errors': []
        }

    def _load_all_speakers(self, project: str) -> Dict[int, Dict]:
        """Load all speakers with embeddings for the project.

        By default, skips speakers already assigned by Phase 3.
        Use --force to include them.
        """
        with get_session() as session:
            filters = ["s.embedding IS NOT NULL"]
            params = {'min_quality': 0.50}

            if project:
                filters.append(":project = ANY(c.projects)")
                params['project'] = project

            if self.start_date:
                filters.append("c.publish_date >= :start_date")
                params['start_date'] = self.start_date
            if self.end_date:
                filters.append("c.publish_date < :end_date")
                params['end_date'] = self.end_date

            # Skip already-assigned speakers unless --force
            if not self.force:
                filters.append("""(
                    s.identification_details IS NULL
                    OR NOT (s.identification_details ? 'phase3')
                    OR s.identification_details->'phase3'->>'status' != 'assigned'
                )""")

            filter_clause = " AND ".join(filters)

            query = text(f"""
                SELECT
                    s.id as speaker_id,
                    s.content_id,
                    s.embedding,
                    s.duration,
                    COALESCE(s.embedding_quality_score, 0.5) as quality,
                    s.speaker_identity_id,
                    s.identification_details->'phase2'->>'status' as phase2_status,
                    s.identification_details->'phase2'->>'identified_name' as phase2_name,
                    c.channel_id,
                    c.duration as episode_duration
                FROM speakers s
                JOIN content c ON s.content_id = c.content_id
                WHERE {filter_clause}
                  AND COALESCE(s.embedding_quality_score, 0.5) >= :min_quality
                  AND c.is_stitched = true
                ORDER BY s.id
            """)

            results = session.execute(query, params).fetchall()

            speakers = {}
            for row in results:
                emb_data = row.embedding
                if isinstance(emb_data, str):
                    emb_data = json.loads(emb_data)
                embedding = np.array(emb_data, dtype=np.float32)
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm

                speakers[row.speaker_id] = {
                    'speaker_id': row.speaker_id,
                    'content_id': row.content_id,
                    'embedding': embedding,
                    'quality': row.quality,
                    'duration': row.duration,
                    'episode_duration': row.episode_duration,
                    'phase2_status': row.phase2_status,
                    'phase2_name': row.phase2_name,
                    'channel_id': row.channel_id,
                    'speaker_identity_id': row.speaker_identity_id
                }

            return speakers

    def _load_text_verified_speakers(
        self, project: str, alias_mapping: Dict[str, str] = None
    ) -> Dict[str, List[Dict]]:
        """Load text-verified speakers grouped by identified name.

        By default, skips speakers already assigned by Phase 3.
        Use --force to include them.

        If alias_mapping is provided, names are normalized before grouping.
        This ensures "Ezra" and "Ezra Levant" end up in the same group
        if they're aliases.
        """
        alias_mapping = alias_mapping or {}

        with get_session() as session:
            filters = [
                "s.identification_details->'phase2'->>'status' = 'certain'",
                "s.identification_details->'phase2'->>'identified_name' IS NOT NULL",
                "s.embedding IS NOT NULL"
            ]
            params = {'min_quality': 0.50}

            if project:
                filters.append(":project = ANY(c.projects)")
                params['project'] = project

            if self.start_date:
                filters.append("c.publish_date >= :start_date")
                params['start_date'] = self.start_date
            if self.end_date:
                filters.append("c.publish_date < :end_date")
                params['end_date'] = self.end_date

            # Skip already-assigned speakers unless --force
            if not self.force:
                filters.append("""(
                    s.identification_details->'phase3'->>'status' IS NULL
                    OR s.identification_details->'phase3'->>'status' != 'assigned'
                )""")

            filter_clause = " AND ".join(filters)

            query = text(f"""
                SELECT
                    s.id as speaker_id,
                    s.content_id,
                    s.embedding,
                    s.duration,
                    COALESCE(s.embedding_quality_score, 0.5) as quality,
                    s.identification_details->'phase2'->>'evidence_type' as evidence_type,
                    s.identification_details->'phase2'->>'identified_name' as identified_name,
                    c.channel_id
                FROM speakers s
                JOIN content c ON s.content_id = c.content_id
                WHERE {filter_clause}
                  AND COALESCE(s.embedding_quality_score, 0.5) >= :min_quality
                ORDER BY s.identification_details->'phase2'->>'identified_name',
                         s.embedding_quality_score DESC NULLS LAST
            """)

            results = session.execute(query, params).fetchall()

            by_name = defaultdict(list)
            alias_normalized_count = 0

            for row in results:
                emb_data = row.embedding
                if isinstance(emb_data, str):
                    emb_data = json.loads(emb_data)
                embedding = np.array(emb_data, dtype=np.float32)
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm

                # Normalize name using Phase 1 alias mapping
                original_name = row.identified_name
                normalized_name = alias_mapping.get(original_name, original_name)
                if normalized_name != original_name:
                    alias_normalized_count += 1

                by_name[normalized_name].append({
                    'speaker_id': row.speaker_id,
                    'content_id': row.content_id,
                    'embedding': embedding,
                    'quality': row.quality,
                    'duration': row.duration,
                    'evidence_type': row.evidence_type,
                    'channel_id': row.channel_id,
                    'original_name': original_name  # Keep track of original
                })

            if alias_normalized_count > 0:
                logger.info(f"  Normalized {alias_normalized_count} speakers via Phase 1 aliases")

            return dict(by_name)

    def _build_faiss_index(self, speakers: Dict[int, Dict]) -> Tuple[faiss.IndexFlatIP, List[int], np.ndarray]:
        """Build FAISS index from all speaker embeddings."""
        speaker_ids = list(speakers.keys())
        embeddings = np.array([speakers[sid]['embedding'] for sid in speaker_ids], dtype=np.float32)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        return index, speaker_ids, embeddings

    def _precompute_knn(
        self,
        index: faiss.IndexFlatIP,
        embeddings: np.ndarray,
        k: int = 128
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Pre-compute k-NN for all speakers."""
        sims, indices = index.search(embeddings, k)
        return indices, sims

    def _load_alias_mappings(self) -> Dict[str, str]:
        """Load alias -> canonical name mappings from channel_host_cache.

        Phase 1 already consolidates name variations (e.g., "Ezra" -> "Ezra Levant").
        We use these mappings to normalize names before clustering.
        """
        with get_session() as session:
            query = text("""
                SELECT host_name, aliases
                FROM channel_host_cache
                WHERE aliases IS NOT NULL
                  AND array_length(aliases, 1) > 1
            """)
            results = session.execute(query).fetchall()

            mapping = {}
            for row in results:
                canonical = row.host_name
                for alias in row.aliases:
                    if alias and alias != canonical:
                        # Map alias to canonical (prefer longer/more complete names)
                        if alias not in mapping:
                            mapping[alias] = canonical
                        elif len(canonical) > len(mapping[alias]):
                            mapping[alias] = canonical

            return mapping

    def _normalize_name(self, name: str, alias_mapping: Dict[str, str]) -> str:
        """Normalize a name using alias mapping."""
        return alias_mapping.get(name, name)

    def _find_dense_core(
        self,
        speakers: List[Dict],
        threshold: float = None
    ) -> Tuple[List[Dict], List[Dict], Dict]:
        """
        Find the largest subset where ALL pairs have similarity >= threshold.

        This is the Dense Core Detection algorithm - a greedy maximal clique approach
        that finds a clean subset of speakers before building the prototype.

        Key insight: LLM labels have ~15% error rate. Rather than building a prototype
        from ALL verified speakers (contaminated), we first find the dense core where
        every speaker is similar to every other speaker. This clean core becomes the
        prototype foundation.

        Algorithm (Greedy Maximal Clique):
        1. Build pairwise similarity matrix (O(n²))
        2. Find seed: speaker with highest minimum similarity to any other
        3. Greedily expand: add speakers who have sim >= threshold to ALL current members
        4. Return the dense core

        Returns:
            Tuple of (core_speakers, excluded_speakers, stats)
        """
        if threshold is None:
            threshold = self.config.tau_anchor

        n = len(speakers)

        # Edge cases
        if n == 0:
            return [], [], {'checked': 0, 'core_size': 0, 'excluded': 0}
        if n == 1:
            return speakers, [], {'checked': 1, 'core_size': 1, 'excluded': 0}
        if n == 2:
            # For 2 speakers, just check if they're similar enough
            sim = float(np.dot(speakers[0]['embedding'], speakers[1]['embedding']))
            if sim >= threshold:
                return speakers, [], {'checked': 2, 'core_size': 2, 'excluded': 0, 'min_sim': sim}
            else:
                # Return the one with higher quality
                if speakers[0].get('quality', 0) >= speakers[1].get('quality', 0):
                    return [speakers[0]], [speakers[1]], {'checked': 2, 'core_size': 1, 'excluded': 1, 'min_sim': sim}
                else:
                    return [speakers[1]], [speakers[0]], {'checked': 2, 'core_size': 1, 'excluded': 1, 'min_sim': sim}

        # Build pairwise similarity matrix (O(n²) via matrix multiply)
        embeddings = np.array([s['embedding'] for s in speakers], dtype=np.float32)
        sim_matrix = embeddings @ embeddings.T

        # Find seed: speaker with highest "worst connection" (most central)
        # This is the speaker who is most similar to their least similar neighbor
        min_sims = np.min(sim_matrix, axis=1)  # Worst similarity for each speaker
        seed_idx = int(np.argmax(min_sims))    # Most central speaker

        # Greedy clique expansion
        core_indices = {seed_idx}
        candidates = set(range(n)) - core_indices

        while candidates:
            best_candidate = None
            best_min_sim = -1

            for c in candidates:
                # Check similarity to ALL current core members
                min_sim_to_core = min(float(sim_matrix[c, m]) for m in core_indices)
                if min_sim_to_core >= threshold and min_sim_to_core > best_min_sim:
                    best_min_sim = min_sim_to_core
                    best_candidate = c

            if best_candidate is None:
                break  # No more candidates pass threshold to all core members

            core_indices.add(best_candidate)
            candidates.remove(best_candidate)

        # Build result lists
        core_speakers = [speakers[i] for i in sorted(core_indices)]
        excluded_speakers = [speakers[i] for i in range(n) if i not in core_indices]

        # Compute stats for the core
        core_embeddings = np.array([s['embedding'] for s in core_speakers], dtype=np.float32)
        if len(core_speakers) >= 2:
            core_sim_matrix = core_embeddings @ core_embeddings.T
            # Get min similarity (excluding diagonal)
            np.fill_diagonal(core_sim_matrix, 1.0)  # Ignore self-similarity
            min_core_sim = float(np.min(core_sim_matrix))
            mean_core_sim = float((np.sum(core_sim_matrix) - len(core_speakers)) / (len(core_speakers) * (len(core_speakers) - 1)))
        else:
            min_core_sim = 1.0
            mean_core_sim = 1.0

        stats = {
            'checked': n,
            'core_size': len(core_speakers),
            'excluded': len(excluded_speakers),
            'min_sim': min_core_sim,
            'mean_sim': mean_core_sim,
            'seed_idx': seed_idx,
            'seed_min_sim': float(min_sims[seed_idx])
        }

        return core_speakers, excluded_speakers, stats

    def _detect_name_collisions(
        self,
        speakers: List[Dict]
    ) -> List[List[Dict]]:
        """
        Detect name collisions - same name, different people.

        Clusters speakers with the same name by embedding similarity.
        Returns list of clusters, where each cluster is a distinct person.

        If the number of verified speakers exceeds collision_skip_if_high_consensus,
        we skip collision detection entirely - if 50+ LLM labels agree on a name,
        the probability of it being different people is essentially zero.
        """
        if len(speakers) < 2:
            return [speakers]

        # Skip collision detection for high-consensus names
        if len(speakers) >= self.config.collision_skip_if_high_consensus:
            logger.info(f"    Skipping collision detection ({len(speakers)} verified speakers = high consensus)")
            return [speakers]

        embeddings = np.array([s['embedding'] for s in speakers], dtype=np.float32)
        n_speakers = len(embeddings)

        # Build small FAISS index
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        # Find neighbors
        K = min(50, n_speakers)
        sims, indices = index.search(embeddings, K)

        # Union-find clustering at collision_threshold
        parent = list(range(n_speakers))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i in range(n_speakers):
            for j, sim in zip(indices[i], sims[i]):
                if i != j and sim >= self.config.collision_threshold:
                    union(i, j)

        # Group by cluster
        cluster_members = defaultdict(list)
        for i in range(n_speakers):
            cluster_members[find(i)].append(i)

        # Convert to speaker lists
        clusters = []
        for members in cluster_members.values():
            clusters.append([speakers[m] for m in members])

        # Sort by size descending
        clusters.sort(key=lambda c: len(c), reverse=True)

        return clusters

    def _select_best_anchor(self, speakers: List[Dict]) -> Dict:
        """Select the best anchor from a group of verified speakers."""
        # Prefer high quality, high duration, self-introduction evidence
        def score(s):
            quality_score = s['quality'] * 100
            duration_score = min(s['duration'] / 60, 10)  # Cap at 10 minutes
            evidence_bonus = 20 if s.get('evidence_type') == 'self_introduction' else 0
            return quality_score + duration_score + evidence_bonus

        return max(speakers, key=score)

    def _grow_cluster_multigate(
        self,
        anchor: Dict,
        all_speakers: Dict[int, Dict],
        speaker_ids: List[int],
        embeddings: np.ndarray,
        knn_indices: np.ndarray,
        knn_sims: np.ndarray,
        id_to_idx: Dict[int, int],
        is_high_confidence: bool = False,
        n_verified: int = 1,
        verified_speakers: List[Dict] = None,
        verbose: bool = False
    ) -> Tuple[CanopyCluster, Dict]:
        """
        Grow a cluster from anchor using multi-gate validation.

        Uses frontier-based expansion: start with anchor's neighbors,
        then expand frontier as new members are added.

        For HIGH-CONFIDENCE clusters (2+ verified), we seed the cluster with
        ALL verified speakers. This builds a stronger prototype and sentinels
        from multiple trusted sources.

        Args:
            is_high_confidence: True if 2+ verified speakers agree on the name
            n_verified: Number of verified speakers for this name group
            verified_speakers: List of all verified speakers for this name group

        Returns:
            Tuple of (cluster, gate_stats) for logging
        """
        anchor_id = anchor['speaker_id']
        anchor_idx = id_to_idx[anchor_id]
        anchor_name = anchor.get('identified_name') or all_speakers[anchor_id].get('phase2_name')

        cluster = CanopyCluster(
            anchor_id=anchor_id,
            anchor_idx=anchor_idx,
            anchor_embedding=anchor['embedding'],
            anchor_name=anchor_name,
            config=self.config,
            is_high_confidence=is_high_confidence,
            n_verified=n_verified
        )

        # Mark anchor as claimed
        self.claimed_speakers.add(anchor_id)

        # For HIGH-CONFIDENCE clusters, use Dense Core Detection to find clean subset
        # Key insight: LLM labels have ~15% error rate. Rather than building prototype
        # from ALL verified speakers (contaminated), find dense core FIRST where
        # every speaker is similar to every other speaker.
        dense_core_stats = None
        if is_high_confidence and verified_speakers:
            # Find dense core BEFORE seeding - this is the clean subset
            core_speakers, excluded_speakers, dense_core_stats = self._find_dense_core(
                verified_speakers, threshold=self.config.tau_anchor
            )

            # Seed cluster with ONLY dense core speakers (clean prototype)
            for sp in core_speakers:
                sp_id = sp['speaker_id']
                if sp_id != anchor_id and sp_id not in self.claimed_speakers:
                    if sp_id in id_to_idx:
                        cluster.add_member(sp_id, sp['embedding'], anchor_name, "dense_core")
                        self.claimed_speakers.add(sp_id)

            # Update prototype from clean core
            cluster.update_prototype(force=True)

            # Give excluded speakers a second chance: validate against clean prototype
            # Many excluded speakers are valid (just not pairwise-similar to ALL core members)
            # but they may still pass the multi-gate validation against the clean prototype
            readmitted = 0
            truly_excluded = 0
            for sp in excluded_speakers:
                sp_id = sp['speaker_id']
                if sp_id in self.claimed_speakers:
                    continue

                # Validate against clean prototype using same gates as expansion
                passed, reason = cluster.validate_candidate(
                    candidate_emb=sp['embedding'],
                    candidate_name=anchor_name,  # They have the same name
                    anchor_knn_sims=knn_sims[id_to_idx[anchor_id]] if anchor_id in id_to_idx else np.ones(10)
                )

                if passed:
                    # Readmit to cluster with special marker
                    cluster.add_member(sp_id, sp['embedding'], anchor_name, "readmitted_after_dense_core")
                    self.claimed_speakers.add(sp_id)
                    readmitted += 1
                else:
                    # Truly excluded - mark as claimed to prevent joining other clusters
                    self.claimed_speakers.add(sp_id)
                    cluster.gate_results[sp_id] = f"excluded:{reason}"
                    truly_excluded += 1

            if readmitted > 0 or truly_excluded > 0:
                dense_core_stats['readmitted'] = readmitted
                dense_core_stats['truly_excluded'] = truly_excluded
                # NOTE: Intentionally NOT updating prototype/sentinels after readmissions
                # Keep sentinels from dense core only for consistent validation
                # The prototype will naturally evolve during expansion

        # Get anchor's neighbors
        anchor_idx = id_to_idx[anchor_id]
        anchor_knn = knn_indices[anchor_idx]
        anchor_knn_sims = knn_sims[anchor_idx]

        # Initialize frontier with anchor's close neighbors
        frontier = set()
        highest_neighbor_sim = 0.0
        neighbors_above_threshold = 0
        for nbr_idx, sim in zip(anchor_knn[:self.config.k_expand], anchor_knn_sims[:self.config.k_expand]):
            if sim > highest_neighbor_sim and nbr_idx != anchor_idx:
                highest_neighbor_sim = sim
            if sim >= self.config.tau_frontier:
                neighbors_above_threshold += 1
                nbr_id = speaker_ids[nbr_idx]
                if nbr_id not in self.claimed_speakers:
                    frontier.add(nbr_id)

        initial_frontier_size = len(frontier)

        # Debug: log when frontier is empty or small
        if initial_frontier_size == 0 and verbose:
            logger.debug(f"    Empty frontier! Highest neighbor sim: {highest_neighbor_sim:.3f}, "
                        f"threshold: {self.config.tau_frontier}")

        # Gate statistics
        gate_stats = {
            'initial_frontier': initial_frontier_size,
            'highest_neighbor_sim': highest_neighbor_sim,
            'candidates_evaluated': 0,
            'passed': 0,
            'rejected_by_gate': {
                'proto_sim': 0,
                'density': 0,
                'sentinel': 0,
                'name_conflict': 0
            }
        }

        # Process frontier
        # Gates ensure quality - no arbitrary limits needed
        while frontier:
            # Get next candidate (closest to prototype)
            best_candidate = None
            best_sim = -1

            for cand_id in frontier:
                cand_emb = all_speakers[cand_id]['embedding']
                sim = np.dot(cand_emb, cluster.prototype)
                if sim > best_sim:
                    best_sim = sim
                    best_candidate = cand_id

            if best_candidate is None:
                break

            frontier.remove(best_candidate)
            gate_stats['candidates_evaluated'] += 1

            if best_candidate in self.claimed_speakers:
                continue

            # Get candidate info
            cand_data = all_speakers[best_candidate]
            cand_emb = cand_data['embedding']
            cand_name = cand_data.get('phase2_name')
            cand_idx = id_to_idx[best_candidate]
            cand_knn = knn_indices[cand_idx]

            # Validate through multi-gate (using array indices for mutual rank)
            passed, reason = cluster.validate_candidate(
                candidate_emb=cand_emb,
                candidate_name=cand_name,
                anchor_knn_sims=anchor_knn_sims
            )

            if passed:
                cluster.add_member(best_candidate, cand_emb, cand_name, reason)
                self.claimed_speakers.add(best_candidate)
                gate_stats['passed'] += 1

                # Expand frontier with new member's neighbors
                for nbr_idx in cand_knn[:self.config.k_expand]:
                    nbr_id = speaker_ids[nbr_idx]
                    if nbr_id not in self.claimed_speakers and nbr_id not in cluster.members:
                        frontier.add(nbr_id)
            else:
                # Track rejection reason
                if 'proto_sim' in reason:
                    gate_stats['rejected_by_gate']['proto_sim'] += 1
                elif 'density' in reason:
                    gate_stats['rejected_by_gate']['density'] += 1
                elif 'sentinel' in reason:
                    gate_stats['rejected_by_gate']['sentinel'] += 1
                elif 'name_conflict' in reason:
                    gate_stats['rejected_by_gate']['name_conflict'] += 1

        # Add dense core stats
        gate_stats['dense_core'] = dense_core_stats

        # Final prototype update
        cluster.update_prototype(force=True)

        return cluster, gate_stats

    def _build_quality_weighted_centroid(self, cluster: CanopyCluster) -> Tuple[np.ndarray, float, int, List[np.ndarray]]:
        """Build centroid from dense core members only, plus variation centroids.

        The number of variations is dynamic based on embedding diversity:
        - Tight cluster (high mean similarity) → fewer variations needed
        - Spread cluster (low mean similarity) → more variations to capture range

        Returns:
            Tuple of (centroid, quality, sample_count, variation_centroids)
            - centroid: Average of dense core members
            - variation_centroids: Diverse embeddings from dense core for robust matching
        """
        # Prioritize dense core members for centroid
        dense_core_members = [
            (mid, cluster.member_embeddings[mid])
            for mid in cluster.members
            if cluster.gate_results.get(mid) in ("dense_core", "anchor")
        ]

        # Fallback to all members if no dense core (shouldn't happen)
        if not dense_core_members:
            dense_core_members = [(mid, cluster.member_embeddings[mid]) for mid in cluster.members]

        embeddings = [emb for _, emb in dense_core_members]

        # Limit to max samples for centroid
        if len(embeddings) > self.config.centroid_max_samples:
            embeddings = embeddings[:self.config.centroid_max_samples]

        # Build primary centroid from dense core
        centroid = np.mean(embeddings, axis=0)
        centroid = centroid / np.linalg.norm(centroid)

        # Compute embedding diversity to determine number of variations
        # Higher mean similarity = tighter cluster = fewer variations needed
        all_embs = [emb for _, emb in dense_core_members]
        n_embs = len(all_embs)

        if n_embs <= 2:
            # Very few samples - just use what we have
            n_variations = n_embs
            mean_sim = 1.0
        else:
            # Compute mean pairwise similarity
            emb_matrix = np.array(all_embs, dtype=np.float32)
            sim_matrix = emb_matrix @ emb_matrix.T
            # Get upper triangle (excluding diagonal)
            upper_tri = sim_matrix[np.triu_indices(n_embs, k=1)]
            mean_sim = float(np.mean(upper_tri))

            # Dynamic variation count based on diversity:
            # - mean_sim > 0.90: very tight, 1 variation
            # - mean_sim > 0.85: tight, 2 variations
            # - mean_sim > 0.80: moderate, 3 variations
            # - mean_sim > 0.75: spread, 4 variations
            # - mean_sim <= 0.75: very spread, 5+ variations
            if mean_sim > 0.90:
                n_variations = 1
            elif mean_sim > 0.85:
                n_variations = 2
            elif mean_sim > 0.80:
                n_variations = 3
            elif mean_sim > 0.75:
                n_variations = 4
            else:
                n_variations = min(6, n_embs)  # Cap at 6

        # Can't have more variations than embeddings
        n_variations = min(n_variations, n_embs)

        # Build variation centroids (diverse embeddings from dense core)
        variation_centroids = []
        if n_variations >= 1:
            # Start with embedding closest to centroid (most representative)
            sims_to_centroid = [float(np.dot(emb, centroid)) for emb in all_embs]
            first_idx = int(np.argmax(sims_to_centroid))
            variation_centroids.append(all_embs[first_idx])
            used_indices = {first_idx}

            # Greedily add diverse embeddings
            while len(variation_centroids) < n_variations and len(used_indices) < n_embs:
                best_idx = None
                best_min_sim = 2.0  # Start high, find minimum

                for i, emb in enumerate(all_embs):
                    if i in used_indices:
                        continue
                    # Find embedding with lowest max similarity to current variations
                    # (most different from all current variations)
                    max_sim_to_variations = max(float(np.dot(emb, v)) for v in variation_centroids)
                    if max_sim_to_variations < best_min_sim:
                        best_min_sim = max_sim_to_variations
                        best_idx = i

                if best_idx is not None:
                    variation_centroids.append(all_embs[best_idx])
                    used_indices.add(best_idx)
                else:
                    break

        logger.debug(f"  Diversity: mean_sim={mean_sim:.3f}, n_variations={len(variation_centroids)}")

        return centroid, 0.90, len(dense_core_members), variation_centroids

    def _infer_role(self, cluster: CanopyCluster, all_speakers: Dict[int, Dict]) -> str:
        """Infer role (host/guest) from cluster characteristics."""
        channels = set()
        episodes = set()

        for mid in cluster.members:
            if mid in all_speakers:
                channels.add(all_speakers[mid].get('channel_id'))
                episodes.add(all_speakers[mid].get('content_id'))

        if len(episodes) >= 10:
            return 'host'
        elif len(episodes) >= 5 and len(channels) == 1:
            return 'co_host'
        else:
            return 'guest'

    def _compute_cluster_stats(
        self,
        cluster: CanopyCluster,
        all_speakers: Dict[int, Dict]
    ) -> Dict:
        """Compute statistics for a cluster (hours spoken, channels, episodes)."""
        total_duration = 0.0
        channels = defaultdict(int)
        episodes = set()

        for mid in cluster.members:
            if mid in all_speakers:
                sp = all_speakers[mid]
                total_duration += sp.get('duration', 0) or 0
                episodes.add(sp.get('content_id'))
                channel_id = sp.get('channel_id')
                if channel_id:
                    channels[channel_id] += 1

        # Get channel names for top channels
        top_channels = []
        if channels:
            sorted_channels = sorted(channels.items(), key=lambda x: x[1], reverse=True)
            # Fetch channel names from DB
            channel_ids = [ch_id for ch_id, _ in sorted_channels[:5]]
            with get_session() as session:
                result = session.execute(
                    text("SELECT id, display_name FROM channels WHERE id = ANY(:ids)"),
                    {'ids': channel_ids}
                ).fetchall()
                id_to_name = {r.id: r.display_name for r in result}

            for ch_id, count in sorted_channels[:5]:
                ch_name = id_to_name.get(ch_id, f"ch_{ch_id}")
                # Truncate long names
                if len(ch_name) > 20:
                    ch_name = ch_name[:17] + "..."
                top_channels.append((ch_name, count))

        return {
            'total_hours': total_duration / 3600,
            'n_episodes': len(episodes),
            'n_channels': len(channels),
            'top_channels': top_channels
        }

    def _store_identity_centroids(
        self,
        identity_id: int,
        primary_centroid: np.ndarray,
        variation_centroids: List[np.ndarray],
        source_method: str = 'dense_core'
    ):
        """Store primary centroid and variation centroids in identity_centroids table.

        Args:
            identity_id: Speaker identity ID
            primary_centroid: Main centroid (from dense core)
            variation_centroids: Diverse embeddings for robust matching (dynamic count based on diversity)
            source_method: Method used to generate centroids
        """
        if self.dry_run:
            return

        with get_session() as session:
            # Delete existing centroids for this identity (in case of update)
            session.execute(
                text("DELETE FROM identity_centroids WHERE identity_id = :identity_id"),
                {'identity_id': identity_id}
            )

            # Insert primary centroid
            session.execute(
                text("""
                    INSERT INTO identity_centroids (identity_id, centroid_type, centroid, source_method)
                    VALUES (:identity_id, 'primary', :centroid, :source_method)
                """),
                {
                    'identity_id': identity_id,
                    'centroid': primary_centroid.tolist(),
                    'source_method': source_method
                }
            )

            # Insert variation centroids (dynamic count)
            for i, variation in enumerate(variation_centroids):
                session.execute(
                    text("""
                        INSERT INTO identity_centroids (identity_id, centroid_type, centroid, sentinel_index, source_method)
                        VALUES (:identity_id, 'variation', :centroid, :sentinel_index, :source_method)
                    """),
                    {
                        'identity_id': identity_id,
                        'centroid': variation.tolist(),
                        'sentinel_index': i,
                        'source_method': source_method
                    }
                )

            session.commit()

    def _get_identity_with_centroids(self, name: str) -> Optional[Dict]:
        """Get existing identity with stored centroids.

        Args:
            name: Speaker name to look up

        Returns:
            Dict with identity_id, primary_name, primary_centroid, variation_centroids
            or None if no identity or no centroids exist
        """
        with get_session() as session:
            # Get identity with primary centroid
            result = session.execute(
                text("""
                    SELECT si.id as identity_id, si.primary_name,
                           ic_primary.centroid as primary_centroid
                    FROM speaker_identities si
                    JOIN identity_centroids ic_primary
                        ON si.id = ic_primary.identity_id
                        AND ic_primary.centroid_type = 'primary'
                    WHERE LOWER(si.primary_name) = LOWER(:name)
                      AND si.is_active = TRUE
                    ORDER BY si.id
                    LIMIT 1
                """),
                {'name': name}
            ).fetchone()

            if not result:
                return None

            identity_id = result.identity_id

            # Parse pgvector format: "[0.1,0.2,...]" -> numpy array
            def parse_pgvector(vec_str):
                if isinstance(vec_str, str):
                    # pgvector returns string like "[0.1,0.2,...]"
                    cleaned = vec_str.strip('[]')
                    return np.array([float(x) for x in cleaned.split(',')], dtype=np.float32)
                elif isinstance(vec_str, (list, tuple)):
                    return np.array(vec_str, dtype=np.float32)
                else:
                    return np.array(vec_str, dtype=np.float32)

            primary_centroid = parse_pgvector(result.primary_centroid)

            # Get variation centroids (dynamic count)
            variations_result = session.execute(
                text("""
                    SELECT centroid
                    FROM identity_centroids
                    WHERE identity_id = :identity_id
                      AND centroid_type = 'variation'
                    ORDER BY sentinel_index
                """),
                {'identity_id': identity_id}
            ).fetchall()

            variation_centroids = [parse_pgvector(row.centroid) for row in variations_result]

            return {
                'identity_id': identity_id,
                'primary_name': result.primary_name,
                'primary_centroid': primary_centroid,
                'variation_centroids': variation_centroids
            }

    def _validate_and_assign_incremental(
        self,
        name: str,
        speakers: List[Dict],
        identity_id: int,
        primary_centroid: np.ndarray,
        variation_centroids: List[np.ndarray]
    ) -> Dict:
        """Validate new speakers against stored centroids (incremental path).

        For existing identities with centroids, we validate new text-verified
        speakers directly against the stored centroids instead of building
        a new cluster. This is faster and uses the authoritative centroids
        from the first run.

        Args:
            name: Speaker name for logging
            speakers: List of speaker dicts with 'speaker_id' and 'embedding'
            identity_id: Existing identity ID
            primary_centroid: Stored primary centroid
            variation_centroids: Stored variation centroids (dynamic count)

        Returns:
            Dict with assigned count, rejected count, and rejection reasons
        """
        assigned = []
        rejected = []

        n_variations = len(variation_centroids)
        # Dynamic threshold: require ~40% of variations to pass, minimum 1
        # 1 variation → 1 required (100%)
        # 2 variations → 1 required (50%)
        # 3 variations → 2 required (67%)
        # 4 variations → 2 required (50%)
        # 5 variations → 2 required (40%)
        # 6 variations → 3 required (50%)
        import math
        min_variation_passes = max(1, math.ceil(n_variations * 0.4))

        for sp in speakers:
            sp_id = sp['speaker_id']

            # Skip if already claimed
            if sp_id in self.claimed_speakers:
                continue

            emb = sp['embedding']

            # Gate 1: Primary centroid similarity
            primary_sim = float(np.dot(emb, primary_centroid))
            if primary_sim < self.config.tau_anchor:  # 0.68
                rejected.append((sp_id, f"primary_sim={primary_sim:.3f}"))
                self.claimed_speakers.add(sp_id)  # Mark as processed
                continue

            # Gate 2: Variation check (proportional threshold)
            if n_variations > 0:
                variation_passes = sum(
                    1 for v in variation_centroids
                    if float(np.dot(emb, v)) >= self.config.tau_sent  # 0.65
                )
                if variation_passes < min_variation_passes:
                    rejected.append((sp_id, f"variation_passes={variation_passes}/{n_variations}"))
                    self.claimed_speakers.add(sp_id)  # Mark as processed
                    continue

            # Passed both gates - add to assigned
            assigned.append(sp_id)
            self.claimed_speakers.add(sp_id)

        # Batch assign to identity
        if assigned and not self.dry_run:
            self._assign_speakers_incremental(assigned, identity_id, name)

        self.stats['speakers_assigned'] += len(assigned)

        return {
            'assigned': len(assigned),
            'rejected': len(rejected),
            'n_variations': n_variations,
            'min_passes_required': min_variation_passes,
            'rejection_reasons': rejected[:10]  # Limit for logging
        }

    def _assign_speakers_incremental(
        self,
        speaker_ids: List[int],
        identity_id: int,
        identity_name: str
    ):
        """Batch assign speakers to existing identity (incremental path)."""
        timestamp = datetime.now(timezone.utc).isoformat()

        with get_session() as session:
            for speaker_id in speaker_ids:
                phase_entry = {
                    'status': 'assigned',
                    'timestamp': timestamp,
                    'method': 'incremental_centroid_validation',
                    'identity_id': identity_id,
                    'identity_name': identity_name
                }

                session.execute(
                    text("""
                        UPDATE speakers SET
                            speaker_identity_id = :identity_id,
                            identification_details = jsonb_set(
                                COALESCE(identification_details, '{}'),
                                '{phase3}',
                                CAST(:phase_entry AS jsonb)
                            )
                        WHERE id = :speaker_id
                    """),
                    {
                        'identity_id': identity_id,
                        'speaker_id': speaker_id,
                        'phase_entry': json.dumps(phase_entry)
                    }
                )

            session.commit()

    def _create_identity(
        self,
        name: str,
        role: str,
        centroid: np.ndarray,
        quality: float,
        sample_count: int,
        cluster: CanopyCluster,
        sentinel_centroids: List[np.ndarray] = None
    ) -> Tuple[Optional[int], bool]:
        """Create or update speaker identity. Returns (identity_id, is_new). Returns (-1, True) in dry_run mode."""
        if self.dry_run:
            # Track what would happen without DB writes
            self.stats['identities_created'] += 1  # Assume new in dry run
            return -1, True  # Sentinel for "would create"

        with get_session() as session:
            # Check for existing identity
            existing = session.execute(
                text("""
                    SELECT id, verification_metadata
                    FROM speaker_identities
                    WHERE LOWER(primary_name) = LOWER(:name)
                      AND is_active = TRUE
                    ORDER BY id
                    LIMIT 1
                """),
                {'name': name}
            ).fetchone()

            is_new_identity = False
            if existing:
                identity_id = existing.id
                self.stats['identities_updated'] += 1

                # Update metadata but DON'T replace centroids
                # Existing centroids are authoritative (from first run with full speaker set)
                session.execute(
                    text("""
                        UPDATE speaker_identities
                        SET updated_at = NOW()
                        WHERE id = :id
                    """),
                    {'id': identity_id}
                )
            else:
                # Create new identity
                is_new_identity = True
                metadata = {
                    'centroid': centroid.tolist(),
                    'centroid_quality': quality,
                    'centroid_sample_count': sample_count,
                    'centroid_updated_at': datetime.now(timezone.utc).isoformat(),
                    'centroid_source': 'anchor_canopy_phase3'
                }

                result = session.execute(
                    text("""
                        INSERT INTO speaker_identities (
                            primary_name,
                            role,
                            verification_status,
                            is_active,
                            centroid_source,
                            text_verified_count,
                            verification_metadata,
                            created_at,
                            updated_at
                        ) VALUES (
                            :name,
                            :role,
                            'text_verified',
                            TRUE,
                            'text_verified',
                            :sample_count,
                            CAST(:metadata AS jsonb),
                            NOW(),
                            NOW()
                        )
                        RETURNING id
                    """),
                    {
                        'name': name,
                        'role': role,
                        'sample_count': sample_count,
                        'metadata': json.dumps(metadata)
                    }
                )
                identity_id = result.fetchone()[0]
                self.stats['identities_created'] += 1

            # Check if identity has any centroids (may have been wiped)
            has_centroids = False
            if not is_new_identity:
                centroid_count = session.execute(
                    text("SELECT COUNT(*) FROM identity_centroids WHERE identity_id = :id"),
                    {'id': identity_id}
                ).fetchone()[0]
                has_centroids = centroid_count > 0

            session.commit()

            # Store centroids if:
            # 1. New identity, OR
            # 2. Existing identity with no centroids (was wiped for rebase)
            if sentinel_centroids and (is_new_identity or not has_centroids):
                self._store_identity_centroids(
                    identity_id=identity_id,
                    primary_centroid=centroid,
                    variation_centroids=sentinel_centroids,
                    source_method='dense_core'
                )
                if not is_new_identity:
                    logger.info(f"  Stored centroids for existing identity (was wiped)")

            return identity_id, is_new_identity

    def _create_unnamed_identity(
        self,
        centroid: np.ndarray,
        cluster: CanopyCluster,
        sentinel_centroids: List[np.ndarray] = None
    ) -> Optional[int]:
        """Create identity without a name for unnamed clusters. Returns -1 in dry_run mode."""
        if self.dry_run:
            self.stats['unnamed_clusters_created'] += 1
            return -1  # Sentinel for "would create"

        with get_session() as session:
            metadata = {
                'centroid': centroid.tolist(),
                'centroid_sample_count': len(cluster.members),
                'centroid_updated_at': datetime.now(timezone.utc).isoformat(),
                'centroid_source': 'unnamed_cluster_phase3'
            }

            result = session.execute(
                text("""
                    INSERT INTO speaker_identities (
                        primary_name,
                        role,
                        verification_status,
                        is_active,
                        centroid_source,
                        text_verified_count,
                        verification_metadata,
                        created_at,
                        updated_at
                    ) VALUES (
                        NULL,
                        'unknown',
                        'unnamed_cluster',
                        TRUE,
                        'unnamed_cluster',
                        0,
                        CAST(:metadata AS jsonb),
                        NOW(),
                        NOW()
                    )
                    RETURNING id
                """),
                {'metadata': json.dumps(metadata)}
            )
            identity_id = result.fetchone()[0]
            session.commit()

            # Store centroids in separate table for indexable lookup
            if sentinel_centroids:
                self._store_identity_centroids(
                    identity_id=identity_id,
                    primary_centroid=centroid,
                    sentinel_centroids=sentinel_centroids,
                    source_method='unnamed_cluster'
                )

            self.stats['unnamed_clusters_created'] += 1
            return identity_id

    def _assign_cluster_members(
        self,
        cluster: CanopyCluster,
        identity_id: int,
        identity_name: Optional[str]
    ):
        """Batch assign all cluster members to identity."""
        speaker_ids = list(cluster.members)

        # Track assignments (even in dry_run)
        self.stats['speakers_assigned'] += len(speaker_ids)

        if self.dry_run:
            return

        timestamp = datetime.now(timezone.utc).isoformat()

        # Build phase3 entry
        for speaker_id in speaker_ids:
            is_text_verified = cluster.member_names.get(speaker_id) is not None
            gate_result = cluster.gate_results.get(speaker_id, "unknown")

            phase_entry = {
                'status': 'assigned',
                'timestamp': timestamp,
                'method': 'anchor_canopy_phase3',
                'identity_id': identity_id,
                'identity_name': identity_name,
                'cluster_size': len(cluster.members),
                'is_text_verified': is_text_verified,
                'gate_result': gate_result
            }

            # Confidence: higher for text-verified, slightly lower for expanded
            confidence = 0.95 if is_text_verified else 0.85

            with get_session() as session:
                session.execute(
                    text(f"""
                        UPDATE speakers
                        SET speaker_identity_id = :identity_id,
                            assignment_confidence = :confidence,
                            assignment_phase = '{PHASE_KEY}',
                            identification_details = jsonb_set(
                                COALESCE(identification_details, '{{}}'::jsonb),
                                ARRAY['{PHASE_KEY}'],
                                CAST(:phase_entry AS jsonb)
                            ),
                            updated_at = NOW()
                        WHERE id = :speaker_id
                    """),
                    {
                        'identity_id': identity_id,
                        'confidence': confidence,
                        'speaker_id': speaker_id,
                        'phase_entry': json.dumps(phase_entry)
                    }
                )
                session.commit()

    def _find_unnamed_clusters(
        self,
        all_speakers: Dict[int, Dict],
        speaker_ids: List[int],
        embeddings: np.ndarray,
        knn_indices: np.ndarray,
        knn_sims: np.ndarray,
        id_to_idx: Dict[int, int]
    ) -> List[CanopyCluster]:
        """Find clusters among unclaimed speakers (no text evidence)."""
        unclaimed = [sid for sid in all_speakers.keys() if sid not in self.claimed_speakers]

        if not unclaimed:
            return []

        logger.info(f"Finding unnamed clusters among {len(unclaimed)} unclaimed speakers...")

        # Sort unclaimed by quality and duration for anchor selection
        def anchor_score(sid):
            s = all_speakers[sid]
            return s['quality'] * 100 + min(s.get('duration', 0) / 60, 10)

        unclaimed.sort(key=anchor_score, reverse=True)

        unnamed_clusters = []

        for anchor_id in tqdm(unclaimed[:1000], desc="Processing potential unnamed anchors"):
            if anchor_id in self.claimed_speakers:
                continue

            anchor_data = all_speakers[anchor_id]
            anchor_idx = id_to_idx[anchor_id]

            # Create anchor without name
            cluster = CanopyCluster(
                anchor_id=anchor_id,
                anchor_idx=anchor_idx,
                anchor_embedding=anchor_data['embedding'],
                anchor_name=None,
                config=self.config
            )

            self.claimed_speakers.add(anchor_id)

            # Get anchor's neighbors
            anchor_knn = knn_indices[anchor_idx]
            anchor_knn_sims = knn_sims[anchor_idx]

            # Simple frontier-based expansion (relaxed for unnamed)
            for nbr_idx, sim in zip(anchor_knn[:50], anchor_knn_sims[:50]):
                if sim < 0.72:  # Higher threshold for unnamed
                    break

                nbr_id = speaker_ids[nbr_idx]
                if nbr_id in self.claimed_speakers:
                    continue

                nbr_data = all_speakers[nbr_id]

                # Skip if this speaker has a verified name (should have been claimed already)
                if nbr_data.get('phase2_name'):
                    continue

                nbr_idx_local = id_to_idx[nbr_id]
                nbr_knn = knn_indices[nbr_idx_local]

                # Simplified validation for unnamed clusters (use anchor_idx not anchor_id)
                # Check mutual k-NN (within first 50 neighbors)
                if anchor_idx in nbr_knn[:50]:
                    cluster.add_member(nbr_id, nbr_data['embedding'], None, f"sim={sim:.3f}")
                    self.claimed_speakers.add(nbr_id)

                if len(cluster.members) >= 50:  # Cap unnamed cluster size
                    break

            # Only keep clusters with minimum size
            if len(cluster.members) >= self.config.min_unnamed_cluster_size:
                cluster.update_prototype(force=True)
                unnamed_clusters.append(cluster)

        return unnamed_clusters

    async def run(self, project: str = None) -> Dict:
        """
        Run anchor-canopy clustering.

        Args:
            project: Project to filter to

        Returns:
            Stats dict
        """
        # Load date range from project config
        if project and not self.start_date and not self.end_date:
            start_date, end_date = get_project_date_range(project)
            self.start_date = start_date
            self.end_date = end_date

        logger.info("=" * 80)
        logger.info("ANCHOR-CANOPY CLUSTERING (Phase 3 - New Pipeline)")
        logger.info("=" * 80)
        logger.info(f"Mode: {'DRY RUN' if self.dry_run else 'APPLY'}")
        logger.info(f"Force: {'YES (reprocessing all)' if self.force else 'NO (skipping already assigned)'}")
        logger.info(f"Project: {project or 'ALL'}")
        if self.start_date or self.end_date:
            logger.info(f"Date range: {self.start_date or 'any'} to {self.end_date or 'any'}")
        logger.info(f"Config: tau_anchor={self.config.tau_anchor}, beta={self.config.beta}, "
                   f"tau_sent={self.config.tau_sent}")
        logger.info(f"Confidence: min_verified={self.config.min_verified_for_high_confidence}, "
                   f"expand_single_verified={self.config.expand_single_verified}")
        logger.info("-" * 80)

        # Step 1: Load Phase 1 alias mappings
        logger.info("Loading Phase 1 alias mappings...")
        alias_mapping = self._load_alias_mappings()
        logger.info(f"Loaded {len(alias_mapping)} alias → canonical name mappings")

        # Step 2: Load all speakers
        logger.info("Loading all speakers...")
        all_speakers = self._load_all_speakers(project)
        logger.info(f"Loaded {len(all_speakers)} speakers with embeddings")

        if not all_speakers:
            logger.warning("No speakers found!")
            return self.stats

        # Step 3: Load text-verified speakers (with alias normalization)
        logger.info("Loading text-verified speakers (Phase 2 certain)...")
        speakers_by_name = self._load_text_verified_speakers(project, alias_mapping)
        self.stats['unique_names_found'] = len(speakers_by_name)

        total_verified = sum(len(s) for s in speakers_by_name.values())
        logger.info(f"Found {len(speakers_by_name)} unique names, {total_verified} text-verified speakers")

        # Step 4: Build FAISS index
        logger.info("Building FAISS index...")
        index, speaker_ids, embeddings = self._build_faiss_index(all_speakers)
        id_to_idx = {sid: idx for idx, sid in enumerate(speaker_ids)}
        logger.info(f"Built index with {len(speaker_ids)} speakers (dim={embeddings.shape[1]})")

        # Step 5: Pre-compute k-NN
        logger.info(f"Pre-computing k-NN (k={self.config.k0})...")
        knn_indices, knn_sims = self._precompute_knn(index, embeddings, k=self.config.k0)
        logger.info("k-NN computation complete")

        logger.info("-" * 80)
        logger.info("PROCESSING TEXT-VERIFIED NAME GROUPS")
        logger.info("-" * 80)

        # Step 6: Process each name group
        # Sort names to process full names first (more specific before less specific)
        # This ensures "Ezra Levant" claims speakers before "Ezra"
        def name_priority(name):
            """Score names: full names > single names > short/numeric names."""
            # Skip numeric-only names
            if name.replace(' ', '').replace('-', '').isdigit():
                return -1  # Will be filtered out

            parts = name.split()
            n_parts = len(parts)
            name_len = len(name)

            # Full names (2+ parts) get priority
            if n_parts >= 2:
                return 1000 + name_len  # Full name + length bonus
            elif n_parts == 1 and name_len >= 4:
                return 500 + name_len   # Single name but substantial
            else:
                return name_len         # Short names last

        # Filter and sort names
        valid_names = [(name, speakers) for name, speakers in speakers_by_name.items()
                       if name_priority(name) > 0]
        skipped_names = len(speakers_by_name) - len(valid_names)
        if skipped_names > 0:
            logger.info(f"Skipped {skipped_names} invalid names (numeric-only, etc.)")

        # Apply name filter if specified (for testing)
        if self.name_filter:
            filter_lower = self.name_filter.lower()
            valid_names = [(name, speakers) for name, speakers in valid_names
                          if filter_lower in name.lower()]
            logger.info(f"Filtered to {len(valid_names)} names matching '{self.name_filter}'")

        # Sort by priority descending (full names first, then by length)
        sorted_names = sorted(valid_names, key=lambda x: name_priority(x[0]), reverse=True)

        logger.info(f"Processing {len(sorted_names)} name groups (full names first)")

        anchor_num = 0
        incremental_count = 0
        baseline_count = 0

        for name, verified_speakers in sorted_names:
            self.stats['names_processed'] += 1

            try:
                # Check if identity with centroids already exists (incremental path)
                existing = self._get_identity_with_centroids(name)

                if existing:
                    # INCREMENTAL PATH: Validate new speakers against stored centroids
                    # This preserves the authoritative centroids from the first run
                    incremental_count += 1
                    n_vars = len(existing['variation_centroids'])
                    logger.info(f"\n{'='*60}")
                    logger.info(f"[INCREMENTAL] \"{name}\" - validating {len(verified_speakers)} speakers")
                    logger.info(f"  Existing identity: ID {existing['identity_id']}, {n_vars} variations stored")

                    result = self._validate_and_assign_incremental(
                        name=name,
                        speakers=verified_speakers,
                        identity_id=existing['identity_id'],
                        primary_centroid=existing['primary_centroid'],
                        variation_centroids=existing['variation_centroids']
                    )

                    logger.info(f"  ✓ Assigned: {result['assigned']}, Rejected: {result['rejected']} "
                               f"(require {result['min_passes_required']}/{result['n_variations']} variations)")
                    if result['rejection_reasons']:
                        reasons_sample = result['rejection_reasons'][:3]
                        for sp_id, reason in reasons_sample:
                            logger.info(f"    Rejected {sp_id}: {reason}")

                    continue  # Skip baseline processing for this name

                # BASELINE PATH: Full cluster building (no existing centroids)
                baseline_count += 1

                # Add identified_name to speakers for anchor selection
                for sp in verified_speakers:
                    sp['identified_name'] = name

                # Detect name collisions
                person_clusters = self._detect_name_collisions(verified_speakers)

                if len(person_clusters) > 1:
                    logger.info(f"\n⚠ NAME COLLISION: '{name}' has {len(person_clusters)} distinct people")
                    self.stats['name_collisions_detected'] += 1

                # Process each distinct person
                for i, person_speakers in enumerate(person_clusters):
                    final_name = name if i == 0 else f"{name} ({i + 1})"
                    anchor_num += 1
                    n_verified = len(person_speakers)

                    # Determine confidence level
                    # HIGH confidence: 2+ verified speakers agree on the name
                    # (multiple independent LLM calls agreeing is strong evidence)
                    is_high_confidence = n_verified >= self.config.min_verified_for_high_confidence

                    # Select best anchor
                    anchor = self._select_best_anchor(person_speakers)
                    anchor_id = anchor['speaker_id']

                    # Log anchor info
                    confidence_label = "HIGH" if is_high_confidence else "LOW"
                    logger.info(f"\n{'='*60}")
                    logger.info(f"Anchor #{anchor_num}: \"{final_name}\" (speaker_id: {anchor_id})")
                    logger.info(f"  Confidence: {confidence_label} ({n_verified} verified speakers)")
                    logger.info(f"  Evidence: {anchor.get('evidence_type', 'unknown')}, "
                               f"Quality: {anchor['quality']:.2f}")

                    # For LOW confidence (single verified), skip expansion if configured
                    if not is_high_confidence and not self.config.expand_single_verified:
                        logger.info(f"  ⚠ Skipping expansion (single verified, unreliable)")
                        # Still mark as claimed and create minimal cluster
                        self.claimed_speakers.add(anchor_id)

                        # Create a cluster with just the verified speakers (no expansion)
                        cluster = CanopyCluster(
                            anchor_id=anchor_id,
                            anchor_idx=id_to_idx[anchor_id],
                            anchor_embedding=anchor['embedding'],
                            anchor_name=final_name,
                            config=self.config,
                            is_high_confidence=False,
                            n_verified=n_verified
                        )
                        # Add any other verified speakers from same group
                        for sp in person_speakers:
                            if sp['speaker_id'] != anchor_id:
                                self.claimed_speakers.add(sp['speaker_id'])
                                cluster.add_member(
                                    sp['speaker_id'],
                                    sp['embedding'],
                                    final_name,
                                    "text_verified"
                                )
                        gate_stats = {
                            'initial_frontier': 0,
                            'highest_neighbor_sim': 0.0,
                            'candidates_evaluated': 0,
                            'passed': 0,
                            'rejected_by_gate': {'proto_sim': 0, 'density': 0, 'sentinel': 0, 'name_conflict': 0}
                        }
                    else:
                        # Grow cluster with gate stats (HIGH confidence or expand_single_verified=True)
                        cluster, gate_stats = self._grow_cluster_multigate(
                            anchor=anchor,
                            all_speakers=all_speakers,
                            speaker_ids=speaker_ids,
                            embeddings=embeddings,
                            knn_indices=knn_indices,
                            knn_sims=knn_sims,
                            id_to_idx=id_to_idx,
                            is_high_confidence=is_high_confidence,
                            n_verified=n_verified,
                            verified_speakers=person_speakers if is_high_confidence else None
                        )

                    self.clusters.append(cluster)
                    self.stats['clusters_created'] += 1
                    if is_high_confidence:
                        self.stats['high_confidence_clusters'] += 1
                    else:
                        self.stats['low_confidence_clusters'] += 1

                    # Log dense core detection for verified speakers
                    dense_core = gate_stats.get('dense_core')
                    if dense_core and dense_core.get('excluded', 0) > 0:
                        readmitted = dense_core.get('readmitted', 0)
                        truly_excluded = dense_core.get('truly_excluded', 0)
                        if readmitted > 0:
                            logger.info(f"  Dense core: {dense_core['core_size']}/{dense_core['checked']} verified, "
                                       f"then {readmitted} readmitted, {truly_excluded} excluded "
                                       f"(min_sim={dense_core['min_sim']:.2f})")
                        else:
                            logger.info(f"  Dense core: {dense_core['core_size']}/{dense_core['checked']} verified "
                                       f"(excluded {dense_core['excluded']} outliers, min_sim={dense_core['min_sim']:.2f})")
                    elif dense_core:
                        logger.info(f"  Dense core: {dense_core['core_size']}/{dense_core['checked']} verified "
                                   f"(all in core, min_sim={dense_core['min_sim']:.2f})")

                    # Log expansion statistics
                    total_rejected = sum(gate_stats['rejected_by_gate'].values())
                    total_evaluated = gate_stats['candidates_evaluated']

                    if total_evaluated > 0:
                        logger.info(f"  Expansion: {gate_stats['passed']} added, {total_evaluated} evaluated")
                        if total_rejected > 0:
                            logger.info(f"    Rejected: proto_sim={gate_stats['rejected_by_gate']['proto_sim']}, "
                                       f"density={gate_stats['rejected_by_gate']['density']}, "
                                       f"sentinel={gate_stats['rejected_by_gate']['sentinel']}, "
                                       f"name_conflict={gate_stats['rejected_by_gate']['name_conflict']}")
                    elif gate_stats['initial_frontier'] == 0:
                        logger.info(f"  Expansion: no candidates (highest neighbor sim: {gate_stats['highest_neighbor_sim']:.3f})")

                    # Build centroid from dense core + sentinel centroids
                    centroid, quality, sample_count, sentinel_centroids = self._build_quality_weighted_centroid(cluster)

                    # Infer role
                    role = self._infer_role(cluster, all_speakers)

                    # Create identity
                    identity_id, is_new_identity = self._create_identity(
                        name=final_name,
                        role=role,
                        centroid=centroid,
                        quality=quality,
                        sample_count=sample_count,
                        cluster=cluster,
                        sentinel_centroids=sentinel_centroids
                    )

                    # Assign members (identity_id can be -1 for dry_run)
                    if identity_id is not None:
                        self._assign_cluster_members(cluster, identity_id, final_name)

                    # Log result with cluster stats
                    cluster_stats = self._compute_cluster_stats(cluster, all_speakers)
                    logger.info(f"  ✓ Cluster: {len(cluster.members)} speakers, "
                               f"{cluster_stats['total_hours']:.1f}h spoken, "
                               f"{cluster_stats['n_episodes']} episodes, "
                               f"{cluster_stats['n_channels']} channels")
                    if cluster_stats['top_channels']:
                        top_ch = ", ".join(f"{ch}({cnt})" for ch, cnt in cluster_stats['top_channels'][:3])
                        logger.info(f"    Top channels: {top_ch}")
                    if identity_id and identity_id != -1:
                        action = "created" if is_new_identity else "updated"
                        logger.info(f"  ✓ Identity {action}: \"{final_name}\" (ID: {identity_id}, role: {role})")
                    elif identity_id == -1:
                        logger.info(f"  ✓ [DRY RUN] Would create identity: \"{final_name}\" (role: {role})")

            except Exception as e:
                logger.error(f"Error processing name '{name}': {e}")
                import traceback
                traceback.print_exc()
                self.stats['errors'].append(f"{name}: {str(e)}")

        # Log incremental vs baseline processing counts
        logger.info("-" * 80)
        logger.info(f"Processing mode breakdown:")
        logger.info(f"  - Incremental (existing centroids): {incremental_count} names")
        logger.info(f"  - Baseline (new clusters): {baseline_count} names")
        logger.info("-" * 80)

        # Step 7: Unnamed clusters (disabled for Phase 3 - focus on named speakers)
        # Unnamed clustering can be done in a future phase with different validation
        logger.info("SKIPPING UNNAMED CLUSTERS (Phase 3 focuses on text-verified names)")
        logger.info("-" * 80)

        # Final stats
        self.stats['speakers_unclaimed'] = len(all_speakers) - len(self.claimed_speakers)

        self._print_summary()
        return self.stats

    def wipe_centroids(self, project: str = None) -> Dict:
        """Wipe centroids and phase3 assignments for a fresh rebuild.

        This is a "rebase" operation that:
        1. Deletes centroids from identity_centroids
        2. Clears phase3 assignments from speakers (unlinks speaker_identity_id)
        3. Preserves speaker_identities metadata (gender, location, etc.)

        Args:
            project: Project to wipe (None = all, or use name_filter for specific speaker)

        Returns:
            Dict with counts of wiped data
        """
        if self.dry_run:
            logger.info("[DRY RUN] Would wipe centroids - no changes made")
            return {'dry_run': True}

        stats = {
            'centroids_deleted': 0,
            'speakers_unlinked': 0,
            'identities_affected': 0
        }

        with get_session() as session:
            # Build filter for which identities to wipe
            if self.name_filter:
                # Wipe specific speaker by name
                identity_filter = """
                    si.id IN (
                        SELECT id FROM speaker_identities
                        WHERE LOWER(primary_name) LIKE LOWER(:name_filter)
                    )
                """
                params = {'name_filter': f'%{self.name_filter}%'}
                logger.info(f"Wiping centroids for speakers matching '{self.name_filter}'")
            elif project:
                # Wipe all identities that have speakers in this project
                identity_filter = """
                    si.id IN (
                        SELECT DISTINCT s.speaker_identity_id
                        FROM speakers s
                        JOIN content c ON s.content_id = c.content_id
                        WHERE :project = ANY(c.projects)
                          AND s.speaker_identity_id IS NOT NULL
                    )
                """
                params = {'project': project}
                logger.info(f"Wiping centroids for project '{project}'")
            else:
                # Wipe ALL centroids
                identity_filter = "1=1"
                params = {}
                logger.info("Wiping ALL centroids")

            # Step 1: Count and delete centroids
            count_result = session.execute(
                text(f"""
                    SELECT COUNT(*) FROM identity_centroids ic
                    JOIN speaker_identities si ON ic.identity_id = si.id
                    WHERE {identity_filter}
                """),
                params
            ).fetchone()
            stats['centroids_deleted'] = count_result[0]

            session.execute(
                text(f"""
                    DELETE FROM identity_centroids
                    WHERE identity_id IN (
                        SELECT si.id FROM speaker_identities si
                        WHERE {identity_filter}
                    )
                """),
                params
            )

            # Step 2: Clear phase3 assignments from speakers
            # Build speaker filter
            if self.name_filter:
                speaker_filter = """
                    speaker_identity_id IN (
                        SELECT id FROM speaker_identities
                        WHERE LOWER(primary_name) LIKE LOWER(:name_filter)
                    )
                """
            elif project:
                speaker_filter = """
                    content_id IN (
                        SELECT content_id FROM content
                        WHERE :project = ANY(projects)
                    )
                    AND identification_details ? 'phase3'
                """
            else:
                speaker_filter = "identification_details ? 'phase3'"

            # Count speakers to unlink
            count_result = session.execute(
                text(f"SELECT COUNT(*) FROM speakers WHERE {speaker_filter}"),
                params
            ).fetchone()
            stats['speakers_unlinked'] = count_result[0]

            # Unlink speakers and remove phase3 from identification_details
            session.execute(
                text(f"""
                    UPDATE speakers SET
                        speaker_identity_id = NULL,
                        assignment_phase = NULL,
                        assignment_confidence = NULL,
                        identification_details = identification_details - 'phase3',
                        updated_at = NOW()
                    WHERE {speaker_filter}
                """),
                params
            )

            # Count affected identities
            if self.name_filter:
                count_result = session.execute(
                    text("SELECT COUNT(*) FROM speaker_identities WHERE LOWER(primary_name) LIKE LOWER(:name_filter)"),
                    params
                ).fetchone()
            elif project:
                count_result = session.execute(
                    text(f"""
                        SELECT COUNT(DISTINCT si.id)
                        FROM speaker_identities si
                        WHERE {identity_filter}
                    """),
                    params
                ).fetchone()
            else:
                count_result = session.execute(
                    text("SELECT COUNT(*) FROM speaker_identities")
                ).fetchone()
            stats['identities_affected'] = count_result[0]

            session.commit()

        logger.info(f"Wiped: {stats['centroids_deleted']} centroids, "
                   f"{stats['speakers_unlinked']} speaker assignments, "
                   f"{stats['identities_affected']} identities affected")
        logger.info("Speaker identity metadata preserved (gender, location, etc.)")

        return stats

    def _print_summary(self):
        """Print summary statistics."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Unique names found: {self.stats['unique_names_found']}")
        logger.info(f"Names processed: {self.stats['names_processed']}")
        logger.info(f"Name collisions detected: {self.stats['name_collisions_detected']}")
        logger.info(f"Clusters created: {self.stats['clusters_created']}")
        logger.info(f"  - High confidence (2+ verified): {self.stats['high_confidence_clusters']}")
        logger.info(f"  - Low confidence (1 verified): {self.stats['low_confidence_clusters']}")
        logger.info(f"Identities created: {self.stats['identities_created']}")
        logger.info(f"Identities updated: {self.stats['identities_updated']}")
        logger.info(f"Speakers assigned: {self.stats['speakers_assigned']}")
        logger.info(f"Unnamed clusters: {self.stats['unnamed_clusters_created']}")
        logger.info(f"Speakers in unnamed: {self.stats['speakers_in_unnamed']}")
        logger.info(f"Speakers unclaimed: {self.stats['speakers_unclaimed']}")
        if self.stats['errors']:
            logger.warning(f"Errors: {len(self.stats['errors'])}")
        logger.info("=" * 80)


async def main():
    parser = argparse.ArgumentParser(
        description='Phase 3: Anchor-Canopy Clustering with Text-Verified Anchors',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (skips already-assigned speakers)
  python -m src.speaker_identification.strategies.anchor_verified_clustering --project CPRMV

  # Apply to project
  python -m src.speaker_identification.strategies.anchor_verified_clustering \\
      --project CPRMV --apply

  # Wipe and rebuild specific speaker (rebase)
  python -m src.speaker_identification.strategies.anchor_verified_clustering \\
      --project CPRMV --name "Ezra Levant" --wipe --apply

  # Wipe and rebuild entire project
  python -m src.speaker_identification.strategies.anchor_verified_clustering \\
      --project CPRMV --wipe --apply
"""
    )

    parser.add_argument('--project', type=str, help='Filter to project')
    parser.add_argument('--name', type=str, help='Filter to specific speaker name (for testing)')
    parser.add_argument('--apply', action='store_true', help='Apply changes (default: dry run)')
    parser.add_argument('--force', action='store_true',
                       help='Reprocess speakers already assigned by Phase 3 (default: skip them)')
    parser.add_argument('--wipe', action='store_true',
                       help='Wipe existing centroids before processing (rebase). '
                            'Clears centroids and phase3 assignments but preserves identity metadata.')
    parser.add_argument('--tau-anchor', type=float, default=0.68,
                       help='Anchor similarity threshold (default: 0.68)')
    parser.add_argument('--beta', type=float, default=0.82,
                       help='Relative density factor (default: 0.82)')
    parser.add_argument('--min-verified', type=int, default=2,
                       help='Minimum verified speakers for high-confidence (default: 2)')
    parser.add_argument('--expand-single', action='store_true',
                       help='Also expand single-verified clusters (default: skip them)')

    args = parser.parse_args()

    config = Phase3ClusterConfig(
        tau_anchor=args.tau_anchor,
        beta=args.beta,
        min_verified_for_high_confidence=args.min_verified,
        expand_single_verified=args.expand_single
    )

    strategy = AnchorVerifiedClusteringStrategy(
        config=config,
        dry_run=not args.apply,
        force=args.force or args.wipe,  # Wipe implies force (reprocess all)
        name_filter=args.name
    )

    try:
        # Wipe centroids first if requested (rebase)
        if args.wipe:
            strategy.wipe_centroids(project=args.project)

        await strategy.run(project=args.project)
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
