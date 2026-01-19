#!/usr/bin/env python3
"""
Phase 3 (NEW): Label Propagation with Conflict Detection
=========================================================

Propagates speaker labels from Phase 2 anchors using k-NN weighted voting,
WITH conflict detection for subsequent LLM resolution.

This phase runs FIRST (before LLM name verification) to identify only
the conflicts that actually matter - where k-NN would assign conflicting
names to similar voice clusters.

Pipeline Position:
    Phase 2: Text evidence collection → verified anchors
    Phase 3: THIS - Label propagation with conflict detection
    Phase 4: LLM conflict resolution (only verifies actual conflicts)

Key Innovation:
    Old approach: LLM-verify ALL name pairs with centroid similarity ≥ 0.70
                  (~2,746 pairs, most don't actually conflict during assignment)
    New approach: Run k-NN first, only LLM-verify pairs that ACTUALLY conflict
                  (typically 10-50 pairs where assignment is contested)

Conflict Types Detected:
    1. CONTESTED_ASSIGNMENT: Top-2 labels have similar vote weights for a speaker
    2. CLUSTER_SPLIT: Same voice cluster has speakers assigned different names
    3. NAME_CENTROID_CONFLICT: Name centroids very similar, assigned to same speakers

Output:
    - Speakers assigned (non-conflicting) → identification_details['phase3_propagation']
    - Conflicts → config/propagation_conflicts.json for Phase 4 LLM resolution

Usage:
    # Dry run
    python -m src.speaker_identification.strategies.label_propagation_with_conflicts \\
        --project CPRMV

    # Apply
    python -m src.speaker_identification.strategies.label_propagation_with_conflicts \\
        --project CPRMV --apply
"""

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import faiss
import numpy as np
from tqdm import tqdm
import yaml

from src.utils.paths import get_project_root
project_root = str(get_project_root())
sys.path.append(project_root)

from sqlalchemy import text

from src.database.session import get_session
from src.utils.logger import setup_worker_logger
from src.utils.config import get_project_date_range

# Paths
NAME_ALIASES_PATH = Path(project_root) / 'config' / 'name_aliases.yaml'
CONFLICTS_PATH = Path(project_root) / 'config' / 'propagation_conflicts.json'

logger = setup_worker_logger('speaker_identification.label_propagation_conflicts')

# Console logging
import logging
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

PHASE_KEY = "phase3_propagation"


@dataclass
class ConflictRecord:
    """Record of a name pair conflict detected during propagation."""
    name_a: str
    name_b: str
    centroid_similarity: float
    conflict_type: str  # "contested_assignment" | "cluster_split" | "centroid_conflict"
    affected_speakers: int
    affected_hours: float
    sample_speaker_ids: List[int] = field(default_factory=list)
    evidence: Dict = field(default_factory=dict)


@dataclass
class LabelPropagationWithConflictsConfig:
    """Configuration for label propagation with conflict detection."""
    # k-NN parameters
    k_neighbors: int = 75
    k_build: int = 128

    # Similarity weighting
    min_similarity: float = 0.55
    similarity_power: float = 2.0

    # Confidence thresholds
    min_confidence_to_assign: float = 0.25
    high_confidence_threshold: float = 0.60

    # Label requirements
    min_labeled_neighbors: int = 1

    # Embedding quality
    min_embedding_quality: float = 0.50

    # Conflict detection thresholds
    contested_vote_ratio: float = 0.70  # If 2nd-best has >= 70% of best's votes, it's contested
    cluster_split_threshold: float = 0.75  # Similarity threshold for "same cluster"
    name_centroid_conflict_threshold: float = 0.70  # Name centroid similarity for conflict detection

    # Metadata constraints (from original Phase 4)
    use_metadata_constraint: bool = True
    high_duration_pct: float = 0.10
    metadata_similarity_boost: float = 0.10


class LabelPropagationWithConflictsStrategy:
    """
    Phase 3: Label Propagation with Conflict Detection.

    Runs k-NN label propagation to assign names to speakers, while
    detecting actual conflicts that need LLM resolution.
    """

    def __init__(
        self,
        config: LabelPropagationWithConflictsConfig = None,
        dry_run: bool = True,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_anchors: Optional[int] = None,
        name_filter: Optional[str] = None
    ):
        self.config = config or LabelPropagationWithConflictsConfig()
        self.dry_run = dry_run
        self.start_date = start_date
        self.end_date = end_date
        self.max_anchors = max_anchors
        self.name_filter = name_filter

        self.stats = {
            'total_speakers': 0,
            'labeled_speakers': 0,
            'unlabeled_speakers': 0,
            'unique_names': 0,
            'speakers_assigned': 0,
            'high_confidence_assignments': 0,
            'low_confidence_assignments': 0,
            'speakers_unassigned': 0,
            'speakers_with_conflicts': 0,
            'conflicts_detected': 0,
            'contested_assignments': 0,
            'cluster_splits': 0,
            # Speaking time
            'total_speaking_hours': 0.0,
            'assigned_speaking_hours': 0.0,
            'conflict_speaking_hours': 0.0,
            'unassigned_speaking_hours': 0.0,
            'identities_created': 0,
            'identities_updated': 0,
            'errors': []
        }

        self.assignments: Dict[int, Dict] = {}
        self.conflicts: List[ConflictRecord] = []

    def _load_all_speakers(self, project: str) -> Dict[int, Dict]:
        """Load all speakers with embeddings for the project."""
        with get_session() as session:
            filters = ["s.embedding IS NOT NULL"]
            params = {'min_quality': self.config.min_embedding_quality}

            if project:
                filters.append(":project = ANY(c.projects)")
                params['project'] = project

            if self.start_date:
                filters.append("c.publish_date >= :start_date")
                params['start_date'] = self.start_date
            if self.end_date:
                filters.append("c.publish_date < :end_date")
                params['end_date'] = self.end_date

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
                    s.identification_details->'phase2'->>'evidence_type' as evidence_type,
                    c.channel_id,
                    c.duration as episode_duration,
                    c.hosts as episode_hosts,
                    c.guests as episode_guests
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

                # Parse episode metadata
                episode_names = set()
                if row.episode_hosts:
                    hosts_data = row.episode_hosts if isinstance(row.episode_hosts, list) else []
                    for h in hosts_data:
                        if isinstance(h, dict) and h.get('name'):
                            episode_names.add(h['name'].lower())
                if row.episode_guests:
                    guests_data = row.episode_guests if isinstance(row.episode_guests, list) else []
                    for g in guests_data:
                        if isinstance(g, dict) and g.get('name'):
                            episode_names.add(g['name'].lower())

                duration_pct = (row.duration / row.episode_duration) if row.episode_duration and row.episode_duration > 0 else 0

                speakers[row.speaker_id] = {
                    'speaker_id': row.speaker_id,
                    'content_id': row.content_id,
                    'embedding': embedding,
                    'quality': row.quality,
                    'duration': row.duration,
                    'episode_duration': row.episode_duration,
                    'duration_pct': duration_pct,
                    'phase2_status': row.phase2_status,
                    'phase2_name': row.phase2_name,
                    'evidence_type': row.evidence_type,
                    'channel_id': row.channel_id,
                    'speaker_identity_id': row.speaker_identity_id,
                    'episode_names': episode_names
                }

            return speakers

    def _load_name_aliases(self) -> Tuple[Dict[str, str], Set[str], Set[frozenset]]:
        """Load name aliases from config file."""
        alias_to_canonical = {}
        unresolved_handles = set()
        do_not_merge = set()

        if not NAME_ALIASES_PATH.exists():
            logger.info(f"  No alias file found at {NAME_ALIASES_PATH}")
            return alias_to_canonical, unresolved_handles, do_not_merge

        try:
            with open(NAME_ALIASES_PATH, 'r') as f:
                config = yaml.safe_load(f)

            if not config:
                return alias_to_canonical, unresolved_handles, do_not_merge

            # Load aliases
            aliases = config.get('aliases', {})
            for canonical, alias_list in aliases.items():
                if alias_list:
                    for alias in alias_list:
                        alias_to_canonical[alias.lower()] = canonical

            # Load unresolved handles
            handles = config.get('unresolved_handles', [])
            if handles:
                unresolved_handles = set(h.lower() for h in handles)

            # Load do-not-merge pairs
            dnm_list = config.get('do_not_merge', [])
            if dnm_list:
                for pair in dnm_list:
                    if isinstance(pair, list) and len(pair) >= 2:
                        do_not_merge.add(frozenset(n.lower() for n in pair))

            logger.info(f"  Loaded {len(alias_to_canonical)} aliases, "
                       f"{len(unresolved_handles)} unresolved handles, "
                       f"{len(do_not_merge)} do-not-merge pairs")

        except Exception as e:
            logger.warning(f"  Error loading aliases: {e}")

        return alias_to_canonical, unresolved_handles, do_not_merge

    def _apply_static_aliases(
        self,
        speakers_by_name: Dict[str, List[Dict]],
        alias_to_canonical: Dict[str, str]
    ) -> Tuple[Dict[str, List[Dict]], Dict[str, str]]:
        """Apply static aliases from config file."""
        name_mappings = {}
        merged_speakers_by_name = {}

        canonical_groups: Dict[str, List[str]] = defaultdict(list)

        for name in speakers_by_name.keys():
            name_lower = name.lower()

            if name_lower in alias_to_canonical:
                canonical = alias_to_canonical[name_lower]
                canonical_groups[canonical].append(name)
                if name != canonical:
                    name_mappings[name] = canonical
            else:
                canonical_groups[name].append(name)

        for canonical, name_variants in canonical_groups.items():
            merged_speakers = []
            for variant in name_variants:
                if variant in speakers_by_name:
                    merged_speakers.extend(speakers_by_name[variant])

            if merged_speakers:
                merged_speakers_by_name[canonical] = merged_speakers

                if len(name_variants) > 1:
                    others = [v for v in name_variants if v != canonical]
                    logger.info(f"  [ALIAS] {others} -> '{canonical}'")

        return merged_speakers_by_name, name_mappings

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

    def _build_name_centroids(
        self,
        speakers_by_name: Dict[str, List[Dict]]
    ) -> Dict[str, np.ndarray]:
        """Build centroid embeddings for each name."""
        centroids = {}

        for name, speakers in speakers_by_name.items():
            embeddings = [sp['embedding'] for sp in speakers if sp.get('embedding') is not None]
            if embeddings:
                centroid = np.mean(embeddings, axis=0)
                centroid = centroid / np.linalg.norm(centroid)
                centroids[name] = centroid

        return centroids

    def _propagate_labels_with_conflict_detection(
        self,
        all_speakers: Dict[int, Dict],
        speaker_ids: List[int],
        embeddings: np.ndarray,
        knn_indices: np.ndarray,
        knn_sims: np.ndarray,
        id_to_idx: Dict[int, int],
        labels: Dict[int, str],
        label_to_identity: Dict[str, int],
        do_not_merge: Set[frozenset]
    ) -> Tuple[Dict[int, Dict], List[Tuple[str, str, List[int]]]]:
        """
        Single-pass label propagation WITH conflict detection.

        Returns:
            - assignments: Dict of speaker_id -> assignment info
            - contested_pairs: List of (name_a, name_b, affected_speaker_ids) for contested assignments
        """
        results = {}
        k = self.config.k_neighbors

        unlabeled_ids = [sid for sid in speaker_ids if sid not in labels]

        logger.info(f"Propagating labels to {len(unlabeled_ids)} unlabeled speakers...")

        # Track contested assignments by name pair
        contested_by_pair: Dict[frozenset, List[int]] = defaultdict(list)

        for speaker_id in tqdm(unlabeled_ids, desc="Label propagation"):
            idx = id_to_idx[speaker_id]
            speaker_data = all_speakers[speaker_id]

            is_high_duration = speaker_data.get('duration_pct', 0) >= self.config.high_duration_pct
            episode_names = speaker_data.get('episode_names', set())

            neighbor_indices = knn_indices[idx][:k]
            neighbor_sims = knn_sims[idx][:k]

            votes: Dict[str, float] = defaultdict(float)
            labeled_neighbor_count = 0
            total_weight = 0.0

            for nbr_idx, sim in zip(neighbor_indices, neighbor_sims):
                if sim < self.config.min_similarity:
                    continue

                nbr_id = speaker_ids[nbr_idx]
                if nbr_id == speaker_id:
                    continue

                if nbr_id in labels:
                    label = labels[nbr_id]
                    weight = float(sim ** self.config.similarity_power)

                    if is_high_duration and self.config.use_metadata_constraint and episode_names:
                        if label.lower() in episode_names:
                            weight *= (1.0 + self.config.metadata_similarity_boost)

                    votes[label] += weight
                    total_weight += weight
                    labeled_neighbor_count += 1

            if labeled_neighbor_count < self.config.min_labeled_neighbors:
                results[speaker_id] = {
                    'label': None,
                    'confidence': 0.0,
                    'n_labeled_neighbors': labeled_neighbor_count,
                    'status': 'unassigned',
                    'reason': f'insufficient_labeled_neighbors_{labeled_neighbor_count}'
                }
                continue

            if not votes:
                results[speaker_id] = {
                    'label': None,
                    'confidence': 0.0,
                    'n_labeled_neighbors': labeled_neighbor_count,
                    'status': 'unassigned',
                    'reason': 'no_votes'
                }
                continue

            # Sort votes by weight
            sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
            best_label, best_weight = sorted_votes[0]
            confidence = best_weight / total_weight if total_weight > 0 else 0.0

            # CHECK FOR CONTESTED ASSIGNMENT
            is_contested = False
            conflict_label = None
            conflict_confidence = None

            if len(sorted_votes) >= 2:
                second_label, second_weight = sorted_votes[1]
                vote_ratio = second_weight / best_weight if best_weight > 0 else 0

                # If second-best has significant votes, this is contested
                if vote_ratio >= self.config.contested_vote_ratio:
                    # Check if this pair is in do_not_merge (already confirmed different)
                    pair_key = frozenset([best_label.lower(), second_label.lower()])
                    if pair_key not in do_not_merge:
                        is_contested = True
                        conflict_label = second_label
                        conflict_confidence = second_weight / total_weight if total_weight > 0 else 0

                        # Track this contested pair
                        contested_by_pair[pair_key].append(speaker_id)

            if confidence < self.config.min_confidence_to_assign:
                results[speaker_id] = {
                    'label': best_label,
                    'confidence': confidence,
                    'n_labeled_neighbors': labeled_neighbor_count,
                    'status': 'conflict' if is_contested else 'unassigned',
                    'conflict_label': conflict_label,
                    'conflict_confidence': conflict_confidence,
                    'reason': f'confidence_too_low_{confidence:.2f}',
                    'identity_id': label_to_identity.get(best_label)
                }
                continue

            is_high_confidence = confidence >= self.config.high_confidence_threshold

            results[speaker_id] = {
                'label': best_label,
                'confidence': confidence,
                'n_labeled_neighbors': labeled_neighbor_count,
                'is_high_confidence': is_high_confidence,
                'status': 'conflict' if is_contested else 'assigned',
                'conflict_label': conflict_label,
                'conflict_confidence': conflict_confidence,
                'identity_id': label_to_identity.get(best_label),
                'assigned': not is_contested  # Only truly assigned if not contested
            }

        # Convert contested pairs to list
        contested_pairs = [
            (list(pair_key)[0], list(pair_key)[1], speaker_ids)
            for pair_key, speaker_ids in contested_by_pair.items()
            if len(speaker_ids) > 0
        ]

        return results, contested_pairs

    def _detect_cluster_splits(
        self,
        all_speakers: Dict[int, Dict],
        results: Dict[int, Dict],
        speaker_ids: List[int],
        embeddings: np.ndarray,
        knn_indices: np.ndarray,
        knn_sims: np.ndarray,
        id_to_idx: Dict[int, int],
        labels: Dict[int, str],
        do_not_merge: Set[frozenset]
    ) -> List[Tuple[str, str, List[int]]]:
        """
        Detect cluster splits: same voice cluster assigned different names.

        This catches cases where the propagation "split" on a boundary -
        speakers with similar voices ended up with different names.
        """
        cluster_splits = []
        checked_pairs = set()

        # For each assigned speaker, check if nearby speakers have different names
        for speaker_id, result in results.items():
            if result.get('status') != 'assigned':
                continue

            my_label = result.get('label')
            if not my_label:
                continue

            idx = id_to_idx[speaker_id]
            neighbor_indices = knn_indices[idx][:30]  # Check close neighbors
            neighbor_sims = knn_sims[idx][:30]

            for nbr_idx, sim in zip(neighbor_indices, neighbor_sims):
                if sim < self.config.cluster_split_threshold:
                    continue

                nbr_id = speaker_ids[nbr_idx]
                if nbr_id == speaker_id:
                    continue

                # Check assigned neighbors
                nbr_result = results.get(nbr_id)
                if not nbr_result or nbr_result.get('status') != 'assigned':
                    continue

                nbr_label = nbr_result.get('label')
                if not nbr_label or nbr_label == my_label:
                    continue

                # Different labels for similar speakers - potential cluster split
                pair_key = frozenset([my_label.lower(), nbr_label.lower()])

                if pair_key in checked_pairs:
                    continue
                if pair_key in do_not_merge:
                    continue

                checked_pairs.add(pair_key)

                # Find all speakers involved in this split
                involved = []
                for sp_id, sp_result in results.items():
                    if sp_result.get('label') in [my_label, nbr_label]:
                        involved.append(sp_id)

                cluster_splits.append((my_label, nbr_label, involved))

        return cluster_splits

    def _build_conflict_records(
        self,
        contested_pairs: List[Tuple[str, str, List[int]]],
        cluster_splits: List[Tuple[str, str, List[int]]],
        all_speakers: Dict[int, Dict],
        name_centroids: Dict[str, np.ndarray]
    ) -> List[ConflictRecord]:
        """Build conflict records for Phase 4 LLM verification."""
        conflicts = []
        seen_pairs = set()

        # Process contested assignments
        for name_a, name_b, speaker_ids in contested_pairs:
            pair_key = frozenset([name_a.lower(), name_b.lower()])
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            # Calculate affected hours
            affected_hours = sum(
                (all_speakers[sid].get('duration', 0) or 0) / 3600
                for sid in speaker_ids
                if sid in all_speakers
            )

            # Get centroid similarity if available
            centroid_sim = 0.0
            if name_a in name_centroids and name_b in name_centroids:
                centroid_sim = float(np.dot(name_centroids[name_a], name_centroids[name_b]))

            conflicts.append(ConflictRecord(
                name_a=name_a,
                name_b=name_b,
                centroid_similarity=centroid_sim,
                conflict_type="contested_assignment",
                affected_speakers=len(speaker_ids),
                affected_hours=affected_hours,
                sample_speaker_ids=speaker_ids[:10],
                evidence={'vote_ratio': 'high'}
            ))

            self.stats['contested_assignments'] += 1

        # Process cluster splits
        for name_a, name_b, speaker_ids in cluster_splits:
            pair_key = frozenset([name_a.lower(), name_b.lower()])
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            affected_hours = sum(
                (all_speakers[sid].get('duration', 0) or 0) / 3600
                for sid in speaker_ids
                if sid in all_speakers
            )

            centroid_sim = 0.0
            if name_a in name_centroids and name_b in name_centroids:
                centroid_sim = float(np.dot(name_centroids[name_a], name_centroids[name_b]))

            conflicts.append(ConflictRecord(
                name_a=name_a,
                name_b=name_b,
                centroid_similarity=centroid_sim,
                conflict_type="cluster_split",
                affected_speakers=len(speaker_ids),
                affected_hours=affected_hours,
                sample_speaker_ids=speaker_ids[:10],
                evidence={'nearby_different_labels': True}
            ))

            self.stats['cluster_splits'] += 1

        return conflicts

    def _write_conflicts_file(self, conflicts: List[ConflictRecord]) -> None:
        """Write conflicts to JSON file for Phase 4."""
        output = {
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'total_conflicts': len(conflicts),
            'conflicts': [
                {
                    'name_a': c.name_a,
                    'name_b': c.name_b,
                    'centroid_similarity': round(c.centroid_similarity, 3),
                    'conflict_type': c.conflict_type,
                    'affected_speakers': c.affected_speakers,
                    'affected_hours': round(c.affected_hours, 1),
                    'sample_speaker_ids': c.sample_speaker_ids,
                    'evidence': c.evidence
                }
                for c in conflicts
            ]
        }

        try:
            with open(CONFLICTS_PATH, 'w') as f:
                json.dump(output, f, indent=2)
            logger.info(f"  Conflicts written to: {CONFLICTS_PATH}")
        except Exception as e:
            logger.error(f"  Failed to write conflicts: {e}")

    def _create_or_get_identity(self, name: str, role: str = 'unknown') -> Optional[int]:
        """Create or get existing identity."""
        if self.dry_run:
            self.stats['identities_created'] += 1
            return -1

        with get_session() as session:
            existing = session.execute(
                text("""
                    SELECT id FROM speaker_identities
                    WHERE LOWER(primary_name) = LOWER(:name)
                      AND is_active = TRUE
                    ORDER BY id LIMIT 1
                """),
                {'name': name}
            ).fetchone()

            if existing:
                self.stats['identities_updated'] += 1
                return existing.id

            result = session.execute(
                text("""
                    INSERT INTO speaker_identities (
                        primary_name, role, verification_status, is_active,
                        centroid_source, created_at, updated_at
                    ) VALUES (
                        :name, :role, 'label_propagation', TRUE,
                        'label_propagation', NOW(), NOW()
                    )
                    RETURNING id
                """),
                {'name': name, 'role': role}
            )
            identity_id = result.fetchone()[0]
            session.commit()

            self.stats['identities_created'] += 1
            return identity_id

    def _assign_speakers(self, assignments: Dict[int, Dict], labels: Dict[int, str], label_to_identity: Dict[str, int]):
        """Batch assign speakers to identities."""
        if self.dry_run:
            return

        timestamp = datetime.now(timezone.utc).isoformat()

        with get_session() as session:
            # Assign anchors first
            for speaker_id, label in labels.items():
                identity_id = label_to_identity.get(label)
                if not identity_id or identity_id == -1:
                    continue

                phase_entry = {
                    'status': 'assigned',
                    'timestamp': timestamp,
                    'method': 'label_propagation_anchor',
                    'identity_id': identity_id,
                    'identity_name': label,
                    'confidence': 1.0,
                    'is_anchor': True
                }

                session.execute(
                    text(f"""
                        UPDATE speakers SET
                            speaker_identity_id = :identity_id,
                            assignment_confidence = 1.0,
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
                        'speaker_id': speaker_id,
                        'phase_entry': json.dumps(phase_entry)
                    }
                )

            # Assign propagated labels (non-conflict only)
            for speaker_id, assignment in assignments.items():
                status = assignment.get('status')
                if status != 'assigned':
                    continue

                identity_id = assignment.get('identity_id')
                if not identity_id or identity_id == -1:
                    continue

                confidence = assignment['confidence']
                is_high = assignment.get('is_high_confidence', False)

                phase_entry = {
                    'status': 'assigned',
                    'timestamp': timestamp,
                    'method': 'label_propagation',
                    'identity_id': identity_id,
                    'identity_name': assignment['label'],
                    'confidence': confidence,
                    'is_high_confidence': is_high,
                    'n_labeled_neighbors': assignment['n_labeled_neighbors']
                }

                session.execute(
                    text(f"""
                        UPDATE speakers SET
                            speaker_identity_id = :identity_id,
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

            # Mark conflicted speakers
            for speaker_id, assignment in assignments.items():
                if assignment.get('status') != 'conflict':
                    continue

                phase_entry = {
                    'status': 'conflict',
                    'timestamp': timestamp,
                    'method': 'label_propagation',
                    'primary_label': assignment.get('label'),
                    'primary_confidence': assignment.get('confidence'),
                    'conflict_label': assignment.get('conflict_label'),
                    'conflict_confidence': assignment.get('conflict_confidence'),
                    'awaiting_resolution': True
                }

                session.execute(
                    text(f"""
                        UPDATE speakers SET
                            identification_details = jsonb_set(
                                COALESCE(identification_details, '{{}}'::jsonb),
                                ARRAY['{PHASE_KEY}'],
                                CAST(:phase_entry AS jsonb)
                            ),
                            updated_at = NOW()
                        WHERE id = :speaker_id
                    """),
                    {
                        'speaker_id': speaker_id,
                        'phase_entry': json.dumps(phase_entry)
                    }
                )

            session.commit()

    async def run(self, project: str = None) -> Dict:
        """Run label propagation with conflict detection."""
        if project and not self.start_date and not self.end_date:
            start_date, end_date = get_project_date_range(project)
            self.start_date = start_date
            self.end_date = end_date

        logger.info("=" * 80)
        logger.info("PHASE 3: LABEL PROPAGATION WITH CONFLICT DETECTION")
        logger.info("=" * 80)
        logger.info(f"Mode: {'DRY RUN' if self.dry_run else 'APPLY'}")
        logger.info(f"Project: {project or 'ALL'}")
        if self.start_date or self.end_date:
            logger.info(f"Date range: {self.start_date or 'any'} to {self.end_date or 'any'}")
        logger.info(f"Config: k={self.config.k_neighbors}, min_sim={self.config.min_similarity}, "
                   f"contested_ratio={self.config.contested_vote_ratio}")
        logger.info("-" * 80)

        # Step 1: Load all speakers
        logger.info("Loading all speakers...")
        all_speakers = self._load_all_speakers(project)
        self.stats['total_speakers'] = len(all_speakers)
        logger.info(f"Loaded {len(all_speakers)} speakers with embeddings")

        if not all_speakers:
            logger.warning("No speakers found!")
            return self.stats

        # Step 2: Build FAISS index
        logger.info("Building FAISS index...")
        index, speaker_ids, embeddings = self._build_faiss_index(all_speakers)
        id_to_idx = {sid: idx for idx, sid in enumerate(speaker_ids)}
        logger.info(f"Built index with {len(speaker_ids)} speakers")

        # Step 3: Pre-compute k-NN
        logger.info(f"Pre-computing k-NN (k={self.config.k_build})...")
        knn_indices, knn_sims = self._precompute_knn(index, embeddings, k=self.config.k_build)

        # Step 4: Extract labeled speakers (Phase 2 certain)
        logger.info("-" * 80)
        logger.info("EXTRACTING LABELED SPEAKERS")
        logger.info("-" * 80)

        speakers_by_name: Dict[str, List[Dict]] = defaultdict(list)
        for sid, sp in all_speakers.items():
            if sp.get('phase2_status') == 'certain' and sp.get('phase2_name'):
                speakers_by_name[sp['phase2_name']].append(sp)

        # Apply name filter if specified
        if self.name_filter:
            filter_lower = self.name_filter.lower()
            speakers_by_name = {
                name: speakers for name, speakers in speakers_by_name.items()
                if filter_lower in name.lower()
            }

        # Apply max_anchors limit
        if self.max_anchors and len(speakers_by_name) > self.max_anchors:
            sorted_names = sorted(speakers_by_name.items(), key=lambda x: len(x[1]), reverse=True)
            speakers_by_name = dict(sorted_names[:self.max_anchors])

        # Step 5: Load and apply static aliases
        logger.info("Loading name aliases...")
        alias_to_canonical, unresolved_handles, do_not_merge = self._load_name_aliases()

        speakers_by_name, name_mappings = self._apply_static_aliases(
            speakers_by_name, alias_to_canonical
        )

        if name_mappings:
            logger.info(f"Applied {len(name_mappings)} static alias merges")

        self.stats['unique_names'] = len(speakers_by_name)

        # Build labels dict
        labels: Dict[int, str] = {}
        label_to_identity: Dict[str, int] = {}

        for name, verified_speakers in speakers_by_name.items():
            identity_id = self._create_or_get_identity(name)
            label_to_identity[name] = identity_id

            for sp in verified_speakers:
                labels[sp['speaker_id']] = name

            logger.info(f"  Label: \"{name}\" - {len(verified_speakers)} anchors")

        self.stats['labeled_speakers'] = len(labels)
        self.stats['unlabeled_speakers'] = len(all_speakers) - len(labels)

        logger.info(f"Total labeled: {len(labels)}, Unlabeled: {self.stats['unlabeled_speakers']}")

        # Build name centroids for conflict detection
        name_centroids = self._build_name_centroids(speakers_by_name)

        # Step 6: Run label propagation with conflict detection
        logger.info("-" * 80)
        logger.info("RUNNING LABEL PROPAGATION")
        logger.info("-" * 80)

        results, contested_pairs = self._propagate_labels_with_conflict_detection(
            all_speakers=all_speakers,
            speaker_ids=speaker_ids,
            embeddings=embeddings,
            knn_indices=knn_indices,
            knn_sims=knn_sims,
            id_to_idx=id_to_idx,
            labels=labels,
            label_to_identity=label_to_identity,
            do_not_merge=do_not_merge
        )

        self.assignments = results

        # Step 7: Detect cluster splits
        logger.info("Detecting cluster splits...")
        cluster_splits = self._detect_cluster_splits(
            all_speakers=all_speakers,
            results=results,
            speaker_ids=speaker_ids,
            embeddings=embeddings,
            knn_indices=knn_indices,
            knn_sims=knn_sims,
            id_to_idx=id_to_idx,
            labels=labels,
            do_not_merge=do_not_merge
        )

        # Step 8: Build conflict records
        conflicts = self._build_conflict_records(
            contested_pairs, cluster_splits, all_speakers, name_centroids
        )
        self.conflicts = conflicts
        self.stats['conflicts_detected'] = len(conflicts)

        # Count results
        assigned = [r for r in results.values() if r.get('status') == 'assigned']
        conflicted = [r for r in results.values() if r.get('status') == 'conflict']
        unassigned = [r for r in results.values() if r.get('status') == 'unassigned']
        high_conf = [r for r in assigned if r.get('is_high_confidence')]

        self.stats['speakers_assigned'] = len(assigned)
        self.stats['high_confidence_assignments'] = len(high_conf)
        self.stats['low_confidence_assignments'] = len(assigned) - len(high_conf)
        self.stats['speakers_with_conflicts'] = len(conflicted)
        self.stats['speakers_unassigned'] = len(unassigned)

        # Calculate speaking time metrics
        total_duration = sum(sp.get('duration', 0) or 0 for sp in all_speakers.values())
        anchor_duration = sum(all_speakers[sid].get('duration', 0) or 0 for sid in labels.keys())
        assigned_duration = sum(
            all_speakers[sp_id].get('duration', 0) or 0
            for sp_id, r in results.items()
            if r.get('status') == 'assigned'
        )
        conflict_duration = sum(
            all_speakers[sp_id].get('duration', 0) or 0
            for sp_id, r in results.items()
            if r.get('status') == 'conflict'
        )

        self.stats['total_speaking_hours'] = total_duration / 3600
        self.stats['assigned_speaking_hours'] = (anchor_duration + assigned_duration) / 3600
        self.stats['conflict_speaking_hours'] = conflict_duration / 3600
        self.stats['unassigned_speaking_hours'] = (total_duration - anchor_duration - assigned_duration - conflict_duration) / 3600

        # Step 9: Apply assignments (non-conflict only)
        logger.info("-" * 80)
        logger.info("APPLYING ASSIGNMENTS")
        logger.info("-" * 80)

        self._assign_speakers(results, labels, label_to_identity)
        logger.info(f"Assigned {len(labels)} anchors + {self.stats['speakers_assigned']} propagated")
        logger.info(f"Marked {self.stats['speakers_with_conflicts']} speakers with conflicts")

        # Step 10: Write conflicts file for Phase 4
        if conflicts:
            logger.info("-" * 80)
            logger.info("WRITING CONFLICTS FOR PHASE 4")
            logger.info("-" * 80)
            self._write_conflicts_file(conflicts)

        # Print summary
        self._print_summary()

        return self.stats

    def _print_summary(self):
        """Print summary statistics."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)

        logger.info("SPEAKER COUNTS:")
        logger.info(f"  Total speakers: {self.stats['total_speakers']}")
        logger.info(f"  Unique names (anchors): {self.stats['unique_names']}")
        logger.info(f"  Labeled speakers (anchors): {self.stats['labeled_speakers']}")
        logger.info(f"  Speakers assigned: {self.stats['speakers_assigned']}")
        logger.info(f"    - High confidence: {self.stats['high_confidence_assignments']}")
        logger.info(f"    - Low confidence: {self.stats['low_confidence_assignments']}")
        logger.info(f"  Speakers with conflicts: {self.stats['speakers_with_conflicts']}")
        logger.info(f"  Speakers unassigned: {self.stats['speakers_unassigned']}")

        logger.info("")
        logger.info("CONFLICT DETECTION:")
        logger.info(f"  Total conflicts detected: {self.stats['conflicts_detected']}")
        logger.info(f"    - Contested assignments: {self.stats['contested_assignments']}")
        logger.info(f"    - Cluster splits: {self.stats['cluster_splits']}")

        if self.conflicts:
            logger.info("")
            logger.info("  Conflicts to resolve in Phase 4:")
            for c in self.conflicts[:10]:
                logger.info(f"    • {c.name_a} vs {c.name_b}: {c.affected_speakers} speakers, "
                           f"{c.affected_hours:.1f}h ({c.conflict_type})")
            if len(self.conflicts) > 10:
                logger.info(f"    ... and {len(self.conflicts) - 10} more")

        logger.info("")
        logger.info("SPEAKING TIME:")
        logger.info(f"  Total: {self.stats['total_speaking_hours']:.1f} hours")
        logger.info(f"  Assigned: {self.stats['assigned_speaking_hours']:.1f} hours")
        logger.info(f"  With conflicts: {self.stats['conflict_speaking_hours']:.1f} hours")
        logger.info(f"  Unassigned: {self.stats['unassigned_speaking_hours']:.1f} hours")

        coverage = (self.stats['assigned_speaking_hours'] / self.stats['total_speaking_hours'] * 100) if self.stats['total_speaking_hours'] > 0 else 0
        logger.info(f"  >>> COVERAGE (assigned only): {coverage:.1f}%")

        if self.stats['conflicts_detected'] > 0:
            logger.info("")
            logger.info("⚠️  Run Phase 4 to resolve conflicts via LLM verification")
        else:
            logger.info("")
            logger.info("✓ No conflicts detected - Phase 4 not needed")

        logger.info("=" * 80)


async def main():
    parser = argparse.ArgumentParser(
        description='Phase 3: Label Propagation with Conflict Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run
  python -m src.speaker_identification.strategies.label_propagation_with_conflicts \\
      --project CPRMV

  # Apply
  python -m src.speaker_identification.strategies.label_propagation_with_conflicts \\
      --project CPRMV --apply

  # Test with limited anchors
  python -m src.speaker_identification.strategies.label_propagation_with_conflicts \\
      --project CPRMV --max-anchors 25 --apply
"""
    )

    parser.add_argument('--project', type=str, help='Filter to project')
    parser.add_argument('--name', type=str, help='Filter to specific speaker name')
    parser.add_argument('--apply', action='store_true', help='Apply changes (default: dry run)')
    parser.add_argument('--max-anchors', type=int, default=None,
                       help='Max unique names to process (for testing)')

    # Config overrides
    parser.add_argument('--k', type=int, default=75,
                       help='Number of neighbors for voting (default: 75)')
    parser.add_argument('--min-sim', type=float, default=0.55,
                       help='Minimum similarity for neighbor vote (default: 0.55)')
    parser.add_argument('--min-conf', type=float, default=0.25,
                       help='Minimum confidence to assign (default: 0.25)')
    parser.add_argument('--contested-ratio', type=float, default=0.70,
                       help='Vote ratio threshold for contested assignment (default: 0.70)')

    args = parser.parse_args()

    config = LabelPropagationWithConflictsConfig(
        k_neighbors=args.k,
        min_similarity=args.min_sim,
        min_confidence_to_assign=args.min_conf,
        contested_vote_ratio=args.contested_ratio
    )

    strategy = LabelPropagationWithConflictsStrategy(
        config=config,
        dry_run=not args.apply,
        max_anchors=args.max_anchors,
        name_filter=args.name
    )

    try:
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
