#!/usr/bin/env python3
"""
Single-Pass Label Propagation for Speaker Identification (Phase 3 Alternative)
===============================================================================

Alternative Phase 3 approach: Instead of iterative cluster expansion from anchors,
perform a SINGLE PASS over all speakers and assign each a "likely score" based on
neighboring labels.

Key Differences from Anchor-Canopy Clustering:
----------------------------------------------
1. SINGLE PASS: O(n) pass over all speakers, no iterative expansion
2. CONFIDENCE SCORES: Each assignment gets a 0.0-1.0 confidence based on neighbor agreement
3. SOFT BOUNDARIES: Handles gradual transitions between speakers naturally
4. NO GATES: Simple weighted voting instead of multi-gate validation
5. FASTER: Much faster than iterative expansion for large speaker sets

Algorithm:
1. Load all speakers with embeddings
2. Build FAISS index, pre-compute k-NN (k=50-100)
3. For text-verified speakers, set their label as known (from Phase 2)
4. Single pass over ALL unlabeled speakers:
   - Look at k nearest neighbors
   - Count label occurrences among neighbors (weighted by similarity)
   - Assign most common label with confidence = weighted_votes / total_weight
5. Output: Each speaker gets (predicted_name, confidence_score)

Usage:
    # Dry run
    python -m src.speaker_identification.strategies.label_propagation_clustering --project CPRMV

    # Apply
    python -m src.speaker_identification.strategies.label_propagation_clustering \\
        --project CPRMV --apply

    # Test with limited anchors
    python -m src.speaker_identification.strategies.label_propagation_clustering \\
        --project CPRMV --max-anchors 25
"""

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import faiss
import numpy as np
from tqdm import tqdm

from src.utils.paths import get_project_root
project_root = str(get_project_root())
sys.path.append(project_root)

from sqlalchemy import text

from src.database.session import get_session
from src.utils.logger import setup_worker_logger
from src.utils.config import get_project_date_range

logger = setup_worker_logger('speaker_identification.label_propagation')

# Console logging
import logging
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

PHASE_KEY = "phase3_lp"  # Distinct from anchor-canopy phase3


@dataclass
class LabelPropagationConfig:
    """Configuration for single-pass label propagation.

    These parameters control the label propagation behavior and
    confidence thresholds for assignment.
    """
    # k-NN parameters
    k_neighbors: int = 75          # Number of neighbors to consider for voting
    k_build: int = 128             # k for FAISS search (larger than k_neighbors for buffer)

    # Similarity weighting
    min_similarity: float = 0.55   # Minimum similarity to count as a "neighbor vote"
    similarity_power: float = 2.0  # Power to raise similarity for weighting (higher = more weight to close neighbors)

    # Confidence thresholds
    min_confidence_to_assign: float = 0.25  # Minimum confidence score to assign (lower = more assignments)
    high_confidence_threshold: float = 0.60  # Above this = high confidence assignment

    # Label requirements
    min_labeled_neighbors: int = 1  # Require at least N labeled neighbors to make assignment

    # Name collision handling (same name, different people)
    collision_threshold: float = 0.65  # Cluster same-name verified speakers at this threshold

    # Embedding quality
    min_embedding_quality: float = 0.50

    # Stage 3b: Metadata-constrained assignment for high-duration speakers
    use_metadata_constraint: bool = True   # Use episode hosts/guests to constrain assignments
    high_duration_pct: float = 0.10        # Speakers with >= this % are high-duration (targets for 3b, matches Phase 2)
    metadata_similarity_boost: float = 0.10  # Bonus similarity when label matches episode metadata


class LabelPropagationStrategy:
    """
    Phase 3 Alternative: Single-Pass Label Propagation.

    Uses text-verified speakers as "labeled" nodes, then propagates
    labels to unlabeled speakers based on k-NN weighted voting.
    """

    def __init__(
        self,
        config: LabelPropagationConfig = None,
        dry_run: bool = True,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_anchors: Optional[int] = None,
        name_filter: Optional[str] = None
    ):
        self.config = config or LabelPropagationConfig()
        self.dry_run = dry_run
        self.start_date = start_date
        self.end_date = end_date
        self.max_anchors = max_anchors  # Limit unique names to process (for testing)
        self.name_filter = name_filter

        # Tracking
        self.stats = {
            'total_speakers': 0,
            'labeled_speakers': 0,
            'unlabeled_speakers': 0,
            'unique_names': 0,
            'name_collisions_detected': 0,
            'speakers_assigned': 0,
            'high_confidence_assignments': 0,
            'low_confidence_assignments': 0,
            'speakers_unassigned': 0,
            'identities_created': 0,
            'identities_updated': 0,
            # Speaking time metrics
            'total_speaking_hours': 0.0,
            'assigned_speaking_hours': 0.0,
            'anchor_speaking_hours': 0.0,
            'propagated_speaking_hours': 0.0,
            'unassigned_speaking_hours': 0.0,
            'speaking_time_coverage_pct': 0.0,
            # Stage 3b metrics (high-duration speakers)
            'high_duration_total_hours': 0.0,
            'high_duration_assigned_hours': 0.0,
            'high_duration_coverage_pct': 0.0,
            'metadata_boost_assignments': 0,
            'direct_metadata_assignments': 0,
            'errors': []
        }

        # Results storage
        self.assignments: Dict[int, Dict] = {}  # speaker_id -> {label, confidence, ...}

    def _load_all_speakers(self, project: str) -> Dict[int, Dict]:
        """Load all speakers with embeddings for the project.

        Also loads episode metadata (hosts/guests) for Stage 3b inference.
        """
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

                # Parse episode metadata (hosts/guests) for Stage 3b
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

                # Calculate duration percentage
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
                    'episode_names': episode_names  # Host/guest names from metadata
                }

            return speakers

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

    def _detect_name_collisions(
        self,
        speakers: List[Dict]
    ) -> List[List[Dict]]:
        """
        Detect name collisions - same name, different people.

        Clusters speakers with the same name by embedding similarity.
        Returns list of clusters, where each cluster is a distinct person.
        """
        if len(speakers) < 2:
            return [speakers]

        # Skip collision detection for very high consensus
        if len(speakers) >= 50:
            logger.info(f"    Skipping collision detection ({len(speakers)} verified = high consensus)")
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

        # Union-find clustering
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

    def _propagate_labels(
        self,
        all_speakers: Dict[int, Dict],
        speaker_ids: List[int],
        embeddings: np.ndarray,
        knn_indices: np.ndarray,
        knn_sims: np.ndarray,
        id_to_idx: Dict[int, int],
        labels: Dict[int, str],  # speaker_id -> label (name)
        label_to_identity: Dict[str, int]  # label -> identity_id
    ) -> Dict[int, Dict]:
        """
        Single-pass label propagation with Stage 3b metadata constraints.

        For each unlabeled speaker:
        1. Look at k nearest neighbors
        2. Count weighted votes from labeled neighbors
        3. Stage 3b: For high-duration speakers (20%+), boost votes for labels
           that match episode metadata (hosts/guests)
        4. Assign most common label with confidence score

        Returns:
            Dict of speaker_id -> {label, confidence, n_labeled_neighbors, ...}
        """
        results = {}
        k = self.config.k_neighbors

        # Get unlabeled speaker indices
        unlabeled_ids = [sid for sid in speaker_ids if sid not in labels]

        # Count high-duration speakers for logging
        high_duration_count = sum(
            1 for sid in unlabeled_ids
            if all_speakers[sid].get('duration_pct', 0) >= self.config.high_duration_pct
        )

        logger.info(f"Propagating labels to {len(unlabeled_ids)} unlabeled speakers...")
        logger.info(f"  Stage 3b: {high_duration_count} high-duration speakers (>={self.config.high_duration_pct*100:.0f}%) "
                   f"will use metadata constraints")

        metadata_boost_count = 0

        for speaker_id in tqdm(unlabeled_ids, desc="Label propagation"):
            idx = id_to_idx[speaker_id]
            speaker_data = all_speakers[speaker_id]

            # Check if high-duration speaker (Stage 3b candidate)
            is_high_duration = speaker_data.get('duration_pct', 0) >= self.config.high_duration_pct
            episode_names = speaker_data.get('episode_names', set())

            # Get k nearest neighbors and their similarities
            neighbor_indices = knn_indices[idx][:k]
            neighbor_sims = knn_sims[idx][:k]

            # Weighted voting from labeled neighbors
            votes: Dict[str, float] = defaultdict(float)
            labeled_neighbor_count = 0
            total_weight = 0.0
            metadata_matched = False

            for nbr_idx, sim in zip(neighbor_indices, neighbor_sims):
                if sim < self.config.min_similarity:
                    continue  # Skip neighbors below similarity threshold

                nbr_id = speaker_ids[nbr_idx]
                if nbr_id == speaker_id:
                    continue  # Skip self

                if nbr_id in labels:
                    label = labels[nbr_id]

                    # Base weight: similarity raised to power
                    weight = float(sim ** self.config.similarity_power)

                    # Stage 3b: Boost weight if label matches episode metadata
                    # This helps high-duration speakers get assigned to known hosts/guests
                    if is_high_duration and self.config.use_metadata_constraint and episode_names:
                        if label.lower() in episode_names:
                            weight *= (1.0 + self.config.metadata_similarity_boost)
                            metadata_matched = True

                    votes[label] += weight
                    total_weight += weight
                    labeled_neighbor_count += 1

            # Skip if not enough labeled neighbors
            if labeled_neighbor_count < self.config.min_labeled_neighbors:
                results[speaker_id] = {
                    'label': None,
                    'confidence': 0.0,
                    'n_labeled_neighbors': labeled_neighbor_count,
                    'reason': f'insufficient_labeled_neighbors_{labeled_neighbor_count}'
                }
                continue

            # Find winning label
            if not votes:
                results[speaker_id] = {
                    'label': None,
                    'confidence': 0.0,
                    'n_labeled_neighbors': labeled_neighbor_count,
                    'reason': 'no_votes'
                }
                continue

            # Get label with highest weighted vote
            best_label = max(votes.keys(), key=lambda l: votes[l])
            best_weight = votes[best_label]

            # Confidence = proportion of weight for winning label
            confidence = best_weight / total_weight if total_weight > 0 else 0.0

            # Track label agreement (how many distinct labels voted)
            # This helps identify noisy/uncertain regions
            n_distinct_labels = len(votes)

            # Track metadata boost
            if metadata_matched and best_label.lower() in episode_names:
                metadata_boost_count += 1

            # Check confidence threshold
            if confidence < self.config.min_confidence_to_assign:
                results[speaker_id] = {
                    'label': best_label,
                    'confidence': confidence,
                    'n_labeled_neighbors': labeled_neighbor_count,
                    'n_distinct_labels': n_distinct_labels,
                    'reason': f'confidence_too_low_{confidence:.2f}',
                    'identity_id': label_to_identity.get(best_label),
                    'metadata_matched': metadata_matched and best_label.lower() in episode_names
                }
                continue

            # Assignment!
            is_high_confidence = confidence >= self.config.high_confidence_threshold

            results[speaker_id] = {
                'label': best_label,
                'confidence': confidence,
                'n_labeled_neighbors': labeled_neighbor_count,
                'n_distinct_labels': n_distinct_labels,
                'total_weight': total_weight,
                'best_weight': best_weight,
                'is_high_confidence': is_high_confidence,
                'identity_id': label_to_identity.get(best_label),
                'metadata_matched': metadata_matched and best_label.lower() in episode_names,
                'assigned': True
            }

        logger.info(f"  Stage 3b: {metadata_boost_count} assignments benefited from metadata match")

        # Stage 3b Part 2: Direct metadata assignment for unassigned high-duration speakers
        # For speakers that couldn't be assigned via propagation but have episode metadata,
        # try to match them directly to centroids of known hosts/guests
        if self.config.use_metadata_constraint:
            unassigned_high_dur = [
                sp_id for sp_id, r in results.items()
                if not r.get('assigned')
                and all_speakers[sp_id].get('duration_pct', 0) >= self.config.high_duration_pct
                and all_speakers[sp_id].get('episode_names')
            ]

            if unassigned_high_dur:
                logger.info(f"  Stage 3b direct assignment: {len(unassigned_high_dur)} unassigned high-duration speakers with metadata")

                # Build label centroids from anchors
                label_centroids = self._build_label_centroids(all_speakers, labels, id_to_idx, embeddings)

                direct_assigned = 0
                for sp_id in unassigned_high_dur:
                    sp_data = all_speakers[sp_id]
                    sp_emb = sp_data['embedding']
                    episode_names = sp_data.get('episode_names', set())

                    if not episode_names:
                        continue

                    # Try to match against centroids of names in this episode
                    best_match = None
                    best_sim = 0.0

                    for name in episode_names:
                        # Check if we have a centroid for this name
                        # Try exact match first, then case variations
                        centroid = None
                        for label in label_centroids:
                            if label.lower() == name.lower():
                                centroid = label_centroids[label]
                                matching_label = label
                                break

                        if centroid is None:
                            continue

                        sim = float(np.dot(sp_emb, centroid))
                        if sim > best_sim and sim >= 0.60:  # Direct assignment threshold
                            best_sim = sim
                            best_match = matching_label

                    if best_match:
                        results[sp_id] = {
                            'label': best_match,
                            'confidence': best_sim,
                            'n_labeled_neighbors': 0,
                            'n_distinct_labels': 1,
                            'is_high_confidence': best_sim >= self.config.high_confidence_threshold,
                            'identity_id': label_to_identity.get(best_match),
                            'metadata_matched': True,
                            'direct_metadata_assignment': True,
                            'assigned': True
                        }
                        direct_assigned += 1

                logger.info(f"  Stage 3b direct assignment: {direct_assigned} speakers assigned via centroid matching")
                self.stats['direct_metadata_assignments'] = direct_assigned

        return results

    def _build_label_centroids(
        self,
        all_speakers: Dict[int, Dict],
        labels: Dict[int, str],
        id_to_idx: Dict[int, int],
        embeddings: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Build centroid embeddings for each label from anchor speakers."""
        label_embeddings: Dict[str, List[np.ndarray]] = defaultdict(list)

        for sp_id, label in labels.items():
            if sp_id in all_speakers:
                label_embeddings[label].append(all_speakers[sp_id]['embedding'])

        # Compute mean centroid for each label
        centroids = {}
        for label, embs in label_embeddings.items():
            if embs:
                centroid = np.mean(embs, axis=0)
                centroid = centroid / np.linalg.norm(centroid)
                centroids[label] = centroid

        return centroids

    def _create_or_get_identity(self, name: str, role: str = 'unknown') -> Optional[int]:
        """Create or get existing identity. Returns -1 in dry_run mode."""
        if self.dry_run:
            self.stats['identities_created'] += 1
            return -1

        with get_session() as session:
            # Check for existing
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

            # Create new
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

    def _assign_speakers(self, assignments: Dict[int, Dict]):
        """Batch assign speakers to identities."""
        if self.dry_run:
            return

        timestamp = datetime.now(timezone.utc).isoformat()

        with get_session() as session:
            for speaker_id, assignment in assignments.items():
                if not assignment.get('assigned'):
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

            session.commit()

    def _assign_labeled_speakers(self, labels: Dict[int, str], label_to_identity: Dict[str, int]):
        """Assign the text-verified speakers to their identities too."""
        if self.dry_run:
            return

        timestamp = datetime.now(timezone.utc).isoformat()

        with get_session() as session:
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

            session.commit()

    async def run(self, project: str = None) -> Dict:
        """
        Run single-pass label propagation.

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
        logger.info("SINGLE-PASS LABEL PROPAGATION (Phase 3 Alternative)")
        logger.info("=" * 80)
        logger.info(f"Mode: {'DRY RUN' if self.dry_run else 'APPLY'}")
        logger.info(f"Project: {project or 'ALL'}")
        if self.start_date or self.end_date:
            logger.info(f"Date range: {self.start_date or 'any'} to {self.end_date or 'any'}")
        logger.info(f"Config: k={self.config.k_neighbors}, min_sim={self.config.min_similarity}, "
                   f"min_conf={self.config.min_confidence_to_assign}")
        if self.max_anchors:
            logger.info(f"Max anchors: {self.max_anchors} (testing mode)")
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
        logger.info(f"Built index with {len(speaker_ids)} speakers (dim={embeddings.shape[1]})")

        # Step 3: Pre-compute k-NN
        logger.info(f"Pre-computing k-NN (k={self.config.k_build})...")
        knn_indices, knn_sims = self._precompute_knn(index, embeddings, k=self.config.k_build)
        logger.info("k-NN computation complete")

        # Step 4: Extract labeled speakers (Phase 2 certain)
        logger.info("-" * 80)
        logger.info("EXTRACTING LABELED SPEAKERS (Phase 2 certain)")
        logger.info("-" * 80)

        # Group by name for collision detection
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
            logger.info(f"Filtered to {len(speakers_by_name)} names matching '{self.name_filter}'")

        # Apply max_anchors limit
        if self.max_anchors and len(speakers_by_name) > self.max_anchors:
            # Sort by count descending, take top N
            sorted_names = sorted(speakers_by_name.items(), key=lambda x: len(x[1]), reverse=True)
            speakers_by_name = dict(sorted_names[:self.max_anchors])
            logger.info(f"Limited to {len(speakers_by_name)} name groups")

        self.stats['unique_names'] = len(speakers_by_name)

        # Build labels dict and handle collisions
        labels: Dict[int, str] = {}  # speaker_id -> label (name or name (2), etc.)
        label_to_identity: Dict[str, int] = {}  # label -> identity_id

        for name, verified_speakers in speakers_by_name.items():
            # Detect name collisions
            person_clusters = self._detect_name_collisions(verified_speakers)

            if len(person_clusters) > 1:
                logger.info(f"⚠ NAME COLLISION: '{name}' has {len(person_clusters)} distinct people")
                self.stats['name_collisions_detected'] += 1

            # Create identity and assign labels for each cluster
            for i, cluster_speakers in enumerate(person_clusters):
                final_name = name if i == 0 else f"{name} ({i + 1})"

                # Create or get identity
                identity_id = self._create_or_get_identity(final_name)
                label_to_identity[final_name] = identity_id

                # Assign labels to all verified speakers in this cluster
                for sp in cluster_speakers:
                    labels[sp['speaker_id']] = final_name

                logger.info(f"  Label: \"{final_name}\" - {len(cluster_speakers)} verified speakers")

        self.stats['labeled_speakers'] = len(labels)
        self.stats['unlabeled_speakers'] = len(all_speakers) - len(labels)

        logger.info(f"Total labeled: {len(labels)}, Unlabeled: {self.stats['unlabeled_speakers']}")

        # Step 5: Single-pass label propagation
        logger.info("-" * 80)
        logger.info("RUNNING LABEL PROPAGATION")
        logger.info("-" * 80)

        results = self._propagate_labels(
            all_speakers=all_speakers,
            speaker_ids=speaker_ids,
            embeddings=embeddings,
            knn_indices=knn_indices,
            knn_sims=knn_sims,
            id_to_idx=id_to_idx,
            labels=labels,
            label_to_identity=label_to_identity
        )

        self.assignments = results

        # Count results and speaking time
        assigned = [r for r in results.values() if r.get('assigned')]
        unassigned = [r for r in results.values() if not r.get('assigned')]
        high_conf = [r for r in assigned if r.get('is_high_confidence')]

        self.stats['speakers_assigned'] = len(assigned)
        self.stats['high_confidence_assignments'] = len(high_conf)
        self.stats['low_confidence_assignments'] = len(assigned) - len(high_conf)
        self.stats['speakers_unassigned'] = len(unassigned)

        # Calculate speaking time metrics
        total_duration = sum(sp.get('duration', 0) or 0 for sp in all_speakers.values())
        anchor_duration = sum(all_speakers[sid].get('duration', 0) or 0 for sid in labels.keys())
        propagated_duration = sum(
            all_speakers[sp_id].get('duration', 0) or 0
            for sp_id, r in results.items()
            if r.get('assigned')
        )
        assigned_duration = anchor_duration + propagated_duration
        unassigned_duration = total_duration - assigned_duration

        self.stats['total_speaking_hours'] = total_duration / 3600
        self.stats['anchor_speaking_hours'] = anchor_duration / 3600
        self.stats['propagated_speaking_hours'] = propagated_duration / 3600
        self.stats['assigned_speaking_hours'] = assigned_duration / 3600
        self.stats['unassigned_speaking_hours'] = unassigned_duration / 3600
        self.stats['speaking_time_coverage_pct'] = (assigned_duration / total_duration * 100) if total_duration > 0 else 0

        # Stage 3b metrics: High-duration speakers (20%+)
        high_dur_threshold = self.config.high_duration_pct
        high_dur_total = sum(
            sp.get('duration', 0) or 0
            for sp in all_speakers.values()
            if sp.get('duration_pct', 0) >= high_dur_threshold
        )
        high_dur_anchor = sum(
            all_speakers[sid].get('duration', 0) or 0
            for sid in labels.keys()
            if all_speakers[sid].get('duration_pct', 0) >= high_dur_threshold
        )
        high_dur_propagated = sum(
            all_speakers[sp_id].get('duration', 0) or 0
            for sp_id, r in results.items()
            if r.get('assigned') and all_speakers[sp_id].get('duration_pct', 0) >= high_dur_threshold
        )
        high_dur_assigned = high_dur_anchor + high_dur_propagated

        self.stats['high_duration_total_hours'] = high_dur_total / 3600
        self.stats['high_duration_assigned_hours'] = high_dur_assigned / 3600
        self.stats['high_duration_coverage_pct'] = (high_dur_assigned / high_dur_total * 100) if high_dur_total > 0 else 0

        # Count metadata-boosted assignments
        self.stats['metadata_boost_assignments'] = sum(
            1 for r in results.values()
            if r.get('assigned') and r.get('metadata_matched')
        )

        # Step 6: Apply assignments
        logger.info("-" * 80)
        logger.info("APPLYING ASSIGNMENTS")
        logger.info("-" * 80)

        # Assign labeled speakers (anchors) first
        self._assign_labeled_speakers(labels, label_to_identity)
        logger.info(f"Assigned {len(labels)} labeled speakers (anchors)")

        # Assign propagated labels
        self._assign_speakers(results)
        logger.info(f"Assigned {self.stats['speakers_assigned']} speakers via propagation")

        # Step 7: Summary by identity with speaking time
        logger.info("-" * 80)
        logger.info("RESULTS BY IDENTITY (sorted by speaking time)")
        logger.info("-" * 80)

        # Group by label with speaking time
        by_label: Dict[str, Dict] = defaultdict(lambda: {
            'speakers': [], 'anchor_ids': [], 'propagated_ids': [],
            'anchor_hours': 0.0, 'propagated_hours': 0.0
        })

        for sp_id, r in results.items():
            if r.get('assigned') and r.get('label'):
                label = r['label']
                by_label[label]['speakers'].append(r)
                by_label[label]['propagated_ids'].append(sp_id)
                by_label[label]['propagated_hours'] += (all_speakers[sp_id].get('duration', 0) or 0) / 3600

        # Add labeled speakers (anchors)
        for sp_id, label in labels.items():
            by_label[label]['speakers'].append({'speaker_id': sp_id, 'confidence': 1.0, 'is_anchor': True})
            by_label[label]['anchor_ids'].append(sp_id)
            by_label[label]['anchor_hours'] += (all_speakers[sp_id].get('duration', 0) or 0) / 3600

        # Sort by total speaking time (not count)
        sorted_labels = sorted(
            by_label.items(),
            key=lambda x: x[1]['anchor_hours'] + x[1]['propagated_hours'],
            reverse=True
        )

        for label, data in sorted_labels[:20]:  # Top 20
            n_anchors = len(data['anchor_ids'])
            n_propagated = len(data['propagated_ids'])
            total_hours = data['anchor_hours'] + data['propagated_hours']
            avg_conf = np.mean([m['confidence'] for m in data['speakers'] if not m.get('is_anchor')] or [0])
            logger.info(f"  {label}: {total_hours:.1f}h ({n_anchors} anchors + {n_propagated} propagated, avg_conf={avg_conf:.2f})")

        if len(sorted_labels) > 20:
            logger.info(f"  ... and {len(sorted_labels) - 20} more identities")

        # Label agreement analysis (helps understand anchor noise)
        logger.info("-" * 80)
        logger.info("LABEL AGREEMENT ANALYSIS (anchor noise indicator)")
        logger.info("-" * 80)

        # Count assignments by number of distinct labels in neighbors
        agreement_buckets = defaultdict(lambda: {'count': 0, 'hours': 0.0})
        for sp_id, r in results.items():
            if r.get('assigned'):
                n_labels = r.get('n_distinct_labels', 1)
                agreement_buckets[n_labels]['count'] += 1
                agreement_buckets[n_labels]['hours'] += (all_speakers[sp_id].get('duration', 0) or 0) / 3600

        total_assigned_hours = sum(b['hours'] for b in agreement_buckets.values())
        for n_labels in sorted(agreement_buckets.keys()):
            bucket = agreement_buckets[n_labels]
            pct_hours = (bucket['hours'] / total_assigned_hours * 100) if total_assigned_hours > 0 else 0
            if n_labels == 1:
                desc = "unanimous (all neighbors agree)"
            elif n_labels == 2:
                desc = "2 competing labels"
            else:
                desc = f"{n_labels} competing labels"
            logger.info(f"  {desc}: {bucket['count']} speakers, {bucket['hours']:.1f}h ({pct_hours:.1f}%)")

        # Show low-confidence assignments (potential noise)
        low_conf_assigned = [r for r in results.values() if r.get('assigned') and not r.get('is_high_confidence')]
        if low_conf_assigned:
            low_conf_hours = sum((all_speakers.get(sp_id, {}).get('duration', 0) or 0) / 3600
                                 for sp_id, r in results.items()
                                 if r.get('assigned') and not r.get('is_high_confidence'))
            logger.info(f"")
            logger.info(f"  ⚠ Low confidence assignments: {len(low_conf_assigned)} speakers, {low_conf_hours:.1f}h")
            logger.info(f"    (confidence {self.config.min_confidence_to_assign}-{self.config.high_confidence_threshold})")

        # Print summary
        self._print_summary()

        return self.stats

    def _print_summary(self):
        """Print summary statistics."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)

        # Speaker counts
        logger.info("SPEAKER COUNTS:")
        logger.info(f"  Total speakers: {self.stats['total_speakers']}")
        logger.info(f"  Unique names (anchors): {self.stats['unique_names']}")
        logger.info(f"  Name collisions detected: {self.stats['name_collisions_detected']}")
        logger.info(f"  Labeled speakers (anchors): {self.stats['labeled_speakers']}")
        logger.info(f"  Unlabeled speakers: {self.stats['unlabeled_speakers']}")
        logger.info(f"  Speakers assigned via propagation: {self.stats['speakers_assigned']}")
        logger.info(f"    - High confidence (>{self.config.high_confidence_threshold}): {self.stats['high_confidence_assignments']}")
        logger.info(f"    - Low confidence: {self.stats['low_confidence_assignments']}")
        logger.info(f"  Speakers unassigned: {self.stats['speakers_unassigned']}")

        # Speaking time metrics (the key goal: 90% coverage)
        logger.info("")
        logger.info("SPEAKING TIME COVERAGE:")
        logger.info(f"  Total speaking time: {self.stats['total_speaking_hours']:.1f} hours")
        logger.info(f"  Anchor speaking time: {self.stats['anchor_speaking_hours']:.1f} hours")
        logger.info(f"  Propagated speaking time: {self.stats['propagated_speaking_hours']:.1f} hours")
        logger.info(f"  ────────────────────────────────────")
        logger.info(f"  ASSIGNED SPEAKING TIME: {self.stats['assigned_speaking_hours']:.1f} hours")
        logger.info(f"  Unassigned speaking time: {self.stats['unassigned_speaking_hours']:.1f} hours")
        logger.info("")
        coverage = self.stats['speaking_time_coverage_pct']
        target = 90.0
        status = "✓ TARGET MET" if coverage >= target else f"✗ {target - coverage:.1f}% below target"
        logger.info(f"  >>> OVERALL COVERAGE: {coverage:.1f}% of speaking time labeled ({status})")

        # Stage 3b: High-duration speaker coverage
        logger.info("")
        logger.info(f"STAGE 3B - HIGH-DURATION SPEAKERS ({self.config.high_duration_pct*100:.0f}%+ of episode):")
        logger.info(f"  Total high-duration time: {self.stats['high_duration_total_hours']:.1f} hours")
        logger.info(f"  Assigned high-duration time: {self.stats['high_duration_assigned_hours']:.1f} hours")
        hd_coverage = self.stats['high_duration_coverage_pct']
        hd_status = "✓ TARGET MET" if hd_coverage >= target else f"✗ {target - hd_coverage:.1f}% below target"
        logger.info(f"  >>> HIGH-DURATION COVERAGE: {hd_coverage:.1f}% ({hd_status})")
        if self.stats['metadata_boost_assignments'] > 0:
            logger.info(f"  Metadata-boosted assignments: {self.stats['metadata_boost_assignments']}")
        if self.stats.get('direct_metadata_assignments', 0) > 0:
            logger.info(f"  Direct centroid assignments: {self.stats['direct_metadata_assignments']}")

        logger.info("")
        logger.info("IDENTITIES:")
        logger.info(f"  Identities created: {self.stats['identities_created']}")
        logger.info(f"  Identities updated: {self.stats['identities_updated']}")
        if self.stats['errors']:
            logger.warning(f"Errors: {len(self.stats['errors'])}")
        logger.info("=" * 80)


async def main():
    parser = argparse.ArgumentParser(
        description='Phase 3 Alternative: Single-Pass Label Propagation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run
  python -m src.speaker_identification.strategies.label_propagation_clustering --project CPRMV

  # Apply
  python -m src.speaker_identification.strategies.label_propagation_clustering \\
      --project CPRMV --apply

  # Test with limited anchors
  python -m src.speaker_identification.strategies.label_propagation_clustering \\
      --project CPRMV --max-anchors 25 --apply
"""
    )

    parser.add_argument('--project', type=str, help='Filter to project')
    parser.add_argument('--name', type=str, help='Filter to specific speaker name (for testing)')
    parser.add_argument('--apply', action='store_true', help='Apply changes (default: dry run)')
    parser.add_argument('--max-anchors', type=int, default=None,
                       help='Max unique names to process (for testing)')

    # Config overrides
    parser.add_argument('--k', type=int, default=75,
                       help='Number of neighbors for voting (default: 75)')
    parser.add_argument('--min-sim', type=float, default=0.55,
                       help='Minimum similarity for neighbor vote (default: 0.55)')
    parser.add_argument('--min-conf', type=float, default=0.30,
                       help='Minimum confidence to assign (default: 0.30)')
    parser.add_argument('--min-labeled', type=int, default=3,
                       help='Minimum labeled neighbors required (default: 3)')

    args = parser.parse_args()

    config = LabelPropagationConfig(
        k_neighbors=args.k,
        min_similarity=args.min_sim,
        min_confidence_to_assign=args.min_conf,
        min_labeled_neighbors=args.min_labeled
    )

    strategy = LabelPropagationStrategy(
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
