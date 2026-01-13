#!/usr/bin/env python3
"""
Identity Centroid Generation (Phase 4)
=======================================

Builds centroids for speaker identities that don't have one yet.
Primarily processes guests identified in Phase 3.

Process:
1. Find identities without centroids that have assigned speakers
2. For single-speaker identities: use that embedding as centroid
3. For multi-speaker identities: cluster to verify same person
   - Single cluster: build quality-weighted centroid
   - Multiple clusters: auto-split identity (name collision)

Usage:
    # Run on all identities needing centroids
    python -m src.speaker_identification.strategies.centroid_generation --apply

    # Run on specific project
    python -m src.speaker_identification.strategies.centroid_generation \
        --project CPRMV --apply

    # Run on single identity
    python -m src.speaker_identification.strategies.centroid_generation \
        --identity-id 55921 --apply
"""

import argparse
import asyncio
import sys
from pathlib import Path
from src.utils.paths import get_project_root, get_config_path
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime, timezone
from collections import defaultdict

import numpy as np
import faiss

project_root = str(get_project_root())
sys.path.append(project_root)

from src.database.session import get_session
from src.utils.logger import setup_worker_logger
from sqlalchemy import text
import json

logger = setup_worker_logger('speaker_identification.centroid_generation')

# Console logging
import logging
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)


class CentroidGenerationStrategy:
    """
    Phase 4: Build centroids for speaker identities without existing centroids.

    Processes identities that:
    - Have assigned speakers with embeddings
    - Do NOT have an existing centroid (skip Phase 2 hosts)

    For each identity:
    1. Load all assigned speaker embeddings
    2. Cluster to verify they're the same person
    3. Build quality-weighted centroid from largest cluster
    4. Auto-split if multiple clusters detected (name collision)
    """

    def __init__(
        self,
        cluster_threshold: float = 0.78,
        min_quality: float = 0.50,
        centroid_min_quality: float = 0.65,
        centroid_max_samples: int = 50,
        dry_run: bool = True
    ):
        """
        Initialize strategy.

        Args:
            cluster_threshold: Similarity threshold for clustering (0.78)
            min_quality: Minimum embedding quality to include (0.50)
            centroid_min_quality: Preferred quality for centroid (0.65)
            centroid_max_samples: Max samples for centroid averaging (50)
            dry_run: If True, don't make DB changes
        """
        self.cluster_threshold = cluster_threshold
        self.min_quality = min_quality
        self.centroid_min_quality = centroid_min_quality
        self.centroid_max_samples = centroid_max_samples
        self.dry_run = dry_run

        self.stats = {
            'identities_processed': 0,
            'identities_skipped_no_speakers': 0,
            'centroids_created': 0,
            'single_speaker_centroids': 0,
            'multi_speaker_centroids': 0,
            'identities_split': 0,
            'new_identities_from_splits': 0,
            'speakers_reassigned': 0,
            'errors': []
        }

    def _get_identities_needing_centroids(
        self,
        project: str = None,
        identity_id: int = None
    ) -> List[Dict]:
        """
        Get identities without centroids that have assigned speakers.

        Args:
            project: Filter to identities with speakers in this project
            identity_id: Specific identity ID to process

        Returns:
            List of identity dicts
        """
        with get_session() as session:
            if identity_id:
                query = text("""
                    SELECT
                        si.id,
                        si.primary_name,
                        si.role,
                        COUNT(s.id) as speaker_count
                    FROM speaker_identities si
                    JOIN speakers s ON s.speaker_identity_id = si.id
                    WHERE si.id = :identity_id
                      AND si.is_active = TRUE
                      AND (
                          si.verification_metadata IS NULL
                          OR si.verification_metadata->>'centroid' IS NULL
                      )
                      AND s.embedding IS NOT NULL
                    GROUP BY si.id
                    HAVING COUNT(s.id) > 0
                """)
                results = session.execute(query, {'identity_id': identity_id}).fetchall()
            elif project:
                query = text("""
                    SELECT
                        si.id,
                        si.primary_name,
                        si.role,
                        COUNT(DISTINCT s.id) as speaker_count
                    FROM speaker_identities si
                    JOIN speakers s ON s.speaker_identity_id = si.id
                    JOIN content c ON s.content_id = c.content_id
                    WHERE si.is_active = TRUE
                      AND (
                          si.verification_metadata IS NULL
                          OR si.verification_metadata->>'centroid' IS NULL
                      )
                      AND s.embedding IS NOT NULL
                      AND :project = ANY(c.projects)
                    GROUP BY si.id
                    HAVING COUNT(DISTINCT s.id) > 0
                    ORDER BY COUNT(DISTINCT s.id) DESC
                """)
                results = session.execute(query, {'project': project}).fetchall()
            else:
                query = text("""
                    SELECT
                        si.id,
                        si.primary_name,
                        si.role,
                        COUNT(s.id) as speaker_count
                    FROM speaker_identities si
                    JOIN speakers s ON s.speaker_identity_id = si.id
                    WHERE si.is_active = TRUE
                      AND (
                          si.verification_metadata IS NULL
                          OR si.verification_metadata->>'centroid' IS NULL
                      )
                      AND s.embedding IS NOT NULL
                    GROUP BY si.id
                    HAVING COUNT(s.id) > 0
                    ORDER BY COUNT(s.id) DESC
                """)
                results = session.execute(query).fetchall()

            return [dict(row._mapping) for row in results]

    def _get_assigned_speakers(self, identity_id: int) -> List[Dict]:
        """
        Get all speakers assigned to an identity with their embeddings.

        Args:
            identity_id: Speaker identity ID

        Returns:
            List of speaker dicts with id, content_id, embedding, quality
        """
        with get_session() as session:
            query = text("""
                SELECT
                    s.id,
                    s.content_id,
                    s.embedding,
                    COALESCE(s.embedding_quality_score, 0.5) as quality,
                    s.duration
                FROM speakers s
                WHERE s.speaker_identity_id = :identity_id
                  AND s.embedding IS NOT NULL
                  AND COALESCE(s.embedding_quality_score, 0.5) >= :min_quality
                ORDER BY s.embedding_quality_score DESC NULLS LAST
            """)
            results = session.execute(query, {
                'identity_id': identity_id,
                'min_quality': self.min_quality
            }).fetchall()

            speakers = []
            for row in results:
                # Handle embedding as string or list
                emb_data = row.embedding
                if isinstance(emb_data, str):
                    emb_data = json.loads(emb_data)
                embedding = np.array(emb_data, dtype=np.float32)
                # Normalize
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                speakers.append({
                    'id': row.id,
                    'content_id': row.content_id,
                    'embedding': embedding,
                    'quality': row.quality,
                    'duration': row.duration
                })

            return speakers

    def _cluster_speakers(
        self,
        speakers: List[Dict],
        embeddings: np.ndarray
    ) -> List[Dict]:
        """
        Cluster speakers by embedding similarity.

        Uses FAISS for efficient similarity search and union-find for clustering.

        Args:
            speakers: List of speaker dicts
            embeddings: Normalized embedding matrix (n_speakers x 512)

        Returns:
            List of clusters: [{'members': [indices], 'episode_count': int}]
        """
        n_speakers = len(embeddings)
        if n_speakers < 2:
            return [{'members': list(range(n_speakers)), 'episode_count': n_speakers}]

        # Build FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        # Find neighbors above threshold
        K = min(50, n_speakers)
        similarities, indices = index.search(embeddings, K)

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

        # Merge speakers above threshold
        for i in range(n_speakers):
            for j, sim in zip(indices[i], similarities[i]):
                if i != j and sim >= self.cluster_threshold:
                    union(i, j)

        # Build clusters
        cluster_members = defaultdict(list)
        for i in range(n_speakers):
            cluster_members[find(i)].append(i)

        # Convert to list format
        clusters = []
        for members in cluster_members.values():
            # Count unique episodes
            episode_ids = set(speakers[m]['content_id'] for m in members)
            clusters.append({
                'members': members,
                'episode_count': len(episode_ids)
            })

        # Sort by size descending
        clusters.sort(key=lambda c: len(c['members']), reverse=True)

        return clusters

    def _build_centroid_from_cluster(
        self,
        speakers: List[Dict],
        embeddings: np.ndarray,
        cluster: Dict
    ) -> Tuple[np.ndarray, float, int]:
        """
        Build quality-weighted centroid from cluster members.

        Args:
            speakers: Full speaker list
            embeddings: Full embedding matrix
            cluster: Cluster dict with 'members' list

        Returns:
            (centroid, avg_quality, sample_count)
        """
        members = cluster['members']

        # Get qualities and sort by quality descending
        idx_quality = [(idx, speakers[idx]['quality']) for idx in members]
        idx_quality.sort(key=lambda x: x[1], reverse=True)

        # Select high-quality embeddings, cap at max_samples
        selected = []
        for idx, quality in idx_quality:
            if quality >= self.centroid_min_quality or len(selected) < 5:
                selected.append((idx, quality))
            if len(selected) >= self.centroid_max_samples:
                break

        # Extract embeddings and qualities
        selected_indices = [s[0] for s in selected]
        selected_qualities = np.array([s[1] for s in selected])
        selected_embeddings = embeddings[selected_indices]

        # Compute quality-weighted centroid
        weights = selected_qualities / np.sum(selected_qualities)
        centroid = np.average(selected_embeddings, axis=0, weights=weights)
        centroid = centroid / np.linalg.norm(centroid)
        avg_quality = float(np.mean(selected_qualities))

        return centroid, avg_quality, len(selected)

    def _save_centroid(
        self,
        identity_id: int,
        centroid: np.ndarray,
        quality: float,
        sample_count: int
    ):
        """Save centroid to speaker_identity verification_metadata."""
        if self.dry_run:
            return

        with get_session() as session:
            # Get existing metadata
            result = session.execute(
                text("SELECT verification_metadata FROM speaker_identities WHERE id = :id"),
                {'id': identity_id}
            ).fetchone()

            metadata = result.verification_metadata if result and result.verification_metadata else {}

            # Update with centroid data
            metadata['centroid'] = centroid.tolist()
            metadata['centroid_quality'] = quality
            metadata['centroid_sample_count'] = sample_count
            metadata['centroid_updated_at'] = datetime.now(timezone.utc).isoformat()

            session.execute(
                text("""
                    UPDATE speaker_identities
                    SET verification_metadata = :metadata,
                        updated_at = NOW()
                    WHERE id = :id
                """),
                {'id': identity_id, 'metadata': json.dumps(metadata)}
            )
            session.commit()

    def _create_split_identity(
        self,
        original_identity: Dict,
        new_name: str
    ) -> int:
        """
        Create a new identity for a split cluster.

        Args:
            original_identity: Original identity dict
            new_name: Name for new identity (e.g., "John Smith (2)")

        Returns:
            New identity ID
        """
        if self.dry_run:
            return -1  # Placeholder for dry run

        with get_session() as session:
            result = session.execute(
                text("""
                    INSERT INTO speaker_identities (
                        primary_name,
                        role,
                        verification_status,
                        verification_metadata,
                        is_active,
                        created_at,
                        updated_at
                    ) VALUES (
                        :name,
                        :role,
                        'llm_identified',
                        :metadata,
                        TRUE,
                        NOW(),
                        NOW()
                    )
                    RETURNING id
                """),
                {
                    'name': new_name,
                    'role': original_identity.get('role', 'guest'),
                    'metadata': json.dumps({
                        'split_from': original_identity['id'],
                        'split_from_name': original_identity['primary_name'],
                        'split_at': datetime.now(timezone.utc).isoformat()
                    })
                }
            )
            new_id = result.fetchone()[0]
            session.commit()

            logger.info(f"Created split identity: {new_name} (ID: {new_id})")
            return new_id

    def _reassign_speakers(
        self,
        speaker_ids: List[int],
        new_identity_id: int
    ):
        """
        Reassign speakers to a new identity.

        Args:
            speaker_ids: List of speaker IDs to reassign
            new_identity_id: New identity ID
        """
        if self.dry_run:
            return

        with get_session() as session:
            session.execute(
                text("""
                    UPDATE speakers
                    SET speaker_identity_id = :identity_id,
                        assignment_method = 'centroid_split',
                        updated_at = NOW()
                    WHERE id = ANY(:speaker_ids)
                """),
                {
                    'identity_id': new_identity_id,
                    'speaker_ids': speaker_ids
                }
            )
            session.commit()

        self.stats['speakers_reassigned'] += len(speaker_ids)

    def _process_identity(self, identity: Dict) -> bool:
        """
        Process a single identity: build centroid, handle splits.

        Args:
            identity: Identity dict with id, primary_name, role, speaker_count

        Returns:
            True if centroid created, False otherwise
        """
        identity_id = identity['id']
        name = identity['primary_name']

        # Get speakers
        speakers = self._get_assigned_speakers(identity_id)
        if not speakers:
            self.stats['identities_skipped_no_speakers'] += 1
            return False

        logger.info(f"Processing: {name} (ID: {identity_id}, {len(speakers)} speakers)")

        if len(speakers) == 1:
            # Single speaker - use its embedding
            centroid = speakers[0]['embedding']
            quality = speakers[0]['quality']
            self._save_centroid(identity_id, centroid, quality, 1)
            self.stats['single_speaker_centroids'] += 1
            self.stats['centroids_created'] += 1
            logger.info(f"  Single-speaker centroid (quality: {quality:.3f})")
            return True

        # Multiple speakers - cluster
        embeddings = np.array([s['embedding'] for s in speakers], dtype=np.float32)
        clusters = self._cluster_speakers(speakers, embeddings)

        if len(clusters) == 1:
            # All speakers cluster together - good
            centroid, quality, sample_count = self._build_centroid_from_cluster(
                speakers, embeddings, clusters[0]
            )
            self._save_centroid(identity_id, centroid, quality, sample_count)
            self.stats['multi_speaker_centroids'] += 1
            self.stats['centroids_created'] += 1
            logger.info(
                f"  Multi-speaker centroid: {sample_count} samples, "
                f"quality: {quality:.3f}"
            )
            return True

        # Multiple clusters - auto-split
        logger.warning(
            f"  Name collision detected: {len(clusters)} clusters for '{name}'"
        )
        self.stats['identities_split'] += 1

        # Largest cluster keeps original identity
        main_cluster = clusters[0]
        centroid, quality, sample_count = self._build_centroid_from_cluster(
            speakers, embeddings, main_cluster
        )
        self._save_centroid(identity_id, centroid, quality, sample_count)
        self.stats['centroids_created'] += 1
        logger.info(
            f"  Main cluster: {len(main_cluster['members'])} speakers → "
            f"keeps '{name}'"
        )

        # Create new identities for outlier clusters
        for i, outlier_cluster in enumerate(clusters[1:], start=2):
            new_name = f"{name} ({i})"
            new_identity_id = self._create_split_identity(identity, new_name)

            if new_identity_id > 0:  # Not dry run
                # Build centroid for new identity
                centroid, quality, sample_count = self._build_centroid_from_cluster(
                    speakers, embeddings, outlier_cluster
                )
                self._save_centroid(new_identity_id, centroid, quality, sample_count)

                # Reassign speakers
                speaker_ids = [speakers[m]['id'] for m in outlier_cluster['members']]
                self._reassign_speakers(speaker_ids, new_identity_id)

            self.stats['new_identities_from_splits'] += 1
            self.stats['centroids_created'] += 1
            logger.info(
                f"  Split cluster {i}: {len(outlier_cluster['members'])} speakers → "
                f"'{new_name}'"
            )

        return True

    async def run(
        self,
        project: str = None,
        identity_id: int = None
    ) -> Dict:
        """
        Run centroid generation.

        Args:
            project: Filter to project
            identity_id: Process single identity

        Returns:
            Stats dict
        """
        logger.info("=" * 80)
        logger.info("IDENTITY CENTROID GENERATION (Phase 4)")
        logger.info("=" * 80)
        logger.info(f"Mode: {'DRY RUN' if self.dry_run else 'APPLY'}")
        logger.info(f"Cluster threshold: {self.cluster_threshold}")
        logger.info(f"Min quality: {self.min_quality}")
        if project:
            logger.info(f"Project: {project}")
        if identity_id:
            logger.info(f"Identity ID: {identity_id}")
        logger.info("-" * 80)

        # Get identities
        identities = self._get_identities_needing_centroids(project, identity_id)
        logger.info(f"Found {len(identities)} identities needing centroids")
        logger.info("-" * 80)

        # Process each identity
        for identity in identities:
            self.stats['identities_processed'] += 1
            try:
                self._process_identity(identity)
            except Exception as e:
                logger.error(f"Error processing identity {identity['id']}: {e}")
                self.stats['errors'].append(f"{identity['id']}: {str(e)}")

        # Summary
        self._print_summary()
        return self.stats

    def _print_summary(self):
        """Print summary statistics."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Identities processed: {self.stats['identities_processed']}")
        logger.info(f"Identities skipped (no speakers): {self.stats['identities_skipped_no_speakers']}")
        logger.info(f"Centroids created: {self.stats['centroids_created']}")
        logger.info(f"  - Single-speaker: {self.stats['single_speaker_centroids']}")
        logger.info(f"  - Multi-speaker: {self.stats['multi_speaker_centroids']}")
        logger.info(f"Identities split (name collision): {self.stats['identities_split']}")
        logger.info(f"New identities from splits: {self.stats['new_identities_from_splits']}")
        logger.info(f"Speakers reassigned: {self.stats['speakers_reassigned']}")
        if self.stats['errors']:
            logger.warning(f"Errors: {len(self.stats['errors'])}")
        logger.info("=" * 80)


async def main():
    parser = argparse.ArgumentParser(
        description='Phase 4: Identity Centroid Generation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run on all identities needing centroids
  python -m src.speaker_identification.strategies.centroid_generation

  # Apply to specific project
  python -m src.speaker_identification.strategies.centroid_generation \\
      --project CPRMV --apply

  # Process single identity
  python -m src.speaker_identification.strategies.centroid_generation \\
      --identity-id 55921 --apply
"""
    )

    parser.add_argument('--project', type=str, help='Filter to project')
    parser.add_argument('--identity-id', type=int, help='Process single identity')
    parser.add_argument('--apply', action='store_true', help='Apply changes (default: dry run)')
    parser.add_argument('--cluster-threshold', type=float, default=0.78,
                       help='Clustering threshold (default: 0.78)')
    parser.add_argument('--min-quality', type=float, default=0.50,
                       help='Min embedding quality (default: 0.50)')

    args = parser.parse_args()

    strategy = CentroidGenerationStrategy(
        cluster_threshold=args.cluster_threshold,
        min_quality=args.min_quality,
        dry_run=not args.apply
    )

    try:
        await strategy.run(
            project=args.project,
            identity_id=args.identity_id
        )
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
