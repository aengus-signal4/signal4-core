#!/usr/bin/env python3
"""
Identity Merge Detection (Phase 5)
===================================

Detects and merges duplicate speaker identities that were created separately
due to name variations, transcription errors, or independent identification
across different episodes/channels.

Process:
1. Build FAISS index of all identity centroids
2. Find high-similarity pairs (>= 0.85 threshold)
3. Gather evidence: name similarity, co-appearance check, transcript samples
4. LLM verification to determine if same person
5. Execute merges for confirmed duplicates

Safety Mechanisms:
- Co-appearance veto: Never merge identities that appear in same episode
- Dry-run by default: Preview merges before applying
- Audit trail: All merges recorded with LLM reasoning

Usage:
    # Dry run on all identities
    python -m src.speaker_identification.strategies.identity_merge_detection

    # Run on specific project
    python -m src.speaker_identification.strategies.identity_merge_detection \\
        --project CPRMV --apply

    # Adjust threshold
    python -m src.speaker_identification.strategies.identity_merge_detection \\
        --threshold 0.90 --apply
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from src.utils.paths import get_project_root, get_config_path
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict

import faiss
import numpy as np
from sqlalchemy import text
from tqdm import tqdm

project_root = str(get_project_root())
sys.path.append(project_root)

from src.speaker_identification.core.llm_client import MLXLLMClient
from src.speaker_identification.core.context_builder import ContextBuilder
from src.speaker_identification.prompts import PromptRegistry
from src.database.session import get_session
from src.utils.logger import setup_worker_logger

logger = setup_worker_logger('speaker_identification.identity_merge_detection')

# Console logging
import logging
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

# Default thresholds
CENTROID_SIMILARITY_THRESHOLD = 0.85  # Conservative threshold for merge consideration
MIN_CONFIDENCE_TO_MERGE = ['certain', 'very_likely']  # LLM confidence required
MAX_CANDIDATES_PER_RUN = 100  # Limit LLM calls per run


class IdentityMergeDetectionStrategy:
    """
    Phase 5: Detect and merge duplicate speaker identities.

    Compares all identity centroids using FAISS, identifies high-similarity
    pairs, gathers evidence, and uses LLM to verify if they're the same person.
    """

    def __init__(
        self,
        similarity_threshold: float = CENTROID_SIMILARITY_THRESHOLD,
        max_candidates: int = MAX_CANDIDATES_PER_RUN,
        dry_run: bool = True
    ):
        """
        Initialize strategy.

        Args:
            similarity_threshold: Minimum centroid similarity to consider merge (default 0.85)
            max_candidates: Maximum candidate pairs to process per run
            dry_run: If True, don't make DB changes
        """
        self.similarity_threshold = similarity_threshold
        self.max_candidates = max_candidates
        self.dry_run = dry_run

        self.llm_client = MLXLLMClient()
        self.context_builder = ContextBuilder()

        self.stats = {
            'identities_with_centroids': 0,
            'candidate_pairs_found': 0,
            'pairs_processed': 0,
            'co_appearance_vetoed': 0,
            'llm_confirmed_same': 0,
            'llm_rejected_different': 0,
            'merges_executed': 0,
            'speakers_reassigned': 0,
            'centroids_rebuilt': 0,
            'errors': []
        }

    def _load_all_centroids(self) -> List[Dict]:
        """Load all active identity centroids with metadata."""
        with get_session() as session:
            query = text("""
                SELECT
                    si.id as identity_id,
                    si.primary_name,
                    si.role,
                    si.verification_metadata,
                    si.created_at,
                    COUNT(DISTINCT s.id) as speaker_count,
                    COUNT(DISTINCT s.content_id) as episode_count,
                    SUM(s.duration) as total_duration,
                    ARRAY_AGG(DISTINCT ch.display_name) FILTER (WHERE ch.display_name IS NOT NULL) as channels
                FROM speaker_identities si
                LEFT JOIN speakers s ON s.speaker_identity_id = si.id
                LEFT JOIN content c ON s.content_id = c.content_id
                LEFT JOIN channels ch ON c.channel_id = ch.id
                WHERE si.is_active = TRUE
                  AND si.verification_metadata ? 'centroid'
                GROUP BY si.id
            """)
            results = session.execute(query).fetchall()

            centroids = []
            for row in results:
                metadata = row.verification_metadata or {}
                centroid_list = metadata.get('centroid')
                if centroid_list:
                    centroids.append({
                        'identity_id': row.identity_id,
                        'name': row.primary_name,
                        'role': row.role,
                        'centroid': np.array(centroid_list, dtype=np.float32),
                        'quality': metadata.get('centroid_quality', 0.0),
                        'sample_count': metadata.get('centroid_sample_count', 0),
                        'speaker_count': row.speaker_count or 0,
                        'episode_count': row.episode_count or 0,
                        'total_duration': row.total_duration or 0,
                        'channels': row.channels or [],
                        'created_at': row.created_at
                    })

            return centroids

    def _build_faiss_index(self, centroids: List[Dict]) -> Tuple[faiss.IndexFlatIP, np.ndarray]:
        """Build FAISS index from centroids for similarity search."""
        if not centroids:
            return None, None

        # Extract and normalize centroids
        centroid_matrix = np.array([c['centroid'] for c in centroids], dtype=np.float32)
        faiss.normalize_L2(centroid_matrix)

        # Build inner product index
        dim = centroid_matrix.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(centroid_matrix)

        return index, centroid_matrix

    def _find_merge_candidates(
        self,
        centroids: List[Dict],
        faiss_index: faiss.IndexFlatIP,
        centroid_matrix: np.ndarray
    ) -> List[Tuple[int, int, float]]:
        """
        Find all centroid pairs above similarity threshold.

        Returns:
            List of (identity_id_a, identity_id_b, similarity) tuples
        """
        n = len(centroids)
        k = min(10, n)  # Search top-10 neighbors

        # Query each centroid against all others
        similarities, indices = faiss_index.search(centroid_matrix, k)

        # Collect pairs above threshold (avoid duplicates with i < j)
        candidates = []
        seen_pairs = set()

        for i in range(n):
            for j_idx in range(k):
                j = int(indices[i][j_idx])
                sim = float(similarities[i][j_idx])

                # Skip self and already-seen pairs
                if i == j:
                    continue

                pair_key = (min(i, j), max(i, j))
                if pair_key in seen_pairs:
                    continue

                if sim >= self.similarity_threshold:
                    seen_pairs.add(pair_key)
                    candidates.append((
                        centroids[i]['identity_id'],
                        centroids[j]['identity_id'],
                        sim
                    ))

        # Sort by similarity descending
        candidates.sort(key=lambda x: x[2], reverse=True)

        return candidates

    def _check_co_appearance(self, identity_a: int, identity_b: int) -> bool:
        """
        Check if two identities appear in the same episode.

        If they do, they CANNOT be the same person (safety veto).

        Returns:
            True if they co-appear (should NOT merge), False otherwise
        """
        with get_session() as session:
            query = text("""
                SELECT COUNT(*) as overlap
                FROM speakers s1
                JOIN speakers s2 ON s1.content_id = s2.content_id
                WHERE s1.speaker_identity_id = :id_a
                  AND s2.speaker_identity_id = :id_b
            """)
            result = session.execute(query, {'id_a': identity_a, 'id_b': identity_b}).fetchone()
            return result.overlap > 0

    def _get_transcript_samples(self, identity_id: int, max_samples: int = 3) -> List[Dict]:
        """
        Get transcript samples for an identity.

        Prioritizes self-introductions and high-quality samples.
        """
        with get_session() as session:
            # Get speakers for this identity, ordered by quality
            query = text("""
                SELECT
                    s.id as speaker_id,
                    s.content_id,
                    c.title as episode_title,
                    s.duration,
                    c.duration as ep_duration,
                    s.embedding_quality_score as quality
                FROM speakers s
                JOIN content c ON s.content_id = c.content_id
                WHERE s.speaker_identity_id = :identity_id
                  AND c.is_stitched = TRUE
                ORDER BY s.embedding_quality_score DESC NULLS LAST
                LIMIT :max_samples
            """)
            speakers = session.execute(query, {
                'identity_id': identity_id,
                'max_samples': max_samples
            }).fetchall()

        samples = []
        for speaker in speakers:
            context = self.context_builder.get_speaker_transcript_context(speaker.speaker_id)
            if context:
                duration_pct = (speaker.duration / speaker.ep_duration * 100) if speaker.ep_duration else 0
                samples.append({
                    'episode_title': speaker.episode_title,
                    'duration_pct': duration_pct,
                    'first_utterance': context.get('first_utterance', '')[:500],
                    'last_utterance': context.get('last_utterance', '')[:500],
                    'total_turns': context.get('total_turns', 0)
                })

        return samples

    def _get_identity_metadata(self, identity_id: int) -> Dict:
        """Get full metadata for an identity."""
        with get_session() as session:
            query = text("""
                SELECT
                    si.id,
                    si.primary_name,
                    si.role,
                    si.verification_metadata,
                    si.created_at,
                    COUNT(DISTINCT s.id) as speaker_count,
                    COUNT(DISTINCT s.content_id) as episode_count,
                    COALESCE(SUM(s.duration), 0) as total_duration,
                    ARRAY_AGG(DISTINCT ch.display_name) FILTER (WHERE ch.display_name IS NOT NULL) as channels,
                    MIN(c.publish_date) as first_appearance
                FROM speaker_identities si
                LEFT JOIN speakers s ON s.speaker_identity_id = si.id
                LEFT JOIN content c ON s.content_id = c.content_id
                LEFT JOIN channels ch ON c.channel_id = ch.id
                WHERE si.id = :identity_id
                GROUP BY si.id
            """)
            row = session.execute(query, {'identity_id': identity_id}).fetchone()

            if not row:
                return {}

            return {
                'id': row.id,
                'name': row.primary_name,
                'role': row.role,
                'speaker_count': row.speaker_count or 0,
                'episode_count': row.episode_count or 0,
                'total_duration': row.total_duration or 0,
                'channels': row.channels or [],
                'first_appearance': row.first_appearance,
                'verification_metadata': row.verification_metadata or {}
            }

    async def _verify_merge_with_llm(
        self,
        identity_a: Dict,
        identity_b: Dict,
        similarity: float,
        samples_a: List[Dict],
        samples_b: List[Dict]
    ) -> Dict:
        """
        Use LLM to determine if two identities are the same person.

        Returns:
            Dict with: is_same_person, confidence, keep_identity, canonical_name, reasoning
        """
        # Format transcript samples
        def format_samples(samples: List[Dict]) -> str:
            if not samples:
                return "  (No transcript samples available)"
            text = ""
            for i, s in enumerate(samples, 1):
                text += f"\n  Sample {i} (from \"{s['episode_title']}\"):\n"
                text += f"    Duration: {s['duration_pct']:.1f}% of episode, {s['total_turns']} turns\n"
                text += f"    First: \"{s['first_utterance'][:200]}...\"\n"
                text += f"    Last: \"{s['last_utterance'][:200]}...\"\n"
            return text

        samples_a_text = format_samples(samples_a)
        samples_b_text = format_samples(samples_b)

        # Format channels
        channels_a = ", ".join(identity_a.get('channels', [])[:5]) or "Unknown"
        channels_b = ", ".join(identity_b.get('channels', [])[:5]) or "Unknown"

        # Format duration
        def format_duration(seconds):
            hours = seconds / 3600
            return f"{hours:.1f} hours"

        # Build prompt
        prompt = PromptRegistry.phase4_identity_merge_verification(
            name_a=identity_a['name'],
            count_a=identity_a['speaker_count'],
            episode_count_a=identity_a['episode_count'],
            duration_a=format_duration(identity_a['total_duration']),
            channels_a=channels_a,
            roles_a=identity_a.get('role', 'unknown'),
            first_appearance_a=str(identity_a.get('first_appearance', 'Unknown')),
            samples_a=samples_a_text,
            name_b=identity_b['name'],
            count_b=identity_b['speaker_count'],
            episode_count_b=identity_b['episode_count'],
            duration_b=format_duration(identity_b['total_duration']),
            channels_b=channels_b,
            roles_b=identity_b.get('role', 'unknown'),
            first_appearance_b=str(identity_b.get('first_appearance', 'Unknown')),
            samples_b=samples_b_text,
            similarity=similarity
        )

        # Call LLM
        response = await self.llm_client._call_llm(prompt, priority=3)

        # Parse response
        try:
            response = response.strip()
            if response.startswith("```"):
                lines = response.split("\n")
                response = "\n".join(lines[1:-1])
            data = json.loads(response)

            return {
                'is_same_person': data.get('is_same_person', False),
                'confidence': data.get('confidence', 'unlikely'),
                'keep_identity': data.get('keep_identity'),
                'canonical_name': data.get('canonical_name'),
                'reasoning': data.get('reasoning', ''),
                'evidence': data.get('evidence', {})
            }
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            return {
                'is_same_person': False,
                'confidence': 'unlikely',
                'keep_identity': None,
                'canonical_name': None,
                'reasoning': f'JSON parse error: {e}',
                'evidence': {}
            }

    def _execute_merge(
        self,
        keep_identity: Dict,
        merge_identity: Dict,
        canonical_name: Optional[str],
        llm_result: Dict
    ):
        """
        Execute the merge: reassign speakers and deactivate merged identity.

        Args:
            keep_identity: Identity to keep (dict with 'id' and 'name')
            merge_identity: Identity to merge away (dict with 'id' and 'name')
            canonical_name: Optional corrected name from LLM
            llm_result: Full LLM verification result for audit trail
        """
        if self.dry_run:
            return

        with get_session() as session:
            keep_id = keep_identity['id']
            merge_id = merge_identity['id']

            # 1. Reassign all speakers from merge -> keep
            result = session.execute(
                text("""
                    UPDATE speakers
                    SET speaker_identity_id = :keep_id,
                        assignment_method = 'identity_merge',
                        updated_at = NOW()
                    WHERE speaker_identity_id = :merge_id
                    RETURNING id
                """),
                {'keep_id': keep_id, 'merge_id': merge_id}
            )
            reassigned_count = len(result.fetchall())
            self.stats['speakers_reassigned'] += reassigned_count

            # 2. Update canonical name if suggested
            if canonical_name and canonical_name != keep_identity['name']:
                session.execute(
                    text("""
                        UPDATE speaker_identities
                        SET primary_name = :name,
                            updated_at = NOW()
                        WHERE id = :id
                    """),
                    {'id': keep_id, 'name': canonical_name}
                )

            # 3. Record merge in verification_metadata (audit trail)
            session.execute(
                text("""
                    UPDATE speaker_identities
                    SET verification_metadata = verification_metadata || :merge_record,
                        updated_at = NOW()
                    WHERE id = :id
                """),
                {
                    'id': keep_id,
                    'merge_record': json.dumps({
                        'merged_identities': [{
                            'id': merge_id,
                            'name': merge_identity['name'],
                            'merged_at': datetime.now(timezone.utc).isoformat(),
                            'reasoning': llm_result.get('reasoning', ''),
                            'confidence': llm_result.get('confidence', '')
                        }]
                    })
                }
            )

            # 4. Deactivate merged identity
            session.execute(
                text("""
                    UPDATE speaker_identities
                    SET is_active = FALSE,
                        verification_metadata = verification_metadata || :deactivate_record,
                        updated_at = NOW()
                    WHERE id = :id
                """),
                {
                    'id': merge_id,
                    'deactivate_record': json.dumps({
                        'merged_into': keep_id,
                        'merged_at': datetime.now(timezone.utc).isoformat(),
                        'reasoning': llm_result.get('reasoning', '')
                    })
                }
            )

            session.commit()
            self.stats['merges_executed'] += 1

    def _rebuild_centroid(self, identity_id: int):
        """Rebuild centroid for identity after merge (using same logic as Phase 4)."""
        # Import here to avoid circular dependency
        from src.speaker_identification.strategies.centroid_generation import CentroidGenerationStrategy

        strategy = CentroidGenerationStrategy(dry_run=self.dry_run)

        # Get identity info
        with get_session() as session:
            result = session.execute(
                text("SELECT id, primary_name, role FROM speaker_identities WHERE id = :id"),
                {'id': identity_id}
            ).fetchone()

            if result:
                identity = {
                    'id': result.id,
                    'primary_name': result.primary_name,
                    'role': result.role
                }
                # Clear existing centroid so it gets rebuilt
                session.execute(
                    text("""
                        UPDATE speaker_identities
                        SET verification_metadata = verification_metadata - 'centroid' - 'centroid_quality' - 'centroid_sample_count',
                            updated_at = NOW()
                        WHERE id = :id
                    """),
                    {'id': identity_id}
                )
                session.commit()

        # Process identity to rebuild centroid
        if result:
            strategy._process_identity(identity)
            self.stats['centroids_rebuilt'] += 1

    async def run(self, project: str = None) -> Dict:
        """
        Run identity merge detection.

        Args:
            project: Optional project filter (uses identities with speakers in this project)

        Returns:
            Stats dict
        """
        logger.info("=" * 80)
        logger.info("IDENTITY MERGE DETECTION (Phase 5)")
        logger.info("=" * 80)
        logger.info(f"Mode: {'DRY RUN' if self.dry_run else 'APPLY'}")
        logger.info(f"Similarity threshold: {self.similarity_threshold}")
        logger.info(f"Max candidates per run: {self.max_candidates}")
        if project:
            logger.info(f"Project: {project}")
        logger.info("-" * 80)

        # Load all centroids
        centroids = self._load_all_centroids()
        self.stats['identities_with_centroids'] = len(centroids)
        logger.info(f"Loaded {len(centroids)} identities with centroids")

        if len(centroids) < 2:
            logger.info("Need at least 2 identities to check for merges")
            return self.stats

        # Build FAISS index
        faiss_index, centroid_matrix = self._build_faiss_index(centroids)
        logger.info(f"Built FAISS index (dim={centroid_matrix.shape[1]})")

        # Find merge candidates
        candidates = self._find_merge_candidates(centroids, faiss_index, centroid_matrix)
        self.stats['candidate_pairs_found'] = len(candidates)
        logger.info(f"Found {len(candidates)} candidate pairs above threshold {self.similarity_threshold}")

        if not candidates:
            logger.info("No merge candidates found")
            return self.stats

        # Limit candidates
        if len(candidates) > self.max_candidates:
            logger.info(f"Limiting to top {self.max_candidates} candidates")
            candidates = candidates[:self.max_candidates]

        # Build lookup for centroids
        centroid_by_id = {c['identity_id']: c for c in centroids}

        # Process each candidate pair
        logger.info("-" * 80)
        logger.info(f"Processing {len(candidates)} candidate pairs...")

        for id_a, id_b, similarity in tqdm(candidates, desc="Merge candidates"):
            self.stats['pairs_processed'] += 1

            centroid_a = centroid_by_id.get(id_a)
            centroid_b = centroid_by_id.get(id_b)

            if not centroid_a or not centroid_b:
                continue

            name_a = centroid_a['name']
            name_b = centroid_b['name']

            tqdm.write(f"\nCandidate: {name_a} (ID:{id_a}) vs {name_b} (ID:{id_b}) | sim={similarity:.3f}")

            # Safety check: co-appearance veto
            if self._check_co_appearance(id_a, id_b):
                tqdm.write(f"  ✗ VETO: Co-appear in same episode - cannot be same person")
                self.stats['co_appearance_vetoed'] += 1
                continue

            # Gather evidence
            identity_a = self._get_identity_metadata(id_a)
            identity_b = self._get_identity_metadata(id_b)
            samples_a = self._get_transcript_samples(id_a)
            samples_b = self._get_transcript_samples(id_b)

            # LLM verification
            try:
                result = await self._verify_merge_with_llm(
                    identity_a, identity_b, similarity,
                    samples_a, samples_b
                )

                is_same = result.get('is_same_person', False)
                confidence = result.get('confidence', 'unlikely')
                keep = result.get('keep_identity')
                canonical_name = result.get('canonical_name')
                reasoning = result.get('reasoning', '')[:100]

                if is_same and confidence in MIN_CONFIDENCE_TO_MERGE:
                    self.stats['llm_confirmed_same'] += 1
                    tqdm.write(f"  ✓ SAME PERSON ({confidence}): {reasoning}")

                    # Determine which to keep
                    if keep == 'A':
                        keep_identity = identity_a
                        merge_identity = identity_b
                    else:
                        keep_identity = identity_b
                        merge_identity = identity_a

                    tqdm.write(f"    Keep: {keep_identity['name']} (ID:{keep_identity['id']})")
                    tqdm.write(f"    Merge: {merge_identity['name']} (ID:{merge_identity['id']})")

                    if canonical_name:
                        tqdm.write(f"    Canonical name: {canonical_name}")

                    # Execute merge
                    self._execute_merge(keep_identity, merge_identity, canonical_name, result)

                    # Rebuild centroid for kept identity
                    if not self.dry_run:
                        self._rebuild_centroid(keep_identity['id'])

                else:
                    self.stats['llm_rejected_different'] += 1
                    tqdm.write(f"  ✗ DIFFERENT ({confidence}): {reasoning}")

            except Exception as e:
                logger.error(f"Error processing pair ({id_a}, {id_b}): {e}")
                self.stats['errors'].append(str(e))

        self._print_summary()
        return self.stats

    def _print_summary(self):
        """Print summary statistics."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Identities with centroids: {self.stats['identities_with_centroids']}")
        logger.info(f"Candidate pairs found: {self.stats['candidate_pairs_found']}")
        logger.info(f"Pairs processed: {self.stats['pairs_processed']}")
        logger.info(f"Co-appearance vetoed: {self.stats['co_appearance_vetoed']}")
        logger.info(f"LLM confirmed same person: {self.stats['llm_confirmed_same']}")
        logger.info(f"LLM rejected (different): {self.stats['llm_rejected_different']}")
        logger.info(f"Merges executed: {self.stats['merges_executed']}")
        logger.info(f"Speakers reassigned: {self.stats['speakers_reassigned']}")
        logger.info(f"Centroids rebuilt: {self.stats['centroids_rebuilt']}")
        if self.stats['errors']:
            logger.warning(f"Errors: {len(self.stats['errors'])}")
        logger.info("=" * 80)


async def main():
    parser = argparse.ArgumentParser(
        description='Phase 5: Identity Merge Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run on all identities
  python -m src.speaker_identification.strategies.identity_merge_detection

  # Apply merges on specific project
  python -m src.speaker_identification.strategies.identity_merge_detection \\
      --project CPRMV --apply

  # Higher threshold (more conservative)
  python -m src.speaker_identification.strategies.identity_merge_detection \\
      --threshold 0.90 --apply
"""
    )

    parser.add_argument('--project', type=str, help='Filter to project')
    parser.add_argument('--apply', action='store_true', help='Apply changes (default: dry run)')
    parser.add_argument('--threshold', type=float, default=CENTROID_SIMILARITY_THRESHOLD,
                       help=f'Centroid similarity threshold (default: {CENTROID_SIMILARITY_THRESHOLD})')
    parser.add_argument('--max-candidates', type=int, default=MAX_CANDIDATES_PER_RUN,
                       help=f'Max candidate pairs to process (default: {MAX_CANDIDATES_PER_RUN})')

    args = parser.parse_args()

    strategy = IdentityMergeDetectionStrategy(
        similarity_threshold=args.threshold,
        max_candidates=args.max_candidates,
        dry_run=not args.apply
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
    finally:
        # Clean up LLM client
        if strategy.llm_client:
            await strategy.llm_client.close()


if __name__ == '__main__':
    asyncio.run(main())
