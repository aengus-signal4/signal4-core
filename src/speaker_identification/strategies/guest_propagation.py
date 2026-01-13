#!/usr/bin/env python3
"""
Guest Propagation Strategy (Phase 3)
=====================================

Identifies unassigned speakers by:
1. Comparing embeddings to existing identity centroids
2. Using LLM verification with transcript context
3. Caching LLM results in speakers.llm_identification

Tiered matching approach:
- Similarity >= 0.80: Auto-match (embedding alone sufficient)
- Similarity >= 0.65 + LLM "very_likely": Match
- Similarity >= 0.40 + LLM "certain": Match (explicit name mention)

Usage:
    # Run on single channel
    python -m src.speaker_identification.strategies.guest_propagation \\
        --channel-id 6569 --apply

    # Run on project
    python -m src.speaker_identification.strategies.guest_propagation \\
        --project CPRMV --apply

    # Dry run
    python -m src.speaker_identification.strategies.guest_propagation \\
        --channel-id 6569
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from src.utils.paths import get_project_root, get_config_path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
from sqlalchemy import text
from tqdm import tqdm

project_root = str(get_project_root())
sys.path.append(project_root)

from src.speaker_identification.core.llm_client import MLXLLMClient
from src.speaker_identification.core.context_builder import ContextBuilder
from src.speaker_identification.core.identity_manager import IdentityManager
from src.speaker_identification.core.phase_tracking import PhaseTracker
from src.speaker_identification.prompts import PromptRegistry
from src.database.session import get_session
from src.database.models import IdentificationStatus
from src.utils.logger import setup_worker_logger
from src.utils.config import get_project_date_range

logger = setup_worker_logger('speaker_identification.guest_propagation')

# Tiered thresholds
EMBEDDING_AUTO_MATCH = 0.80      # Embedding alone sufficient
EMBEDDING_LLM_REQUIRED = 0.65   # + LLM "very_likely" required
EMBEDDING_MIN_THRESHOLD = 0.40  # Below this = rejected; 0.40-0.65 = skip (retry next run)


class GuestPropagationStrategy:
    """
    Phase 3: Propagate guest identities using embedding + LLM verification.

    For each unassigned speaker with embedding:
    1. Compare to all existing identity centroids
    2. Apply tiered matching rules
    3. Cache LLM results to avoid re-running
    4. Assign speaker_identity_id for matches
    """

    def __init__(
        self,
        min_duration_pct: float = 0.05,
        min_quality: float = 0.50,
        dry_run: bool = True,
        max_speakers: int = None,
        skip_llm_cached: bool = True,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ):
        """
        Initialize strategy.

        Args:
            min_duration_pct: Minimum % of episode for speaker consideration (default 5%)
            min_quality: Minimum speaker embedding quality score
            dry_run: If True, don't make DB changes
            max_speakers: Maximum speakers to process
            skip_llm_cached: Skip speakers with existing llm_identification
            start_date: Filter content published on or after this date (YYYY-MM-DD)
            end_date: Filter content published before this date (YYYY-MM-DD)
        """
        self.min_duration_pct = min_duration_pct
        self.min_quality = min_quality
        self.dry_run = dry_run
        self.max_speakers = max_speakers
        self.skip_llm_cached = skip_llm_cached
        self.start_date = start_date
        self.end_date = end_date

        self.llm_client = MLXLLMClient()
        self.context_builder = ContextBuilder()
        self.identity_manager = IdentityManager()

        self.stats = {
            'speakers_processed': 0,
            'auto_matches': 0,
            'llm_matches': 0,
            'skipped_for_retry': 0,
            'not_verified': 0,
            'speakers_assigned': 0,
            'errors': []
        }

    def _load_all_centroids(self) -> List[Dict]:
        """Load all identity centroids from speaker_identities."""
        with get_session() as session:
            query = text("""
                SELECT
                    id as identity_id,
                    primary_name,
                    role,
                    verification_metadata
                FROM speaker_identities
                WHERE is_active = TRUE
                  AND verification_metadata ? 'centroid'
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
                        'sample_count': metadata.get('centroid_sample_count', 0)
                    })

            return centroids

    def _build_faiss_index(self, centroids: List[Dict]) -> Tuple[faiss.IndexFlatIP, np.ndarray, List[int], List[str]]:
        """
        Build a FAISS index from identity centroids for fast similarity search.

        Returns:
            Tuple of (faiss_index, centroid_matrix, identity_ids, identity_names)
        """
        if not centroids:
            return None, None, [], []

        # Extract centroid embeddings and normalize for cosine similarity
        centroid_matrix = np.array([c['centroid'] for c in centroids], dtype=np.float32)
        faiss.normalize_L2(centroid_matrix)

        # Build inner product index (equivalent to cosine similarity on normalized vectors)
        dim = centroid_matrix.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(centroid_matrix)

        identity_ids = [c['identity_id'] for c in centroids]
        identity_names = [c['name'] for c in centroids]

        logger.info(f"Built FAISS index with {len(centroids)} centroids (dim={dim})")

        return index, centroid_matrix, identity_ids, identity_names

    def _parse_embedding(self, embedding) -> np.ndarray:
        """Parse embedding from various formats (list, numpy array, or JSON string)."""
        if isinstance(embedding, np.ndarray):
            return embedding.astype(np.float32)
        elif isinstance(embedding, str):
            return np.array(json.loads(embedding), dtype=np.float32)
        else:
            return np.array(embedding, dtype=np.float32)

    def _batch_similarity_search(
        self,
        speakers: List[Dict],
        faiss_index: faiss.IndexFlatIP,
        centroid_matrix: np.ndarray,
        identity_ids: List[int],
        identity_names: List[str]
    ) -> Dict[str, List[Dict]]:
        """
        Perform batch similarity search and bucket speakers by threshold tier.

        For LLM tier, includes ALL candidates above the LLM threshold so the LLM
        can determine which (if any) is the correct match.

        Returns:
            Dict with keys: 'auto_match', 'needs_llm', 'skip_for_now', 'below_threshold'
            Each value is a list of dicts with speaker info and match details.
        """
        if not speakers:
            return {'auto_match': [], 'needs_llm': [], 'skip_for_now': [], 'below_threshold': []}

        # Extract speaker embeddings (handle JSON strings from DB)
        speaker_embeddings = np.array(
            [self._parse_embedding(s['embedding']) for s in speakers],
            dtype=np.float32
        )
        faiss.normalize_L2(speaker_embeddings)

        # Compute full similarity matrix for speakers that need multiple candidates
        # For efficiency, first do top-1 to bucket, then compute full similarities only for LLM tier
        scores_top1, indices_top1 = faiss_index.search(speaker_embeddings, 1)

        # Bucket by threshold
        buckets = {
            'auto_match': [],       # >= 0.80 with SINGLE match: auto-assign
            'needs_llm': [],        # >= 0.65 OR multiple >= 0.80: needs LLM
            'skip_for_now': [],     # < 0.65: skip, retry next run (new centroids may help)
        }

        # Track which speakers need full similarity computation
        # This includes: (1) 0.65-0.80 tier, (2) potential multi-0.80 conflicts
        potential_llm_indices = []

        for i, speaker in enumerate(speakers):
            best_score = float(scores_top1[i][0])
            best_idx = int(indices_top1[i][0])

            if best_score >= EMBEDDING_LLM_REQUIRED:
                # Could be auto-match or LLM - need to check for multiple high matches
                potential_llm_indices.append(i)
            else:
                # Below 0.65 - skip for now, retry next run (new centroids may help)
                buckets['skip_for_now'].append({
                    'speaker': speaker,
                    'best_similarity': best_score,
                    'best_identity_id': identity_ids[best_idx],
                    'best_identity_name': identity_names[best_idx]
                })

        # For speakers >= 0.65, compute full similarities to check for multiple high matches
        if potential_llm_indices:
            llm_embeddings = speaker_embeddings[potential_llm_indices]
            # Full dot product against all centroids
            all_similarities = np.dot(llm_embeddings, centroid_matrix.T)

            for idx, speaker_idx in enumerate(potential_llm_indices):
                speaker = speakers[speaker_idx]
                sims = all_similarities[idx]

                # Find all candidates above LLM threshold (0.65)
                candidates = []
                for j, sim in enumerate(sims):
                    if sim >= EMBEDDING_LLM_REQUIRED:
                        candidates.append({
                            'identity_id': identity_ids[j],
                            'identity_name': identity_names[j],
                            'similarity': float(sim)
                        })

                # Sort by similarity descending
                candidates.sort(key=lambda x: x['similarity'], reverse=True)

                if not candidates:
                    continue

                best_sim = candidates[0]['similarity']

                # Check if this is a clean auto-match:
                # - Best match >= 0.80
                # - No other candidate >= 0.80 (or second-best is significantly lower)
                auto_match_candidates = [c for c in candidates if c['similarity'] >= EMBEDDING_AUTO_MATCH]

                if len(auto_match_candidates) == 1:
                    # Single 0.80+ match - auto-assign
                    buckets['auto_match'].append({
                        'speaker': speaker,
                        'best_similarity': best_sim,
                        'best_identity_id': candidates[0]['identity_id'],
                        'best_identity_name': candidates[0]['identity_name']
                    })
                elif len(auto_match_candidates) > 1:
                    # Multiple 0.80+ matches - needs LLM to disambiguate
                    buckets['needs_llm'].append({
                        'speaker': speaker,
                        'candidates': auto_match_candidates,  # Only the 0.80+ candidates
                        'best_similarity': best_sim,
                        'best_identity_id': candidates[0]['identity_id'],
                        'best_identity_name': candidates[0]['identity_name'],
                        'reason': 'multiple_auto_match'
                    })
                else:
                    # Best is 0.65-0.80 - needs LLM
                    buckets['needs_llm'].append({
                        'speaker': speaker,
                        'candidates': candidates,  # All candidates >= 0.65
                        'best_similarity': best_sim,
                        'best_identity_id': candidates[0]['identity_id'],
                        'best_identity_name': candidates[0]['identity_name']
                    })

        return buckets

    def _parse_episode_hosts(self, content_hosts) -> List[str]:
        """
        Parse episode-level hosts from content.hosts JSONB field.

        Args:
            content_hosts: JSONB data from content.hosts (list of dicts or strings)

        Returns:
            List of host names
        """
        if not content_hosts:
            return []

        # Handle JSON string if not already parsed
        if isinstance(content_hosts, str):
            try:
                content_hosts = json.loads(content_hosts)
            except json.JSONDecodeError:
                return []

        # Extract names from list of dicts or strings
        host_names = []
        for h in content_hosts:
            if isinstance(h, dict):
                name = h.get('name') or h.get('primary_name')
                if name:
                    host_names.append(name)
            elif isinstance(h, str):
                host_names.append(h)

        return host_names

    def _get_unassigned_speakers(
        self,
        channel_id: int = None,
        project: str = None
    ) -> List[Dict]:
        """Get unassigned speakers with embeddings."""
        with get_session() as session:
            filters = []
            params = {
                'min_duration_pct': self.min_duration_pct,
                'min_quality': self.min_quality
            }

            if channel_id:
                filters.append("c.channel_id = :channel_id")
                params['channel_id'] = channel_id

            if project:
                filters.append(":project = ANY(c.projects)")
                params['project'] = project

            if self.skip_llm_cached:
                # Use per-phase tracking: skip speakers already processed by Phase 5
                # Also include retry_eligible from the phase JSONB
                filters.append(f"s.speaker_identity_id IS NULL AND {PhaseTracker.get_phase_filter_sql(5, 's')}")

            # Date range filters
            if self.start_date:
                filters.append("c.publish_date >= :start_date")
                params['start_date'] = self.start_date
            if self.end_date:
                filters.append("c.publish_date < :end_date")
                params['end_date'] = self.end_date

            filter_clause = " AND ".join(filters) if filters else "TRUE"

            if self.max_speakers:
                params['limit'] = self.max_speakers
                limit_clause = "LIMIT :limit"
            else:
                limit_clause = ""

            query = text(f"""
                SELECT
                    s.id as speaker_id,
                    s.embedding,
                    s.duration,
                    s.embedding_quality_score as quality,
                    c.content_id,
                    c.title,
                    c.description,
                    c.duration as ep_duration,
                    c.channel_id,
                    c.hosts as content_hosts,
                    ch.display_name as channel_name
                FROM speakers s
                JOIN content c ON s.content_id = c.content_id
                JOIN channels ch ON c.channel_id = ch.id
                WHERE s.speaker_identity_id IS NULL
                  AND {PhaseTracker.get_phase_filter_sql(5, 's')}
                  AND s.embedding IS NOT NULL
                  AND c.is_stitched = true
                  AND c.duration > 0
                  AND s.duration > 0
                  AND (s.duration / c.duration) >= :min_duration_pct
                  AND COALESCE(s.embedding_quality_score, 1.0) >= :min_quality
                  AND {filter_clause}
                ORDER BY c.channel_id, s.duration DESC
                {limit_clause}
            """)

            results = session.execute(query, params).fetchall()

            return [dict(row._mapping) for row in results]

    async def _verify_with_llm(
        self,
        speaker: Dict,
        candidates: List[Dict]
    ) -> Dict:
        """
        Call LLM to verify speaker identity against multiple candidates.

        Args:
            speaker: Speaker dict with episode info
            candidates: List of candidate dicts with 'identity_id', 'identity_name', 'similarity'

        Returns dict with: is_match, confidence, reasoning, identified_name, matched_identity_id
        """
        # Get transcript context
        context = self.context_builder.get_speaker_transcript_context(speaker['speaker_id'])
        if not context:
            return {
                'is_match': False,
                'confidence': 'unlikely',
                'reasoning': 'No transcript context available',
                'identified_name': None,
                'matched_identity_id': None
            }

        duration_pct = (speaker['duration'] / speaker['ep_duration'] * 100)

        # Parse episode-level hosts from metadata
        known_hosts = self._parse_episode_hosts(speaker.get('content_hosts'))

        # Build 6-turn transcript context with speaker names when known
        turn_before_first = context.get('turn_before_first')
        turn_after_first = context.get('turn_after_first')
        turn_before_last = context.get('turn_before_last')
        turn_after_last = context.get('turn_after_last')
        first_utterance = context.get('first_utterance', 'N/A')
        last_utterance = context.get('last_utterance', 'N/A')

        # Get speaker names for surrounding turns (from assigned identities)
        before_first_name = context.get('turn_before_first_speaker_name')
        after_first_name = context.get('turn_after_first_speaker_name')
        before_last_name = context.get('turn_before_last_speaker_name')
        after_last_name = context.get('turn_after_last_speaker_name')

        # Format speaker labels: use name if known, else generic label
        def _label(name: str, fallback: str) -> str:
            return name if name else fallback

        transcript_section = f"""
--- FIRST APPEARANCE ---
{_label(before_first_name, '[Speaker before]')}: ...{turn_before_first[-300:] if turn_before_first else 'N/A'}
UNKNOWN SPEAKER: {first_utterance}
{_label(after_first_name, '[Speaker after]')}: {turn_after_first[:300] if turn_after_first else 'N/A'}...

--- LAST APPEARANCE ---
{_label(before_last_name, '[Speaker before]')}: ...{turn_before_last[-300:] if turn_before_last else 'N/A'}
UNKNOWN SPEAKER: {last_utterance}
{_label(after_last_name, '[Speaker after]')}: {turn_after_last[:300] if turn_after_last else 'N/A'}..."""

        # Build prompt using registry - multi-candidate version
        prompt = PromptRegistry.phase5_multi_candidate_verification(
            candidates=candidates,
            episode_title=speaker['title'],
            episode_description=speaker['description'],
            known_hosts=known_hosts,
            transcript_section=transcript_section,
            duration_pct=duration_pct,
            total_turns=context.get('total_turns', 0)
        )

        # Call LLM
        response = await self.llm_client._call_llm(prompt, priority=2)

        # Parse response
        try:
            response = response.strip()
            if response.startswith("```"):
                lines = response.split("\n")
                response = "\n".join(lines[1:-1])
            data = json.loads(response)

            # Extract evidence fields (new structured format)
            evidence = data.get('evidence', {})
            identified_name = data.get('identified_name')

            # Find the matching identity_id from candidates
            matched_identity_id = None
            matched_similarity = None
            if identified_name:
                for c in candidates:
                    if c['identity_name'].lower() == identified_name.lower():
                        matched_identity_id = c['identity_id']
                        matched_similarity = c['similarity']
                        break

            return {
                'is_match': data.get('is_match', False),
                'confidence': data.get('confidence', 'unlikely'),
                'reasoning': data.get('reasoning', ''),
                'identified_name': identified_name,
                'matched_identity_id': matched_identity_id,
                'matched_similarity': matched_similarity,
                # Categorical evidence fields
                'evidence_type': evidence.get('type', 'none'),
                'evidence_source': evidence.get('speaker_source', 'unknown'),
                'evidence_quote': evidence.get('quote', '')
            }
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            return {
                'is_match': False,
                'confidence': 'unlikely',
                'reasoning': f'JSON parse error: {e}',
                'identified_name': None,
                'matched_identity_id': None,
                'matched_similarity': None,
                'evidence_type': 'none',
                'evidence_source': 'unknown',
                'evidence_quote': ''
            }

    def _save_llm_identification(
        self,
        speaker_id: int,
        result: Dict,
        candidates: List[Dict],
        method: str = 'guest_propagation'
    ):
        """Save LLM identification result to speakers table using per-phase tracking."""
        if self.dry_run:
            return

        # Determine phase status based on result
        is_match = result.get('is_match', False)
        confidence = result.get('confidence', 'unlikely')

        if not is_match:
            if confidence == 'probably':
                phase_status = PhaseTracker.STATUS_RETRY_ELIGIBLE
            else:
                phase_status = PhaseTracker.STATUS_REJECTED
        else:
            # Will be upgraded to 'assigned' if assignment succeeds
            phase_status = PhaseTracker.STATUS_RETRY_ELIGIBLE

        # Record in per-phase JSONB structure
        PhaseTracker.record_result(
            speaker_id=speaker_id,
            phase=5,
            status=phase_status,
            result={
                'identified_name': result.get('identified_name'),
                'matched_identity_id': result.get('matched_identity_id'),
                'candidates': [{'name': c['identity_name'], 'sim': round(c['similarity'], 3)} for c in candidates],
                'is_match': is_match,
                'confidence': confidence,
                'reasoning': result.get('reasoning', ''),
                'embedding_similarity': round(result.get('matched_similarity') or 0, 4),
                # Categorical evidence fields
                'evidence_type': result.get('evidence_type', 'none'),
                'evidence_source': result.get('evidence_source', 'unknown'),
                'evidence_quote': result.get('evidence_quote', '')
            },
            method=method,
            dry_run=self.dry_run
        )

    def _assign_speaker_identity(
        self,
        speaker_id: int,
        identity_id: int,
        confidence: float,
        method: str,
        candidate_name: str = None
    ):
        """Assign speaker to identity."""
        if self.dry_run:
            return

        with get_session() as session:
            session.execute(
                text("""
                    UPDATE speakers
                    SET speaker_identity_id = :identity_id,
                        assignment_confidence = :confidence,
                        assignment_method = :method,
                        identification_status = :status,
                        updated_at = NOW()
                    WHERE id = :speaker_id
                """),
                {
                    'speaker_id': speaker_id,
                    'identity_id': identity_id,
                    'confidence': confidence,
                    'method': method,
                    'status': IdentificationStatus.ASSIGNED
                }
            )
            session.commit()

        # Also update phase status to 'assigned'
        PhaseTracker.record_result(
            speaker_id=speaker_id,
            phase=5,
            status=PhaseTracker.STATUS_ASSIGNED,
            result={
                'identified_name': candidate_name,
                'confidence': confidence
            },
            method=method,
            identity_id=identity_id,
            dry_run=self.dry_run
        )

    async def run(
        self,
        channel_id: int = None,
        project: str = None
    ) -> Dict:
        """
        Run guest propagation using FAISS for efficient batch similarity search.

        Args:
            channel_id: Optional channel to filter to
            project: Optional project to filter to

        Returns:
            Stats dict
        """
        # Load date range from project config if not already set
        if project and not self.start_date and not self.end_date:
            start_date, end_date = get_project_date_range(project)
            self.start_date = start_date
            self.end_date = end_date

        logger.info("=" * 80)
        logger.info("GUEST PROPAGATION (Phase 5) - FAISS Batch Mode")
        logger.info("=" * 80)
        logger.info(f"Mode: {'DRY RUN' if self.dry_run else 'APPLY'}")
        logger.info(f"Thresholds:")
        logger.info(f"  - >= {EMBEDDING_AUTO_MATCH}: Auto-match (single high match)")
        logger.info(f"  - >= {EMBEDDING_LLM_REQUIRED}: LLM verification (or multiple 0.80+ matches)")
        logger.info(f"  - < {EMBEDDING_LLM_REQUIRED}: Skip for now (retry next run)")
        if self.start_date or self.end_date:
            logger.info(f"Date range: {self.start_date or 'any'} to {self.end_date or 'any'}")

        # Load centroids and build FAISS index
        centroids = self._load_all_centroids()
        if not centroids:
            logger.warning("No identity centroids found!")
            return self.stats

        logger.info(f"\nLoaded {len(centroids)} identity centroids")

        faiss_index, centroid_matrix, identity_ids, identity_names = self._build_faiss_index(centroids)

        # Get unassigned speakers
        speakers = self._get_unassigned_speakers(channel_id, project)
        if not speakers:
            logger.info("No unassigned speakers found")
            return self.stats

        logger.info(f"Loaded {len(speakers)} unassigned speakers")

        # Batch similarity search and bucket by threshold
        logger.info("Running FAISS batch similarity search...")
        buckets = self._batch_similarity_search(speakers, faiss_index, centroid_matrix, identity_ids, identity_names)

        # Report counts upfront
        logger.info("-" * 80)
        logger.info("SIMILARITY SEARCH RESULTS:")
        logger.info(f"  Auto-match (>= {EMBEDDING_AUTO_MATCH}, single): {len(buckets['auto_match']):,} speakers")
        logger.info(f"  Needs LLM (>= {EMBEDDING_LLM_REQUIRED}):          {len(buckets['needs_llm']):,} speakers")
        logger.info(f"  Skip for now (< {EMBEDDING_LLM_REQUIRED}):        {len(buckets['skip_for_now']):,} speakers (retry next run)")
        logger.info("-" * 80)

        logger.info(f"PLAN: {len(buckets['auto_match']):,} auto-assigns, {len(buckets['needs_llm']):,} LLM calls")
        logger.info("-" * 80)

        # Phase 1: Process auto-matches (no LLM needed)
        logger.info(f"\n[1/3] Processing {len(buckets['auto_match']):,} auto-matches...")
        for match_info in tqdm(buckets['auto_match'], desc="Auto-matches"):
            self.stats['speakers_processed'] += 1
            speaker = match_info['speaker']
            best_sim = match_info['best_similarity']
            best_name = match_info['best_identity_name']
            best_identity_id = match_info['best_identity_id']

            try:
                self.stats['auto_matches'] += 1
                tqdm.write(
                    f"  ✓ AUTO: {speaker['channel_name'][:20]} | "
                    f"{speaker['title'][:40]} | {best_name} (sim={best_sim:.3f})"
                )

                # Record phase tracking for auto-match
                if not self.dry_run:
                    PhaseTracker.record_result(
                        speaker_id=speaker['speaker_id'],
                        phase=5,
                        status=PhaseTracker.STATUS_ASSIGNED,
                        result={
                            'identified_name': best_name,
                            'matched_identity_id': best_identity_id,
                            'embedding_similarity': round(best_sim, 4),
                            'method': 'auto_embedding',
                            'reasoning': 'Single high-confidence embedding match (>= 0.80)'
                        },
                        method='auto_embedding',
                        identity_id=best_identity_id,
                        dry_run=self.dry_run
                    )

                self._assign_speaker_identity(
                    speaker['speaker_id'],
                    best_identity_id,
                    best_sim,
                    'auto_embedding',
                    candidate_name=best_name
                )
                self.stats['speakers_assigned'] += 1

            except Exception as e:
                logger.error(f"Error processing speaker {speaker['speaker_id']}: {e}")
                self.stats['errors'].append(str(e))

        # Phase 2: Skip low-similarity speakers (don't mark them - they'll be retried next run)
        logger.info(f"\n[2/3] Skipping {len(buckets['skip_for_now']):,} low-similarity speakers (will retry next run)...")
        self.stats['skipped_for_retry'] = len(buckets['skip_for_now'])

        # Phase 3: Process LLM-required tier (with multiple candidates)
        logger.info(f"\n[3/3] Processing {len(buckets['needs_llm']):,} speakers requiring LLM verification...")

        for match_info in tqdm(buckets['needs_llm'], desc="LLM verification"):
            self.stats['speakers_processed'] += 1
            speaker = match_info['speaker']
            candidates = match_info['candidates']  # All candidates >= 0.65

            try:
                # Call LLM with all candidates
                result = await self._verify_with_llm(speaker, candidates)

                # Save LLM result
                self._save_llm_identification(
                    speaker['speaker_id'],
                    result,
                    candidates
                )

                is_match = result.get('is_match', False)
                confidence = result.get('confidence', 'unlikely')
                identified_name = result.get('identified_name')
                matched_identity_id = result.get('matched_identity_id')
                matched_similarity = result.get('matched_similarity')

                # Check if match criteria met (need "very_likely" or "certain" AND a valid identity_id)
                matched = False
                if is_match and confidence in ('very_likely', 'certain') and matched_identity_id:
                    self.stats['llm_matches'] += 1
                    matched = True

                if matched:
                    candidate_count = len(candidates)
                    tqdm.write(
                        f"  ✓ LLM: {speaker['channel_name'][:20]} | "
                        f"{speaker['title'][:40]} | {identified_name} ({confidence}, sim={matched_similarity:.3f}, {candidate_count} candidates)"
                    )
                    self._assign_speaker_identity(
                        speaker['speaker_id'],
                        matched_identity_id,
                        matched_similarity,
                        f'llm_{confidence}',
                        candidate_name=identified_name
                    )
                    self.stats['speakers_assigned'] += 1
                else:
                    self.stats['not_verified'] += 1

            except Exception as e:
                logger.error(f"Error processing speaker {speaker['speaker_id']}: {e}")
                self.stats['errors'].append(str(e))

        self._print_summary()
        return self.stats

    def _print_summary(self):
        """Print summary statistics."""
        logger.info("\n" + "=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Speakers processed: {self.stats['speakers_processed']}")
        logger.info(f"Auto-matches (>= {EMBEDDING_AUTO_MATCH}, single): {self.stats['auto_matches']}")
        logger.info(f"LLM matches: {self.stats['llm_matches']}")
        logger.info(f"Skipped for retry (< {EMBEDDING_LLM_REQUIRED}): {self.stats['skipped_for_retry']}")
        logger.info(f"Not verified by LLM: {self.stats['not_verified']}")
        total = self.stats['auto_matches'] + self.stats['llm_matches']
        logger.info(f"TOTAL MATCHES: {total}")
        logger.info(f"Speakers assigned: {self.stats['speakers_assigned']}")
        if self.stats['errors']:
            logger.warning(f"Errors: {len(self.stats['errors'])}")
        logger.info("=" * 80)


async def main():
    parser = argparse.ArgumentParser(
        description='Phase 3: Guest propagation via embedding + LLM',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--channel-id', type=int, help='Process single channel')
    parser.add_argument('--project', type=str, help='Filter to project')
    parser.add_argument('--max-speakers', type=int, help='Max speakers to process')
    parser.add_argument('--apply', action='store_true', help='Apply changes (default: dry run)')
    parser.add_argument('--min-duration-pct', type=float, default=0.15,
                       help='Min duration %% (default: 0.15)')
    parser.add_argument('--min-quality', type=float, default=0.50,
                       help='Min embedding quality (default: 0.50)')
    parser.add_argument('--include-cached', action='store_true',
                       help='Include speakers with existing LLM results')

    args = parser.parse_args()

    strategy = GuestPropagationStrategy(
        min_duration_pct=args.min_duration_pct,
        min_quality=args.min_quality,
        dry_run=not args.apply,
        max_speakers=args.max_speakers,
        skip_llm_cached=not args.include_cached
    )

    try:
        await strategy.run(
            channel_id=args.channel_id,
            project=args.project
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
