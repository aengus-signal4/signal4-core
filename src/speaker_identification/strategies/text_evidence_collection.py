#!/usr/bin/env python3
"""
Text Evidence Collection Strategy (Phase 2)
============================================

NEW pipeline Phase 2: Find ROCK-SOLID transcript evidence for speaker identity.

This phase processes ALL speakers with ≥10% duration and finds text-based evidence
(self-introduction, being addressed, being introduced). This creates the "ground truth"
for building clean, uncontaminated centroids in Phase 3.

Key differences from old guest_identification.py:
- Processes ALL speakers (not just those with known hosts)
- Does NOT create/assign identities - just collects evidence
- Binary output: "certain" evidence or "none" (no middle ground)
- Stores evidence in new columns: text_evidence_status, evidence_type, evidence_quote

Usage:
    # Run on single channel
    python -m src.speaker_identification.strategies.text_evidence_collection \\
        --channel-id 6569 --apply

    # Run on project
    python -m src.speaker_identification.strategies.text_evidence_collection \\
        --project CPRMV --apply

    # Dry run
    python -m src.speaker_identification.strategies.text_evidence_collection \\
        --channel-id 6569
"""

import argparse
import asyncio
import json
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from src.utils.paths import get_project_root, get_config_path
from typing import Dict, List, Optional

from sqlalchemy import text
from tqdm import tqdm

project_root = str(get_project_root())
sys.path.append(project_root)

from src.speaker_identification.core.llm_client import MLXLLMClient
from src.speaker_identification.core.context_builder import ContextBuilder
from src.speaker_identification.prompts import PromptRegistry
from src.database.session import get_session
from src.utils.logger import setup_worker_logger
from src.utils.config import get_project_date_range, load_config
from src.utils.priority import calculate_priority_by_date

logger = setup_worker_logger('speaker_identification.text_evidence_collection')

# Phase key for identification_details JSONB
PHASE_KEY = "phase2"
PHASE_NUMBER = 2


class TextEvidenceCollectionStrategy:
    """
    Phase 2: Collect text-based evidence for speaker identification.

    This is the first phase of the text-evidence-first pipeline.
    It processes ALL speakers meeting duration thresholds and finds
    rock-solid transcript evidence (self-ID, addressed, introduced).

    Output is stored in identification_details JSONB under 'phase2' key:
    - status: 'certain', 'none', 'short_utterance'
    - evidence_type: 'self_intro', 'addressed', 'introduced', 'none'
    - identified_name: The name found (if any)
    - reasoning: LLM explanation

    This phase does NOT create identities or make assignments.

    Supports parallel LLM processing via asyncio.Semaphore to take advantage
    of multiple LLM providers in the load balancer.
    """

    def __init__(
        self,
        min_duration_pct: float = 0.10,
        min_quality: float = 0.50,
        dry_run: bool = True,
        max_speakers: int = None,
        skip_processed: bool = True,
        include_assigned: bool = False,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_concurrent: int = 15
    ):
        """
        Initialize strategy.

        Args:
            min_duration_pct: Minimum % of episode for speaker consideration (default 10%)
            min_quality: Minimum speaker embedding quality score
            dry_run: If True, don't make DB changes
            max_speakers: Maximum speakers to process (for testing)
            skip_processed: Skip speakers with existing text_evidence_status
            include_assigned: Include speakers already assigned to an identity (for re-verification)
            start_date: Filter content published on or after this date (YYYY-MM-DD)
            end_date: Filter content published before this date (YYYY-MM-DD)
            max_concurrent: Maximum concurrent LLM requests (default 15)
        """
        self.min_duration_pct = min_duration_pct
        self.min_quality = min_quality
        self.dry_run = dry_run
        self.max_speakers = max_speakers
        self.skip_processed = skip_processed
        self.include_assigned = include_assigned
        self.start_date = start_date
        self.end_date = end_date
        self.max_concurrent = max_concurrent

        self.llm_client = MLXLLMClient(tier="tier_1")
        self.context_builder = ContextBuilder()

        # Load project priorities from config for ordering
        config = load_config()
        active_projects = config.get('active_projects', {})
        self.project_priorities = {
            name: cfg.get('priority', 0)
            for name, cfg in active_projects.items()
        }

        # Semaphore for controlling concurrent LLM requests
        self._semaphore: Optional[asyncio.Semaphore] = None

        # Graceful shutdown flag
        self._shutdown_requested = False
        self._active_tasks: List[asyncio.Task] = []

        self.stats = {
            'speakers_processed': 0,
            'evidence_found_certain': 0,
            'evidence_type_self_intro': 0,
            'evidence_type_addressed': 0,
            'evidence_type_introduced': 0,
            'evidence_none': 0,
            'short_utterance': 0,
            'skipped_already_processed': 0,
            'errors': []
        }

    def _get_speakers_for_processing(
        self,
        channel_id: int = None,
        project: str = None,
        projects: List[str] = None,
        limit: int = None
    ) -> List[Dict]:
        """
        Get speakers meeting duration threshold for text evidence collection.

        Unlike old guest_identification which required known hosts,
        this processes EVERY speaker with sufficient duration.

        Prioritization (same as other task types):
        1. Recent content (last 30 days) - gets 10M priority boost
        2. Then by project priority (higher priority projects first)
        3. Then by publish date (newer content first)

        Args:
            channel_id: Optional single channel to filter to
            project: Optional single project to filter to
            projects: Optional list of projects to filter to (for universal processing)
            limit: Optional limit on number of speakers to return (for batch processing)

        Returns speakers where:
        - duration_pct >= min_duration_pct
        - embedding_quality_score >= min_quality
        - text_evidence_status IS NULL (unless include_assigned=True)
        """
        with get_session() as session:
            filters = []
            params = {
                'min_duration_pct': self.min_duration_pct,
                'min_quality': self.min_quality
            }

            if channel_id:
                filters.append("c.channel_id = :channel_id")
                params['channel_id'] = channel_id

            if projects:
                # Filter to any of the specified projects using array overlap
                # Build the array literal directly since SQLAlchemy has issues with array casting
                projects_literal = "ARRAY[" + ",".join(f"'{p}'" for p in projects) + "]::varchar[]"
                filters.append(f"c.projects && {projects_literal}")
            elif project:
                filters.append(":project = ANY(c.projects)")
                params['project'] = project

            # Skip already processed speakers (have phase2 in identification_details)
            if self.skip_processed:
                filters.append(f"(s.identification_details->'{PHASE_KEY}' IS NULL)")

            # By default, only process unassigned speakers
            # Set include_assigned=True to re-verify legacy assignments
            if not self.include_assigned:
                filters.append("s.speaker_identity_id IS NULL")

            # Date range filters
            if self.start_date:
                filters.append("c.publish_date >= :start_date")
                params['start_date'] = self.start_date
            if self.end_date:
                filters.append("c.publish_date < :end_date")
                params['end_date'] = self.end_date

            filter_clause = " AND ".join(filters) if filters else "TRUE"

            # Use limit parameter if provided, otherwise fall back to max_speakers
            effective_limit = limit if limit is not None else self.max_speakers
            if effective_limit:
                params['limit'] = effective_limit
                limit_clause = "LIMIT :limit"
            else:
                limit_clause = ""

            # Build project priority CASE statement for the priority calculation
            # This mirrors calculate_priority_by_date logic in SQL
            if self.project_priorities:
                priority_cases = " ".join(
                    f"WHEN '{proj}' = ANY(c.projects) THEN {priority}"
                    for proj, priority in self.project_priorities.items()
                )
                project_priority_expr = f"CASE {priority_cases} ELSE 1 END"
            else:
                project_priority_expr = "1"

            # Priority ordering that matches calculate_priority_by_date:
            # - Recent content (< 30 days) gets 10M boost
            # - Then project priority bands (priority * 1M)
            # - Then date_priority within each band
            query = text(f"""
                SELECT
                    s.id as speaker_id,
                    s.duration,
                    s.segment_count,
                    s.embedding_quality_score as quality,
                    s.speaker_identity_id,
                    c.content_id,
                    c.title,
                    c.description,
                    c.publish_date,
                    c.duration as ep_duration,
                    c.channel_id,
                    c.hosts as content_hosts,
                    c.guests as content_guests,
                    ch.display_name as channel_name,
                    -- Calculate priority matching calculate_priority_by_date
                    (
                        -- Base date priority
                        (EXTRACT(EPOCH FROM c.publish_date)::bigint / 86400 - 20000) * 1000
                        -- Add project priority band
                        + ({project_priority_expr}) * 1000000
                        -- Add recency boost for content < 30 days old
                        + CASE
                            WHEN c.publish_date >= (NOW() - INTERVAL '30 days') THEN 10000000
                            ELSE 0
                          END
                    ) as calculated_priority
                FROM speakers s
                JOIN content c ON s.content_id = c.content_id
                JOIN channels ch ON c.channel_id = ch.id
                WHERE s.duration >= 30
                  AND s.segment_count >= 2
                  AND c.duration > 0
                  AND c.is_stitched = true
                  AND (s.duration / c.duration) >= :min_duration_pct
                  AND COALESCE(s.embedding_quality_score, 1.0) >= :min_quality
                  AND {filter_clause}
                  -- Only include content that has been migrated to sentences table
                  AND EXISTS (
                      SELECT 1 FROM sentences sen
                      JOIN content co ON sen.content_id = co.id
                      WHERE co.content_id = c.content_id
                      LIMIT 1
                  )
                ORDER BY calculated_priority DESC, s.duration DESC
                {limit_clause}
            """)

            results = session.execute(query, params).fetchall()
            return [dict(row._mapping) for row in results]

    def _count_speakers_for_processing(
        self,
        channel_id: int = None,
        project: str = None,
        projects: List[str] = None
    ) -> int:
        """
        Count speakers needing Phase 2 processing.

        Uses the same filtering criteria as _get_speakers_for_processing.

        Args:
            channel_id: Optional single channel to filter to
            project: Optional single project to filter to
            projects: Optional list of projects to filter to

        Returns:
            Total count of speakers needing processing
        """
        with get_session() as session:
            filters = []
            params = {
                'min_duration_pct': self.min_duration_pct,
                'min_quality': self.min_quality
            }

            if channel_id:
                filters.append("c.channel_id = :channel_id")
                params['channel_id'] = channel_id

            if projects:
                projects_literal = "ARRAY[" + ",".join(f"'{p}'" for p in projects) + "]::varchar[]"
                filters.append(f"c.projects && {projects_literal}")
            elif project:
                filters.append(":project = ANY(c.projects)")
                params['project'] = project

            if self.skip_processed:
                filters.append(f"(s.identification_details->'{PHASE_KEY}' IS NULL)")

            if not self.include_assigned:
                filters.append("s.speaker_identity_id IS NULL")

            if self.start_date:
                filters.append("c.publish_date >= :start_date")
                params['start_date'] = self.start_date
            if self.end_date:
                filters.append("c.publish_date < :end_date")
                params['end_date'] = self.end_date

            filter_clause = " AND ".join(filters) if filters else "TRUE"

            query = text(f"""
                SELECT COUNT(*) as total
                FROM speakers s
                JOIN content c ON s.content_id = c.content_id
                WHERE s.duration >= 30
                  AND s.segment_count >= 2
                  AND c.duration > 0
                  AND c.is_stitched = true
                  AND (s.duration / c.duration) >= :min_duration_pct
                  AND COALESCE(s.embedding_quality_score, 1.0) >= :min_quality
                  AND {filter_clause}
                  AND EXISTS (
                      SELECT 1 FROM sentences sen
                      JOIN content co ON sen.content_id = co.id
                      WHERE co.content_id = c.content_id
                      LIMIT 1
                  )
            """)

            result = session.execute(query, params).fetchone()
            return result.total if result else 0

    def _parse_possible_speakers(self, speaker: Dict) -> List[str]:
        """
        Parse possible speaker names from episode metadata.

        Combines content.hosts and content.guests into a single list.
        These are "hints" from Phase 1 metadata extraction.
        """
        possible = []

        # Parse hosts
        hosts = speaker.get('content_hosts')
        if hosts:
            if isinstance(hosts, str):
                try:
                    hosts = json.loads(hosts)
                except json.JSONDecodeError:
                    hosts = []
            for h in hosts:
                name = h.get('name') if isinstance(h, dict) else h
                if name and name not in possible:
                    possible.append(name)

        # Parse guests
        guests = speaker.get('content_guests')
        if guests:
            if isinstance(guests, str):
                try:
                    guests = json.loads(guests)
                except json.JSONDecodeError:
                    guests = []
            for g in guests:
                name = g.get('name') if isinstance(g, dict) else g
                if name and name not in possible:
                    possible.append(name)

        return possible

    def _build_transcript_context(self, context: Dict) -> str:
        """
        Build formatted transcript context for LLM prompt.

        Uses the 6-turn pattern from context_builder.
        NOTE: We intentionally do NOT include assigned speaker names to avoid
        circular reasoning - the LLM should identify speakers from transcript
        content alone, not from potentially-wrong existing assignments.
        """
        def _truncate_middle(text: str, max_chars: int = 600) -> str:
            """Truncate long text, keeping start and end."""
            if not text or len(text) <= max_chars:
                return text
            half = max_chars // 2
            return f"{text[:half]}...{text[-half:]}"

        turn_before_first = context.get('turn_before_first')
        turn_after_first = context.get('turn_after_first')
        turn_before_last = context.get('turn_before_last')
        turn_after_last = context.get('turn_after_last')
        first_utterance = context.get('first_utterance', 'N/A')
        last_utterance = context.get('last_utterance', 'N/A')

        # Truncate unknown speaker utterances (can be very long)
        first_utterance_truncated = _truncate_middle(first_utterance, 600)
        last_utterance_truncated = _truncate_middle(last_utterance, 600)

        transcript_section = f"""
--- FIRST APPEARANCE ---
[Previous speaker]: ...{turn_before_first[-300:] if turn_before_first else 'N/A'}
[UNKNOWN SPEAKER]: {first_utterance_truncated}
[Next speaker]: {turn_after_first[:300] if turn_after_first else 'N/A'}...

--- LAST APPEARANCE ---
[Previous speaker]: ...{turn_before_last[-300:] if turn_before_last else 'N/A'}
[UNKNOWN SPEAKER]: {last_utterance_truncated}
[Next speaker]: {turn_after_last[:300] if turn_after_last else 'N/A'}..."""

        return transcript_section

    async def _find_text_evidence(self, speaker: Dict) -> Dict:
        """
        Use LLM to find text evidence proving speaker identity.

        Returns:
            Dict with: evidence_found, reasoning, evidence_type, identified_name
        """
        # Get transcript context
        context = self.context_builder.get_speaker_transcript_context(speaker['speaker_id'])
        if not context:
            return {
                'evidence_found': False,
                'reasoning': 'No transcript context available',
                'evidence_type': 'none',
                'identified_name': None
            }

        duration_pct = (speaker['duration'] / speaker['ep_duration'] * 100)
        total_turns = context.get('total_turns', 0)

        # Build possible speakers list from metadata
        possible_speakers = self._parse_possible_speakers(speaker)

        # Build transcript context
        transcript_context = self._build_transcript_context(context)

        # Build prompt using registry
        prompt = PromptRegistry.phase2_text_evidence_collection(
            episode_title=speaker['title'],
            episode_description=speaker['description'],
            possible_speakers=possible_speakers,
            transcript_context=transcript_context,
            duration_pct=duration_pct,
            total_turns=total_turns
        )

        # Log transcript context for debugging
        logger.info(f"\n{'='*80}\nSPEAKER {speaker['speaker_id']} TRANSCRIPT CONTEXT:\n{'='*80}{transcript_context}\n{'='*80}")

        # Call LLM with retry on JSON parse error
        max_retries = 2
        last_error = None

        for attempt in range(max_retries):
            response = await self.llm_client._call_llm(prompt, priority=2)

            # Log full response for debugging
            logger.info(f"\n{'='*80}\nRESPONSE FOR SPEAKER {speaker['speaker_id']}:\n{'='*80}\n{response}\n{'='*80}")

            # Parse response
            try:
                response = response.strip()
                if response.startswith("```"):
                    lines = response.split("\n")
                    response = "\n".join(lines[1:-1])
                data = json.loads(response)

                return {
                    'evidence_found': data.get('evidence_found', False),
                    'reasoning': data.get('reasoning', ''),
                    'evidence_type': data.get('evidence_type', 'none'),
                    'identified_name': data.get('identified_name')
                }
            except json.JSONDecodeError as e:
                last_error = e
                if attempt < max_retries - 1:
                    logger.debug(f"JSON parse error (attempt {attempt + 1}), retrying: {e}")
                    continue
                else:
                    logger.warning(f"JSON parse error after {max_retries} attempts: {e}")

        # All retries failed
        return {
            'evidence_found': False,
            'reasoning': f'JSON parse error after {max_retries} attempts: {last_error}',
            'evidence_type': 'none',
            'identified_name': None
        }

    async def _process_speaker_with_semaphore(self, speaker: Dict) -> Dict:
        """
        Process a single speaker with semaphore to control concurrency.

        Returns:
            Dict with speaker info and result for stats tracking
        """
        async with self._semaphore:
            try:
                duration_pct = (speaker['duration'] / speaker['ep_duration'] * 100)
                result = await self._find_text_evidence(speaker)

                # Save result
                self._save_text_evidence(speaker['speaker_id'], result, duration_pct)

                return {
                    'speaker_id': speaker['speaker_id'],
                    'content_id': speaker['content_id'],
                    'title': speaker['title'],
                    'duration_pct': duration_pct,
                    'result': result,
                    'error': None
                }
            except Exception as e:
                logger.error(f"Error processing speaker {speaker['speaker_id']}: {e}")
                return {
                    'speaker_id': speaker['speaker_id'],
                    'content_id': speaker['content_id'],
                    'title': speaker['title'],
                    'duration_pct': 0,
                    'result': None,
                    'error': str(e)
                }

    def _save_text_evidence(
        self,
        speaker_id: int,
        result: Dict,
        duration_pct: float
    ):
        """
        Save text evidence result to identification_details JSONB.

        Stores under 'phase2' key:
        - status: 'certain', 'none', 'short_utterance'
        - evidence_type: 'self_intro', 'addressed', 'introduced', 'none'
        - identified_name: Name found (if any)
        - reasoning: LLM explanation
        """
        if self.dry_run:
            return

        evidence_found = result.get('evidence_found', False)
        evidence_type = result.get('evidence_type', 'none')
        identified_name = result.get('identified_name')
        reasoning = result.get('reasoning', '')

        # Determine status
        if evidence_type == 'none' and 'short utterance' in reasoning.lower():
            status = 'short_utterance'
        elif evidence_found and evidence_type in ['self_intro', 'addressed', 'introduced']:
            status = 'certain'
        else:
            status = 'none'

        # Build phase entry for identification_details JSONB
        phase_entry = {
            'status': status,
            'timestamp': datetime.utcnow().isoformat(),
            'evidence_found': evidence_found,
            'evidence_type': evidence_type,
            'identified_name': identified_name,
            'reasoning': reasoning
        }

        with get_session() as session:
            # Update identification_details JSONB with phase2 results
            query = text(f"""
                UPDATE speakers
                SET identification_details = jsonb_set(
                    COALESCE(identification_details, '{{}}'::jsonb),
                    ARRAY['{PHASE_KEY}'],
                    CAST(:phase_entry AS jsonb)
                ),
                updated_at = NOW()
                WHERE id = :speaker_id
            """)

            session.execute(query, {
                'speaker_id': speaker_id,
                'phase_entry': json.dumps(phase_entry)
            })

            session.commit()

    async def run(
        self,
        channel_id: int = None,
        project: str = None,
        projects: List[str] = None,
        batch_size: int = None
    ) -> Dict:
        """
        Run text evidence collection with parallel LLM processing.

        Args:
            channel_id: Optional channel to filter to
            project: Optional single project to filter to
            projects: Optional list of projects to filter to (universal processing)
            batch_size: If provided, process this many speakers then recheck priority
                        for new content. Continues until no more speakers remain.

        Returns:
            Stats dict
        """
        # Load date range from project config if not already set (only for single project)
        if project and not projects and not self.start_date and not self.end_date:
            start_date, end_date = get_project_date_range(project)
            self.start_date = start_date
            self.end_date = end_date

        logger.info("=" * 80)
        logger.info("TEXT EVIDENCE COLLECTION (Phase 2 - New Pipeline)")
        logger.info("=" * 80)
        logger.info(f"Mode: {'DRY RUN' if self.dry_run else 'APPLY'}")
        if projects:
            logger.info(f"Projects: {', '.join(projects)}")
        elif project:
            logger.info(f"Project: {project}")
        logger.info(f"Min duration: {self.min_duration_pct * 100:.1f}%")
        logger.info(f"Include assigned speakers: {self.include_assigned}")
        logger.info(f"Skip already processed: {self.skip_processed}")
        logger.info(f"Max concurrent LLM requests: {self.max_concurrent}")
        if batch_size:
            logger.info(f"Batch size: {batch_size} (will recheck priority after each batch)")
        if self.start_date or self.end_date:
            logger.info(f"Date range: {self.start_date or 'any'} to {self.end_date or 'any'}")

        # Initialize semaphore for concurrent LLM requests
        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        self._shutdown_requested = False

        if batch_size:
            # Batch mode: process batch_size speakers, then requery for priority updates
            await self._run_batched(channel_id, project, projects, batch_size)
        else:
            # Original mode: get all speakers upfront and process
            await self._run_all_at_once(channel_id, project, projects)

        self._print_summary()
        return self.stats

    async def _run_all_at_once(
        self,
        channel_id: int = None,
        project: str = None,
        projects: List[str] = None
    ):
        """Original processing mode: get all speakers upfront and process in parallel."""
        # Get speakers - use projects list if provided, otherwise single project
        speakers = self._get_speakers_for_processing(channel_id, project, projects)
        if not speakers:
            logger.info("No speakers found for processing")
            return

        # Count unique episodes
        unique_episodes = len(set(s['content_id'] for s in speakers))
        logger.info(f"Processing {len(speakers)} speakers across {unique_episodes} episodes")
        logger.info("-" * 80)

        # Create explicit tasks for proper cancellation handling
        self._active_tasks = [
            asyncio.create_task(self._process_speaker_with_semaphore(speaker))
            for speaker in speakers
        ]

        # Process in parallel with progress bar
        results = []
        completed_count = 0
        try:
            with tqdm(total=len(self._active_tasks), desc="Text evidence collection (parallel)") as pbar:
                # Use asyncio.as_completed to update progress as tasks finish
                for future in asyncio.as_completed(self._active_tasks):
                    if self._shutdown_requested:
                        break
                    try:
                        result = await future
                        results.append(result)
                        self._update_stats_from_result(result)
                        completed_count += 1
                        pbar.update(1)
                    except asyncio.CancelledError:
                        # Task was cancelled during shutdown
                        continue
        except (asyncio.CancelledError, KeyboardInterrupt):
            logger.info("\nInterrupt received during processing")

        # If shutdown was requested, cancel remaining tasks
        if self._shutdown_requested:
            await self._cancel_pending_tasks()
            logger.info(f"✓ Graceful shutdown complete. Processed {completed_count}/{len(speakers)} speakers.")

    async def _run_batched(
        self,
        channel_id: int = None,
        project: str = None,
        projects: List[str] = None,
        batch_size: int = 50
    ):
        """
        Batch processing mode: process batch_size speakers, then requery for priority.

        This allows new high-priority content (recent uploads) to jump ahead of
        older backlog items between batches.
        """
        # Get total count upfront for progress visibility
        total_speakers = self._count_speakers_for_processing(channel_id, project, projects)
        total_batches = (total_speakers + batch_size - 1) // batch_size if total_speakers > 0 else 0

        logger.info(f"Total speakers to process: {total_speakers:,} ({total_batches} batches)")

        batch_num = 0
        total_processed = 0

        # Create a single progress bar for all speakers
        pbar = tqdm(total=total_speakers, desc="Processing", unit="spk")

        try:
            while not self._shutdown_requested:
                batch_num += 1

                # Query for next batch of highest-priority speakers
                speakers = self._get_speakers_for_processing(
                    channel_id, project, projects, limit=batch_size
                )

                if not speakers:
                    logger.info(f"No more speakers to process after {total_processed} total")
                    break

                # Show what we're processing
                unique_episodes = len(set(s['content_id'] for s in speakers))

                # Check if any are recent (last 30 days) vs backlog
                now = datetime.now(timezone.utc)
                recent_count = sum(
                    1 for s in speakers
                    if s.get('publish_date') and
                    (now - s['publish_date']).days < 30
                )

                # Update progress bar description with batch info
                pbar.set_description(f"Batch {batch_num}/{total_batches}")
                pbar.set_postfix_str(f"{recent_count} recent, {unique_episodes} episodes")

                # Create tasks for this batch
                self._active_tasks = [
                    asyncio.create_task(self._process_speaker_with_semaphore(speaker))
                    for speaker in speakers
                ]

                # Process batch
                batch_completed = 0
                try:
                    for future in asyncio.as_completed(self._active_tasks):
                        if self._shutdown_requested:
                            break
                        try:
                            result = await future
                            self._update_stats_from_result(result)
                            batch_completed += 1
                            total_processed += 1
                            pbar.update(1)
                        except asyncio.CancelledError:
                            continue
                except (asyncio.CancelledError, KeyboardInterrupt):
                    logger.info("\nInterrupt received during processing")
                    break

                # If we processed fewer than batch_size, we're done
                if len(speakers) < batch_size:
                    logger.info("Reached end of queue")
                    break
        finally:
            pbar.close()

        if self._shutdown_requested:
            await self._cancel_pending_tasks()

    def _update_stats_from_result(self, processed: Dict):
        """Update stats from a single processed speaker result."""
        self.stats['speakers_processed'] += 1

        if processed['error']:
            self.stats['errors'].append(processed['error'])
            return

        result = processed['result']
        if not result:
            return

        duration_pct = processed['duration_pct']
        evidence_found = result.get('evidence_found', False)
        evidence_type = result.get('evidence_type', 'none')
        identified_name = result.get('identified_name', 'unknown')
        reasoning = result.get('reasoning', '')[:80]

        if evidence_found and evidence_type != 'none':
            self.stats['evidence_found_certain'] += 1

            # Track by evidence type
            if evidence_type == 'self_intro':
                self.stats['evidence_type_self_intro'] += 1
            elif evidence_type == 'addressed':
                self.stats['evidence_type_addressed'] += 1
            elif evidence_type == 'introduced':
                self.stats['evidence_type_introduced'] += 1

            logger.info(
                f"  ✓ Speaker {processed['speaker_id']} ({duration_pct:.1f}%) → "
                f"{identified_name} ({evidence_type})"
            )
        else:
            if 'short utterance' in reasoning.lower():
                self.stats['short_utterance'] += 1
                logger.debug(
                    f"  - Speaker {processed['speaker_id']} ({duration_pct:.1f}%) → "
                    f"short utterance"
                )
            else:
                self.stats['evidence_none'] += 1
                logger.debug(
                    f"  ✗ Speaker {processed['speaker_id']} ({duration_pct:.1f}%) → "
                    f"no evidence"
                )

    def request_shutdown(self):
        """Request graceful shutdown - allows in-progress tasks to complete."""
        if not self._shutdown_requested:
            self._shutdown_requested = True
            logger.info("\n⚠️  Shutdown requested - waiting for in-progress tasks to complete...")
            logger.info("   (Press Ctrl+C again to force quit)")

    async def _cancel_pending_tasks(self):
        """Cancel tasks that haven't started yet, let in-progress ones finish."""
        cancelled_count = 0
        in_progress_count = 0

        for task in self._active_tasks:
            if not task.done():
                # Check if task is waiting on semaphore (not started yet)
                # vs actually running (has acquired semaphore)
                task.cancel()
                cancelled_count += 1

        if cancelled_count > 0:
            # Wait for all cancellations to complete
            await asyncio.gather(*self._active_tasks, return_exceptions=True)
            logger.info(f"   Cancelled {cancelled_count} pending tasks")

    def _print_summary(self):
        """Print summary statistics."""
        logger.info("\n" + "=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Speakers processed: {self.stats['speakers_processed']}")
        logger.info(f"Evidence found (certain): {self.stats['evidence_found_certain']}")
        logger.info(f"  - Self-introduction: {self.stats['evidence_type_self_intro']}")
        logger.info(f"  - Addressed by name: {self.stats['evidence_type_addressed']}")
        logger.info(f"  - Introduced: {self.stats['evidence_type_introduced']}")
        logger.info(f"No evidence: {self.stats['evidence_none']}")
        logger.info(f"Short utterance (skipped): {self.stats['short_utterance']}")

        if self.stats['speakers_processed'] > 0:
            hit_rate = self.stats['evidence_found_certain'] / self.stats['speakers_processed'] * 100
            logger.info(f"\nText evidence hit rate: {hit_rate:.1f}%")

        if self.stats['errors']:
            logger.warning(f"Errors: {len(self.stats['errors'])}")
        logger.info("=" * 80)


async def main():
    parser = argparse.ArgumentParser(
        description='Phase 2: Text evidence collection for speaker identification',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--channel-id', type=int, help='Process single channel')
    parser.add_argument('--project', type=str, help='Filter to project')
    parser.add_argument('--max-speakers', type=int, help='Max speakers to process')
    parser.add_argument('--apply', action='store_true', help='Apply changes (default: dry run)')
    parser.add_argument('--min-duration-pct', type=float, default=0.10,
                       help='Min duration %% (default: 0.10)')
    parser.add_argument('--min-quality', type=float, default=0.50,
                       help='Min embedding quality (default: 0.50)')
    parser.add_argument('--include-assigned', action='store_true',
                       help='Include speakers already assigned (for re-verification)')
    parser.add_argument('--include-processed', action='store_true',
                       help='Re-process speakers with existing text_evidence_status')
    parser.add_argument('--max-concurrent', type=int, default=15,
                       help='Max concurrent LLM requests (default: 15)')

    args = parser.parse_args()

    strategy = TextEvidenceCollectionStrategy(
        min_duration_pct=args.min_duration_pct,
        min_quality=args.min_quality,
        dry_run=not args.apply,
        max_speakers=args.max_speakers,
        skip_processed=not args.include_processed,
        include_assigned=args.include_assigned,
        max_concurrent=args.max_concurrent
    )

    # Set up signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()
    shutdown_count = [0]  # Use list to allow modification in closure

    def handle_shutdown_signal():
        shutdown_count[0] += 1
        if shutdown_count[0] == 1:
            strategy.request_shutdown()
        else:
            logger.info("\nForce quit requested")
            sys.exit(1)

    # Register signal handlers
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, handle_shutdown_signal)

    try:
        await strategy.run(
            channel_id=args.channel_id,
            project=args.project
        )
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Clean up HTTP client session
        await strategy.llm_client.close()


if __name__ == '__main__':
    asyncio.run(main())
