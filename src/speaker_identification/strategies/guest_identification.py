#!/usr/bin/env python3
"""
Guest Identification Strategy (Phase 2B)
=========================================

Identifies unassigned speakers using:
1. Known host from channel/episode
2. Episode metadata (title, description, expected guests)
3. Transcript context (6-turn pattern)
4. LLM identification

This is for episodes where we know the host but have unassigned speakers.
Unlike guest_propagation, this doesn't use embedding similarity - it's pure
transcript + metadata based identification.

Usage:
    # Run on single channel
    python -m src.speaker_identification.strategies.guest_identification \\
        --channel-id 6569 --apply

    # Run on project
    python -m src.speaker_identification.strategies.guest_identification \\
        --project CPRMV --apply

    # Dry run
    python -m src.speaker_identification.strategies.guest_identification \\
        --channel-id 6569
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from src.utils.paths import get_project_root, get_config_path
from typing import Dict, List, Optional

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

logger = setup_worker_logger('speaker_identification.guest_identification')


class GuestIdentificationStrategy:
    """
    Phase 2B: Identify guests using host context + metadata.

    For episodes with known hosts:
    1. Find unassigned speakers >= min_duration_pct
    2. Use LLM with transcript + metadata to identify
    3. Create/match identity and assign speaker
    """

    def __init__(
        self,
        min_duration_pct: float = 0.10,
        min_quality: float = 0.50,
        dry_run: bool = True,
        max_speakers: int = None,
        skip_llm_cached: bool = True,
        require_named_guests: bool = False,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ):
        """
        Initialize strategy.

        Args:
            min_duration_pct: Minimum % of episode for speaker consideration
            min_quality: Minimum speaker embedding quality score
            dry_run: If True, don't make DB changes
            max_speakers: Maximum speakers to process
            skip_llm_cached: Skip speakers with existing llm_identification
            require_named_guests: Only process episodes with content.guests populated
            start_date: Filter content published on or after this date (YYYY-MM-DD)
            end_date: Filter content published before this date (YYYY-MM-DD)
        """
        self.min_duration_pct = min_duration_pct
        self.min_quality = min_quality
        self.dry_run = dry_run
        self.max_speakers = max_speakers
        self.skip_llm_cached = skip_llm_cached
        self.require_named_guests = require_named_guests
        self.start_date = start_date
        self.end_date = end_date

        self.llm_client = MLXLLMClient()
        self.context_builder = ContextBuilder()
        self.identity_manager = IdentityManager()

        self.stats = {
            'speakers_processed': 0,
            'identified_certain': 0,
            'identified_very_likely': 0,
            'identified_probably': 0,
            'unknown': 0,
            'identities_created': 0,
            'identities_matched': 0,
            'speakers_assigned': 0,
            'llm_cached_skipped': 0,
            'errors': []
        }

    def _get_speakers_with_known_hosts(
        self,
        channel_id: int = None,
        project: str = None
    ) -> List[Dict]:
        """
        Get unassigned speakers from episodes with known hosts.

        Returns speakers where:
        - Episode has a host assigned (via speaker_identity with role='host')
        - Speaker is unassigned
        - Speaker meets duration/quality thresholds
        """
        with get_session() as session:
            filters = []
            params = {
                'min_duration_pct': self.min_duration_pct,
                'min_quality': self.min_quality
            }

            if channel_id:
                filters.append("he.channel_id = :channel_id")
                params['channel_id'] = channel_id

            if project:
                filters.append(":project = ANY(he.projects)")
                params['project'] = project

            if self.skip_llm_cached:
                # Use per-phase tracking: skip speakers already processed by Phase 3
                filters.append(f"s.speaker_identity_id IS NULL AND {PhaseTracker.get_phase_filter_sql(3, 's')}")

            if self.require_named_guests:
                filters.append("he.guests IS NOT NULL AND he.guests != '[]'::jsonb")

            # Date range filters
            if self.start_date:
                filters.append("he.publish_date >= :start_date")
                params['start_date'] = self.start_date
            if self.end_date:
                filters.append("he.publish_date < :end_date")
                params['end_date'] = self.end_date

            filter_clause = " AND ".join(filters) if filters else "TRUE"

            if self.max_speakers:
                params['limit'] = self.max_speakers
                limit_clause = "LIMIT :limit"
            else:
                limit_clause = ""

            query = text(f"""
                WITH host_episodes AS (
                    -- Episodes with at least one host assigned
                    SELECT DISTINCT
                        c.content_id,
                        c.title,
                        c.description,
                        c.duration as ep_duration,
                        c.channel_id,
                        c.projects,
                        c.guests,
                        c.hosts as content_hosts,
                        c.publish_date,
                        ch.display_name as channel_name,
                        si.primary_name as host_name
                    FROM content c
                    JOIN channels ch ON c.channel_id = ch.id
                    JOIN speakers host_s ON host_s.content_id = c.content_id
                    JOIN speaker_identities si ON host_s.speaker_identity_id = si.id
                    WHERE si.role = 'host'
                      AND c.is_stitched = true
                      AND c.duration > 0
                )
                SELECT
                    s.id as speaker_id,
                    s.embedding,
                    s.duration,
                    s.embedding_quality_score as quality,
                    he.content_id,
                    he.title,
                    he.description,
                    he.ep_duration,
                    he.channel_id,
                    he.channel_name,
                    he.host_name,
                    he.guests,
                    he.content_hosts
                FROM host_episodes he
                JOIN speakers s ON s.content_id = he.content_id
                WHERE s.speaker_identity_id IS NULL
                  AND {PhaseTracker.get_phase_filter_sql(3, 's')}
                  AND s.duration > 0
                  AND (s.duration / he.ep_duration) >= :min_duration_pct
                  AND COALESCE(s.embedding_quality_score, 1.0) >= :min_quality
                  AND {filter_clause}
                ORDER BY he.channel_id, s.duration DESC
                {limit_clause}
            """)

            results = session.execute(query, params).fetchall()
            return [dict(row._mapping) for row in results]

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

    def _parse_episode_guests(self, guests) -> List[str]:
        """
        Parse episode-level guests from content.guests JSONB field.

        Args:
            guests: JSONB data from content.guests (list of dicts or strings)

        Returns:
            List of guest names
        """
        if not guests:
            return []

        # Handle JSON string if not already parsed
        if isinstance(guests, str):
            try:
                guests = json.loads(guests)
            except json.JSONDecodeError:
                return []

        # Extract names from list of dicts or strings
        guest_names = []
        for g in guests:
            if isinstance(g, dict):
                name = g.get('name') or g.get('primary_name')
                if name:
                    guest_names.append(name)
            elif isinstance(g, str):
                guest_names.append(g)

        return guest_names

    async def _identify_guest(
        self,
        speaker: Dict
    ) -> Dict:
        """
        Use LLM to identify a guest speaker.

        Args:
            speaker: Speaker dict with episode context

        Returns:
            Dict with: speaker_name, role, confidence, reasoning
        """
        # Get transcript context
        context = self.context_builder.get_speaker_transcript_context(speaker['speaker_id'])
        if not context:
            return {
                'speaker_name': 'unknown',
                'role': 'unknown',
                'confidence': 'unlikely',
                'reasoning': 'No transcript context available'
            }

        duration_pct = (speaker['duration'] / speaker['ep_duration'] * 100)

        # Parse episode-level host and guest names from metadata
        # These come from content.hosts and content.guests (extracted by Phase 1B)
        episode_hosts = self._parse_episode_hosts(speaker.get('content_hosts'))
        expected_guests = self._parse_episode_guests(speaker.get('guests'))

        # Build combined list of possible speakers (hosts + guests, deduplicated)
        # Include the assigned host name plus any from episode metadata
        possible_speakers = [speaker['host_name']]  # Primary host from speaker_identities
        for h in episode_hosts:
            if h and h.lower() not in [p.lower() for p in possible_speakers]:
                possible_speakers.append(h)
        for g in expected_guests:
            if g and g.lower() not in [p.lower() for p in possible_speakers]:
                possible_speakers.append(g)

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

        def _truncate_middle(text: str, max_chars: int = 600) -> str:
            """Truncate long text, keeping start and end."""
            if not text or len(text) <= max_chars:
                return text
            half = max_chars // 2
            return f"{text[:half]}...{text[-half:]}"

        # Truncate unknown speaker utterances (can be very long)
        first_utterance_truncated = _truncate_middle(first_utterance, 600)
        last_utterance_truncated = _truncate_middle(last_utterance, 600)

        transcript_section = f"""
--- FIRST APPEARANCE ---
{_label(before_first_name, '[Speaker before]')}: ...{turn_before_first[-300:] if turn_before_first else 'N/A'}
[UNKNOWN SPEAKER]: {first_utterance_truncated}
{_label(after_first_name, '[Speaker after]')}: {turn_after_first[:300] if turn_after_first else 'N/A'}...

--- LAST APPEARANCE ---
{_label(before_last_name, '[Speaker before]')}: ...{turn_before_last[-300:] if turn_before_last else 'N/A'}
[UNKNOWN SPEAKER]: {last_utterance_truncated}
{_label(after_last_name, '[Speaker after]')}: {turn_after_last[:300] if turn_after_last else 'N/A'}..."""

        # Build prompt using registry
        prompt = PromptRegistry.phase3_guest_identification(
            episode_title=speaker['title'],
            episode_description=speaker['description'],
            possible_speakers=possible_speakers,
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

            return {
                'speaker_name': data.get('speaker_name', 'unknown'),
                'role': data.get('role', 'unknown'),
                'confidence': data.get('confidence', 'unlikely'),
                'reasoning': data.get('reasoning', ''),
                # New categorical evidence fields
                'evidence_type': evidence.get('type', 'none'),
                'evidence_source': evidence.get('speaker_source', 'unknown'),
                'evidence_quote': evidence.get('quote', '')
            }
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            return {
                'speaker_name': 'unknown',
                'role': 'unknown',
                'confidence': 'unlikely',
                'reasoning': f'JSON parse error: {e}',
                'evidence_type': 'none',
                'evidence_source': 'unknown',
                'evidence_quote': ''
            }

    def _save_llm_identification(
        self,
        speaker_id: int,
        result: Dict,
        method: str = 'guest_identification'
    ):
        """Save LLM identification result to speakers table using per-phase tracking."""
        if self.dry_run:
            return

        # Determine phase status based on result
        speaker_name = result.get('speaker_name', 'unknown')
        confidence = result.get('confidence', 'unlikely')

        if speaker_name == 'unknown':
            phase_status = PhaseTracker.STATUS_REJECTED
        elif confidence == 'probably':
            phase_status = PhaseTracker.STATUS_RETRY_ELIGIBLE
        elif confidence == 'unlikely':
            phase_status = PhaseTracker.STATUS_REJECTED
        else:
            # certain/very_likely - will be upgraded to 'assigned' by _create_or_assign_identity
            phase_status = PhaseTracker.STATUS_RETRY_ELIGIBLE

        # Record in per-phase JSONB structure
        PhaseTracker.record_result(
            speaker_id=speaker_id,
            phase=3,
            status=phase_status,
            result={
                'identified_name': result.get('speaker_name'),
                'role': result.get('role', 'unknown'),
                'confidence': confidence,
                'reasoning': result.get('reasoning', ''),
                # New categorical evidence fields
                'evidence_type': result.get('evidence_type', 'none'),
                'evidence_source': result.get('evidence_source', 'unknown'),
                'evidence_quote': result.get('evidence_quote', '')
            },
            method=method,
            dry_run=self.dry_run
        )

    def _create_or_assign_identity(
        self,
        speaker_id: int,
        speaker_name: str,
        role: str,
        confidence: str,
        channel_id: int
    ):
        """Create or match identity and assign speaker."""
        if self.dry_run:
            return

        # Map confidence to numeric value
        confidence_map = {
            'certain': 0.95,
            'very_likely': 0.85,
            'probably': 0.70,
            'unlikely': 0.50
        }
        conf_value = confidence_map.get(confidence, 0.50)

        # Try to find or create identity
        identity_id = self.identity_manager.create_or_match_identity(
            name=speaker_name,
            role=role,
            confidence=conf_value,
            method='guest_identification',
            metadata={'source_channel_id': channel_id}
        )

        if identity_id:
            # Check if this was a new creation or existing match
            with get_session() as session:
                result = session.execute(
                    text("SELECT created_at FROM speaker_identities WHERE id = :id"),
                    {'id': identity_id}
                ).fetchone()

                # If created very recently, it's new
                from datetime import timedelta, timezone
                now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
                if result and (now_utc - result.created_at) < timedelta(seconds=5):
                    self.stats['identities_created'] += 1
                else:
                    self.stats['identities_matched'] += 1

            # Assign speaker
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
                        'confidence': conf_value,
                        'method': f'guest_identification_{confidence}',
                        'status': IdentificationStatus.ASSIGNED
                    }
                )
                session.commit()

            # Also update phase status to 'assigned'
            PhaseTracker.record_result(
                speaker_id=speaker_id,
                phase=3,
                status=PhaseTracker.STATUS_ASSIGNED,
                result={
                    'identified_name': speaker_name,
                    'role': role,
                    'confidence': confidence
                },
                method=f'guest_identification_{confidence}',
                identity_id=identity_id,
                dry_run=self.dry_run
            )

            self.stats['speakers_assigned'] += 1

    async def run(
        self,
        channel_id: int = None,
        project: str = None
    ) -> Dict:
        """
        Run guest identification.

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
        logger.info("GUEST IDENTIFICATION (Phase 3)")
        logger.info("=" * 80)
        logger.info(f"Mode: {'DRY RUN' if self.dry_run else 'APPLY'}")
        logger.info(f"Min duration: {self.min_duration_pct * 100:.1f}%")
        logger.info(f"Require named guests: {self.require_named_guests}")
        logger.info(f"Skip already processed: {self.skip_llm_cached}")
        if self.start_date or self.end_date:
            logger.info(f"Date range: {self.start_date or 'any'} to {self.end_date or 'any'}")

        # Get speakers
        speakers = self._get_speakers_with_known_hosts(channel_id, project)
        if not speakers:
            logger.info("No speakers found with known hosts")
            return self.stats

        # Count unique episodes
        unique_episodes = len(set(s['content_id'] for s in speakers))
        logger.info(f"Processing {len(speakers)} speakers across {unique_episodes} episodes")

        # Show expected guests if available
        episodes_with_guests = sum(1 for s in speakers if s.get('guests'))
        if episodes_with_guests:
            logger.info(f"Episodes with named guests in metadata: {episodes_with_guests}")
        logger.info("-" * 80)

        # Process speakers
        current_episode = None
        for speaker in tqdm(speakers, desc="Guest identification"):
            self.stats['speakers_processed'] += 1

            try:
                duration_pct = (speaker['duration'] / speaker['ep_duration'] * 100)

                # Log episode change
                if speaker['content_id'] != current_episode:
                    current_episode = speaker['content_id']
                    expected_guests = []
                    if speaker.get('guests'):
                        guests_data = speaker['guests']
                        if isinstance(guests_data, str):
                            guests_data = json.loads(guests_data)
                        expected_guests = [g.get('name', g) if isinstance(g, dict) else g for g in guests_data]

                    logger.info(f"Episode: {speaker['title'][:70]}")
                    logger.info(f"  Host: {speaker['host_name']}")
                    if expected_guests:
                        logger.info(f"  Expected guests: {', '.join(expected_guests)}")

                # Call LLM
                result = await self._identify_guest(speaker)

                # Save LLM result
                self._save_llm_identification(speaker['speaker_id'], result)

                speaker_name = result.get('speaker_name', 'unknown')
                confidence = result.get('confidence', 'unlikely')
                role = result.get('role', 'unknown')
                reasoning = result.get('reasoning', '')[:80]

                # Track confidence stats
                if speaker_name != 'unknown':
                    if confidence == 'certain':
                        self.stats['identified_certain'] += 1
                    elif confidence == 'very_likely':
                        self.stats['identified_very_likely'] += 1
                    elif confidence == 'probably':
                        self.stats['identified_probably'] += 1
                    else:
                        self.stats['unknown'] += 1

                    # Only assign for high confidence
                    if confidence in ['certain', 'very_likely']:
                        logger.info(
                            f"  ✓ Speaker {speaker['speaker_id']} ({duration_pct:.1f}%) → "
                            f"{speaker_name} ({role}, {confidence})"
                        )
                        logger.info(f"    Reasoning: {reasoning}")
                        self._create_or_assign_identity(
                            speaker['speaker_id'],
                            speaker_name,
                            role,
                            confidence,
                            speaker['channel_id']
                        )
                    else:
                        logger.info(
                            f"  ? Speaker {speaker['speaker_id']} ({duration_pct:.1f}%) → "
                            f"{speaker_name} ({confidence}) - not assigned"
                        )
                else:
                    self.stats['unknown'] += 1
                    logger.info(
                        f"  ✗ Speaker {speaker['speaker_id']} ({duration_pct:.1f}%) → unknown"
                    )
                    if reasoning:
                        logger.info(f"    Reasoning: {reasoning}")

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
        logger.info(f"Identified (certain): {self.stats['identified_certain']}")
        logger.info(f"Identified (very_likely): {self.stats['identified_very_likely']}")
        logger.info(f"Identified (probably): {self.stats['identified_probably']}")
        logger.info(f"Unknown: {self.stats['unknown']}")
        logger.info(f"Identities created: {self.stats['identities_created']}")
        logger.info(f"Identities matched: {self.stats['identities_matched']}")
        logger.info(f"Speakers assigned: {self.stats['speakers_assigned']}")
        if self.stats['errors']:
            logger.warning(f"Errors: {len(self.stats['errors'])}")
        logger.info("=" * 80)


async def main():
    parser = argparse.ArgumentParser(
        description='Phase 2B: Guest identification via host context + metadata',
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
    parser.add_argument('--require-named-guests', action='store_true',
                       help='Only process episodes with content.guests populated')
    parser.add_argument('--include-cached', action='store_true',
                       help='Include speakers with existing LLM results')

    args = parser.parse_args()

    strategy = GuestIdentificationStrategy(
        min_duration_pct=args.min_duration_pct,
        min_quality=args.min_quality,
        dry_run=not args.apply,
        max_speakers=args.max_speakers,
        skip_llm_cached=not args.include_cached,
        require_named_guests=args.require_named_guests
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
    finally:
        # Clean up HTTP client session
        await strategy.llm_client.close()


if __name__ == '__main__':
    asyncio.run(main())
