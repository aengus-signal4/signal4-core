#!/usr/bin/env python3
"""
Phase 4 (NEW): Conflict Resolution via LLM
==========================================

Resolves name conflicts detected in Phase 3 (label propagation) using LLM verification.

This phase only runs when Phase 3 detected conflicts - where k-NN propagation
would have assigned conflicting names to similar voice clusters.

Pipeline Position:
    Phase 2: Text evidence collection → verified anchors
    Phase 3: Label propagation with conflict detection
    Phase 4: THIS - LLM resolution of actual conflicts

Key Innovation:
    Old approach: LLM-verify ALL ~2,746 name pairs with centroid similarity ≥ 0.70
    New approach: LLM-verify only ~10-50 pairs that ACTUALLY conflict during assignment

Input:
    config/propagation_conflicts.json (from Phase 3)

Output:
    - Updated name_aliases.yaml with resolved merges/do_not_merge
    - Updated speakers with resolved assignments

Usage:
    # Dry run (preview LLM decisions)
    python -m src.speaker_identification.strategies.conflict_resolution --project CPRMV

    # Apply
    python -m src.speaker_identification.strategies.conflict_resolution --project CPRMV --apply
"""

import argparse
import asyncio
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import yaml

from src.utils.paths import get_project_root
project_root = str(get_project_root())
sys.path.append(project_root)

from sqlalchemy import text

from src.database.session import get_session
from src.utils.logger import setup_worker_logger
from src.speaker_identification.core.llm_client import MLXLLMClient
from src.speaker_identification.prompts import PromptRegistry

# Paths
NAME_ALIASES_PATH = Path(project_root) / 'config' / 'name_aliases.yaml'
CONFLICTS_PATH = Path(project_root) / 'config' / 'propagation_conflicts.json'
CONFLICTS_RESOLVED_PATH = Path(project_root) / 'config' / 'propagation_conflicts_resolved.json'

logger = setup_worker_logger('speaker_identification.conflict_resolution')

# Console logging
import logging
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

PHASE_KEY = "phase4_resolution"


@dataclass
class ConflictResolutionConfig:
    """Configuration for conflict resolution."""
    # LLM settings
    llm_tier: str = "tier_1"
    llm_priority: int = 1  # Top priority
    llm_batch_size: int = 10

    # Auto-approve thresholds
    auto_approve_sim: float = 0.995  # Very high similarity
    auto_approve_patterns: bool = True


class ConflictResolutionStrategy:
    """
    Phase 4: Resolve name conflicts via LLM verification.

    Loads conflicts from Phase 3, verifies with LLM, updates aliases,
    and re-assigns conflicted speakers.
    """

    def __init__(
        self,
        config: ConflictResolutionConfig = None,
        dry_run: bool = True,
    ):
        self.config = config or ConflictResolutionConfig()
        self.dry_run = dry_run

        self.stats = {
            'conflicts_loaded': 0,
            'auto_approved': 0,
            'llm_verified': 0,
            'llm_same_person': 0,
            'llm_different_people': 0,
            'llm_needs_review': 0,
            'speakers_reassigned': 0,
            'llm_calls': 0,
            'errors': []
        }

        # Initialize LLM client
        self.llm_client = MLXLLMClient(tier=self.config.llm_tier)

    def _load_conflicts(self) -> List[Dict]:
        """Load conflicts from Phase 3 output file."""
        if not CONFLICTS_PATH.exists():
            logger.info(f"No conflicts file found at {CONFLICTS_PATH}")
            return []

        try:
            with open(CONFLICTS_PATH, 'r') as f:
                data = json.load(f)

            conflicts = data.get('conflicts', [])
            logger.info(f"Loaded {len(conflicts)} conflicts from Phase 3")
            logger.info(f"  Generated at: {data.get('generated_at', 'unknown')}")
            return conflicts

        except Exception as e:
            logger.error(f"Error loading conflicts: {e}")
            return []

    def _load_name_aliases(self) -> Tuple[Dict[str, str], Set[frozenset]]:
        """Load current name aliases and do-not-merge pairs."""
        alias_to_canonical = {}
        do_not_merge = set()

        if not NAME_ALIASES_PATH.exists():
            return alias_to_canonical, do_not_merge

        try:
            with open(NAME_ALIASES_PATH, 'r') as f:
                config = yaml.safe_load(f) or {}

            aliases = config.get('aliases', {})
            for canonical, alias_list in aliases.items():
                if alias_list:
                    for alias in alias_list:
                        alias_to_canonical[alias.lower()] = canonical

            dnm_list = config.get('do_not_merge', [])
            if dnm_list:
                for pair in dnm_list:
                    if isinstance(pair, list) and len(pair) >= 2:
                        do_not_merge.add(frozenset(n.lower() for n in pair))

        except Exception as e:
            logger.warning(f"Error loading aliases: {e}")

        return alias_to_canonical, do_not_merge

    def _is_auto_approvable(self, name_a: str, name_b: str, similarity: float) -> Tuple[bool, str]:
        """Check if a name pair can be auto-approved without LLM."""
        if not self.config.auto_approve_patterns:
            return False, ""

        # Very high similarity
        if similarity >= self.config.auto_approve_sim:
            return True, "very_high_similarity"

        # Title prefixes
        title_prefixes = ['Dr. ', 'Prof. ', 'The Honourable ', 'Hon. ', 'Rev. ', 'Pastor ', 'Sir ']
        for prefix in title_prefixes:
            if name_a.startswith(prefix) and name_a[len(prefix):] == name_b:
                return True, "title_prefix"
            if name_b.startswith(prefix) and name_b[len(prefix):] == name_a:
                return True, "title_prefix"

        # Exact match ignoring case
        if name_a.lower() == name_b.lower():
            return True, "case_match"

        return False, ""

    def _get_speaker_samples(self, name: str, speaker_ids: List[int]) -> Dict:
        """Get sample transcript excerpts for a name's speakers."""
        samples = {
            'hours': 0.0,
            'episodes': 0,
            'sample_text': ''
        }

        with get_session() as session:
            # Get aggregate stats
            result = session.execute(text("""
                SELECT
                    COUNT(DISTINCT s.content_id) as episodes,
                    SUM(s.duration) / 3600.0 as hours
                FROM speakers s
                WHERE s.id = ANY(:speaker_ids)
            """), {'speaker_ids': speaker_ids[:100]}).fetchone()

            if result:
                samples['hours'] = float(result.hours or 0)
                samples['episodes'] = int(result.episodes or 0)

            # Get a sample transcript
            if speaker_ids:
                sample_result = session.execute(text("""
                    SELECT st.text
                    FROM speaker_transcriptions st
                    JOIN content c ON st.content_id = c.id
                    JOIN speakers s ON c.content_id = s.content_id AND st.speaker_id = s.id
                    WHERE s.id = ANY(:speaker_ids)
                      AND LENGTH(st.text) > 50
                    ORDER BY st.end_time - st.start_time DESC
                    LIMIT 1
                """), {'speaker_ids': speaker_ids[:10]}).fetchone()

                if sample_result:
                    samples['sample_text'] = sample_result.text[:200]

        return samples

    async def _verify_conflicts_with_llm(
        self,
        conflicts: List[Dict]
    ) -> List[Dict]:
        """Verify conflicts using LLM in batches."""
        decisions = []

        # Build batch data
        batch_data = []
        for conflict in conflicts:
            name_a = conflict['name_a']
            name_b = conflict['name_b']

            # Get speaker samples for both names
            speaker_ids_a = conflict.get('sample_speaker_ids', [])
            speaker_ids_b = []  # We'll need to query for the other name's speakers

            # Query for speakers with each name
            with get_session() as session:
                # Get speakers for name_a
                result_a = session.execute(text("""
                    SELECT s.id, s.content_id
                    FROM speakers s
                    WHERE s.identification_details->'phase2'->>'identified_name' = :name
                       OR s.identification_details->'phase3_propagation'->>'primary_label' = :name
                    LIMIT 50
                """), {'name': name_a}).fetchall()
                speaker_ids_a = [r.id for r in result_a]
                content_ids_a = set(r.content_id for r in result_a)

                # Get speakers for name_b
                result_b = session.execute(text("""
                    SELECT s.id, s.content_id
                    FROM speakers s
                    WHERE s.identification_details->'phase2'->>'identified_name' = :name
                       OR s.identification_details->'phase3_propagation'->>'primary_label' = :name
                    LIMIT 50
                """), {'name': name_b}).fetchall()
                speaker_ids_b = [r.id for r in result_b]
                content_ids_b = set(r.content_id for r in result_b)

            # Check for co-occurrence
            shared_episodes = len(content_ids_a & content_ids_b)

            samples_a = self._get_speaker_samples(name_a, speaker_ids_a)
            samples_b = self._get_speaker_samples(name_b, speaker_ids_b)

            batch_data.append({
                'name_a': name_a,
                'name_b': name_b,
                'similarity': conflict.get('centroid_similarity', 0),
                'hours_a': samples_a['hours'],
                'hours_b': samples_b['hours'],
                'episodes_a': samples_a['episodes'],
                'episodes_b': samples_b['episodes'],
                'shared_episodes': shared_episodes,
                'sample_a': samples_a['sample_text'],
                'sample_b': samples_b['sample_text'],
                'conflict_type': conflict.get('conflict_type', 'unknown'),
                'affected_speakers': conflict.get('affected_speakers', 0),
            })

        # Process in batches
        total_batches = (len(batch_data) + self.config.llm_batch_size - 1) // self.config.llm_batch_size

        for batch_start in range(0, len(batch_data), self.config.llm_batch_size):
            batch = batch_data[batch_start:batch_start + self.config.llm_batch_size]
            batch_num = batch_start // self.config.llm_batch_size + 1
            logger.info(f"  LLM batch {batch_num}/{total_batches} ({len(batch)} conflicts)...")

            # Build prompt
            prompt = PromptRegistry.phase3_name_pair_batch(batch)

            try:
                self.stats['llm_calls'] += 1
                response = await self.llm_client._call_llm(prompt, priority=self.config.llm_priority)

                # Parse JSON response
                json_match = re.search(r'\[[\s\S]*\]', response)
                if json_match:
                    llm_decisions = json.loads(json_match.group())

                    for i, decision in enumerate(llm_decisions):
                        if i < len(batch):
                            decisions.append({
                                'name_a': batch[i]['name_a'],
                                'name_b': batch[i]['name_b'],
                                'decision': decision.get('decision', 'needs_review'),
                                'canonical_name': decision.get('canonical_name'),
                                'confidence': decision.get('confidence', 'low'),
                                'reasoning': decision.get('reasoning', ''),
                                'conflict_type': batch[i]['conflict_type'],
                                'affected_speakers': batch[i]['affected_speakers'],
                            })

                            # Update stats
                            d = decision.get('decision', 'needs_review')
                            if d == 'same_person':
                                self.stats['llm_same_person'] += 1
                            elif d == 'different_people':
                                self.stats['llm_different_people'] += 1
                            else:
                                self.stats['llm_needs_review'] += 1
                else:
                    logger.warning(f"Could not parse LLM response for batch {batch_num}")
                    for p in batch:
                        decisions.append({
                            'name_a': p['name_a'],
                            'name_b': p['name_b'],
                            'decision': 'needs_review',
                            'canonical_name': None,
                            'confidence': 'low',
                            'reasoning': 'LLM parse error',
                            'conflict_type': p['conflict_type'],
                            'affected_speakers': p['affected_speakers'],
                        })
                        self.stats['llm_needs_review'] += 1

            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                for p in batch:
                    decisions.append({
                        'name_a': p['name_a'],
                        'name_b': p['name_b'],
                        'decision': 'needs_review',
                        'canonical_name': None,
                        'confidence': 'low',
                        'reasoning': f'LLM error: {str(e)[:50]}',
                        'conflict_type': p['conflict_type'],
                        'affected_speakers': p['affected_speakers'],
                    })
                    self.stats['llm_needs_review'] += 1

            logger.info(f"    Batch {batch_num}: "
                       f"{len([d for d in decisions[-len(batch):] if d['decision'] == 'same_person'])} same, "
                       f"{len([d for d in decisions[-len(batch):] if d['decision'] == 'different_people'])} different, "
                       f"{len([d for d in decisions[-len(batch):] if d['decision'] == 'needs_review'])} review")

        return decisions

    def _update_aliases_file(
        self,
        same_person: List[Dict],
        different_people: List[Dict]
    ) -> None:
        """Update name_aliases.yaml with LLM decisions."""
        if self.dry_run:
            logger.info("  [DRY RUN] Would update aliases file")
            return

        try:
            # Load current config
            if NAME_ALIASES_PATH.exists():
                with open(NAME_ALIASES_PATH, 'r') as f:
                    aliases_config = yaml.safe_load(f) or {}
            else:
                aliases_config = {}

            if 'aliases' not in aliases_config:
                aliases_config['aliases'] = {}
            if 'do_not_merge' not in aliases_config:
                aliases_config['do_not_merge'] = []

            # Add same-person merges
            for decision in same_person:
                canonical = decision.get('canonical_name') or decision['name_a']
                variant = decision['name_b'] if canonical == decision['name_a'] else decision['name_a']

                if canonical not in aliases_config['aliases']:
                    aliases_config['aliases'][canonical] = []
                if variant not in aliases_config['aliases'][canonical]:
                    aliases_config['aliases'][canonical].append(variant)
                    logger.info(f"  [MERGE] {variant} -> {canonical}")

            # Add different-people to do_not_merge
            for decision in different_people:
                dnm_pair = [decision['name_a'], decision['name_b']]
                exists = any(
                    set(p) == set(dnm_pair)
                    for p in aliases_config['do_not_merge']
                )
                if not exists:
                    aliases_config['do_not_merge'].append(dnm_pair)
                    logger.info(f"  [DO-NOT-MERGE] {decision['name_a']} / {decision['name_b']}")

            # Write updated config
            with open(NAME_ALIASES_PATH, 'w') as f:
                yaml.dump(aliases_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

            logger.info(f"  Updated {NAME_ALIASES_PATH}")

        except Exception as e:
            logger.error(f"  Failed to update aliases: {e}")
            self.stats['errors'].append(str(e))

    def _reassign_conflicted_speakers(
        self,
        same_person: List[Dict],
        different_people: List[Dict]
    ) -> None:
        """Re-assign speakers that were marked as conflicted."""
        if self.dry_run:
            logger.info("  [DRY RUN] Would reassign conflicted speakers")
            return

        timestamp = datetime.now(timezone.utc).isoformat()

        with get_session() as session:
            # For same-person merges: update to use canonical name
            for decision in same_person:
                canonical = decision.get('canonical_name') or decision['name_a']
                variant = decision['name_b'] if canonical == decision['name_a'] else decision['name_a']

                # Get or create identity for canonical name
                identity_result = session.execute(text("""
                    SELECT id FROM speaker_identities
                    WHERE LOWER(primary_name) = LOWER(:name)
                      AND is_active = TRUE
                    ORDER BY id LIMIT 1
                """), {'name': canonical}).fetchone()

                if identity_result:
                    identity_id = identity_result.id
                else:
                    # Create new identity
                    result = session.execute(text("""
                        INSERT INTO speaker_identities (
                            primary_name, verification_status, is_active,
                            created_at, updated_at
                        ) VALUES (
                            :name, 'conflict_resolved', TRUE, NOW(), NOW()
                        )
                        RETURNING id
                    """), {'name': canonical})
                    identity_id = result.fetchone()[0]

                # Update speakers with conflict involving these names
                session.execute(text(f"""
                    UPDATE speakers SET
                        speaker_identity_id = :identity_id,
                        assignment_phase = '{PHASE_KEY}',
                        identification_details = jsonb_set(
                            COALESCE(identification_details, '{{}}'::jsonb),
                            ARRAY['{PHASE_KEY}'],
                            :phase_entry::jsonb
                        ),
                        updated_at = NOW()
                    WHERE identification_details->'phase3_propagation'->>'status' = 'conflict'
                      AND (
                          identification_details->'phase3_propagation'->>'primary_label' IN (:name_a, :name_b)
                          OR identification_details->'phase3_propagation'->>'conflict_label' IN (:name_a, :name_b)
                      )
                """), {
                    'identity_id': identity_id,
                    'phase_entry': json.dumps({
                        'status': 'resolved',
                        'timestamp': timestamp,
                        'resolution': 'same_person',
                        'canonical_name': canonical,
                        'identity_id': identity_id
                    }),
                    'name_a': decision['name_a'],
                    'name_b': decision['name_b']
                })

                self.stats['speakers_reassigned'] += 1

            # For different-people: assign to their respective identities
            for decision in different_people:
                for name in [decision['name_a'], decision['name_b']]:
                    # Get or create identity
                    identity_result = session.execute(text("""
                        SELECT id FROM speaker_identities
                        WHERE LOWER(primary_name) = LOWER(:name)
                          AND is_active = TRUE
                        ORDER BY id LIMIT 1
                    """), {'name': name}).fetchone()

                    if identity_result:
                        identity_id = identity_result.id
                    else:
                        result = session.execute(text("""
                            INSERT INTO speaker_identities (
                                primary_name, verification_status, is_active,
                                created_at, updated_at
                            ) VALUES (
                                :name, 'conflict_resolved', TRUE, NOW(), NOW()
                            )
                            RETURNING id
                        """), {'name': name})
                        identity_id = result.fetchone()[0]

                    # Update speakers with this primary label
                    session.execute(text(f"""
                        UPDATE speakers SET
                            speaker_identity_id = :identity_id,
                            assignment_phase = '{PHASE_KEY}',
                            identification_details = jsonb_set(
                                COALESCE(identification_details, '{{}}'::jsonb),
                                ARRAY['{PHASE_KEY}'],
                                :phase_entry::jsonb
                            ),
                            updated_at = NOW()
                        WHERE identification_details->'phase3_propagation'->>'status' = 'conflict'
                          AND identification_details->'phase3_propagation'->>'primary_label' = :name
                    """), {
                        'identity_id': identity_id,
                        'phase_entry': json.dumps({
                            'status': 'resolved',
                            'timestamp': timestamp,
                            'resolution': 'different_people',
                            'assigned_name': name,
                            'identity_id': identity_id
                        }),
                        'name': name
                    })

            session.commit()

    def _write_resolved_conflicts(self, decisions: List[Dict]) -> None:
        """Write resolved conflicts to file for audit."""
        output = {
            'resolved_at': datetime.now(timezone.utc).isoformat(),
            'total_resolved': len(decisions),
            'same_person': [d for d in decisions if d['decision'] == 'same_person'],
            'different_people': [d for d in decisions if d['decision'] == 'different_people'],
            'needs_review': [d for d in decisions if d['decision'] == 'needs_review'],
        }

        try:
            with open(CONFLICTS_RESOLVED_PATH, 'w') as f:
                json.dump(output, f, indent=2)
            logger.info(f"  Resolved conflicts written to: {CONFLICTS_RESOLVED_PATH}")
        except Exception as e:
            logger.warning(f"  Failed to write resolved conflicts: {e}")

    async def run(self, project: str = None) -> Dict:
        """Run conflict resolution."""
        logger.info("=" * 80)
        logger.info("PHASE 4: CONFLICT RESOLUTION VIA LLM")
        logger.info("=" * 80)
        logger.info(f"Mode: {'DRY RUN' if self.dry_run else 'APPLY'}")
        logger.info("-" * 80)

        # Step 1: Load conflicts
        logger.info("Loading conflicts from Phase 3...")
        conflicts = self._load_conflicts()
        self.stats['conflicts_loaded'] = len(conflicts)

        if not conflicts:
            logger.info("No conflicts to resolve - Phase 4 complete!")
            return self.stats

        # Step 2: Load current aliases (to skip already-resolved)
        alias_to_canonical, do_not_merge = self._load_name_aliases()

        # Filter out already-resolved conflicts
        unresolved = []
        for conflict in conflicts:
            pair_key = frozenset([conflict['name_a'].lower(), conflict['name_b'].lower()])

            # Check if already in do_not_merge
            if pair_key in do_not_merge:
                logger.info(f"  Skipping {conflict['name_a']} / {conflict['name_b']} - already in do_not_merge")
                continue

            # Check if already merged via aliases
            a_canonical = alias_to_canonical.get(conflict['name_a'].lower(), conflict['name_a'].lower())
            b_canonical = alias_to_canonical.get(conflict['name_b'].lower(), conflict['name_b'].lower())
            if a_canonical == b_canonical:
                logger.info(f"  Skipping {conflict['name_a']} / {conflict['name_b']} - already merged")
                continue

            unresolved.append(conflict)

        logger.info(f"  {len(unresolved)} conflicts need resolution")

        if not unresolved:
            logger.info("All conflicts already resolved - Phase 4 complete!")
            return self.stats

        # Step 3: Check for auto-approvable conflicts
        auto_approved = []
        llm_needed = []

        for conflict in unresolved:
            is_auto, reason = self._is_auto_approvable(
                conflict['name_a'],
                conflict['name_b'],
                conflict.get('centroid_similarity', 0)
            )
            if is_auto:
                auto_approved.append({
                    **conflict,
                    'decision': 'same_person',
                    'canonical_name': conflict['name_a'],  # Pick first as canonical
                    'confidence': 'auto',
                    'reasoning': reason
                })
                self.stats['auto_approved'] += 1
            else:
                llm_needed.append(conflict)

        logger.info(f"  {len(auto_approved)} auto-approved, {len(llm_needed)} need LLM verification")

        # Step 4: LLM verification
        llm_decisions = []
        if llm_needed:
            logger.info("-" * 80)
            logger.info("LLM VERIFICATION")
            logger.info("-" * 80)
            llm_decisions = await self._verify_conflicts_with_llm(llm_needed)
            self.stats['llm_verified'] = len(llm_decisions)

        # Combine all decisions
        all_decisions = auto_approved + llm_decisions

        # Step 5: Categorize results
        same_person = [d for d in all_decisions if d['decision'] == 'same_person' and d.get('confidence') in ['auto', 'certain', 'high']]
        different_people = [d for d in all_decisions if d['decision'] == 'different_people']
        needs_review = [d for d in all_decisions if d['decision'] == 'needs_review' or (d['decision'] == 'same_person' and d.get('confidence') not in ['auto', 'certain', 'high'])]

        logger.info("-" * 80)
        logger.info("RESULTS")
        logger.info("-" * 80)
        logger.info(f"  Same person (will merge): {len(same_person)}")
        logger.info(f"  Different people (do-not-merge): {len(different_people)}")
        logger.info(f"  Needs human review: {len(needs_review)}")

        if same_person:
            logger.info("")
            logger.info("  MERGES:")
            for d in same_person[:10]:
                logger.info(f"    • {d['name_a']} = {d['name_b']} ({d.get('reasoning', '')})")
            if len(same_person) > 10:
                logger.info(f"    ... and {len(same_person) - 10} more")

        if different_people:
            logger.info("")
            logger.info("  DO-NOT-MERGE:")
            for d in different_people[:10]:
                logger.info(f"    • {d['name_a']} ≠ {d['name_b']} ({d.get('reasoning', '')})")

        # Step 6: Update aliases file
        logger.info("-" * 80)
        logger.info("UPDATING ALIASES")
        logger.info("-" * 80)
        self._update_aliases_file(same_person, different_people)

        # Step 7: Reassign conflicted speakers
        logger.info("-" * 80)
        logger.info("REASSIGNING SPEAKERS")
        logger.info("-" * 80)
        self._reassign_conflicted_speakers(same_person, different_people)

        # Step 8: Write resolved conflicts
        self._write_resolved_conflicts(all_decisions)

        # Print summary
        self._print_summary(needs_review)

        return self.stats

    def _print_summary(self, needs_review: List[Dict]):
        """Print summary statistics."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)
        logger.info(f"  Conflicts loaded: {self.stats['conflicts_loaded']}")
        logger.info(f"  Auto-approved: {self.stats['auto_approved']}")
        logger.info(f"  LLM verified: {self.stats['llm_verified']}")
        logger.info(f"    - Same person: {self.stats['llm_same_person']}")
        logger.info(f"    - Different people: {self.stats['llm_different_people']}")
        logger.info(f"    - Needs review: {self.stats['llm_needs_review']}")
        logger.info(f"  LLM calls: {self.stats['llm_calls']}")
        logger.info(f"  Speakers reassigned: {self.stats['speakers_reassigned']}")

        if needs_review:
            logger.info("")
            logger.info("⚠️  ITEMS NEEDING HUMAN REVIEW:")
            for d in needs_review:
                logger.info(f"    • {d['name_a']} vs {d['name_b']}: {d.get('reasoning', 'uncertain')}")

        if self.stats['errors']:
            logger.info("")
            logger.info(f"⚠️  Errors: {len(self.stats['errors'])}")

        logger.info("=" * 80)


async def main():
    parser = argparse.ArgumentParser(
        description='Phase 4: Conflict Resolution via LLM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (preview LLM decisions)
  python -m src.speaker_identification.strategies.conflict_resolution --project CPRMV

  # Apply
  python -m src.speaker_identification.strategies.conflict_resolution --project CPRMV --apply
"""
    )

    parser.add_argument('--project', type=str, help='Project (for filtering, optional)')
    parser.add_argument('--apply', action='store_true', help='Apply changes (default: dry run)')
    parser.add_argument('--llm-batch-size', type=int, default=10, help='Conflicts per LLM call')

    args = parser.parse_args()

    config = ConflictResolutionConfig(
        llm_batch_size=args.llm_batch_size
    )

    strategy = ConflictResolutionStrategy(
        config=config,
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


if __name__ == '__main__':
    asyncio.run(main())
