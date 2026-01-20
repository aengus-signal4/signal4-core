#!/usr/bin/env python3
"""
Speaker Identification Orchestrator
====================================

Runs the full speaker identification pipeline.

PIPELINE PHASES:
================
Phase 1: Metadata Extraction
    - 1A: Extract hosts from channel descriptions
    - 1B: Extract speakers/guests from episode metadata
    → Populates: channels.hosts, content.hosts, content.guests, content.mentioned

Phase 2: Text Evidence Collection
    - Process ALL speakers with ≥10% duration
    - LLM finds text evidence (self-ID, addressed, introduced)
    - Binary output: "certain" or "none"
    → Populates: speakers.identification_details['phase2']

Phase 3: Speaker Clustering (three modes available via --phase3-mode)
    Mode 'anchor': Anchor-Canopy Clustering
        - Text-verified speakers become anchors
        - Multi-gate validation for cluster expansion
        - More conservative, higher precision
    Mode 'propagation': Label Propagation (legacy)
        - k-NN weighted voting from labeled neighbors
        - Confidence scores based on neighbor agreement
        - Faster, handles gradual transitions better
    Mode 'propagation-conflicts' (default, RECOMMENDED):
        - k-NN label propagation WITH conflict detection
        - Only flags actual conflicts for Phase 4 LLM verification
        - Much more efficient than LLM-verifying all name pairs
    → Populates: speaker_identities, speakers.speaker_identity_id

Phase 4: Conflict Resolution / Identity Merge Detection
    If Phase 3 used 'propagation-conflicts':
        - LLM-verifies only the actual conflicts detected in Phase 3
        - Much faster than old approach (10-50 pairs vs 2,746)
    Otherwise (legacy):
        - Find duplicate identities via centroid similarity
        - LLM verification + merge
    → Updates name_aliases.yaml, reassigns conflicted speakers

Phase 5: Speaker Hydration
    - Retrieve detailed biographical information about identified speakers
    - Uses external APIs (LinkedIn, etc.) to enrich speaker profiles
    → Populates: speaker_identities.bio, occupation, organization, social_profiles, etc.

Usage:
    # Dry run on project (recommended: uses new conflict-based pipeline)
    python -m src.speaker_identification.orchestrator --project CPRMV

    # Apply all phases
    python -m src.speaker_identification.orchestrator --project CPRMV --apply

    # Run specific phases
    python -m src.speaker_identification.orchestrator --project CPRMV --phases 2,3,4 --apply

    # Use legacy propagation mode (without conflict detection)
    python -m src.speaker_identification.orchestrator --project CPRMV --phase3-mode propagation

    # Re-verify all legacy assignments
    python -m src.speaker_identification.orchestrator --project CPRMV --include-assigned --apply
"""

import argparse
import asyncio
import signal
import sys
from datetime import datetime
from pathlib import Path
from src.utils.paths import get_project_root, get_config_path
from typing import List, Optional, Dict

from tqdm import tqdm

project_root = str(get_project_root())
sys.path.append(project_root)

from src.speaker_identification.strategies.metadata_identification import MetadataIdentificationStrategy
from src.speaker_identification.strategies.identity_merge_detection import IdentityMergeDetectionStrategy
from src.speaker_identification.strategies.text_evidence_collection import TextEvidenceCollectionStrategy
from src.speaker_identification.strategies.anchor_verified_clustering import AnchorVerifiedClusteringStrategy
from src.speaker_identification.strategies.label_propagation_clustering import LabelPropagationStrategy
from src.speaker_identification.strategies.label_propagation_with_conflicts import LabelPropagationWithConflictsStrategy
from src.speaker_identification.strategies.conflict_resolution import ConflictResolutionStrategy
from src.speaker_identification.strategies.speaker_hydration import SpeakerHydrationStrategy
from src.database.session import get_session
from src.utils.logger import setup_worker_logger
from sqlalchemy import text

logger = setup_worker_logger('speaker_identification.orchestrator')

# Console logging
import logging
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)


class SpeakerIdentificationOrchestrator:
    """
    Orchestrates the full speaker identification pipeline.
    """

    def __init__(
        self,
        dry_run: bool = True,
        phases: List[int] = None,
        max_channels: int = None,
        max_episodes: int = None,
        max_speakers: int = None,
        min_duration_pct: float = 0.10,
        start_channel_id: int = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        include_assigned: bool = False,
        max_concurrent: int = 15,
        batch_size: int = None,
        max_anchors: int = None,
        phase3_mode: str = 'propagation-conflicts'
    ):
        """
        Initialize orchestrator.

        Args:
            dry_run: If True, don't make DB changes
            phases: List of phases to run (default: 1,2,3,4,5)
            max_channels: Max channels to process
            max_episodes: Max episodes per channel (Phase 1)
            max_speakers: Max speakers per phase
            min_duration_pct: Minimum speaker duration %
            start_channel_id: Resume from this channel ID
            start_date: Filter content published on or after this date (YYYY-MM-DD)
            end_date: Filter content published before this date (YYYY-MM-DD)
            include_assigned: If True, re-verify speakers already assigned
            max_concurrent: Max concurrent LLM requests for Phase 2 (default: 15)
            batch_size: Process N speakers per batch, then recheck priority (default: None = all at once)
            max_anchors: Max anchor name groups to process in Phase 3 (for testing)
            phase3_mode: Phase 3 algorithm:
                'propagation-conflicts' (default): k-NN propagation with conflict detection (RECOMMENDED)
                'anchor': anchor-canopy clustering
                'propagation': legacy label propagation without conflict detection
        """
        self.dry_run = dry_run
        self.include_assigned = include_assigned
        self.max_concurrent = max_concurrent
        self.batch_size = batch_size
        self.max_anchors = max_anchors
        self.phase3_mode = phase3_mode

        # Default phases
        if phases is None:
            self.phases = [1, 2, 3, 4, 5]
        else:
            self.phases = phases

        self.max_channels = max_channels
        self.max_episodes = max_episodes
        self.max_speakers = max_speakers
        self.min_duration_pct = min_duration_pct
        self.start_channel_id = start_channel_id
        self.start_date = start_date
        self.end_date = end_date

        self.stats = {
            'phase1': {},
            'phase2': {},
            'phase3': {},
            'phase4': {},  # Identity Merge Detection (was Phase 5)
            'phase5': {},  # Speaker Hydration (new)
            'total_time': 0
        }

        # Track current strategy for graceful shutdown
        self._current_strategy = None
        self._shutdown_requested = False

    def request_shutdown(self):
        """Request graceful shutdown - propagates to current strategy."""
        if not self._shutdown_requested:
            self._shutdown_requested = True
            logger.info("\n⚠️  Shutdown requested - completing current phase...")
            if self._current_strategy and hasattr(self._current_strategy, 'request_shutdown'):
                self._current_strategy.request_shutdown()

    def _get_channels(self, project: str = None, channel_id: int = None) -> List[Dict]:
        """Get channels to process."""
        with get_session() as session:
            if channel_id:
                query = text("""
                    SELECT
                        c.id,
                        c.display_name,
                        c.platform,
                        c.hosts,
                        COUNT(DISTINCT co.content_id) as episode_count
                    FROM channels c
                    JOIN content co ON c.id = co.channel_id
                    WHERE c.id = :channel_id
                      AND c.status = 'active'
                    GROUP BY c.id
                """)
                results = session.execute(query, {'channel_id': channel_id}).fetchall()
            else:
                query = text("""
                    SELECT
                        c.id,
                        c.display_name,
                        c.platform,
                        c.hosts,
                        COUNT(DISTINCT co.content_id) as episode_count
                    FROM channels c
                    JOIN content co ON c.id = co.channel_id
                    WHERE c.status = 'active'
                      AND :project = ANY(co.projects)
                      AND (:start_id IS NULL OR c.id >= :start_id)
                    GROUP BY c.id
                    ORDER BY c.id
                """)
                results = session.execute(query, {
                    'project': project,
                    'start_id': self.start_channel_id
                }).fetchall()

            channels = [dict(row._mapping) for row in results]

            if self.max_channels:
                channels = channels[:self.max_channels]

            return channels

    async def run_phase1(self, projects: List[str] = None, channel_id: int = None):
        """
        Phase 1: Metadata extraction.

        Extracts hosts from channel descriptions and speakers from episode metadata.

        When batch_size is set:
        - Processes episodes by priority (recent first, then project priority)
        - Runs Phase 1A on-demand when first encountering a channel
        - Runs Phase 1C consolidation after each batch
        - Re-queries for priority updates between batches
        """
        logger.info("")
        logger.info("=" * 80)
        logger.info("PHASE 1: METADATA EXTRACTION")
        logger.info("=" * 80)

        strategy = MetadataIdentificationStrategy(
            dry_run=self.dry_run,
            max_episodes=self.max_episodes,
            min_duration_pct=self.min_duration_pct,
            start_date=self.start_date,
            end_date=self.end_date
        )

        # Track for graceful shutdown
        self._current_strategy = strategy
        if self._shutdown_requested:
            strategy.request_shutdown()

        if channel_id:
            await strategy.run_single_channel(channel_id)
        elif projects:
            # Run on all active projects at once
            # Use batch_size if provided for priority-based processing
            await strategy.run_all_projects_filtered(
                projects=projects,
                max_channels=self.max_channels,
                batch_size=self.batch_size
            )
        else:
            # Fallback to run_all_projects which reads from config
            await strategy.run_all_projects(
                max_channels=self.max_channels
            )

        self._current_strategy = None
        self.stats['phase1'] = strategy.stats
        return strategy.stats

    async def run_phase2(self, project: str = None, projects: List[str] = None, channel_id: int = None):
        """
        Phase 2: Text Evidence Collection.

        Processes ALL speakers with ≥10% duration to find text-based evidence
        (self-introduction, being addressed, being introduced).

        This creates the "ground truth" for building clean centroids.
        Uses parallel LLM processing to take advantage of multiple providers.

        Prioritization (when batch_size is set):
        1. Recent content (last 30 days) - processed first regardless of project
        2. Then by project priority (higher priority projects first)
        3. After each batch, re-queries to allow new content to jump the queue

        Args:
            project: Single project to filter to
            projects: List of projects to process universally (shows total progress)
            channel_id: Single channel to process
        """
        logger.info("")
        logger.info("=" * 80)
        logger.info("PHASE 2: TEXT EVIDENCE COLLECTION")
        logger.info("=" * 80)

        strategy = TextEvidenceCollectionStrategy(
            min_duration_pct=self.min_duration_pct,
            dry_run=self.dry_run,
            max_speakers=self.max_speakers,
            include_assigned=self.include_assigned,
            start_date=self.start_date,
            end_date=self.end_date,
            max_concurrent=self.max_concurrent
        )

        # Track for graceful shutdown
        self._current_strategy = strategy
        if self._shutdown_requested:
            strategy.request_shutdown()

        stats = await strategy.run(
            channel_id=channel_id,
            project=project,
            projects=projects,
            batch_size=self.batch_size
        )

        self._current_strategy = None
        self.stats['phase2'] = stats
        return stats

    async def run_phase3(self, project: str = None, channel_id: int = None):
        """
        Phase 3: Speaker Clustering and Assignment.

        Three modes available (controlled by --phase3-mode):

        1. 'propagation-conflicts' (default, RECOMMENDED): Label Propagation with Conflict Detection
           - Uses k-NN weighted voting to propagate labels
           - Detects actual conflicts (contested assignments, cluster splits)
           - Only flags conflicts for Phase 4 LLM verification
           - Much more efficient than verifying all name pairs

        2. 'anchor': Anchor-Canopy Clustering
           - Uses text-verified speakers from Phase 2 as trusted anchors
           - Expands clusters using multi-gate validation
           - More conservative, higher precision

        3. 'propagation': Legacy Single-Pass Label Propagation
           - Uses k-NN weighted voting to propagate labels
           - No conflict detection (use with legacy Phase 4)

        Creates/updates speaker identities and assigns all cluster members.
        """
        logger.info("")
        logger.info("=" * 80)

        if self.phase3_mode == 'propagation-conflicts':
            logger.info("PHASE 3: LABEL PROPAGATION WITH CONFLICT DETECTION")
            logger.info("=" * 80)

            strategy = LabelPropagationWithConflictsStrategy(
                dry_run=self.dry_run,
                start_date=self.start_date,
                end_date=self.end_date,
                max_anchors=self.max_anchors
            )
        elif self.phase3_mode == 'propagation':
            logger.info("PHASE 3: LABEL PROPAGATION (Legacy)")
            logger.info("=" * 80)

            strategy = LabelPropagationStrategy(
                dry_run=self.dry_run,
                start_date=self.start_date,
                end_date=self.end_date,
                max_anchors=self.max_anchors
            )
        else:
            logger.info("PHASE 3: ANCHOR-CANOPY CLUSTERING")
            logger.info("=" * 80)

            strategy = AnchorVerifiedClusteringStrategy(
                dry_run=self.dry_run,
                start_date=self.start_date,
                end_date=self.end_date,
                max_anchors=self.max_anchors
            )

        stats = await strategy.run(project=project)

        self.stats['phase3'] = stats
        return stats

    async def run_phase4(self, project: str = None, channel_id: int = None):
        """
        Phase 4: Conflict Resolution / Identity Merge Detection.

        Behavior depends on Phase 3 mode:

        1. If Phase 3 used 'propagation-conflicts' (default):
           - Uses ConflictResolutionStrategy
           - LLM-verifies only the actual conflicts detected in Phase 3
           - Much faster than legacy approach (10-50 pairs vs 2,746)

        2. If Phase 3 used 'anchor' or 'propagation' (legacy):
           - Uses IdentityMergeDetectionStrategy
           - Finds duplicate identities via centroid similarity
           - LLM verifies all potential merges
        """
        logger.info("")
        logger.info("=" * 80)

        if self.phase3_mode == 'propagation-conflicts':
            logger.info("PHASE 4: CONFLICT RESOLUTION VIA LLM")
            logger.info("=" * 80)

            strategy = ConflictResolutionStrategy(
                dry_run=self.dry_run
            )
        else:
            logger.info("PHASE 4: IDENTITY MERGE DETECTION (Legacy)")
            logger.info("=" * 80)

            strategy = IdentityMergeDetectionStrategy(
                dry_run=self.dry_run
            )

        stats = await strategy.run(project=project)

        self.stats['phase4'] = stats
        return stats

    async def run_phase5(self, project: str = None, channel_id: int = None):
        """
        Phase 5: Speaker Hydration.

        Retrieves detailed biographical information about identified speakers
        using external APIs (LinkedIn, etc.) to enrich speaker profiles.
        """
        logger.info("")
        logger.info("=" * 80)
        logger.info("PHASE 5: SPEAKER HYDRATION")
        logger.info("=" * 80)

        strategy = SpeakerHydrationStrategy(
            dry_run=self.dry_run
        )

        stats = await strategy.run(project=project)

        self.stats['phase5'] = stats
        return stats

    async def run(
        self,
        projects: List[str] = None,
        channel_id: int = None
    ) -> Dict:
        """
        Run the full pipeline.

        Args:
            projects: List of projects to process (Phase 2+ runs per-project)
            channel_id: Single channel to process

        Returns:
            Combined stats dict
        """
        start_time = datetime.now()

        logger.info("=" * 80)
        logger.info("SPEAKER IDENTIFICATION ORCHESTRATOR")
        logger.info("=" * 80)
        logger.info(f"Mode: {'DRY RUN' if self.dry_run else 'APPLY CHANGES'}")
        logger.info(f"Phases to run: {self.phases}")
        if projects:
            logger.info(f"Projects: {projects}")
        if channel_id:
            logger.info(f"Channel ID: {channel_id}")
        logger.info(f"Min duration: {self.min_duration_pct * 100:.0f}%")
        if self.batch_size:
            logger.info(f"Batch size: {self.batch_size} (priority recheck after each batch)")
        if self.start_date or self.end_date:
            logger.info(f"Date range: {self.start_date or 'any'} to {self.end_date or 'any'}")
        if self.include_assigned:
            logger.info("Including previously assigned speakers (re-verification mode)")
        logger.info("")

        # Phase 1: Run ONCE globally across all specified projects
        if 1 in self.phases:
            await self.run_phase1(projects=projects, channel_id=channel_id)

        # Phase 2+: Run phases
        if channel_id:
            # Single channel mode - run all phases
            if 2 in self.phases:
                await self.run_phase2(project=None, channel_id=channel_id)
            if 3 in self.phases:
                await self.run_phase3(project=None, channel_id=channel_id)
            if 4 in self.phases:
                await self.run_phase4(project=None, channel_id=channel_id)
            if 5 in self.phases:
                await self.run_phase5(project=None, channel_id=channel_id)
        elif projects:
            # Phase 2: Run ONCE universally across all projects (single tqdm)
            if 2 in self.phases:
                await self.run_phase2(projects=projects, channel_id=None)

            # Phase 3+: Still run per-project (they need project-specific logic)
            for project in projects:
                if 3 in self.phases or 4 in self.phases or 5 in self.phases:
                    logger.info("")
                    logger.info("=" * 80)
                    logger.info(f"PROCESSING PROJECT: {project}")
                    logger.info("=" * 80)

                if 3 in self.phases:
                    await self.run_phase3(project=project, channel_id=None)
                if 4 in self.phases:
                    await self.run_phase4(project=project, channel_id=None)
                if 5 in self.phases:
                    await self.run_phase5(project=project, channel_id=None)

        # Final summary
        elapsed = (datetime.now() - start_time).total_seconds()
        self.stats['total_time'] = elapsed

        logger.info("")
        logger.info("=" * 80)
        logger.info("ORCHESTRATOR COMPLETE")
        logger.info("=" * 80)

        if 1 in self.phases:
            p1 = self.stats['phase1']
            logger.info(f"Phase 1: {p1.get('episodes_processed', 0)} episodes, {p1.get('speakers_identified', 0)} speakers")

        if 2 in self.phases:
            p2 = self.stats['phase2']
            logger.info(f"Phase 2: {p2.get('evidence_found_certain', 0)} with certain evidence, "
                       f"{p2.get('evidence_none', 0)} no evidence")

        if 3 in self.phases:
            p3 = self.stats['phase3']
            logger.info(f"Phase 3: {p3.get('clusters_created', 0)} clusters, "
                       f"{p3.get('speakers_assigned', 0)} assigned, "
                       f"{p3.get('name_collisions_detected', 0)} collisions, "
                       f"{p3.get('unnamed_clusters_created', 0)} unnamed")

        if 4 in self.phases:
            p4 = self.stats['phase4']
            logger.info(f"Phase 4: {p4.get('llm_confirmed_same', 0)} merges confirmed, {p4.get('merges_executed', 0)} executed")

        if 5 in self.phases:
            p5 = self.stats['phase5']
            logger.info(f"Phase 5: {p5.get('identities_hydrated', 0)} identities hydrated, "
                       f"{p5.get('api_calls', 0)} API calls")

        logger.info(f"Total time: {elapsed/60:.1f} minutes")
        logger.info("=" * 80)

        return self.stats


async def main():
    parser = argparse.ArgumentParser(
        description='Speaker Identification Orchestrator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
PHASES:
  1 = Metadata extraction (channel hosts, episode speakers from descriptions)
  2 = Text Evidence Collection (find transcript evidence: self-intro, addressed, introduced)
  3 = Speaker Clustering (--phase3-mode: 'propagation-conflicts', 'anchor', or 'propagation')
      - propagation-conflicts (default): k-NN with conflict detection → efficient Phase 4
      - anchor: anchor-canopy clustering
      - propagation: legacy k-NN without conflict detection
  4 = Conflict Resolution / Identity Merge (depends on Phase 3 mode)
      - If propagation-conflicts: LLM verifies only actual conflicts (10-50 pairs)
      - Otherwise: legacy identity merge detection
  5 = Speaker Hydration (enrich profiles with external data: LinkedIn, etc.)

Examples:
  # Dry run on project
  python -m src.speaker_identification.orchestrator --project CPRMV

  # Apply all phases
  python -m src.speaker_identification.orchestrator --project CPRMV --apply

  # Run specific phases
  python -m src.speaker_identification.orchestrator --project CPRMV --phases 2,3,4 --apply

  # Re-verify all assignments
  python -m src.speaker_identification.orchestrator --project CPRMV --include-assigned --apply

  # Run on all active projects
  python -m src.speaker_identification.orchestrator --all-active --apply
"""
    )

    parser.add_argument('--project', type=str, help='Single project to process')
    parser.add_argument('--projects', type=str, help='Comma-separated list of projects to process')
    parser.add_argument('--all-active', action='store_true', help='Process all active projects from config')
    parser.add_argument('--channel-id', type=int, help='Single channel ID')
    parser.add_argument('--phases', type=str, default=None,
                       help='Comma-separated phases to run (default: 1,2,3,4,5)')
    parser.add_argument('--apply', action='store_true', help='Apply changes (default: dry run)')
    parser.add_argument('--include-assigned', action='store_true',
                       help='Re-verify speakers already assigned')
    parser.add_argument('--max-channels', type=int, help='Max channels to process')
    parser.add_argument('--max-episodes', type=int, help='Max episodes per channel (Phase 1)')
    parser.add_argument('--max-speakers', type=int, help='Max speakers (Phase 2, 4)')
    parser.add_argument('--min-duration-pct', type=float, default=0.08,
                       help='Min speaker duration %% (default: 0.08)')
    parser.add_argument('--start-channel-id', type=int,
                       help='Resume from channel ID')
    parser.add_argument('--max-concurrent', type=int, default=15,
                       help='Max concurrent LLM requests for Phase 2 (default: 15)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Process N items per batch, then recheck priority for new content. '
                            'Phase 1: N episodes per batch. Phase 2: N speakers per batch. '
                            'Allows recent uploads to jump ahead of backlog between batches.')
    parser.add_argument('--max-anchors', type=int, default=None,
                       help='Max anchor name groups to process in Phase 3 (for testing)')
    parser.add_argument('--phase3-mode', type=str,
                       choices=['propagation-conflicts', 'anchor', 'propagation'],
                       default='propagation-conflicts',
                       help='Phase 3 algorithm: propagation-conflicts (default, RECOMMENDED: '
                            'k-NN with conflict detection for Phase 4 LLM resolution), '
                            'anchor (anchor-canopy clustering), or '
                            'propagation (legacy label propagation without conflict detection)')

    args = parser.parse_args()

    # Determine projects to process
    projects_to_process = []

    if args.all_active:
        # Load active projects from config
        from src.utils.config import load_config
        config = load_config()
        active_projects = config.get('active_projects', {})
        projects_to_process = [
            name for name, cfg in active_projects.items()
            if cfg.get('enabled', False)
        ]
        # Sort by priority (higher priority first)
        projects_to_process.sort(
            key=lambda p: active_projects[p].get('priority', 0),
            reverse=True
        )
        logger.info(f"Active projects: {projects_to_process}")

    elif args.projects:
        projects_to_process = [p.strip() for p in args.projects.split(',')]

    elif args.project:
        projects_to_process = [args.project]

    elif not args.channel_id:
        parser.error('Either --project, --projects, --all-active, or --channel-id is required')

    # Parse phases (if provided)
    phases = None
    if args.phases:
        phases = [int(p.strip()) for p in args.phases.split(',')]

    orchestrator = SpeakerIdentificationOrchestrator(
        dry_run=not args.apply,
        phases=phases,
        max_channels=args.max_channels,
        max_episodes=args.max_episodes,
        max_speakers=args.max_speakers,
        min_duration_pct=args.min_duration_pct,
        start_channel_id=args.start_channel_id,
        include_assigned=args.include_assigned,
        max_concurrent=args.max_concurrent,
        batch_size=args.batch_size,
        max_anchors=args.max_anchors,
        phase3_mode=args.phase3_mode
    )

    # Set up signal handlers for graceful shutdown
    loop = asyncio.get_running_loop()
    shutdown_count = [0]  # Use list to allow modification in closure

    def handle_shutdown_signal():
        shutdown_count[0] += 1
        if shutdown_count[0] == 1:
            orchestrator.request_shutdown()
        else:
            logger.info("\nForce quit requested")
            sys.exit(1)

    # Register signal handlers
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, handle_shutdown_signal)

    try:
        if args.channel_id:
            # Single channel mode
            stats = await orchestrator.run(channel_id=args.channel_id)
        else:
            # Pass projects list - orchestrator handles Phase 1 globally, Phase 2+ per-project
            stats = await orchestrator.run(projects=projects_to_process)

        # Print machine-readable summary for orchestrator
        import json
        summary = {
            'phases_run': args.phases,
            'total_time_minutes': round(stats.get('total_time', 0) / 60, 1)
        }
        # Add phase-specific stats
        if 1 in args.phases:
            p1 = stats.get('phase1', {})
            summary['phase1_episodes_processed'] = p1.get('episodes_processed', 0)
            summary['phase1_speakers_identified'] = p1.get('speakers_identified', 0)
        if 2 in args.phases:
            p2 = stats.get('phase2', {})
            summary['phase2_evidence_certain'] = p2.get('evidence_found_certain', 0)
            summary['phase2_evidence_none'] = p2.get('evidence_none', 0)
        if 3 in args.phases:
            p3 = stats.get('phase3', {})
            summary['phase3_clusters_created'] = p3.get('clusters_created', 0)
            summary['phase3_speakers_assigned'] = p3.get('speakers_assigned', 0)
        if 4 in args.phases:
            p4 = stats.get('phase4', {})
            summary['phase4_merges_confirmed'] = p4.get('llm_confirmed_same', 0)
            summary['phase4_merges_executed'] = p4.get('merges_executed', 0)
        if 5 in args.phases:
            p5 = stats.get('phase5', {})
            summary['phase5_identities_hydrated'] = p5.get('identities_hydrated', 0)
        print(f"TASK_SUMMARY: {json.dumps(summary)}")

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
