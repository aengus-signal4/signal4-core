#!/usr/bin/env python3
"""
Metadata-Based Speaker Identification (Phase 1)
================================================

Two-phase speaker identification using only channel and episode descriptions:
- Phase 1A: Extract hosts from channel descriptions
- Phase 1B: Assign episode speakers using hosts + episode metadata
- Phase 1C: Consolidate name variations to canonical forms

Usage:
    # Run on all active projects (default)
    python -m src.speaker_identification.strategies.metadata_identification --apply

    # Test on single channel
    python -m src.speaker_identification.strategies.metadata_identification \\
        --channel-id 6109 --apply

    # Run on specific project
    python -m src.speaker_identification.strategies.metadata_identification \\
        --project CPRMV --apply

    # Dry run (no --apply)
    python -m src.speaker_identification.strategies.metadata_identification \\
        --channel-id 6109
"""

import argparse
import asyncio
import sys
from pathlib import Path
from src.utils.paths import get_project_root, get_config_path
from typing import Dict, List, Optional
from datetime import datetime, timezone

from tqdm import tqdm

# Add project root to path
project_root = str(get_project_root())
sys.path.append(project_root)

from src.speaker_identification.core.llm_client import MLXLLMClient
from src.speaker_identification.core.context_builder import ContextBuilder
from src.speaker_identification.core.identity_manager import IdentityManager
from src.utils.logger import setup_worker_logger
from src.utils.config import get_project_date_range, get_active_projects, load_config

logger = setup_worker_logger('speaker_identification.metadata_strategy')


class MetadataIdentificationStrategy:
    """
    Phase 1: Metadata-only speaker identification.

    Process:
    1. Phase 1A: Extract hosts from channel descriptions
    2. Cache hosts in channel_host_cache
    3. Phase 1B: For each episode, assign speakers using hosts + episode metadata
    4. Create SpeakerIdentity records and assign speakers
    """

    def __init__(
        self,
        min_confidence: float = 0.60,
        min_duration_pct: float = 0.10,
        min_quality: float = 0.65,
        dry_run: bool = True,
        max_episodes: int = None,
        episode_progress_callback = None,
        batch_size: int = 8,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ):
        """
        Initialize strategy.

        Args:
            min_confidence: Minimum confidence to create identities
            min_duration_pct: Minimum % of episode for speaker consideration
            min_quality: Minimum speaker quality score
            dry_run: If True, don't make DB changes
            max_episodes: Maximum episodes to process per channel
            episode_progress_callback: Optional callback(episode_num, total) for progress tracking
            batch_size: Number of concurrent LLM requests (default 8, saturates 2 endpoints)
            start_date: Filter content published on or after this date (YYYY-MM-DD)
            end_date: Filter content published before this date (YYYY-MM-DD)
        """
        self.min_confidence = min_confidence
        self.min_duration_pct = min_duration_pct
        self.min_quality = min_quality
        self.dry_run = dry_run
        self.max_episodes = max_episodes
        self.episode_progress_callback = episode_progress_callback
        self.batch_size = batch_size
        self.start_date = start_date
        self.end_date = end_date

        self.llm_client = MLXLLMClient(tier="tier_3")
        self.context_builder = ContextBuilder()
        self.identity_manager = IdentityManager()

        # Load project priorities from config
        config = load_config()
        active_projects = config.get('active_projects', {})
        self.project_priorities = {
            name: cfg.get('priority', 1)
            for name, cfg in active_projects.items()
        }

        # Progress bar (set during run_all_projects/run_multiple_channels)
        self._pbar = None

        # Track channels that have had Phase 1A run (for on-demand processing)
        self._channels_with_hosts: Dict[int, List[str]] = {}

        # Graceful shutdown
        self._shutdown_requested = False

        # Tracking
        self.stats = {
            'channels_processed': 0,
            'hosts_extracted': 0,
            'hosts_cached': 0,
            'episodes_processed': 0,
            'speakers_identified': 0,
            'speakers_assigned': 0,
            'identities_created': 0,
            'identities_matched': 0,
            'errors': []
        }

    async def run_single_channel(self, channel_id: int) -> Dict:
        """
        Run Phase 1A + 1B on a single channel.

        Args:
            channel_id: Channel ID to process

        Returns:
            Stats dict for this channel
        """
        logger.info("=" * 80)
        logger.info(f"Processing Channel ID: {channel_id}")
        logger.info("=" * 80)

        # Phase 1A: Extract hosts
        hosts = await self._phase_1a_extract_hosts(channel_id)

        if not hosts:
            logger.info(f"No channel-level hosts identified for channel {channel_id}")
            logger.info("Continuing with Phase 1B to extract episode-level speakers from metadata")

        # Phase 1B: Process episodes (even without known hosts - will extract guests/speakers from metadata)
        await self._phase_1b_process_episodes(channel_id, hosts or [])

        # Phase 1C: Consolidate name variations
        await self._phase_1c_consolidate_names(channel_id)

        return self.stats

    async def run_consolidation_only(
        self,
        project: Optional[str] = None,
        channel_id: Optional[int] = None,
        max_channels: Optional[int] = None
    ) -> Dict:
        """
        Run only Phase 1C consolidation on channels that already have hosts populated.

        Args:
            project: Filter to specific project
            channel_id: Process single channel
            max_channels: Maximum number of channels to process

        Returns:
            Stats dict
        """
        print(f"\n{'='*80}")
        print("HOST NAME CONSOLIDATION (Phase 1C only)")
        print(f"{'='*80}")
        print(f"Mode: {'DRY RUN' if self.dry_run else 'APPLY'}")
        if project:
            print(f"Project: {project}")
        print()

        if channel_id:
            # Single channel
            channels = [{'channel_id': channel_id, 'channel_name': f'Channel {channel_id}', 'unique_names': 0}]
        else:
            # Get channels needing consolidation
            channels = self.context_builder.get_channels_needing_consolidation(
                project=project,
                min_unique_names=3
            )

        if not channels:
            print("No channels need consolidation.")
            return self.stats

        if max_channels:
            channels = channels[:max_channels]

        print(f"Found {len(channels)} channels to consolidate")
        print()

        # Progress bar
        self._pbar = tqdm(total=len(channels), desc="Consolidating", unit="ch")

        try:
            for channel in channels:
                self._pbar.set_postfix_str(f"{channel.get('channel_name', '')[:30]}")
                try:
                    await self._phase_1c_consolidate_names(channel['channel_id'])
                    self.stats['channels_processed'] += 1
                except Exception as e:
                    logger.error(f"Error consolidating channel {channel['channel_id']}: {e}")
                    self.stats['errors'].append(f"Channel {channel['channel_id']}: {str(e)}")
                finally:
                    self._pbar.update(1)
        finally:
            self._pbar.close()
            self._pbar = None

        # Print summary
        print(f"\nConsolidated {self.stats.get('names_consolidated', 0)} name variations")
        if self.stats['errors']:
            print(f"Errors: {len(self.stats['errors'])}")

        return self.stats

    async def run_all_projects_filtered(
        self,
        projects: List[str],
        max_channels: Optional[int] = None,
        batch_size: int = None
    ) -> Dict:
        """
        Run Phase 1 on specified projects with tqdm progress.

        Args:
            projects: List of project names to process
            max_channels: Maximum number of channels to process
            batch_size: If provided, process episodes by priority in batches

        Returns:
            Aggregate stats dict
        """
        if batch_size:
            return await self.run_batched(projects=projects, batch_size=batch_size)
        return await self._run_projects(projects, max_channels)

    async def run_batched(
        self,
        projects: List[str] = None,
        batch_size: int = 50
    ) -> Dict:
        """
        Run Phase 1 processing episodes by priority in batches.

        This is the recommended mode for ongoing processing as it ensures:
        1. Recent content (last 30 days) is processed first
        2. Higher priority projects are processed before lower priority
        3. New content can jump the queue between batches

        Flow:
        1. Query N highest-priority episodes needing processing
        2. For each episode:
           - Run Phase 1A on-demand if channel not yet processed
           - Run Phase 1B to extract speakers from metadata
        3. Repeat until no episodes remain or shutdown requested
        4. At end: Run Phase 1C consolidation for all channels that had episodes processed

        Args:
            projects: List of projects to filter to
            batch_size: Number of episodes per batch (default 50)

        Returns:
            Stats dict
        """
        # Get total count upfront
        total_episodes = self.context_builder.count_episodes_by_priority(
            projects=projects,
            start_date=self.start_date,
            end_date=self.end_date
        )
        total_batches = (total_episodes + batch_size - 1) // batch_size if total_episodes > 0 else 0

        print(f"\n{'='*80}")
        print("METADATA-BASED SPEAKER IDENTIFICATION (Priority Batched)")
        print(f"{'='*80}")
        print(f"Mode: {'DRY RUN' if self.dry_run else 'APPLY'}")
        if projects:
            print(f"Projects: {', '.join(projects)}")
        print(f"Batch size: {batch_size} episodes")
        print(f"Total episodes: {total_episodes:,} ({total_batches} batches)")
        print(f"Priority: Recent (30d) first, then by project priority")
        print()

        batch_num = 0
        total_processed = 0
        self._shutdown_requested = False

        # Create a single progress bar for all episodes
        pbar = tqdm(total=total_episodes, desc="Processing", unit="ep")

        try:
            while not self._shutdown_requested:
                batch_num += 1

                # Query next batch of highest-priority episodes
                episodes = self.context_builder.get_episodes_by_priority(
                    projects=projects,
                    project_priorities=self.project_priorities,
                    limit=batch_size,
                    start_date=self.start_date,
                    end_date=self.end_date
                )

                if not episodes:
                    logger.info(f"No more episodes to process after {total_processed} total")
                    break

                # Count recent vs backlog
                now = datetime.now(timezone.utc)
                recent_count = sum(
                    1 for ep in episodes
                    if ep.get('publish_date') and
                    (now - ep['publish_date']).days < 30
                )

                # Count unique channels in batch
                channels_in_batch = set(ep['channel_id'] for ep in episodes)

                # Update progress bar description with batch info
                pbar.set_description(f"Batch {batch_num}/{total_batches}")
                pbar.set_postfix_str(f"{recent_count} recent, {len(channels_in_batch)} channels")

                # Process episodes
                batch_completed = 0
                for episode in episodes:
                    if self._shutdown_requested:
                        break

                    try:
                        await self._process_episode_with_channel_setup(episode)
                        batch_completed += 1
                        total_processed += 1
                        self.stats['episodes_processed'] += 1
                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"Error processing episode {episode['content_id']}: {e}")
                        self.stats['errors'].append(f"Episode {episode['content_id']}: {str(e)}")
                        pbar.update(1)

                # If we processed fewer than batch_size, we're done
                if len(episodes) < batch_size:
                    logger.info("Reached end of queue")
                    break
        finally:
            pbar.close()

        # Note: Phase 1C (name consolidation) is skipped in batched mode.
        # Run it separately when needed:
        #   python -m src.speaker_identification.strategies.metadata_identification --consolidate-only --apply

        if self._shutdown_requested:
            logger.info(f"✓ Graceful shutdown. Processed {total_processed} episodes total.")

        self._print_summary()
        return self.stats

    async def _process_episode_with_channel_setup(self, episode: Dict):
        """
        Process a single episode, running Phase 1A on-demand if needed.

        Args:
            episode: Episode dict from get_episodes_by_priority
        """
        channel_id = episode['channel_id']

        # Check if we've already processed this channel's hosts
        if channel_id not in self._channels_with_hosts:
            # Run Phase 1A on-demand for this channel
            logger.debug(f"Running Phase 1A on-demand for channel {channel_id} ({episode['channel_name']})")
            hosts = await self._phase_1a_extract_hosts(channel_id)
            self._channels_with_hosts[channel_id] = hosts or []

        channel_hosts = self._channels_with_hosts[channel_id]

        # Process the episode (Phase 1B logic)
        await self._process_single_episode(
            episode={
                'content_id': episode['content_id'],
                'title': episode['title'],
                'description': episode['description'],
                'publish_date': episode['publish_date'].isoformat() if episode['publish_date'] else ''
            },
            channel_hosts=channel_hosts,
            channel_id=channel_id
        )

    def request_shutdown(self):
        """Request graceful shutdown."""
        if not self._shutdown_requested:
            self._shutdown_requested = True
            logger.info("\n⚠️  Shutdown requested - completing current episode...")

    async def run_all_projects(
        self,
        max_channels: Optional[int] = None
    ) -> Dict:
        """
        Run Phase 1 on all active projects (from config) with tqdm progress.

        Args:
            max_channels: Maximum number of channels to process

        Returns:
            Aggregate stats dict
        """
        # Get all active projects from config
        projects = get_active_projects()
        return await self._run_projects(projects, max_channels)

    async def _run_projects(
        self,
        projects: List[str],
        max_channels: Optional[int] = None
    ) -> Dict:
        """
        Internal method to run Phase 1 on specified projects.

        Args:
            projects: List of project names to process
            max_channels: Maximum number of channels to process

        Returns:
            Aggregate stats dict
        """
        print(f"\n{'='*80}")
        print("METADATA-BASED SPEAKER IDENTIFICATION")
        print(f"{'='*80}")
        print(f"Mode: {'DRY RUN' if self.dry_run else 'APPLY'}")
        print(f"Active projects: {', '.join(projects)}")
        print()

        # Get work summary across all projects
        work_summary = self.context_builder.get_work_summary(projects)

        if work_summary['total_episodes'] == 0:
            print("No episodes need processing.")
            return self.stats

        # Apply max_channels limit
        channels_to_process = work_summary['channels']
        if max_channels:
            channels_to_process = channels_to_process[:max_channels]

        total_episodes = sum(ch['episode_count'] for ch in channels_to_process)

        # Print summary
        print(f"Work Summary:")
        print(f"  Channels: {len(channels_to_process)}")
        print(f"  Episodes: {total_episodes}")
        print()
        print("Top 10 channels by episode count:")
        for ch in channels_to_process[:10]:
            print(f"  {ch['channel_name'][:40]:<40} ({ch['platform']:<10}) {ch['episode_count']:>5} episodes")
        if len(channels_to_process) > 10:
            remaining = sum(ch['episode_count'] for ch in channels_to_process[10:])
            print(f"  ... and {len(channels_to_process) - 10} more channels with {remaining} episodes")
        print()

        # Create single progress bar for all episodes
        self._pbar = tqdm(
            total=total_episodes,
            desc="Episodes",
            unit="ep"
        )

        try:
            for channel in channels_to_process:
                self._pbar.set_postfix_str(f"{channel['channel_name'][:30]}")
                try:
                    await self.run_single_channel(channel['channel_id'])
                    self.stats['channels_processed'] += 1
                except Exception as e:
                    logger.error(f"Error processing channel {channel['channel_id']}: {e}")
                    self.stats['errors'].append(f"Channel {channel['channel_id']}: {str(e)}")
                    continue
        finally:
            self._pbar.close()
            self._pbar = None

        self._print_summary()
        return self.stats

    async def run_multiple_channels(
        self,
        project: Optional[str] = None,
        max_channels: Optional[int] = None
    ) -> Dict:
        """
        Run Phase 1A + 1B + 1C on multiple channels with tqdm progress.

        Args:
            project: Filter to specific project
            max_channels: Maximum number of channels to process

        Returns:
            Aggregate stats dict
        """
        # Load date range from project config if not already set
        if project and not self.start_date and not self.end_date:
            start_date, end_date = get_project_date_range(project)
            self.start_date = start_date
            self.end_date = end_date

        print(f"\n{'='*80}")
        print("METADATA-BASED SPEAKER IDENTIFICATION")
        print(f"{'='*80}")
        print(f"Project: {project}")
        print(f"Mode: {'DRY RUN' if self.dry_run else 'APPLY'}")
        if self.start_date or self.end_date:
            print(f"Date range: {self.start_date or 'any'} to {self.end_date or 'any'}")
        print()

        # Get work summary for this project
        work_summary = self.context_builder.get_work_summary([project] if project else None)

        if work_summary['total_episodes'] == 0:
            print("No episodes need processing.")
            return self.stats

        # Apply max_channels limit
        channels_to_process = work_summary['channels']
        if max_channels:
            channels_to_process = channels_to_process[:max_channels]

        total_episodes = sum(ch['episode_count'] for ch in channels_to_process)

        # Print summary
        print(f"Work Summary:")
        print(f"  Channels: {len(channels_to_process)}")
        print(f"  Episodes: {total_episodes}")
        print()
        print("Top 10 channels by episode count:")
        for ch in channels_to_process[:10]:
            print(f"  {ch['channel_name'][:40]:<40} ({ch['platform']:<10}) {ch['episode_count']:>5} episodes")
        if len(channels_to_process) > 10:
            remaining = sum(ch['episode_count'] for ch in channels_to_process[10:])
            print(f"  ... and {len(channels_to_process) - 10} more channels with {remaining} episodes")
        print()

        # Create single progress bar for all episodes
        self._pbar = tqdm(
            total=total_episodes,
            desc="Episodes",
            unit="ep"
        )

        try:
            for channel in channels_to_process:
                self._pbar.set_postfix_str(f"{channel['channel_name'][:30]}")
                try:
                    await self.run_single_channel(channel['channel_id'])
                    self.stats['channels_processed'] += 1
                except Exception as e:
                    logger.error(f"Error processing channel {channel['channel_id']}: {e}")
                    self.stats['errors'].append(f"Channel {channel['channel_id']}: {str(e)}")
                    continue
        finally:
            self._pbar.close()
            self._pbar = None

        self._print_summary()
        return self.stats

    async def _phase_1a_extract_hosts(self, channel_id: int) -> List[str]:
        """
        Phase 1A: Extract hosts from channel description.

        For multi-host channels, also promotes frequent episode-level hosts
        to the channel level cache.

        Args:
            channel_id: Channel ID

        Returns:
            List of host names
        """
        logger.info("\n--- Phase 1A: Host Extraction ---")

        # First, sync hosts from episode frequency data
        # This promotes any host appearing in 10+ episodes to channel level
        promoted_hosts = self.context_builder.sync_channel_hosts_from_episodes(
            channel_id=channel_id,
            min_episodes=10
        )
        if promoted_hosts:
            logger.info(f"Promoted {len(promoted_hosts)} new hosts from episode data")

        # Now check cache (which now includes promoted hosts)
        cached_hosts = self.context_builder.get_cached_channel_hosts(channel_id)
        if cached_hosts:
            host_names = [h['name'] for h in cached_hosts]
            logger.info(f"Using {len(host_names)} cached hosts: {', '.join(host_names)}")
            self.stats['hosts_cached'] += len(host_names)
            return host_names

        # Get channel context
        channel = self.context_builder.get_channel_context(channel_id)
        if not channel or not channel['description']:
            logger.warning(f"No channel description for channel {channel_id}")
            return []

        logger.info(f"Channel: {channel['name']} ({channel['platform']})")
        logger.info(f"Description: {channel['description'][:200]}...")

        # Call LLM for host identification
        try:
            hosts_data = await self.llm_client.identify_channel_hosts(
                channel['name'],
                channel['description'],
                channel['platform']
            )

            if not hosts_data:
                logger.info("No hosts identified by LLM")
                return []

            logger.info(f"LLM identified {len(hosts_data)} host(s):")
            for host in hosts_data:
                logger.info(
                    f"  - {host['name']} "
                    f"(confidence: {host['confidence']})"
                )
                logger.info(f"    Reasoning: {host.get('reasoning', 'N/A')}")

            # Save hosts to channels.hosts
            if not self.dry_run:
                self.context_builder.save_channel_hosts(
                    channel_id=channel_id,
                    hosts=hosts_data
                )
                logger.info(f"Saved {len(hosts_data)} hosts to channel")
            else:
                logger.info(f"[DRY RUN] Would save {len(hosts_data)} hosts")

            self.stats['hosts_extracted'] += len(hosts_data)
            return [h['name'] for h in hosts_data]

        except Exception as e:
            logger.error(f"Error in Phase 1A: {e}")
            self.stats['errors'].append(f"Phase 1A channel {channel_id}: {str(e)}")
            return []

    async def _phase_1b_process_episodes(
        self,
        channel_id: int,
        channel_hosts: List[str]
    ):
        """
        Phase 1B: Process episodes and assign speakers.

        Uses concurrent processing with a semaphore to keep N requests in flight,
        distributing load across multiple LLM endpoints.

        Args:
            channel_id: Channel ID
            channel_hosts: List of host names from Phase 1A
        """
        logger.info("\n--- Phase 1B: Episode Speaker Assignment ---")

        # Get episodes that don't have speakers populated yet
        episodes = self.context_builder.get_episodes_with_unassigned_speakers(
            channel_id=channel_id,
            min_quality=self.min_quality,
            limit=self.max_episodes,
            start_date=self.start_date,
            end_date=self.end_date
        )

        if not episodes:
            logger.info("No episodes with unassigned speakers")
            return

        logger.debug(f"Processing {len(episodes)} episodes with {self.batch_size} concurrent requests")

        # Track completed count for progress reporting
        completed_lock = asyncio.Lock()

        async def process_with_limit(episode: Dict, idx: int):
            """Process a single episode with semaphore-controlled concurrency."""

            async with semaphore:
                logger.debug(f"[Episode {idx}/{len(episodes)}] Starting: {episode['title'][:60]}")

                try:
                    await self._process_single_episode(episode, channel_hosts, channel_id)

                    async with completed_lock:
                        self.stats['episodes_processed'] += 1
                        # Update parent progress bar if exists
                        if self._pbar:
                            self._pbar.update(1)

                        # Trigger progress callback
                        if self.episode_progress_callback:
                            self.episode_progress_callback(self.stats['episodes_processed'], len(episodes))

                except Exception as e:
                    logger.error(f"Error processing episode {episode['content_id']}: {e}")
                    async with completed_lock:
                        self.stats['errors'].append(f"Episode {episode['content_id']}: {str(e)}")
                        # Still update progress on error
                        if self._pbar:
                            self._pbar.update(1)

        # Semaphore controls max concurrent requests
        semaphore = asyncio.Semaphore(self.batch_size)

        # Launch all tasks - semaphore keeps only batch_size in flight at once
        tasks = [process_with_limit(ep, i + 1) for i, ep in enumerate(episodes)]
        await asyncio.gather(*tasks)

    async def _process_single_episode(
        self,
        episode: Dict,
        channel_hosts: List[str],
        channel_id: int
    ):
        """
        Process a single episode - extract speakers from metadata only.

        Args:
            episode: Episode dict from context_builder
            channel_hosts: List of host names
            channel_id: Channel ID for name normalization
        """
        # Call LLM to extract speakers and mentioned from metadata
        try:
            result = await self.llm_client.extract_episode_speakers(
                channel_hosts=channel_hosts,
                episode_title=episode['title'],
                episode_description=episode['description'],
                publish_date=episode['publish_date']
            )

            episode_speakers = result.get('speakers', [])
            mentioned_people = result.get('mentioned', [])

            if not episode_speakers and not mentioned_people:
                logger.debug(f"  No speakers/mentioned extracted: {episode['title'][:60]}")
                # Mark as processed even with no results
                if not self.dry_run:
                    self.context_builder.mark_metadata_speakers_extracted(episode['content_id'])
                return

            logger.debug(f"  LLM extracted {len(episode_speakers)} speaker(s), {len(mentioned_people)} mentioned")

            # Filter speakers by categorical confidence (only "certain" and "very_likely")
            confident_speakers = [
                s for s in episode_speakers
                if s.get('confidence') in ['certain', 'very_likely']
                and s.get('name')
            ]

            logger.debug(f"  {episode['title'][:60]}: {len(confident_speakers)} speakers")

            for speaker in confident_speakers:
                logger.info(
                    f"    - {speaker['name']} "
                    f"({speaker['role']}, confidence: {speaker['confidence']})"
                )
                logger.info(f"      Reasoning: {speaker['reasoning']}")

            if mentioned_people:
                logger.info(f"  Mentioned: {', '.join([m['name'] for m in mentioned_people])}")

            # Save to content.hosts, content.guests, content.mentioned columns
            metadata_to_save = {
                'speakers': confident_speakers,
                'mentioned': mentioned_people
            }

            if not self.dry_run and (confident_speakers or mentioned_people):
                self.context_builder.save_content_speakers(
                    content_id=episode['content_id'],
                    speakers=metadata_to_save,
                    channel_id=channel_id
                )

                self.stats['speakers_identified'] += len(confident_speakers)
                hosts_count = sum(1 for s in confident_speakers if s.get('role') == 'host')
                guests_count = sum(1 for s in confident_speakers if s.get('role') == 'guest')
                logger.info(f"  ✓ Saved {hosts_count} host(s), {guests_count} guest(s), {len(mentioned_people)} mentioned")

            else:
                logger.info(f"  [DRY RUN] Would save {len(confident_speakers)} speakers")

        except Exception as e:
            logger.error(f"Error extracting speakers: {e}")
            raise

    async def _phase_1c_consolidate_names(self, channel_id: int):
        """
        Phase 1C: Consolidate name variations to canonical forms.

        Gets the distribution of host names across all episodes for a channel,
        uses LLM to identify variations of the same person, then updates
        content.hosts to use canonical names.

        Args:
            channel_id: Channel ID to consolidate
        """
        logger.info("\n--- Phase 1C: Name Consolidation ---")

        # Get host name distribution for this channel
        host_distribution = self.context_builder.get_host_name_distribution(channel_id)

        if not host_distribution or len(host_distribution) < 2:
            logger.info("Not enough host variations to consolidate")
            return

        logger.info(f"Found {len(host_distribution)} unique host names")

        # Get channel context for the LLM
        channel = self.context_builder.get_channel_context(channel_id)
        if not channel:
            logger.warning(f"Could not get channel context for {channel_id}")
            return

        # Call LLM to consolidate names - use tier_2 (tier_1 not always available)
        tier2_client = MLXLLMClient(tier="tier_2")
        try:
            consolidations = await tier2_client.consolidate_channel_names(
                channel_name=channel['name'],
                channel_description=channel.get('description', ''),
                host_distribution=host_distribution
            )

            if not consolidations:
                logger.info("No consolidations returned by LLM")
                return

            # Build variation -> canonical mapping and aliases per person
            name_mapping = {}
            aliases_by_canonical = {}
            for person in consolidations:
                canonical = person['canonical_name']
                variations = person.get('variations', [])
                # Store all variations (including canonical) as aliases
                aliases_by_canonical[canonical] = variations
                for variation in variations:
                    if variation != canonical:
                        name_mapping[variation] = canonical

            if not name_mapping:
                logger.info("No name variations to consolidate")
                return

            logger.info(f"Consolidating {len(name_mapping)} name variations:")
            for old_name, new_name in sorted(name_mapping.items()):
                logger.info(f"  {old_name} -> {new_name}")

            # Apply consolidations to content.hosts and save aliases
            if not self.dry_run:
                updated_count = self.context_builder.apply_host_name_consolidations(
                    channel_id=channel_id,
                    name_mapping=name_mapping
                )
                logger.info(f"Updated {updated_count} episodes with consolidated names")
                self.stats['names_consolidated'] = self.stats.get('names_consolidated', 0) + len(name_mapping)

                # Save aliases to channel_host_cache for future lookups
                for canonical, aliases in aliases_by_canonical.items():
                    if len(aliases) > 1:  # Only save if there are actual aliases
                        self.context_builder.save_host_aliases(
                            channel_id=channel_id,
                            canonical_name=canonical,
                            aliases=aliases
                        )
                logger.info(f"Saved aliases for {len(aliases_by_canonical)} hosts")
            else:
                logger.info(f"[DRY RUN] Would update episodes with {len(name_mapping)} name mappings")

        except Exception as e:
            logger.error(f"Error in Phase 1C: {e}")
            self.stats['errors'].append(f"Phase 1C channel {channel_id}: {str(e)}")
        finally:
            await tier2_client.close()

    def _print_summary(self):
        """Print summary statistics."""
        logger.info("\n" + "=" * 80)
        logger.info("SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Channels processed: {self.stats['channels_processed']}")
        logger.info(f"Hosts extracted: {self.stats['hosts_extracted']}")
        logger.info(f"Hosts from cache: {self.stats['hosts_cached']}")
        logger.info(f"Episodes processed: {self.stats['episodes_processed']}")
        logger.info(f"Speakers identified: {self.stats['speakers_identified']}")
        logger.info(f"Speakers assigned: {self.stats['speakers_assigned']}")
        logger.info(f"Identities created: {self.stats['identities_created']}")
        logger.info(f"Identities matched: {self.stats['identities_matched']}")
        if self.stats['errors']:
            logger.warning(f"Errors: {len(self.stats['errors'])}")
        logger.info("=" * 80)


async def main():
    parser = argparse.ArgumentParser(
        description='Phase 1: Metadata-based speaker identification',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--channel-id', type=int, help='Process single channel')
    parser.add_argument('--project', type=str, help='Filter to project (e.g., CPRMV)')
    parser.add_argument('--max-channels', type=int, help='Max channels to process')
    parser.add_argument('--max-episodes', type=int, help='Max episodes per channel to process')
    parser.add_argument('--apply', action='store_true', help='Apply changes (default: dry run)')
    parser.add_argument('--consolidate-only', action='store_true',
                       help='Run only Phase 1C consolidation on channels that already have hosts')
    parser.add_argument('--min-confidence', type=float, default=0.60,
                       help='Min confidence for assignment (default: 0.60)')
    parser.add_argument('--min-duration-pct', type=float, default=0.10,
                       help='Min duration %% for speaker (default: 0.10)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Concurrent LLM requests (default: 8)')

    args = parser.parse_args()

    strategy = MetadataIdentificationStrategy(
        min_confidence=args.min_confidence,
        min_duration_pct=args.min_duration_pct,
        dry_run=not args.apply,
        max_episodes=args.max_episodes,
        batch_size=args.batch_size
    )

    try:
        if args.consolidate_only:
            # Run consolidation only on channels with hosts
            await strategy.run_consolidation_only(
                project=args.project,
                channel_id=args.channel_id,
                max_channels=args.max_channels
            )
        elif args.channel_id:
            await strategy.run_single_channel(args.channel_id)
        elif args.project:
            # Specific project
            await strategy.run_multiple_channels(
                project=args.project,
                max_channels=args.max_channels
            )
        else:
            # All active projects (default)
            await strategy.run_all_projects(
                max_channels=args.max_channels
            )

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
