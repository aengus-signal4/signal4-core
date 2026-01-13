#!/usr/bin/env python3
"""
Production Run: Populate hosts and speakers for all active channels
====================================================================

Runs Phase 1A + 1B on all active channels in specified projects.
Designed to be run incrementally and safely with progress tracking.

Usage:
    # Dry run to see what would be processed
    python -m src.speaker_identification.run_production --project CPRMV

    # Process all channels (applies changes)
    python -m src.speaker_identification.run_production --project CPRMV --apply

    # Limit channels and episodes for testing
    python -m src.speaker_identification.run_production --project CPRMV --apply --max-channels 10 --max-episodes 50

    # Resume from specific channel
    python -m src.speaker_identification.run_production --project CPRMV --apply --start-channel-id 21726
"""

import argparse
import asyncio
import sys
from pathlib import Path
from src.utils.paths import get_project_root, get_config_path
from typing import Optional
from datetime import datetime
from tqdm import tqdm

project_root = str(get_project_root())
sys.path.append(project_root)

from src.speaker_identification.strategies.metadata_identification import MetadataIdentificationStrategy
from src.speaker_identification.core.context_builder import ContextBuilder
from src.utils.logger import setup_worker_logger
from sqlalchemy import text
from src.database.session import get_session

# Setup logger - will write to logs/content_processing/worker_speaker_identification.production.log
logger = setup_worker_logger('speaker_identification.production')

# Also log to console
import logging
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)


def get_active_channels(
    project: str,
    min_unassigned: int = 5,
    start_channel_id: Optional[int] = None
):
    """Get active channels for a project."""
    with get_session() as session:
        query = text("""
            SELECT DISTINCT
                c.id,
                c.display_name,
                c.platform,
                c.status,
                c.hosts,
                COUNT(DISTINCT co.content_id) as episode_count,
                COUNT(DISTINCT s.id) FILTER (
                    WHERE s.speaker_identity_id IS NULL
                    AND s.embedding_quality_score >= 0.65
                ) as unassigned_speakers,
                COUNT(DISTINCT co.content_id) FILTER (
                    WHERE co.hosts IS NULL
                    OR co.hosts = '[]'::jsonb
                ) as episodes_without_speakers
            FROM channels c
            JOIN content co ON c.id = co.channel_id
            LEFT JOIN speakers s ON s.content_id = co.content_id AND s.embedding IS NOT NULL
            WHERE c.status = 'active'
              AND :project = ANY(co.projects)
              AND (:start_id IS NULL OR c.id >= :start_id)
            GROUP BY c.id, c.display_name, c.platform, c.status, c.hosts
            HAVING COUNT(DISTINCT co.content_id) > 0
            ORDER BY c.id
        """)

        results = session.execute(query, {
            'project': project,
            'start_id': start_channel_id
        }).fetchall()

        return [
            {
                'id': row.id,
                'display_name': row.display_name,
                'platform': row.platform,
                'status': row.status,
                'has_hosts': bool(row.hosts and len(row.hosts) > 0),
                'episode_count': row.episode_count,
                'unassigned_speakers': row.unassigned_speakers,
                'episodes_without_speakers': row.episodes_without_speakers
            }
            for row in results
        ]


async def main():
    parser = argparse.ArgumentParser(
        description='Production run: Populate hosts and speakers for active channels',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--project', type=str,
                       help='Project name (e.g., CPRMV)')
    parser.add_argument('--channel-id', type=int,
                       help='Process single channel ID')
    parser.add_argument('--apply', action='store_true',
                       help='Apply changes (default: dry run)')
    parser.add_argument('--max-channels', type=int,
                       help='Max channels to process (for testing)')
    parser.add_argument('--max-episodes', type=int,
                       help='Max episodes per channel (default: all)')
    parser.add_argument('--start-channel-id', type=int,
                       help='Start from specific channel ID (for resuming)')
    parser.add_argument('--min-confidence', type=float, default=0.60,
                       help='Min confidence for assignment (default: 0.60)')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for LLM requests (default: 4)')

    args = parser.parse_args()

    # Validate arguments
    if not args.project and not args.channel_id:
        parser.error('Either --project or --channel-id is required')

    # Check log file location
    log_dir = Path.home() / "logs" / "content_processing"
    log_file = log_dir / "worker_speaker_identification.production.log"

    logger.info("=" * 80)
    logger.info("SPEAKER IDENTIFICATION - PRODUCTION RUN")
    logger.info("=" * 80)
    logger.info(f"Log file: {log_file}")
    if args.channel_id:
        logger.info(f"Channel ID: {args.channel_id}")
    else:
        logger.info(f"Project: {args.project}")
    logger.info(f"Mode: {'APPLY CHANGES' if args.apply else 'DRY RUN'}")
    logger.info(f"Max channels: {args.max_channels or 'All'}")
    logger.info(f"Max episodes per channel: {args.max_episodes or 'All'}")
    if args.start_channel_id:
        logger.info(f"Starting from channel ID: {args.start_channel_id}")
    logger.info("")

    # Get active channels
    if args.channel_id:
        # Single channel mode
        logger.info(f"Loading channel {args.channel_id}...")
        with get_session() as session:
            result = session.execute(text("""
                SELECT
                    c.id,
                    c.display_name,
                    c.platform,
                    c.status,
                    c.hosts,
                    COUNT(DISTINCT co.content_id) as episode_count,
                    0 as unassigned_speakers,
                    COUNT(DISTINCT co.content_id) FILTER (
                        WHERE co.hosts IS NULL
                        OR co.hosts = '[]'::jsonb
                    ) as episodes_without_speakers
                FROM channels c
                JOIN content co ON c.id = co.channel_id
                WHERE c.id = :channel_id
                GROUP BY c.id, c.display_name, c.platform, c.status, c.hosts
            """), {'channel_id': args.channel_id}).fetchone()

            if not result:
                logger.error(f"Channel {args.channel_id} not found")
                return

            channels = [{
                'id': result.id,
                'display_name': result.display_name,
                'platform': result.platform,
                'status': result.status,
                'has_hosts': bool(result.hosts and len(result.hosts) > 0),
                'episode_count': result.episode_count,
                'unassigned_speakers': result.unassigned_speakers,
                'episodes_without_speakers': result.episodes_without_speakers
            }]
    else:
        # Project mode
        logger.info("Loading active channels...")
        channels = get_active_channels(
            project=args.project,
            start_channel_id=args.start_channel_id
        )

    if not channels:
        if args.channel_id:
            logger.info(f"Channel {args.channel_id} not found")
        else:
            logger.info(f"No active channels found for project '{args.project}'")
        return

    logger.info(f"Found {len(channels)} active channels")
    logger.info("")

    # Summary stats
    total_episodes = sum(c['episode_count'] for c in channels)
    total_unassigned = sum(c['unassigned_speakers'] for c in channels)
    channels_without_hosts = sum(1 for c in channels if not c['has_hosts'])
    total_episodes_without_speakers = sum(c['episodes_without_speakers'] for c in channels)

    logger.info("Summary:")
    logger.info(f"  Total episodes: {total_episodes:,}")
    logger.info(f"  Unassigned speakers: {total_unassigned:,}")
    logger.info(f"  Channels without hosts: {channels_without_hosts}")
    logger.info(f"  Episodes without speakers: {total_episodes_without_speakers:,}")
    logger.info("")

    if args.max_channels:
        channels = channels[:args.max_channels]
        logger.info(f"Limited to {len(channels)} channels")
        logger.info("")

    # Track progress
    start_time = datetime.now()
    channels_processed = 0
    channels_failed = 0

    # Progress bar for channels
    channel_pbar = tqdm(
        total=len(channels),
        desc="Channels",
        unit="channel",
        position=0,
        leave=True
    )

    # Progress bar for episodes (will be updated per channel)
    episode_pbar = tqdm(
        total=0,
        desc="Episodes",
        unit="ep",
        position=1,
        leave=False
    )

    # Episode progress callback
    def update_episode_progress(current, total):
        episode_pbar.update(1)

    # Initialize strategy
    strategy = MetadataIdentificationStrategy(
        min_confidence=args.min_confidence,
        dry_run=not args.apply,
        max_episodes=args.max_episodes,
        episode_progress_callback=update_episode_progress,
        batch_size=args.batch_size
    )

    logger.info(f"Batch size: {args.batch_size}")

    # Process each channel
    for i, channel in enumerate(channels, 1):
        channel_pbar.set_description(f"Channel: {channel['display_name'][:40]}")

        # Reset episode progress bar for this channel
        episode_pbar.reset(total=min(channel['episodes_without_speakers'], args.max_episodes or 999999))

        logger.info("")
        logger.info("=" * 80)
        logger.info(f"[{i}/{len(channels)}] Processing Channel: {channel['display_name']}")
        logger.info("=" * 80)
        logger.info(f"  ID: {channel['id']}")
        logger.info(f"  Platform: {channel['platform']}")
        logger.info(f"  Episodes: {channel['episode_count']:,}")
        logger.info(f"  Episodes without speakers: {channel['episodes_without_speakers']:,}")
        logger.info(f"  Has hosts: {'Yes' if channel['has_hosts'] else 'No'}")
        logger.info("")

        try:
            await strategy.run_single_channel(channel['id'])
            channels_processed += 1
            channel_pbar.update(1)

            # Update progress bar postfix with stats
            channel_pbar.set_postfix({
                'hosts': strategy.stats['hosts_extracted'],
                'speakers': strategy.stats['speakers_identified'],
                'episodes': strategy.stats['episodes_processed']
            })

        except KeyboardInterrupt:
            channel_pbar.close()
            episode_pbar.close()
            logger.warning("\n\nInterrupted by user!")
            logger.info(f"Processed {channels_processed}/{len(channels)} channels before interruption")
            logger.info(f"Resume with: --start-channel-id {channel['id']}")
            sys.exit(1)

        except Exception as e:
            logger.error(f"Error processing channel {channel['id']}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            channels_failed += 1
            channel_pbar.update(1)
            continue

    channel_pbar.close()
    episode_pbar.close()

    # Final summary
    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info("")
    logger.info("=" * 80)
    logger.info("PRODUCTION RUN COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Channels processed: {channels_processed}/{len(channels)}")
    logger.info(f"Channels failed: {channels_failed}")
    logger.info(f"Hosts extracted: {strategy.stats['hosts_extracted']}")
    logger.info(f"Speakers identified: {strategy.stats['speakers_identified']}")
    logger.info(f"Episodes processed: {strategy.stats['episodes_processed']}")
    logger.info(f"Elapsed time: {elapsed/60:.1f} minutes")
    logger.info("=" * 80)

    if strategy.stats['errors']:
        logger.warning(f"\nErrors encountered: {len(strategy.stats['errors'])}")
        for error in strategy.stats['errors'][:10]:
            logger.warning(f"  - {error}")


if __name__ == '__main__':
    asyncio.run(main())
