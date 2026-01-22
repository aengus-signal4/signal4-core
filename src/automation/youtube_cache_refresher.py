#!/usr/bin/env python3
"""
YouTube Video Cache Refresher
=============================

Scheduled task that refreshes the YouTube video cache for validated podcast channels.
This keeps the cache up-to-date so episode matching can use recent videos.

Usage:
    # Refresh cache for all validated channels
    python -m src.automation.youtube_cache_refresher

    # Refresh with limit
    python -m src.automation.youtube_cache_refresher --limit 50

    # Also match new episodes after refreshing
    python -m src.automation.youtube_cache_refresher --match-episodes --episode-limit 20

    # Dry run (show what would be refreshed)
    python -m src.automation.youtube_cache_refresher --dry-run
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from typing import Dict, List

from sqlalchemy import desc

from src.database.models import Channel
from src.database.session import get_session
from src.ingestion.youtube_video_cache_indexer import YouTubeVideoCacheIndexer
from src.ingestion.podcast_video_matcher import PodcastVideoMatcher
from src.utils.logger import setup_indexer_logger


def setup_cli_logger(name: str) -> logging.Logger:
    """Set up a logger that outputs to console for CLI usage."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(console)
    logger.propagate = False
    return logger


async def refresh_youtube_cache(
    limit: int = None,
    match_episodes: bool = False,
    episode_limit: int = 50,
    dry_run: bool = False,
    logger: logging.Logger = None
) -> Dict:
    """
    Refresh YouTube video cache for validated podcast channels.

    Args:
        limit: Maximum number of channels to refresh (None = all)
        match_episodes: Also match new episodes after refreshing cache
        episode_limit: Max episodes per channel to match
        dry_run: Just show what would be done, don't actually refresh
        logger: Logger instance

    Returns:
        Summary dict with statistics
    """
    logger = logger or setup_cli_logger('youtube_cache_refresher')
    start_time = time.time()

    logger.info("=" * 60)
    logger.info("YouTube Video Cache Refresher")
    logger.info("=" * 60)

    # Get validated podcast channels with YouTube matches
    with get_session() as session:
        query = session.query(Channel).filter(
            Channel.platform == 'podcast',
            Channel.status == 'active',
            Channel.platform_metadata['video_links']['youtube']['validated'].astext == 'true'
        ).order_by(desc(Channel.importance_score))

        if limit:
            query = query.limit(limit)

        channels = query.all()

        # Also get channels not yet validated (for initial validation)
        unvalidated_query = session.query(Channel).filter(
            Channel.platform == 'podcast',
            Channel.status == 'active',
            Channel.platform_metadata['video_links']['youtube']['channel_id'].isnot(None),
            Channel.platform_metadata['video_links']['youtube']['validated'].is_(None)
        ).order_by(desc(Channel.importance_score))

        if limit:
            remaining = limit - len(channels)
            if remaining > 0:
                unvalidated_query = unvalidated_query.limit(remaining)
            else:
                unvalidated_query = unvalidated_query.limit(0)

        unvalidated_channels = unvalidated_query.all()

        # Combine lists
        all_channels = channels + unvalidated_channels
        channel_data = [
            {
                'id': c.id,
                'name': c.display_name,
                'youtube_channel_id': c.platform_metadata.get('video_links', {}).get('youtube', {}).get('channel_id'),
                'validated': c.platform_metadata.get('video_links', {}).get('youtube', {}).get('validated'),
                'importance': c.importance_score
            }
            for c in all_channels
        ]

    logger.info(f"Found {len(channels)} validated channels")
    logger.info(f"Found {len(unvalidated_channels)} unvalidated channels")
    logger.info(f"Total to process: {len(channel_data)}")

    if dry_run:
        logger.info("\n[DRY RUN] Would process these channels:")
        for ch in channel_data[:20]:
            status = "validated" if ch['validated'] else "needs validation"
            logger.info(f"  - {ch['name']} ({status})")
        if len(channel_data) > 20:
            logger.info(f"  ... and {len(channel_data) - 20} more")
        return {
            'status': 'dry_run',
            'channels_found': len(channel_data),
            'validated': len(channels),
            'unvalidated': len(unvalidated_channels)
        }

    # Process channels
    indexer = YouTubeVideoCacheIndexer(logger=logger)
    matcher = PodcastVideoMatcher(logger=logger, use_llm=False, skip_rumble=True) if match_episodes else None

    stats = {
        'channels_processed': 0,
        'channels_refreshed': 0,
        'channels_validated': 0,
        'channels_false_positive': 0,
        'channels_skipped': 0,
        'channels_error': 0,
        'videos_indexed': 0,
        'episodes_matched': 0,
        'quota_used': 0,
    }

    for i, ch in enumerate(channel_data):
        logger.info(f"\n[{i+1}/{len(channel_data)}] {ch['name']}")

        try:
            if ch['validated']:
                # Already validated - just refresh the cache
                result = await indexer.index_channel_videos(
                    youtube_channel_id=ch['youtube_channel_id'],
                    max_videos=500,
                    force=True  # Force refresh even if recently indexed
                )

                if result.get('status') == 'success':
                    stats['channels_refreshed'] += 1
                    stats['videos_indexed'] += result.get('videos_indexed', 0)
                    logger.info(f"  Refreshed: {result.get('videos_indexed', 0)} videos")
                elif result.get('status') == 'skipped':
                    stats['channels_skipped'] += 1
                    logger.info(f"  Skipped: {result.get('reason', 'unknown')}")
                else:
                    stats['channels_error'] += 1
                    logger.warning(f"  Error: {result.get('error', 'unknown')}")

            else:
                # Not yet validated - run validation
                if matcher:
                    result = await matcher.match_episodes_for_channel_cached(
                        podcast_channel_id=ch['id'],
                        episode_limit=episode_limit,
                        force=False
                    )
                else:
                    # Just index without matching
                    result = await indexer.index_for_podcast_channel(
                        podcast_channel_id=ch['id'],
                        max_videos=50,  # Start with validation page only
                        force=False
                    )

                if result.get('status') == 'success':
                    stats['channels_validated'] += 1
                    stats['videos_indexed'] += result.get('cache_video_count', result.get('videos_indexed', 0))
                    stats['episodes_matched'] += result.get('matched', 0)
                    logger.info(f"  Validated: {result.get('matched', 0)} episodes matched")
                elif result.get('status') == 'false_positive':
                    stats['channels_false_positive'] += 1
                    logger.info(f"  False positive: {result.get('reason', 'no matches')}")
                elif result.get('status') == 'skipped':
                    stats['channels_skipped'] += 1
                    logger.info(f"  Skipped: {result.get('reason', 'unknown')}")
                else:
                    stats['channels_error'] += 1
                    logger.warning(f"  Error: {result.get('error', 'unknown')}")

            stats['channels_processed'] += 1

            # Match episodes for validated channels if requested
            if match_episodes and ch['validated'] and matcher:
                match_result = await matcher.match_episodes_for_channel_cached(
                    podcast_channel_id=ch['id'],
                    episode_limit=episode_limit,
                    force=False
                )
                if match_result.get('status') == 'success':
                    stats['episodes_matched'] += match_result.get('matched', 0)

        except Exception as e:
            logger.error(f"  Exception: {e}")
            stats['channels_error'] += 1

        # Brief pause between channels
        await asyncio.sleep(0.5)

    # Get final quota usage
    stats['quota_used'] = indexer.get_total_quota_usage()

    duration = time.time() - start_time

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Duration: {duration:.1f}s")
    logger.info(f"Channels processed: {stats['channels_processed']}")
    logger.info(f"  - Refreshed: {stats['channels_refreshed']}")
    logger.info(f"  - Newly validated: {stats['channels_validated']}")
    logger.info(f"  - False positives: {stats['channels_false_positive']}")
    logger.info(f"  - Skipped: {stats['channels_skipped']}")
    logger.info(f"  - Errors: {stats['channels_error']}")
    logger.info(f"Videos indexed: {stats['videos_indexed']}")
    logger.info(f"Episodes matched: {stats['episodes_matched']}")
    logger.info(f"YouTube API quota used: {stats['quota_used']}")

    # Output JSON summary for scheduled task manager
    summary = {
        'status': 'success',
        'duration_seconds': round(duration, 1),
        **stats
    }
    print(f"\n__TASK_SUMMARY__: {json.dumps(summary)}")

    return summary


async def main():
    parser = argparse.ArgumentParser(
        description='Refresh YouTube video cache for validated podcast channels'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum channels to process (default: all)'
    )
    parser.add_argument(
        '--match-episodes',
        action='store_true',
        help='Also match new episodes after refreshing cache'
    )
    parser.add_argument(
        '--episode-limit',
        type=int,
        default=50,
        help='Max episodes per channel to match (default: 50)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually doing it'
    )

    args = parser.parse_args()

    await refresh_youtube_cache(
        limit=args.limit,
        match_episodes=args.match_episodes,
        episode_limit=args.episode_limit,
        dry_run=args.dry_run
    )


if __name__ == '__main__':
    asyncio.run(main())
