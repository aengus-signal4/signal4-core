#!/usr/bin/env python3
"""
YouTube Video Cache Indexer
===========================

Indexes YouTube channel videos into a cache table for efficient podcast-to-YouTube
episode matching. This is completely isolated from the download pipeline - it does
NOT create Channel records, ChannelSource records, or Content records.

The cache enables:
- One-time indexing of a channel's videos
- Fast episode matching without live API calls
- Reuse of existing YouTube API patterns

Usage:
    # Index videos for a podcast channel by ID
    indexer = YouTubeVideoCacheIndexer()
    await indexer.index_for_podcast_channel(5881)

    # Index videos directly by YouTube channel ID
    await indexer.index_channel_videos("UCxxxxx", max_videos=500)
"""

import asyncio
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

import isodate
from dotenv import load_dotenv
from googleapiclient.discovery import build
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm.attributes import flag_modified

from src.database.models import Channel, YouTubeVideoCache
from src.database.session import get_session
from src.utils.logger import setup_indexer_logger
from src.utils.paths import get_env_path

# Load environment variables
load_dotenv(get_env_path())


def parse_duration(duration_str: str) -> int:
    """Convert ISO 8601 duration to seconds."""
    try:
        duration = isodate.parse_duration(duration_str)
        return int(duration.total_seconds())
    except (isodate.ISO8601Error, TypeError, ValueError):
        return 0


class YouTubeVideoCacheIndexer:
    """
    Indexes YouTube channel videos into cache table for episode matching.

    This indexer is completely isolated from the download pipeline:
    - Does NOT create Channel records
    - Does NOT create ChannelSource records
    - Does NOT create Content records
    - Does NOT trigger any downloads

    It simply caches video metadata for fast episode matching.
    """

    # Quota costs for different operations
    QUOTA_COSTS = {
        'channels.list': 1,
        'playlistItems.list': 1,
        'videos.list': 1,
    }
    DAILY_QUOTA_LIMIT = 10000

    def __init__(self, logger=None):
        self.logger = logger or setup_indexer_logger('youtube_video_cache')
        self.api_keys = []
        self.current_key_index = 0
        self.quota_usage = {}
        self.youtube = None

        # Load API keys
        api_keys_str = os.getenv('YOUTUBE_API_KEYS')
        if not api_keys_str:
            self.logger.warning("No YouTube API keys found - cache indexing disabled")
            return

        self.api_keys = [k.strip() for k in api_keys_str.split(',') if k.strip()]
        if not self.api_keys:
            self.logger.warning("No valid YouTube API keys - cache indexing disabled")
            return

        # Initialize quota tracking
        for key in self.api_keys:
            self.quota_usage[key] = 0

        # Initialize client
        self._init_youtube_client()

    def _init_youtube_client(self):
        """Initialize YouTube API client with current key."""
        if not self.api_keys:
            return
        self.youtube = build('youtube', 'v3', developerKey=self.api_keys[self.current_key_index])

    def _rotate_api_key(self):
        """Rotate to the next API key when quota is exhausted."""
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self._init_youtube_client()
        self.logger.info(f"Rotated to API key index {self.current_key_index}")

    async def _execute_request(self, request, operation_name: str):
        """Execute YouTube API request with retry and key rotation."""
        if not self.youtube:
            return None

        loop = asyncio.get_event_loop()

        for _ in range(len(self.api_keys)):
            try:
                result = await loop.run_in_executor(None, request.execute)

                # Track quota usage
                current_key = self.api_keys[self.current_key_index]
                cost = self.QUOTA_COSTS.get(operation_name, 1)
                self.quota_usage[current_key] += cost

                return result

            except Exception as e:
                if 'quota' in str(e).lower():
                    self.logger.warning(f"API key {self.current_key_index} quota exceeded, rotating")
                    self._rotate_api_key()
                    continue
                raise

        self.logger.error("All YouTube API keys exhausted")
        return None

    def get_total_quota_usage(self) -> int:
        """Get total quota usage across all keys."""
        return sum(self.quota_usage.values())

    async def index_channel_videos(
        self,
        youtube_channel_id: str,
        max_videos: int = 500,
        force: bool = False
    ) -> Dict:
        """
        Fetch and cache all videos from a YouTube channel.

        Args:
            youtube_channel_id: YouTube channel ID (UC...)
            max_videos: Maximum number of videos to index (default 500)
            force: If True, re-index even if recently indexed

        Returns:
            Dict with status, videos_indexed, quota_used, etc.
        """
        if not self.youtube:
            return {'status': 'error', 'error': 'YouTube API not initialized'}

        self.logger.info(f"Indexing videos for YouTube channel: {youtube_channel_id}")

        # Check if recently indexed (unless force=True)
        if not force:
            with get_session() as session:
                recent = session.query(YouTubeVideoCache).filter(
                    YouTubeVideoCache.youtube_channel_id == youtube_channel_id
                ).first()

                if recent and recent.indexed_at:
                    days_since = (datetime.now(timezone.utc) - recent.indexed_at.replace(tzinfo=timezone.utc)).days
                    if days_since < 30:
                        self.logger.info(f"Channel {youtube_channel_id} was indexed {days_since} days ago, skipping (use force=True to re-index)")
                        return {
                            'status': 'skipped',
                            'reason': f'indexed {days_since} days ago',
                            'youtube_channel_id': youtube_channel_id,
                        }

        try:
            # Get channel info and uploads playlist
            ch_request = self.youtube.channels().list(
                part="contentDetails,snippet,statistics",
                id=youtube_channel_id
            )
            ch_response = await self._execute_request(ch_request, 'channels.list')

            if not ch_response or not ch_response.get('items'):
                return {'status': 'error', 'error': f'Channel not found: {youtube_channel_id}'}

            channel_info = ch_response['items'][0]
            channel_name = channel_info['snippet']['title']
            uploads_playlist = channel_info['contentDetails']['relatedPlaylists']['uploads']
            total_videos = int(channel_info['statistics'].get('videoCount', 0))

            self.logger.info(f"Channel '{channel_name}' has {total_videos} videos, indexing up to {max_videos}")

            # Fetch videos from uploads playlist
            videos_indexed = 0
            page_token = None
            all_videos = []

            while videos_indexed < max_videos:
                pl_request = self.youtube.playlistItems().list(
                    part="snippet,contentDetails",
                    playlistId=uploads_playlist,
                    maxResults=min(50, max_videos - videos_indexed),
                    pageToken=page_token
                )
                pl_response = await self._execute_request(pl_request, 'playlistItems.list')

                if not pl_response:
                    break

                items = pl_response.get('items', [])
                if not items:
                    break

                # Get video IDs for batch detail request
                video_ids = [item['contentDetails']['videoId'] for item in items]

                # Get detailed video info
                video_request = self.youtube.videos().list(
                    part="snippet,contentDetails,statistics",
                    id=','.join(video_ids)
                )
                video_response = await self._execute_request(video_request, 'videos.list')

                if video_response:
                    for video in video_response.get('items', []):
                        video_data = self._extract_video_data(youtube_channel_id, video)
                        all_videos.append(video_data)
                        videos_indexed += 1

                page_token = pl_response.get('nextPageToken')
                if not page_token:
                    break

            # Upsert all videos into cache
            if all_videos:
                self._upsert_videos(all_videos)

            quota_used = self.get_total_quota_usage()
            self.logger.info(f"Indexed {videos_indexed} videos for channel '{channel_name}' (quota used: {quota_used})")

            return {
                'status': 'success',
                'youtube_channel_id': youtube_channel_id,
                'channel_name': channel_name,
                'videos_indexed': videos_indexed,
                'total_videos': total_videos,
                'quota_used': quota_used,
            }

        except Exception as e:
            self.logger.error(f"Error indexing channel {youtube_channel_id}: {e}")
            return {'status': 'error', 'error': str(e)}

    def _extract_video_data(self, youtube_channel_id: str, video: Dict) -> Dict:
        """Extract video data from API response."""
        snippet = video.get('snippet', {})
        content_details = video.get('contentDetails', {})
        statistics = video.get('statistics', {})

        # Parse publish date
        publish_date = None
        if snippet.get('publishedAt'):
            try:
                publish_date = datetime.fromisoformat(
                    snippet['publishedAt'].replace('Z', '+00:00')
                )
            except ValueError:
                pass

        # Get best thumbnail URL
        thumbnails = snippet.get('thumbnails', {})
        thumbnail_url = None
        for quality in ['maxres', 'high', 'medium', 'default']:
            if quality in thumbnails:
                thumbnail_url = thumbnails[quality]['url']
                break

        return {
            'youtube_channel_id': youtube_channel_id,
            'video_id': video['id'],
            'title': snippet.get('title'),
            'description': snippet.get('description', '')[:5000],  # Limit description size
            'publish_date': publish_date,
            'duration': parse_duration(content_details.get('duration', '')),
            'view_count': int(statistics.get('viewCount', 0)) if statistics.get('viewCount') else None,
            'thumbnail_url': thumbnail_url,
            'meta_data': {
                'tags': snippet.get('tags', []),
                'category_id': snippet.get('categoryId'),
                'live_status': snippet.get('liveBroadcastContent'),
                'like_count': int(statistics.get('likeCount', 0)) if statistics.get('likeCount') else None,
                'comment_count': int(statistics.get('commentCount', 0)) if statistics.get('commentCount') else None,
            },
            'indexed_at': datetime.now(timezone.utc),
        }

    def _upsert_videos(self, videos: List[Dict]) -> int:
        """Upsert videos into cache table."""
        if not videos:
            return 0

        with get_session() as session:
            # Use PostgreSQL INSERT ... ON CONFLICT for upsert
            stmt = insert(YouTubeVideoCache).values(videos)
            stmt = stmt.on_conflict_do_update(
                constraint='uq_youtube_video_cache_channel_video',
                set_={
                    'title': stmt.excluded.title,
                    'description': stmt.excluded.description,
                    'publish_date': stmt.excluded.publish_date,
                    'duration': stmt.excluded.duration,
                    'view_count': stmt.excluded.view_count,
                    'thumbnail_url': stmt.excluded.thumbnail_url,
                    'meta_data': stmt.excluded.meta_data,
                    'indexed_at': stmt.excluded.indexed_at,
                }
            )
            session.execute(stmt)
            session.commit()

        return len(videos)

    async def index_for_podcast_channel(
        self,
        podcast_channel_id: int,
        max_videos: int = 500,
        force: bool = False
    ) -> Dict:
        """
        Index YouTube videos for a podcast's matched YouTube channel.

        This looks up the podcast channel's platform_metadata['video_links']['youtube']
        to find the matched YouTube channel, then indexes its videos.

        Args:
            podcast_channel_id: Database ID of the podcast channel
            max_videos: Maximum number of videos to index
            force: If True, re-index even if recently indexed

        Returns:
            Dict with status, videos_indexed, etc.
        """
        with get_session() as session:
            channel = session.query(Channel).filter(
                Channel.id == podcast_channel_id,
                Channel.platform == 'podcast'
            ).first()

            if not channel:
                return {'status': 'error', 'error': f'Podcast channel not found: {podcast_channel_id}'}

            # Get YouTube match from platform_metadata
            pm = channel.platform_metadata or {}
            video_links = pm.get('video_links', {})
            youtube_info = video_links.get('youtube', {})

            youtube_channel_id = youtube_info.get('channel_id')
            if not youtube_channel_id:
                return {
                    'status': 'error',
                    'error': f'No YouTube channel matched for podcast {channel.display_name}',
                    'podcast_channel_id': podcast_channel_id,
                }

            self.logger.info(f"Indexing YouTube channel {youtube_channel_id} for podcast '{channel.display_name}'")

        # Index the YouTube channel
        result = await self.index_channel_videos(youtube_channel_id, max_videos, force)

        # Update podcast channel's platform_metadata with indexed timestamp
        if result.get('status') == 'success':
            with get_session() as session:
                db_channel = session.query(Channel).filter(Channel.id == podcast_channel_id).first()
                if db_channel:
                    pm = dict(db_channel.platform_metadata or {})
                    if 'video_links' not in pm:
                        pm['video_links'] = {}
                    if 'youtube' not in pm['video_links']:
                        pm['video_links']['youtube'] = {}
                    pm['video_links']['youtube']['cache_indexed_at'] = datetime.now(timezone.utc).isoformat()
                    pm['video_links']['youtube']['cache_video_count'] = result.get('videos_indexed', 0)
                    db_channel.platform_metadata = pm
                    flag_modified(db_channel, 'platform_metadata')
                    session.commit()

        result['podcast_channel_id'] = podcast_channel_id
        return result

    def get_cached_video_count(self, youtube_channel_id: str) -> int:
        """Get the number of cached videos for a YouTube channel."""
        with get_session() as session:
            return session.query(YouTubeVideoCache).filter(
                YouTubeVideoCache.youtube_channel_id == youtube_channel_id
            ).count()

    def get_cached_videos(
        self,
        youtube_channel_id: str,
        limit: int = None
    ) -> List[YouTubeVideoCache]:
        """Get cached videos for a YouTube channel."""
        with get_session() as session:
            query = session.query(YouTubeVideoCache).filter(
                YouTubeVideoCache.youtube_channel_id == youtube_channel_id
            ).order_by(YouTubeVideoCache.publish_date.desc())

            if limit:
                query = query.limit(limit)

            return query.all()
