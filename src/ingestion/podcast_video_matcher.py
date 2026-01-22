#!/usr/bin/env python3
"""
Podcast-to-Video Channel Matcher
================================

Discovers YouTube and Rumble channels for podcasts WITHOUT triggering downloads.

This tool audits podcast channels to find their corresponding video channels on
YouTube (primary) and Rumble (fallback). All linkage is stored in platform_metadata
JSONB fields, NOT in ChannelSource (which would trigger the indexing pipeline).

Storage:
- Channel level: channels.platform_metadata['video_links']
- Episode level: content.meta_data['video_link']

Usage:
    python -m src.ingestion.podcast_video_matcher --limit 100
    python -m src.ingestion.podcast_video_matcher --min-importance 5.0
    python -m src.ingestion.podcast_video_matcher --channel-id 12345
"""

import asyncio
import argparse
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple, Any

import yt_dlp
from ddgs import DDGS
from googleapiclient.discovery import build
from sqlalchemy import desc
from sqlalchemy.orm.attributes import flag_modified

from src.database.session import get_session
from src.database.models import Channel, Content
from src.utils.logger import setup_indexer_logger
from src.utils.llm_client import LLMClient
from src.utils.paths import get_env_path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(get_env_path())


class PodcastVideoMatcher:
    """
    Discovers YouTube/Rumble channels for podcasts WITHOUT triggering downloads.

    Stores all linkage in platform_metadata/meta_data JSONB fields,
    NOT in ChannelSource (which would trigger the indexing pipeline).
    """

    # YouTube API quota costs
    QUOTA_COSTS = {
        'search.list': 100,
        'channels.list': 1,
        'playlistItems.list': 1,
    }
    DAILY_QUOTA_LIMIT = 10000

    # Matching thresholds
    NAME_EXACT_THRESHOLD = 0.95
    NAME_FUZZY_THRESHOLD = 0.85  # Raised from 0.80 to reduce false positives
    MIN_CONFIDENCE_TO_STORE = 0.80  # Don't store matches below this confidence
    EPISODE_TITLE_THRESHOLD = 0.75
    DURATION_TOLERANCE_PERCENT = 0.10  # 10% tolerance
    DATE_TOLERANCE_DAYS = 7

    def __init__(self, logger=None, use_llm: bool = True):
        self.logger = logger or setup_indexer_logger('podcast_video_matcher')
        self.use_llm = use_llm

        # YouTube API setup
        self.api_keys = []
        self.current_key_index = 0
        self.quota_usage = {}
        self.youtube = None
        self._init_youtube_client()

        # LLM client for intelligent matching (tier_1 for best accuracy)
        self.llm_client = LLMClient(tier="tier_1", priority=3, task_type="analysis") if use_llm else None

        # yt-dlp for Rumble (no API needed)
        self.ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
            'ignoreerrors': True,
        }

        # Audit statistics
        self.stats = {
            'channels_audited': 0,
            'youtube_found': 0,
            'rumble_found': 0,
            'no_match': 0,
            'already_audited': 0,
            'episodes_matched': 0,
            'quota_used': 0,
            'llm_calls': 0,
        }

    def _init_youtube_client(self):
        """Initialize YouTube API client with API key rotation."""
        api_keys_str = os.getenv('YOUTUBE_API_KEYS')
        if not api_keys_str:
            self.logger.warning("No YouTube API keys found - YouTube matching disabled")
            return

        self.api_keys = [k.strip() for k in api_keys_str.split(',') if k.strip()]
        if not self.api_keys:
            self.logger.warning("No valid YouTube API keys - YouTube matching disabled")
            return

        for key in self.api_keys:
            self.quota_usage[key] = 0

        self.youtube = build('youtube', 'v3', developerKey=self.api_keys[self.current_key_index])
        self.logger.info(f"Initialized YouTube client with {len(self.api_keys)} API keys")

    def _rotate_api_key(self):
        """Rotate to next API key when quota is exhausted."""
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self.youtube = build('youtube', 'v3', developerKey=self.api_keys[self.current_key_index])
        self.logger.info(f"Rotated to API key index {self.current_key_index}")

    async def _execute_youtube_request(self, request, operation_name: str):
        """Execute YouTube API request with retry and key rotation."""
        if not self.youtube:
            return None

        loop = asyncio.get_event_loop()

        for _ in range(len(self.api_keys)):
            try:
                result = await loop.run_in_executor(None, request.execute)

                # Track quota
                current_key = self.api_keys[self.current_key_index]
                cost = self.QUOTA_COSTS.get(operation_name, 1)
                self.quota_usage[current_key] += cost
                self.stats['quota_used'] += cost

                return result

            except Exception as e:
                if 'quota' in str(e).lower():
                    self.logger.warning(f"API key {self.current_key_index} quota exceeded, rotating")
                    self._rotate_api_key()
                    continue
                raise

        self.logger.error("All YouTube API keys exhausted")
        return None

    def _similarity_ratio(self, s1: str, s2: str) -> float:
        """Calculate similarity ratio between two strings."""
        # Normalize strings
        s1 = re.sub(r'[^\w\s]', '', s1.lower().strip())
        s2 = re.sub(r'[^\w\s]', '', s2.lower().strip())
        return SequenceMatcher(None, s1, s2).ratio()

    def _normalize_podcast_name(self, name: str) -> str:
        """Normalize podcast name for searching."""
        # Remove common podcast suffixes
        name = re.sub(r'\s*(podcast|show|radio|audio|official)s?\s*$', '', name, flags=re.IGNORECASE)
        # Remove special characters but keep spaces
        name = re.sub(r'[^\w\s]', ' ', name)
        # Collapse multiple spaces
        name = re.sub(r'\s+', ' ', name).strip()
        return name

    def _extract_youtube_channel_id(self, url: str) -> Optional[str]:
        """Extract YouTube channel ID from various URL formats."""
        patterns = [
            r'youtube\.com/channel/([a-zA-Z0-9_-]+)',  # /channel/UCxxxx
            r'youtube\.com/@([a-zA-Z0-9_-]+)',         # /@handle
            r'youtube\.com/c/([a-zA-Z0-9_-]+)',        # /c/customname
            r'youtube\.com/user/([a-zA-Z0-9_-]+)',     # /user/username
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    def _search_duckduckgo(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search DuckDuckGo and return results."""
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
                return results
        except Exception as e:
            self.logger.debug(f"DuckDuckGo search error: {e}")
            return []

    async def search_youtube_channel(self, podcast_name: str, creator: str = None) -> Optional[Dict]:
        """
        Search for YouTube channel using DuckDuckGo (free), then verify with YouTube API (1 unit).

        This approach reduces API quota from 100 units (search.list) to 1 unit (channels.list).

        Returns match info or None if no confident match found.
        """
        # Build search queries for DuckDuckGo
        # Multiple query variations to maximize channel discovery
        normalized = self._normalize_podcast_name(podcast_name)
        search_queries = [
            f'"{podcast_name}" site:youtube.com/channel',   # Exact name, channel URLs only
            f'"{podcast_name}" youtube channel official',   # Exact name with channel keyword
            f'"{normalized}" site:youtube.com/channel',     # Normalized name
        ]
        if creator and creator.lower() != podcast_name.lower():
            search_queries.append(f'"{podcast_name}" "{creator}" site:youtube.com')

        # Collect candidate channel IDs/handles from DuckDuckGo results
        candidates = []  # List of (identifier, is_channel_id, source_title)

        for query in search_queries:
            results = self._search_duckduckgo(query, max_results=5)

            for result in results:
                url = result.get('href', '')
                title = result.get('title', '')

                if 'youtube.com' not in url:
                    continue

                identifier = self._extract_youtube_channel_id(url)
                if identifier:
                    is_channel_id = identifier.startswith('UC')
                    candidates.append((identifier, is_channel_id, title, url))

        if not candidates:
            self.logger.debug(f"No YouTube candidates found via DuckDuckGo for '{podcast_name}'")
            return None

        # Deduplicate candidates
        seen = set()
        unique_candidates = []
        for c in candidates:
            if c[0] not in seen:
                seen.add(c[0])
                unique_candidates.append(c)

        best_match = None
        best_confidence = 0.0

        # Verify each candidate with YouTube API (only 1 unit per call!)
        for identifier, is_channel_id, source_title, source_url in unique_candidates[:5]:  # Limit to 5
            try:
                if is_channel_id:
                    # Direct channel ID lookup
                    request = self.youtube.channels().list(
                        part="snippet,statistics",
                        id=identifier
                    )
                else:
                    # Handle/username - need to use forHandle parameter
                    request = self.youtube.channels().list(
                        part="snippet,statistics",
                        forHandle=identifier
                    )

                response = await self._execute_youtube_request(request, 'channels.list')

                if not response or not response.get('items'):
                    continue

                item = response['items'][0]
                channel_id = item['id']
                channel_title = item['snippet']['title']
                channel_description = item['snippet'].get('description', '')
                stats = item.get('statistics', {})

                # Calculate match confidence
                name_similarity = self._similarity_ratio(podcast_name, channel_title)

                # Also check if YouTube channel name is a substring of podcast name
                # (e.g., "The Diary Of A CEO" in "The Diary Of A CEO with Steven Bartlett")
                yt_name_lower = channel_title.lower().strip()
                podcast_name_lower = podcast_name.lower().strip()
                is_substring = (
                    yt_name_lower in podcast_name_lower or
                    podcast_name_lower in yt_name_lower
                )

                # Check for exact/near-exact match
                if name_similarity >= self.NAME_EXACT_THRESHOLD:
                    confidence = 0.95
                    method = 'name_exact'
                elif name_similarity >= self.NAME_FUZZY_THRESHOLD:
                    confidence = 0.85
                    method = 'name_fuzzy'
                elif is_substring and len(yt_name_lower) > 10:
                    # YouTube name is contained in podcast name (or vice versa)
                    # Require minimum length to avoid false positives
                    confidence = 0.90
                    method = 'name_substring'
                else:
                    # Check if podcast name appears in description
                    if podcast_name.lower() in channel_description.lower():
                        confidence = 0.75
                        method = 'description_match'
                    else:
                        continue

                # Boost confidence if creator matches
                if creator:
                    creator_in_title = creator.lower() in channel_title.lower()
                    creator_in_desc = creator.lower() in channel_description.lower()
                    if creator_in_title or creator_in_desc:
                        confidence = min(confidence + 0.10, 0.98)

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = {
                        'platform': 'youtube',
                        'channel_id': channel_id,
                        'channel_name': channel_title,
                        'channel_url': f'https://www.youtube.com/channel/{channel_id}',
                        'match_confidence': round(confidence, 2),
                        'match_method': method,
                        'search_source': 'duckduckgo',
                        'subscriber_count': int(stats.get('subscriberCount', 0)),
                        'video_count': int(stats.get('videoCount', 0)),
                    }

            except Exception as e:
                self.logger.debug(f"Error verifying YouTube channel {identifier}: {e}")
                continue

        return best_match

    async def search_rumble_channel(self, podcast_name: str, creator: str = None) -> Optional[Dict]:
        """
        Search Rumble for a channel matching the podcast name using yt-dlp.

        Returns match info or None if no confident match found.
        """
        search_queries = [
            self._normalize_podcast_name(podcast_name),
        ]
        if creator and creator != podcast_name:
            search_queries.append(creator)

        best_match = None
        best_confidence = 0.0

        for query in search_queries:
            if not query:
                continue

            try:
                # Rumble search URL
                search_url = f"https://rumble.com/search/channel?q={query.replace(' ', '+')}"

                with yt_dlp.YoutubeDL({**self.ydl_opts, 'extract_flat': 'in_playlist'}) as ydl:
                    # This extracts search results
                    try:
                        # Try direct channel URL patterns first
                        channel_url = f"https://rumble.com/c/{query.replace(' ', '')}"
                        channel_info = ydl.extract_info(channel_url, download=False)

                        if channel_info and channel_info.get('title'):
                            channel_title = channel_info['title']
                            name_similarity = self._similarity_ratio(podcast_name, channel_title)

                            if name_similarity >= self.NAME_FUZZY_THRESHOLD:
                                confidence = 0.75 if name_similarity >= self.NAME_EXACT_THRESHOLD else 0.65

                                if confidence > best_confidence:
                                    best_confidence = confidence
                                    best_match = {
                                        'platform': 'rumble',
                                        'channel_id': channel_info.get('id', query),
                                        'channel_name': channel_title,
                                        'channel_url': channel_url,
                                        'match_confidence': round(confidence, 2),
                                        'match_method': 'direct_url',
                                        'search_query': query,
                                        'video_count': len(channel_info.get('entries', [])),
                                    }
                    except Exception:
                        pass  # Channel doesn't exist at that URL

            except Exception as e:
                self.logger.debug(f"Rumble search error for '{query}': {e}")
                continue

        return best_match

    async def audit_channel(self, channel: Channel, skip_if_audited: bool = True) -> Dict:
        """
        Audit a podcast channel to find corresponding YouTube/Rumble channels.

        Stores results in channel.platform_metadata['video_links'].
        Does NOT create ChannelSource entries.

        Args:
            channel: Podcast Channel to audit
            skip_if_audited: Skip if already audited within last 90 days

        Returns:
            Audit result dict
        """
        pm = channel.platform_metadata or {}
        video_links = pm.get('video_links', {})

        # Check if recently audited
        if skip_if_audited and video_links.get('audited_at'):
            try:
                last_audit = datetime.fromisoformat(video_links['audited_at'])
                days_since = (datetime.now(timezone.utc).replace(tzinfo=None) - last_audit).days
                if days_since < 90:
                    self.stats['already_audited'] += 1
                    return {
                        'status': 'skipped',
                        'reason': f'audited {days_since} days ago',
                        'channel_id': channel.id,
                        'channel_name': channel.display_name,
                    }
            except Exception:
                pass

        self.stats['channels_audited'] += 1
        podcast_name = channel.display_name
        creator = pm.get('creator', '')

        self.logger.info(f"Auditing: {podcast_name}")

        result = {
            'channel_id': channel.id,
            'channel_name': podcast_name,
            'youtube': None,
            'rumble': None,
        }

        # Try YouTube first (primary)
        youtube_match = await self.search_youtube_channel(podcast_name, creator)
        if youtube_match:
            confidence = youtube_match['match_confidence']
            if confidence >= self.MIN_CONFIDENCE_TO_STORE:
                result['youtube'] = youtube_match
                self.stats['youtube_found'] += 1
                self.logger.info(f"  YouTube match: {youtube_match['channel_name']} "
                               f"(confidence: {confidence})")
            else:
                self.logger.info(f"  YouTube candidate rejected (low confidence {confidence}): "
                               f"{youtube_match['channel_name']}")
                youtube_match = None  # Don't store

        # Try Rumble as fallback (or in addition for redundancy)
        rumble_match = await self.search_rumble_channel(podcast_name, creator)
        if rumble_match:
            confidence = rumble_match['match_confidence']
            if confidence >= self.MIN_CONFIDENCE_TO_STORE:
                result['rumble'] = rumble_match
                self.stats['rumble_found'] += 1
                self.logger.info(f"  Rumble match: {rumble_match['channel_name']} "
                               f"(confidence: {confidence})")
            else:
                self.logger.info(f"  Rumble candidate rejected (low confidence {confidence}): "
                               f"{rumble_match['channel_name']}")
                rumble_match = None  # Don't store

        if not youtube_match and not rumble_match:
            self.stats['no_match'] += 1
            self.logger.info(f"  No video channel found")

        # Store results in platform_metadata (NOT ChannelSource!)
        video_links = {
            'audited_at': datetime.now(timezone.utc).isoformat(),
        }
        if youtube_match:
            video_links['youtube'] = youtube_match
        if rumble_match:
            video_links['rumble'] = rumble_match

        # Update channel
        with get_session() as session:
            db_channel = session.query(Channel).filter(Channel.id == channel.id).first()
            if db_channel:
                pm = dict(db_channel.platform_metadata or {})
                pm['video_links'] = video_links
                db_channel.platform_metadata = pm
                flag_modified(db_channel, 'platform_metadata')
                session.commit()

        result['status'] = 'success'
        return result

    async def match_episode_to_video(
        self,
        episode: Content,
        video_channel_id: str,
        platform: str = 'youtube'
    ) -> Optional[Dict]:
        """
        Match a podcast episode to a YouTube/Rumble video.

        Uses title similarity, duration matching, and publish date proximity.
        Stores result in episode.meta_data['video_link'].

        Args:
            episode: Podcast Content record
            video_channel_id: YouTube channel ID or Rumble channel slug
            platform: 'youtube' or 'rumble'

        Returns:
            Match info or None
        """
        if platform == 'youtube':
            return await self._match_episode_youtube(episode, video_channel_id)
        elif platform == 'rumble':
            return await self._match_episode_rumble(episode, video_channel_id)
        return None

    async def _match_episode_youtube(self, episode: Content, channel_id: str) -> Optional[Dict]:
        """Match episode to YouTube video."""
        if not self.youtube:
            return None

        try:
            # Get channel's uploads playlist
            ch_request = self.youtube.channels().list(
                part="contentDetails",
                id=channel_id
            )
            ch_response = await self._execute_youtube_request(ch_request, 'channels.list')

            if not ch_response or not ch_response.get('items'):
                return None

            uploads_playlist = ch_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']

            # Search through videos (limit to recent ones for efficiency)
            # We'll fetch multiple pages if needed
            best_match = None
            best_score = 0.0
            page_token = None
            pages_checked = 0
            max_pages = 5  # Limit API calls

            episode_title = episode.title or ''
            episode_duration = episode.duration or 0
            episode_date = episode.publish_date

            while pages_checked < max_pages:
                pl_request = self.youtube.playlistItems().list(
                    part="snippet,contentDetails",
                    playlistId=uploads_playlist,
                    maxResults=50,
                    pageToken=page_token
                )
                pl_response = await self._execute_youtube_request(pl_request, 'playlistItems.list')

                if not pl_response:
                    break

                for item in pl_response.get('items', []):
                    video_title = item['snippet']['title']
                    video_id = item['contentDetails']['videoId']
                    video_date_str = item['snippet'].get('publishedAt', '')

                    # Calculate title similarity
                    title_sim = self._similarity_ratio(episode_title, video_title)

                    if title_sim < self.EPISODE_TITLE_THRESHOLD:
                        continue

                    # Base score from title
                    score = title_sim

                    # Boost if dates are close
                    if episode_date and video_date_str:
                        try:
                            video_date = datetime.fromisoformat(video_date_str.replace('Z', '+00:00'))
                            if video_date.tzinfo:
                                video_date = video_date.replace(tzinfo=None)
                            if episode_date.tzinfo:
                                episode_date_naive = episode_date.replace(tzinfo=None)
                            else:
                                episode_date_naive = episode_date

                            days_diff = abs((video_date - episode_date_naive).days)
                            if days_diff <= self.DATE_TOLERANCE_DAYS:
                                score += 0.15
                        except Exception:
                            pass

                    if score > best_score:
                        best_score = score
                        best_match = {
                            'platform': 'youtube',
                            'video_id': video_id,
                            'video_url': f'https://www.youtube.com/watch?v={video_id}',
                            'video_title': video_title,
                            'match_confidence': round(score, 2),
                            'match_method': 'title_date',
                            'matched_at': datetime.now(timezone.utc).isoformat(),
                        }

                page_token = pl_response.get('nextPageToken')
                if not page_token:
                    break
                pages_checked += 1

            if best_match and best_match['match_confidence'] >= self.EPISODE_TITLE_THRESHOLD:
                self.stats['episodes_matched'] += 1

                # Store in episode's meta_data
                with get_session() as session:
                    db_episode = session.query(Content).filter(Content.id == episode.id).first()
                    if db_episode:
                        meta = dict(db_episode.meta_data or {})
                        meta['video_link'] = best_match
                        db_episode.meta_data = meta
                        flag_modified(db_episode, 'meta_data')
                        session.commit()

                return best_match

        except Exception as e:
            self.logger.warning(f"Error matching episode to YouTube: {e}")

        return None

    async def _match_episode_rumble(self, episode: Content, channel_slug: str) -> Optional[Dict]:
        """Match episode to Rumble video using yt-dlp."""
        try:
            channel_url = f"https://rumble.com/c/{channel_slug}"

            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                channel_info = ydl.extract_info(channel_url, download=False)

                if not channel_info:
                    return None

                episode_title = episode.title or ''
                best_match = None
                best_score = 0.0

                for video in channel_info.get('entries', [])[:100]:  # Limit to recent videos
                    if not video:
                        continue

                    video_title = video.get('title', '')
                    title_sim = self._similarity_ratio(episode_title, video_title)

                    if title_sim >= self.EPISODE_TITLE_THRESHOLD and title_sim > best_score:
                        best_score = title_sim
                        video_id = video.get('id', '')
                        best_match = {
                            'platform': 'rumble',
                            'video_id': video_id,
                            'video_url': video.get('url') or video.get('webpage_url', ''),
                            'video_title': video_title,
                            'match_confidence': round(title_sim, 2),
                            'match_method': 'title',
                            'matched_at': datetime.now(timezone.utc).isoformat(),
                        }

                if best_match:
                    self.stats['episodes_matched'] += 1

                    # Store in episode's meta_data
                    with get_session() as session:
                        db_episode = session.query(Content).filter(Content.id == episode.id).first()
                        if db_episode:
                            meta = dict(db_episode.meta_data or {})
                            meta['video_link'] = best_match
                            db_episode.meta_data = meta
                            flag_modified(db_episode, 'meta_data')
                            session.commit()

                    return best_match

        except Exception as e:
            self.logger.debug(f"Error matching episode to Rumble: {e}")

        return None

    async def run_audit(
        self,
        limit: int = 100,
        min_importance: float = 0.0,
        channel_id: int = None,
        match_episodes: bool = False,
        episode_limit: int = 10,
    ) -> Dict:
        """
        Run the audit process on podcast channels.

        Args:
            limit: Maximum number of channels to audit
            min_importance: Minimum importance_score to include
            channel_id: Specific channel ID to audit (ignores limit/importance)
            match_episodes: Whether to also match episodes to videos
            episode_limit: Max episodes per channel to match

        Returns:
            Summary statistics
        """
        self.logger.info("=" * 60)
        self.logger.info("Starting Podcast-to-Video Channel Audit")
        self.logger.info("=" * 60)

        start_time = time.time()
        results = []

        with get_session() as session:
            if channel_id:
                # Audit specific channel
                channels = session.query(Channel).filter(
                    Channel.id == channel_id,
                    Channel.platform == 'podcast'
                ).all()
            else:
                # Query podcast channels by importance
                query = session.query(Channel).filter(
                    Channel.platform == 'podcast',
                    Channel.status == 'active'
                )

                if min_importance > 0:
                    query = query.filter(Channel.importance_score >= min_importance)

                # Order by importance (highest first)
                query = query.order_by(desc(Channel.importance_score))

                if limit:
                    query = query.limit(limit)

                channels = query.all()

            self.logger.info(f"Found {len(channels)} channels to audit")

            for i, channel in enumerate(channels):
                self.logger.info(f"\n[{i+1}/{len(channels)}] {channel.display_name}")

                try:
                    result = await self.audit_channel(channel)
                    results.append(result)

                    # Match episodes if requested and video channel found
                    if match_episodes and result.get('status') == 'success':
                        video_info = result.get('youtube') or result.get('rumble')
                        if video_info:
                            await self._match_channel_episodes(
                                channel,
                                video_info,
                                episode_limit,
                                session
                            )

                except Exception as e:
                    self.logger.error(f"Error auditing {channel.display_name}: {e}")
                    results.append({
                        'status': 'error',
                        'channel_id': channel.id,
                        'channel_name': channel.display_name,
                        'error': str(e),
                    })

                # Rate limiting
                await asyncio.sleep(0.5)

        duration = time.time() - start_time

        # Generate summary
        summary = {
            'duration_seconds': round(duration, 1),
            'channels_audited': self.stats['channels_audited'],
            'youtube_found': self.stats['youtube_found'],
            'rumble_found': self.stats['rumble_found'],
            'no_match': self.stats['no_match'],
            'already_audited': self.stats['already_audited'],
            'episodes_matched': self.stats['episodes_matched'],
            'youtube_quota_used': self.stats['quota_used'],
            'results': results,
        }

        # Log summary
        self.logger.info("\n" + "=" * 60)
        self.logger.info("AUDIT COMPLETE")
        self.logger.info("=" * 60)
        self.logger.info(f"Duration: {duration:.1f}s")
        self.logger.info(f"Channels audited: {self.stats['channels_audited']}")
        self.logger.info(f"YouTube matches: {self.stats['youtube_found']}")
        self.logger.info(f"Rumble matches: {self.stats['rumble_found']}")
        self.logger.info(f"No match: {self.stats['no_match']}")
        self.logger.info(f"Already audited (skipped): {self.stats['already_audited']}")
        self.logger.info(f"Episodes matched: {self.stats['episodes_matched']}")
        self.logger.info(f"YouTube API quota used: {self.stats['quota_used']}")

        return summary

    async def _match_channel_episodes(
        self,
        channel: Channel,
        video_info: Dict,
        limit: int,
        session
    ):
        """Match recent episodes from a channel to videos."""
        platform = video_info['platform']
        channel_id = video_info['channel_id']

        # Get recent episodes for this channel
        episodes = session.query(Content).filter(
            Content.channel_id == channel.id,
            Content.platform == 'podcast'
        ).order_by(desc(Content.publish_date)).limit(limit).all()

        self.logger.info(f"  Matching {len(episodes)} recent episodes...")

        for episode in episodes:
            # Skip if already matched
            if episode.meta_data and episode.meta_data.get('video_link'):
                continue

            match = await self.match_episode_to_video(episode, channel_id, platform)
            if match:
                self.logger.info(f"    Matched: {episode.title[:50]}... -> {match['video_title'][:50]}...")


def setup_cli_logger(name: str) -> logging.Logger:
    """Set up a logger that outputs to console for CLI usage."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Remove any existing handlers
    logger.handlers = []

    # Add console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console)

    # Prevent propagation to avoid duplicate logs
    logger.propagate = False

    return logger


async def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Audit podcast channels for YouTube/Rumble video equivalents'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=100,
        help='Maximum channels to audit (default: 100)'
    )
    parser.add_argument(
        '--min-importance',
        type=float,
        default=0.0,
        help='Minimum importance_score threshold (default: 0)'
    )
    parser.add_argument(
        '--channel-id',
        type=int,
        help='Audit a specific channel by ID'
    )
    parser.add_argument(
        '--match-episodes',
        action='store_true',
        help='Also match episodes to videos'
    )
    parser.add_argument(
        '--episode-limit',
        type=int,
        default=10,
        help='Max episodes per channel to match (default: 10)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Re-audit even if recently audited'
    )

    args = parser.parse_args()

    # Use CLI logger for console output
    cli_logger = setup_cli_logger('podcast_video_matcher')
    matcher = PodcastVideoMatcher(logger=cli_logger)

    # If force flag, we need to modify the audit to not skip
    if args.force:
        # Temporarily disable skip logic by setting threshold to 0 days
        original_method = matcher.audit_channel
        async def force_audit(channel, skip_if_audited=True):
            return await original_method(channel, skip_if_audited=False)
        matcher.audit_channel = force_audit

    summary = await matcher.run_audit(
        limit=args.limit,
        min_importance=args.min_importance,
        channel_id=args.channel_id,
        match_episodes=args.match_episodes,
        episode_limit=args.episode_limit,
    )

    return summary


if __name__ == '__main__':
    asyncio.run(main())
