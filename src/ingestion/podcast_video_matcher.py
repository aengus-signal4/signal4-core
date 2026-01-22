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
from src.database.models import Channel, Content, ChannelSource
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

    # Channel name patterns to filter out (derivative channels, not main)
    DERIVATIVE_CHANNEL_PATTERNS = [
        r'\bshorts?\b',      # "Shorts" channels
        r'\bhighlights?\b',  # "Highlights" channels
        r'\bclips?\b',       # "Clips" channels
        r'\b-\s*topic\b',    # "- Topic" (YouTube auto-generated)
        r'\btopic$',         # ends with "Topic"
        r'\btrailers?\b',    # "Trailers" channels
        r'\bpodcast\s+highlights?\b',  # "Podcast Highlights"
        r'\bbest\s+of\b',    # "Best of" compilations
        r'\bmoments?\b',     # "Moments" channels
    ]

    def __init__(self, logger=None, use_llm: bool = True, skip_rumble: bool = False):
        self.logger = logger or setup_indexer_logger('podcast_video_matcher')
        self.use_llm = use_llm
        self.skip_rumble = skip_rumble

        # YouTube API setup
        self.api_keys = []
        self.current_key_index = 0
        self.quota_usage = {}
        self.youtube = None
        self._init_youtube_client()

        # LLM client for intelligent matching (tier_1 for best accuracy, priority 1 for processing)
        self.llm_client = LLMClient(tier="tier_1", priority=1, task_type="analysis") if use_llm else None

        # yt-dlp for Rumble (no API needed) - add timeout to prevent hanging
        self.ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
            'ignoreerrors': True,
            'socket_timeout': 15,  # 15 second timeout
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

    def _is_derivative_channel(self, channel_name: str) -> bool:
        """
        Check if a YouTube channel name indicates it's a derivative channel
        (Shorts, Highlights, Clips, Topic, etc.) rather than the main channel.
        """
        name_lower = channel_name.lower()
        for pattern in self.DERIVATIVE_CHANNEL_PATTERNS:
            if re.search(pattern, name_lower, re.IGNORECASE):
                return True
        return False

    def _check_indexed_channels(self, podcast_name: str, creator: str = None) -> Optional[Dict]:
        """
        Check if we already have a matching YouTube/Rumble channel indexed in our database.
        This provides a high-confidence match without using external APIs.
        """
        with get_session() as session:
            # Search for matching indexed YouTube channels
            normalized = self._normalize_podcast_name(podcast_name).lower()

            # Query indexed YouTube channels that might match
            query = session.query(Channel).filter(
                Channel.platform.in_(['youtube', 'rumble']),
                Channel.status == 'active'
            )

            candidates = query.all()
            best_match = None
            best_similarity = 0.0

            for ch in candidates:
                ch_name = ch.display_name or ''
                ch_name_lower = ch_name.lower()

                # Skip derivative channels
                if self._is_derivative_channel(ch_name):
                    continue

                # Check name similarity
                similarity = self._similarity_ratio(podcast_name, ch_name)

                # Also check against creator name
                creator_sim = 0.0
                if creator:
                    creator_sim = self._similarity_ratio(creator, ch_name)

                best_sim = max(similarity, creator_sim)

                if best_sim >= self.NAME_FUZZY_THRESHOLD and best_sim > best_similarity:
                    best_similarity = best_sim
                    best_match = {
                        'platform': ch.platform,
                        'channel_id': ch.channel_key,
                        'channel_name': ch_name,
                        'channel_url': f'https://www.youtube.com/channel/{ch.channel_key}' if ch.platform == 'youtube' else f'https://rumble.com/c/{ch.channel_key}',
                        'match_confidence': round(best_sim + 0.05, 2),  # Boost for already indexed
                        'match_method': 'indexed_channel',
                        'subscriber_count': ch.platform_metadata.get('subscriber_count', 0) if ch.platform_metadata else 0,
                    }

            if best_match:
                self.logger.info(f"  Found indexed channel match: {best_match['channel_name']}")

            return best_match

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

    def _extract_video_urls_from_text(self, text: str) -> Dict[str, List[str]]:
        """
        Extract YouTube and Rumble channel URLs from text (e.g., podcast description).

        Returns dict with 'youtube' and 'rumble' lists of URLs found.
        """
        if not text:
            return {'youtube': [], 'rumble': []}

        youtube_urls = []
        rumble_urls = []

        # YouTube patterns
        yt_patterns = [
            r'https?://(?:www\.)?youtube\.com/channel/[a-zA-Z0-9_-]+',
            r'https?://(?:www\.)?youtube\.com/@[a-zA-Z0-9_-]+',
            r'https?://(?:www\.)?youtube\.com/c/[a-zA-Z0-9_-]+',
            r'https?://(?:www\.)?youtube\.com/user/[a-zA-Z0-9_-]+',
        ]
        for pattern in yt_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            youtube_urls.extend(matches)

        # Rumble patterns
        rumble_patterns = [
            r'https?://(?:www\.)?rumble\.com/c/[a-zA-Z0-9_-]+',
            r'https?://(?:www\.)?rumble\.com/user/[a-zA-Z0-9_-]+',
        ]
        for pattern in rumble_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            rumble_urls.extend(matches)

        # Deduplicate
        return {
            'youtube': list(set(youtube_urls)),
            'rumble': list(set(rumble_urls)),
        }

    def _search_duckduckgo(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search DuckDuckGo and return results."""
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
                return results
        except Exception as e:
            self.logger.debug(f"DuckDuckGo search error: {e}")
            return []

    async def _llm_select_best_channel(
        self,
        podcast_name: str,
        podcast_description: str,
        creator: str,
        candidates: List[Dict]
    ) -> Optional[Dict]:
        """
        Use LLM to intelligently select the best YouTube channel match.

        Args:
            podcast_name: Name of the podcast
            podcast_description: Description of the podcast
            creator: Creator/author name
            candidates: List of YouTube channel candidates with their metadata

        Returns:
            Best matching candidate or None if no good match
        """
        if not self.llm_client or not candidates:
            return None

        # Build candidate descriptions for the LLM
        candidate_text = ""
        for i, c in enumerate(candidates, 1):
            sub_count = c.get('subscriber_count', 0)
            sub_str = f"{sub_count:,}" if sub_count else "Unknown"
            candidate_text += f"""
Candidate {i}:
- Channel Name: {c.get('channel_name', 'Unknown')}
- Subscribers: {sub_str}
- Videos: {c.get('video_count', 'Unknown')}
- Description: {c.get('description', 'No description')[:300]}
"""

        prompt = f"""You are helping match a podcast to its official YouTube channel.

PODCAST INFORMATION:
- Name: {podcast_name}
- Creator: {creator or 'Unknown'}
- Description: {podcast_description[:500] if podcast_description else 'No description'}

YOUTUBE CHANNEL CANDIDATES:
{candidate_text}

TASK: Determine which YouTube channel (if any) is the MAIN official channel for this podcast.

STRICT REJECTION RULES - ALWAYS respond "NONE" if:
1. Channel name contains "Shorts", "Highlights", "Clips", "Topic", "Best of", or "Moments" - these are NEVER the main channel
2. Channel name is very different from the podcast name (e.g., "Enterprise Pivot" for podcast "Pivot")
3. Channel appears to be a fan channel, compilation channel, or unrelated channel
4. Channel has suspiciously few subscribers compared to the podcast's popularity

SELECTION RULES - Only select a channel if:
1. Channel name closely matches the podcast name OR the host/creator's name
2. Channel has a substantial subscriber count (typically 10,000+ for popular podcasts)
3. Channel description mentions the podcast or its content
4. You are CONFIDENT this is the official main channel where full episodes are posted

When in doubt, respond "NONE". It's better to have no match than a wrong match.

Respond with ONLY the candidate number (1, 2, 3, etc.) or "NONE". No explanation needed."""

        try:
            self.stats['llm_calls'] += 1
            response = await self.llm_client.call_simple(prompt, max_tokens=10, temperature=0.0)
            response = response.strip().upper()

            if response == "NONE":
                return None

            # Parse the candidate number
            match = re.search(r'(\d+)', response)
            if match:
                idx = int(match.group(1)) - 1
                if 0 <= idx < len(candidates):
                    selected = candidates[idx]
                    selected['match_method'] = 'llm_selected'
                    selected['match_confidence'] = 0.95  # High confidence for LLM selection
                    return selected

        except Exception as e:
            self.logger.warning(f"LLM channel selection failed: {e}")

        return None

    async def search_youtube_channel(self, podcast_name: str, creator: str = None, description: str = None) -> Optional[Dict]:
        """
        Search for YouTube channel using multiple strategies:

        1. Check if we already have an indexed channel matching this podcast (free, high confidence)
        2. Check podcast description for explicit YouTube URLs (highest confidence)
        3. Search DuckDuckGo for candidates (free)
        4. Verify candidates with YouTube API (1 unit each)
        5. Filter out derivative channels (Shorts, Highlights, Topic, etc.)
        6. Use LLM to intelligently select the best match

        Returns match info or None if no confident match found.
        """
        # FIRST: Check if we already have this channel indexed in our database
        indexed_match = self._check_indexed_channels(podcast_name, creator)
        if indexed_match and indexed_match['platform'] == 'youtube':
            self.logger.debug(f"Found indexed YouTube channel for '{podcast_name}'")
            return indexed_match

        raw_candidates = []

        # SECOND: Check podcast description for explicit YouTube URLs
        if description:
            extracted = self._extract_video_urls_from_text(description)
            for url in extracted.get('youtube', []):
                identifier = self._extract_youtube_channel_id(url)
                if identifier:
                    is_channel_id = identifier.startswith('UC')
                    raw_candidates.append((identifier, is_channel_id, 'from_description', url))
                    self.logger.debug(f"Found YouTube URL in description: {url}")

        # THIRD: Search DuckDuckGo for additional candidates
        normalized = self._normalize_podcast_name(podcast_name)

        # Extract potential creator name from podcast name (e.g., "On Purpose with Jay Shetty" -> "Jay Shetty")
        extracted_creator = None
        with_patterns = [
            r'(?:with|by|from|featuring)\s+(.+?)(?:\s*[-:|]|$)',
            r"^(.+?)(?:'s|s')\s+",  # "Someone's Podcast" -> "Someone"
        ]
        for pattern in with_patterns:
            match = re.search(pattern, podcast_name, re.IGNORECASE)
            if match:
                extracted_creator = match.group(1).strip()
                break

        search_queries = [
            f'"{podcast_name}" site:youtube.com/channel',   # Exact name, channel URLs only
            f'"{podcast_name}" youtube channel official',   # Exact name with channel keyword
            f'"{normalized}" site:youtube.com/channel',     # Normalized name
            f'"{podcast_name}" site:youtube.com/@',         # Handle URLs
        ]

        # Add creator-based searches
        if creator and creator.lower() != podcast_name.lower():
            search_queries.append(f'"{podcast_name}" "{creator}" site:youtube.com')
            search_queries.append(f'"{creator}" youtube channel')
            search_queries.append(f'{creator.replace(" ", "")}Podcast youtube')  # "JayShettypodcast youtube"

        # Add extracted creator searches (e.g., "Jay Shetty" from "On Purpose with Jay Shetty")
        if extracted_creator and extracted_creator.lower() != podcast_name.lower():
            search_queries.append(f'{extracted_creator.replace(" ", "")}Podcast youtube')  # "JayShettyPodcast youtube"
            search_queries.append(f'"{extracted_creator}" podcast youtube channel')
            search_queries.append(f'"{extracted_creator}" youtube channel official')

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
                    raw_candidates.append((identifier, is_channel_id, title, url))

        if not raw_candidates:
            self.logger.debug(f"No YouTube candidates found via DuckDuckGo for '{podcast_name}'")
            return None

        # Deduplicate candidates
        seen = set()
        unique_candidates = []
        for c in raw_candidates:
            if c[0] not in seen:
                seen.add(c[0])
                unique_candidates.append(c)

        # Verify each candidate with YouTube API and collect full metadata
        verified_candidates = []

        for identifier, is_channel_id, source_title, source_url in unique_candidates[:8]:
            try:
                if is_channel_id:
                    request = self.youtube.channels().list(
                        part="snippet,statistics",
                        id=identifier
                    )
                else:
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

                # Track if this came from podcast description (high confidence source)
                from_description = source_title == 'from_description'

                verified_candidates.append({
                    'platform': 'youtube',
                    'channel_id': channel_id,
                    'channel_name': channel_title,
                    'channel_url': f'https://www.youtube.com/channel/{channel_id}',
                    'description': channel_description[:500],
                    'subscriber_count': int(stats.get('subscriberCount', 0)),
                    'video_count': int(stats.get('videoCount', 0)),
                    'search_source': 'podcast_description' if from_description else 'duckduckgo',
                    'from_description': from_description,
                })

            except Exception as e:
                self.logger.debug(f"Error verifying YouTube channel {identifier}: {e}")
                continue

        if not verified_candidates:
            return None

        # If we found a URL directly in the podcast description, use it with high confidence
        # (but still filter out derivative channels)
        description_matches = [c for c in verified_candidates if c.get('from_description')]
        if description_matches:
            best = description_matches[0]
            # Only use if it's not a derivative channel
            if not self._is_derivative_channel(best['channel_name']):
                best['match_method'] = 'description_url'
                best['match_confidence'] = 0.99  # Very high confidence - explicitly mentioned
                self.logger.debug(f"Using YouTube URL found in podcast description: {best['channel_url']}")
                return best
            else:
                self.logger.debug(f"Skipping derivative channel from description: {best['channel_name']}")

        # Filter out derivative channels (Shorts, Highlights, Topic, etc.) before LLM selection
        main_candidates = [
            c for c in verified_candidates
            if not self._is_derivative_channel(c['channel_name'])
        ]

        # Log what we filtered
        filtered_count = len(verified_candidates) - len(main_candidates)
        if filtered_count > 0:
            filtered_names = [c['channel_name'] for c in verified_candidates if self._is_derivative_channel(c['channel_name'])]
            self.logger.debug(f"Filtered out {filtered_count} derivative channels: {filtered_names}")

        # Sort by subscriber count (prefer larger channels)
        main_candidates.sort(key=lambda x: x.get('subscriber_count', 0), reverse=True)

        if not main_candidates:
            self.logger.debug(f"No non-derivative YouTube channels found for '{podcast_name}'")
            return None

        # Use LLM to select the best channel if enabled
        if self.use_llm and self.llm_client:
            llm_result = await self._llm_select_best_channel(
                podcast_name=podcast_name,
                podcast_description=description or '',
                creator=creator or '',
                candidates=main_candidates
            )
            if llm_result:
                return llm_result

        # Fallback: Use heuristic matching if LLM is disabled or fails
        best_match = None
        best_confidence = 0.0

        for candidate in main_candidates:
            channel_title = candidate['channel_name']
            channel_description = candidate.get('description', '')

            name_similarity = self._similarity_ratio(podcast_name, channel_title)

            yt_name_lower = channel_title.lower().strip()
            podcast_name_lower = podcast_name.lower().strip()
            is_substring = (
                yt_name_lower in podcast_name_lower or
                podcast_name_lower in yt_name_lower
            )

            if name_similarity >= self.NAME_EXACT_THRESHOLD:
                confidence = 0.95
                method = 'name_exact'
            elif name_similarity >= self.NAME_FUZZY_THRESHOLD:
                confidence = 0.85
                method = 'name_fuzzy'
            elif is_substring and len(yt_name_lower) > 10:
                confidence = 0.90
                method = 'name_substring'
            else:
                if podcast_name.lower() in channel_description.lower():
                    confidence = 0.75
                    method = 'description_match'
                else:
                    continue

            if creator:
                creator_in_title = creator.lower() in channel_title.lower()
                creator_in_desc = creator.lower() in channel_description.lower()
                if creator_in_title or creator_in_desc:
                    confidence = min(confidence + 0.10, 0.98)

            if confidence > best_confidence:
                best_confidence = confidence
                best_match = {
                    **candidate,
                    'match_confidence': round(confidence, 2),
                    'match_method': method,
                }

        return best_match

    def _extract_rumble_channel_slug(self, url: str) -> Optional[str]:
        """Extract Rumble channel slug from URL."""
        patterns = [
            r'rumble\.com/c/([a-zA-Z0-9_-]+)',  # /c/channelname
            r'rumble\.com/user/([a-zA-Z0-9_-]+)',  # /user/username
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

    async def search_rumble_channel(self, podcast_name: str, creator: str = None, description: str = None) -> Optional[Dict]:
        """
        Search Rumble for a channel matching the podcast name.

        Strategy:
        1. Check if we already have an indexed Rumble channel
        2. Check podcast description for explicit Rumble URLs
        3. Search DuckDuckGo for Rumble channel candidates
        4. Try direct URL guessing as fallback
        5. Verify channels with yt-dlp (if possible)

        Returns match info or None if no confident match found.
        """
        # FIRST: Check if we already have this channel indexed in our database
        indexed_match = self._check_indexed_channels(podcast_name, creator)
        if indexed_match and indexed_match['platform'] == 'rumble':
            self.logger.debug(f"Found indexed Rumble channel for '{podcast_name}'")
            return indexed_match

        raw_candidates = []

        # SECOND: Check podcast description for explicit Rumble URLs
        if description:
            extracted = self._extract_video_urls_from_text(description)
            for url in extracted.get('rumble', []):
                slug = self._extract_rumble_channel_slug(url)
                if slug:
                    raw_candidates.append({
                        'slug': slug,
                        'url': url,
                        'source': 'podcast_description',
                    })
                    self.logger.debug(f"Found Rumble URL in description: {url}")

        # THIRD: Search DuckDuckGo for Rumble channel pages
        normalized = self._normalize_podcast_name(podcast_name)
        search_queries = [
            f'"{podcast_name}" site:rumble.com/c/',
            f'"{normalized}" site:rumble.com/c/',
            f'"{podcast_name}" rumble channel',
        ]
        if creator and creator.lower() != podcast_name.lower():
            search_queries.append(f'"{creator}" site:rumble.com/c/')

        for query in search_queries:
            results = self._search_duckduckgo(query, max_results=5)

            for result in results:
                url = result.get('href', '')
                if 'rumble.com' not in url:
                    continue

                slug = self._extract_rumble_channel_slug(url)
                if slug:
                    raw_candidates.append({
                        'slug': slug,
                        'url': url,
                        'source': 'duckduckgo',
                    })

        # FOURTH: Try direct URL guessing based on podcast name
        guesses = [
            normalized.replace(' ', ''),  # "JoeRogan"
            normalized.replace(' ', '').lower(),  # "joerogan"
            normalized.replace(' ', '-').lower(),  # "joe-rogan"
            normalized.replace(' ', '_').lower(),  # "joe_rogan"
        ]
        if creator:
            creator_norm = self._normalize_podcast_name(creator)
            guesses.extend([
                creator_norm.replace(' ', ''),
                creator_norm.replace(' ', '').lower(),
            ])

        for guess in guesses:
            if guess:
                raw_candidates.append({
                    'slug': guess,
                    'url': f'https://rumble.com/c/{guess}',
                    'source': 'url_guess',
                })

        # Deduplicate by slug
        seen_slugs = set()
        unique_candidates = []
        for c in raw_candidates:
            slug = c['slug'].lower()
            if slug not in seen_slugs:
                seen_slugs.add(slug)
                unique_candidates.append(c)

        if not unique_candidates:
            return None

        # Verify candidates with yt-dlp
        best_match = None
        best_confidence = 0.0

        for candidate in unique_candidates[:10]:  # Limit to prevent too many requests
            slug = candidate['slug']
            channel_url = f"https://rumble.com/c/{slug}"

            try:
                with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                    channel_info = ydl.extract_info(channel_url, download=False)

                    if channel_info and channel_info.get('title'):
                        channel_title = channel_info['title']
                        name_similarity = self._similarity_ratio(podcast_name, channel_title)

                        # Also check creator match
                        creator_similarity = 0.0
                        if creator:
                            creator_similarity = self._similarity_ratio(creator, channel_title)

                        best_sim = max(name_similarity, creator_similarity)

                        # Assign confidence based on source and similarity
                        if candidate['source'] == 'podcast_description':
                            confidence = 0.95  # High confidence - explicitly mentioned
                            method = 'description_url'
                        elif best_sim >= self.NAME_EXACT_THRESHOLD:
                            confidence = 0.90
                            method = 'name_exact'
                        elif best_sim >= self.NAME_FUZZY_THRESHOLD:
                            confidence = 0.80
                            method = 'name_fuzzy'
                        else:
                            continue  # Skip low similarity matches

                        if confidence > best_confidence:
                            best_confidence = confidence
                            best_match = {
                                'platform': 'rumble',
                                'channel_id': slug,
                                'channel_name': channel_title,
                                'channel_url': channel_url,
                                'match_confidence': round(confidence, 2),
                                'match_method': method,
                                'video_count': len(channel_info.get('entries', [])),
                            }

                            # If we found it from description, use it immediately
                            if candidate['source'] == 'podcast_description':
                                self.logger.debug(f"Using Rumble URL from podcast description: {channel_url}")
                                return best_match

            except Exception as e:
                self.logger.debug(f"Rumble verification error for '{slug}': {e}")
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
        podcast_description = channel.description or ''

        self.logger.info(f"Auditing: {podcast_name}")

        result = {
            'channel_id': channel.id,
            'channel_name': podcast_name,
            'youtube': None,
            'rumble': None,
        }

        # Try YouTube first (primary)
        youtube_match = await self.search_youtube_channel(podcast_name, creator, podcast_description)
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
        rumble_match = None
        if not self.skip_rumble:
            rumble_match = await self.search_rumble_channel(podcast_name, creator, podcast_description)
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

    def _match_episode_youtube_cached(
        self,
        episode: Content,
        youtube_channel_id: str
    ) -> Optional[Dict]:
        """
        Match episode to YouTube video using cached data (no API calls).

        This is much faster and doesn't consume API quota. It requires that the
        YouTube channel's videos have been previously indexed into the cache.

        Args:
            episode: Podcast Content record
            youtube_channel_id: YouTube channel ID (UC...)

        Returns:
            Match info or None
        """
        from src.database.models import YouTubeVideoCache

        episode_title = episode.title or ''
        episode_date = episode.publish_date

        with get_session() as session:
            # Query all cached videos for this channel
            candidates = session.query(YouTubeVideoCache).filter(
                YouTubeVideoCache.youtube_channel_id == youtube_channel_id
            ).all()

            if not candidates:
                self.logger.debug(f"No cached videos found for channel {youtube_channel_id}")
                return None

            best_match = None
            best_score = 0.0

            for video in candidates:
                video_title = video.title or ''

                # Calculate title similarity
                title_sim = self._similarity_ratio(episode_title, video_title)

                if title_sim < self.EPISODE_TITLE_THRESHOLD:
                    continue

                # Base score from title
                score = title_sim

                # Boost if dates are close
                if episode_date and video.publish_date:
                    try:
                        video_date = video.publish_date
                        # Handle timezone-aware/naive comparison
                        if video_date.tzinfo:
                            video_date = video_date.replace(tzinfo=None)
                        if episode_date.tzinfo:
                            episode_date_naive = episode_date.replace(tzinfo=None)
                        else:
                            episode_date_naive = episode_date

                        days_diff = abs((video_date - episode_date_naive).days)
                        if days_diff <= self.DATE_TOLERANCE_DAYS:
                            score += 0.15
                        elif days_diff <= 14:
                            score += 0.10
                        elif days_diff <= 30:
                            score += 0.05
                    except Exception:
                        pass

                # Boost for duration match if available
                if episode.duration and video.duration:
                    duration_diff = abs(episode.duration - video.duration)
                    tolerance = episode.duration * self.DURATION_TOLERANCE_PERCENT
                    if duration_diff <= tolerance:
                        score += 0.10

                if score > best_score:
                    best_score = score
                    best_match = {
                        'platform': 'youtube',
                        'video_id': video.video_id,
                        'video_url': f'https://www.youtube.com/watch?v={video.video_id}',
                        'video_title': video_title,
                        'match_confidence': round(score, 2),
                        'match_method': 'cached_title_date',
                        'matched_at': datetime.now(timezone.utc).isoformat(),
                    }

            if best_match and best_match['match_confidence'] >= self.EPISODE_TITLE_THRESHOLD:
                self.stats['episodes_matched'] += 1

                # Store in episode's meta_data
                db_episode = session.query(Content).filter(Content.id == episode.id).first()
                if db_episode:
                    meta = dict(db_episode.meta_data or {})
                    meta['video_link'] = best_match
                    db_episode.meta_data = meta
                    flag_modified(db_episode, 'meta_data')
                    session.commit()

                return best_match

        return None

    async def match_episodes_for_channel_cached(
        self,
        podcast_channel_id: int,
        episode_limit: int = 100,
        force: bool = False,
        min_match_ratio: float = 0.1
    ) -> Dict:
        """
        Match episodes to YouTube videos using cached data (no API calls).

        This method uses a validation-first approach:
        1. Gets the podcast channel and its matched YouTube channel
        2. First indexes only 1 page (50 videos) to validate the match
        3. Tries to match recent episodes against those videos
        4. If no matches found, marks YouTube channel as false positive and returns
        5. Only if matches are found, continues to index more videos

        Args:
            podcast_channel_id: Database ID of the podcast channel
            episode_limit: Maximum number of episodes to match
            force: Re-match even if episode already has a video_link
            min_match_ratio: Minimum ratio of matched episodes to consider valid (default 0.1 = 10%)

        Returns:
            Dict with status and match statistics
        """
        from src.ingestion.youtube_video_cache_indexer import YouTubeVideoCacheIndexer

        self.logger.info(f"Matching episodes for podcast channel {podcast_channel_id} using cache")

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

            # Check if already marked as false positive
            if youtube_info.get('validated') is False:
                return {
                    'status': 'skipped',
                    'reason': 'YouTube channel previously marked as false positive',
                    'podcast_channel_id': podcast_channel_id,
                    'youtube_channel_id': youtube_channel_id,
                }

            channel_name = channel.display_name

        self.logger.info(f"Using YouTube channel {youtube_channel_id} for podcast '{channel_name}'")

        indexer = YouTubeVideoCacheIndexer(logger=self.logger)
        cache_count = indexer.get_cached_video_count(youtube_channel_id)

        # Phase 1: Validation - index only 1 page (50 videos) first
        if cache_count == 0:
            self.logger.info(f"No cached videos found, fetching first page (50 videos) for validation")
            index_result = await indexer.index_channel_videos(youtube_channel_id, max_videos=50)
            if index_result.get('status') != 'success':
                return {
                    'status': 'error',
                    'error': f'Failed to index YouTube channel: {index_result.get("error")}',
                    'podcast_channel_id': podcast_channel_id,
                }
            cache_count = index_result.get('videos_indexed', 0)

        self.logger.info(f"Cache has {cache_count} videos for validation")

        # Get recent episodes to validate against (use smaller set for validation)
        validation_limit = min(20, episode_limit)
        with get_session() as session:
            episodes = session.query(Content).filter(
                Content.channel_id == podcast_channel_id,
                Content.platform == 'podcast'
            ).order_by(desc(Content.publish_date)).limit(validation_limit).all()

        if not episodes:
            return {
                'status': 'error',
                'error': 'No episodes found for this podcast channel',
                'podcast_channel_id': podcast_channel_id,
            }

        self.logger.info(f"Validating with {len(episodes)} recent episodes against {cache_count} cached videos")

        # Phase 2: Validation matching
        validation_matched = 0
        for episode in episodes:
            # Skip if already matched (unless force)
            if not force and episode.meta_data and episode.meta_data.get('video_link'):
                validation_matched += 1  # Count existing matches
                continue

            match = self._match_episode_youtube_cached(episode, youtube_channel_id)
            if match:
                validation_matched += 1

        match_ratio = validation_matched / len(episodes) if episodes else 0
        self.logger.info(f"Validation: {validation_matched}/{len(episodes)} episodes matched ({match_ratio:.1%})")

        # Phase 3: Check if this is a false positive
        if validation_matched == 0 or match_ratio < min_match_ratio:
            self.logger.warning(f"YouTube channel {youtube_channel_id} appears to be a FALSE POSITIVE for '{channel_name}'")
            self.logger.warning(f"Only {validation_matched}/{len(episodes)} episodes matched ({match_ratio:.1%})")

            # Mark as false positive in platform_metadata
            self._mark_youtube_channel_invalid(podcast_channel_id, youtube_channel_id, match_ratio)

            return {
                'status': 'false_positive',
                'podcast_channel_id': podcast_channel_id,
                'channel_name': channel_name,
                'youtube_channel_id': youtube_channel_id,
                'validation_episodes': len(episodes),
                'validation_matched': validation_matched,
                'match_ratio': round(match_ratio, 3),
                'reason': f'Only {match_ratio:.1%} of episodes matched - below {min_match_ratio:.1%} threshold',
            }

        # Phase 4: Valid match - index more videos if needed
        if cache_count < 500:
            self.logger.info(f"Valid match confirmed! Indexing full video history (up to 500)")
            await indexer.index_channel_videos(youtube_channel_id, max_videos=500, force=True)
            cache_count = indexer.get_cached_video_count(youtube_channel_id)

        # Mark as validated
        self._mark_youtube_channel_validated(podcast_channel_id, youtube_channel_id)

        # Phase 5: Full matching
        with get_session() as session:
            query = session.query(Content).filter(
                Content.channel_id == podcast_channel_id,
                Content.platform == 'podcast'
            ).order_by(desc(Content.publish_date))

            if not force:
                # Filter out episodes that already have a video_link
                # Use JSONB ? operator via SQLAlchemy's has_key
                from sqlalchemy import not_, text
                query = query.filter(
                    not_(Content.meta_data.op('?')('video_link'))
                )

            episodes = query.limit(episode_limit).all()

        self.logger.info(f"Full matching: {len(episodes)} episodes against {cache_count} cached videos")

        matched = 0
        skipped = 0
        failed = 0

        for episode in episodes:
            try:
                if not force and episode.meta_data and episode.meta_data.get('video_link'):
                    skipped += 1
                    continue

                match = self._match_episode_youtube_cached(episode, youtube_channel_id)
                if match:
                    matched += 1
                    self.logger.debug(f"Matched: {episode.title[:50]}... -> {match['video_title'][:50]}...")
                else:
                    failed += 1
                    self.logger.debug(f"No match found for: {episode.title[:50]}...")
            except Exception as e:
                self.logger.warning(f"Error matching episode {episode.id}: {e}")
                failed += 1

        self.logger.info(f"Matching complete: {matched} matched, {skipped} skipped, {failed} no match")

        return {
            'status': 'success',
            'podcast_channel_id': podcast_channel_id,
            'channel_name': channel_name,
            'youtube_channel_id': youtube_channel_id,
            'episodes_processed': len(episodes),
            'matched': matched,
            'skipped': skipped,
            'no_match': failed,
            'cache_video_count': cache_count,
            'validated': True,
        }

    def _mark_youtube_channel_invalid(
        self,
        podcast_channel_id: int,
        youtube_channel_id: str,
        match_ratio: float
    ) -> None:
        """Mark a YouTube channel match as a false positive."""
        with get_session() as session:
            channel = session.query(Channel).filter(Channel.id == podcast_channel_id).first()
            if channel:
                pm = dict(channel.platform_metadata or {})
                if 'video_links' not in pm:
                    pm['video_links'] = {}
                if 'youtube' not in pm['video_links']:
                    pm['video_links']['youtube'] = {}

                pm['video_links']['youtube']['validated'] = False
                pm['video_links']['youtube']['validation_date'] = datetime.now(timezone.utc).isoformat()
                pm['video_links']['youtube']['validation_match_ratio'] = round(match_ratio, 3)
                pm['video_links']['youtube']['validation_reason'] = 'false_positive_no_episode_matches'

                channel.platform_metadata = pm
                flag_modified(channel, 'platform_metadata')
                session.commit()
                self.logger.info(f"Marked YouTube channel {youtube_channel_id} as false positive for channel {podcast_channel_id}")

    def _mark_youtube_channel_validated(
        self,
        podcast_channel_id: int,
        youtube_channel_id: str
    ) -> None:
        """Mark a YouTube channel match as validated."""
        with get_session() as session:
            channel = session.query(Channel).filter(Channel.id == podcast_channel_id).first()
            if channel:
                pm = dict(channel.platform_metadata or {})
                if 'video_links' not in pm:
                    pm['video_links'] = {}
                if 'youtube' not in pm['video_links']:
                    pm['video_links']['youtube'] = {}

                pm['video_links']['youtube']['validated'] = True
                pm['video_links']['youtube']['validation_date'] = datetime.now(timezone.utc).isoformat()

                channel.platform_metadata = pm
                flag_modified(channel, 'platform_metadata')
                session.commit()
                self.logger.debug(f"Marked YouTube channel {youtube_channel_id} as validated for channel {podcast_channel_id}")

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
            'llm_calls': self.stats['llm_calls'],
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
        self.logger.info(f"LLM calls: {self.stats['llm_calls']}")

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

    # Subcommands for different operations
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Default audit command (backward compatible)
    audit_parser = subparsers.add_parser('audit', help='Audit podcast channels for YouTube/Rumble matches')
    audit_parser.add_argument(
        '--limit',
        type=int,
        default=100,
        help='Maximum channels to audit (default: 100)'
    )
    audit_parser.add_argument(
        '--min-importance',
        type=float,
        default=0.0,
        help='Minimum importance_score threshold (default: 0)'
    )
    audit_parser.add_argument(
        '--channel-id',
        type=int,
        help='Audit a specific channel by ID'
    )
    audit_parser.add_argument(
        '--match-episodes',
        action='store_true',
        help='Also match episodes to videos'
    )
    audit_parser.add_argument(
        '--episode-limit',
        type=int,
        default=10,
        help='Max episodes per channel to match (default: 10)'
    )
    audit_parser.add_argument(
        '--force',
        action='store_true',
        help='Re-audit even if recently audited'
    )
    audit_parser.add_argument(
        '--no-llm',
        action='store_true',
        help='Disable LLM-based channel selection (use heuristics only)'
    )
    audit_parser.add_argument(
        '--skip-rumble',
        action='store_true',
        help='Skip Rumble search (YouTube only) for faster auditing'
    )

    # Index videos command - cache YouTube videos for a channel
    index_parser = subparsers.add_parser('index-videos', help='Index YouTube videos into cache for episode matching')
    index_parser.add_argument(
        '--channel-id',
        type=int,
        required=True,
        help='Podcast channel ID to index YouTube videos for'
    )
    index_parser.add_argument(
        '--max-videos',
        type=int,
        default=500,
        help='Maximum videos to index (default: 500)'
    )
    index_parser.add_argument(
        '--force',
        action='store_true',
        help='Re-index even if recently indexed'
    )

    # Match episodes command - match episodes using cached videos
    match_parser = subparsers.add_parser('match-episodes', help='Match podcast episodes to cached YouTube videos')
    match_parser.add_argument(
        '--channel-id',
        type=int,
        required=True,
        help='Podcast channel ID to match episodes for'
    )
    match_parser.add_argument(
        '--limit',
        type=int,
        default=100,
        help='Maximum episodes to match (default: 100)'
    )
    match_parser.add_argument(
        '--force',
        action='store_true',
        help='Re-match even if episode already has video_link'
    )

    # Match all command - match ALL episodes for validated channels
    match_all_parser = subparsers.add_parser('match-all', help='Match ALL episodes for validated channels (bulk matching)')
    match_all_parser.add_argument(
        '--channel-id',
        type=int,
        help='Specific channel ID (if not set, processes all validated channels)'
    )
    match_all_parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum channels to process (default: all)'
    )
    match_all_parser.add_argument(
        '--force',
        action='store_true',
        help='Re-match even if episode already has video_link'
    )

    # Clear match command - remove YouTube match from a podcast channel
    clear_parser = subparsers.add_parser('clear-match', help='Clear YouTube match from a podcast channel (for false positives)')
    clear_parser.add_argument(
        '--channel-id',
        type=int,
        required=True,
        help='Podcast channel ID to clear YouTube match from'
    )
    clear_parser.add_argument(
        '--delete-cache',
        action='store_true',
        help='Also delete cached videos for this YouTube channel'
    )

    # Stats command - show cache and validation statistics
    stats_parser = subparsers.add_parser('stats', help='Show cache and validation statistics')
    stats_parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed per-channel statistics'
    )

    # Index and match command - do both in one go
    both_parser = subparsers.add_parser('index-and-match', help='Index videos and match episodes in one command')
    both_parser.add_argument(
        '--channel-id',
        type=int,
        help='Specific podcast channel ID (if not set, processes channels with YouTube matches)'
    )
    both_parser.add_argument(
        '--limit',
        type=int,
        default=10,
        help='Maximum channels to process (default: 10)'
    )
    both_parser.add_argument(
        '--episode-limit',
        type=int,
        default=50,
        help='Maximum episodes per channel to match (default: 50)'
    )
    both_parser.add_argument(
        '--max-videos',
        type=int,
        default=500,
        help='Maximum videos to index per channel (default: 500)'
    )
    both_parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-indexing and re-matching'
    )

    # Also support old-style arguments for backward compatibility
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
    parser.add_argument(
        '--no-llm',
        action='store_true',
        help='Disable LLM-based channel selection (use heuristics only)'
    )
    parser.add_argument(
        '--skip-rumble',
        action='store_true',
        help='Skip Rumble search (YouTube only) for faster auditing'
    )

    args = parser.parse_args()

    # Use CLI logger for console output
    cli_logger = setup_cli_logger('podcast_video_matcher')

    # Handle subcommands
    if args.command == 'index-videos':
        from src.ingestion.youtube_video_cache_indexer import YouTubeVideoCacheIndexer
        indexer = YouTubeVideoCacheIndexer(logger=cli_logger)
        result = await indexer.index_for_podcast_channel(
            podcast_channel_id=args.channel_id,
            max_videos=args.max_videos,
            force=args.force
        )
        cli_logger.info(f"\nResult: {result}")
        return result

    elif args.command == 'match-episodes':
        matcher = PodcastVideoMatcher(logger=cli_logger, use_llm=False, skip_rumble=True)
        result = await matcher.match_episodes_for_channel_cached(
            podcast_channel_id=args.channel_id,
            episode_limit=args.limit,
            force=args.force
        )
        cli_logger.info(f"\nResult: {result}")
        return result

    elif args.command == 'match-all':
        # Match ALL episodes for validated channels (direct matching, skips validation)
        from src.database.models import YouTubeVideoCache

        cli_logger.info("=" * 60)
        cli_logger.info("BULK EPISODE MATCHING - All Validated Channels")
        cli_logger.info("=" * 60)

        # Get validated channels
        with get_session() as session:
            query = session.query(Channel).filter(
                Channel.platform == 'podcast',
                Channel.platform_metadata['video_links']['youtube']['validated'].astext == 'true'
            ).order_by(desc(Channel.importance_score))

            if args.channel_id:
                query = query.filter(Channel.id == args.channel_id)
            elif args.limit:
                query = query.limit(args.limit)

            channels = query.all()
            channel_data = [
                {
                    'id': c.id,
                    'name': c.display_name,
                    'youtube_channel_id': c.platform_metadata.get('video_links', {}).get('youtube', {}).get('channel_id'),
                }
                for c in channels
            ]

        cli_logger.info(f"Found {len(channel_data)} validated channels to process")

        matcher = PodcastVideoMatcher(logger=cli_logger, use_llm=False, skip_rumble=True)
        total_matched = 0
        total_processed = 0
        total_skipped = 0
        results = []

        for i, ch in enumerate(channel_data):
            cli_logger.info(f"\n[{i+1}/{len(channel_data)}] {ch['name']}")

            youtube_channel_id = ch['youtube_channel_id']
            if not youtube_channel_id:
                cli_logger.warning(f"  No YouTube channel ID found, skipping")
                continue

            # Get episodes to match (skip already matched unless force)
            with get_session() as session:
                query = session.query(Content).filter(
                    Content.channel_id == ch['id'],
                    Content.platform == 'podcast'
                )

                total_episodes = query.count()

                if not args.force:
                    query = query.filter(
                        not_(Content.meta_data.op('?')('video_link'))
                    )

                episodes = query.order_by(desc(Content.publish_date)).all()
                already_matched = total_episodes - len(episodes)

            cli_logger.info(f"  Episodes: {total_episodes} total, {already_matched} already matched, {len(episodes)} to process")

            if not episodes:
                cli_logger.info(f"  Skipping - no unmatched episodes")
                continue

            # Check cache has videos
            cache_count = session.query(YouTubeVideoCache).filter(
                YouTubeVideoCache.youtube_channel_id == youtube_channel_id
            ).count()

            if cache_count == 0:
                cli_logger.warning(f"  No cached videos for {youtube_channel_id}, skipping")
                continue

            cli_logger.info(f"  Matching against {cache_count} cached videos...")

            # Direct matching (no validation, already validated)
            matched = 0
            skipped = 0
            no_match = 0

            for episode in episodes:
                try:
                    match = matcher._match_episode_youtube_cached(episode, youtube_channel_id)
                    if match:
                        matched += 1
                    else:
                        no_match += 1
                except Exception as e:
                    cli_logger.debug(f"  Error matching episode {episode.id}: {e}")
                    no_match += 1

            total_matched += matched
            total_processed += len(episodes)
            total_skipped += skipped

            cli_logger.info(f"  Result: {matched} matched, {no_match} no match")
            results.append({
                'channel_id': ch['id'],
                'channel_name': ch['name'],
                'episodes_processed': len(episodes),
                'matched': matched,
                'no_match': no_match
            })

        cli_logger.info(f"\n{'='*60}")
        cli_logger.info("SUMMARY")
        cli_logger.info('='*60)
        cli_logger.info(f"Channels processed: {len(channel_data)}")
        cli_logger.info(f"Total episodes processed: {total_processed}")
        cli_logger.info(f"Total new matches: {total_matched}")

        return {
            'status': 'success',
            'channels_processed': len(channel_data),
            'episodes_processed': total_processed,
            'episodes_matched': total_matched,
            'results': results
        }

    elif args.command == 'clear-match':
        # Clear YouTube match from a podcast channel
        from src.database.models import YouTubeVideoCache

        with get_session() as session:
            channel = session.query(Channel).filter(
                Channel.id == args.channel_id,
                Channel.platform == 'podcast'
            ).first()

            if not channel:
                cli_logger.error(f"Podcast channel not found: {args.channel_id}")
                return {'status': 'error', 'error': 'Channel not found'}

            pm = channel.platform_metadata or {}
            video_links = pm.get('video_links', {})
            youtube_info = video_links.get('youtube', {})
            youtube_channel_id = youtube_info.get('channel_id')

            if not youtube_channel_id:
                cli_logger.info(f"Channel {args.channel_id} ({channel.display_name}) has no YouTube match to clear")
                return {'status': 'no_match', 'channel_id': args.channel_id}

            cli_logger.info(f"Clearing YouTube match for '{channel.display_name}'")
            cli_logger.info(f"  YouTube channel: {youtube_info.get('channel_name')} ({youtube_channel_id})")

            # Remove the youtube section from video_links
            if 'youtube' in video_links:
                del video_links['youtube']
                pm['video_links'] = video_links
                channel.platform_metadata = pm
                flag_modified(channel, 'platform_metadata')

            # Optionally delete cached videos
            deleted_cache = 0
            if args.delete_cache and youtube_channel_id:
                deleted_cache = session.query(YouTubeVideoCache).filter(
                    YouTubeVideoCache.youtube_channel_id == youtube_channel_id
                ).delete()
                cli_logger.info(f"  Deleted {deleted_cache} cached videos")

            session.commit()

        cli_logger.info(f"YouTube match cleared for channel {args.channel_id}")
        return {
            'status': 'success',
            'channel_id': args.channel_id,
            'youtube_channel_id': youtube_channel_id,
            'cache_deleted': deleted_cache
        }

    elif args.command == 'stats':
        # Show cache and validation statistics
        from src.database.models import YouTubeVideoCache

        with get_session() as session:
            # Get validation stats
            total_with_youtube = session.query(Channel).filter(
                Channel.platform == 'podcast',
                Channel.platform_metadata['video_links']['youtube']['channel_id'].isnot(None)
            ).count()

            validated_true = session.query(Channel).filter(
                Channel.platform == 'podcast',
                Channel.platform_metadata['video_links']['youtube']['validated'].astext == 'true'
            ).count()

            validated_false = session.query(Channel).filter(
                Channel.platform == 'podcast',
                Channel.platform_metadata['video_links']['youtube']['validated'].astext == 'false'
            ).count()

            not_validated = total_with_youtube - validated_true - validated_false

            # Get cache stats
            from sqlalchemy import func
            cache_stats = session.query(
                func.count(YouTubeVideoCache.id).label('total_videos'),
                func.count(func.distinct(YouTubeVideoCache.youtube_channel_id)).label('channels_cached')
            ).first()

            # Get episode match stats
            episodes_with_match = session.query(Content).filter(
                Content.platform == 'podcast',
                Content.meta_data.op('?')('video_link')
            ).count()

            total_podcast_episodes = session.query(Content).filter(
                Content.platform == 'podcast'
            ).count()

        cli_logger.info("=" * 60)
        cli_logger.info("YOUTUBE VIDEO CACHE STATISTICS")
        cli_logger.info("=" * 60)
        cli_logger.info("")
        cli_logger.info("Channel Validation Status:")
        cli_logger.info(f"  Total with YouTube match:  {total_with_youtube:,}")
        cli_logger.info(f"  Validated (confirmed):     {validated_true:,}")
        cli_logger.info(f"  False positives:           {validated_false:,}")
        cli_logger.info(f"  Not yet validated:         {not_validated:,}")
        cli_logger.info("")
        cli_logger.info("Video Cache:")
        cli_logger.info(f"  Total cached videos:       {cache_stats.total_videos:,}")
        cli_logger.info(f"  Channels in cache:         {cache_stats.channels_cached:,}")
        cli_logger.info("")
        cli_logger.info("Episode Matching:")
        cli_logger.info(f"  Episodes with video link:  {episodes_with_match:,}")
        cli_logger.info(f"  Total podcast episodes:    {total_podcast_episodes:,}")
        if total_podcast_episodes > 0:
            cli_logger.info(f"  Match rate:                {episodes_with_match/total_podcast_episodes*100:.2f}%")

        if args.verbose:
            cli_logger.info("")
            cli_logger.info("Per-Channel Cache Details:")
            cli_logger.info("-" * 60)

            cache_by_channel = session.query(
                YouTubeVideoCache.youtube_channel_id,
                func.count(YouTubeVideoCache.id).label('video_count'),
                func.max(YouTubeVideoCache.indexed_at).label('last_indexed')
            ).group_by(YouTubeVideoCache.youtube_channel_id).order_by(
                func.count(YouTubeVideoCache.id).desc()
            ).limit(20).all()

            for row in cache_by_channel:
                cli_logger.info(f"  {row.youtube_channel_id}: {row.video_count} videos")

        return {
            'total_with_youtube': total_with_youtube,
            'validated_true': validated_true,
            'validated_false': validated_false,
            'not_validated': not_validated,
            'cached_videos': cache_stats.total_videos,
            'channels_cached': cache_stats.channels_cached,
            'episodes_matched': episodes_with_match,
        }

    elif args.command == 'index-and-match':
        from src.ingestion.youtube_video_cache_indexer import YouTubeVideoCacheIndexer

        if args.channel_id:
            # Process single channel
            channel_ids = [args.channel_id]
        else:
            # Get channels that have YouTube matches
            with get_session() as session:
                channels = session.query(Channel).filter(
                    Channel.platform == 'podcast',
                    Channel.status == 'active',
                    Channel.platform_metadata['video_links']['youtube']['channel_id'].isnot(None)
                ).order_by(desc(Channel.importance_score)).limit(args.limit).all()
                channel_ids = [c.id for c in channels]
                cli_logger.info(f"Found {len(channel_ids)} channels with YouTube matches")

        results = []
        for channel_id in channel_ids:
            cli_logger.info(f"\n{'='*60}")
            cli_logger.info(f"Processing channel {channel_id}")
            cli_logger.info('='*60)

            # Index videos
            indexer = YouTubeVideoCacheIndexer(logger=cli_logger)
            index_result = await indexer.index_for_podcast_channel(
                podcast_channel_id=channel_id,
                max_videos=args.max_videos,
                force=args.force
            )

            if index_result.get('status') == 'error':
                cli_logger.warning(f"Index failed: {index_result.get('error')}")
                results.append({'channel_id': channel_id, 'index': index_result, 'match': None})
                continue

            # Match episodes
            matcher = PodcastVideoMatcher(logger=cli_logger, use_llm=False, skip_rumble=True)
            match_result = await matcher.match_episodes_for_channel_cached(
                podcast_channel_id=channel_id,
                episode_limit=args.episode_limit,
                force=args.force
            )

            results.append({
                'channel_id': channel_id,
                'index': index_result,
                'match': match_result
            })

        # Summary
        cli_logger.info(f"\n{'='*60}")
        cli_logger.info("SUMMARY")
        cli_logger.info('='*60)
        total_indexed = sum(r['index'].get('videos_indexed', 0) for r in results if r['index'])
        total_matched = sum(r['match'].get('matched', 0) for r in results if r['match'])
        cli_logger.info(f"Channels processed: {len(results)}")
        cli_logger.info(f"Total videos indexed: {total_indexed}")
        cli_logger.info(f"Total episodes matched: {total_matched}")

        return results

    else:
        # Default: run audit (backward compatible)
        matcher = PodcastVideoMatcher(
            logger=cli_logger,
            use_llm=not getattr(args, 'no_llm', False),
            skip_rumble=getattr(args, 'skip_rumble', False)
        )

        # If force flag, we need to modify the audit to not skip
        if getattr(args, 'force', False):
            original_method = matcher.audit_channel
            async def force_audit(channel, skip_if_audited=True):
                return await original_method(channel, skip_if_audited=False)
            matcher.audit_channel = force_audit

        summary = await matcher.run_audit(
            limit=getattr(args, 'limit', 100),
            min_importance=getattr(args, 'min_importance', 0.0),
            channel_id=getattr(args, 'channel_id', None),
            match_episodes=getattr(args, 'match_episodes', False),
            episode_limit=getattr(args, 'episode_limit', 10),
        )

        return summary


if __name__ == '__main__':
    asyncio.run(main())
