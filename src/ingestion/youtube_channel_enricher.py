#!/usr/bin/env python3
"""
YouTube Channel Enricher

Enriches YouTube channels with metadata from YouTube Data API v3.
Similar to podcast_enricher.py but for YouTube channels.

Features:
- Fetch channel metadata (subscribers, views, video count, etc.)
- Store in unified channels table
- Track enrichment timestamps
- Support bulk enrichment
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from src.utils.paths import get_project_root, get_config_path
from typing import Dict, Optional, List
from googleapiclient.discovery import build
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(get_project_root()))

from src.database.session import get_session
from src.database.models import Channel
from src.utils.logger import setup_worker_logger
from src.utils.project_utils import normalize_language_code

logger = setup_worker_logger('youtube_channel_enricher')

# Load environment variables
load_dotenv()


class YouTubeChannelEnricher:
    """Enrich YouTube channels with metadata from YouTube Data API"""

    def __init__(self):
        """Initialize YouTube channel enricher"""
        # Load API keys
        api_keys_str = os.getenv('YOUTUBE_API_KEYS')
        if not api_keys_str:
            raise ValueError("No YouTube API keys found in environment")

        self.api_keys = [k.strip() for k in api_keys_str.split(',') if k.strip()]
        if not self.api_keys:
            raise ValueError("No valid YouTube API keys found")

        self.current_key_index = 0
        self.youtube = build('youtube', 'v3', developerKey=self.api_keys[self.current_key_index])

        logger.info(f"Initialized YouTubeChannelEnricher with {len(self.api_keys)} API keys")

    def _rotate_api_key(self):
        """Rotate to the next API key"""
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self.youtube = build('youtube', 'v3', developerKey=self.api_keys[self.current_key_index])
        logger.info(f"Rotated to API key index {self.current_key_index}")

    def _extract_channel_id(self, channel_url: str) -> Optional[str]:
        """
        Extract channel ID from various YouTube URL formats.

        Supported formats:
        - youtube.com/channel/UCxxxxx
        - youtube.com/@handle
        - youtube.com/c/customurl
        - youtube.com/user/username

        Returns:
            Channel ID (UCxxxxx) or None if can't extract
        """
        import re
        from urllib.parse import unquote

        # Decode URL
        decoded_url = unquote(unquote(channel_url))

        # Try to match channel ID directly
        patterns = [
            r'youtube\.com/channel/([^/?&]+)',  # Direct channel ID
            r'youtube\.com/@([^/?&]+)',         # Handle
            r'youtube\.com/c/([^/?&]+)',        # Custom URL
            r'youtube\.com/user/([^/?&]+)'      # Legacy username
        ]

        for pattern in patterns:
            match = re.search(pattern, decoded_url)
            if match:
                identifier = match.group(1)

                # If it's a direct channel ID (UC...), return it
                if identifier.startswith('UC'):
                    return identifier

                # For handles/usernames, need to resolve via API
                return self._resolve_channel_id(identifier)

        logger.warning(f"Could not extract channel ID from URL: {channel_url}")
        return None

    def _resolve_channel_id(self, handle_or_username: str) -> Optional[str]:
        """
        Resolve a handle/username to channel ID via YouTube API.

        This is expensive (100 quota units) so use sparingly.
        """
        try:
            # Try search API to resolve handle/username
            request = self.youtube.search().list(
                part="snippet",
                q=handle_or_username,
                type="channel",
                maxResults=1
            )
            response = request.execute()

            if response.get('items'):
                channel_id = response['items'][0]['snippet']['channelId']
                logger.info(f"Resolved '{handle_or_username}' to channel ID: {channel_id}")
                return channel_id

            logger.warning(f"Could not resolve '{handle_or_username}' to channel ID")
            return None

        except Exception as e:
            logger.error(f"Error resolving channel ID for '{handle_or_username}': {e}")
            return None

    def fetch_channel_metadata(self, channel_id: str, max_retries: int = 3) -> Optional[Dict]:
        """
        Fetch channel metadata from YouTube Data API.

        Args:
            channel_id: YouTube channel ID (UCxxxxx)
            max_retries: Number of retry attempts

        Returns:
            Dictionary with channel metadata or None if failed
        """
        for attempt in range(max_retries):
            try:
                # Rate limiting - be respectful
                time.sleep(1.0)

                # Fetch channel details
                request = self.youtube.channels().list(
                    part="snippet,contentDetails,statistics,brandingSettings,topicDetails",
                    id=channel_id
                )
                response = request.execute()

                if not response.get('items'):
                    logger.warning(f"Channel not found: {channel_id}")
                    return None

                channel = response['items'][0]
                snippet = channel.get('snippet', {})
                statistics = channel.get('statistics', {})
                branding = channel.get('brandingSettings', {}).get('channel', {})
                topics = channel.get('topicDetails', {})

                metadata = {
                    'channel_id': channel_id,
                    'title': snippet.get('title', ''),
                    'description': snippet.get('description', ''),
                    'custom_url': snippet.get('customUrl', ''),
                    'published_at': snippet.get('publishedAt', ''),
                    'country': snippet.get('country', ''),
                    'thumbnail_url': snippet.get('thumbnails', {}).get('high', {}).get('url', ''),
                    'subscriber_count': int(statistics.get('subscriberCount', 0)),
                    'video_count': int(statistics.get('videoCount', 0)),
                    'view_count': int(statistics.get('viewCount', 0)),
                    'keywords': branding.get('keywords', ''),
                    'topic_categories': topics.get('topicCategories', []),
                    'uploads_playlist_id': channel.get('contentDetails', {}).get('relatedPlaylists', {}).get('uploads', '')
                }

                logger.info(f"Fetched metadata for channel: {metadata['title']} ({metadata['subscriber_count']:,} subscribers)")
                return metadata

            except Exception as e:
                error_str = str(e).lower()
                if 'quota' in error_str and attempt < max_retries - 1:
                    logger.warning(f"Quota exceeded, rotating API key (attempt {attempt + 1})")
                    self._rotate_api_key()
                    time.sleep(2)
                    continue

                logger.error(f"Error fetching channel metadata (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)

        return None

    def enrich_channel(self, channel_url: str) -> Optional[Channel]:
        """
        Enrich a single YouTube channel and store in database.

        Args:
            channel_url: YouTube channel URL

        Returns:
            Channel object if successful, None otherwise
        """
        # Extract channel ID
        channel_id = self._extract_channel_id(channel_url)
        if not channel_id:
            logger.error(f"Could not extract channel ID from URL: {channel_url}")
            return None

        # Fetch metadata
        metadata = self.fetch_channel_metadata(channel_id)
        if not metadata:
            logger.error(f"Could not fetch metadata for channel: {channel_id}")
            return None

        # Infer language from country or description
        language = self._infer_language(metadata.get('country', ''), metadata.get('description', ''))

        # Store in database
        with get_session() as session:
            # Check if channel already exists
            channel = session.query(Channel).filter_by(
                channel_key=channel_id,
                platform='youtube'
            ).first()

            if channel:
                # Update existing channel
                logger.info(f"Updating existing channel: {metadata['title']}")
                channel.display_name = metadata['title']
                channel.description = metadata['description']
                channel.language = normalize_language_code(language) if language else channel.language
                channel.platform_metadata = {
                    'channel_id': channel_id,
                    'subscriber_count': metadata['subscriber_count'],
                    'video_count': metadata['video_count'],
                    'view_count': metadata['view_count'],
                    'country': metadata['country'],
                    'published_at': metadata['published_at'],
                    'custom_url': metadata['custom_url'],
                    'keywords': metadata['keywords'],
                    'topic_categories': metadata['topic_categories'],
                    'thumbnail_url': metadata['thumbnail_url'],
                    'uploads_playlist_id': metadata['uploads_playlist_id'],
                    'last_enriched': datetime.utcnow().isoformat()
                }
                channel.updated_at = datetime.utcnow()
            else:
                # Create new channel
                logger.info(f"Creating new channel: {metadata['title']}")
                channel = Channel(
                    channel_key=channel_id,
                    display_name=metadata['title'],
                    platform='youtube',
                    primary_url=f"https://www.youtube.com/channel/{channel_id}",
                    language=language,
                    description=metadata['description'],
                    status='discovered',  # Will be set to 'active' when added to projects
                    platform_metadata={
                        'channel_id': channel_id,
                        'subscriber_count': metadata['subscriber_count'],
                        'video_count': metadata['video_count'],
                        'view_count': metadata['view_count'],
                        'country': metadata['country'],
                        'published_at': metadata['published_at'],
                        'custom_url': metadata['custom_url'],
                        'keywords': metadata['keywords'],
                        'topic_categories': metadata['topic_categories'],
                        'thumbnail_url': metadata['thumbnail_url'],
                        'uploads_playlist_id': metadata['uploads_playlist_id'],
                        'last_enriched': datetime.utcnow().isoformat()
                    },
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                session.add(channel)

            session.commit()
            logger.info(f"✓ Enriched channel: {channel.display_name} (ID: {channel.id})")
            return channel

    def _infer_language(self, country: str, description: str) -> str:
        """
        Infer language from country code or description.

        Simple heuristic - can be improved with language detection libraries.
        """
        # Country code to language mapping (simplified)
        country_to_lang = {
            'US': 'en', 'GB': 'en', 'CA': 'en', 'AU': 'en', 'NZ': 'en', 'IE': 'en',
            'FR': 'fr', 'BE': 'fr', 'CH': 'fr',
            'DE': 'de', 'AT': 'de',
            'ES': 'es', 'MX': 'es', 'AR': 'es', 'CO': 'es',
            'IT': 'it',
            'PT': 'pt', 'BR': 'pt',
            'RO': 'ro',
            'PL': 'pl',
            'NL': 'nl',
            'SE': 'sv',
            'NO': 'no',
            'DK': 'da',
            'FI': 'fi'
        }

        if country in country_to_lang:
            return country_to_lang[country]

        # Default to English
        return 'en'

    def enrich_channels_bulk(self, channel_urls: List[str]) -> Dict[str, int]:
        """
        Enrich multiple YouTube channels.

        Args:
            channel_urls: List of YouTube channel URLs

        Returns:
            Statistics dictionary
        """
        stats = {
            'total': len(channel_urls),
            'successful': 0,
            'failed': 0,
            'errors': []
        }

        logger.info(f"Starting bulk enrichment of {len(channel_urls)} YouTube channels")

        for i, url in enumerate(channel_urls, 1):
            try:
                logger.info(f"[{i}/{len(channel_urls)}] Enriching: {url}")
                channel = self.enrich_channel(url)
                if channel:
                    stats['successful'] += 1
                else:
                    stats['failed'] += 1
                    stats['errors'].append(f"Failed to enrich: {url}")
            except Exception as e:
                stats['failed'] += 1
                error_msg = f"Error enriching {url}: {str(e)}"
                stats['errors'].append(error_msg)
                logger.error(error_msg)

        logger.info("="*70)
        logger.info("Bulk Enrichment Summary")
        logger.info("="*70)
        logger.info(f"Total channels: {stats['total']}")
        logger.info(f"Successful: {stats['successful']}")
        logger.info(f"Failed: {stats['failed']}")
        logger.info("="*70)

        return stats


def enrich_youtube_channel(channel_url: str) -> Optional[Channel]:
    """
    Convenience function to enrich a single YouTube channel.

    Args:
        channel_url: YouTube channel URL

    Returns:
        Channel object if successful, None otherwise
    """
    enricher = YouTubeChannelEnricher()
    return enricher.enrich_channel(channel_url)


if __name__ == "__main__":
    # Simple CLI for testing
    if len(sys.argv) > 1:
        url = sys.argv[1]
        print(f"Enriching channel: {url}")
        channel = enrich_youtube_channel(url)
        if channel:
            print(f"✓ Success: {channel.display_name}")
        else:
            print("✗ Failed")
    else:
        print("Usage: python youtube_channel_enricher.py <youtube_channel_url>")
