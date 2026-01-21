"""
YouTube Thumbnail Downloader

Downloads YouTube channel thumbnails and stores them in S3.
Thumbnails are stored at: thumbnails/channels/{channel_id}.{ext}

YouTube channel thumbnails require fetching the actual thumbnail URL from the
YouTube Data API (the URL contains a hash, not just the channel ID).

Usage:
    # Download thumbnails for all YouTube channels (5 second rate limit)
    uv run python -m src.ingestion.youtube_thumbnails

    # Download for specific channel IDs
    uv run python -m src.ingestion.youtube_thumbnails --channel-ids 123 456 789

    # Test mode - process only 3 channels
    uv run python -m src.ingestion.youtube_thumbnails --test

    # Force re-download existing thumbnails
    uv run python -m src.ingestion.youtube_thumbnails --force
"""

import asyncio
import aiohttp
import logging
import argparse
import os
import re
from typing import Optional, List, Dict, Tuple
from datetime import datetime
from io import BytesIO
from PIL import Image
from googleapiclient.discovery import build
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from signal4/.env
from src.utils.paths import get_env_path
load_dotenv(get_env_path())

from src.database.session import get_session
from src.database.models import Channel
from src.storage.s3_utils import S3Storage, S3StorageConfig
from src.utils.logger import setup_worker_logger
from sqlalchemy.orm.attributes import flag_modified


# S3 folder for thumbnails (shared with podcasts)
THUMBNAIL_PREFIX = "thumbnails/channels"

# Target thumbnail size (width x height)
THUMBNAIL_SIZE = (400, 400)


def extract_youtube_channel_id(url: str) -> Optional[str]:
    """
    Extract YouTube channel ID from various URL formats.

    Args:
        url: YouTube channel URL (e.g., https://www.youtube.com/channel/UC...)

    Returns:
        Channel ID (UC...) or None if not found
    """
    if not url:
        return None

    # Direct channel ID format: /channel/UC...
    match = re.search(r'youtube\.com/channel/([UC][a-zA-Z0-9_-]+)', url)
    if match:
        return match.group(1)

    return None


class YouTubeThumbnailDownloader:
    """Downloads and stores YouTube channel thumbnails in S3."""

    def __init__(
        self,
        s3_storage: S3Storage,
        logger: logging.Logger = None,
        request_delay: float = 5.0,  # 5 seconds between requests for throughput
        timeout: int = 30
    ):
        """
        Initialize the thumbnail downloader.

        Args:
            s3_storage: S3Storage instance for storing thumbnails
            logger: Logger instance
            request_delay: Delay between requests in seconds (default 5s for YouTube)
            timeout: HTTP request timeout in seconds
        """
        self.s3 = s3_storage
        self.logger = logger or setup_worker_logger('youtube_thumbnails')
        self.request_delay = request_delay
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None
        self._youtube = None
        self._init_youtube_client()

    def _init_youtube_client(self):
        """Initialize YouTube API client."""
        try:
            api_keys_str = os.getenv('YOUTUBE_API_KEYS')
            if api_keys_str:
                api_key = api_keys_str.split(',')[0].strip()
                self._youtube = build('youtube', 'v3', developerKey=api_key)
                self.logger.info("YouTube API client initialized")
            else:
                self.logger.warning("No YOUTUBE_API_KEYS found, will use stored URLs only")
        except Exception as e:
            self.logger.warning(f"Could not initialize YouTube API client: {e}")

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                headers={
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                    'Accept': 'image/*,*/*;q=0.8',
                }
            )
        return self._session

    async def close(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()

    def get_thumbnail_s3_key(self, channel_id: int, extension: str = 'jpg') -> str:
        """Get the S3 key for a channel's thumbnail."""
        return f"{THUMBNAIL_PREFIX}/{channel_id}.{extension}"

    def thumbnail_exists(self, channel_id: int) -> Tuple[bool, Optional[str]]:
        """
        Check if a thumbnail already exists for a channel.

        Returns:
            Tuple of (exists: bool, s3_key: str or None)
        """
        for ext in ['jpg', 'png', 'webp', 'gif']:
            s3_key = self.get_thumbnail_s3_key(channel_id, ext)
            if self.s3.file_exists(s3_key):
                return True, s3_key
        return False, None

    async def _get_thumbnail_url_from_api(self, youtube_channel_id: str) -> Optional[str]:
        """
        Get thumbnail URL from YouTube Data API.

        Args:
            youtube_channel_id: YouTube channel ID (UC...)

        Returns:
            Thumbnail URL or None
        """
        if not self._youtube:
            return None

        try:
            loop = asyncio.get_event_loop()
            request = self._youtube.channels().list(
                part="snippet",
                id=youtube_channel_id
            )
            response = await loop.run_in_executor(None, request.execute)

            if response.get('items'):
                thumbnails = response['items'][0]['snippet'].get('thumbnails', {})
                # Prefer high quality, then medium, then default
                for quality in ['high', 'medium', 'default']:
                    if quality in thumbnails:
                        return thumbnails[quality]['url']

        except Exception as e:
            self.logger.warning(f"API error getting thumbnail for {youtube_channel_id}: {e}")

        return None

    async def _get_thumbnail_url_from_page(self, youtube_channel_id: str) -> Optional[str]:
        """
        Scrape thumbnail URL from YouTube channel page.

        Fallback method when API is not available.

        Args:
            youtube_channel_id: YouTube channel ID (UC...)

        Returns:
            Thumbnail URL or None
        """
        session = await self._get_session()
        url = f"https://www.youtube.com/channel/{youtube_channel_id}"

        try:
            async with session.get(url) as response:
                if response.status != 200:
                    return None

                html = await response.text()

                # Look for avatar/thumbnail URL in the page
                # YouTube embeds channel thumbnail in various meta tags and JSON
                patterns = [
                    r'"avatar":\{"thumbnails":\[\{"url":"([^"]+)"',
                    r'"channelAvatar":\{"thumbnails":\[\{"url":"([^"]+)"',
                    r'<meta property="og:image" content="([^"]+)"',
                    r'"thumbnails":\[\{"url":"(https://yt3\.googleusercontent\.com/[^"]+)"',
                ]

                for pattern in patterns:
                    match = re.search(pattern, html)
                    if match:
                        thumb_url = match.group(1)
                        # Clean up escaped characters
                        thumb_url = thumb_url.replace('\\u0026', '&')
                        self.logger.debug(f"Found thumbnail URL via scraping: {thumb_url[:60]}...")
                        return thumb_url

        except Exception as e:
            self.logger.debug(f"Error scraping page for {youtube_channel_id}: {e}")

        return None

    async def _fetch_youtube_thumbnail(self, youtube_channel_id: str, stored_url: str = None) -> Optional[Tuple[bytes, str, str]]:
        """
        Fetch thumbnail from YouTube.

        Tries multiple sources in order:
        1. Stored URL (if available)
        2. YouTube Data API (if configured)
        3. Page scraping (fallback)

        Args:
            youtube_channel_id: YouTube channel ID (UC...)
            stored_url: Previously stored thumbnail URL

        Returns:
            Tuple of (image_bytes, extension, url) or None on failure
        """
        session = await self._get_session()

        # URLs to try in order
        urls_to_try = []

        # First try stored URL if available
        if stored_url:
            urls_to_try.append(stored_url)

        # Then try to get URL from API
        api_url = await self._get_thumbnail_url_from_api(youtube_channel_id)
        if api_url and api_url not in urls_to_try:
            urls_to_try.append(api_url)

        # Then try scraping the page
        scraped_url = await self._get_thumbnail_url_from_page(youtube_channel_id)
        if scraped_url and scraped_url not in urls_to_try:
            urls_to_try.append(scraped_url)

        self.logger.debug(f"URLs to try for {youtube_channel_id}: {len(urls_to_try)}")

        # Try each URL
        for url in urls_to_try:
            try:
                async with session.get(url, allow_redirects=True) as response:
                    if response.status == 200:
                        image_data = await response.read()
                        if len(image_data) > 100:
                            result = await self._process_image(image_data)
                            if result:
                                return result[0], result[1], url
            except Exception as e:
                self.logger.debug(f"Failed to fetch from {url}: {e}")
                continue

        return None

    async def _process_image(self, image_data: bytes) -> Optional[Tuple[bytes, str]]:
        """Process and resize image data."""
        try:
            img = Image.open(BytesIO(image_data))

            # Convert RGBA to RGB if needed (for JPEG)
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')

            # Resize to thumbnail size (maintain aspect ratio)
            img.thumbnail(THUMBNAIL_SIZE, Image.Resampling.LANCZOS)

            # Save to bytes
            output = BytesIO()
            img.save(output, format='JPEG', quality=85, optimize=True)
            output.seek(0)

            return output.read(), 'jpg'

        except Exception as e:
            self.logger.error(f"Error processing image: {e}")
            return None

    async def download_thumbnail(
        self,
        channel_id: int,
        youtube_channel_id: str,
        force: bool = False,
        stored_image_url: Optional[str] = None
    ) -> Dict:
        """
        Download and store a thumbnail for a YouTube channel.

        Args:
            channel_id: Database channel ID
            youtube_channel_id: YouTube channel ID (UC...)
            force: If True, re-download even if thumbnail exists
            stored_image_url: Previously stored image URL for change detection

        Returns:
            Dict with status information
        """
        result = {
            'channel_id': channel_id,
            'youtube_channel_id': youtube_channel_id,
            'success': False,
            'message': '',
            's3_key': None,
            'image_url': None,
            'url_changed': False
        }

        if not youtube_channel_id:
            result['message'] = 'No YouTube channel ID'
            return result

        # First, get the current thumbnail URL from API (needed for change detection)
        # We do this BEFORE checking S3 existence so we can detect URL changes
        current_url = await self._get_thumbnail_url_from_api(youtube_channel_id)
        if not current_url:
            # Fallback to scraping if API fails
            current_url = await self._get_thumbnail_url_from_page(youtube_channel_id)

        if current_url:
            result['image_url'] = current_url

            # Check if URL changed - if so, force re-download
            if stored_image_url and stored_image_url != current_url:
                result['url_changed'] = True
                self.logger.info(f"Channel {channel_id}: Image URL changed, will re-download")
                force = True

        # Check if already exists (unless force or URL changed)
        if not force:
            exists, existing_key = self.thumbnail_exists(channel_id)
            if exists:
                result['success'] = True
                result['message'] = 'Already exists'
                result['s3_key'] = existing_key
                return result

        # Download thumbnail from YouTube
        image_result = await self._fetch_youtube_thumbnail(youtube_channel_id, stored_image_url)
        if not image_result:
            result['message'] = 'Failed to download from YouTube'
            return result

        image_bytes, extension, actual_url = image_result
        result['image_url'] = actual_url

        # Upload to S3
        s3_key = self.get_thumbnail_s3_key(channel_id, extension)
        try:
            self.s3._client.put_object(
                Bucket=self.s3.config.bucket_name,
                Key=s3_key,
                Body=image_bytes,
                ContentType=f'image/{extension}'
            )

            result['success'] = True
            if result['url_changed']:
                result['message'] = 'Updated (URL changed)'
            else:
                result['message'] = 'Downloaded successfully'
            result['s3_key'] = s3_key
            result['size_bytes'] = len(image_bytes)

            self.logger.info(f"Uploaded thumbnail for channel {channel_id}: {s3_key} ({len(image_bytes)} bytes)")

        except Exception as e:
            result['message'] = f'S3 upload failed: {e}'
            self.logger.error(f"Failed to upload thumbnail for channel {channel_id}: {e}")

        return result

    async def download_thumbnails_batch(
        self,
        channels: List[Dict],
        force: bool = False
    ) -> Dict:
        """
        Download thumbnails for a batch of channels with rate limiting.

        Args:
            channels: List of dicts with keys:
                - 'id': database channel ID (required)
                - 'youtube_channel_id': YouTube channel ID UC... (required)
                - 'stored_image_url': previously stored image URL (optional)
            force: If True, re-download existing thumbnails

        Returns:
            Summary dict with statistics
        """
        results = {
            'total': len(channels),
            'success': 0,
            'updated': 0,
            'skipped': 0,
            'failed': 0,
            'errors': [],
            'url_updates': []
        }

        for i, channel in enumerate(channels):
            channel_id = channel['id']
            youtube_channel_id = channel.get('youtube_channel_id')

            self.logger.info(f"Processing channel {i+1}/{len(channels)}: {channel_id} (YT: {youtube_channel_id})")

            if not youtube_channel_id:
                results['failed'] += 1
                results['errors'].append({
                    'channel_id': channel_id,
                    'error': 'No YouTube channel ID'
                })
                continue

            result = await self.download_thumbnail(
                channel_id,
                youtube_channel_id,
                force,
                stored_image_url=channel.get('stored_image_url')
            )

            if result['success']:
                if result['message'] == 'Already exists':
                    results['skipped'] += 1
                elif result.get('url_changed'):
                    results['updated'] += 1
                    if result.get('image_url'):
                        results['url_updates'].append({
                            'channel_id': channel_id,
                            'image_url': result['image_url']
                        })
                else:
                    results['success'] += 1
                    if result.get('image_url'):
                        results['url_updates'].append({
                            'channel_id': channel_id,
                            'image_url': result['image_url']
                        })
            else:
                results['failed'] += 1
                results['errors'].append({
                    'channel_id': channel_id,
                    'error': result['message']
                })

            # Rate limiting - 5 seconds between requests
            if i < len(channels) - 1:
                await asyncio.sleep(self.request_delay)

        return results


def get_thumbnail_url(s3_storage: S3Storage, channel_id: int, expires_in: int = 3600) -> Optional[str]:
    """
    Get a pre-signed URL for a channel's thumbnail.

    Args:
        s3_storage: S3Storage instance
        channel_id: Channel ID
        expires_in: URL expiration time in seconds (default 1 hour)

    Returns:
        Pre-signed URL or None if thumbnail doesn't exist
    """
    for ext in ['jpg', 'png', 'webp', 'gif']:
        s3_key = f"{THUMBNAIL_PREFIX}/{channel_id}.{ext}"
        if s3_storage.file_exists(s3_key):
            return s3_storage.get_file_url(s3_key, expires_in)
    return None


async def main():
    """Main entry point for the YouTube thumbnail downloader."""
    parser = argparse.ArgumentParser(description='Download YouTube channel thumbnails to S3')
    parser.add_argument('--channel-ids', type=int, nargs='+', help='Specific channel IDs to process')
    parser.add_argument('--test', action='store_true', help='Test mode - process only 3 channels')
    parser.add_argument('--force', action='store_true', help='Re-download existing thumbnails')
    parser.add_argument('--delay', type=float, default=5.0, help='Delay between requests (seconds)')
    args = parser.parse_args()

    logger = setup_worker_logger('youtube_thumbnails')
    logger.info("Starting YouTube thumbnail downloader")

    # Initialize S3
    s3_config = S3StorageConfig()
    s3_storage = S3Storage(s3_config)

    # Get channels to process
    with get_session() as session:
        query = session.query(Channel).filter(
            Channel.platform == 'youtube'
        )

        if args.channel_ids:
            query = query.filter(Channel.id.in_(args.channel_ids))

        channels = query.all()

        if args.test:
            channels = channels[:3]

        channel_data = []
        for c in channels:
            youtube_channel_id = extract_youtube_channel_id(c.primary_url)
            channel_data.append({
                'id': c.id,
                'display_name': c.display_name,
                'youtube_channel_id': youtube_channel_id,
                'stored_image_url': (c.platform_metadata or {}).get('image_url')
            })

    logger.info(f"Found {len(channel_data)} channels to process")

    if not channel_data:
        logger.warning("No channels found to process")
        return

    # Download thumbnails
    downloader = YouTubeThumbnailDownloader(
        s3_storage=s3_storage,
        logger=logger,
        request_delay=args.delay
    )

    try:
        results = await downloader.download_thumbnails_batch(channel_data, force=args.force)

        logger.info(f"\n{'='*50}")
        logger.info("YouTube Thumbnail Download Summary:")
        logger.info(f"  Total: {results['total']}")
        logger.info(f"  New downloads: {results['success']}")
        logger.info(f"  Updated (URL changed): {results['updated']}")
        logger.info(f"  Skipped (already exist): {results['skipped']}")
        logger.info(f"  Failed: {results['failed']}")

        if results['errors']:
            logger.info(f"\nErrors ({len(results['errors'])}):")
            for err in results['errors'][:10]:
                logger.info(f"  Channel {err['channel_id']}: {err['error']}")

        # Update database with new image URLs
        if results['url_updates']:
            logger.info(f"\nUpdating {len(results['url_updates'])} channel image URLs in database...")
            with get_session() as session:
                for update in results['url_updates']:
                    channel = session.query(Channel).filter(Channel.id == update['channel_id']).first()
                    if channel:
                        pm = dict(channel.platform_metadata or {})
                        pm['image_url'] = update['image_url']
                        pm['thumbnail_updated_at'] = datetime.utcnow().isoformat()
                        channel.platform_metadata = pm
                        flag_modified(channel, 'platform_metadata')
                session.commit()
            logger.info("Database updated successfully")

    finally:
        await downloader.close()


if __name__ == '__main__':
    asyncio.run(main())
