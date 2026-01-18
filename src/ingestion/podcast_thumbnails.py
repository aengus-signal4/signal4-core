"""
Podcast Thumbnail Downloader

Downloads podcast channel thumbnails from RSS feeds and stores them in S3.
Thumbnails are stored at: thumbnails/channels/{channel_id}.{ext}

Usage:
    # Download thumbnails for all podcast channels (respects rate limits)
    uv run python -m src.ingestion.podcast_thumbnails

    # Download for specific channel IDs
    uv run python -m src.ingestion.podcast_thumbnails --channel-ids 123 456 789

    # Test mode - process only 3 channels
    uv run python -m src.ingestion.podcast_thumbnails --test

    # Force re-download existing thumbnails
    uv run python -m src.ingestion.podcast_thumbnails --force
"""

import asyncio
import aiohttp
import feedparser
import logging
import argparse
import os
from typing import Optional, List, Dict, Tuple
from datetime import datetime
from io import BytesIO
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv

# Load environment variables from core/.env
_env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(_env_path)

from src.database.session import get_session
from src.database.models import Channel
from src.storage.s3_utils import S3Storage, S3StorageConfig
from src.utils.logger import setup_worker_logger
from sqlalchemy import text
from sqlalchemy.orm.attributes import flag_modified


# S3 folder for thumbnails
THUMBNAIL_PREFIX = "thumbnails/channels"

# Supported image formats and their extensions
SUPPORTED_FORMATS = {
    'image/jpeg': 'jpg',
    'image/jpg': 'jpg',
    'image/png': 'png',
    'image/webp': 'webp',
    'image/gif': 'gif',
}

# Target thumbnail size (width x height)
THUMBNAIL_SIZE = (400, 400)


class PodcastThumbnailDownloader:
    """Downloads and stores podcast thumbnails in S3."""

    def __init__(
        self,
        s3_storage: S3Storage,
        logger: logging.Logger = None,
        request_delay: float = 1.0,
        timeout: int = 30
    ):
        """
        Initialize the thumbnail downloader.

        Args:
            s3_storage: S3Storage instance for storing thumbnails
            logger: Logger instance
            request_delay: Delay between requests in seconds (rate limiting)
            timeout: HTTP request timeout in seconds
        """
        self.s3 = s3_storage
        self.logger = logger or setup_worker_logger('podcast_thumbnails')
        self.request_delay = request_delay
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None

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

    async def _fetch_image_url_from_feed(self, feed_url: str) -> Optional[str]:
        """
        Fetch the image URL from a podcast RSS feed.

        Args:
            feed_url: RSS feed URL

        Returns:
            Image URL or None if not found
        """
        try:
            session = await self._get_session()
            async with session.get(feed_url) as response:
                if response.status != 200:
                    self.logger.warning(f"Failed to fetch feed {feed_url}: HTTP {response.status}")
                    return None

                content = await response.text()
                feed = feedparser.parse(content)

                if not feed.feed:
                    return None

                # Try various image locations in RSS feed
                feed_data = feed.feed

                # itunes:image (most common for podcasts)
                if hasattr(feed_data, 'image') and hasattr(feed_data.image, 'href'):
                    return feed_data.image.href

                # Check for itunes_image attribute
                if hasattr(feed_data, 'itunes_image'):
                    if hasattr(feed_data.itunes_image, 'href'):
                        return feed_data.itunes_image.href
                    elif isinstance(feed_data.itunes_image, dict):
                        return feed_data.itunes_image.get('href')

                # Standard RSS image
                if 'image' in feed_data:
                    img = feed_data['image']
                    if isinstance(img, dict) and 'href' in img:
                        return img['href']
                    elif hasattr(img, 'url'):
                        return img.url

                return None

        except asyncio.TimeoutError:
            self.logger.warning(f"Timeout fetching feed {feed_url}")
            return None
        except Exception as e:
            self.logger.error(f"Error fetching feed {feed_url}: {e}")
            return None

    async def _download_and_process_image(self, image_url: str) -> Optional[Tuple[bytes, str]]:
        """
        Download an image and process it (resize, convert format).

        Args:
            image_url: URL of the image to download

        Returns:
            Tuple of (image_bytes, extension) or None on failure
        """
        try:
            session = await self._get_session()
            async with session.get(image_url) as response:
                if response.status != 200:
                    self.logger.warning(f"Failed to download image {image_url}: HTTP {response.status}")
                    return None

                content_type = response.headers.get('Content-Type', '').split(';')[0].strip().lower()
                image_data = await response.read()

                if len(image_data) < 100:
                    self.logger.warning(f"Image too small ({len(image_data)} bytes): {image_url}")
                    return None

                # Process image with PIL
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
                    # Return original if processing fails
                    ext = SUPPORTED_FORMATS.get(content_type, 'jpg')
                    return image_data, ext

        except asyncio.TimeoutError:
            self.logger.warning(f"Timeout downloading image {image_url}")
            return None
        except Exception as e:
            self.logger.error(f"Error downloading image {image_url}: {e}")
            return None

    async def download_thumbnail(
        self,
        channel_id: int,
        feed_url: str,
        force: bool = False,
        stored_image_url: Optional[str] = None
    ) -> Dict:
        """
        Download and store a thumbnail for a podcast channel.

        Supports change detection: if stored_image_url is provided and the RSS feed
        has a different image URL, the thumbnail will be re-downloaded even if one
        already exists in S3.

        Args:
            channel_id: Database channel ID
            feed_url: RSS feed URL to fetch image from
            force: If True, re-download even if thumbnail exists
            stored_image_url: Previously stored image URL for change detection

        Returns:
            Dict with status information including:
                - success: bool
                - message: str
                - s3_key: str or None
                - image_url: str or None (the source image URL, for storing in DB)
                - url_changed: bool (whether the image URL changed)
        """
        result = {
            'channel_id': channel_id,
            'feed_url': feed_url,
            'success': False,
            'message': '',
            's3_key': None,
            'image_url': None,
            'url_changed': False
        }

        # Fetch image URL from RSS feed first (we need it for change detection)
        image_url = await self._fetch_image_url_from_feed(feed_url)
        if not image_url:
            result['message'] = 'No image URL in feed'
            return result

        result['image_url'] = image_url

        # Check if image URL changed
        if stored_image_url and stored_image_url != image_url:
            result['url_changed'] = True
            self.logger.info(f"Channel {channel_id}: Image URL changed, will re-download")
            force = True  # Force re-download when URL changes

        # Check if already exists (unless force or URL changed)
        if not force:
            exists, existing_key = self.thumbnail_exists(channel_id)
            if exists:
                result['success'] = True
                result['message'] = 'Already exists'
                result['s3_key'] = existing_key
                return result

        # Download and process image
        image_result = await self._download_and_process_image(image_url)
        if not image_result:
            result['message'] = 'Failed to download image'
            return result

        image_bytes, extension = image_result

        # Upload to S3
        s3_key = self.get_thumbnail_s3_key(channel_id, extension)
        try:
            # Use put_object directly for bytes
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
                - 'id': channel ID (required)
                - 'primary_url': RSS feed URL (required)
                - 'stored_image_url': previously stored image URL (optional, for change detection)
            force: If True, re-download existing thumbnails

        Returns:
            Summary dict with statistics including 'updated' count for URL changes
        """
        results = {
            'total': len(channels),
            'success': 0,
            'updated': 0,  # Count of thumbnails updated due to URL change
            'skipped': 0,
            'failed': 0,
            'errors': [],
            'url_updates': []  # List of (channel_id, new_image_url) for DB updates
        }

        for i, channel in enumerate(channels):
            channel_id = channel['id']
            feed_url = channel['primary_url']
            stored_image_url = channel.get('stored_image_url')

            self.logger.info(f"Processing channel {i+1}/{len(channels)}: {channel_id}")

            result = await self.download_thumbnail(
                channel_id, feed_url, force, stored_image_url=stored_image_url
            )

            if result['success']:
                if result['message'] == 'Already exists':
                    results['skipped'] += 1
                elif result.get('url_changed'):
                    results['updated'] += 1
                    # Track URL updates for DB
                    if result.get('image_url'):
                        results['url_updates'].append({
                            'channel_id': channel_id,
                            'image_url': result['image_url']
                        })
                else:
                    results['success'] += 1
                    # Track new URLs for DB
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

            # Rate limiting - be respectful
            if i < len(channels) - 1:
                await asyncio.sleep(self.request_delay)

        return results


def get_thumbnail_url(s3_storage: S3Storage, channel_id: int, expires_in: int = 3600) -> Optional[str]:
    """
    Get a pre-signed URL for a channel's thumbnail.

    This is the quick access utility function.

    Args:
        s3_storage: S3Storage instance
        channel_id: Channel ID
        expires_in: URL expiration time in seconds (default 1 hour)

    Returns:
        Pre-signed URL or None if thumbnail doesn't exist
    """
    # Check for thumbnail in various formats
    for ext in ['jpg', 'png', 'webp', 'gif']:
        s3_key = f"{THUMBNAIL_PREFIX}/{channel_id}.{ext}"
        if s3_storage.file_exists(s3_key):
            return s3_storage.get_file_url(s3_key, expires_in)
    return None


def get_thumbnail_s3_key(channel_id: int) -> Optional[str]:
    """
    Get the S3 key for a channel's thumbnail (without checking if it exists).

    Args:
        channel_id: Channel ID

    Returns:
        S3 key string (assumes jpg format)
    """
    return f"{THUMBNAIL_PREFIX}/{channel_id}.jpg"


async def main():
    """Main entry point for the thumbnail downloader."""
    parser = argparse.ArgumentParser(description='Download podcast thumbnails to S3')
    parser.add_argument('--channel-ids', type=int, nargs='+', help='Specific channel IDs to process')
    parser.add_argument('--test', action='store_true', help='Test mode - process only 3 channels')
    parser.add_argument('--force', action='store_true', help='Re-download existing thumbnails')
    parser.add_argument('--delay', type=float, default=1.0, help='Delay between requests (seconds)')
    args = parser.parse_args()

    logger = setup_worker_logger('podcast_thumbnails')
    logger.info("Starting podcast thumbnail downloader")

    # Initialize S3
    s3_config = S3StorageConfig()
    s3_storage = S3Storage(s3_config)

    # Get channels to process
    with get_session() as session:
        query = session.query(Channel).filter(
            Channel.platform == 'podcast'
        )

        if args.channel_ids:
            query = query.filter(Channel.id.in_(args.channel_ids))

        channels = query.all()

        if args.test:
            channels = channels[:3]

        channel_data = [
            {
                'id': c.id,
                'primary_url': c.primary_url,
                'display_name': c.display_name,
                'stored_image_url': (c.platform_metadata or {}).get('image_url')
            }
            for c in channels
        ]

    logger.info(f"Found {len(channel_data)} channels to process")

    if not channel_data:
        logger.warning("No channels found to process")
        return

    # Download thumbnails
    downloader = PodcastThumbnailDownloader(
        s3_storage=s3_storage,
        logger=logger,
        request_delay=args.delay
    )

    try:
        results = await downloader.download_thumbnails_batch(channel_data, force=args.force)

        logger.info(f"\n{'='*50}")
        logger.info("Thumbnail Download Summary:")
        logger.info(f"  Total: {results['total']}")
        logger.info(f"  New downloads: {results['success']}")
        logger.info(f"  Updated (URL changed): {results['updated']}")
        logger.info(f"  Skipped (already exist): {results['skipped']}")
        logger.info(f"  Failed: {results['failed']}")

        if results['errors']:
            logger.info(f"\nErrors ({len(results['errors'])}):")
            for err in results['errors'][:10]:  # Show first 10 errors
                logger.info(f"  Channel {err['channel_id']}: {err['error']}")

        # Update database with new image URLs
        if results['url_updates']:
            logger.info(f"\nUpdating {len(results['url_updates'])} channel image URLs in database...")
            with get_session() as session:
                for update in results['url_updates']:
                    channel = session.query(Channel).filter(Channel.id == update['channel_id']).first()
                    if channel:
                        pm = dict(channel.platform_metadata or {})  # Make a copy
                        pm['image_url'] = update['image_url']
                        pm['thumbnail_updated_at'] = datetime.utcnow().isoformat()
                        channel.platform_metadata = pm
                        flag_modified(channel, 'platform_metadata')  # Required for JSONB updates
                session.commit()
            logger.info("Database updated successfully")

    finally:
        await downloader.close()


if __name__ == '__main__':
    asyncio.run(main())
