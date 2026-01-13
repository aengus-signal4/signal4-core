#!/usr/bin/env python3
import sys
from pathlib import Path

from src.utils.paths import get_project_root
from src.utils.config import load_config
import asyncio
import logging
import json
import yaml
from typing import Dict, Optional
from datetime import datetime, timezone
import yt_dlp
import random
import argparse
from yt_dlp.utils import DownloadError
import os
import re

# Add the project root to Python path
sys.path.append(str(get_project_root()))

from src.utils.logger import setup_worker_logger
from src.storage.s3_utils import S3StorageConfig, S3Storage
from src.utils.datetime_encoder import DateTimeEncoder
from src.database.session import get_session
from src.database.models import Content

logger = setup_worker_logger('download_rumble')

class RumbleDownloader:
    def __init__(self, proxy: Optional[str] = None, user_agent: Optional[str] = None):
        """Initialize Rumble downloader with configuration"""
        # Load config
        self.config = load_config()
            
        # Get Rumble downloader specific config
        self.rumble_config = self.config['processing'].get('rumble_downloader', {})
        logger.debug(f"Loaded Rumble downloader config: {self.rumble_config}")
        
        # Set up temp directory for downloads
        self.temp_dir = Path("/tmp/rumble_downloads")
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Using temp directory: {self.temp_dir}")
        
        # Initialize S3 storage
        s3_config = S3StorageConfig(
            endpoint_url=self.config['storage']['s3']['endpoint_url'],
            access_key=self.config['storage']['s3']['access_key'],
            secret_key=self.config['storage']['s3']['secret_key'],
            bucket_name=self.config['storage']['s3']['bucket_name'],
            use_ssl=self.config['storage']['s3']['use_ssl']
        )
        self.s3_storage = S3Storage(s3_config)
        logger.debug("Initialized S3 storage connection")
        
        # Download settings
        self.download_retries = self.rumble_config.get('download_retries', 3)
        self.socket_timeout = self.rumble_config.get('socket_timeout', 30)
        self.video_format = self.rumble_config.get('video_format', 'bestvideo[height<=360][ext=mp4]+bestaudio[ext=m4a]/best[height<=360][ext=mp4]/best[ext=mp4]')
        
        # Get and parse throttle rate
        self.throttle_rate_str = self.rumble_config.get('throttle_rate')
        self.throttle_rate_bytes = self._parse_rate_limit(self.throttle_rate_str)
        
        log_msg = f"Download settings: retries={self.download_retries}, timeout={self.socket_timeout}s, format={self.video_format}"
        if self.throttle_rate_bytes:
            log_msg += f", throttle_rate={self.throttle_rate_str} ({self.throttle_rate_bytes} B/s)"
        logger.debug(log_msg)

        # Set proxy and user-agent
        self.proxy = proxy
        self.user_agent = user_agent

    def _parse_rate_limit(self, rate_str: Optional[str | int]) -> Optional[int]:
        """Parse rate limit string (e.g., 50K, 1M) or int into bytes per second."""
        if rate_str is None:
            return None
        
        # If it's already an integer, assume it's bytes/second
        if isinstance(rate_str, int):
            if rate_str > 0:
                logger.debug(f"Using integer throttle_rate: {rate_str} B/s")
                return rate_str
            else:
                logger.warning(f"Non-positive integer rate limit: {rate_str}. Ignoring throttle.")
                return None

        # If not an int, treat as string and parse
        rate_str_parsed = str(rate_str)
        
        match = re.match(r'^(\d+(?:\.\d+)?)([KMG])?$', rate_str_parsed.strip().upper())
        if not match:
            logger.warning(f"Invalid rate limit format: '{rate_str_parsed}'. Ignoring throttle.")
            return None
            
        value = float(match.group(1))
        unit = match.group(2)
        
        if unit == 'K':
            return int(value * 1024)
        elif unit == 'M':
            return int(value * 1024 * 1024)
        elif unit == 'G':
            return int(value * 1024 * 1024 * 1024)
        else:
            return int(value)

    async def _get_video_url(self, content_id: str) -> Optional[str]:
        """Get the video URL from database"""
        try:
            with get_session() as session:
                content = session.query(Content).filter_by(
                    content_id=content_id,
                    platform='rumble'
                ).first()
                
                if content and content.meta_data:
                    video_url = content.meta_data.get('webpage_url')
                    if video_url:
                        # Clean up the URL - remove .html and query parameters
                        # Example: https://rumble.com/v6u23jp-ragecast-540-815pm-est.html?e9s=src_v1_ucp_a
                        # Becomes: https://rumble.com/v6u23jp-ragecast-540-815pm-est
                        video_url = video_url.split('?')[0]  # Remove query parameters
                        if video_url.endswith('.html'):
                            video_url = video_url[:-5]  # Remove .html extension
                        
                        logger.debug(f"Found and cleaned video URL for {content_id}: {video_url}")
                        return video_url
                    else:
                        logger.warning(f"No webpage_url in metadata for {content_id}")
                else:
                    logger.warning(f"Content not found in database for {content_id}")
                    
            return None
        except Exception as e:
            logger.error(f"Error getting video URL from database: {str(e)}")
            return None

    async def _check_content_exists(self, content_id: str) -> bool:
        """Check if content already exists in S3"""
        try:
            # Check for meta.json
            meta_key = f"content/{content_id}/meta.json"
            meta_exists = self.s3_storage.file_exists(meta_key)
            
            # Check for source.mp4
            source_key = f"content/{content_id}/source.mp4"
            source_exists = self.s3_storage.file_exists(source_key)
            
            logger.debug(f"Content existence check for {content_id}: meta={meta_exists}, source={source_exists}")
            return meta_exists and source_exists
            
        except Exception as e:
            logger.error(f"Error checking if content exists in S3: {str(e)}")
            return False

    async def _download_video(self, video_url: str, output_path: Path) -> Dict:
        """Download video using yt-dlp"""
        try:
            logger.info(f"Starting download for {video_url} to {output_path}")
            
            # Configure yt-dlp
            ydl_opts = {
                'format': self.video_format,
                'outtmpl': str(output_path),
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
                'nocheckcertificate': True,
                'ignoreerrors': False,
                'no_color': True,
                'socket_timeout': self.socket_timeout,
                'retries': self.download_retries
            }
            
            # Add rate limit if configured
            if self.throttle_rate_bytes:
                ydl_opts['ratelimit'] = self.throttle_rate_bytes
                logger.debug(f"Applying rate limit: {self.throttle_rate_str}")
            
            # Add proxy if configured
            if self.proxy:
                ydl_opts['proxy'] = self.proxy
                logger.debug(f"Using proxy: {self.proxy}")
            
            # Add user-agent if configured
            if self.user_agent:
                ydl_opts['user_agent'] = self.user_agent
                logger.debug(f"Using User-Agent: {self.user_agent}")
            
            # Run download
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                try:
                    logger.debug(f"Executing yt-dlp download with options: {ydl_opts}")
                    ydl.download([video_url])
                except DownloadError as e:
                    error_message = str(e)
                    logger.error(f"yt-dlp download failed: {error_message}")
                    
                    # Check for permanent failure conditions
                    permanent_failures = [
                        "Video unavailable",
                        "has been removed",
                        "This video is private",
                        "This video is no longer available",
                        "This video has been removed for violating",
                        "This content is not available in your country"
                    ]
                    
                    is_permanent_failure = any(msg in error_message for msg in permanent_failures)
                    
                    if is_permanent_failure:
                        logger.warning(f"Permanent download failure detected for {video_url}: {error_message}")
                        return {'status': 'failed', 'error': error_message, 'permanent': True}
                    else:
                        logger.error(f"Unhandled yt-dlp DownloadError for {video_url}: {error_message}")
                        return {'status': 'error', 'error': error_message}
            
            if output_path.exists() and output_path.stat().st_size > 0:
                logger.info(f"Successfully downloaded video to {output_path} ({output_path.stat().st_size} bytes)")
                return {'status': 'success'}
            else:
                logger.error("Download failed - file not created or empty")
                raise Exception("Download failed - file not created or empty")
                
        except Exception as e:
            logger.error(f"Error in _download_video: {str(e)}")
            return {'status': 'error', 'error': str(e)}

    async def _upload_to_s3(self, content_id: str, file_path: str) -> bool:
        """Upload file to S3"""
        try:    
            logger.debug(f"Starting S3 upload for content {content_id}")
            
            # Create metadata
            meta_data = {
                'platform': 'rumble',
                'content_id': content_id,
                'source_extension': '.mp4',
                'updated_at': datetime.now(timezone.utc).isoformat()
            }

            # Create temporary meta.json file
            meta_path = self.temp_dir / f"{content_id}_meta.json"
            with open(meta_path, 'w') as f:
                json.dump(meta_data, f, indent=2, cls=DateTimeEncoder)
            logger.debug(f"Created metadata file at {meta_path}")

            # Upload meta.json
            meta_key = f"content/{content_id}/meta.json"
            meta_success = self.s3_storage.upload_file(str(meta_path), meta_key)
            if not meta_success:
                logger.error(f"Failed to upload metadata for {content_id}")
                return False
            logger.debug(f"Successfully uploaded metadata to {meta_key}")

            # Clean up meta.json
            meta_path.unlink()
            logger.debug("Cleaned up temporary metadata file")

            # Upload source file
            source_key = f"content/{content_id}/source.mp4"
            source_success = self.s3_storage.upload_file(file_path, source_key)
            if not source_success:
                logger.error(f"Failed to upload source file for {content_id}")
                return False
            logger.debug(f"Successfully uploaded source file to {source_key}")

            return True

        except Exception as e:
            logger.error(f"Error uploading to S3: {str(e)}")
            return False

    async def process_content(self, content_id: str) -> Dict:
        """Process a single Rumble video download"""
        try:
            # Check if content already exists in S3
            if await self._check_content_exists(content_id):
                logger.info(f"Content {content_id} already exists in S3, skipping download")
                return {
                    'status': 'skipped', 
                    'reason': 'already_exists',
                    'message': 'Content already exists in S3',
                    'skip_wait_time': True
                }

            # Get the actual video URL from database
            video_url = await self._get_video_url(content_id)
            if not video_url:
                logger.error(f"Could not find video URL for content {content_id}")
                return {'status': 'failed', 'error': f'Video URL not found for content {content_id}', 'permanent': True}
            
            # Download video
            temp_path = self.temp_dir / f"{content_id}.mp4"
            download_result = await self._download_video(video_url, temp_path)
            
            if download_result['status'] != 'success':
                logger.error(f"Download failed for {content_id}: {download_result.get('error', 'Unknown error')}")
                return download_result

            # Upload to S3
            upload_success = await self._upload_to_s3(content_id, str(temp_path))
            if not upload_success:
                logger.error(f"Failed to upload {content_id} to S3")
                return {'status': 'error', 'error': 'Failed to upload to S3'}

            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()
                logger.debug(f"Cleaned up temporary file {temp_path}")

            logger.info(f"Successfully processed content {content_id}")
            return {'status': 'success'}

        except Exception as e:
            logger.error(f"Error processing content {content_id}: {str(e)}")
            return {'status': 'error', 'error': str(e)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download Rumble video')
    parser.add_argument('--content', required=True, help='Content ID to process')
    parser.add_argument('--proxy', help='Proxy server address (e.g., http://user:pass@host:port)')
    parser.add_argument('--user-agent', help='Custom User-Agent string to use')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Send preliminary messages to stderr or logger
    logger.info(f"Attempting to download video: {args.content}")
    if args.proxy:
        logger.info(f"Using proxy: {args.proxy}")
    if args.user_agent:
        logger.info(f"Using User-Agent: {args.user_agent}")
        
    async def main():
        downloader = RumbleDownloader(
            proxy=args.proxy, 
            user_agent=args.user_agent
        )
        result = await downloader.process_content(args.content)
        # Print ONLY the final JSON result to stdout
        print(json.dumps(result, cls=DateTimeEncoder))
    
    asyncio.run(main()) 