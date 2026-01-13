#!/usr/bin/env python3
import sys
from pathlib import Path
import asyncio
import logging
import json
from typing import Dict, Optional
from datetime import datetime, timezone
import yt_dlp
import random
import argparse
from yt_dlp.utils import DownloadError
import os
import re # Added for parsing rate limit string

# Add the project root to Python path
from src.utils.paths import get_project_root
sys.path.append(str(get_project_root()))

from src.utils.logger import setup_worker_logger
from src.utils.config import load_config
from src.storage.s3_utils import S3StorageConfig, S3Storage, create_s3_storage_from_config
from src.utils.datetime_encoder import DateTimeEncoder
from src.utils.error_codes import ErrorCode, create_error_result, create_skipped_result

logger = setup_worker_logger('download_youtube')

class YouTubeDownloader:
    def __init__(self, cookie_profile: Optional[str] = None, proxy: Optional[str] = None, user_agent: Optional[str] = None):
        """Initialize YouTube downloader with configuration"""
        # Load config using centralized loader
        self.config = load_config()
            
        # Get YouTube downloader specific config
        self.yt_config = self.config['processing'].get('downloader', {})
        logger.debug(f"Loaded YouTube downloader config: {self.yt_config}")
        
        # Set up temp directory for downloads
        self.temp_dir = Path("/tmp/youtube_downloads")
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Using temp directory: {self.temp_dir}")
        
        # Store S3 config for fresh connections
        self.s3_config_dict = self.config['storage']['s3']
        
        # Set up authentication and user agent
        self.cookie_file = None
        self.effective_user_agent = user_agent  # Default to passed user_agent
        
        if cookie_profile:
            self.cookie_file = Path('data/cookies') / f"{cookie_profile}_cookies.txt"
            if not self.cookie_file.exists():
                logger.warning(f"Cookie file not found: {self.cookie_file}")
                self.cookie_file = None
            else:
                logger.info(f"Using cookie file: {self.cookie_file}")
                # Get user agent from cookie profile config (overrides passed user_agent)
                cookie_behavior = self.config['processing'].get('youtube_downloader', {}).get('cookie_behavior', {})
                for profile in cookie_behavior.get('profiles', []):
                    if profile.get('name') == cookie_profile:
                        profile_ua = profile.get('user_agent')
                        if profile_ua:
                            self.effective_user_agent = profile_ua
                            logger.info(f"Using user agent from profile: {self.effective_user_agent}")
                        break
        
        # Download settings
        self.download_retries = self.yt_config.get('download_retries', 3)
        self.socket_timeout = self.yt_config.get('socket_timeout', 30)
        # Limit to 360p for bandwidth/storage efficiency
        # Prefer combined formats (18) over separate video+audio to handle flaky networks better
        self.video_format = self.yt_config.get('video_format', 'best[height<=360][ext=mp4]/18/best[height<=360]/best')
        
        # Get and parse throttle rate
        self.throttle_rate_str = self.yt_config.get('throttle_rate')
        self.throttle_rate_bytes = self._parse_rate_limit(self.throttle_rate_str)
        
        log_msg = f"Download settings: retries={self.download_retries}, timeout={self.socket_timeout}s, format={self.video_format}"
        if self.throttle_rate_bytes:
            log_msg += f", throttle_rate={self.throttle_rate_str} ({self.throttle_rate_bytes} B/s)"
        logger.debug(log_msg)

        # Set proxy
        self.proxy = proxy

    def _get_fresh_s3_storage(self) -> S3Storage:
        """Get a fresh S3 storage connection to avoid stale connections."""
        return create_s3_storage_from_config(self.s3_config_dict)

    def _needs_authentication(self, error_message: str) -> bool:
        """Check if error indicates authentication is needed."""
        auth_indicators = [
            "Sign in to confirm your age",
            "This video is private",
            "Members-only content",
            "Join this channel to get access",
            "This video requires payment",
            "Use --cookies"
        ]
        return any(msg in error_message for msg in auth_indicators)
    
    def _handle_download_error(self, video_url: str, error_message: str) -> Dict:
        """Handle download errors and return appropriate error result."""
        logger.error(f"Download failed for {video_url}: {error_message}")
        
        # Authentication errors
        if "Use --cookies" in error_message or "The following content is not available" in error_message:
            return create_error_result(
                error_code=ErrorCode.YOUTUBE_AUTH,
                error_message=error_message,
                permanent=False
            )
        
        # Members-only content
        if "Join this channel to get access" in error_message or "Members-only content" in error_message:
            return create_error_result(
                error_code=ErrorCode.MEMBERS_ONLY,
                error_message=error_message,
                permanent=True,
                skip_state_audit=True
            )
        
        # Age-restricted content
        if "Sign in to confirm your age" in error_message:
            return create_error_result(
                error_code=ErrorCode.AGE_RESTRICTED,
                error_message=error_message,
                permanent=True,
                skip_state_audit=True
            )
        
        # Video unavailable (removed, private, etc.)
        unavailable_indicators = [
            "Video unavailable",
            "has been removed by the uploader",
            "This video is private",
            "This video is no longer available",
            "This video has been removed for violating"
        ]
        if any(msg in error_message for msg in unavailable_indicators):
            return create_error_result(
                error_code=ErrorCode.VIDEO_UNAVAILABLE,
                error_message=error_message,
                permanent=True,
                skip_state_audit=True
            )
        
        # Geo-restricted
        if "This video is not available in your country" in error_message:
            return create_error_result(
                error_code=ErrorCode.ACCESS_DENIED,
                error_message=error_message,
                permanent=True,
                skip_state_audit=True
            )
        
        # Corrupt/empty download
        if "0 bytes read" in error_message:
            return create_error_result(
                error_code=ErrorCode.CORRUPT_MEDIA,
                error_message=error_message,
                permanent=True,
                skip_state_audit=True
            )
        
        # Live event not started - treat as permanent since we don't know when it will start
        if "This live event will begin" in error_message:
            return create_error_result(
                error_code=ErrorCode.VIDEO_UNAVAILABLE,
                error_message=error_message,
                permanent=True,
                skip_state_audit=True
            )
        
        # Live stream recording not available - permanent error
        if "This live stream recording is not available" in error_message:
            return create_error_result(
                error_code=ErrorCode.VIDEO_UNAVAILABLE,
                error_message=error_message,
                permanent=True,
                skip_state_audit=True
            )
        
        # Video offline - permanent error
        if "Offline." in error_message:
            return create_error_result(
                error_code=ErrorCode.VIDEO_UNAVAILABLE,
                error_message=error_message,
                permanent=True,
                skip_state_audit=True
            )
        
        # Live streams or videos with no formats found (typically live content)
        if "No video formats found" in error_message:
            return create_error_result(
                error_code=ErrorCode.LIVE_STREAM,
                error_message=error_message,
                permanent=True,
                skip_state_audit=True
            )
        
        # Default: treat as transient error
        return create_error_result(
            error_code=ErrorCode.NETWORK_ERROR,
            error_message=error_message,
            permanent=False
        )
    
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
        rate_str_parsed = str(rate_str) # Convert to string first
        
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
        else: # No unit or invalid unit
            return int(value)

    async def _validate_video_file(self, file_path: str, min_size_bytes: int = 50000) -> bool:
        """Validate that a video file is not corrupted and has reasonable size.

        Args:
            file_path: Path to video file
            min_size_bytes: Minimum file size in bytes (default 50KB)

        Returns:
            True if file is valid, False otherwise
        """
        try:
            import subprocess

            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size < min_size_bytes:
                logger.warning(f"File too small: {file_size} bytes (minimum {min_size_bytes})")
                return False

            # Use ffprobe to validate video structure
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                 '-show_entries', 'stream=duration', '-of', 'default=noprint_wrappers=1:nokey=1',
                 file_path],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0:
                logger.warning(f"ffprobe validation failed: {result.stderr}")
                return False

            # Check if we got a valid duration
            try:
                duration = float(result.stdout.strip())
                if duration <= 0:
                    logger.warning(f"Invalid duration: {duration}")
                    return False

                # Sanity check: file size should be reasonable for the duration
                # Minimum expected bitrate: ~10kbps (1.25 KB/s) for any real video
                # This catches truncated files that have valid headers but missing media data
                min_expected_size = duration * 1250  # 10kbps in bytes
                if file_size < min_expected_size:
                    logger.warning(f"File size ({file_size:,} bytes) too small for duration ({duration:.1f}s). "
                                  f"Expected at least {min_expected_size:,.0f} bytes. File appears truncated.")
                    return False

                logger.debug(f"Video validated: {file_size:,} bytes, {duration:.1f}s duration")
                return True
            except (ValueError, AttributeError):
                logger.warning(f"Could not parse duration from ffprobe output: {result.stdout}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("ffprobe validation timed out")
            return False
        except Exception as e:
            logger.error(f"Error validating video file: {str(e)}")
            return False

    async def _check_content_exists(self, content_id: str) -> bool:
        """Check if content already exists in S3 and validate it.

        If content exists but is corrupted, delete it and return False to trigger re-download.
        """
        try:
            # Use fresh S3 connection
            fresh_s3 = self._get_fresh_s3_storage()

            # Check for meta.json
            meta_key = f"content/{content_id}/meta.json"
            meta_exists = fresh_s3.file_exists(meta_key)

            # Check for source.mp4
            source_key = f"content/{content_id}/source.mp4"
            source_exists = fresh_s3.file_exists(source_key)

            if not (meta_exists and source_exists):
                logger.debug(f"Content does not exist in S3: meta={meta_exists}, source={source_exists}")
                return False

            # Validate the source file
            logger.info(f"Content exists in S3, validating file integrity for {content_id}")

            # Check file size via head_object
            try:
                response = fresh_s3._client.head_object(Bucket=fresh_s3.config.bucket_name, Key=source_key)
                file_size = response.get('ContentLength', 0)

                # Minimum reasonable size for video: 50KB
                min_size = 50000
                if file_size < min_size:
                    logger.warning(f"S3 file too small ({file_size:,} bytes < {min_size:,}), deleting corrupted file")
                    fresh_s3.delete_file(source_key)
                    fresh_s3.delete_file(meta_key)
                    return False

                logger.debug(f"S3 file size validation passed: {file_size:,} bytes")

            except Exception as e:
                logger.error(f"Error checking S3 file size: {str(e)}")
                # If we can't check size, assume it needs re-download
                return False

            # Optionally download and validate with ffprobe for thorough check
            # (only for files under 5MB to avoid expensive downloads)
            if file_size < 5 * 1024 * 1024:
                import tempfile
                logger.info(f"File under 5MB, performing deep validation with ffprobe")

                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
                    tmp_path = tmp.name

                try:
                    if fresh_s3.download_file(source_key, tmp_path):
                        is_valid = await self._validate_video_file(tmp_path)

                        if not is_valid:
                            logger.warning(f"S3 file failed validation, deleting corrupted file")
                            fresh_s3.delete_file(source_key)
                            fresh_s3.delete_file(meta_key)
                            os.unlink(tmp_path)
                            return False

                        logger.info(f"S3 file passed deep validation")
                        os.unlink(tmp_path)
                    else:
                        logger.warning(f"Failed to download file for validation")
                        return False

                except Exception as e:
                    logger.error(f"Error during deep validation: {str(e)}")
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                    # If validation fails, delete the corrupted file
                    fresh_s3.delete_file(source_key)
                    fresh_s3.delete_file(meta_key)
                    return False

            logger.info(f"Content {content_id} exists and is valid in S3")
            return True

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
                'quiet': False,  # Enable output to see what's happening
                'no_warnings': False,  # Show warnings
                'extract_flat': False,
                'nocheckcertificate': True,
                'ignoreerrors': False,
                'no_color': True,
                'socket_timeout': 60,  # Longer timeout for slow connections
                'retries': 20,  # More retries for flaky networks
                'fragment_retries': 20,  # Retry individual fragments
                'file_access_retries': 5,  # Retry file operations
                'http_chunk_size': 1048576,  # 1MB chunks to handle connection drops better
                'noresizebuffer': True,  # Don't resize buffer (more stable)
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
            if self.effective_user_agent:
                ydl_opts['user_agent'] = self.effective_user_agent
                logger.debug(f"Using User-Agent: {self.effective_user_agent}")

            # Run download
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                error_message = None
                retry_with_cookies = False
                retry_with_default_format = False

                try:
                    ydl.download([video_url])
                except DownloadError as e:
                    error_message = str(e)
                    # Check if we should retry with cookies
                    retry_with_cookies = self.cookie_file and self._needs_authentication(error_message)
                    # Check if format issue - retry with default format
                    # This includes 404 errors which often occur when specific formats are unavailable
                    if ("Requested format is not available" in error_message or
                        "No video formats found" in error_message or
                        "HTTP Error 404" in error_message):
                        retry_with_default_format = True

                # Retry with cookies if needed
                if retry_with_cookies and not retry_with_default_format:
                    logger.info(f"Authentication required for {video_url}. Retrying with cookies...")
                    ydl_opts_with_cookies = dict(ydl_opts)  # Create a fresh copy
                    ydl_opts_with_cookies['cookiefile'] = str(self.cookie_file)

                    try:
                        with yt_dlp.YoutubeDL(ydl_opts_with_cookies) as ydl_retry:
                            ydl_retry.download([video_url])
                            logger.info(f"Successfully downloaded {video_url} with cookies")
                            error_message = None  # Clear error on success
                    except DownloadError as retry_error:
                        error_message = str(retry_error)
                        if "The following content is not available on this app" in error_message:
                            logger.error(f"Cookies appear to be invalid: {error_message}")
                        # Check if format issue after cookie retry
                        if ("Requested format is not available" in error_message or
                            "No video formats found" in error_message or
                            "HTTP Error 404" in error_message):
                            retry_with_default_format = True

                # Retry with default format as last resort
                if retry_with_default_format:
                    logger.warning(f"Format '{self.video_format}' not available. Retrying with default format...")
                    ydl_opts_default = dict(ydl_opts)
                    # Remove format specification to use yt-dlp's default selection
                    del ydl_opts_default['format']

                    try:
                        with yt_dlp.YoutubeDL(ydl_opts_default) as ydl_default:
                            ydl_default.download([video_url])
                            logger.info(f"Successfully downloaded {video_url} with default format")
                            error_message = None  # Clear error on success
                    except DownloadError as default_error:
                        error_message = str(default_error)
                        logger.error(f"Failed even with default format: {error_message}")

                # Handle any remaining errors
                if error_message:
                    return self._handle_download_error(video_url, error_message)
            
            # Check download success and validate file
            if output_path.exists() and output_path.stat().st_size > 0:
                logger.info(f"Downloaded video to {output_path} ({output_path.stat().st_size} bytes), validating...")

                # Validate the downloaded file
                is_valid = await self._validate_video_file(str(output_path))
                if not is_valid:
                    logger.error("Downloaded file failed validation (corrupted or too small)")
                    # Clean up invalid file
                    if output_path.exists():
                        output_path.unlink()
                    # If download "succeeded" but gave us corrupted/tiny file, it's a permanent issue
                    # (video unavailable, geo-restricted, or YouTube returned error page)
                    return create_error_result(
                        error_code=ErrorCode.CORRUPT_MEDIA,
                        error_message="Downloaded file failed validation - video may be unavailable or restricted",
                        permanent=True,
                        skip_state_audit=True
                    )

                logger.info(f"Successfully downloaded and validated video")
                return {'status': 'success'}
            else:
                logger.error("Download failed - file not created or empty")
                return create_error_result(
                    error_code=ErrorCode.PROCESS_FAILED,
                    error_message="Download failed - file not created or empty",
                    permanent=True,
                    skip_state_audit=True
                )
                
        except Exception as e:
            logger.error(f"Unexpected error in _download_video: {str(e)}")
            return create_error_result(
                error_code=ErrorCode.UNKNOWN_ERROR,
                error_message=str(e),
                permanent=False
            )

    async def _upload_to_s3(self, content_id: str, file_path: str) -> bool:
        """Upload file to S3"""
        try:    
            logger.debug(f"Starting S3 upload for content {content_id}")
            
            # Use fresh S3 connection
            fresh_s3 = self._get_fresh_s3_storage()
            
            # Create metadata
            meta_data = {
                'platform': 'youtube',
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
            meta_success = fresh_s3.upload_file(str(meta_path), meta_key)
            if not meta_success:
                logger.error(f"Failed to upload metadata for {content_id}")
                return False
            logger.debug(f"Successfully uploaded metadata to {meta_key}")

            # Clean up meta.json
            meta_path.unlink()
            logger.debug("Cleaned up temporary metadata file")

            # Upload source file
            source_key = f"content/{content_id}/source.mp4"
            source_success = fresh_s3.upload_file(file_path, source_key)
            if not source_success:
                logger.error(f"Failed to upload source file for {content_id}")
                return False
            logger.debug(f"Successfully uploaded source file to {source_key}")

            return True

        except Exception as e:
            logger.error(f"Error uploading to S3: {str(e)}")
            return False

    def _update_content_state(self, content_id: str, is_downloaded: bool, blocked_download: bool = False) -> None:
        """Update content download state in database.

        Args:
            content_id: Content ID
            is_downloaded: Whether content is successfully downloaded
            blocked_download: Whether download should be blocked
        """
        try:
            from src.database.session import get_session
            from src.database.models import Content
            from datetime import datetime, timezone

            with get_session() as session:
                content = session.query(Content).filter(Content.content_id == content_id).first()
                if content:
                    content.is_downloaded = is_downloaded
                    content.blocked_download = blocked_download
                    content.last_updated = datetime.now(timezone.utc)
                    session.commit()
                    logger.info(f"Updated content state: is_downloaded={is_downloaded}, blocked_download={blocked_download}")
                else:
                    logger.warning(f"Content {content_id} not found in database")

        except Exception as e:
            logger.error(f"Error updating content state: {str(e)}")

    async def process_content(self, content_id: str) -> Dict:
        """Process a single YouTube video download"""
        try:
            # Check if content already exists in S3 (will delete and reset if corrupted)
            content_exists = await self._check_content_exists(content_id)

            if content_exists:
                logger.info(f"Content {content_id} already exists and is valid in S3, skipping download")
                return create_skipped_result(
                    reason='Content already exists in S3',
                    skip_wait_time=True
                )

            # If we reach here and file was corrupted, _check_content_exists already:
            # 1. Deleted the corrupted file from S3
            # 2. We need to ensure is_downloaded = False before attempting download
            logger.info(f"Ensuring is_downloaded=False before download attempt")
            self._update_content_state(content_id, is_downloaded=False, blocked_download=False)

            # Download video
            video_url = f"https://www.youtube.com/watch?v={content_id}"
            temp_path = self.temp_dir / f"{content_id}.mp4"
            download_result = await self._download_video(video_url, temp_path)

            if download_result['status'] != 'success':
                # If download failed with permanent error, update database
                if download_result.get('permanent'):
                    logger.warning(f"Permanent download failure for {content_id}, setting blocked_download=True")
                    self._update_content_state(content_id, is_downloaded=False, blocked_download=True)

                # Error already logged in _download_video or _handle_download_error
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
            return create_error_result(
                error_code=ErrorCode.UNKNOWN_ERROR,
                error_message=str(e),
                permanent=False
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download YouTube video')
    parser.add_argument('--content', required=True, help='Content ID to process')
    parser.add_argument('--cookies', help='Cookie profile name to use (e.g., "tom")')
    parser.add_argument('--proxy', help='Proxy server address (e.g., http://user:pass@host:port)')
    parser.add_argument('--user-agent', help='Custom User-Agent string to use')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    async def main():
        downloader = YouTubeDownloader(
            cookie_profile=args.cookies, 
            proxy=args.proxy, 
            user_agent=args.user_agent
        )
        result = await downloader.process_content(args.content)
        # Print ONLY the final JSON result to stdout
        print(json.dumps(result, cls=DateTimeEncoder))
    
    asyncio.run(main()) 