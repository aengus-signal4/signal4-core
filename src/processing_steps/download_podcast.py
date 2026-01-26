#!/usr/bin/env python3
# Centralized environment setup (must be before other imports)
from src.utils.env_setup import setup_env
setup_env()

import sys
import os
from pathlib import Path

from src.utils.paths import get_project_root
from src.utils.config import load_config
import asyncio
import logging
import json
import yaml
from typing import Dict, List, Optional
from datetime import datetime
import tempfile
import shutil
import argparse
import uuid
import aiohttp
from urllib.parse import urlparse, urljoin
from fake_useragent import UserAgent
import socket
from datetime import timezone
import yt_dlp
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy.exc import NoResultFound
from src.database.models import Content
from src.database.session import get_session
import requests
import ssl
import warnings
import random

# Add the project root to Python path
sys.path.append(str(get_project_root()))

# Import required modules
from src.utils.logger import setup_worker_logger
from src.storage.s3_utils import S3StorageConfig, S3Storage
from src.storage.content_storage import ContentStorageManager
from src.utils.media_utils import get_audio_duration # Import shared duration function
from src.utils.error_codes import ErrorCode, create_error_result, create_success_result, create_skipped_result
import subprocess # Keep subprocess for the fallback delete

logger = setup_worker_logger('download_podcast')

class PodcastDownloader:
    def __init__(self):
        """Initialize the podcast downloader"""
        # Load config
        self.config = load_config()
            
        # Initialize S3 storage
        s3_config = S3StorageConfig(
            endpoint_url=self.config['storage']['s3']['endpoint_url'],
            access_key=self.config['storage']['s3']['access_key'],
            secret_key=self.config['storage']['s3']['secret_key'],
            bucket_name=self.config['storage']['s3']['bucket_name'],
            use_ssl=self.config['storage']['s3']['use_ssl']
        )
        self.s3_storage = S3Storage(s3_config)
        self.storage_manager = ContentStorageManager(self.s3_storage)
        
        # Set up temp directory
        self.temp_dir = Path("/tmp/podcast_downloads")
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize UserAgent for rotation
        try:
            self.ua = UserAgent()
        except Exception as e:
            logger.warning(f"Could not initialize UserAgent, using fallback: {str(e)}")
            self.ua = None
            
        # Common browser headers
        self.base_headers = {
            'Accept': 'audio/webm,audio/ogg,audio/wav,audio/*;q=0.9,application/ogg;q=0.7,video/*;q=0.6,*/*;q=0.5',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Sec-Fetch-Dest': 'audio',
            'Sec-Fetch-Mode': 'no-cors',
            'Sec-Fetch-Site': 'cross-site',
            'Pragma': 'no-cache',
            'Cache-Control': 'no-cache',
            'TE': 'trailers',
        }

    def _get_headers(self, url: str) -> Dict:
        """Generate realistic browser headers for a request"""
        parsed_url = urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}" if parsed_url.netloc else url
        
        headers = {
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Sec-Fetch-Dest': 'audio',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'cross-site',
            'Sec-Fetch-User': '?1',
            'Pragma': 'no-cache',
            'Cache-Control': 'no-cache',
            'Range': 'bytes=0-',
            'Referer': base_url,
            'Origin': base_url,
            'Host': parsed_url.netloc if parsed_url.netloc else '',
            'User-Agent': self.ua.random if self.ua else 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            'DNT': '1',
        }
        
        if not headers['Host']:
            del headers['Host']

        return headers

    async def _download_with_retry(self, url: str, output_path: Path, session: aiohttp.ClientSession) -> Dict:
        """Download a single audio file with retry logic, including HEAD check and redirects."""
        max_retries = 1
        retry_delay = 2
        head_timeout = 10
        get_timeout = 60

        for attempt in range(max_retries):
            try:
                headers = self._get_headers(url)
                current_url = url
                redirect_count = 0
                max_redirects = 10
                head_failed = False
                last_error = None

                logger.debug(f"Attempt {attempt + 1}: Downloading {current_url} to {output_path}")

                while redirect_count < max_redirects:
                    try:
                        if not head_failed:
                            try:
                                async with session.head(current_url, headers=headers, allow_redirects=False, timeout=head_timeout) as head_response:
                                    logger.debug(f"HEAD request to {current_url} returned status {head_response.status}")
                                    if head_response.status in (301, 302, 303, 307, 308):
                                        location = head_response.headers.get('Location')
                                        if not location:
                                             raise aiohttp.ClientError("Redirect location header missing")

                                        # Check for error page redirects
                                        if 'error' in location.lower() or 'not-found' in location.lower():
                                            logger.error(f"HEAD redirect to error page detected: {location}")
                                            return create_error_result(ErrorCode.NOT_FOUND, f'Content no longer available (redirects to error page)', permanent=True, skip_state_audit=True)

                                        current_url = urljoin(current_url, location)
                                        redirect_count += 1
                                        headers = self._get_headers(current_url)
                                        logger.debug(f"Following redirect to: {current_url}")
                                        continue
                                    elif head_response.status == 403:
                                        logger.warning(f"HEAD request returned 403 Forbidden for {current_url}. Will try GET request and yt-dlp fallback.")
                                        head_failed = True
                                    elif head_response.status == 404:
                                        logger.error(f"HEAD request returned 404 Not Found for {current_url}")
                                        return create_error_result(ErrorCode.NOT_FOUND, 'Resource not found (404)', permanent=True)
                                    elif head_response.status >= 400:
                                        logger.warning(f"HEAD request failed with status {head_response.status}. Proceeding to GET.")
                                        head_failed = True
                            except asyncio.TimeoutError:
                                logger.warning(f"HEAD request timed out for {current_url}. Proceeding to GET.")
                                head_failed = True
                            except aiohttp.ClientError as head_err:
                                logger.warning(f"HEAD request failed for {current_url}: {head_err}. Proceeding to GET.")
                                head_failed = True

                        # Skip preliminary validation as it can interfere with actual download
                        # Some servers don't allow multiple requests for the same resource 
                        
                        # Now, attempt the actual streaming download
                        async with session.get(current_url, headers=headers, allow_redirects=False, timeout=get_timeout) as response:
                            logger.debug(f"GET request to {current_url} returned status {response.status}")
                            if response.status in (301, 302, 303, 307, 308):
                                location = response.headers.get('Location')
                                if not location:
                                     raise aiohttp.ClientError("Redirect location header missing")

                                # Check for error page redirects indicating content unavailable
                                if 'error' in location.lower() or 'not-found' in location.lower():
                                    logger.error(f"Redirect to error page detected: {location}")
                                    return create_error_result(ErrorCode.NOT_FOUND, f'Content no longer available (redirects to error page)', permanent=True, skip_state_audit=True)

                                current_url = urljoin(current_url, location)
                                redirect_count += 1
                                headers = self._get_headers(current_url)
                                logger.debug(f"Following redirect (from GET) to: {current_url}")
                                continue
                            elif response.status in (200, 206):
                                total_size = int(response.headers.get('Content-Length', 0))
                                downloaded_size = 0
                                logger.debug(f"Starting download (expected size: {total_size} bytes)")

                                import aiofiles
                                async with aiofiles.open(output_path, 'wb') as f:
                                    async for chunk in response.content.iter_chunked(8192):
                                        if not chunk:
                                            break
                                        await f.write(chunk)
                                        downloaded_size += len(chunk)

                                logger.info(f"Finished downloading {downloaded_size} bytes to {output_path}")

                                if not output_path.exists() or downloaded_size == 0:
                                     logger.error("Download finished but file is missing or empty.")
                                     return create_error_result(ErrorCode.EMPTY_RESULT, 'File missing or empty after download', permanent=True, skip_state_audit=True)

                                if total_size > 0 and downloaded_size < total_size:
                                    logger.warning(f"Incomplete download: Expected {total_size} bytes, got {downloaded_size} bytes.")

                                return create_success_result()
                            elif response.status in (403, 429, 401, 503):
                                error_text = await response.text()
                                logger.warning(f"Download failed with status {response.status}. Will attempt yt-dlp fallback. Reason: {error_text[:200]}")
                                return await self._try_ytdlp_download(url, output_path)
                            elif response.status == 400:
                                error_text = await response.text()
                                # Check for known permanent failure patterns (e.g., RSS disabled)
                                if "disabled" in error_text.lower() or "ValidationError" in error_text:
                                    logger.error(f"HTTP 400 with permanent failure indicator: {error_text[:200]}")
                                    return create_error_result(ErrorCode.FEED_DISABLED, f"HTTP 400: content/feed disabled", permanent=True)
                                # Otherwise try yt-dlp fallback
                                logger.warning(f"Download failed with status 400. Will attempt yt-dlp fallback. Reason: {error_text[:200]}")
                                return await self._try_ytdlp_download(url, output_path)
                            elif response.status >= 400:
                                error_text = await response.text()
                                logger.error(f"Download failed with status {response.status}: {error_text[:200]}")
                                last_error = f"HTTP status {response.status}"
                                break

                    except asyncio.TimeoutError:
                         logger.error(f"Request timed out for {current_url}")
                         last_error = "Timeout"
                         break
                    except aiohttp.ClientError as e:
                         logger.error(f"Client error during download from {current_url}: {e}")
                         last_error = f"ClientError: {e}"
                         break

                if redirect_count >= max_redirects:
                    logger.error(f"Too many redirects encountered starting from {url}. Attempting yt-dlp fallback.")
                    return await self._try_ytdlp_download(url, output_path)

                if last_error:
                     logger.warning(f"Aiohttp download failed: {last_error}. Attempting yt-dlp fallback.")
                     return await self._try_ytdlp_download(url, output_path)

            except Exception as e:
                logger.error(f"Unexpected error during download attempt {attempt + 1}: {e}", exc_info=True)
                logger.warning(f"Unexpected error encountered. Attempting yt-dlp fallback.")
                return await self._try_ytdlp_download(url, output_path)

        logger.error(f"All download attempts failed for {url}. Final attempt with yt-dlp.")
        return await self._try_ytdlp_download(url, output_path)

    async def _try_requests_download(self, url: str, output_path: Path) -> Dict:
        """Attempt download using requests library with permissive SSL as last resort"""
        logger.info(f"Attempting download with requests library (permissive SSL) for URL: {url}")

        try:
            # Suppress SSL warnings since we're intentionally using insecure connections
            warnings.filterwarnings('ignore', message='Unverified HTTPS request')

            # Full browser-like headers to avoid 429s
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache',
                'Sec-Ch-Ua': '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
                'Sec-Ch-Ua-Mobile': '?0',
                'Sec-Ch-Ua-Platform': '"macOS"',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none',
                'Sec-Fetch-User': '?1',
                'Upgrade-Insecure-Requests': '1',
                'DNT': '1',
            }

            # First, try to extract the final destination URL by parsing known redirect patterns
            # Many podcast CDNs use predictable URL structures
            target_url = url

            # Check if this is a tracking redirect chain to megaphone.fm or similar
            if 'traffic.megaphone.fm' in url or 'megaphone.fm' in url:
                # Extract the megaphone.fm part directly
                import re
                match = re.search(r'(traffic\.megaphone\.fm/[^?]+\?[^&\s]+)', url)
                if match:
                    target_url = f"https://{match.group(1)}"
                    logger.info(f"Extracted direct megaphone URL: {target_url}")

            # Try the direct/extracted URL first
            try:
                response = requests.get(
                    target_url,
                    headers=headers,
                    stream=True,
                    timeout=60,
                    verify=False,
                    allow_redirects=True
                )
                response.raise_for_status()
            except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
                # If direct URL fails and we didn't extract it, try full chain
                if target_url != url:
                    logger.warning(f"Direct URL failed with {e}, trying full redirect chain")
                    response = requests.get(
                        url,
                        headers=headers,
                        stream=True,
                        timeout=60,
                        verify=False,
                        allow_redirects=True
                    )
                    response.raise_for_status()
                else:
                    raise

            logger.info(f"Successfully connected via requests (status {response.status_code})")
            logger.debug(f"Final URL after redirects: {response.url}")

            content_length = response.headers.get('Content-Length')
            if content_length:
                logger.info(f"Downloading {int(content_length) / (1024*1024):.2f} MB")

            # Download the file
            downloaded = 0
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if downloaded % (10 * 1024 * 1024) < 8192:  # Log every 10MB
                            logger.debug(f"Downloaded: {downloaded / (1024*1024):.1f} MB")

            logger.info(f"Successfully downloaded {downloaded / (1024*1024):.2f} MB via requests")

            if output_path.exists() and output_path.stat().st_size > 0:
                return create_success_result()
            else:
                return create_error_result(ErrorCode.EMPTY_RESULT, 'Download completed but file is empty', permanent=True, skip_state_audit=True)

        except requests.exceptions.SSLError as e:
            logger.error(f"Requests SSL error (even with verify=False): {e}")
            return create_error_result(ErrorCode.NOT_FOUND, f"SSL error persists - content unavailable", permanent=True)
        except requests.exceptions.RequestException as e:
            logger.error(f"Requests library failed: {e}")
            return create_error_result(ErrorCode.PROCESS_FAILED, f"Requests download failed: {e}", permanent=True, skip_state_audit=True)
        except Exception as e:
            logger.error(f"Unexpected error in requests download: {e}", exc_info=True)
            return create_error_result(ErrorCode.UNKNOWN_ERROR, f"Requests download error: {e}", permanent=True, skip_state_audit=True)

    def _extract_ytdlp_error(self, stderr_text: str) -> str:
        """Extract just the ERROR lines from yt-dlp output, filtering out debug noise."""
        error_lines = []
        for line in stderr_text.split('\n'):
            line = line.strip()
            if line.startswith('ERROR:'):
                # Remove the ERROR: prefix for cleaner messages
                error_lines.append(line[6:].strip())

        if error_lines:
            return '; '.join(error_lines)

        # Fallback: if no ERROR lines found, look for common error patterns
        for line in stderr_text.split('\n'):
            if 'HTTP Error' in line or 'Unable to' in line:
                return line.strip()

        # Last resort: return truncated stderr
        return stderr_text[:200] + '...' if len(stderr_text) > 200 else stderr_text

    async def _try_ytdlp_download(self, url: str, output_path: Path, retry_attempt: int = 0) -> Dict:
        """Attempt to download using yt-dlp as a fallback method with exponential backoff for 429 errors"""
        # Clean the URL
        url = url.strip()

        # Exponential backoff settings for 429 errors (5min -> 20min -> 60min -> 180min)
        retry_delays = [300, 1200, 3600, 10800]  # 5, 20, 60, 180 minutes in seconds
        max_retries = len(retry_delays)

        logger.info(f"Attempting fallback download using yt-dlp for URL: {url} (attempt {retry_attempt + 1}/{max_retries + 1})")
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # --- NEW: Ensure target path is clean before yt-dlp ---
            if output_path.exists():
                logger.warning(f"Output path {output_path} exists before yt-dlp fallback. Deleting it.")
                try:
                    output_path.unlink()
                except OSError as e:
                    logger.error(f"Failed to delete existing output file {output_path} before yt-dlp: {e}")
                    # If we can't delete it, yt-dlp will likely fail anyway or behave unexpectedly.
                    return create_error_result(ErrorCode.PROCESS_FAILED, f'Failed to delete pre-existing output file for yt-dlp: {e}', permanent=True, skip_state_audit=True)
            # --- END NEW ---

            # Build the yt-dlp command with browser-like headers
            # Use a recent Chrome user agent and full browser headers to avoid 429s
            parsed_url = urlparse(url)
            referer_base = f"{parsed_url.scheme}://{parsed_url.netloc}/"

            # Rotate user agent for each attempt
            user_agents = [
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.2 Safari/605.1.15',
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0',
                'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:133.0) Gecko/20100101 Firefox/133.0',
            ]
            user_agent = user_agents[retry_attempt % len(user_agents)]

            cmd = [
                'yt-dlp',
                '--output', str(output_path.with_suffix('.%(ext)s')),
                '--no-check-certificate',
                '--ignore-errors',
                '--no-color',
                '--verbose',
                '--user-agent', user_agent,
                '--referer', referer_base,
                # Full browser-like headers
                '--add-header', 'Accept:text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
                '--add-header', 'Accept-Language:en-US,en;q=0.9',
                '--add-header', 'Accept-Encoding:gzip, deflate, br',
                '--add-header', 'Cache-Control:no-cache',
                '--add-header', 'Pragma:no-cache',
                '--add-header', 'Sec-Ch-Ua:"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
                '--add-header', 'Sec-Ch-Ua-Mobile:?0',
                '--add-header', 'Sec-Ch-Ua-Platform:"macOS"',
                '--add-header', 'Sec-Fetch-Dest:document',
                '--add-header', 'Sec-Fetch-Mode:navigate',
                '--add-header', 'Sec-Fetch-Site:none',
                '--add-header', 'Sec-Fetch-User:?1',
                '--add-header', 'Upgrade-Insecure-Requests:1',
                '--add-header', 'DNT:1',
                '--no-playlist',
                '--extract-audio',
                '--audio-format', 'mp3',
                '--audio-quality', '0',
                # Add sleep between requests to appear more human-like
                '--sleep-requests', '1',
                url
            ]

            logger.debug(f"Executing yt-dlp command: {' '.join(cmd)}")

            # Run the command
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            # Log the output
            if stdout:
                logger.debug(f"yt-dlp stdout:\n{stdout.decode()}")
            if stderr:
                logger.debug(f"yt-dlp stderr:\n{stderr.decode()}")

            if process.returncode != 0:
                stderr_text = stderr.decode()
                logger.error(f"yt-dlp command failed with return code {process.returncode}")

                # Check for 429 Too Many Requests - retry with exponential backoff
                if "429" in stderr_text and ("Too Many Requests" in stderr_text or "HTTP Error 429" in stderr_text):
                    if retry_attempt < max_retries:
                        # Use predefined delay schedule with jitter
                        delay = retry_delays[retry_attempt]
                        jitter = random.uniform(0, delay * 0.1)  # Add up to 10% jitter
                        total_delay = delay + jitter
                        logger.warning(f"HTTP 429 Too Many Requests - waiting {total_delay/60:.1f} minutes before retry {retry_attempt + 2}/{max_retries + 1}")
                        await asyncio.sleep(total_delay)
                        return await self._try_ytdlp_download(url, output_path, retry_attempt + 1)
                    else:
                        logger.error(f"HTTP 429 Too Many Requests - max retries ({max_retries}) exceeded, blocking content")
                        return create_error_result(
                            ErrorCode.RATE_LIMITED,
                            f"Rate limited (429) after {max_retries + 1} attempts over ~4.4 hours. Content blocked for review.",
                            permanent=True
                        )

                # Check for SSL record layer failure - try requests library as final fallback
                if "[SSL] record layer failure" in stderr_text or "SSLError" in stderr_text:
                    logger.warning(f"yt-dlp SSL error detected - trying requests library with permissive SSL")
                    return await self._try_requests_download(url, output_path)

                # Check for 400 Bad Request - often means content/feed disabled or invalid URL
                if "400" in stderr_text and "Bad Request" in stderr_text:
                    logger.error(f"HTTP 400 Bad Request detected - content likely unavailable or feed disabled")
                    return create_error_result(ErrorCode.FEED_DISABLED, f"HTTP 400 Bad Request: content unavailable", permanent=True)

                # Check for 502 Bad Gateway - typically means content unavailable in redirect chains
                if "502" in stderr_text and "Bad Gateway" in stderr_text:
                    logger.error(f"HTTP 502 Bad Gateway detected - content likely unavailable")
                    return create_error_result(ErrorCode.NOT_FOUND, f"HTTP 502 Bad Gateway: content unavailable", permanent=True)

                # Check for 410 Gone - content permanently deleted
                if "410" in stderr_text and "Gone" in stderr_text:
                    logger.error(f"HTTP 410 Gone detected - content permanently deleted")
                    return create_error_result(ErrorCode.CONTENT_GONE, f"HTTP 410 Gone: content permanently deleted", permanent=True)

                # Before failing, check if a file was actually downloaded despite the non-zero return code
                # This handles cases like ffprobe warnings that cause yt-dlp to exit non-zero
                # but the download itself succeeded
                possible_extensions = ['.mp3', '.m4a', '.aac', '.ogg', '.opus', '.wav']
                base_name = output_path.stem
                for ext in possible_extensions:
                    potential_file = output_path.with_name(base_name + ext)
                    if potential_file.exists() and potential_file.stat().st_size > 0:
                        logger.warning(f"yt-dlp exited with non-zero code but file exists: {potential_file} ({potential_file.stat().st_size} bytes)")
                        logger.info(f"Proceeding with download despite yt-dlp warning: {stderr_text[:200]}")
                        # File exists, fall through to normal success path by breaking out of this block
                        break
                else:
                    # Also check the original output path
                    if output_path.exists() and output_path.stat().st_size > 0:
                        logger.warning(f"yt-dlp exited with non-zero code but file exists at original path: {output_path}")
                    else:
                        # No file found, this is a real failure
                        error_msg = self._extract_ytdlp_error(stderr_text)
                        return create_error_result(ErrorCode.PROCESS_FAILED, f"yt-dlp command failed: {error_msg}", permanent=True, skip_state_audit=True)

            # Check for downloaded file
            downloaded_file = None
            possible_extensions = ['.mp3', '.m4a', '.aac', '.ogg', '.opus', '.wav']
            base_name = output_path.stem

            for ext in possible_extensions:
                potential_file = output_path.with_name(base_name + ext)
                if potential_file.exists() and potential_file.stat().st_size > 0:
                    downloaded_file = potential_file
                    logger.info(f"yt-dlp downloaded file: {downloaded_file}")
                    break

            if not downloaded_file:
                if output_path.exists() and output_path.stat().st_size > 0:
                    downloaded_file = output_path
                    logger.info(f"yt-dlp used the original path: {downloaded_file}")
                else:
                    video_extensions = ['.mp4', '.mkv', '.webm', '.mov']
                    for ext in video_extensions:
                        potential_file = output_path.with_name(base_name + ext)
                        if potential_file.exists() and potential_file.stat().st_size > 0:
                            logger.warning(f"yt-dlp downloaded a video file ({potential_file}) when audio was expected. This might cause issues downstream.")
                            raise Exception("yt-dlp downloaded video instead of audio.")

                    logger.error(f"yt-dlp download finished, but expected output file ({output_path.name} with common audio extensions) not found or empty in {output_path.parent}")
                    raise FileNotFoundError("yt-dlp download failed - output file not found or empty")

            if downloaded_file != output_path:
                logger.debug(f"Renaming {downloaded_file} to {output_path}")
                try:
                    shutil.move(str(downloaded_file), str(output_path))
                except Exception as move_err:
                    logger.error(f"Failed to rename downloaded file {downloaded_file} to {output_path}: {move_err}")
                    raise Exception(f"Failed to prepare output file: {move_err}") from move_err

            if output_path.exists() and output_path.stat().st_size > 0:
                logger.info(f"yt-dlp download successful, final file at: {output_path}")
                return create_success_result()
            else:
                logger.error("Output file check failed after potential rename.")
                raise Exception("Download successful but final output file validation failed.")

        except FileNotFoundError as fnf_err:
            logger.error(f"yt-dlp fallback failed: {str(fnf_err)}")
            return create_error_result(ErrorCode.NOT_FOUND, str(fnf_err), permanent=True, skip_state_audit=True)
        except Exception as e:
            error_detail = str(e)
            if hasattr(e, 'exc_info') and e.exc_info and e.exc_info[1]:
                error_detail = str(e.exc_info[1])
            logger.error(f"yt-dlp fallback failed: {error_detail}", exc_info=False)
            
            # Check for HTTP 403 Forbidden errors
            if "HTTP Error 403: Forbidden" in error_detail or "403: Forbidden" in error_detail:
                logger.error(f"HTTP 403 Forbidden error detected - marking as ACCESS_DENIED for permanent ban")
                return create_error_result(ErrorCode.ACCESS_DENIED, f"HTTP 403 Forbidden: {error_detail}", permanent=True)
            
            return create_error_result(ErrorCode.PROCESS_FAILED, f"yt-dlp download failed: {error_detail}", permanent=True, skip_state_audit=True)

    def _run_ytdlp(self, url: str, ydl_opts: dict) -> None:
        """Run yt-dlp download in a synchronous context"""
        logger.debug(f"Executing yt-dlp download for {url}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                ydl.download([url])
                logger.debug(f"yt-dlp execution finished for {url}")
            except yt_dlp.utils.DownloadError as download_error:
                 logger.error(f"yt-dlp encountered a download error for {url}: {download_error}")
                 raise Exception(f"yt-dlp download error: {download_error}") from download_error
            except Exception as e:
                logger.error(f"Unexpected error during yt-dlp execution for {url}: {e}", exc_info=True)
                raise Exception(f"yt-dlp execution failed: {e}") from e

    async def download_single_episode(self, episode_url: str, content_id: str) -> Dict:
        """Downloads a single podcast episode given its direct URL."""
        episode_temp_dir = self.temp_dir / content_id
        episode_temp_dir.mkdir(parents=True, exist_ok=True)
        # Determine the potential source key (assuming mp3 for now, actual download might use different)
        s3_key_base = f"content/{content_id}/source"
        s3_key_mp3 = f"{s3_key_base}.mp3"

        logger.info(f"Starting download for episode: {episode_url} (content_id: {content_id})")
        
        # Check if URL contains patterns that are known to require yt-dlp
        url_lower = episode_url.lower()
        force_ytdlp = any([
            'cloudfront.net' in url_lower and 'signature=' in url_lower,
            'expires=' in url_lower and 'key-pair-id=' in url_lower,
            'louder-with-crowder' in url_lower,
            # Add more patterns here as needed
        ])
        
        if force_ytdlp:
            logger.info(f"URL pattern suggests direct download may fail, using yt-dlp directly: {episode_url}")

        # Check if *any* source file exists first (more robust than assuming mp3)
        # (We rely on the download logic to determine the final extension later)
        source_key_to_check = None
        extensions = ['.mp3', '.mp4', '.wav', '.m4a', '.webm', '.mkv', '.avi', '.mov']
        for ext in extensions:
            potential_key = f"{s3_key_base}{ext}"
            if self.s3_storage.file_exists(potential_key):
                source_key_to_check = potential_key
                logger.info(f"Found existing source candidate: {source_key_to_check}")
                break
        # Fallback check without extension
        if not source_key_to_check and self.s3_storage.file_exists(s3_key_base):
            source_key_to_check = s3_key_base
            logger.info(f"Found existing source candidate (no ext): {source_key_to_check}")

        if source_key_to_check:
            # Verify existing S3 file is valid audio before skipping
            temp_verify_path = episode_temp_dir / f"verify_{Path(source_key_to_check).name}"
            is_valid = False
            try:
                logger.debug(f"Attempting to download {source_key_to_check} for validation to {temp_verify_path}")
                if not self.s3_storage.download_file(source_key_to_check, str(temp_verify_path)):
                    logger.warning(f"Could not download existing S3 file {source_key_to_check} for validation")
                    # Assume it\'s corrupted or inaccessible, delete and retry download
                    self.s3_storage.delete_file(source_key_to_check)
                    logger.info(f"Deleted potentially invalid S3 file {source_key_to_check}, will re-download")
                    return create_error_result(ErrorCode.S3_CONNECTION_ERROR, 'Could not download existing S3 file for validation')

                # Use the shared get_audio_duration function for validation
                logger.debug(f"Validating downloaded file {temp_verify_path} using get_audio_duration")
                duration = get_audio_duration(str(temp_verify_path))
                
                if duration is not None and duration > 1.0: # Check if duration is valid and longer than 1 second
                    is_valid = True
                    logger.info(f"Existing file {source_key_to_check} validation successful (Duration: {duration:.2f}s). Skipping download.")
                else:
                    logger.warning(f"Existing file {source_key_to_check} validation failed. get_audio_duration returned: {duration}")

            except Exception as e:
                logger.warning(f"Error during validation check for {source_key_to_check}: {e}")
                is_valid = False # Treat any validation exception as invalid
            finally:
                # Clean up the temporary verification file
                if temp_verify_path.exists():
                    try:
                        temp_verify_path.unlink()
                    except OSError as e:
                        logger.warning(f"Failed to delete temp verification file {temp_verify_path}: {e}")

            if not is_valid:
                # If validation failed, delete the file from S3 and force re-download
                logger.warning(f"Validation failed for {source_key_to_check}. Deleting and triggering re-download.")
                self.s3_storage.delete_file(source_key_to_check)
                # Don\'t return retry here, just fall through to the main download logic
                logger.info(f"Deleted invalid S3 file {source_key_to_check}, proceeding to download.")
            else:
                # Validation succeeded, skip the download
                logger.info(f"Source audio already exists and is valid in S3 for {content_id} at {source_key_to_check}, skipping download.")
                if episode_temp_dir.exists():
                    try:
                        shutil.rmtree(episode_temp_dir)
                    except Exception as cleanup_err:
                         logger.warning(f"Could not clean up temporary directory {episode_temp_dir} after skip: {cleanup_err}")
                return create_skipped_result(
                    'Source audio file already exists in S3 and is valid.',
                    skip_wait_time=True,
                    data={'s3_key': source_key_to_check}
                )
        
        # --- If no valid existing file found, proceed with download ---
        logger.info(f"No valid existing source file found for {content_id}. Proceeding with download from URL.")
        
        # Define the *intended* output path and S3 key for the new download
        # We use mp3 as the target format for yt-dlp fallback and general consistency
        temp_output_filename = f"{content_id}.mp3" 
        temp_output_path = episode_temp_dir / temp_output_filename
        s3_key_target = f"{s3_key_base}.mp3" # Target S3 key is .mp3

        download_result = {'status': 'error', 'error': 'Download did not initiate'}
        try:
            # If we detected patterns that require yt-dlp, skip straight to it
            if force_ytdlp:
                download_result = await self._try_ytdlp_download(episode_url, temp_output_path)
            else:
                async with aiohttp.ClientSession() as session:
                    download_result = await self._download_with_retry(episode_url, temp_output_path, session)

            if download_result['status'] == 'completed':
                logger.info(f"Successfully downloaded episode to {temp_output_path}")
                if not temp_output_path.exists() or temp_output_path.stat().st_size == 0:
                     logger.error(f"Downloaded file {temp_output_path} is missing or empty after successful status.")
                     return create_error_result(ErrorCode.EMPTY_RESULT, 'Downloaded file missing or empty', permanent=True, skip_state_audit=True)

                # Optional: Add a final validation check on the newly downloaded file
                logger.debug(f"Validating newly downloaded file: {temp_output_path}")
                final_duration = get_audio_duration(str(temp_output_path))
                logger.debug(f"Audio duration validation result: {final_duration}")
                if final_duration is None or final_duration <= 1.0:
                    logger.error(f"Newly downloaded file {temp_output_path} failed validation (duration: {final_duration}). Not uploading.")
                    return create_error_result(ErrorCode.CORRUPT_MEDIA, 'Downloaded file failed validation', permanent=True, skip_state_audit=True)
                logger.info(f"Newly downloaded file validation successful (Duration: {final_duration:.2f}s). Proceeding with upload.")

                logger.info(f"Uploading {temp_output_path} to S3 at {s3_key_target}")
                if self.s3_storage.upload_file(str(temp_output_path), s3_key_target):
                    logger.info(f"Successfully uploaded {s3_key_target} to S3.")
                    return create_success_result(
                        data={'s3_key': s3_key_target, 'local_path': str(temp_output_path)},
                        message='Episode downloaded and uploaded successfully.'
                    )
                else:
                    logger.error(f"Failed to upload {temp_output_path} to S3.")
                    return create_error_result(ErrorCode.S3_CONNECTION_ERROR, 'S3 upload failed')
            else:
                logger.error(f"Failed to download episode {episode_url}. Reason: {download_result.get('error', 'Unknown')}")
                logger.debug(f"Download result was: {download_result}")
                # If the error is due to a bad URL, return a clear error for pipeline_manager
                error_msg = download_result.get('error', '')
                if (
                    'Unsupported URL' in error_msg or
                    'not found' in error_msg.lower() or
                    '404' in error_msg or
                    'Bad URL' in error_msg
                ):
                    return create_error_result(ErrorCode.BAD_URL, 'Bad URL', permanent=True)
                return download_result

        except Exception as e:
            logger.error(f"Unexpected error during download process for {episode_url}: {str(e)}", exc_info=True)
            return create_error_result(ErrorCode.UNKNOWN_ERROR, f'Unexpected download error: {str(e)}', permanent=True, skip_state_audit=True)
        finally:
            if episode_temp_dir.exists():
                logger.debug(f"Cleaning up temporary directory: {episode_temp_dir}")
                try:
                    shutil.rmtree(episode_temp_dir)
                except Exception as cleanup_err:
                    logger.warning(f"Could not clean up temporary directory {episode_temp_dir}: {cleanup_err}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download podcast episode')
    parser.add_argument('--content', required=True, help='Content ID to process')
    parser.add_argument('--url', help='Direct URL to the audio file (optional)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    logger.info(f"Attempting to download podcast: {args.content}")
    
    async def main():
        downloader = PodcastDownloader()
        
        # If URL is not provided via command line, try to get it from database
        if not args.url:
            logger.info(f"No URL provided, fetching from database for content_id: {args.content}")
            with get_session() as session:
                content = session.query(Content).filter_by(content_id=args.content).first()
                if not content:
                    error_result = {
                        'status': 'failed',
                        'error': f"Content not found in database: {args.content}",
                        'error_code': 'not_found',
                        'permanent': True
                    }
                    print(json.dumps(error_result))
                    return
                
                # Try to get audio URL from metadata
                url = content.meta_data.get('audio_url') or content.meta_data.get('episode_url')
                if not url:
                    error_result = {
                        'status': 'failed',
                        'error': f"No audio URL found in metadata for content: {args.content}",
                        'error_code': 'bad_url',
                        'permanent': True
                    }
                    print(json.dumps(error_result))
                    return
        else:
            url = args.url
            
        logger.info(f"Downloading from URL: {url}")
        result = await downloader.download_single_episode(url, args.content)
        
        # Print ONLY the final JSON result to stdout
        print(json.dumps(result))
    
    asyncio.run(main()) 