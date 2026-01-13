#!/usr/bin/env python3
"""
Rumble Channel Indexer
=====================

Indexes Rumble channels using yt-dlp to extract video metadata.
"""

import asyncio
import logging
import time
import re
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import pytz
import yt_dlp
from pathlib import Path

from src.database.session import get_session
from src.database.manager import DatabaseManager
from src.database.models import Content
from src.utils.logger import setup_indexer_logger, setup_task_logger
from src.utils.db_utils import standardize_source_name
from src.utils.content_id import generate_content_id
from src.utils.config import load_config

class RumbleIndexer:
    def __init__(self, test_mode: bool = False):
        self.test_mode = test_mode
        self.logger = setup_indexer_logger('rumble')
        self.task_logger = setup_task_logger('rumble')
        self.logger.setLevel(logging.INFO)
        
        # Configure yt-dlp options
        self.ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': 'in_playlist',  # Extract flat for playlists but get full info for individual videos
            'ignoreerrors': True,  # Skip unavailable videos
        }
    
    def _parse_duration(self, duration_value) -> int:
        """Convert duration to seconds"""
        try:
            # yt-dlp typically returns duration as integer seconds for Rumble
            if isinstance(duration_value, (int, float)):
                return int(duration_value)
            # If it's a string, try to parse it as integer
            if isinstance(duration_value, str):
                return int(duration_value)
            return 0
        except (TypeError, ValueError) as e:
            self.logger.warning(f"Could not parse duration {duration_value}: {str(e)}")
            return 0
    
    def _get_channel_id(self, channel_url: str) -> str:
        """Extract channel ID from URL"""
        # Example URL: https://rumble.com/c/CHANNELNAME
        try:
            parts = channel_url.strip('/').split('/')
            if 'c' in parts:
                channel_idx = parts.index('c')
                if channel_idx + 1 < len(parts):
                    return parts[channel_idx + 1]
            return parts[-1]  # Fallback to last part
        except Exception as e:
            self.logger.error(f"Error extracting channel ID from {channel_url}: {str(e)}")
            return channel_url
    
    async def _add_video_to_db(self, video_id: str, channel_id: str, publish_date: datetime,
                              title: str, description: str, video_details: Dict, project: str,
                              session: get_session, project_sources: dict = None) -> None:
        """Add or update a video in the database"""
        try:
            # Load config for thresholds
            config = load_config()
            short_threshold = config['processing'].get('short_video_threshold', 180)
            
            # Ensure publish_date is timezone-aware
            if publish_date.tzinfo is None:
                publish_date = pytz.UTC.localize(publish_date)
            else:
                publish_date = publish_date.astimezone(pytz.UTC)
            
            # Get duration
            duration = self._parse_duration(video_details.get('duration', 0))
            
            # Determine if it's a short (less than threshold)
            is_short = duration < short_threshold
            
            # Import here to avoid circular imports
            from src.utils.project_utils import get_language_for_channel
            
            # Get channel URL and language
            channel_url = f"https://rumble.com/c/{channel_id}"
            main_language = get_language_for_channel(channel_url, project_sources)
            
            content_data = {
                "content_id": generate_content_id('rumble', video_id=video_id),
                "platform": "rumble",
                "channel_name": video_details.get('channel', channel_id),
                "channel_url": channel_url,
                "title": title,
                "description": description,
                "publish_date": publish_date,
                "duration": duration,
                "is_downloaded": False,
                "is_converted": False,
                "is_transcribed": False,
                "is_diarized": False,
                "is_stitched": False,
                "is_embedded": False,
                "is_short": is_short,
                "processing_priority": 2,  # Medium priority for Rumble
                "projects": project,
                "main_language": main_language,  # Get language from sources.csv
                "meta_data": {
                    'view_count': video_details.get('view_count', 0),
                    'like_count': video_details.get('like_count', 0),
                    'duration_string': str(duration),
                    'thumbnail': video_details.get('thumbnail', ''),
                    'webpage_url': video_details.get('webpage_url', '')
                },
                "blocked_download": False,
                "total_chunks": None,
                "chunks_processed": 0,
                "chunks_status": {}
            }
            
            db = DatabaseManager(session)
            
            # Check if content already exists for this project
            existing_content = session.query(Content).filter(
                Content.content_id == video_id,
                Content.projects.any(project)
            ).first()
            already_exists_for_project = existing_content is not None
            
            # Add or update content
            content = db.add_content(content_data)
                        
        except Exception as e:
            self.logger.error(f"Error adding video {video_id} to database: {str(e)}")
            raise
    
    async def index_channels(self, channel_urls: List[str], project: str, progress_callback=None, project_sources: dict = None) -> List[Dict]:
        """Index multiple Rumble channels"""
        results = []
        total_channels = len(channel_urls)
        processed = 0
        
        self.logger.info(f"Starting indexing of {total_channels} Rumble channels")
        
        for url in channel_urls:
            try:
                self.logger.info(f"Starting channel {processed + 1}/{total_channels}: {url}")
                result = await self.index_channel(url, project, progress_callback, project_sources)
                results.append(result)
                
                processed += 1
                if progress_callback:
                    progress_callback({
                        "current": processed,
                        "total": total_channels,
                        "status": f"Indexed {processed}/{total_channels} channels"
                    })
                
                self.logger.info(f"Indexed channel {url}: {result['status']}")
                if result['status'] == 'success':
                    self.logger.info(f"Found {result.get('video_count', 0)} videos")
                    
            except Exception as e:
                error_msg = f"Error indexing channel {url}: {str(e)}"
                self.logger.error(error_msg)
                results.append({
                    "status": "error",
                    "channel_url": url,
                    "error": error_msg
                })
        
        # Log final summary
        successful = [r for r in results if r.get("status") == "success"]
        failed = [r for r in results if r.get("status") == "error"]
        total_videos = sum(r.get('video_count', 0) for r in successful)
        
        self.logger.info(
            f"\nRumble indexing complete:\n"
            f"  - Processed {len(channel_urls)} channels\n"
            f"  - Successful: {len(successful)}\n"
            f"  - Failed: {len(failed)}\n"
            f"  - Total videos found: {total_videos}"
        )
        
        return results
    
    async def index_channel(self, channel_url: str, project: str, progress_callback=None, project_sources: dict = None) -> Dict:
        """Index a single Rumble channel with optimized duplicate detection"""
        start_time = time.time()
        channel_title = "Unknown"
        channel_id = None
        existing_count = 0
        indexed_count = 0
        total_committed = 0
        
        try:
            self.logger.info(f"Starting to index channel: {channel_url}")
            
            # Get channel ID
            channel_id = self._get_channel_id(channel_url)
            if not channel_id:
                raise ValueError(f"Could not resolve channel ID for {channel_url}")
            
            # Get existing content info with optimized query
            with get_session() as session:
                existing_content = session.query(Content.content_id, Content.publish_date).filter(
                    Content.platform == 'rumble',
                    Content.channel_url == f"https://rumble.com/c/{channel_id}",
                    Content.projects.any(project)
                ).all()
                
                existing_count = len(existing_content)
                # Create a set of existing video IDs for fast lookup
                existing_video_ids = {content.content_id for content in existing_content}
                
                # Get latest publish date
                last_indexed_date = None
                if existing_content:
                    latest_dates = [c.publish_date for c in existing_content if c.publish_date]
                    if latest_dates:
                        last_indexed_date = max(latest_dates)
                        if last_indexed_date.tzinfo is None:
                            last_indexed_date = pytz.UTC.localize(last_indexed_date)
                        else:
                            last_indexed_date = last_indexed_date.astimezone(pytz.UTC)
            
            self.logger.info(f"Found {existing_count} existing videos for channel {channel_url}")
            if last_indexed_date:
                self.logger.info(f"Latest indexed video date: {last_indexed_date}")
            else:
                self.logger.info("No previous content found, will index all videos")
            
            # Extract channel info using yt-dlp with early termination optimization
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                try:
                    channel_info = ydl.extract_info(channel_url, download=False)
                    channel_title = channel_info.get('title', channel_id)
                    entries = channel_info.get('entries', [])
                    total_videos = len(entries)
                    
                    self.logger.info(f"Found channel {channel_title} with {total_videos} videos")
                    
                    # Early termination counters
                    consecutive_old_videos = 0
                    consecutive_existing_videos = 0
                    early_termination_threshold = 10
                    
                    # Process videos in batches with early termination
                    batch_size = 50
                    should_continue = True
                    
                    for i in range(0, len(entries), batch_size):
                        if not should_continue:
                            self.logger.info(f"Early termination triggered after processing {i} videos")
                            break
                            
                        batch = entries[i:i + batch_size]
                        batch_indexed = 0
                        
                        with get_session() as session:
                            for video in batch:
                                try:
                                    # Extract video ID from URL for Rumble
                                    video_url = video.get('url', '')
                                    video_id = video.get('id')
                                    
                                    # If no ID, try to extract from URL
                                    if not video_id and video_url:
                                        # Rumble URLs typically look like: https://rumble.com/v12345a-video-title.html
                                        match = re.search(r'/v([a-zA-Z0-9]+)-', video_url)
                                        if match:
                                            video_id = match.group(1)
                                    
                                    if not video_id:
                                        self.logger.warning(f"Could not extract video ID from URL: {video_url}")
                                        continue
                                    
                                    # Quick check: if video already exists, skip without full extraction
                                    if video_id in existing_video_ids:
                                        consecutive_existing_videos += 1
                                        consecutive_old_videos = 0  # Reset old counter
                                        
                                        if consecutive_existing_videos >= early_termination_threshold:
                                            self.logger.info(
                                                f"Found {consecutive_existing_videos} consecutive existing videos. "
                                                f"Assuming channel is up to date, terminating early."
                                            )
                                            should_continue = False
                                            break
                                        continue
                                    else:
                                        consecutive_existing_videos = 0  # Reset existing counter
                                    
                                    # For Rumble with extract_flat, we always need to fetch full video info
                                    if video.get('_type') == 'url' or 'duration' not in video or video.get('duration') is None:
                                        try:
                                            self.logger.debug(f"Fetching full details for video {video_id} from {video_url}")
                                            video_info = ydl.extract_info(video_url, download=False)
                                            # Update video with full info
                                            video = video_info  # Replace entire video object with full info
                                            # Ensure video_id is set
                                            if not video.get('id'):
                                                video['id'] = video_id
                                        except Exception as e:
                                            self.logger.warning(f"Could not fetch full details for {video_id}: {str(e)}")
                                            continue
                                    
                                    # Parse publish date
                                    publish_date = None
                                    if 'timestamp' in video:
                                        publish_date = datetime.fromtimestamp(video['timestamp'])
                                        publish_date = pytz.UTC.localize(publish_date)
                                    
                                    # Check if video is older than our latest indexed date
                                    if last_indexed_date and publish_date and publish_date <= last_indexed_date:
                                        consecutive_old_videos += 1
                                        consecutive_existing_videos = 0  # Reset existing counter
                                        
                                        if consecutive_old_videos >= early_termination_threshold:
                                            self.logger.info(
                                                f"Found {consecutive_old_videos} consecutive old videos "
                                                f"(older than {last_indexed_date}). Assuming we've reached "
                                                f"previously indexed content, terminating early."
                                            )
                                            should_continue = False
                                            break
                                        continue
                                    else:
                                        consecutive_old_videos = 0  # Reset old counter
                                    
                                    # Add video to database (use video_id we extracted, not video.get('id'))
                                    await self._add_video_to_db(
                                        video_id,
                                        channel_id,
                                        publish_date or datetime.now(pytz.UTC),
                                        video.get('title', ''),
                                        video.get('description', ''),
                                        video,
                                        project,
                                        session,
                                        project_sources
                                    )
                                    batch_indexed += 1
                                    
                                    # Add to existing set to avoid re-processing in subsequent checks
                                    existing_video_ids.add(video_id)
                                    
                                except Exception as e:
                                    self.logger.error(f"Error processing video {video.get('id', 'unknown')}: {str(e)}")
                                    continue
                            
                            # Commit the batch
                            try:
                                session.commit()
                                indexed_count += batch_indexed
                                total_committed += batch_indexed
                                if batch_indexed > 0:
                                    self.logger.info(f"Committed batch of {batch_indexed} videos")
                            except Exception as e:
                                self.logger.error(f"Error committing batch: {str(e)}")
                                session.rollback()
                                continue
                        
                        # Update progress
                        if progress_callback:
                            current_processed = min(i + len(batch), total_videos)
                            progress_callback({
                                "status": f"Indexed {indexed_count} videos from {channel_title} (processed {current_processed}/{total_videos})",
                                "channel_name": channel_title,
                                "current": current_processed,
                                "total": total_videos,
                                "early_termination": not should_continue
                            })
                        
                        # Break out of batch loop if early termination was triggered
                        if not should_continue:
                            break
                    
                except Exception as e:
                    raise ValueError(f"Failed to extract channel info: {str(e)}")
            
            # After indexing is complete, log task event
            duration = time.time() - start_time
            
            # Log to task logger
            self.task_logger.info(f"Indexed Rumble channel: {channel_title} (had {existing_count} videos, found {indexed_count} new, committed {total_committed}) in {duration:.1f}s", extra={
                'task_event': True,
                'task_id': f"index_rumble_{standardize_source_name(channel_url)}",
                'content_id': channel_url,
                'duration': duration,
                'channel_name': channel_title,
                'existing_count': existing_count,
                'new_count': indexed_count,
                'committed_count': total_committed,
                'total_count': existing_count + total_committed,
                'component': 'rumble_indexer'
            })
            
            # Log to file logger
            self.logger.info(
                f"Completed indexing channel {channel_title}:\n"
                f"  - URL: {channel_url}\n"
                f"  - Existing videos: {existing_count}\n"
                f"  - New videos: {indexed_count}\n"
                f"  - Successfully committed: {total_committed}\n"
                f"  - Total videos: {existing_count + total_committed}\n"
                f"  - Duration: {duration:.1f}s"
            )
            
            # Prepare final status message
            status_msg = f"Completed {channel_title}: "
            if indexed_count > 0:
                status_msg += f"added {indexed_count} new videos (had {existing_count} before)"
            else:
                status_msg += "no new videos found"
            
            return {
                "status": "success",
                "channel_url": channel_url,
                "channel_name": channel_title,
                "video_count": indexed_count,
                "existing_count": existing_count,
                "total_count": existing_count + total_committed,
                "committed_count": total_committed,
                "duration": duration,
                "message": status_msg
            }
            
        except Exception as e:
            error_msg = f"Error indexing channel {channel_url}: {str(e)}"
            self.logger.error(error_msg)
            return {
                "status": "error",
                "channel_url": channel_url,
                "error": str(e)
            } 