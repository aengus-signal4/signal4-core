from googleapiclient.discovery import build
from datetime import datetime
import os
from typing import List, Dict, Optional, Callable, Tuple
from ..database.session import get_session
from ..database.manager import DatabaseManager
from ..database.models import Content, Channel
from ..utils.logger import setup_indexer_logger, setup_task_logger
from ..utils.content_id import generate_content_id
import asyncio
from dateutil.parser import parse
import isodate
from dotenv import load_dotenv
import pytz
import logging
import re
from urllib.parse import unquote
from sqlalchemy import text
import time

def parse_duration(duration_str: str) -> int:
    """Convert ISO 8601 duration to seconds"""
    try:
        duration = isodate.parse_duration(duration_str)
        return int(duration.total_seconds())
    except (isodate.ISO8601Error, TypeError, ValueError) as e:
        logging.warning(f"Could not parse duration {duration_str}: {str(e)}")
        return 0

class YouTubeIndexer:
    # Quota costs for different operations
    QUOTA_COSTS = {
        'channels.list': 1,
        'playlistItems.list': 1,
        'videos.list': 1,
        'search.list': 100,  # Most expensive, used for handle resolution
        'activities.list': 1
    }
    
    DAILY_QUOTA_LIMIT = 10000  # YouTube API daily quota limit
    
    def __init__(self, test_mode: bool = False, logger=None):
        # Load environment variables from .env file
        self.test_mode = test_mode
        self.api_keys = []
        self.current_key_index = 0
        self.quota_usage = {}
        self.youtube = None
        self.logger = logger or setup_indexer_logger('youtube')
        self.task_logger = setup_task_logger('youtube')  # Add task logger
        
        # Load API keys
        api_keys_str = os.getenv('YOUTUBE_API_KEYS')
        if not api_keys_str:
            raise ValueError("No YouTube API keys found in environment")
        
        self.api_keys = [k.strip() for k in api_keys_str.split(',') if k.strip()]
        if not self.api_keys:
            raise ValueError("No valid YouTube API keys found")
            
        # Initialize quota usage tracking
        for key in self.api_keys:
            self.quota_usage[key] = 0
        
        # Initialize client
        self._init_youtube_client()

    def _init_youtube_client(self):
        """Initialize YouTube API client with current key"""
        self.youtube = build('youtube', 'v3', developerKey=self.api_keys[self.current_key_index])
    
    def _rotate_api_key(self):
        """Rotate to the next API key"""
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        self._init_youtube_client()
        self.logger.info(f"Rotated to next YouTube API key (index: {self.current_key_index})")
    
    async def _execute_with_retry(self, request, operation_name: str = None):
        """Execute a request with API key rotation on quota errors"""
        loop = asyncio.get_event_loop()
        
        for _ in range(len(self.api_keys)):
            try:
                # Run the synchronous execute() in a thread pool
                result = await loop.run_in_executor(None, request.execute)
                
                # Track quota usage if operation name provided
                if operation_name:
                    current_key = self.api_keys[self.current_key_index]
                    self.quota_usage[current_key] += self.QUOTA_COSTS.get(operation_name, 1)
                    quota_percent = (self.quota_usage[current_key] / self.DAILY_QUOTA_LIMIT) * 100
                    self.logger.debug(
                        f"API Key {self.current_key_index} quota: "
                        f"{self.quota_usage[current_key]}/{self.DAILY_QUOTA_LIMIT} "
                        f"({quota_percent:.1f}%)"
                    )
                
                return result
                
            except Exception as e:
                error_str = str(e).lower()
                if 'quota' in error_str:
                    # If we hit quota limit, try rotating the key
                    current_key = self.api_keys[self.current_key_index]
                    self.logger.warning(
                        f"API key {self.current_key_index} quota exceeded "
                        f"({self.quota_usage[current_key]}/{self.DAILY_QUOTA_LIMIT} units used). "
                        "Rotating to next key."
                    )
                    self._rotate_api_key()
                    continue
                raise
        
        raise Exception("All API keys exhausted")

    def get_quota_usage(self) -> Dict[str, int]:
        """Get current quota usage for all API keys"""
        return self.quota_usage

    def get_total_quota_usage(self) -> int:
        """Get total quota usage across all keys"""
        return sum(self.quota_usage.values())

    async def _get_channel_id(self, channel_url: str) -> Optional[str]:
        """Get channel ID from URL"""
        try:
            # First decode the URL (handles double-encoded URLs)
            decoded_url = unquote(unquote(channel_url))
            self.logger.debug(f"Decoded URL: {decoded_url}")
            
            # Extract channel ID from URL if possible
            url_patterns = [
                r'youtube\.com/channel/([^/?&]+)',  # Direct channel ID
                r'youtube\.com/@([^/?&]+)',         # Handle
                r'youtube\.com/c/([^/?&]+)',        # Custom URL
                r'youtube\.com/user/([^/?&]+)'      # Legacy username
            ]
            
            for pattern in url_patterns:
                match = re.search(pattern, decoded_url)
                if match:
                    identifier = match.group(1)
                    self.logger.debug(f"Extracted identifier from URL: {identifier}")
                    
                    # If it's a direct channel ID (UC...), return it
                    if identifier.startswith('UC'):
                        return identifier
                    
                    # For other formats, we need to resolve to channel ID
                    try:
                        search_request = self.youtube.search().list(
                            part="snippet",
                            q=identifier,
                            type="channel",
                            maxResults=1
                        )
                        search_response = await self._execute_with_retry(search_request, 'search.list')
                        
                        if search_response.get('items'):
                            channel_id = search_response['items'][0]['id']['channelId']
                            self.logger.debug(f"Resolved identifier {identifier} to channel ID: {channel_id}")
                            return channel_id
                            
                    except Exception as e:
                        self.logger.error(f"Error resolving identifier {identifier}: {str(e)}")
                        continue
            
            # If no match or resolution failed, try direct channel lookup
            try:
                # Use the decoded channel name for search
                channel_name = re.search(r'youtube\.com/c/([^/?&]+)', decoded_url)
                if channel_name:
                    search_term = channel_name.group(1).replace('-', ' ')
                else:
                    search_term = decoded_url
                
                self.logger.debug(f"Trying direct channel lookup with search term: {search_term}")
                search_request = self.youtube.search().list(
                    part="snippet",
                    q=search_term,
                    type="channel",
                    maxResults=1
                )
                search_response = await self._execute_with_retry(search_request, 'search.list')
                
                if search_response.get('items'):
                    channel_id = search_response['items'][0]['id']['channelId']
                    channel_title = search_response['items'][0]['snippet']['title']
                    self.logger.debug(f"Found channel: {channel_title} (ID: {channel_id})")
                    return channel_id
                
            except Exception as e:
                self.logger.error(f"Error in direct channel lookup: {str(e)}")
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error processing channel URL {channel_url}: {str(e)}")
            return None
    
    def _get_last_indexed_date(self, channel_url: str, project: str) -> Optional[datetime]:
        """Get the date of the most recently indexed video for a channel"""
        try:
            with get_session() as session:
                # Set a longer statement timeout (120 seconds)
                session.execute(text('SET statement_timeout = 120000'))
                
                # Use explicit index hint and execution options
                latest_content = session.query(Content)\
                    .filter(
                        Content.platform == 'youtube',
                        Content.channel_url == channel_url,
                        Content.projects.any(project)
                    )\
                    .order_by(Content.publish_date.desc())\
                    .execution_options(
                        timeout=120,
                        statement_hint=f"USE INDEX (idx_content_channel_url_projects)"
                    )\
                    .first()
                
                if latest_content and latest_content.publish_date:
                    # Ensure timezone awareness
                    if latest_content.publish_date.tzinfo is None:
                        return pytz.UTC.localize(latest_content.publish_date)
                    return latest_content.publish_date.astimezone(pytz.UTC)
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting last indexed date: {str(e)}")
            return None

    async def _get_recent_videos_from_activities(self, channel_id: str) -> List[Dict]:
        """Get recent videos from channel activities"""
        self.logger.info(f"Getting recent videos from activities for channel {channel_id}")
        
        videos = []
        page_token = None
        
        while True:
            try:
                # Get activities for channel
                activities_request = self.youtube.activities().list(
                    part="snippet,contentDetails",
                    channelId=channel_id,
                    maxResults=50,
                    pageToken=page_token
                )
                activities_response = await self._execute_with_retry(activities_request, 'activities.list')
                
                # Process each activity
                for activity in activities_response.get("items", []):
                    try:
                        # Skip non-upload activities
                        if activity["snippet"]["type"] != "upload":
                            continue
                            
                        video_id = activity["contentDetails"]["upload"]["videoId"]
                        publish_date = datetime.fromisoformat(activity["snippet"]["publishedAt"].replace('Z', '+00:00'))
                        if publish_date.tzinfo is None:
                            publish_date = pytz.UTC.localize(publish_date)
                        else:
                            publish_date = publish_date.astimezone(pytz.UTC)
                        
                        self.logger.debug(f"Found video {video_id} published at {publish_date}")
                        
                        videos.append({
                            "id": video_id,
                            "publish_date": publish_date
                        })
                        
                    except Exception as e:
                        self.logger.error(f"Error processing activity: {str(e)}")
                        continue
                
                # Get next page token
                page_token = activities_response.get("nextPageToken")
                if not page_token:
                    break
                    
            except Exception as e:
                self.logger.error(f"Error getting activities: {str(e)}")
                break
                
        return videos

    async def _get_video_details(self, video_ids: List[str]) -> Dict:
        """Get detailed information for a batch of videos"""
        if not video_ids:
            return {}
            
        video_request = self.youtube.videos().list(
            part="snippet,contentDetails,statistics",
            id=','.join(video_ids)
        )
        
        video_response = await self._execute_with_retry(video_request, 'videos.list')
        return {v['id']: v for v in video_response.get('items', [])}

    async def index_channels(self, channel_urls: List[str], project: str, progress_callback=None) -> List[Dict]:
        """Index multiple YouTube channels"""
        try:
            results = []
            total_channels = len(channel_urls)
            processed = 0
            
            self.logger.info(f"Starting indexing of {total_channels} YouTube channels")
            
            for url in channel_urls:
                try:
                    self.logger.info(f"Indexing channel: {url}")
                    result = await self.index_channel(url, project, progress_callback)
                    results.append(result)
                    
                    processed += 1
                    if progress_callback:
                        progress_callback({
                            "current": processed,
                            "total": total_channels,
                            "status": f"Indexed {processed}/{total_channels} channels"
                        })
                    
                    self.logger.info(f"Channel indexing result: {result}")
                    
                except Exception as e:
                    error_msg = f"Error indexing channel {url}: {str(e)}"
                    self.logger.error(error_msg)
                    results.append({
                        "status": "error",
                        "channel_url": url,
                        "error": error_msg
                    })
            
            # Log summary
            successful = [r for r in results if r.get("status") == "success"]
            failed = [r for r in results if r.get("status") == "error"]
            self.logger.info(f"Indexing complete: {len(successful)} successful, {len(failed)} failed")
            
            return results
            
        except Exception as e:
            error_msg = f"Error in index_channels: {str(e)}"
            self.logger.error(error_msg)
            return [{"status": "error", "error": error_msg}]

    async def index_channel(self, channel_url: str, project: str, progress_callback=None, project_sources: dict = None) -> Dict:
        """Index a single YouTube channel"""
        start_time = time.time()
        channel_title = "Unknown"
        channel_id = None
        existing_count = 0
        indexed_count = 0
        total_committed = 0
        
        try:
            self.logger.info(f"Starting to index channel: {channel_url}")
            
            # Get channel ID
            channel_id = await self._get_channel_id(channel_url)
            if not channel_id:
                raise ValueError(f"Could not resolve channel ID for {channel_url}")
            
            # Get existing content count
            with get_session() as session:
                existing_count = session.query(Content).filter(
                    Content.platform == 'youtube',
                    Content.channel_url == f"https://www.youtube.com/channel/{channel_id}",
                    Content.projects.any(project)
                ).count()
            
            self.logger.info(f"Found {existing_count} existing videos for channel {channel_url}")
            
            # Get channel info
            channel_request = self.youtube.channels().list(
                part="snippet,contentDetails,statistics",
                id=channel_id
            )
            channel_response = await self._execute_with_retry(channel_request, 'channels.list')
            
            if not channel_response.get('items'):
                raise ValueError(f"Channel not found: {channel_id}")
            
            channel = channel_response['items'][0]
            channel_title = channel['snippet']['title']
            total_videos = int(channel['statistics']['videoCount'])
            
            self.logger.info(f"Found channel {channel_title} with {total_videos} total videos")
            
            # Get last indexed date
            with get_session() as session:
                db = DatabaseManager(session)
                latest_content = db.get_latest_content(project, f"https://www.youtube.com/channel/{channel_id}")
                last_indexed_date = None
                if latest_content and latest_content.publish_date:
                    # Ensure timezone awareness
                    last_indexed_date = latest_content.publish_date
                    if last_indexed_date.tzinfo is None:
                        last_indexed_date = pytz.UTC.localize(last_indexed_date)
                    else:
                        last_indexed_date = last_indexed_date.astimezone(pytz.UTC)
                
                if last_indexed_date:
                    self.logger.info(f"Found last indexed date for {channel_title}: {last_indexed_date}")
                    if latest_content:
                        self.logger.debug(f"Latest content: {latest_content.title} (ID: {latest_content.content_id})")
                else:
                    self.logger.info(f"No previous content found for {channel_title}, will index all videos")
            
            # First check recent activity
            self.logger.info(f"Checking recent activity for channel {channel_title}")
            recent_videos = await self._get_recent_videos_from_activities(channel_id)

            # Process recent videos and check for overlap
            found_overlap = False
            if recent_videos:
                # Normalize all video dates to be timezone-aware
                for video in recent_videos:
                    video_date = video['publish_date']
                    if video_date.tzinfo is None:
                        video['publish_date'] = pytz.UTC.localize(video_date)
                    else:
                        video['publish_date'] = video_date.astimezone(pytz.UTC)

                # Check if activity feed contains any video at or before last_indexed_date
                # This indicates we have continuity with existing content
                if last_indexed_date:
                    found_overlap = any(v['publish_date'] <= last_indexed_date for v in recent_videos)

                    # Filter to only new videos (newer than last indexed)
                    new_videos = [v for v in recent_videos if v['publish_date'] > last_indexed_date]
                else:
                    # No previous content - all videos are new
                    new_videos = recent_videos

                self.logger.info(f"Activity feed: {len(recent_videos)} videos, {len(new_videos)} new, overlap={found_overlap}")

                # Add new videos from activity feed
                if new_videos:
                    # Get all video details in batches of 50
                    video_ids = [v['id'] for v in new_videos]
                    video_details = await self._get_video_details(video_ids)

                    with get_session() as session:
                        for video in new_videos:
                            if video['id'] in video_details:
                                details = video_details[video['id']]
                                await self._add_video_to_db(
                                    video['id'],
                                    channel_id,
                                    video['publish_date'],
                                    details['snippet']['title'],
                                    details['snippet']['description'],
                                    details,
                                    project,
                                    session,
                                    project_sources
                                )
                                indexed_count += 1
                                total_committed += 1

                        session.commit()

                        if progress_callback:
                            progress_callback({
                                "status": f"Indexed {indexed_count} new videos from {channel_title}",
                                "channel_name": channel_title,
                                "current": indexed_count,
                                "total": total_videos
                            })

            # If no overlap found, activity feed didn't reach back to last indexed content
            # This means there's a gap - do a full index to catch everything
            if not found_overlap:
                self.logger.info(f"No overlap found in activity feed, starting full index of channel {channel_title}")
                result = await self._index_channel_full(channel_url, channel, channel_id, project, progress_callback, project_sources)
                indexed_count = result.get('video_count', 0)
                total_committed = result.get('committed_count', 0)
            
            # After indexing is complete, log task event
            duration = time.time() - start_time
            
            # Log to task logger for central logging
            self.task_logger.info(f"Indexed YouTube channel: {channel_title} (had {existing_count} videos, found {indexed_count} new, committed {total_committed}) in {duration:.1f}s", extra={
                'task_event': True,
                'task_id': f"index_youtube_{channel_id}",
                'content_id': channel_url,
                'duration': duration,
                'channel_name': channel_title,
                'existing_count': existing_count,
                'new_count': indexed_count,
                'committed_count': total_committed,
                'total_count': existing_count + total_committed,
                'component': 'youtube_indexer'
            })
            
            # Log to file logger for detailed logging
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

    async def _index_channel_full(self, channel_url: str, channel: Dict, channel_id: str, 
                                project: str, progress_callback=None, project_sources: dict = None) -> Dict:
        """Do a full index of all videos from a channel's playlist"""
        uploads_playlist_id = channel['contentDetails']['relatedPlaylists']['uploads']
        total_videos = int(channel['statistics']['videoCount'])
        channel_title = channel['snippet']['title']
        
        self.logger.info(f"Starting full playlist index for channel {channel_title}")
        
        indexed_count = 0
        total_committed = 0
        current_video = 0
        error_count = 0
        
        # Process videos in batches
        next_page_token = None
        
        while True:
            self.logger.debug(f"Fetching playlist items (page token: {next_page_token})")
            playlist_request = self.youtube.playlistItems().list(
                part="snippet,contentDetails",
                playlistId=uploads_playlist_id,
                maxResults=50,
                pageToken=next_page_token
            )
            
            playlist_response = await self._execute_with_retry(playlist_request, 'playlistItems.list')
            
            video_batch = playlist_response.get('items', [])
            if not video_batch:
                self.logger.debug("No more videos in playlist")
                break
                    
            # Collect video IDs for batch processing
            video_ids = [item['contentDetails']['videoId'] for item in video_batch]
            self.logger.debug(f"Processing batch of {len(video_ids)} videos")
            
            # Get detailed video info in one batch request
            video_details = await self._get_video_details(video_ids)
            
            # Store videos in database
            with get_session() as session:
                batch_indexed = 0
                for video in video_details.values():
                    try:
                        # Parse video date and ensure timezone awareness
                        video_date = datetime.fromisoformat(video['snippet']['publishedAt'].replace('Z', '+00:00'))
                        if video_date.tzinfo is None:
                            video_date = pytz.UTC.localize(video_date)
                        else:
                            video_date = video_date.astimezone(pytz.UTC)
                        
                        await self._add_video_to_db(
                            video['id'],
                            channel_id,
                            video_date,
                            video['snippet']['title'],
                            video['snippet']['description'],
                            video,
                            project,
                            session,
                            project_sources
                        )
                        batch_indexed += 1
                        
                    except Exception as e:
                        self.logger.error(f"Error processing video {video['id']}: {str(e)}")
                        error_count += 1
                        continue
                
                # Commit the batch
                try:
                    session.commit()
                    indexed_count += batch_indexed
                    total_committed += batch_indexed
                    if batch_indexed > 0:  # Only log if we actually indexed something
                        self.logger.info(f"Committed batch of {batch_indexed} videos")
                except Exception as e:
                    self.logger.error(f"Error committing batch: {str(e)}")
                    session.rollback()
                    continue
            
            # Update progress
            current_video += len(video_batch)
            if progress_callback:
                progress_callback({
                    "current": current_video,
                    "total": total_videos,
                    "indexed": indexed_count,
                    "status": f"Indexed {indexed_count} videos from {channel_title}"
                })
            
            # Get next page token
            next_page_token = playlist_response.get('nextPageToken')
            if not next_page_token:
                break
        
        return {
            "status": "success",
            "channel_url": channel_url,
            "video_count": indexed_count,
            "committed_count": total_committed,
            "error_count": error_count,
            "message": "Full playlist index complete"
        }

    async def _add_video_to_db(self, video_id: str, yt_channel_id: str, publish_date: datetime,
                              title: str, description: str, video_details: Dict, project: str,
                              session: get_session, project_sources: dict = None) -> None:
        """Add or update a video in the database"""
        try:
            # Load config for thresholds
            from ..utils.config import load_config
            config = load_config()
            short_threshold = config['processing'].get('short_video_threshold', 180)  # Default to 180s if not set

            # Ensure publish_date is timezone-aware
            if publish_date.tzinfo is None:
                publish_date = pytz.UTC.localize(publish_date)
            else:
                publish_date = publish_date.astimezone(pytz.UTC)

            # Get duration
            duration = parse_duration(video_details['contentDetails']['duration'])

            # Get live broadcast content status
            live_status = video_details['snippet'].get('liveBroadcastContent', 'none')
            is_live = live_status == 'live'
            is_upcoming = live_status == 'upcoming'

            # Determine if it's a short (less than threshold)
            is_short = False
            duration_str = video_details['contentDetails'].get('duration', '')
            if duration_str.startswith('PT'):
                # Mark as short if duration is less than threshold
                is_short = duration < short_threshold

            # Import here to avoid circular imports
            from src.utils.project_utils import get_language_for_channel

            # Get channel URL and language
            channel_url = f"https://www.youtube.com/channel/{yt_channel_id}"
            main_language = get_language_for_channel(channel_url, project_sources)

            # Look up database channel_id by primary_url
            db_channel = session.query(Channel).filter(
                Channel.platform == 'youtube',
                Channel.primary_url == channel_url
            ).first()
            db_channel_id = db_channel.id if db_channel else None

            content_data = {
                "content_id": generate_content_id('youtube', video_id=video_id),
                "platform": "youtube",
                "channel_id": db_channel_id,  # Link to channels table
                "channel_name": video_details['snippet']['channelTitle'],
                "channel_url": channel_url,
                "title": title,
                "description": description,
                "publish_date": publish_date,
                "duration": duration,
                "is_downloaded": False,
                "processing_priority": 1,  # Default priority for YouTube
                "projects": project,
                "main_language": main_language,  # Get language from sources.csv
                "is_short": is_short,  # Add is_short to main content data
                "meta_data": {
                    'view_count': int(video_details['statistics'].get('viewCount', 0)),
                    'like_count': int(video_details['statistics'].get('likeCount', 0)),
                    'comment_count': int(video_details['statistics'].get('commentCount', 0)),
                    'is_live': is_live,
                    'is_upcoming': is_upcoming,
                    'live_status': live_status,
                    'duration_string': duration_str
                }
            }
            
            db = DatabaseManager(session)
            
            # Check if content already exists for this project
            existing_content = session.query(Content).filter(
                Content.content_id == video_id,
                Content.projects.any(project)
            ).first()
            already_exists_for_project = existing_content is not None

            # Add or update content (DatabaseManager handles adding project to existing)
            content = db.add_content(content_data)
            
        except Exception as e:
            self.logger.error(f"Error adding video {video_id} to database: {str(e)}")
            raise 