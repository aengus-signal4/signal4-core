import asyncio
import aiohttp
import feedparser
import hashlib
import re
from datetime import datetime
from typing import List, Dict, Optional, Callable
from src.database.session import get_session
from src.database.manager import DatabaseManager
from src.database.models import Content, Channel
from src.utils.logger import setup_indexer_logger, setup_task_logger
from src.utils.db_utils import standardize_source_name
from src.utils.content_id import generate_content_id
import xml.etree.ElementTree as ET
import logging
from sqlalchemy import text
import pytz
from src.utils.config import load_config
import time

class PodcastIndexer:
    def __init__(self, test_mode: bool = False, logger=None):
        self.test_mode = test_mode
        self.logger = logger or setup_indexer_logger('podcast')
        self.task_logger = setup_task_logger('podcast')  # Add task logger
        self.session = None  # aiohttp session for making requests
        
    async def _get_feed_content(self, feed_url: str) -> str:
        """Fetch RSS feed content from URL"""
        if not self.session:
            self.session = aiohttp.ClientSession()
            
        # Browser-like headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1'
        }
        
        max_retries = 3
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                async with self.session.get(feed_url, headers=headers, timeout=30) as response:
                    response.raise_for_status()
                    return await response.text()
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1}/{max_retries} failed to fetch feed {feed_url}: {str(e)}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                else:
                    self.logger.error(f"All attempts failed to fetch feed {feed_url}: {str(e)}")
                    raise
            
    def _parse_duration(self, duration_str: str) -> int:
        """Convert duration string to seconds"""
        try:
            # Handle HH:MM:SS format
            if ':' in duration_str:
                parts = duration_str.split(':')
                if len(parts) == 3:
                    h, m, s = map(int, parts)
                    return h * 3600 + m * 60 + s
                elif len(parts) == 2:
                    m, s = map(int, parts)
                    return m * 60 + s
            # Handle seconds as string
            return int(duration_str)
        except (ValueError, TypeError) as e:
            self.logger.warning(f"Could not parse duration {duration_str}: {str(e)}")
            return 0

    def _get_or_create_channel(self, session, feed_url: str, feed: feedparser.FeedParserDict) -> Optional[int]:
        """
        Get or create a Channel record from RSS feed metadata.
        Updates description if RSS has better data than what's in the database.

        Args:
            session: Database session
            feed_url: RSS feed URL
            feed: Parsed feedparser feed object

        Returns:
            channel_id or None if error
        """
        try:
            feed_data = feed.feed

            # Extract metadata from RSS feed
            display_name = feed_data.get('title', '').strip()
            if not display_name:
                self.logger.warning(f"Feed has no title: {feed_url}")
                return None

            # Get description - prefer feed.description, fallback to subtitle or summary
            rss_description = (
                feed_data.get('description', '') or
                feed_data.get('subtitle', '') or
                feed_data.get('itunes_summary', '') or
                feed_data.get('summary', '')
            ).strip()

            # Get creator/author
            creator = (
                feed_data.get('author', '') or
                feed_data.get('itunes_author', '') or
                ''
            ).strip()

            # Get language
            language = feed_data.get('language', '').strip()

            # Get website
            website = feed_data.get('link', '').strip()

            # Get image
            image_url = ''
            if hasattr(feed_data, 'image') and hasattr(feed_data.image, 'href'):
                image_url = feed_data.image.href
            elif hasattr(feed_data, 'itunes_image') and hasattr(feed_data.itunes_image, 'href'):
                image_url = feed_data.itunes_image.href

            # Look up existing channel by primary_url (the canonical identifier)
            channel = session.query(Channel).filter(
                Channel.platform == 'podcast',
                Channel.primary_url == feed_url
            ).first()

            if channel:
                # Update channel if RSS has better data
                updated = False

                # Update display name if different
                if display_name and channel.display_name != display_name:
                    self.logger.info(f"Updating channel name: {channel.display_name} -> {display_name}")
                    channel.display_name = display_name
                    updated = True

                # Update description if RSS has data and (current is empty OR significantly different)
                if rss_description:
                    current_desc = (channel.description or '').strip()
                    # Update if current is empty or RSS description is substantively different
                    # Check if it's not just a truncation and isn't AI-generated garbage
                    if not current_desc or (
                        len(rss_description) > 100 and
                        rss_description[:50] not in current_desc and
                        current_desc[:50] not in rss_description
                    ):
                        self.logger.info(f"Updating description for {display_name} (had {len(current_desc)} chars, RSS has {len(rss_description)} chars)")
                        channel.description = rss_description
                        updated = True

                # Update platform_metadata
                pm = channel.platform_metadata or {}
                if creator and pm.get('creator') != creator:
                    pm['creator'] = creator
                    updated = True
                if language and pm.get('language') != language:
                    pm['language'] = language
                    updated = True
                if website and pm.get('website') != website:
                    pm['website'] = website
                    updated = True
                if image_url and pm.get('image_url') != image_url:
                    pm['image_url'] = image_url
                    updated = True

                if updated:
                    channel.platform_metadata = pm
                    channel.updated_at = datetime.utcnow()
                    session.commit()
                    self.logger.info(f"Updated channel metadata for: {display_name}")

                return channel.id

            else:
                # Create new channel
                # Generate channel_key from feed_url hash to ensure uniqueness
                url_hash = hashlib.md5(feed_url.encode()).hexdigest()[:8]
                slug = re.sub(r'[^a-z0-9]+', '-', display_name.lower()).strip('-')[:50]
                channel_key = f"podcast:{slug}-{url_hash}"

                new_channel = Channel(
                    channel_key=channel_key,
                    platform='podcast',
                    display_name=display_name,
                    primary_url=feed_url,  # RSS feed URL is the canonical identifier
                    language=language or None,
                    description=rss_description,
                    platform_metadata={
                        'creator': creator,
                        'language': language,
                        'website': website,
                        'image_url': image_url
                    },
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow()
                )
                session.add(new_channel)
                session.commit()
                self.logger.info(f"Created new channel: {display_name} (ID: {new_channel.id})")
                return new_channel.id

        except Exception as e:
            self.logger.error(f"Error getting/creating channel for {feed_url}: {e}")
            return None

    async def index_feeds(self, feed_urls: List[str], project: str, progress_callback=None, project_sources: dict = None) -> List[Dict]:
        """Index multiple podcast feeds"""
        results = []
        total_feeds = len(feed_urls)
        processed = 0
        
        self.logger.info(f"Starting indexing of {total_feeds} podcast feeds")
        
        for url in feed_urls:
            try:
                self.logger.info(f"Starting feed {processed + 1}/{total_feeds}: {url}")
                result = await self.index_feed(url, project, progress_callback, project_sources)
                results.append(result)
                
                processed += 1
                if progress_callback:
                    progress_callback({
                        "current": processed,
                        "total": total_feeds,
                        "status": f"Indexed {processed}/{total_feeds} feeds"
                    })
                
                self.logger.info(f"Indexed feed {url}: {result['status']}")
                if result['status'] == 'success':
                    self.logger.info(f"Found {result.get('episode_count', 0)} episodes")
                
            except Exception as e:
                error_msg = f"Error indexing feed {url}: {str(e)}"
                self.logger.error(error_msg)
                results.append({
                    "status": "error",
                    "feed_url": url,
                    "error": error_msg
                })
        
        # Log final summary
        successful = [r for r in results if r.get("status") == "success"]
        failed = [r for r in results if r.get("status") == "error"]
        total_episodes = sum(r.get("indexed_count", 0) for r in successful)
        
        self.logger.info(
            f"\nPodcast indexing complete:\n"
            f"  - Processed {len(feed_urls)} feeds\n"
            f"  - Successful: {len(successful)}\n"
            f"  - Failed: {len(failed)}\n"
            f"  - Total episodes found: {total_episodes}"
        )
        
        return results
    
    async def index_feed(self, feed_url: str, project: str, progress_callback: Optional[Callable] = None, project_sources: dict = None) -> Dict:
        """Index all episodes from a podcast feed"""
        start_time = time.time()
        try:
            self.logger.info(f"Starting to index feed: {feed_url}")
            
            # Get existing content count
            with get_session() as session:
                existing_count = session.query(Content).filter(
                    Content.platform == 'podcast',
                    Content.channel_url == feed_url,
                    Content.projects.any(project)
                ).count()
            
            self.logger.info(f"Found {existing_count} existing episodes for feed {feed_url}")
            
            # Load config for chunk planning
            config = load_config()
            chunk_size = config['processing']['chunk_size']
            chunk_overlap = config['processing']['transcription']['chunk_overlap']
            
            # Fetch feed content
            feed_content = await self._get_feed_content(feed_url)
            feed = feedparser.parse(feed_content)

            if not feed.entries:
                raise ValueError(f"No episodes found in feed: {feed_url}")

            # Get or create channel record with RSS metadata
            with get_session() as session:
                channel_id = self._get_or_create_channel(session, feed_url, feed)
                if not channel_id:
                    self.logger.warning(f"Could not get/create channel for {feed_url}, episodes will not be linked")

            # Get channel info
            channel_name = feed.feed.title
            total_episodes = len(feed.entries)
            indexed_count = 0
            skipped_count = 0

            self.logger.info(f"Found {total_episodes} episodes in feed (channel_id: {channel_id})")
            
            if progress_callback:
                progress_callback({
                    "status": f"Starting to index podcast: {channel_name} ({total_episodes} total episodes)",
                    "channel_name": channel_name,
                    "current_episode": 0,
                    "total_episodes": total_episodes
                })
            
            # Get the most recent episode date from the database
            with get_session() as session:
                db = DatabaseManager(session)
                latest_content = db.get_latest_content(project, feed_url)
                last_indexed_date = latest_content.publish_date if latest_content else None
                
                if last_indexed_date:
                    self.logger.info(f"Found last indexed date for {channel_name}: {last_indexed_date}")
                else:
                    self.logger.info(f"No previous episodes found for {channel_name}, will index all episodes")
            
            # Process episodes in chronological order (oldest first)
            entries = sorted(feed.entries, key=lambda x: x.get('published_parsed', 0))
            
            # Process episodes in batches to avoid long transactions
            batch_size = 50
            total_committed = 0
            batch_number = 0
            
            for i in range(0, len(entries), batch_size):
                batch = entries[i:i + batch_size]
                batch_number += 1
                batch_indexed = 0
                batch_skipped = 0
                
                with get_session() as session:
                    db = DatabaseManager(session)
                    
                    for entry in batch:
                        try:
                            # Extract episode GUID for deterministic content ID generation
                            episode_guid = entry.get('id', entry.get('guid', entry.get('link', entry.title)))
                            episode_id = generate_content_id('podcast', feed_url=feed_url, episode_guid=episode_guid)
                            
                            # Get publish date
                            publish_date = None
                            if 'published_parsed' in entry:
                                # Create datetime and make it timezone-aware
                                publish_date = datetime(*entry.published_parsed[:6])
                                if publish_date.tzinfo is None:
                                    publish_date = pytz.UTC.localize(publish_date)
                            
                            # Skip if we've already indexed up to this date
                            if last_indexed_date and publish_date:
                                # Ensure both datetimes are timezone-aware
                                if last_indexed_date.tzinfo is None:
                                    last_indexed_date = pytz.UTC.localize(last_indexed_date)
                                if publish_date.tzinfo is None:
                                    publish_date = pytz.UTC.localize(publish_date)
                                
                                if publish_date < last_indexed_date:
                                    batch_skipped += 1
                                    continue
                            
                            # Check if already indexed by ID
                            existing = db.get_content_by_id(episode_id)
                            if existing:
                                batch_skipped += 1
                                continue
                            
                            # Get episode duration
                            duration = 0
                            if hasattr(entry, 'itunes_duration'):
                                duration = self._parse_duration(entry.itunes_duration)
                            
                            # Get episode URL from enclosure
                            episode_url = None
                            if hasattr(entry, 'enclosures') and entry.enclosures:
                                for enclosure in entry.enclosures:
                                    if enclosure.type and enclosure.type.startswith('audio/'):
                                        episode_url = enclosure.href
                                        break
                            
                            if not episode_url:
                                # Fallback to links if no enclosure found
                                for link in entry.get('links', []):
                                    if link.get('type', '').startswith('audio/'):
                                        episode_url = link.get('href')
                                        break
                            
                            if not episode_url:
                                self.logger.warning(f"No audio URL found for episode: {entry.title}")
                                continue
                            
                            # Import here to avoid circular imports
                            from src.utils.project_utils import get_language_for_channel
                            
                            # Get language for this feed
                            main_language = get_language_for_channel(feed_url, project_sources)
                            
                            # Prepare content data
                            content_data = {
                                "content_id": episode_id,
                                "platform": "podcast",
                                "channel_id": channel_id,  # Link to channel
                                "channel_name": channel_name,
                                "channel_url": feed_url,
                                "title": entry.title,
                                "description": entry.get('description', ''),
                                "publish_date": publish_date,
                                "duration": duration,
                                "is_downloaded": False,
                                "is_converted": False,
                                "is_transcribed": False,
                                "processing_priority": 3,  # Highest priority for podcasts
                                "projects": project,
                                "main_language": main_language,  # Get language from sources.csv
                                "meta_data": {
                                    "episode_url": episode_url,
                                    "episode_type": entry.get('itunes_episodetype', 'full'),
                                    "season": entry.get('itunes_season'),
                                    "episode": entry.get('itunes_episode'),
                                    "explicit": entry.get('itunes_explicit', False),
                                    "keywords": entry.get('tags', []),
                                    "has_subtitles": False
                                },
                                "blocked_download": False,
                                "total_chunks": None,
                                "chunks_processed": 0,
                                "chunks_status": {}
                            }
                            
                            # Add to database
                            content = db.add_content(content_data)
                            
                            batch_indexed += 1
                            self.logger.debug(f"Added episode {episode_id}: {entry.title}")
                            
                            # Update progress
                            if progress_callback:
                                status_msg = f"Indexed {indexed_count} new episodes"
                                if skipped_count > 0:
                                    status_msg += f" (skipped {skipped_count} existing episodes)"
                                status_msg += f" from {channel_name}"
                                
                                progress_callback({
                                    "status": status_msg,
                                    "channel_name": channel_name,
                                    "current_episode": indexed_count + skipped_count,
                                    "total_episodes": total_episodes
                                })
                        except Exception as e:
                            self.logger.error(f"Error processing episode {entry.get('title', 'unknown')}: {str(e)}")
                            continue
                    
                    # Commit the batch
                    try:
                        session.commit()
                        total_committed += batch_indexed
                        indexed_count += batch_indexed
                        skipped_count += batch_skipped
                        if batch_indexed > 0:  # Only log if we actually indexed something
                            self.logger.info(f"Batch {batch_number}: {batch_indexed} episodes committed")
                    except Exception as e:
                        self.logger.error(f"Error committing batch {batch_number}: {str(e)}")
                        session.rollback()
                        continue
            
            # After indexing is complete, log task event
            duration = time.time() - start_time
            
            # Log to task logger for central logging
            self.task_logger.info(f"Indexed podcast feed: {channel_name} (had {existing_count} episodes, found {indexed_count} new, committed {total_committed}) in {duration:.1f}s", extra={
                'task_event': True,
                'task_id': f"index_podcast_{standardize_source_name(feed_url)}",
                'content_id': feed_url,
                'duration': duration,
                'channel_name': channel_name,
                'existing_count': existing_count,
                'new_count': indexed_count,
                'committed_count': total_committed,
                'total_count': existing_count + total_committed,
                'component': 'podcast_indexer'
            })
            
            # Log to file logger for detailed logging
            self.logger.info(
                f"Completed indexing feed {channel_name}:\n"
                f"  - URL: {feed_url}\n"
                f"  - Existing episodes: {existing_count}\n"
                f"  - New episodes: {indexed_count}\n"
                f"  - Successfully committed: {total_committed}\n"
                f"  - Skipped episodes: {skipped_count}\n"
                f"  - Total episodes: {existing_count + total_committed}\n"
                f"  - Duration: {duration:.1f}s"
            )
            
            # Prepare final status message
            status_msg = f"Completed {channel_name}: "
            if indexed_count > 0:
                status_msg += f"added {indexed_count} new episodes (had {existing_count} before)"
                if skipped_count > 0:
                    status_msg += f", skipped {skipped_count} existing episodes"
            else:
                status_msg += "no new episodes found"
            
            return {
                "status": "success",
                "channel": channel_name,
                "total_episodes": total_episodes,
                "indexed_count": indexed_count,
                "existing_count": existing_count,
                "total_count": existing_count + indexed_count,
                "skipped_count": skipped_count,
                "duration": duration,
                "message": status_msg
            }
            
        except Exception as e:
            self.logger.error(f"Error indexing podcast feed {feed_url}: {str(e)}")
            return {
                "status": "error",
                "channel": feed_url,
                "error": str(e)
            }
        finally:
            if self.session:
                await self.session.close()
                self.session = None
    
    def get_latest_episode_date(self, feed_url: str, project: str) -> Optional[datetime]:
        """Get the date of the most recently indexed episode for a feed"""
        try:
            with get_session() as session:
                # Set a longer statement timeout (120 seconds)
                session.execute(text('SET statement_timeout = 120000'))
                
                # Use explicit index hint and execution options
                latest_content = session.query(Content)\
                    .filter(
                        Content.platform == 'podcast',
                        Content.channel_url == feed_url,
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
            self.logger.error(f"Error getting latest episode date: {str(e)}")
            return None 