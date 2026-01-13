from typing import List, Dict, Optional, Union
from datetime import datetime, timedelta
from sqlalchemy import or_, and_, func, exists
from sqlalchemy.orm import Session
from sqlalchemy.sql import text
import pandas as pd
from pathlib import Path
from src.utils.paths import get_project_root, get_config_path

from .models import (
    Content, 
    Transcription,
    Source,
    TaskQueue,
    WorkerConfig
)
from ..utils.logger import setup_indexer_logger

logger = setup_indexer_logger('database')

class DatabaseManager:
    # Class-level cache shared across all DatabaseManager instances
    _task_cache = None
    _cache_enabled = False

    def __init__(self, session: Session):
        self.session = session
        self.logger = logger  # Use the logger imported from utils.logger

    @classmethod
    async def initialize_task_cache(cls, config: Dict):
        """Initialize the class-level task queue cache (call once at orchestrator startup)"""
        cls._cache_enabled = config.get('processing', {}).get('task_cache_enabled', True)

        if not cls._cache_enabled:
            logger.info("Task cache disabled in config")
            return

        try:
            # Import here to avoid circular dependency
            import sys
            from pathlib import Path
            sys.path.append(str(get_project_root()))
            from src.orchestration.task_queue_cache import TaskQueueCache

            # Create a pseudo task_manager for cache
            # The cache needs a task_manager with get_next_tasks method
            class CacheTaskManagerAdapter:
                def __init__(self, config):
                    self.config = config
                    self.assigned_tasks = set()
                    self.blocked_task_types = set()

                async def get_next_tasks(self, limit=100, task_types=None, exclude_task_ids=None):
                    """Adapter method that calls DatabaseManager's get_pending_tasks_async"""
                    from ..database.session import get_session

                    with get_session() as session:
                        db = DatabaseManager(session)
                        tasks_objs = db.get_pending_tasks(limit=limit, allowed_types=task_types)

                        # Convert to dict format
                        task_list = []
                        for task in tasks_objs:
                            if exclude_task_ids and str(task.id) in exclude_task_ids:
                                continue

                            task_dict = {
                                'id': task.id,
                                'content_id': task.content_id,
                                'task_type': task.task_type,
                                'priority': task.priority,
                                'input_data': task.input_data or {},
                                'created_at': task.created_at,
                                'attempts': getattr(task, 'attempts', 0)
                            }
                            task_list.append(task_dict)

                        return task_list

            adapter = CacheTaskManagerAdapter(config)

            prefetch_size = config.get('processing', {}).get('task_cache_prefetch_size', 100)
            refresh_threshold = config.get('processing', {}).get('task_cache_refresh_threshold', 20)
            ttl_seconds = config.get('processing', {}).get('task_cache_ttl_seconds', 60)

            cls._task_cache = TaskQueueCache(
                task_manager=adapter,
                prefetch_size=prefetch_size,
                refresh_threshold=refresh_threshold,
                ttl_seconds=ttl_seconds
            )

            await cls._task_cache.start()
            logger.info("Task queue cache initialized and started for DatabaseManager")

        except Exception as e:
            logger.error(f"Failed to initialize task cache: {str(e)}")
            cls._task_cache = None
            cls._cache_enabled = False

    @classmethod
    async def shutdown_task_cache(cls):
        """Shutdown the class-level task queue cache"""
        if cls._task_cache:
            await cls._task_cache.stop()
            logger.info("Task queue cache stopped")
    
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            self.session.close()
            
    def close(self):
        """Explicitly close the session"""
        if self.session:
            self.session.close()
            self.session = None
    
    def add_content(self, content_data: Dict) -> Content:
        """Add new content to database or update if exists"""
        # Convert single project to list if needed
        if 'project' in content_data:
            content_data['projects'] = [content_data.pop('project')]
        elif not isinstance(content_data.get('projects'), list):
            # If projects is a string (legacy), convert to list
            content_data['projects'] = [content_data['projects']] if content_data.get('projects') else []
        
        try:
            # Check if content already exists
            existing = self.session.query(Content).filter_by(
                platform=content_data['platform'],
                content_id=content_data['content_id']
            ).first()
            
            if existing:
                # Update existing content
                self.logger.debug(f"Updating existing content: {content_data['content_id']}")

                # Update projects if needed (merge with existing)
                if content_data.get('projects'):
                    existing_projects = set(existing.projects) if existing.projects else set()
                    new_projects = set(content_data['projects']) if isinstance(content_data['projects'], list) else {content_data['projects']}
                    content_data['projects'] = list(existing_projects | new_projects)

                # Special handling for meta_data
                if content_data.get('meta_data'):
                    if existing.meta_data:
                        # Merge existing and new metadata
                        existing_meta = existing.meta_data if isinstance(existing.meta_data, dict) else {}
                        new_meta = content_data['meta_data']
                        merged_meta = {**existing_meta, **new_meta}
                        content_data['meta_data'] = merged_meta

                # IMPORTANT: Do not overwrite main_language if it's already set
                # Language should only be set once during initial creation or by explicit admin tools
                # This prevents re-indexing from reverting manually corrected languages
                if existing.main_language:
                    content_data.pop('main_language', None)

                # IMPORTANT: Do not overwrite processing state flags when re-indexing
                # These flags track pipeline progress and should never be reset by the indexer
                # The indexer always passes is_downloaded=False for new content, but we must
                # preserve the actual state for existing content
                processing_state_fields = {
                    'is_downloaded', 'is_converted', 'is_transcribed', 'is_diarized',
                    'is_stitched', 'is_embedded', 'is_compressed', 'blocked_download'
                }
                for field in processing_state_fields:
                    content_data.pop(field, None)

                # Update all fields
                for key, value in content_data.items():
                    setattr(existing, key, value)
                content = existing
            else:
                # Create new content
                self.logger.debug(f"Adding new content: {content_data['content_id']}")
                content = Content(**content_data)
                self.session.add(content)
            
            self.session.commit()
            return content
            
        except Exception as e:
            self.logger.error(f"Error adding/updating content: {str(e)}")
            self.session.rollback()
            raise

    def _get_platform_priority(self, platform: str) -> int:
        """Get processing priority for a platform"""
        priorities = {
            'podcast': 3,
            'rumble': 2,
            'youtube': 1
        }
        return priorities.get(platform.lower(), 0)

    def _calculate_duplicate_confidence(self, content_data: Dict, potential_duplicate: Content) -> float:
        """Calculate confidence score for duplicate content match"""
        confidence = 0.0
        
        # Compare titles (removing common words and punctuation)
        title1 = self._normalize_text(content_data['title'])
        title2 = self._normalize_text(potential_duplicate.title)
        title_similarity = self._calculate_text_similarity(title1, title2)
        confidence += title_similarity * 0.4  # Title similarity has 40% weight
        
        # Compare durations
        if abs(content_data['duration'] - potential_duplicate.duration) <= 30:  # Within 30 seconds
            confidence += 0.3  # Duration match has 30% weight
        
        # Compare publish dates
        if abs((content_data['publish_date'] - potential_duplicate.publish_date).total_seconds()) <= 86400:  # Within 24 hours
            confidence += 0.3  # Publish date match has 30% weight
        
        return confidence

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison by removing common patterns and words"""
        import re
        
        # Convert to lowercase
        text = text.lower()
        
        # Common words to remove
        common_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'episode', 'ep', 'part', 'pt', 'video', 'podcast',
            'show', 'full', 'complete', 'official', 'hd', '4k'
        }
        
        # Remove common prefixes/suffixes patterns
        patterns_to_remove = [
            r'^ep\.?\s*\d+\s*[-:]\s*',  # Episode number prefixes (ep. 1 -, ep1:, etc.)
            r'^episode\s*\d+\s*[-:]\s*',  # Episode prefixes
            r'^part\s*\d+\s*[-:]\s*',    # Part number prefixes
            r'^pt\.?\s*\d+\s*[-:]\s*',   # Pt. prefixes
            r'^#\d+\s*[-:]\s*',          # #123 style prefixes
            r'\s*\[.*?\]',               # Anything in square brackets
            r'\s*\(.*?\)',               # Anything in parentheses
            r'\s*\|.*$',                 # Anything after a pipe symbol
            r'\s*-\s*.*$',               # Anything after a dash
            r'\s*:\s*.*$',               # Anything after a colon
            r'\s+\d{1,3}\s*$',           # Episode numbers at the end
            r'\s+HD\s*$',                # HD suffix
            r'\s+4K\s*$',                # 4K suffix
            r'^\d{1,3}\.\s*',            # Leading numbers with dots
            r'[\(\)\[\]\{\}]',           # Remove any remaining brackets
            r'[^\w\s]',                  # Remove any remaining punctuation
        ]
        
        # Apply all removal patterns
        for pattern in patterns_to_remove:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Split into words and remove common words
        words = text.split()
        words = [w for w in words if w not in common_words]
        
        # Remove any remaining numbers
        words = [w for w in words if not w.isdigit()]
        
        # Join words back together
        return ' '.join(words).strip()

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts using Levenshtein distance"""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, text1, text2).ratio()

    def _find_potential_duplicate(self, content_data: Dict) -> Optional[Content]:
        """Find potential duplicate content based on title, duration, and publish date"""
        # Get content with similar duration (within 30 seconds)
        duration_min = content_data['duration'] - 30
        duration_max = content_data['duration'] + 30
        
        # Get content published within 24 hours
        date_min = content_data['publish_date'] - timedelta(days=1)
        date_max = content_data['publish_date'] + timedelta(days=1)
        
        # Query for potential duplicates
        potential_duplicates = self.session.query(Content).filter(
            Content.platform != content_data['platform'],  # Different platform
            Content.duration.between(duration_min, duration_max),  # Similar duration
            Content.publish_date.between(date_min, date_max),  # Similar publish date
            Content.is_duplicate == False  # Not already marked as duplicate
        ).all()
        
        # Find best match based on title similarity
        best_match = None
        best_similarity = 0
        title1 = self._normalize_text(content_data['title'])
        
        for content in potential_duplicates:
            title2 = self._normalize_text(content.title)
            similarity = self._calculate_text_similarity(title1, title2)
            if similarity > best_similarity and similarity > 0.8:  # Require at least 80% title similarity
                best_similarity = similarity
                best_match = content
        
        return best_match

    def get_content_to_process(self, project: str, date_range: Optional[Dict] = None) -> List[Content]:
        """Get content that needs processing, excluding duplicates or using highest priority version"""
        query = self.session.query(Content).filter(
            Content.projects.any(project)
        )
        
        if date_range:
            if date_range.get('start'):
                query = query.filter(Content.publish_date >= date_range['start'])
            if date_range.get('end'):
                query = query.filter(Content.publish_date <= date_range['end'])
        
        # Get non-duplicate content and highest priority version of duplicate content
        query = query.filter(
            or_(
                Content.is_duplicate == False,
                and_(
                    Content.is_duplicate == True,
                    ~exists().where(
                        and_(
                            Content.duplicate_of_id == Content.id,
                            Content.processing_priority > Content.processing_priority
                        )
                    )
                )
            )
        ).order_by(Content.processing_priority.desc())
        
        return query.all()

    def get_unique_content_count(self, project: str) -> int:
        """Get count of unique content (excluding duplicates)"""
        return self.session.query(Content).filter(
            Content.projects.any(project),
            Content.is_duplicate == False
        ).count()

    def get_unique_content_duration(self, project: str) -> float:
        """Get total duration of unique content in hours"""
        result = self.session.query(func.sum(Content.duration))\
            .filter(
                Content.projects.any(project),
                Content.is_duplicate == False
            ).scalar()
        return float(result or 0) / 3600  # Convert seconds to hours
    
    def get_content_by_id(self, content_id: str) -> Optional[Content]:
        """Get content by ID"""
        return self.session.query(Content).filter_by(content_id=content_id).first()
    
    def get_content_by_url(self, url: str) -> Optional[Content]:
        """Get content by URL"""
        return self.session.query(Content).filter_by(url=url).first()
    
    def get_content_for_project(
        self,
        project: str,
        date_range: Optional[Dict] = None,
        limit: Optional[int] = None
    ) -> List[Content]:
        """Get all content for a project with optional filters"""
        query = self.session.query(Content).filter(
            Content.projects.any(project)  # Use LIKE for substring matching
        )
        
        if date_range:
            if date_range.get('start'):
                query = query.filter(Content.publish_date >= date_range['start'])
            if date_range.get('end'):
                query = query.filter(Content.publish_date <= date_range['end'])
        
        if limit:
            query = query.limit(limit)
            
        return query.all()
    
    def add_project_to_content(self, content_id: str, project: str) -> bool:
        """Add a project to content's project list if not already present"""
        try:
            content = self.session.query(Content).filter(Content.id == content_id).first()
            if not content:
                return False
                
            projects = set(content.projects) if content.projects else set()
            if project not in projects:
                projects.add(project)
                content.projects = list(projects)
                self.session.commit()
            return True
        except Exception as e:
            logger.error(f"Error adding project to content: {str(e)}")
            self.session.rollback()
            return False
    
    def remove_project_from_content(self, content_id: str, project: str) -> bool:
        """Remove a project from content's project list"""
        try:
            content = self.session.query(Content).filter(Content.id == content_id).first()
            if not content:
                return False
                
            projects = set(content.projects) if content.projects else set()
            if project in projects:
                projects.remove(project)
                content.projects = list(projects) if projects else []
                self.session.commit()
            return True
        except Exception as e:
            logger.error(f"Error removing project from content: {str(e)}")
            self.session.rollback()
            return False
    
    def add_transcription(self, transcription_data: Dict) -> Transcription:
        """Add transcription to database"""
        try:
            # Start transaction
            self.session.begin_nested()
            
            # Create transcription record
            transcription = Transcription(
                content_id=transcription_data['content_id'],
                full_text=transcription_data['full_text'],
                segments=transcription_data['segments'],
                model_version=transcription_data.get('model_version'),
                processing_status=transcription_data.get('processing_status', 'processed')
            )
            self.session.add(transcription)
            
            try:
                # First try to commit just the transcription
                self.session.flush()
                
                # If transcription was added successfully, update content flag
                content = self.session.query(Content).filter_by(id=transcription_data['content_id']).first()
                if content:
                    content.is_transcribed = True
                    content.updated_at = datetime.utcnow()
                
                # Commit the entire transaction
                self.session.commit()
                return transcription
                
            except Exception as e:
                self.session.rollback()
                self.logger.error(f"Error adding transcription: {str(e)}")
                raise
                
        except Exception as e:
            self.session.rollback()
            self.logger.error(f"Error in transcription transaction: {str(e)}")
            raise
    
    def get_transcription(self, transcription_id: int) -> Optional[Transcription]:
        """Get transcription by ID"""
        return self.session.query(Transcription).filter_by(id=transcription_id).first()
    
    def get_transcriptions_for_content(self, content_id: int) -> List[Transcription]:
        """Get all transcriptions for a piece of content"""
        return self.session.query(Transcription).filter_by(content_id=content_id).all()
    
    def get_transcriptions_for_entity_detection(
        self,
        project: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Transcription]:
        """Get transcriptions that need entity detection"""
        query = self.session.query(Transcription)\
            .join(Content)\
            .filter(Content.projects.any(project))
        
        if start_date:
            query = query.filter(Content.publish_date >= start_date)
        if end_date:
            query = query.filter(Content.publish_date <= end_date)
            
        return query.all()
    
    def debug_check_podcasts(self, project: str) -> None:
        """Debug function to directly check podcast status in database"""
        try:
            # Raw SQL query to check podcast counts with detailed info
            sql = """
            SELECT 
                COUNT(*) as total,
                platform,
                is_downloaded,
                blocked_download,
                publish_date::date as pub_date
            FROM content 
            WHERE projects LIKE :project 
            AND platform = 'podcast'
            GROUP BY platform, is_downloaded, blocked_download, pub_date
            ORDER BY pub_date DESC;
            """
            
            results = self.session.execute(sql, {'project': f'%{project}%'})
            self.logger.info("\n=== Podcast Status Report ===")
            for row in results:
                self.logger.info(
                    f"Count: {row.total}, "
                    f"Downloaded: {row.is_downloaded}, "
                    f"Blocked: {row.blocked_download}, "
                    f"Date: {row.pub_date}"
                )
            
            # Query specifically for undownloaded, unblocked podcasts
            sql_undownloaded = """
            SELECT 
                content_id,
                title,
                publish_date::date as pub_date,
                projects
            FROM content 
            WHERE projects LIKE :project 
            AND platform = 'podcast'
            AND is_downloaded = false
            AND blocked_download = false
            ORDER BY publish_date DESC;
            """
            
            results = self.session.execute(sql_undownloaded, {'project': f'%{project}%'})
            self.logger.info("\n=== Undownloaded Podcasts ===")
            for row in results:
                self.logger.info(
                    f"ID: {row.content_id}, "
                    f"Date: {row.pub_date}, "
                    f"Title: {row.title}"
                )
            
        except Exception as e:
            self.logger.error(f"Error in debug query: {str(e)}")

    def get_content_to_download(
        self,
        project: str,
        platform: str = None,
        date_range: Optional[Dict] = None,
        limit_per_platform: Optional[int] = 500
    ) -> List[Content]:
        """Get content that needs to be downloaded"""
        try:
            # Run debug check for podcasts
            if platform == 'podcast' or platform is None:
                self.debug_check_podcasts(project)
            
            # Debug: Check total podcast counts first
            total_podcasts = self.session.query(Content).filter(
                Content.projects.any(project),
                Content.platform == 'podcast'
            ).count()
            self.logger.info(f"Total podcasts in project: {total_podcasts}")
            
            not_downloaded_podcasts = self.session.query(Content).filter(
                Content.projects.any(project),
                Content.platform == 'podcast',
                Content.is_downloaded == False
            ).count()
            self.logger.info(f"Podcasts not yet downloaded: {not_downloaded_podcasts}")
            
            blocked_podcasts = self.session.query(Content).filter(
                Content.projects.any(project),
                Content.platform == 'podcast',
                Content.blocked_download == True
            ).count()
            self.logger.info(f"Blocked podcasts: {blocked_podcasts}")
            
            # Start with base query
            query = self.session.query(Content).filter(
                Content.projects.any(project),
                Content.is_downloaded == False,  # Not marked as downloaded in database
                Content.blocked_download == False  # Not blocked from downloading
            )
            
            # Debug log initial query count
            initial_count = query.count()
            self.logger.info(f"Initial query found {initial_count} items to download")
            
            # Add platform filter if specified
            if platform:
                query = query.filter(Content.platform == platform)
                platform_count = query.count()
                self.logger.info(f"After platform filter ({platform}): {platform_count} items")
            
            # Add date range filter if specified
            if date_range:
                if date_range.get('start'):
                    query = query.filter(Content.publish_date >= date_range['start'])
                    self.logger.info(f"After start date filter: {query.count()} items")
                if date_range.get('end'):
                    query = query.filter(Content.publish_date <= date_range['end'])
                    self.logger.info(f"After end date filter: {query.count()} items")
            
            # Debug log the SQL query
            self.logger.debug(f"SQL Query: {query}")
            
            # Order by priority and date
            query = query.order_by(
                Content.processing_priority.desc(),
                Content.publish_date.desc()
            )
            
            # Apply limit if specified
            if limit_per_platform:
                query = query.limit(limit_per_platform)
                self.logger.info(f"Limited to {limit_per_platform} items")
            
            results = query.all()
            self.logger.info(f"Final result count: {len(results)} items")
            
            # Debug log some sample results
            if results:
                sample = results[:5]
                self.logger.debug("Sample of results:")
                for item in sample:
                    self.logger.debug(f"  - {item.platform}: {item.title} ({item.publish_date})")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error getting content to download: {str(e)}")
            return []

    def get_content_to_extract_audio(
        self,
        project_name: str,
        platform: Optional[str] = None,
        date_range: Optional[Dict] = None
    ) -> List[Content]:
        """Get content that needs audio extraction"""
        try:
            query = self.session.query(Content).filter(
                Content.projects.any(project_name),
                Content.is_downloaded == True,
                Content.is_converted == False
            )
            
            # Add platform filter if specified
            if platform:
                query = query.filter(Content.platform == platform)
            
            # Add date range filter if specified
            if date_range:
                query = query.filter(
                    Content.publish_date >= date_range['start'],
                    Content.publish_date <= date_range['end']
                )
            
            # Order by download date if available, otherwise publish date
            query = query.order_by(
                Content.download_date.desc() if Content.download_date else Content.publish_date.desc()
            )
            
            return query.all()
            
        except Exception as e:
            self.logger.error(f"Error getting content for audio extraction: {str(e)}")
            return []
    
    def get_total_content_count(self, project: str) -> int:
        """Get total number of content items for a project"""
        return self.session.query(Content).filter(
            Content.projects.any(project)
        ).count()
    
    def get_transcription_by_content(self, content_id: str) -> Optional[Transcription]:
        """Get transcription for content"""
        content = self.get_content_by_id(content_id)
        if not content:
            return None
        
        # Get the most recent transcription
        return self.session.query(Transcription)\
            .filter(Transcription.content_id == content.id)\
            .order_by(Transcription.created_at.desc())\
            .first()
    
    def get_all_content(self, project_name: str) -> List[Content]:
        """Get all content for a project"""
        logger.debug(f"[DEBUG] Getting content for project: {project_name}")
        query = self.session.query(Content).filter(
            Content.projects.any(project_name)
        )
        results = query.all()
        logger.debug(f"[DEBUG] Found {len(results)} items in project {project_name}")
        return results
    
    def get_content_with_transcriptions(self, project: str) -> int:
        """Get count of content with transcriptions"""
        return self.session.query(Content)\
            .join(Transcription)\
            .filter(Content.projects.any(project))\
            .count()
    
    def get_latest_content(self, project: str, channel_url: str) -> Optional[Content]:
        """Get the most recently published content for a channel in a project"""
        self.logger.debug(f"Getting latest content for project '{project}' and channel '{channel_url}'")
        
        try:
            query = self.session.query(Content).filter(
                Content.projects.any(project),
                Content.channel_url == channel_url
            ).order_by(Content.publish_date.desc())
            
            result = query.first()
            if result:
                self.logger.info(f"Found latest content: {result.title} ({result.publish_date})")
            else:
                self.logger.info(f"No content found for channel {channel_url} in project {project}")
                # Debug query
                all_channel_content = self.session.query(Content).filter(
                    Content.channel_url == channel_url
                ).all()
                if all_channel_content:
                    self.logger.debug(f"Found {len(all_channel_content)} items for this channel in other projects:")
                    for content in all_channel_content:
                        self.logger.debug(f"  - Project: {content.projects}, Title: {content.title}, Date: {content.publish_date}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting latest content: {str(e)}")
            return None
    
    def get_total_duration(self, project: str) -> float:
        """Get total duration in hours of all content for a project"""
        result = self.session.query(func.sum(Content.duration))\
            .filter(Content.projects.any(project))\
            .scalar()
        return float(result or 0) / 3600  # Convert seconds to hours
    
    def get_duration_to_process(self, project: str, step: str) -> float:
        """Get total duration in hours of content that needs to be processed for a given step"""
        query = self.session.query(func.sum(Content.duration)).filter(
            Content.projects.any(project)
        )
        
        if step == 'download':
            query = query.filter(Content.is_downloaded == False)
        elif step == 'transcribe':
            query = query.outerjoin(Transcription)\
                .filter(Transcription.id == None)
                
        result = query.scalar()
        return float(result or 0) / 3600  # Convert seconds to hours
    
    def get_project_sources(self, project_name: str) -> Dict[str, List[str]]:
        """Get project sources from database"""
        sources = {}
        try:
            # Query sources for project
            query = self.session.query(Source).filter(
                Source.projects.any(project_name)
            )
            
            # Group by platform
            for source in query.all():
                platform = source.type.lower().strip()
                url = source.url.strip()
                if url:
                    sources.setdefault(platform, []).append(url)
            
            return sources
            
        except Exception as e:
            logger.error(f"Error getting project sources: {str(e)}")
            return sources

    def add_project_sources(self, project_name: str, sources_file: Path) -> bool:
        """Add project sources from CSV file to database"""
        try:
            df = pd.read_csv(sources_file)
            if df.empty:
                logger.error(f"No sources found in {sources_file}")
                return False
                
            # Add each source to database
            for _, row in df.iterrows():
                # Handle YouTube URLs
                if pd.notna(row.get('youtube')):
                    url = row['youtube'].strip()
                    if url:
                        # Check if source already exists
                        existing = self.session.query(Source).filter_by(url=url).first()
                        if existing:
                            # Update projects if needed
                            projects = set(existing.projects) if existing.projects else set()
                            if project_name not in projects:
                                projects.add(project_name)
                                existing.projects = list(projects)
                                existing.updated_at = datetime.utcnow()
                        else:
                            # Create new source
                            source = Source(
                                name=row['channel_name'],
                                type='youtube',
                                url=url,
                                projects=[project_name],
                                description=row.get('description', '')
                            )
                            self.session.add(source)
                
                # Handle Podcast URLs
                if pd.notna(row.get('podcast')):
                    url = row['podcast'].strip()
                    if url:
                        # Check if source already exists
                        existing = self.session.query(Source).filter_by(url=url).first()
                        if existing:
                            # Update projects if needed
                            projects = set(existing.projects) if existing.projects else set()
                            if project_name not in projects:
                                projects.add(project_name)
                                existing.projects = list(projects)
                                existing.updated_at = datetime.utcnow()
                        else:
                            # Create new source
                            source = Source(
                                name=row['channel_name'],
                                type='podcast',
                                url=url,
                                projects=[project_name],
                                description=row.get('description', '')
                            )
                            self.session.add(source)
            
            self.session.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error adding project sources: {str(e)}")
            self.session.rollback()
            return False

    def get_content_to_transcribe(
        self,
        project_name: str,
        platform: Optional[str] = None,
        date_range: Optional[Dict] = None,
        order_by_recent: bool = True
    ) -> List[Content]:
        """Get content that needs transcription (has WAV audio but no transcription)"""
        try:
            # Base query for content in project
            query = self.session.query(Content).filter(
                Content.projects.any(project_name),
                Content.is_duplicate == False,
                # Content must be converted to WAV
                Content.is_converted == True,
                # And not already transcribed
                Content.is_transcribed == False
            )
            
            # Add platform filter if specified
            if platform:
                query = query.filter(Content.platform == platform)
            
            # Add date range filter if specified
            if date_range:
                query = query.filter(
                    Content.publish_date >= date_range['start'],
                    Content.publish_date <= date_range['end']
                )
            
            # Order by publish date if requested
            if order_by_recent:
                query = query.order_by(Content.publish_date.desc())
            
            return query.all()
            
        except Exception as e:
            self.logger.error(f"Error getting content to transcribe: {str(e)}")
            return []

    def mark_download_blocked(self, content_id: str) -> bool:
        """Mark content download as blocked"""
        content = self.get_content_by_id(content_id)
        if content:
            content.blocked_download = True
            try:
                self.session.commit()
                return True
            except Exception as e:
                self.logger.error(f"Error marking download blocked for {content_id}: {str(e)}")
                self.session.rollback()
        return False

    def get_pending_tasks(self, limit: Optional[int] = None, allowed_types: Optional[List[str]] = None) -> List[TaskQueue]:
        """Get pending tasks, optionally filtering by type and applying a limit."""
        try:
            query = self.session.query(TaskQueue)\
                .filter(TaskQueue.status == 'pending')

            # Apply optional filter for allowed task types
            if allowed_types:
                 if not isinstance(allowed_types, list):
                      self.logger.warning("allowed_types should be a list, attempting to use anyway.")
                 # Ensure list is not empty before applying filter
                 if allowed_types:
                      query = query.filter(TaskQueue.task_type.in_(allowed_types))

            # Apply ordering by priority and creation time
            query = query.order_by(TaskQueue.priority.desc(), TaskQueue.created_at.asc())

            # Apply optional limit
            if limit is not None and isinstance(limit, int) and limit > 0:
                 query = query.limit(limit)
            elif limit is not None:
                 self.logger.warning(f"Invalid limit '{limit}' provided, ignoring.")

            return query.all()
        except Exception as e:
            self.logger.error(f"Error getting pending tasks: {str(e)}")
            return []

    async def get_pending_task_cached(self, allowed_types: Optional[List[str]] = None, exclude_task_ids: Optional[set] = None) -> Optional[TaskQueue]:
        """
        Get single pending task using cache (fast path for reactive assignment).

        Args:
            allowed_types: List of acceptable task types
            exclude_task_ids: Set of task IDs to skip

        Returns:
            TaskQueue object or None
        """
        if not self.__class__._cache_enabled or not self.__class__._task_cache or not allowed_types:
            # Fallback to direct DB query
            tasks = self.get_pending_tasks(limit=1, allowed_types=allowed_types)
            return tasks[0] if tasks else None

        # Try cache first
        task_dict = await self.__class__._task_cache.get_next_task(allowed_types, exclude_task_ids)

        if task_dict:
            # Convert dict back to TaskQueue object for V1 compatibility
            task_obj = self.session.query(TaskQueue).filter_by(id=task_dict['id']).first()
            return task_obj

        # Cache miss - fallback to DB
        self.logger.debug(f"Cache miss for task types {allowed_types}, falling back to DB")
        tasks = self.get_pending_tasks(limit=1, allowed_types=allowed_types)
        return tasks[0] if tasks else None

    def get_in_progress_tasks(self) -> List[TaskQueue]:
        """Get all in-progress tasks"""
        try:
            return self.session.query(TaskQueue)\
                .filter(TaskQueue.status == 'in_progress')\
                .order_by(TaskQueue.started_at.asc())\
                .all()
        except Exception as e:
            self.logger.error(f"Error getting in-progress tasks: {str(e)}")
            return []

    def get_task_counts(self) -> Dict[str, int]:
        """Get counts of tasks by status"""
        try:
            counts = {}
            for status in ['pending', 'in_progress', 'done', 'error']:
                count = self.session.query(TaskQueue)\
                    .filter(TaskQueue.status == status)\
                    .count()
                counts[status] = count
            return counts
        except Exception as e:
            self.logger.error(f"Error getting task counts: {str(e)}")
            return {}