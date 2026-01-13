"""State manager for handling database updates."""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, List
from sqlalchemy import text
from .session import get_session
from .models import Content, ContentChunk, TaskQueue, Transcription
from ..utils.logger import setup_worker_logger
from contextlib import contextmanager
import time
import json
import tempfile
from collections import defaultdict
from pathlib import Path
from src.utils.paths import get_project_root, get_config_path
from dataclasses import dataclass, field
from typing import DefaultDict

@dataclass
class StateSummary:
    """Tracks state changes across projects"""
    projects: DefaultDict[str, Dict[str, Dict[str, int]]] = field(
        default_factory=lambda: defaultdict(
            lambda: {
                'content': defaultdict(int),
                'chunks': defaultdict(int),
                'transcripts': defaultdict(int)
            }
        )
    )
    logger: logging.Logger = field(default_factory=lambda: setup_worker_logger('state_summary'))
    
    def add_state_change(self, project: str, category: str, state_type: str, count: int):
        """Record state changes for a project/category/type combination"""
        self.projects[project][category][state_type] += count
        
    def print_summary(self):
        """Print a formatted summary of all state changes"""
        if not self.projects:
            self.logger.info("\n📊 No state changes detected")
            return
            
        self.logger.info("\n📊 State Change Summary")
        self.logger.info("=" * 70)
        
        # Calculate column widths
        project_width = max(len("Project"), max(len(p) for p in self.projects.keys()))
        category_width = 12  # Fixed width for categories
        type_width = max(
            len("State Type"),
            max(len(t) for p in self.projects.values() 
                for c in p.values() 
                for t in c.keys())
        )
        count_width = max(
            len("Count"),
            max(len(str(c)) for p in self.projects.values()
                for cat in p.values()
                for c in cat.values())
        )
        
        # Print header
        header = f"{'Project':<{project_width}} | {'Category':<{category_width}} | {'State Type':<{type_width}} | {'Count':>{count_width}}"
        self.logger.info(header)
        self.logger.info("-" * len(header))
        
        # Print each project's state changes
        total_changes = 0
        for project, categories in sorted(self.projects.items()):
            project_total = sum(count for cat in categories.values() for count in cat.values())
            total_changes += project_total
            
            # Print each category and its states
            for category, states in sorted(categories.items()):
                category_total = sum(states.values())
                
                # Print each state type
                for state_type, count in sorted(states.items()):
                    if count > 0:  # Only show non-zero counts
                        self.logger.info(
                            f"{project:<{project_width}} | "
                            f"{category:<{category_width}} | "
                            f"{state_type:<{type_width}} | "
                            f"{count:>{count_width}}"
                        )
                
                # Print category subtotal if it has multiple states
                if len([c for c in states.values() if c > 0]) > 1:
                    self.logger.info("-" * len(header))
                    self.logger.info(
                        f"{'':<{project_width}} | "
                        f"{category:<{category_width}} | "
                        f"{'SUBTOTAL':<{type_width}} | "
                        f"{category_total:>{count_width}}"
                    )
                    self.logger.info("-" * len(header))
            
            # Print project total if it has multiple categories
            if len([cat for cat in categories.values() if sum(cat.values()) > 0]) > 1:
                self.logger.info("=" * len(header))
                self.logger.info(
                    f"{project:<{project_width}} | "
                    f"{'TOTAL':<{category_width}} | "
                    f"{'':<{type_width}} | "
                    f"{project_total:>{count_width}}"
                )
                self.logger.info("=" * len(header))
        
        # Print grand total if multiple projects
        if len([p for p in self.projects.values() if sum(sum(cat.values()) for cat in p.values()) > 0]) > 1:
            self.logger.info("=" * len(header))
            self.logger.info(
                f"{'TOTAL':<{project_width}} | "
                f"{'(all)':<{category_width}} | "
                f"{'':<{type_width}} | "
                f"{total_changes:>{count_width}}"
            )
            self.logger.info("=" * len(header))

class StateManager:
    """Centralized manager for handling database state updates."""
    
    def __init__(self, logger=None):
        self.logger = logger or setup_worker_logger('state_manager')
        
    @contextmanager
    def _get_session_with_timeout(self, timeout_ms: int = 300000):
        """Get a database session with specified timeout."""
        with get_session() as session:
            try:
                # Set statement timeout
                session.execute(text(f'SET statement_timeout = {timeout_ms}'))
                yield session
                session.commit()
            except Exception as e:
                session.rollback()
                raise e
            
    async def update_download_status(self, content_id: str, success: bool, error: Optional[str] = None, permanent_block: bool = False) -> bool:
        """Update content download status.
        
        Args:
            content_id: The content ID
            success: Whether download succeeded
            error: Optional error message
            permanent_block: Whether this is a permanent block (e.g. 403 error)
        """
        try:
            with self._get_session_with_timeout() as session:
                content = session.query(Content).filter_by(content_id=content_id).first()
                if content:
                    # Get current metadata
                    meta = content.meta_data or {}
                    
                    if success:
                        # On successful download, verify content exists in S3 before updating status
                        paths = content.get_s3_paths()
                        
                        # Initialize S3 storage
                        from ..storage.s3_utils import S3StorageConfig, S3Storage
                        import yaml
                        from pathlib import Path

                        # Load config
                        config_path = get_config_path()
                        with open(config_path) as f:
                            config = yaml.safe_load(f)

                        s3_config = S3StorageConfig(
                            endpoint_url=config['storage']['s3']['endpoint_url'],
                            access_key=config['storage']['s3']['access_key'],
                            secret_key=config['storage']['s3']['secret_key'],
                            bucket_name=config['storage']['s3']['bucket_name'],
                            use_ssl=config['storage']['s3']['use_ssl']
                        )
                        s3_storage = S3Storage(s3_config)
                        
                        # Only update status if file exists in S3
                        if s3_storage.file_exists(paths['source']):
                            content.is_downloaded = True
                            content.s3_source_key = paths['source']
                            
                            # Only remove blocked status if not a permanent block
                            if not content.meta_data.get('permanent_block', False):
                                content.blocked_download = False
                                
                            # Clear error related fields but preserve other metadata
                            meta = content.meta_data.copy() if content.meta_data else {}
                            for key in ['download_error', 'download_attempts', 'retry_after']:
                                meta.pop(key, None)
                            content.meta_data = meta
                        else:
                            # File doesn't exist in S3, don't update status
                            self.logger.warning(f"Download marked as success but file not found in S3 for {content_id}")
                            return False
                    else:
                        # Handle failed download
                        content.is_downloaded = False
                        attempts = meta.get('download_attempts', 0) + 1
                        
                        # Update metadata for failure
                        meta.update({
                            'download_error': error,
                            'last_attempt': datetime.now(timezone.utc).isoformat(),
                            'download_attempts': attempts,
                            'permanent_block': permanent_block
                        })
                        
                        # Set blocked status based on attempts and permanent flag
                        if permanent_block:
                            content.blocked_download = True
                        else:
                            # Block after 3 attempts, but allow retry after 20 days
                            content.blocked_download = attempts >= 3
                            if content.blocked_download:
                                meta['retry_after'] = (
                                    datetime.now(timezone.utc) + timedelta(days=20)
                                ).isoformat()
                        
                        content.meta_data = meta
                    
                    content.last_updated = datetime.now(timezone.utc)
                    return True
                return False
        except Exception as e:
            self.logger.error(f"Error updating download status for {content_id}: {str(e)}")
            return False
            
    async def update_conversion_status(self, content_id: str, total_chunks: int, chunk_data: List[Dict]) -> bool:
        """Update content conversion status and create chunk plan records."""
        try:
            with self._get_session_with_timeout() as session:
                content = session.query(Content).filter_by(content_id=content_id).first()
                if not content:
                    return False
                    
                # Update content status - note is_converted will be set based on actual chunk completion
                content.total_chunks = total_chunks
                content.chunks_processed = 0
                content.chunks_status = {}
                content.last_updated = datetime.now(timezone.utc)
                
                # Create chunk records for any missing chunks
                existing_chunks = {
                    chunk.chunk_index: chunk
                    for chunk in session.query(ContentChunk).filter_by(content_id=content.id).all()
                }
                
                # Create or update chunk records
                for chunk in chunk_data:
                    chunk_index = chunk['index']
                    if chunk_index not in existing_chunks:
                        content_chunk = ContentChunk(
                            content_id=content.id,
                            chunk_index=chunk_index,
                            start_time=chunk['start_time'],
                            end_time=chunk['end_time'],
                            duration=chunk['duration'],
                            status='pending'
                        )
                        session.add(content_chunk)
                        content.chunks_status[str(chunk_index)] = 'pending'
                    else:
                        # Update existing chunk if needed
                        existing_chunk = existing_chunks[chunk_index]
                        if existing_chunk.status == 'failed':
                            existing_chunk.status = 'pending'
                            content.chunks_status[str(chunk_index)] = 'pending'
                
                return True
                
        except Exception as e:
            self.logger.error(f"Error updating conversion status for {content_id}: {str(e)}")
            return False
            
    async def update_chunk_extraction_status(self, content_id: str, chunk_index: int, status: str, 
                                          error: Optional[str] = None,
                                          worker_id: Optional[str] = None) -> bool:
        """Update extraction status of a specific chunk."""
        try:
            with self._get_session_with_timeout() as session:
                content = session.query(Content).filter_by(content_id=content_id).first()
                if not content:
                    return False
                    
                # Update chunk status
                chunk = session.query(ContentChunk).filter_by(
                    content_id=content.id,
                    chunk_index=chunk_index
                ).first()
                
                if not chunk:
                    self.logger.error(f"Chunk {chunk_index} not found for content {content_id}")
                    return False
                
                # Get S3 configuration
                from ..storage.s3_utils import S3StorageConfig, S3Storage
                from ..storage.content_storage import ContentStorageManager
                import yaml
                from pathlib import Path

                # Load config
                config_path = get_config_path()
                with open(config_path) as f:
                    config = yaml.safe_load(f)

                s3_config = S3StorageConfig(
                    endpoint_url=config['storage']['s3']['endpoint_url'],
                    access_key=config['storage']['s3']['access_key'],
                    secret_key=config['storage']['s3']['secret_key'],
                    bucket_name=config['storage']['s3']['bucket_name'],
                    use_ssl=config['storage']['s3']['use_ssl']
                )
                s3_storage = S3Storage(s3_config)
                
                # Get all chunks for this content in one S3 list operation
                chunks_prefix = f"content/{content_id}/chunks/"
                s3_chunks = set()
                for key in s3_storage.list_files(chunks_prefix):
                    try:
                        parts = Path(key).parts
                        if len(parts) >= 5 and parts[-1] == 'audio.wav':
                            try:
                                chunk_idx = int(parts[-2])  # Get index from directory name
                                s3_chunks.add(chunk_idx)
                            except ValueError:
                                continue
                    except (ValueError, IndexError):
                        continue

                # Check if the specific chunk exists in S3
                chunk_exists = chunk_index in s3_chunks
                
                # Update status based on S3 state
                if status == 'processing':
                    chunk.mark_extraction_processing(worker_id)
                elif status == 'completed':
                    # Only mark as completed if chunk exists in S3
                    if chunk_exists:
                        chunk.mark_extraction_completed()
                    else:
                        self.logger.error(f"Cannot mark chunk {chunk_index} as completed - file not found in S3")
                        chunk.mark_extraction_failed("Chunk file not found in S3")
                        return False
                elif status == 'failed':
                    chunk.mark_extraction_failed(error)
                
                # Update all chunk statuses based on S3
                all_chunks = session.query(ContentChunk).filter_by(content_id=content.id).all()
                completed_chunks = 0
                total_chunks = len(all_chunks)
                
                # Batch update all chunks based on S3 state
                for db_chunk in all_chunks:
                    if db_chunk.chunk_index in s3_chunks:
                        if db_chunk.extraction_status != 'completed':
                            db_chunk.mark_extraction_completed()
                        completed_chunks += 1
                    elif db_chunk.extraction_status == 'completed':
                        # If marked completed in DB but not in S3, mark as failed
                        db_chunk.mark_extraction_failed("Chunk file missing from S3")
                
                # Update content status
                content.total_chunks = total_chunks
                content.chunks_processed = completed_chunks
                content.is_converted = (completed_chunks == total_chunks and total_chunks > 0)
                content.last_updated = datetime.now(timezone.utc)
                
                # Update metadata with chunk status
                meta = content.meta_data or {}
                meta.update({
                    'total_chunks': total_chunks,
                    'completed_chunks': completed_chunks,
                    'last_chunk_update': datetime.now(timezone.utc).isoformat(),
                    'chunk_status': {
                        'completed': completed_chunks,
                        'total': total_chunks,
                        'percent_complete': round((completed_chunks / total_chunks * 100) if total_chunks > 0 else 0, 2)
                    }
                })
                content.meta_data = meta
                
                if content.is_converted:
                    self.logger.info(f"Marked content {content_id} as converted with {total_chunks} chunks completed")
                elif completed_chunks < total_chunks:
                    self.logger.info(f"Content {content_id} partially converted: {completed_chunks}/{total_chunks} chunks ({meta['chunk_status']['percent_complete']}%)")
                
                return True
                
        except Exception as e:
            self.logger.error(f"Error updating chunk extraction status for {content_id}:{chunk_index}: {str(e)}")
            return False

    async def update_chunk_transcription_status(self, content_id: str, chunk_index: int, status: str,
                                             error: Optional[str] = None,
                                             worker_id: Optional[str] = None) -> bool:
        """Update transcription status of a specific chunk."""
        try:
            with self._get_session_with_timeout() as session:
                content = session.query(Content).filter_by(content_id=content_id).first()
                if not content:
                    return False
                    
                # Update chunk status
                chunk = session.query(ContentChunk).filter_by(
                    content_id=content.id,
                    chunk_index=chunk_index
                ).first()
                
                if not chunk:
                    self.logger.error(f"Chunk {chunk_index} not found for content {content_id}")
                    return False

                # Get S3 configuration
                from ..storage.s3_utils import S3StorageConfig, S3Storage
                import yaml
                from pathlib import Path

                # Load config
                config_path = get_config_path()
                with open(config_path) as f:
                    config = yaml.safe_load(f)

                s3_config = S3StorageConfig(
                    endpoint_url=config['storage']['s3']['endpoint_url'],
                    access_key=config['storage']['s3']['access_key'],
                    secret_key=config['storage']['s3']['secret_key'],
                    bucket_name=config['storage']['s3']['bucket_name'],
                    use_ssl=config['storage']['s3']['use_ssl']
                )
                s3_storage = S3Storage(s3_config)
                
                # Get all transcripts for this content in one S3 list operation
                chunks_prefix = f"content/{content_id}/chunks/"
                s3_transcripts = set()
                for key in s3_storage.list_files(chunks_prefix):
                    try:
                        parts = Path(key).parts
                        if len(parts) >= 5 and parts[-1] == 'transcript_words.json':  # Changed to transcript_words.json
                            try:
                                chunk_idx = int(parts[-2])  # Get index from directory name
                                s3_transcripts.add(chunk_idx)
                            except ValueError:
                                continue
                    except (ValueError, IndexError):
                        continue

                # Check if the specific chunk exists in S3
                chunk_has_transcript = chunk_index in s3_transcripts
                
                # Update status based on S3 state
                if status == 'processing':
                    chunk.mark_transcription_processing(worker_id)
                elif status == 'completed':
                    # Only mark as completed if transcript exists in S3
                    if chunk_has_transcript:
                        chunk.mark_transcription_completed()
                    else:
                        self.logger.error(f"Cannot mark chunk {chunk_index} as completed - transcript_words.json not found in S3")  # Updated error message
                        chunk.mark_transcription_failed("transcript_words.json file not found in S3")  # Updated error message
                        return False
                elif status == 'failed':
                    chunk.mark_transcription_failed(error)
                
                # Update all chunk statuses based on S3
                all_chunks = session.query(ContentChunk).filter_by(content_id=content.id).all()
                completed_chunks = 0
                total_chunks = len(all_chunks)
                
                # Batch update all chunks based on S3 state
                for db_chunk in all_chunks:
                    if db_chunk.chunk_index in s3_transcripts:
                        if db_chunk.transcription_status != 'completed':
                            db_chunk.mark_transcription_completed()
                        completed_chunks += 1
                    elif db_chunk.transcription_status == 'completed':
                        # If marked completed in DB but not in S3, mark as failed
                        db_chunk.mark_transcription_failed("transcript_words.json file missing from S3")  # Updated error message
                
                # Update content status
                content.chunks_processed = completed_chunks
                
                # Check if all chunks are in 'completed' or 'skipped' status
                all_chunks_transcribed = all(
                    chunk.transcription_status in ['completed', 'skipped'] 
                    for chunk in all_chunks
                )
                
                # Mark as transcribed only if all chunks are transcribed
                if completed_chunks == total_chunks and all_chunks_transcribed:
                    content.is_transcribed = True
                    self.logger.info(f"Content {content_id} marked as transcribed - all {total_chunks} chunks have transcripts")
                else:
                    content.is_transcribed = False
                    self.logger.info(f"Content {content_id} not fully transcribed - {completed_chunks}/{total_chunks} chunks have transcripts")
                
                content.last_updated = datetime.now(timezone.utc)
                session.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"Error updating chunk transcription status for {content_id}:{chunk_index}: {str(e)}")
            return False

    async def update_transcription_status(self, content_id: str, success: bool) -> bool:
        """Update content transcription status.
        
        This method checks if all chunks have been transcribed and updates the is_transcribed flag.
        Note: transcript.json is no longer used, as each chunk has its own transcript file.
        """
        try:
            with self._get_session_with_timeout() as session:
                content = session.query(Content).filter_by(content_id=content_id).first()
                if not content:
                    self.logger.error(f"Content not found: {content_id}")
                    return False
                
                if not success:
                    # If explicitly marked as not successful, update flag and return
                    content.is_transcribed = False
                    content.last_updated = datetime.now(timezone.utc)
                    return True
                
                # Check if all chunks are processed and have transcriptions
                chunks = session.query(ContentChunk).filter_by(content_id=content.id).all()
                
                if not chunks:
                    self.logger.error(f"Cannot mark content {content_id} as transcribed - no chunks found")
                    return False
                
                # All chunks must have a transcription status of 'completed' or 'skipped'
                all_chunks_transcribed = all(
                    chunk.transcription_status in ['completed', 'skipped'] 
                    for chunk in chunks
                )
                
                # Check if the chunks_processed matches total_chunks 
                chunks_match = (content.chunks_processed is not None and 
                               content.total_chunks is not None and 
                               content.chunks_processed == content.total_chunks)
                
                # Update the transcribed status based on chunk transcription status
                content.is_transcribed = all_chunks_transcribed and chunks_match
                content.last_updated = datetime.now(timezone.utc)
                
                if content.is_transcribed:
                    self.logger.info(f"Content {content_id} marked as transcribed - all {len(chunks)} chunks transcribed")
                else:
                    self.logger.info(f"Content {content_id} not marked as transcribed - {sum(1 for c in chunks if c.transcription_status in ['completed', 'skipped'])}/{len(chunks)} chunks transcribed")
                
                return True
        except Exception as e:
            self.logger.error(f"Error updating transcription status for {content_id}: {str(e)}")
            return False

    async def update_stitched_transcript(self, content_id: str, result_data: Dict) -> bool:
        """Update content with stitched transcript data and mark as stitched.
        
        Args:
            content_id: The content ID
            result_data: Dictionary containing transcript data including:
                - duration: Total duration in seconds
                - num_segments: Number of segments
                - has_word_timestamps: Whether word timestamps are present
                - text: Full transcript text
                - segments: List of transcript segments
                - output_path: S3 path of the transcript_diarized.json file
        """
        try:
            with self._get_session_with_timeout() as session:
                content = session.query(Content).filter_by(content_id=content_id).first()
                if not content:
                    self.logger.error(f"Content not found: {content_id}")
                    return False

                # Check if output path is provided (should be transcript_diarized.json)
                output_path = result_data.get('output_path')
                if not output_path or 'transcript_diarized.json' not in output_path:
                    self.logger.error(f"Missing or incorrect output_path in result_data for {content_id}. Expected path to transcript_diarized.json.")
                    return False
                    
                # Verify the output file exists in S3 before proceeding
                from ..storage.s3_utils import S3StorageConfig, S3Storage
                import yaml
                from pathlib import Path

                config_path = get_config_path()
                with open(config_path) as f:
                    config = yaml.safe_load(f)

                s3_config = S3StorageConfig(
                    endpoint_url=config['storage']['s3']['endpoint_url'],
                    access_key=config['storage']['s3']['access_key'],
                    secret_key=config['storage']['s3']['secret_key'],
                    bucket_name=config['storage']['s3']['bucket_name'],
                    use_ssl=config['storage']['s3']['use_ssl']
                )
                s3_storage = S3Storage(s3_config)
                
                if not s3_storage.file_exists(output_path):
                    self.logger.error(f"Cannot mark content {content_id} as stitched. Final file {output_path} not found in S3.")
                    # Optionally set an error in metadata
                    meta = content.meta_data or {}
                    meta['stitch_error'] = f"Post-task check failed: Missing S3 file: {output_path}\n"
                    content.meta_data = meta
                    content.is_stitched = False # Ensure flag is false
                    content.last_updated = datetime.now(timezone.utc)
                    return False

                # Create new transcription record (optional, could be replaced by reading the diarized file later)
                # For now, let's assume we want to store it for quick access
                transcription = Transcription(
                    content_id=content.id,
                    full_text=result_data['text'],
                    segments=result_data['segments'],
                    model_version='whisper_diarized', # Indicate source
                    processing_status='processed'
                )
                session.add(transcription)

                # Update content metadata
                meta = content.meta_data or {}
                meta.update({
                    'transcript_duration': result_data['duration'],
                    'transcript_segments': result_data['num_segments'],
                    'has_word_timestamps': result_data['has_word_timestamps'],
                    'last_stitched': datetime.now(timezone.utc).isoformat()
                })
                # Clear previous stitch error if any
                meta.pop('stitch_error', None)
                content.meta_data = meta
                content.last_updated = datetime.now(timezone.utc)

                # Update S3 paths and mark as stitched
                content.s3_transcript_key = output_path # Store the path to the final stitched file
                content.is_stitched = True # Mark stitching as complete
                # content.is_diarized should have been set by the previous step

                self.logger.info(f"Updated content {content_id} with stitched transcript info. Marked is_stitched=True.")
                return True

        except Exception as e:
            self.logger.error(f"Error updating stitched transcript for {content_id}: {str(e)}")
            return False

    async def update_content_status(self, content_id: str, is_converted: bool = None, 
                                  total_chunks: int = None, chunks_processed: int = None,
                                  is_transcribed: bool = None,
                                  is_diarized: bool = None,
                                  is_stitched: bool = None
                                  ) -> bool:
        """Update content status fields."""
        try:
            with self._get_session_with_timeout() as session:
                content = session.query(Content).filter_by(content_id=content_id).first()
                if not content:
                    return False
                    
                # Update basic fields
                if is_converted is not None:
                    content.is_converted = is_converted
                if total_chunks is not None:
                    content.total_chunks = total_chunks
                if chunks_processed is not None:
                    content.chunks_processed = chunks_processed
                if is_diarized is not None:
                    content.is_diarized = is_diarized
                
                # Update transcribed status - don't touch s3_transcript_key
                # Transcription now happens at the chunk level, no global transcript.json
                if is_transcribed is not None:
                    content.is_transcribed = is_transcribed
                
                # Update stitched status last (highest priority)
                if is_stitched is not None:
                    content.is_stitched = is_stitched
                    if is_stitched:
                        # When marking as stitched, always point to the stitched file
                        stitched_path = f"content/{content_id}/transcript_diarized.json"
                        content.s3_transcript_key = stitched_path
                
                content.last_updated = datetime.now(timezone.utc)
                return True
                
        except Exception as e:
            self.logger.error(f"Error updating content status for {content_id}: {str(e)}")
            return False

    async def check_and_update_states(self) -> Dict:
        """Periodic check of content states and update if needed."""
        results = {
            'checked': 0,
            'updated': 0,
            'errors': []
        }
        
        try:
            with self._get_session_with_timeout() as session:
                # Break the query into smaller, indexed queries
                # 1. Check converted but no chunks
                converted_no_chunks = session.query(Content).filter(
                    Content.is_converted == True,
                    Content.total_chunks == None
                ).limit(100).all()
                
                # 2. Check completed chunks but not transcribed
                completed_not_transcribed = session.query(Content).filter(
                    Content.chunks_processed > 0,
                    Content.chunks_processed == Content.total_chunks,
                    Content.is_transcribed == False
                ).limit(100).all()
                
                # 3. Check transcribed but no transcript.json
                transcribed_no_file = session.query(Content).filter(
                    Content.is_transcribed == True
                ).limit(100).all()
                
                # 4. Check content not updated recently
                not_recently_updated = session.query(Content).filter(
                    Content.last_updated < datetime.now(timezone.utc) - timedelta(hours=24)
                ).limit(100).all()
                
                # Combine results
                contents = list(set(converted_no_chunks + completed_not_transcribed + transcribed_no_file + not_recently_updated))
                
                # Initialize S3 storage once for all checks
                from ..storage.s3_utils import S3StorageConfig, S3Storage
                import yaml

                # Load config
                config_path = get_config_path()
                with open(config_path) as f:
                    config = yaml.safe_load(f)

                s3_config = S3StorageConfig(
                    endpoint_url=config['storage']['s3']['endpoint_url'],
                    access_key=config['storage']['s3']['access_key'],
                    secret_key=config['storage']['s3']['secret_key'],
                    bucket_name=config['storage']['s3']['bucket_name'],
                    use_ssl=config['storage']['s3']['use_ssl']
                )
                s3_storage = S3Storage(s3_config)

                for content in contents:
                    results['checked'] += 1
                    try:
                        paths = content.get_s3_paths()
                        # Check for the *final* diarized transcript
                        diarized_transcript_path = f"content/{content.content_id}/transcript_diarized.json"
                        diarized_transcript_exists = s3_storage.file_exists(diarized_transcript_path)
                        
                        # Check state related to is_diarized flag and transcript_diarized.json
                        if content.is_diarized and not diarized_transcript_exists:
                            content.is_diarized = False
                            # Potentially clear s3_transcript_key if it points to the diarized one?
                            # content.s3_transcript_key = None 
                            content.last_updated = datetime.now(timezone.utc)
                            results['updated'] += 1
                            self.logger.warning(f"Content {content.content_id} marked as diarized but transcript_diarized.json not found. Resetting flag.")
                        elif not content.is_diarized and diarized_transcript_exists:
                            content.is_diarized = True
                            content.s3_transcript_key = diarized_transcript_path # Ensure key points to the correct file
                            content.last_updated = datetime.now(timezone.utc)
                            results['updated'] += 1
                            self.logger.info(f"Found transcript_diarized.json for {content.content_id}, updating is_diarized status.")

                        # You might want to keep checks for 'is_transcribed' based on chunk transcripts if needed
                        # For example, to see if all chunks are done BEFORE stitching
                        # transcript_exists = s3_storage.file_exists(paths['transcript']) # Check for original transcript.json if needed

                        # Check state related to is_stitched flag and transcript_diarized.json
                        stitched_path = f"content/{content.content_id}/transcript_diarized.json"
                        stitched_file_exists = s3_storage.file_exists(stitched_path)
                        if content.is_stitched and not stitched_file_exists:
                            content.is_stitched = False
                            content.last_updated = datetime.now(timezone.utc)
                            results['updated'] += 1
                            self.logger.warning(f"Content {content.content_id} marked as stitched but {stitched_path} not found. Resetting flag.")
                        elif not content.is_stitched and stitched_file_exists:
                            content.is_stitched = True
                            content.s3_transcript_key = stitched_path # Ensure key points to final file
                            content.last_updated = datetime.now(timezone.utc)
                            results['updated'] += 1
                            self.logger.info(f"Found {stitched_path} for {content.content_id}, updating is_stitched status.")
                            
                        # Check state related to is_diarized flag based on diarization.json and speaker_embeddings.json
                        diarization_path = f"content/{content.content_id}/diarization.json"
                        embeddings_path = f"content/{content.content_id}/speaker_embeddings.json"
                        diarization_files_exist = s3_storage.file_exists(diarization_path) and s3_storage.file_exists(embeddings_path)
                        
                        if content.is_diarized and not diarization_files_exist:
                             content.is_diarized = False
                             content.last_updated = datetime.now(timezone.utc)
                             results['updated'] += 1
                             self.logger.warning(f"Content {content.content_id} marked as diarized but diarization/embedding files not found. Resetting flag.")
                        elif not content.is_diarized and diarization_files_exist:
                             content.is_diarized = True
                             content.last_updated = datetime.now(timezone.utc)
                             results['updated'] += 1
                             self.logger.info(f"Found diarization/embedding files for {content.content_id}, updating is_diarized status.")
                             
                        # Continue with other state checks...
                        # ... existing code for checking chunks and other states ...

                    except Exception as e:
                        error_msg = f"Error processing {content.content_id}: {str(e)}"
                        self.logger.error(error_msg)
                        results['errors'].append(error_msg)
                        continue

                return results

        except Exception as e:
            self.logger.error(f"Error in state check: {str(e)}")
            results['errors'].append(str(e))
            return results

    def initialize_legacy_content(self, content_id: str, platform: str = 'youtube') -> bool:
        """Initialize legacy content with proper metadata and chunk plan."""
        try:
            with self._get_session_with_timeout() as session:
                content = session.query(Content).filter_by(content_id=content_id).first()
                if not content:
                    return False

                # Check if content exists in S3
                from ..storage.s3_utils import S3StorageConfig, S3Storage
                from ..storage.content_storage import ContentStorageManager
                import yaml
                from pathlib import Path

                # Load config
                config_path = get_config_path()
                with open(config_path) as f:
                    config = yaml.safe_load(f)

                s3_config = S3StorageConfig(
                    endpoint_url=config['storage']['s3']['endpoint_url'],
                    access_key=config['storage']['s3']['access_key'],
                    secret_key=config['storage']['s3']['secret_key'],
                    bucket_name=config['storage']['s3']['bucket_name'],
                    use_ssl=config['storage']['s3']['use_ssl']
                )
                s3_storage = S3Storage(s3_config)
                storage_manager = ContentStorageManager(s3_storage)

                # Get S3 paths
                paths = content.get_s3_paths()

                # Check if source file exists in S3
                if not s3_storage.file_exists(paths['source']):
                    #self.logger.error(f"Source file not found in S3 for {content_id}")
                    return False

                # Create and upload metadata if it doesn't exist
                if not s3_storage.file_exists(paths['meta']):
                    # Ensure we have a valid metadata dictionary
                    existing_meta = content.meta_data or {}
                    
                    # Create standardized metadata structure
                    metadata = {
                        # Core fields (required)
                        'platform': platform,
                        'content_id': content.content_id,
                        'title': content.title or '',
                        'description': content.description or '',
                        'channel_name': content.channel_name or '',
                        'channel_url': content.channel_url or '',
                        'publish_date': content.publish_date.isoformat() if content.publish_date else None,
                        'duration': content.duration or 0,
                        'source_extension': '.mp4' if platform == 'youtube' else '.mp3',
                        
                        # Platform-specific stats (from existing metadata or defaults)
                        'view_count': existing_meta.get('view_count', 0),
                        'like_count': existing_meta.get('like_count', 0),
                        'comment_count': existing_meta.get('comment_count', 0),
                        
                        # Processing metadata
                        'processing_priority': existing_meta.get('processing_priority', 1),
                        'processing_status': existing_meta.get('processing_status', 'pending'),
                        'processing_attempts': existing_meta.get('processing_attempts', 0),
                        'last_processing_error': existing_meta.get('last_processing_error', None),
                        
                        # Migration metadata
                        'migrated_at': datetime.now(timezone.utc).isoformat(),
                        'migration_source': 'legacy',
                        'original_filename': existing_meta.get('original_filename', f"{content_id}{'.mp4' if platform == 'youtube' else '.mp3'}"),
                        'original_source': existing_meta.get('original_source', 'unknown'),
                        
                        # Transcription metadata (if exists)
                        'transcript_duration': existing_meta.get('transcript_duration'),
                        'total_segments': existing_meta.get('total_segments'),
                        'has_word_timestamps': existing_meta.get('has_word_timestamps', False),
                        'transcription_model': existing_meta.get('transcription_model'),
                        'stitched_at': existing_meta.get('stitched_at'),
                        
                        # Additional fields from existing metadata
                        'video_url': existing_meta.get('video_url', f"https://www.youtube.com/watch?v={content_id}" if platform == 'youtube' else None),
                        'audio_url': existing_meta.get('audio_url'),
                        'episode_url': existing_meta.get('episode_url'),
                        'download_error': existing_meta.get('download_error'),
                        'last_attempt': existing_meta.get('last_attempt'),
                        
                        # Timestamps
                        'created_at': existing_meta.get('created_at', content.created_at.isoformat() if content.created_at else None),
                        'updated_at': datetime.now(timezone.utc).isoformat()
                    }
                    
                    # Create temporary meta.json file
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                        json.dump(metadata, f, indent=2)
                        meta_path = f.name

                    # Upload metadata
                    if s3_storage.upload_file(meta_path, paths['meta']):
                        self.logger.debug(f"Updated metadata for {content.content_id}")
                        content.meta_data = metadata
                        content.s3_metadata_key = paths['meta']
                    else:
                        self.logger.error(f"Failed to upload metadata for {content.content_id}")

                    # Clean up temp file
                    Path(meta_path).unlink()
                    

                # Update content status
                content.is_downloaded = True
                content.s3_metadata_key = paths['meta']
                content.s3_source_key = paths['source']
                content.last_updated = datetime.now(timezone.utc)
                
                return True

        except Exception as e:
            self.logger.error(f"Error initializing legacy content {content_id}: {str(e)}")
            return False

    def _build_s3_map(self, s3_storage) -> Dict:
        """Build a comprehensive map of S3 content state efficiently."""
        s3_stats = {
            'total_files': 0,
            'total_size': 0,
            'by_type': defaultdict(int)
        }
        
        content_details = defaultdict(lambda: {
            'source': False,
            'source_path': None,
            'meta': False,
            'transcript': False,
            'transcript_path': None,
            'diarization': False, # Raw diarization output
            'diarization_path': None,
            'transcript_diarized': False, # Final stitched output
            'transcript_diarized_path': None,
            'embeddings': False, # Added
            'embeddings_path': None,
            'chunk_count': 0,
            'chunks': set(),
            'file_types': set()
        })
        
        content_with_source = set()
        content_with_transcript = set()
        content_with_diarized_transcript = set() # Added
        
        # Use paginator for efficient listing of large buckets
        paginator = s3_storage._client.get_paginator('list_objects_v2')
        
        self.logger.info("Starting S3 content scan...")
        
        try:
            for page in paginator.paginate(Bucket=s3_storage.config.bucket_name, Prefix='content/'):
                if 'Contents' not in page:
                    continue
                    
                for obj in page['Contents']:
                    key = obj['Key']
                    size = obj['Size']
                    s3_stats['total_files'] += 1
                    s3_stats['total_size'] += size
                    
                    try:
                        # Extract content_id from path
                        parts = Path(key).parts
                        if len(parts) >= 2 and parts[0] == 'content':
                            content_id = parts[1]
                            file_name = parts[-1]
                            
                            # Track file type
                            ext = Path(file_name).suffix.lower()
                            if ext:
                                ext = ext[1:]  # Remove the dot
                                content_details[content_id]['file_types'].add(ext)
                                s3_stats['by_type'][ext] += size
                            
                            # Track source files
                            if file_name.startswith('source.'):
                                content_details[content_id]['source'] = True
                                content_details[content_id]['source_path'] = key
                                content_with_source.add(content_id)
                            
                            # Track metadata files
                            elif file_name == 'meta.json':
                                content_details[content_id]['meta'] = True
                            
                            # Track transcript files - specifically transcript.json
                            elif file_name == 'transcript.json':
                                content_details[content_id]['transcript'] = True
                                content_details[content_id]['transcript_path'] = key
                                content_with_transcript.add(content_id)
                                
                            # Track diarization file
                            elif file_name == 'diarization.json':
                                content_details[content_id]['diarization'] = True
                                content_details[content_id]['diarization_path'] = key
                                
                            # Track speaker embeddings file
                            elif file_name == 'speaker_embeddings.json':
                                content_details[content_id]['embeddings'] = True
                                content_details[content_id]['embeddings_path'] = key
                                
                            # Track final diarized transcript file (NEW)
                            elif file_name == 'transcript_diarized.json':
                                content_details[content_id]['transcript_diarized'] = True
                                content_details[content_id]['transcript_diarized_path'] = key
                                content_with_diarized_transcript.add(content_id) # Added
                            
                            # Track chunks
                            elif len(parts) >= 5 and parts[-1] == 'audio.wav':
                                try:
                                    chunk_idx = int(parts[-2])
                                    content_details[content_id]['chunks'].add(chunk_idx)
                                    content_details[content_id]['chunk_count'] += 1
                                except ValueError:
                                    continue
                                
                    except (ValueError, IndexError) as e:
                        self.logger.error(f"Error processing S3 key {key}: {str(e)}")
                        continue
            
            self.logger.info(f"S3 scan complete. Found {len(content_with_source)} content items with source files")
            self.logger.info(f"Found {len(content_with_transcript)} content items with transcript.json")
            self.logger.info(f"Found {len(content_with_diarized_transcript)} content items with transcript_diarized.json") # Added logging
            
            return {
                'stats': s3_stats,
                'content_details': content_details,
                'content_with_source': content_with_source,
                'content_with_transcript': content_with_transcript,
                'content_with_diarized_transcript': content_with_diarized_transcript # Added
            }
            
        except Exception as e:
            self.logger.error(f"Error in _build_s3_map: {str(e)}")
            raise

    def get_processing_summary(self, session, project: str = None, start_date: datetime = None, end_date: datetime = None) -> Dict:
        """Get a clear summary of processing progress.
        
        Args:
            session: Database session
            project: Optional project name to filter by
            start_date: Optional start date to filter by
            end_date: Optional end date to filter by
            
        Returns:
            Dict containing processing statistics by project
        """
        query = text("""
            WITH RECURSIVE project_list AS (
                SELECT DISTINCT 
                    TRIM(unnest(string_to_array(COALESCE(projects, 'default'), ','))) as project_name
                FROM content 
                WHERE COALESCE(projects, 'default') != ''
            ),
            content_stats AS (
                SELECT 
                    p.project_name,
                    COUNT(*) as total_content,
                    COUNT(*) FILTER (WHERE is_downloaded = true) as downloaded_content,
                    COUNT(*) FILTER (WHERE is_converted = true) as converted_content,
                    COUNT(*) FILTER (WHERE is_transcribed = true) as transcribed_content,
                    COUNT(*) FILTER (WHERE is_diarized = true) as diarized_content, -- Added
                    COUNT(*) FILTER (WHERE is_stitched = true) as stitched_content -- Added
                FROM project_list p
                JOIN content c ON 
                    p.project_name = ANY(string_to_array(COALESCE(c.projects, 'default'), ','))
                    OR (p.project_name = 'default' AND c.projects IS NULL)
                WHERE 
                    (CASE WHEN :project IS NOT NULL 
                        THEN p.project_name = :project
                        ELSE true
                    END)
                    AND (CASE WHEN :start_date IS NOT NULL 
                        THEN c.publish_date >= :start_date
                        ELSE true
                    END)
                    AND (CASE WHEN :end_date IS NOT NULL 
                        THEN c.publish_date <= :end_date
                        ELSE true
                    END)
                GROUP BY p.project_name
            ),
            chunk_stats AS (
                SELECT 
                    p.project_name,
                    COUNT(*) as total_chunks,
                    COUNT(*) FILTER (WHERE cc.extraction_status = 'completed') as extracted_chunks,
                    COUNT(*) FILTER (WHERE cc.transcription_status = 'completed') as transcribed_chunks
                FROM project_list p
                JOIN content c ON 
                    p.project_name = ANY(string_to_array(COALESCE(c.projects, 'default'), ','))
                    OR (p.project_name = 'default' AND c.projects IS NULL)
                JOIN content_chunks cc ON c.id = cc.content_id
                WHERE 
                    (CASE WHEN :project IS NOT NULL 
                        THEN p.project_name = :project
                        ELSE true
                    END)
                    AND (CASE WHEN :start_date IS NOT NULL 
                        THEN c.publish_date >= :start_date
                        ELSE true
                    END)
                    AND (CASE WHEN :end_date IS NOT NULL 
                        THEN c.publish_date <= :end_date
                        ELSE true
                    END)
                GROUP BY p.project_name
            )
            SELECT 
                cs.project_name,
                cs.total_content,
                cs.downloaded_content,
                cs.converted_content,
                cs.transcribed_content,
                chs.total_chunks,
                chs.extracted_chunks,
                chs.transcribed_chunks,
                cs.diarized_content, -- Added
                cs.stitched_content -- Added
            FROM content_stats cs
            LEFT JOIN chunk_stats chs ON cs.project_name = chs.project_name
            ORDER BY cs.project_name
        """)
        
        results = session.execute(query, {
            'project': project,
            'start_date': start_date,
            'end_date': end_date
        }).fetchall()
        
        summaries = {}
        for row in results:
            # Calculate percentages
            extracted_pct = round((row.extracted_chunks / row.total_chunks * 100) if row.total_chunks else 0, 1)
            transcribed_pct = round((row.transcribed_chunks / row.total_chunks * 100) if row.total_chunks else 0, 1)
            # stitched_pct = round((row.transcribed_content / row.total_content * 100) if row.total_content else 0, 1)
            diarized_pct = round((row.diarized_content / row.total_content * 100) if row.total_content else 0, 1) # Use diarized count
            stitched_pct = round((row.stitched_content / row.total_content * 100) if row.total_content else 0, 1) # Use stitched count
            
            summaries[row.project_name] = {
                'content': {
                    'total': row.total_content,
                    'downloaded': row.downloaded_content,
                    'converted': row.converted_content,
                    'transcribed': row.transcribed_content, # Keep for chunk-level info
                    'diarized': row.diarized_content, # Added intermediate state
                    'stitched': row.stitched_content # Added final state
                },
                'chunks': {
                    'total': row.total_chunks or 0,
                    'extracted': f"{row.extracted_chunks or 0:,} ({extracted_pct}%)",
                    'transcribed': f"{row.transcribed_chunks or 0:,} ({transcribed_pct}%)"
                },
                'completion': {
                    # 'fully_stitched': f"{row.transcribed_content:,} ({stitched_pct}%)", # Old
                    'fully_diarized (Intermediate)': f"{row.diarized_content:,} ({diarized_pct}%)",
                    'fully_stitched (Final)': f"{row.stitched_content:,} ({stitched_pct}%)" # Use stitched count for final state
                }
            }
            
        return summaries

    async def complete_state_check(self, platform: str = None, project: str = None) -> Dict:
        """Complete state check and initialization of all content.
        
        Args:
            platform: Optional platform to filter content by
            project: Optional project to filter content by
            
        Returns:
            Dict containing analysis results
        """
        results = {
            'start_time': datetime.now(timezone.utc).isoformat(),
            'end_time': None,
            'errors': [],
            'inconsistencies': {
                'download_state': [],
                'conversion_state': [],
                'missing_metadata': [],
                'chunk_state': [],
                'chunk_transcript_state': [], # Renamed from transcript_state
                'diarization_state': [], # Now checks diarization.json + speaker_embeddings.json
                'stitch_state': [] # NEW: Checks final transcript_diarized.json
            },
            'stats': {
                'total_content_s3': 0,
                'total_content_db': 0,
                'total_chunks_s3': 0,
                'total_chunks_db': 0,
                'content_with_metadata': 0,
                'content_with_chunks': 0,
                'content_with_transcripts': 0, # Raw transcript.json
                'content_with_diarization': 0, # Raw diarization.json
                'content_with_embeddings': 0, # Raw speaker_embeddings.json
                'content_with_diarized_transcript': 0 # Added: Final transcript_diarized.json
            }
        }
        
        try:
            # Initialize S3 storage
            from ..storage.s3_utils import S3StorageConfig, S3Storage
            import yaml
            from pathlib import Path

            config_path = get_config_path()
            with open(config_path) as f:
                config = yaml.safe_load(f)
                
            # Get project date range if configured
            start_date = None
            end_date = None
            
            # Look for project settings if project is specified
            if project and project in config.get('active_projects', {}):
                settings = config['active_projects'][project]
                if settings.get('enabled', False):
                    start_date = datetime.strptime(settings['start_date'], '%Y-%m-%d').replace(tzinfo=timezone.utc) if settings.get('start_date') else None
                    end_date = datetime.strptime(settings['end_date'], '%Y-%m-%d').replace(tzinfo=timezone.utc) if settings.get('end_date') else datetime.now(timezone.utc)

            s3_config = S3StorageConfig(
                endpoint_url=config['storage']['s3']['endpoint_url'],
                access_key=config['storage']['s3']['access_key'],
                secret_key=config['storage']['s3']['secret_key'],
                bucket_name=config['storage']['s3']['bucket_name'],
                use_ssl=config['storage']['s3']['use_ssl']
            )
            s3_storage = S3Storage(s3_config)
            
            # Step 1: Build S3 state map
            self.logger.info("Building S3 state map...")
            s3_map = self._build_s3_map(s3_storage)
            content_details = s3_map['content_details']
            content_with_source = s3_map['content_with_source']
            
            # Update stats (add diarization/embeddings)
            results['stats'].update({
                'total_content_s3': len(content_details),
                'content_with_source': len(content_with_source),
                'content_with_metadata': sum(1 for d in content_details.values() if d['meta']),
                'content_with_chunks': sum(1 for d in content_details.values() if d['chunk_count'] > 0),
                'content_with_transcripts': sum(1 for d in content_details.values() if d['transcript']),
                'content_with_diarization': sum(1 for d in content_details.values() if d['diarization']), # Raw diarization
                'content_with_embeddings': sum(1 for d in content_details.values() if d['embeddings']), # Embeddings
                'content_with_diarized_transcript': sum(1 for d in content_details.values() if d['transcript_diarized']), # Added
                'total_chunks_s3': sum(d['chunk_count'] for d in content_details.values())
            })
            
            with self._get_session_with_timeout(timeout_ms=3600000) as session:
                # Get processing summary first
                summary = self.get_processing_summary(session, project, start_date, end_date)
                
                # Print clean status report
                self._print_status_report(summary, project)
                
                # Get all content records for inconsistency checks
                query = session.query(Content)
                if platform:
                    query = query.filter(Content.platform == platform)
                if project:
                    query = query.filter(Content.projects.any(project))
                all_content = query.all()
                results['stats']['total_content_db'] = len(all_content)
                
                # Get all chunk records efficiently
                chunk_query = text("""
                    SELECT 
                        c.content_id,
                        COUNT(*) as total_chunks,
                        COUNT(*) FILTER (WHERE extraction_status = 'completed') as completed_extraction,
                        COUNT(*) FILTER (WHERE transcription_status = 'completed') as completed_transcription
                    FROM content_chunks cc
                    JOIN content c ON c.id = cc.content_id
                    WHERE
                        CASE WHEN :project IS NOT NULL
                            THEN :project = ANY(c.projects)
                            ELSE true
                        END
                    GROUP BY c.content_id
                """)
                chunk_stats = {
                    row.content_id: {
                        'total': row.total_chunks,
                        'completed_extraction': row.completed_extraction,
                        'completed_transcription': row.completed_transcription
                    }
                    for row in session.execute(chunk_query, {'project': project})
                }
                results['stats']['total_chunks_db'] = sum(s['total'] for s in chunk_stats.values())
                
                # Analyze inconsistencies
                for content in all_content:
                    content_id = content.content_id
                    s3_state = content_details.get(content_id, {})
                    db_chunks = chunk_stats.get(content_id, {'total': 0, 'completed_extraction': 0})
                    
                    # Check download state
                    source_exists = s3_state.get('source', False)
                    if source_exists != content.is_downloaded:
                        results['inconsistencies']['download_state'].append({
                            'content_id': content_id,
                            'db_state': content.is_downloaded,
                            's3_state': source_exists,
                            'source_path': s3_state.get('source_path')
                        })
                    
                    # Check metadata state
                    if source_exists and not s3_state.get('meta', False):
                        results['inconsistencies']['missing_metadata'].append({
                            'content_id': content_id
                        })
                    
                    # Check chunk state
                    s3_chunks = s3_state.get('chunks', set())
                    if s3_chunks:
                        if len(s3_chunks) != db_chunks['completed_extraction']:
                            results['inconsistencies']['chunk_state'].append({
                                'content_id': content_id,
                                'db_completed': db_chunks['completed_extraction'],
                                's3_chunks': len(s3_chunks),
                                'chunk_indices': sorted(s3_chunks)
                            })
                        
                        # Check conversion state
                        is_fully_converted = len(s3_chunks) == db_chunks['total'] == db_chunks['completed_extraction']
                        if is_fully_converted != content.is_converted:
                            results['inconsistencies']['conversion_state'].append({
                                'content_id': content_id,
                                'db_state': content.is_converted,
                                'should_be': is_fully_converted,
                                'total_chunks': db_chunks['total'],
                                'completed_chunks': db_chunks['completed_extraction']
                            })
                    
                    # Check transcript state
                    has_transcript = s3_state.get('transcript', False)
                    if has_transcript != content.is_transcribed:
                        results['inconsistencies']['transcript_state'].append({
                            'content_id': content_id,
                            'db_state': content.is_transcribed,
                            's3_state': has_transcript
                        })

                    # Check diarization state
                    has_diarization_file = s3_state.get('diarization', False)
                    has_embeddings_file = s3_state.get('embeddings', False)
                    s3_diarization_complete = has_diarization_file and has_embeddings_file
                    
                    if content.is_diarized and not s3_diarization_complete:
                        results['inconsistencies']['diarization_state'].append({
                            'content_id': content_id,
                            'db_state': True,
                            's3_diarization': has_diarization_file,
                            's3_embeddings': has_embeddings_file
                        })
                    elif not content.is_diarized and s3_diarization_complete:
                        # Also report if DB says not diarized but files exist
                         results['inconsistencies']['diarization_state'].append({
                            'content_id': content_id,
                            'db_state': False,
                            's3_diarization': has_diarization_file,
                            's3_embeddings': has_embeddings_file
                        })
                
                    # Check chunk transcript state (based on individual transcript.json files)
                    has_all_chunk_transcripts = (db_chunks['total'] > 0 and 
                                                 db_chunks['total'] == db_chunks['completed_transcription'])
                    # Compare with is_transcribed flag (which should reflect chunk transcription completion)
                    if has_all_chunk_transcripts != content.is_transcribed:
                         results['inconsistencies']['chunk_transcript_state'].append({
                            'content_id': content_id,
                            'db_state': content.is_transcribed,
                            's3_state_based_on_chunks': has_all_chunk_transcripts,
                            'total_chunks': db_chunks['total'],
                            'transcribed_chunks': db_chunks['completed_transcription']
                        })

                    # Check DIARIZATION/STITCHING state (based on FINAL transcript_diarized.json)
                    has_diarized_transcript_file = s3_state.get('transcript_diarized', False)
                    
                    # Compare DB is_diarized flag with S3 transcript_diarized.json existence
                    if content.is_diarized != has_diarized_transcript_file:
                        results['inconsistencies']['diarization_state'].append({
                            'content_id': content_id,
                            'db_state': content.is_diarized,
                            's3_has_diarized_transcript': has_diarized_transcript_file,
                            's3_path': s3_state.get('transcript_diarized_path') # Include path for debugging
                        })
                
                    # Check DIARIZATION state (based on diarization.json + speaker_embeddings.json)
                    has_diarization_file = s3_state.get('diarization', False)
                    has_embeddings_file = s3_state.get('embeddings', False)
                    s3_diarization_step_complete = has_diarization_file and has_embeddings_file
                    if content.is_diarized != s3_diarization_step_complete:
                        results['inconsistencies']['diarization_state'].append({
                             'content_id': content_id,
                             'db_state': content.is_diarized,
                             's3_diarization': has_diarization_file,
                             's3_embeddings': has_embeddings_file
                         })
                         
                    # Check STITCH state (based on transcript_diarized.json)
                    has_stitched_file = s3_state.get('transcript_diarized', False)
                    if content.is_stitched != has_stitched_file:
                        results['inconsistencies']['stitch_state'].append({
                             'content_id': content_id,
                             'db_state': content.is_stitched,
                             's3_stitched': has_stitched_file,
                             's3_path': s3_state.get('transcript_diarized_path')
                         })
                         
                # Log summary of inconsistencies
                if any(results['inconsistencies'].values()):
                    self.logger.info("\n=== State Inconsistency Summary ===")
                    self.logger.info(f"Inconsistencies found:")
                    self.logger.info(f"- Download state mismatches: {len(results['inconsistencies']['download_state'])}")
                    self.logger.info(f"- Missing metadata files: {len(results['inconsistencies']['missing_metadata'])}")
                    self.logger.info(f"- Chunk state mismatches: {len(results['inconsistencies']['chunk_state'])}")
                    self.logger.info(f"- Conversion state mismatches: {len(results['inconsistencies']['conversion_state'])}")
                    self.logger.info(f"- Chunk Transcript state mismatches: {len(results['inconsistencies']['chunk_transcript_state'])}") # Updated label
                    self.logger.info(f"- Diarization state mismatches (Intermediate): {len(results['inconsistencies']['diarization_state'])}") # Updated label
                    self.logger.info(f"- Stitch state mismatches (Final Output): {len(results['inconsistencies']['stitch_state'])}") # Added
                    self.logger.info("\nTo fix these inconsistencies, call fix_state_inconsistencies() with the results from this analysis.")
                
                results['end_time'] = datetime.now(timezone.utc).isoformat()
                return results
                
        except Exception as e:
            self.logger.error(f"Error in complete state check: {str(e)}")
            results['errors'].append(str(e))
            results['end_time'] = datetime.now(timezone.utc).isoformat()
            return results

    def _print_status_report(self, summaries: Dict, project: str = None):
        """Print a clean status report using StateSummary format.
        
        Args:
            summaries: Dictionary containing processing statistics by project
            project: Optional project name to filter specific project
        """
        state_summary = StateSummary(logger=self.logger)
        
        # Add statistics for each project
        for project_name, summary in summaries.items():
            if project and project != project_name:
                continue
                
            # Content stats
            state_summary.add_state_change(project_name, 'content', 'total', summary['content']['total'])
            state_summary.add_state_change(project_name, 'content', 'downloaded', summary['content']['downloaded'])
            state_summary.add_state_change(project_name, 'content', 'converted', summary['content']['converted'])
            state_summary.add_state_change(project_name, 'content', 'transcribed', summary['content']['transcribed'])
            state_summary.add_state_change(project_name, 'content', 'diarized', summary['content']['diarized']) # Added
            state_summary.add_state_change(project_name, 'content', 'stitched', summary['content']['stitched']) # Added
            
            # Extract chunk counts from formatted strings
            extracted_chunks = int(summary['chunks']['extracted'].split()[0].replace(',', ''))
            transcribed_chunks = int(summary['chunks']['transcribed'].split()[0].replace(',', ''))
            
            # Chunk stats
            state_summary.add_state_change(project_name, 'chunks', 'total', summary['chunks']['total'])
            state_summary.add_state_change(project_name, 'chunks', 'extracted', extracted_chunks)
            state_summary.add_state_change(project_name, 'chunks', 'transcribed', transcribed_chunks)
            
            # Transcript stats (from fully_stitched -> fully_diarized)
            # stitched_count = int(summary['completion']['fully_stitched'].split()[0].replace(',', '')) # Old
            diarized_count = int(summary['completion']['fully_diarized (Intermediate)'].split()[0].replace(',', '')) # Use intermediate diarized
            stitched_count = int(summary['completion']['fully_stitched (Final)'].split()[0].replace(',', '')) # Use final stitched
            state_summary.add_state_change(project_name, 'transcripts', 'diarized', diarized_count) # Use new key/label
            state_summary.add_state_change(project_name, 'transcripts', 'stitched', stitched_count) # Added final state
        
        # Print the formatted summary
        state_summary.print_summary()
        
        # Add completion percentages in a separate section
        self.logger.info("\n📈 Completion Rates by Project")
        self.logger.info("=" * 50)
        
        for project_name, summary in summaries.items():
            if project and project != project_name:
                continue
                
            self.logger.info(f"\n{project_name}:")
            
            # Calculate percentages
            content_total = summary['content']['total']
            if content_total > 0:
                download_pct = (summary['content']['downloaded'] / content_total) * 100
                convert_pct = (summary['content']['converted'] / content_total) * 100
                transcribe_pct = (summary['content']['transcribed'] / content_total) * 100 # Chunk transcripts done
                diarized_pct = (summary['content']['diarized'] / content_total) * 100 # Diarization step done
                stitched_pct = (summary['content']['stitched'] / content_total) * 100 # Stitching step done
                
                self.logger.info(f"Content Downloaded:           {download_pct:.1f}%")
                self.logger.info(f"Content Converted:            {convert_pct:.1f}%")
                self.logger.info(f"Content Transcribed (Chunks): {transcribe_pct:.1f}%") # Clarified label
                self.logger.info(f"Content Diarized (Step):      {diarized_pct:.1f}%") # Added intermediate state
                self.logger.info(f"Content Stitched (Final):     {stitched_pct:.1f}%") # Added final state
            
            # Add chunk completion rates
            chunk_total = summary['chunks']['total']
            if chunk_total > 0:
                # Extract percentages from the formatted strings
                extracted_pct = float(summary['chunks']['extracted'].split('(')[1].strip('%)'))
                transcribed_pct = float(summary['chunks']['transcribed'].split('(')[1].strip('%)'))
                
                self.logger.info(f"Chunks Extracted:   {extracted_pct:.1f}%")
                self.logger.info(f"Chunks Transcribed: {transcribed_pct:.1f}%")
        
        self.logger.info("\n" + "=" * 50)

    def format_size(self, size_bytes: int) -> str:
        """Convert bytes to human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.2f} TB"

    async def fix_blocked_downloads(self) -> Dict:
        """Fix blocked_download flags by checking actual S3 content existence."""
        results = {
            'total_checked': 0,
            'fixed': 0,
            'errors': [],
            'start_time': datetime.now(timezone.utc).isoformat(),
            'source_files_found': 0,
            'database_mismatches': 0
        }
        
        try:
            # Initialize S3 storage
            from ..storage.s3_utils import S3StorageConfig, S3Storage
            import yaml
            from pathlib import Path

            # Load config
            config_path = get_config_path()
            with open(config_path) as f:
                config = yaml.safe_load(f)

            s3_config = S3StorageConfig(
                endpoint_url=config['storage']['s3']['endpoint_url'],
                access_key=config['storage']['s3']['access_key'],
                secret_key=config['storage']['s3']['secret_key'],
                bucket_name=config['storage']['s3']['bucket_name'],
                use_ssl=config['storage']['s3']['use_ssl']
            )
            s3_storage = S3Storage(s3_config)
            
            # Get all files in content directory
            self.logger.info("Scanning S3 storage for source files...")
            content_files = s3_storage.list_files('content/')
            
            # Build set of content IDs that have source files
            has_source = set()
            processed_files = 0
            
            for key in content_files:
                processed_files += 1
                if processed_files % 1000 == 0:
                    self.logger.info(f"Processed {processed_files} files...")
                    
                parts = Path(key).parts
                if len(parts) >= 2 and parts[0] == 'content':
                    content_id = parts[1]
                    file_name = parts[-1]
                    if file_name.startswith('source.'):
                        has_source.add(content_id)
            
            results['source_files_found'] = len(has_source)
            self.logger.info(f"Found {len(has_source)} content items with source files")
            
            # Update database in batches
            with self._get_session_with_timeout(timeout_ms=3600000) as session:  # 1 hour timeout
                # First get database state
                initial_stats = session.execute(text("""
                    SELECT 
                        COUNT(*) as total,
                        COUNT(*) FILTER (WHERE is_downloaded = true) as downloaded,
                        COUNT(*) FILTER (WHERE blocked_download = true) as blocked
                    FROM content
                """)).fetchone()
                
                self.logger.info("\nInitial Database State:")
                self.logger.info(f"Total content: {initial_stats.total}")
                self.logger.info(f"Downloaded: {initial_stats.downloaded}")
                self.logger.info(f"Blocked: {initial_stats.blocked}")
                
                # Get all content records
                query = text("""
                    SELECT id, content_id, blocked_download, is_downloaded
                    FROM content 
                """)
                all_content = session.execute(query).fetchall()
                
                self.logger.info(f"\nChecking {len(all_content)} database records against {len(has_source)} S3 source files")
                
                batch_size = 1000
                total_batches = (len(all_content) + batch_size - 1) // batch_size
                
                for batch_num, start_idx in enumerate(range(0, len(all_content), batch_size)):
                    batch = all_content[start_idx:start_idx + batch_size]
                    batch_updates = 0
                    
                    for row in batch:
                        results['total_checked'] += 1
                        try:
                            content_id = row.content_id
                            has_source_file = content_id in has_source
                            needs_update = False
                            
                            # Check if database state matches S3 state
                            if has_source_file and (not row.is_downloaded or row.blocked_download):
                                needs_update = True
                                results['database_mismatches'] += 1
                            elif not has_source_file and row.is_downloaded:
                                needs_update = True
                                results['database_mismatches'] += 1
                            
                            if needs_update:
                                update_query = text("""
                                    UPDATE content 
                                    SET 
                                        blocked_download = :blocked,
                                        is_downloaded = :downloaded,
                                        last_updated = :now
                                    WHERE id = :id
                                """)
                                session.execute(update_query, {
                                    'id': row.id,
                                    'blocked': not has_source_file,
                                    'downloaded': has_source_file,
                                    'now': datetime.now(timezone.utc)
                                })
                                results['fixed'] += 1
                                batch_updates += 1
                            
                        except Exception as e:
                            error_msg = f"Error processing content {row.content_id}: {str(e)}"
                            self.logger.error(error_msg)
                            results['errors'].append(error_msg)
                            continue
                    
                    # Commit batch and log progress
                    session.commit()
                    self.logger.info(
                        f"Batch {batch_num + 1}/{total_batches}: "
                        f"Processed {batch_size} records, Updated {batch_updates} records "
                        f"(Total fixed: {results['fixed']})"
                    )
                
                # Get final stats
                final_stats = session.execute(text("""
                    SELECT 
                        COUNT(*) as total,
                        COUNT(*) FILTER (WHERE is_downloaded = true) as downloaded,
                        COUNT(*) FILTER (WHERE blocked_download = true) as blocked
                    FROM content
                """)).fetchone()
                
                self.logger.info("\nFinal Database Status:")
                self.logger.info(f"Total content: {final_stats.total}")
                self.logger.info(f"Downloaded: {final_stats.downloaded}")
                self.logger.info(f"Blocked: {final_stats.blocked}")
                
                results['end_time'] = datetime.now(timezone.utc).isoformat()
                duration = (datetime.fromisoformat(results['end_time']) - 
                          datetime.fromisoformat(results['start_time'])).total_seconds() / 60
                
                self.logger.info(f"\nProcessing Summary:")
                self.logger.info(f"Duration: {duration:.1f} minutes")
                self.logger.info(f"Total records checked: {results['total_checked']}")
                self.logger.info(f"Source files found in S3: {results['source_files_found']}")
                self.logger.info(f"Database mismatches found: {results['database_mismatches']}")
                self.logger.info(f"Records fixed: {results['fixed']}")
                self.logger.info(f"Errors encountered: {len(results['errors'])}")
                
                # Verify final state matches S3
                if final_stats.downloaded != len(has_source):
                    self.logger.error(
                        f"WARNING: Final state mismatch - Database shows {final_stats.downloaded} "
                        f"downloaded but found {len(has_source)} source files in S3"
                    )
                
                return results
                
        except Exception as e:
            self.logger.error(f"Error fixing blocked downloads: {str(e)}")
            results['errors'].append(str(e))
            return results

    async def sync_with_s3(self) -> Dict:
        """Synchronize database state with S3 reality in one efficient pass."""
        results = {
            'start_time': datetime.now(timezone.utc).isoformat(),
            'end_time': None,
            'before_state': {},
            'after_state': {},
            'updates': {
                'content': 0,
                'chunks': 0,
                'transcripts': 0
            },
            'errors': []
        }
        
        try:
            # Initialize S3 storage
            from ..storage.s3_utils import S3StorageConfig, S3Storage
            import yaml
            from pathlib import Path

            config_path = get_config_path()
            with open(config_path) as f:
                config = yaml.safe_load(f)

            s3_config = S3StorageConfig(
                endpoint_url=config['storage']['s3']['endpoint_url'],
                access_key=config['storage']['s3']['access_key'],
                secret_key=config['storage']['s3']['secret_key'],
                bucket_name=config['storage']['s3']['bucket_name'],
                use_ssl=config['storage']['s3']['use_ssl']
            )
            s3_storage = S3Storage(s3_config)
            
            with self._get_session_with_timeout(timeout_ms=3600000) as session:
                # Get initial database state
                self.logger.info("Getting initial database state...")
                results['before_state'] = self._get_db_state(session)
                
                # Step 1: Build comprehensive S3 object map
                self.logger.info("Building S3 object map...")
                s3_map = defaultdict(lambda: {
                    'source': None,
                    'meta': None,
                    'diarization': None,
                    'speaker_embeddings': None,
                    'transcript_diarized': None,
                    'chunks': set(),
                    'chunk_transcripts': set()  # Track which chunks have transcripts
                })
                
                for key in s3_storage.list_files('content/'):
                    try:
                        parts = Path(key).parts
                        if len(parts) >= 2 and parts[0] == 'content':
                            content_id = parts[1]
                            file_name = parts[-1]
                            
                            # Map source files - only check for source.* pattern
                            if file_name.startswith('source.'):
                                s3_map[content_id]['source'] = key
                                self.logger.debug(f"Found source file: {key}")
                            
                            # Map metadata
                            elif file_name == 'meta.json':
                                s3_map[content_id]['meta'] = key
                            # Map diarization files
                            elif file_name == 'diarization.json':
                                s3_map[content_id]['diarization'] = key
                            # Map speaker embeddings
                            elif file_name == 'speaker_embeddings.json':
                                s3_map[content_id]['speaker_embeddings'] = key
                            # Map stitched transcript
                            elif file_name == 'transcript_diarized.json':
                                s3_map[content_id]['transcript_diarized'] = key
                            # Map chunks - use the same strict check as audio_extractor.py
                            elif len(parts) >= 5 and parts[-1] == 'audio.wav':
                                try:
                                    chunk_idx = int(parts[-2])  # Get index from directory name
                                    s3_map[content_id]['chunks'].add(chunk_idx)
                                except ValueError:
                                    continue
                            # Map chunk transcripts
                            elif len(parts) >= 5 and parts[-1] == 'transcript.json':
                                try:
                                    chunk_idx = int(parts[-2])  # Get index from directory name
                                    s3_map[content_id]['chunk_transcripts'].add(chunk_idx)
                                except ValueError:
                                    continue
                                
                    except (ValueError, IndexError):
                        continue
                
                self.logger.info(f"Found {len(s3_map)} content items in S3")
                
                # Step 2: Update database in batches
                all_content = session.query(Content).all()
                batch_size = 1000
                
                for i in range(0, len(all_content), batch_size):
                    batch = all_content[i:i + batch_size]
                    batch_updates = {'content': 0, 'chunks': 0, 'transcripts': 0}
                    
                    for content in batch:
                        try:
                            s3_state = s3_map[content.content_id]
                            updates = {}
                            
                            # Check source file
                            has_source = bool(s3_state['source'])
                            if has_source != content.is_downloaded:
                                updates['is_downloaded'] = has_source
                                updates['s3_source_key'] = s3_state['source']
                                updates['blocked_download'] = False if has_source else content.blocked_download
                                batch_updates['content'] += 1
                            
                            # Check chunks - perform a more thorough check
                            s3_chunks = s3_state['chunks']
                            if s3_chunks:
                                # Get all database chunks for this content
                                db_chunks = session.query(ContentChunk).filter_by(content_id=content.id).all()
                                db_chunk_map = {c.chunk_index: c for c in db_chunks}
                                
                                # Verify what chunks exist in S3 with a direct check
                                verified_s3_chunks = set()
                                chunks_prefix = f"content/{content.content_id}/chunks/"
                                for key in s3_storage.list_files(chunks_prefix):
                                    try:
                                        parts = Path(key).parts
                                        if len(parts) >= 5 and parts[-1] == 'audio.wav':
                                            try:
                                                chunk_idx = int(parts[-2])  # Get index from directory name
                                                verified_s3_chunks.add(chunk_idx)
                                            except ValueError:
                                                continue
                                    except (ValueError, IndexError):
                                        continue
                                
                                # Update or create chunks based on verified S3 chunks
                                completed_chunks = 0
                                for chunk_idx in verified_s3_chunks:
                                    if chunk_idx in db_chunk_map:
                                        chunk = db_chunk_map[chunk_idx]
                                        if chunk.extraction_status != 'completed':
                                            chunk.mark_extraction_completed()
                                            batch_updates['chunks'] += 1
                                    else:
                                        # Create new chunk if it doesn't exist in database
                                        new_chunk = ContentChunk(
                                            content_id=content.id,
                                            chunk_index=chunk_idx,
                                            status='completed',
                                            extraction_status='completed',
                                            start_time=0.0,  # Default values, will be updated later
                                            end_time=0.0,
                                            duration=0.0
                                        )
                                        session.add(new_chunk)
                                        batch_updates['chunks'] += 1
                                    completed_chunks += 1
                                
                                # Get total chunks from database after updates
                                total_chunks = len(db_chunks)
                                if verified_s3_chunks and completed_chunks == total_chunks:
                                    # All chunks are complete, mark content as converted
                                    updates['is_converted'] = True
                                    updates['chunks_processed'] = completed_chunks
                                    updates['total_chunks'] = total_chunks
                                    batch_updates['content'] += 1
                                elif verified_s3_chunks:
                                    # Some chunks are complete, update counts
                                    updates['chunks_processed'] = completed_chunks
                                    updates['total_chunks'] = total_chunks
                                    batch_updates['content'] += 1
                            
                            # Check transcript status based on chunk transcripts
                            if s3_state['chunks'] and s3_state['chunk_transcripts']:
                                # All chunks must have transcripts to be considered fully transcribed
                                all_chunks_transcribed = s3_state['chunks'].issubset(s3_state['chunk_transcripts'])
                                
                                if all_chunks_transcribed and not content.is_transcribed:
                                    updates['is_transcribed'] = True
                                    batch_updates['transcripts'] += 1
                                    self.logger.debug(f"Content {content.content_id} all {len(s3_state['chunks'])} chunks transcribed, marking as transcribed")
                            
                            # Check diarization files
                            has_diarization = bool(s3_state['diarization'] and s3_state['speaker_embeddings'])
                            if has_diarization and not content.is_diarized:
                                updates['is_diarized'] = True
                                batch_updates['content'] += 1
                            
                            # Check stitched transcript
                            has_stitched = bool(s3_state['transcript_diarized'])
                            if has_stitched and not content.is_stitched:
                                updates['is_stitched'] = True
                                # Update the transcript key to point to the stitched file
                                updates['s3_transcript_key'] = s3_state['transcript_diarized']
                                batch_updates['transcripts'] += 1
                            
                            # Apply updates
                            if updates:
                                updates['last_updated'] = datetime.now(timezone.utc)
                                for key, value in updates.items():
                                    setattr(content, key, value)
                            
                        except Exception as e:
                            error_msg = f"Error processing content {content.content_id}: {str(e)}"
                            self.logger.error(error_msg)
                            results['errors'].append(error_msg)
                            continue
                    
                    # Commit batch
                    session.commit()
                    
                    # Update total counts
                    for key in batch_updates:
                        results['updates'][key] += batch_updates[key]
                    
                    self.logger.info(
                        f"Batch progress: "
                        f"Content: +{batch_updates['content']}, "
                        f"Chunks: +{batch_updates['chunks']}, "
                        f"Transcripts: +{batch_updates['transcripts']}"
                    )
                
                # Get final database state
                results['after_state'] = self._get_db_state(session)
            
            # Set end time and log summary
            results['end_time'] = datetime.now(timezone.utc).isoformat()
            duration = (datetime.fromisoformat(results['end_time']) - 
                      datetime.fromisoformat(results['start_time'])).total_seconds() / 60
            
            # Calculate changes
            changes = {
                k: results['after_state'][k] - results['before_state'][k]
                for k in results['before_state']
                if not isinstance(results['before_state'][k], dict)
            }
            
            # Calculate chunk changes
            chunk_changes = {
                'total': results['after_state']['chunks']['total'] - results['before_state']['chunks']['total'],
                'extraction': {
                    k: results['after_state']['chunks']['extraction'][k] - results['before_state']['chunks']['extraction'][k]
                    for k in ['completed', 'failed', 'processing']
                },
                'transcription': {
                    k: results['after_state']['chunks']['transcription'][k] - results['before_state']['chunks']['transcription'][k]
                    for k in ['completed', 'failed', 'processing']
                }
            }
            
            self.logger.info(
                f"\n=== State Changes ===\n"
                f"Duration: {duration:.1f} minutes\n"
                f"\nContent State (Before -> After):\n"
                f"Downloaded: {results['before_state']['downloaded']} -> {results['after_state']['downloaded']} ({changes['downloaded']:+d})\n"
                f"Blocked: {results['before_state']['blocked']} -> {results['after_state']['blocked']} ({changes['blocked']:+d})\n"
                f"Converted: {results['before_state']['converted']} -> {results['after_state']['converted']} ({changes['converted']:+d})\n"
                f"With Chunks: {results['before_state']['with_chunks']} -> {results['after_state']['with_chunks']} ({changes['with_chunks']:+d})\n"
                f"Fully Extracted: {results['before_state']['fully_extracted']} -> {results['after_state']['fully_extracted']} ({changes['fully_extracted']:+d})\n"
                f"Fully Transcribed: {results['before_state']['fully_transcribed']} -> {results['after_state']['fully_transcribed']} ({changes['fully_transcribed']:+d})\n"
                f"Fully Diarized: {results['before_state']['fully_diarized']} -> {results['after_state']['fully_diarized']} ({changes['fully_diarized']:+d})\n" # Added
                f"\nChunk Statistics:\n"
                f"Total Chunks: {results['before_state']['chunks']['total']} -> {results['after_state']['chunks']['total']} ({chunk_changes['total']:+d})\n"
                f"Average per Content: {results['after_state']['chunks']['avg_per_content']:.1f}\n"
                f"Min/Max per Content: {results['after_state']['chunks']['min_per_content']}/{results['after_state']['chunks']['max_per_content']}\n"
                f"\nExtraction Status (Before -> After):\n"
                f"Completed: {results['before_state']['chunks']['extraction']['completed']} -> {results['after_state']['chunks']['extraction']['completed']} "
                f"({chunk_changes['extraction']['completed']:+d})\n"
                f"Failed: {results['before_state']['chunks']['extraction']['failed']} -> {results['after_state']['chunks']['extraction']['failed']} "
                f"({chunk_changes['extraction']['failed']:+d})\n"
                f"Processing: {results['before_state']['chunks']['extraction']['processing']} -> {results['after_state']['chunks']['extraction']['processing']} "
                f"({chunk_changes['extraction']['processing']:+d})\n"
                f"Completion Rate: {results['after_state']['chunks']['extraction']['completion_rate']}%\n"
                f"\nTranscription Status (Before -> After):\n"
                f"Completed: {results['before_state']['chunks']['transcription']['completed']} -> {results['after_state']['chunks']['transcription']['completed']} "
                f"({chunk_changes['transcription']['completed']:+d})\n"
                f"Failed: {results['before_state']['chunks']['transcription']['failed']} -> {results['after_state']['chunks']['transcription']['failed']} "
                f"({chunk_changes['transcription']['failed']:+d})\n"
                f"Processing: {results['before_state']['chunks']['transcription']['processing']} -> {results['after_state']['chunks']['transcription']['processing']} "
                f"({chunk_changes['transcription']['processing']:+d})\n"
                f"Completion Rate: {results['after_state']['chunks']['transcription']['completion_rate']}%\n"
                f"\nUpdates Made:\n"
                f"Content records: {results['updates']['content']}\n"
                f"Chunk records: {results['updates']['chunks']}\n"
                f"Transcript records: {results['updates']['transcripts']}\n"
                f"Errors: {len(results['errors'])}"
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in S3 sync: {str(e)}")
            results['errors'].append(str(e))
            results['end_time'] = datetime.now(timezone.utc).isoformat()
            return results
            
    def _get_db_state(self, session) -> Dict:
        """Get current database state statistics including detailed chunk information."""
        stats = session.execute(text("""
            WITH chunk_stats AS (
                SELECT 
                    content_id,
                    COUNT(*) as total_chunks,
                    COUNT(*) FILTER (WHERE extraction_status = 'completed') as completed_extraction,
                    COUNT(*) FILTER (WHERE extraction_status = 'failed') as failed_extraction,
                    COUNT(*) FILTER (WHERE extraction_status = 'processing') as processing_extraction,
                    COUNT(*) FILTER (WHERE transcription_status = 'completed') as completed_transcription,
                    COUNT(*) FILTER (WHERE transcription_status = 'failed') as failed_transcription,
                    COUNT(*) FILTER (WHERE transcription_status = 'processing') as processing_transcription
                FROM content_chunks
                GROUP BY content_id
            ),
            content_stats AS (
                SELECT 
                    COUNT(*) FILTER (WHERE is_downloaded = true) as downloaded,
                    COUNT(*) FILTER (WHERE blocked_download = true) as blocked,
                    COUNT(*) FILTER (WHERE is_converted = true) as converted,
                    COUNT(*) FILTER (WHERE is_transcribed = true) as transcribed,
                    COUNT(*) FILTER (WHERE is_diarized = true) as diarized, -- Added
                    COUNT(DISTINCT CASE WHEN cs.total_chunks > 0 THEN c.id END) as with_chunks,
                    -- Content with all chunks completed
                    COUNT(DISTINCT CASE 
                        WHEN cs.total_chunks > 0 AND cs.total_chunks = cs.completed_extraction 
                        THEN c.id 
                    END) as fully_extracted,
                    -- Content with all chunks transcribed
                    COUNT(DISTINCT CASE 
                        WHEN cs.total_chunks > 0 AND cs.total_chunks = cs.completed_transcription 
                        THEN c.id 
                    END) as fully_transcribed
                    -- Added: Content with final diarized transcript (implicitly where is_diarized = true)
                FROM content c
                LEFT JOIN chunk_stats cs ON cs.content_id = c.id
            ),
            chunk_totals AS (
                SELECT
                    COUNT(*) as total_chunks,
                    COUNT(*) FILTER (WHERE extraction_status = 'completed') as completed_extraction,
                    COUNT(*) FILTER (WHERE extraction_status = 'failed') as failed_extraction,
                    COUNT(*) FILTER (WHERE extraction_status = 'processing') as processing_extraction,
                    COUNT(*) FILTER (WHERE transcription_status = 'completed') as completed_transcription,
                    COUNT(*) FILTER (WHERE transcription_status = 'failed') as failed_transcription,
                    COUNT(*) FILTER (WHERE transcription_status = 'processing') as processing_transcription,
                    -- Average chunks per content
                    ROUND(AVG(cc.chunk_count) FILTER (WHERE cc.chunk_count > 0), 2) as avg_chunks_per_content,
                    MIN(cc.chunk_count) FILTER (WHERE cc.chunk_count > 0) as min_chunks,
                    MAX(cc.chunk_count) as max_chunks
                FROM content_chunks,
                    (SELECT content_id, COUNT(*) as chunk_count 
                     FROM content_chunks 
                     GROUP BY content_id) as cc
            )
            SELECT 
                cs.*,
                ct.*
            FROM content_stats cs, chunk_totals ct
        """)).fetchone()
        
        return {
            # Content state
            'downloaded': stats.downloaded,
            'blocked': stats.blocked,
            'converted': stats.converted,
            'transcribed': stats.transcribed,
            'with_chunks': stats.with_chunks,
            'fully_extracted': stats.fully_extracted,
            'fully_transcribed': stats.fully_transcribed,
            'fully_diarized': stats.diarized, # Added (uses content.is_diarized count)
            
            # Chunk totals
            'chunks': {
                'total': stats.total_chunks,
                'avg_per_content': stats.avg_chunks_per_content,
                'min_per_content': stats.min_chunks,
                'max_per_content': stats.max_chunks,
                
                # Extraction status
                'extraction': {
                    'completed': stats.completed_extraction,
                    'failed': stats.failed_extraction,
                    'processing': stats.processing_extraction,
                    'completion_rate': round(stats.completed_extraction / stats.total_chunks * 100, 2) if stats.total_chunks else 0
                },
                
                # Transcription status
                'transcription': {
                    'completed': stats.completed_transcription,
                    'failed': stats.failed_transcription,
                    'processing': stats.processing_transcription,
                    'completion_rate': round(stats.completed_transcription / stats.total_chunks * 100, 2) if stats.total_chunks else 0
                }
            }
        }

    async def fix_conversion_status(self) -> Dict:
        """Fix the is_converted flag to match actual chunk completion status."""
        results = {
            'total_checked': 0,
            'fixed': 0,
            'errors': [],
            'start_time': datetime.now(timezone.utc).isoformat()
        }
        
        try:
            with self._get_session_with_timeout(timeout_ms=3600000) as session:  # 1 hour timeout
                # First get current state
                initial_stats = session.execute(text("""
                    WITH chunk_stats AS (
                        SELECT 
                            content_id,
                            COUNT(*) as total_chunks,
                            COUNT(*) FILTER (WHERE extraction_status = 'completed') as completed_extraction
                        FROM content_chunks
                        GROUP BY content_id
                    )
                    SELECT 
                        COUNT(*) FILTER (WHERE is_converted = true) as currently_converted,
                        COUNT(DISTINCT CASE 
                            WHEN cs.total_chunks > 0 AND cs.total_chunks = cs.completed_extraction 
                            THEN c.id 
                        END) as should_be_converted
                    FROM content c
                    LEFT JOIN chunk_stats cs ON cs.content_id = c.id
                """)).fetchone()
                
                self.logger.info("\nCurrent State:")
                self.logger.info(f"Content marked as converted: {initial_stats.currently_converted}")
                self.logger.info(f"Content that should be converted: {initial_stats.should_be_converted}")
                
                # Update is_converted flag based on chunk completion
                update_query = text("""
                    WITH chunk_stats AS (
                        SELECT 
                            content_id,
                            COUNT(*) as total_chunks,
                            COUNT(*) FILTER (WHERE extraction_status = 'completed') as completed_extraction
                        FROM content_chunks
                        GROUP BY content_id
                    )
                    UPDATE content c
                    SET 
                        is_converted = CASE 
                            WHEN cs.total_chunks > 0 AND cs.total_chunks = cs.completed_extraction THEN true
                            ELSE false
                        END,
                        last_updated = NOW()
                    FROM chunk_stats cs
                    WHERE cs.content_id = c.id
                    AND (
                        (cs.total_chunks > 0 AND cs.total_chunks = cs.completed_extraction AND NOT c.is_converted)
                        OR
                        (cs.total_chunks = 0 OR cs.total_chunks != cs.completed_extraction) AND c.is_converted
                    )
                """)
                
                result = session.execute(update_query)
                session.commit()
                
                # Get final state
                final_stats = session.execute(text("""
                    WITH chunk_stats AS (
                        SELECT 
                            content_id,
                            COUNT(*) as total_chunks,
                            COUNT(*) FILTER (WHERE extraction_status = 'completed') as completed_extraction
                        FROM content_chunks
                        GROUP BY content_id
                    )
                    SELECT 
                        COUNT(*) FILTER (WHERE is_converted = true) as currently_converted,
                        COUNT(DISTINCT CASE 
                            WHEN cs.total_chunks > 0 AND cs.total_chunks = cs.completed_extraction 
                            THEN c.id 
                        END) as should_be_converted
                    FROM content c
                    LEFT JOIN chunk_stats cs ON cs.content_id = c.id
                """)).fetchone()
                
                results['end_time'] = datetime.now(timezone.utc).isoformat()
                duration = (datetime.fromisoformat(results['end_time']) - 
                          datetime.fromisoformat(results['start_time'])).total_seconds() / 60
                
                self.logger.info("\nFinal State:")
                self.logger.info(f"Content marked as converted: {final_stats.currently_converted}")
                self.logger.info(f"Content that should be converted: {final_stats.should_be_converted}")
                self.logger.info(f"\nChanges made:")
                self.logger.info(f"Duration: {duration:.1f} minutes")
                self.logger.info(f"Fixed conversion status for {result.rowcount} content items")
                
                return results
                
        except Exception as e:
            self.logger.error(f"Error fixing conversion status: {str(e)}")
            results['errors'].append(str(e))
            results['end_time'] = datetime.now(timezone.utc).isoformat()
            return results

    async def fix_state_inconsistencies(self, analysis_results: Dict) -> Dict:
        """Fix inconsistencies identified during state check.
        
        Args:
            analysis_results: Results dictionary from complete_state_check
            
        Returns:
            Dict containing results of the fix operation
        """
        results = {
            'start_time': datetime.now(timezone.utc).isoformat(),
            'end_time': None,
            'fixed': {
                'download_state': 0,
                'conversion_state': 0,
                'chunk_state': 0,
                'chunk_transcript_state': 0, # Renamed
                'diarization_state': 0, # Will fix based on diarization + embeddings files
                'stitch_state': 0 # NEW: Will fix based on transcript_diarized.json
            },
            'errors': []
        }
        
        try:
            with self._get_session_with_timeout(timeout_ms=3600000) as session:
                # Process in batches for better performance
                batch_size = 1000
                
                # 0. Fix download state mismatches first
                if analysis_results['inconsistencies']['download_state']:
                    self.logger.info("\nFixing download state mismatches...")
                    download_updates = 0
                    
                    for i in range(0, len(analysis_results['inconsistencies']['download_state']), batch_size):
                        batch = analysis_results['inconsistencies']['download_state'][i:i + batch_size]
                        batch_updates = 0
                        
                        for mismatch in batch:
                            try:
                                content = session.query(Content).filter_by(content_id=mismatch['content_id']).first()
                                if not content:
                                    continue
                                    
                                # Update download state based on S3
                                if content.is_downloaded != mismatch['s3_state']:
                                    content.is_downloaded = mismatch['s3_state']
                                    if mismatch['s3_state']:
                                        # If file exists in S3, update source key and unblock
                                        content.s3_source_key = mismatch['source_path']
                                        content.blocked_download = False
                                    content.last_updated = datetime.now(timezone.utc)
                                    batch_updates += 1
                                    
                            except Exception as e:
                                error_msg = f"Error fixing download state for {mismatch['content_id']}: {str(e)}"
                                self.logger.error(error_msg)
                                results['errors'].append(error_msg)
                                continue
                        
                        download_updates += batch_updates
                        self.logger.info(f"Fixed {batch_updates} download states in current batch")
                    
                    results['fixed']['download_state'] = download_updates
                    self.logger.info(f"Fixed {download_updates} download states total")
                
                # 1. Fix chunk state mismatches
                if analysis_results['inconsistencies']['chunk_state']:
                    self.logger.info("\nFixing chunk state mismatches...")
                    chunk_updates = 0
                    
                    for mismatch in analysis_results['inconsistencies']['chunk_state']:
                        try:
                            content = session.query(Content).filter_by(content_id=mismatch['content_id']).first()
                            if not content:
                                continue
                                
                            # Update chunk states based on S3
                            chunks = session.query(ContentChunk).filter_by(content_id=content.id).all()
                            for chunk in chunks:
                                if chunk.chunk_index in mismatch['chunk_indices']:
                                    if chunk.extraction_status != 'completed':
                                        chunk.mark_extraction_completed()
                                        chunk_updates += 1
                            
                            # Update content chunk counts
                            content.total_chunks = len(mismatch['chunk_indices'])
                            content.chunks_processed = len(mismatch['chunk_indices'])
                            content.last_updated = datetime.now(timezone.utc)
                            
                        except Exception as e:
                            error_msg = f"Error fixing chunk state for {mismatch['content_id']}: {str(e)}"
                            self.logger.error(error_msg)
                            results['errors'].append(error_msg)
                            continue
                    
                    results['fixed']['chunk_state'] = chunk_updates
                    self.logger.info(f"Fixed {chunk_updates} chunk states")
                    session.commit()
                
                # 2. Fix conversion state mismatches
                if analysis_results['inconsistencies']['conversion_state']:
                    self.logger.info("\nFixing conversion state mismatches...")
                    conversion_updates = 0
                    
                    for mismatch in analysis_results['inconsistencies']['conversion_state']:
                        try:
                            content = session.query(Content).filter_by(content_id=mismatch['content_id']).first()
                            if not content:
                                continue
                                
                            if content.is_converted != mismatch['should_be']:
                                content.is_converted = mismatch['should_be']
                                content.last_updated = datetime.now(timezone.utc)
                                conversion_updates += 1
                                
                        except Exception as e:
                            error_msg = f"Error fixing conversion state for {mismatch['content_id']}: {str(e)}"
                            self.logger.error(error_msg)
                            results['errors'].append(error_msg)
                            continue
                    
                    results['fixed']['conversion_state'] = conversion_updates
                    self.logger.info(f"Fixed {conversion_updates} conversion states")
                    session.commit()
                
                # 3. Fix chunk transcript state mismatches (based on is_transcribed flag)
                if analysis_results['inconsistencies']['chunk_transcript_state']:
                    self.logger.info("\nFixing chunk transcript state mismatches...")
                    transcript_updates = 0
                    
                    for i in range(0, len(analysis_results['inconsistencies']['chunk_transcript_state']), batch_size):
                        batch = analysis_results['inconsistencies']['chunk_transcript_state'][i:i + batch_size]
                        batch_updates = 0
                        
                        for mismatch in batch:
                            try:
                                content = session.query(Content).filter_by(content_id=mismatch['content_id']).first()
                                if not content:
                                    continue
                                
                                # Update is_transcribed based on whether all chunks *should* be transcribed
                                if content.is_transcribed != mismatch['s3_state_based_on_chunks']:
                                    content.is_transcribed = mismatch['s3_state_based_on_chunks']
                                    # Don't update s3_transcript_key here, it's for the final output now
                                    content.last_updated = datetime.now(timezone.utc)
                                    batch_updates += 1
                                    
                            except Exception as e:
                                error_msg = f"Error fixing chunk transcript state for {mismatch['content_id']}: {str(e)}"
                                self.logger.error(error_msg)
                                results['errors'].append(error_msg)
                                continue
                        
                        transcript_updates += batch_updates
                        self.logger.info(f"Fixed {batch_updates} chunk transcript states in current batch")
                        session.commit()
                    
                    results['fixed']['chunk_transcript_state'] = transcript_updates
                    self.logger.info(f"Fixed {transcript_updates} chunk transcript states total")

                # 4. Fix diarization state mismatches (based on diarization.json + speaker_embeddings.json)
                if analysis_results['inconsistencies']['diarization_state']:
                    self.logger.info("\nFixing diarization state mismatches (based on intermediate diarization/embedding files)...")
                    diarization_updates = 0

                    for i in range(0, len(analysis_results['inconsistencies']['diarization_state']), batch_size):
                        batch = analysis_results['inconsistencies']['diarization_state'][i:i + batch_size]
                        batch_updates = 0

                        for mismatch in batch:
                            try:
                                content = session.query(Content).filter_by(content_id=mismatch['content_id']).first()
                                if not content:
                                    continue

                                # Determine S3 state based on BOTH files existing
                                s3_diarization_step_complete = mismatch['s3_diarization'] and mismatch['s3_embeddings']

                                # Fix if DB state (is_diarized) doesn't match S3 reality
                                if content.is_diarized != s3_diarization_step_complete:
                                    content.is_diarized = s3_diarization_step_complete
                                    content.last_updated = datetime.now(timezone.utc)
                                    batch_updates += 1
                                    self.logger.debug(f"Setting is_diarized={s3_diarization_step_complete} for {content.content_id} based on diarization/embedding files.")

                            except Exception as e:
                                error_msg = f"Error fixing diarization state for {mismatch['content_id']}: {str(e)}"
                                self.logger.error(error_msg)
                                results['errors'].append(error_msg)
                                continue

                        diarization_updates += batch_updates
                        self.logger.info(f"Fixed {batch_updates} diarization states in current batch")
                        session.commit() # Commit after each batch for this step too

                    results['fixed']['diarization_state'] = diarization_updates
                    self.logger.info(f"Fixed {diarization_updates} diarization states total")

                # 5. Fix stitch state mismatches (based on transcript_diarized.json)
                if analysis_results['inconsistencies']['stitch_state']:
                    self.logger.info("\nFixing stitch state mismatches (based on final transcript_diarized.json)...")
                    stitch_updates = 0
                    
                    for i in range(0, len(analysis_results['inconsistencies']['stitch_state']), batch_size):
                        batch = analysis_results['inconsistencies']['stitch_state'][i:i + batch_size]
                        batch_updates = 0
                        
                        for mismatch in batch:
                             try:
                                content = session.query(Content).filter_by(content_id=mismatch['content_id']).first()
                                if not content:
                                    continue
                                    
                                s3_stitch_complete = mismatch['s3_stitched']
                                
                                if content.is_stitched != s3_stitch_complete:
                                    content.is_stitched = s3_stitch_complete
                                    if s3_stitch_complete:
                                        content.s3_transcript_key = mismatch['s3_path'] # Ensure key points to final file
                                    # else: # Optional: Clear key if file missing?
                                    #    content.s3_transcript_key = None 
                                    content.last_updated = datetime.now(timezone.utc)
                                    batch_updates += 1
                                    self.logger.debug(f"Setting is_stitched={s3_stitch_complete} for {content.content_id} based on transcript_diarized.json")
                             except Exception as e:
                                error_msg = f"Error fixing stitch state for {mismatch['content_id']}: {str(e)}"
                                self.logger.error(error_msg)
                                results['errors'].append(error_msg)
                                continue
                        
                        stitch_updates += batch_updates
                        self.logger.info(f"Fixed {batch_updates} stitch states in current batch")
                        session.commit()
                    
                    results['fixed']['stitch_state'] = stitch_updates
                    self.logger.info(f"Fixed {stitch_updates} stitch states total")

                # Log final summary
                results['end_time'] = datetime.now(timezone.utc).isoformat()
                duration = (datetime.fromisoformat(results['end_time']) - 
                          datetime.fromisoformat(results['start_time'])).total_seconds() / 60
                
                self.logger.info(f"\n=== Fix Summary ===")
                self.logger.info(f"Duration: {duration:.1f} minutes")
                self.logger.info(f"Fixed download states: {results['fixed']['download_state']}")
                self.logger.info(f"Fixed conversion states: {results['fixed']['conversion_state']}")
                self.logger.info(f"Fixed chunk states: {results['fixed']['chunk_state']}")
                self.logger.info(f"Fixed chunk transcript states: {results['fixed']['chunk_transcript_state']}") # Updated label
                self.logger.info(f"Fixed diarization states (Intermediate): {results['fixed']['diarization_state']}") # Updated label
                self.logger.info(f"Fixed stitch states (Final Output): {results['fixed']['stitch_state']}") # Added
                self.logger.info(f"Total errors: {len(results['errors'])}")
                
                return results
                
        except Exception as e:
            self.logger.error(f"Error fixing state inconsistencies: {str(e)}")
            results['errors'].append(str(e))
            results['end_time'] = datetime.now(timezone.utc).isoformat()
            return results

    def _detect_diarization_method_from_file(self, s3_storage, diarization_s3_path: str) -> Optional[str]:
        """Detect diarization method by analyzing the diarization.json file structure.
        
        Args:
            s3_storage: S3Storage instance
            diarization_s3_path: S3 path to the diarization.json file
        
        Returns:
            Detected diarization method string or None if detection fails
        """
        try:
            import json
            import gzip
            from io import BytesIO
            
            # Download the file content
            content_bytes = s3_storage.download_file_bytes(diarization_s3_path)
            if not content_bytes:
                self.logger.warning(f"Could not download diarization file from {diarization_s3_path}")
                return None
            
            # Try to parse as JSON (could be compressed or uncompressed)
            try:
                # First try direct JSON parsing
                diarization_data = json.loads(content_bytes.decode('utf-8'))
            except (UnicodeDecodeError, json.JSONDecodeError):
                # Try gzip decompression first
                try:
                    with gzip.open(BytesIO(content_bytes), 'rt') as gz_file:
                        diarization_data = json.load(gz_file)
                except:
                    self.logger.warning(f"Could not parse diarization file from {diarization_s3_path}")
                    return None
            
            # Analyze structure to determine method
            if isinstance(diarization_data, dict):
                # Check if it has an explicit method field first (most reliable)
                if 'method' in diarization_data:
                    return diarization_data['method']
                elif 'diarization_method' in diarization_data:
                    return diarization_data['diarization_method']
                
                # Check for FluidAudio characteristics
                elif 'speakerEmbeddings' in diarization_data or 'speaker_centroids' in diarization_data:
                    # FluidAudio includes speaker embeddings/centroids
                    return 'fluid_audio'
                
                # Check for PyAnnote characteristics  
                elif 'speakers' in diarization_data or ('segments' in diarization_data and 'method' not in diarization_data):
                    # Standard PyAnnote structure (only if method field is not present)
                    return 'pyannote3.1'
                
            # If we can't determine from structure, return None
            self.logger.warning(f"Could not determine diarization method from file structure in {diarization_s3_path}")
            return None
            
        except Exception as e:
            self.logger.warning(f"Error detecting diarization method from {diarization_s3_path}: {str(e)}")
            return None
    
    async def update_diarization_complete_status(self, content_id: str, success: bool, result_data: Optional[Dict] = None, error: Optional[str] = None, diarization_method: Optional[str] = None) -> bool:
        """Update content diarization status based on diarization.py completion.
        
        Args:
            content_id: The content ID
            success: Whether diarization succeeded
            result_data: Optional dict containing paths if successful (diarization_path, embeddings_path)
            error: Optional error message if failed
            diarization_method: Method used for diarization ('pyannote3.1', 'fluid_audio', etc.)
        
        Returns:
            True if status was updated, False otherwise.
        """
        try:
            with self._get_session_with_timeout() as session:
                content = session.query(Content).filter_by(content_id=content_id).first()
                if not content:
                    self.logger.error(f"Content not found: {content_id}")
                    return False

                if success:
                    # Verify that the expected files exist in S3 before marking as diarized
                    if not result_data or 'diarization_path' not in result_data:
                        self.logger.error(f"Missing diarization_path in result data for {content_id}")
                        return False
                        
                    # Initialize S3 storage
                    from ..storage.s3_utils import S3StorageConfig, S3Storage
                    import yaml
                    from pathlib import Path

                    config_path = get_config_path()
                    with open(config_path) as f:
                        config = yaml.safe_load(f)

                    s3_config = S3StorageConfig(
                        endpoint_url=config['storage']['s3']['endpoint_url'],
                        access_key=config['storage']['s3']['access_key'],
                        secret_key=config['storage']['s3']['secret_key'],
                        bucket_name=config['storage']['s3']['bucket_name'],
                        use_ssl=config['storage']['s3']['use_ssl']
                    )
                    s3_storage = S3Storage(s3_config)
                    
                    diarization_exists = s3_storage.file_exists(result_data['diarization_path'])
                    
                    # Check for embeddings_path if it exists in result_data (PyAnnote provides this, FluidAudio may not)
                    embeddings_exist = True  # Default to True
                    if 'embeddings_path' in result_data:
                        embeddings_exist = s3_storage.file_exists(result_data['embeddings_path'])

                    if diarization_exists and embeddings_exist:
                        content.is_diarized = True
                        
                        # If diarization_method is not provided, try to detect it from the diarization file
                        if not diarization_method:
                            diarization_method = self._detect_diarization_method_from_file(
                                s3_storage, result_data['diarization_path']
                            )
                        
                        if diarization_method:
                            content.diarization_method = diarization_method
                        # Clear any previous error in metadata if needed
                        meta = content.meta_data or {}
                        meta.pop('diarization_error', None)
                        content.meta_data = meta
                        content.last_updated = datetime.now(timezone.utc)
                        method_msg = f" using {diarization_method}" if diarization_method else ""
                        self.logger.info(f"Marked content {content_id} as diarized (is_diarized=True){method_msg}")
                        return True
                    else:
                        missing = []
                        if not diarization_exists: 
                            missing.append("diarization.json")
                        if 'embeddings_path' in result_data and not embeddings_exist: 
                            missing.append("speaker_embeddings.json")
                        
                        self.logger.error(f"Cannot mark content {content_id} as diarized. Missing S3 files: {', '.join(missing)}")
                        content.is_diarized = False # Ensure it's False if files are missing
                        meta = content.meta_data or {}
                        meta['diarization_error'] = f"Post-task check failed: Missing S3 files: {', '.join(missing)}\n"
                        content.meta_data = meta
                        content.last_updated = datetime.now(timezone.utc)
                        return False
                else:
                    # Handle failed diarization
                    content.is_diarized = False
                    meta = content.meta_data or {}
                    meta.update({
                        'diarization_error': error,
                        'last_diarization_attempt': datetime.now(timezone.utc).isoformat()
                    })
                    content.meta_data = meta
                    content.last_updated = datetime.now(timezone.utc)

                    # If error indicates missing audio file (404), mark content as not converted
                    if error and ('404' in error or 'not found' in error.lower()):
                        content.is_converted = False
                        self.logger.warning(f"Marked content {content_id} as NOT converted due to missing audio file: {error}")

                    self.logger.warning(f"Marked content {content_id} as NOT diarized due to error: {error}")
                    return True # Return True because we successfully updated the status to failed

        except Exception as e:
            self.logger.error(f"Error updating diarization status for {content_id}: {str(e)}")
            return False

    async def update_stitched_transcript(self, content_id: str, result_data: Dict) -> bool:
        """Update content with stitched transcript data and mark as stitched.
        
        Args:
            content_id: The content ID
            result_data: Dictionary containing transcript data including:
                - duration: Total duration in seconds
                - num_segments: Number of segments
                - has_word_timestamps: Whether word timestamps are present
                - text: Full transcript text
                - segments: List of transcript segments
                - output_path: S3 path of the transcript_diarized.json file
        """
        try:
            with self._get_session_with_timeout() as session:
                content = session.query(Content).filter_by(content_id=content_id).first()
                if not content:
                    self.logger.error(f"Content not found: {content_id}")
                    return False

                # Check if output path is provided (should be transcript_diarized.json)
                output_path = result_data.get('output_path')
                if not output_path or 'transcript_diarized.json' not in output_path:
                    self.logger.error(f"Missing or incorrect output_path in result_data for {content_id}. Expected path to transcript_diarized.json.")
                    return False
                    
                # Verify the output file exists in S3 before proceeding
                from ..storage.s3_utils import S3StorageConfig, S3Storage
                import yaml
                from pathlib import Path

                config_path = get_config_path()
                with open(config_path) as f:
                    config = yaml.safe_load(f)

                s3_config = S3StorageConfig(
                    endpoint_url=config['storage']['s3']['endpoint_url'],
                    access_key=config['storage']['s3']['access_key'],
                    secret_key=config['storage']['s3']['secret_key'],
                    bucket_name=config['storage']['s3']['bucket_name'],
                    use_ssl=config['storage']['s3']['use_ssl']
                )
                s3_storage = S3Storage(s3_config)
                
                if not s3_storage.file_exists(output_path):
                    self.logger.error(f"Cannot mark content {content_id} as stitched. Final file {output_path} not found in S3.")
                    # Optionally set an error in metadata
                    meta = content.meta_data or {}
                    meta['stitch_error'] = f"Post-task check failed: Missing S3 file: {output_path}\n"
                    content.meta_data = meta
                    content.is_stitched = False # Ensure flag is false
                    content.last_updated = datetime.now(timezone.utc)
                    return False

                # Create new transcription record (optional, could be replaced by reading the diarized file later)
                # For now, let's assume we want to store it for quick access
                transcription = Transcription(
                    content_id=content.id,
                    full_text=result_data['text'],
                    segments=result_data['segments'],
                    model_version='whisper_diarized', # Indicate source
                    processing_status='processed'
                )
                session.add(transcription)

                # Update content metadata
                meta = content.meta_data or {}
                meta.update({
                    'transcript_duration': result_data['duration'],
                    'transcript_segments': result_data['num_segments'],
                    'has_word_timestamps': result_data['has_word_timestamps'],
                    'last_stitched': datetime.now(timezone.utc).isoformat()
                })
                # Clear previous stitch error if any
                meta.pop('stitch_error', None)
                content.meta_data = meta
                content.last_updated = datetime.now(timezone.utc)

                # Update S3 paths and mark as stitched
                content.s3_transcript_key = output_path # Store the path to the final stitched file
                content.is_stitched = True # Mark stitching as complete
                # content.is_diarized should have been set by the previous step

                self.logger.info(f"Updated content {content_id} with stitched transcript info. Marked is_stitched=True.")
                return True

        except Exception as e:
            self.logger.error(f"Error updating stitched transcript for {content_id}: {str(e)}")
            return False

    async def update_content_status(self, content_id: str, is_converted: bool = None, 
                                  total_chunks: int = None, chunks_processed: int = None,
                                  is_transcribed: bool = None,
                                  is_diarized: bool = None,
                                  is_stitched: bool = None
                                  ) -> bool:
        """Update content status fields."""
        try:
            with self._get_session_with_timeout() as session:
                content = session.query(Content).filter_by(content_id=content_id).first()
                if not content:
                    return False
                    
                # Update basic fields
                if is_converted is not None:
                    content.is_converted = is_converted
                if total_chunks is not None:
                    content.total_chunks = total_chunks
                if chunks_processed is not None:
                    content.chunks_processed = chunks_processed
                if is_diarized is not None:
                    content.is_diarized = is_diarized
                
                # Update transcribed status - don't touch s3_transcript_key
                # Transcription now happens at the chunk level, no global transcript.json
                if is_transcribed is not None:
                    content.is_transcribed = is_transcribed
                
                # Update stitched status last (highest priority)
                if is_stitched is not None:
                    content.is_stitched = is_stitched
                    if is_stitched:
                        # When marking as stitched, always point to the stitched file
                        stitched_path = f"content/{content_id}/transcript_diarized.json"
                        content.s3_transcript_key = stitched_path
                
                content.last_updated = datetime.now(timezone.utc)
                return True
                
        except Exception as e:
            self.logger.error(f"Error updating content status for {content_id}: {str(e)}")
            return False