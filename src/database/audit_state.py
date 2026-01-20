#!/usr/bin/env python3
import sys
from pathlib import Path
from src.utils.paths import get_project_root, get_config_path
import argparse
import logging
import yaml
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv(get_project_root() / '.env')
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime, timezone
from sqlalchemy import text
from sqlalchemy.orm import Session
import json

# Add the project root to Python path
sys.path.append(str(get_project_root()))

from src.utils.logger import setup_worker_logger
from src.database.session import get_session
from src.database.models import Content, ContentChunk
from src.storage.s3_utils import S3Storage, S3StorageConfig
from src.storage.config import get_storage_config

@dataclass
class AuditSummary:
    """Tracks audit results across projects"""
    projects: Dict[str, Dict[str, Dict[str, int]]] = field(
        default_factory=lambda: defaultdict(
            lambda: {
                'missing_files': defaultdict(int),
                'inconsistent_states': defaultdict(int),
                'total_checked': 0
            }
        )
    )
    logger: logging.Logger = field(default_factory=lambda: setup_worker_logger('audit_summary'))
    
    def add_missing_file(self, project: str, file_type: str, count: int = 1):
        """Record missing files for a project/file_type combination"""
        self.projects[project]['missing_files'][file_type] += count
        
    def add_inconsistent_state(self, project: str, state_type: str, count: int = 1):
        """Record inconsistent states for a project/state_type combination"""
        self.projects[project]['inconsistent_states'][state_type] += count
        
    def increment_total_checked(self, project: str, count: int = 1):
        """Increment total checked count for a project"""
        self.projects[project]['total_checked'] += count
        
    def print_summary(self):
        """Print a formatted summary of all audit results"""
        if not self.projects:
            self.logger.info("\n📊 No audit results to report")
            return
            
        self.logger.info("\n📊 S3 File Audit Summary")
        self.logger.info("=" * 80)
        
        # Calculate column widths with default values for empty collections
        project_width = max(len("Project"), max((len(p) for p in self.projects.keys()), default=0))
        type_width = max(
            len("File Type"),
            max((len(t) for p in self.projects.values() 
                for t in p['missing_files'].keys()), default=0)
        )
        
        # Calculate count width by checking all numeric values in the nested structure
        count_values = []
        for project in self.projects.values():
            # Add missing files counts
            count_values.extend(project['missing_files'].values())
            # Add inconsistent states counts
            count_values.extend(project['inconsistent_states'].values())
            # Add total checked
            count_values.append(project['total_checked'])
        
        count_width = max(
            len("Count"),
            max((len(str(c)) for c in count_values), default=0)
        )
        
        # Print header
        header = f"{'Project':<{project_width}} | {'File Type':<{type_width}} | {'Missing':>{count_width}} | {'Total Checked':>{count_width}}"
        self.logger.info(header)
        self.logger.info("-" * len(header))
        
        # Print each project's results
        for project, results in sorted(self.projects.items()):
            total_checked = results['total_checked']
            missing_files = results['missing_files']
            
            if not missing_files:
                self.logger.info(f"{project:<{project_width}} | {'All files present':<{type_width}} | {'0':>{count_width}} | {total_checked:>{count_width}}")
                continue
                
            # Print each missing file type
            for file_type, count in sorted(missing_files.items()):
                self.logger.info(
                    f"{project:<{project_width}} | "
                    f"{file_type:<{type_width}} | "
                    f"{count:>{count_width}} | "
                    f"{total_checked:>{count_width}}"
                )
            
            # Print project subtotal
            total_missing = sum(missing_files.values())
            self.logger.info("-" * len(header))
            self.logger.info(
                f"{project:<{project_width}} | "
                f"{'TOTAL MISSING':<{type_width}} | "
                f"{total_missing:>{count_width}} | "
                f"{total_checked:>{count_width}}"
            )
            self.logger.info("=" * len(header))
            
            # Print inconsistent states if any
            if results['inconsistent_states']:
                self.logger.info("\nInconsistent States:")
                for state_type, count in sorted(results['inconsistent_states'].items()):
                    self.logger.info(f"  {state_type}: {count}")
                self.logger.info("=" * len(header))

class StateAuditor:
    """Audits the state of content processing outputs in S3 against database state."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = setup_worker_logger('state_auditor')
        self.s3_storage = S3Storage(S3StorageConfig.from_dict(config['storage']['s3']))
        self.audit_summary = AuditSummary()
        self.content_map = {}  # Will store the complete S3 content map
        
    async def build_content_map(self) -> None:
        """Build a complete map of all content and their files in S3"""
        try:
            self.logger.info("Building content map from S3...")
            
            # List all files in the content directory
            content_files = self.s3_storage.list_files("content/")
            
            # Process files to build content map
            for file_path in content_files:
                # Skip the content/ prefix
                parts = file_path.split('/')
                if len(parts) < 2:  # Skip files directly in content/
                    continue
                    
                content_id = parts[1]
                if content_id not in self.content_map:
                    self.content_map[content_id] = {
                        'files': set(),
                        'chunks': defaultdict(set)
                    }
                
                # Get the relative path after content/{content_id}/
                relative_path = '/'.join(parts[2:])
                
                # Check if it's a chunk file
                if relative_path.startswith("chunks/"):
                    chunk_parts = relative_path.split('/')
                    if len(chunk_parts) >= 3:  # chunks/{index}/file
                        chunk_index = chunk_parts[1]
                        self.content_map[content_id]['chunks'][chunk_index].add('/'.join(chunk_parts[2:]))
                else:
                    self.content_map[content_id]['files'].add(relative_path)
            
            self.logger.info(f"Content map built with {len(self.content_map)} content items")
            
        except Exception as e:
            self.logger.error(f"Error building content map: {str(e)}")
            raise
            
    def check_source_files(self, content_id: str) -> bool:
        """Check if source files exist using the content map"""
        try:
            if content_id not in self.content_map:
                return False
                
            files = self.content_map[content_id]['files']
            
            # Check for audio.wav
            if 'audio.wav' in files:
                return True
                
            # Check for source files
            source_extensions = ['.mp4', '.mp3', '.wav']
            for ext in source_extensions:
                if f'source{ext}' in files:
                    return True
                    
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking source files for {content_id}: {str(e)}")
            return False
            
    def check_chunk_files(self, content_id: str, index: int) -> bool:
        """Check if chunk files exist using the content map"""
        try:
            if content_id not in self.content_map:
                return False
                
            chunk_files = self.content_map[content_id]['chunks'].get(str(index), set())
            return 'audio.wav' in chunk_files and 'transcription_words.json' in chunk_files
            
        except Exception as e:
            self.logger.error(f"Error checking chunk files for {content_id} chunk {index}: {str(e)}")
            return False
            
    def check_diarization_files(self, content_id: str) -> bool:
        """Check if diarization files exist using the content map"""
        try:
            if content_id not in self.content_map:
                return False
                
            files = self.content_map[content_id]['files']
            return 'diarization.json' in files and 'speaker_embeddings.json' in files
            
        except Exception as e:
            self.logger.error(f"Error checking diarization files for {content_id}: {str(e)}")
            return False
            
    def check_stitch_files(self, content_id: str) -> bool:
        """Check if stitch files exist using the content map"""
        try:
            if content_id not in self.content_map:
                return False
                
            files = self.content_map[content_id]['files']
            return 'stitched_transcript.json' in files
            
        except Exception as e:
            self.logger.error(f"Error checking stitch files for {content_id}: {str(e)}")
            return False
            
    def check_chunks(self, content_id: str) -> bool:
        """Check if audio.wav exists in both content and chunk directories"""
        try:
            if content_id not in self.content_map:
                return False
                
            # Check if main audio.wav exists
            if 'audio.wav' not in self.content_map[content_id]['files']:
                return False
                
            # Check if we have any chunks
            if not self.content_map[content_id]['chunks']:
                return False
                
            # Check each chunk has an audio.wav file
            for chunk_files in self.content_map[content_id]['chunks'].values():
                if 'audio.wav' not in chunk_files:
                    return False
                    
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking audio.wav files for {content_id}: {str(e)}")
            return False
            
    async def check_content_state(self, content_id: str) -> Dict:
        """Check the state of a content item using the content map"""
        try:
            # Get content metadata
            if 'meta.json' not in self.content_map[content_id]['files']:
                return {
                    'content_id': content_id,
                    'state': 'missing',
                    'inconsistent': True,
                    'missing_files': ['meta.json']
                }
                
            content_meta = self.s3_storage.read_json(f"content/{content_id}/meta.json")
            if not content_meta:
                return {
                    'content_id': content_id,
                    'state': 'missing',
                    'inconsistent': True,
                    'missing_files': ['meta.json']
                }
                
            # Check source files
            source_exists = self.check_source_files(content_id)
            if not source_exists:
                return {
                    'content_id': content_id,
                    'state': content_meta.get('state', 'unknown'),
                    'inconsistent': True,
                    'missing_files': ['source files']
                }
                
            # Check state-specific files
            state = content_meta.get('state', 'unknown')
            inconsistent = False
            missing_files = []
            
            if state == 'downloaded':
                if 'audio.wav' not in self.content_map[content_id]['files']:
                    inconsistent = True
                    missing_files.append('audio.wav')
                    
            elif state == 'chunked':
                if not self.check_chunks(content_id):
                    inconsistent = True
                    missing_files.append('chunk files')
                    
            elif state == 'diarized':
                if not self.check_diarization_files(content_id):
                    inconsistent = True
                    missing_files.append('diarization files')
                    
            elif state == 'transcribed':
                if not self.check_chunks(content_id):
                    inconsistent = True
                    missing_files.append('chunk files')
                    
            elif state == 'stitched':
                if not self.check_stitch_files(content_id):
                    inconsistent = True
                    missing_files.append('stitched transcript')
                    
            return {
                'content_id': content_id,
                'state': state,
                'inconsistent': inconsistent,
                'missing_files': missing_files
            }
            
        except Exception as e:
            self.logger.error(f"Error checking state for {content_id}: {str(e)}")
            return {
                'content_id': content_id,
                'state': 'error',
                'inconsistent': True,
                'missing_files': ['error during check']
            }
            
    async def audit_content(self, project: Optional[str] = None, 
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None) -> Dict:
        """Audit all content processing outputs in S3 against database state."""
        try:
            with get_session() as session:
                # Build base query
                query = text("""
                    SELECT 
                        c.id,
                        c.content_id,
                        c.projects,
                        c.platform,
                        c.is_downloaded,
                        c.is_converted,
                        c.is_transcribed,
                        c.is_diarized,
                        c.is_stitched,
                        c.publish_date,
                        COUNT(cc.id) as total_chunks,
                        COUNT(cc.id) FILTER (WHERE cc.extraction_status = 'completed') as extracted_chunks,
                        COUNT(cc.id) FILTER (WHERE cc.transcription_status = 'completed') as transcribed_chunks
                    FROM content c
                    LEFT JOIN content_chunks cc ON c.id = cc.content_id
                    WHERE 
                        (:project IS NULL OR :project = ANY(string_to_array(c.projects, ',')))
                        AND (:start_date IS NULL OR c.publish_date >= :start_date)
                        AND (:end_date IS NULL OR c.publish_date <= :end_date)
                    GROUP BY c.id, c.content_id, c.projects, c.platform, c.is_downloaded,
                             c.is_converted, c.is_transcribed, c.is_diarized, c.is_stitched, c.publish_date
                """)
                
                results = session.execute(query, {
                    'project': project,
                    'start_date': start_date,
                    'end_date': end_date
                })
                
                for row in results:
                    # Get project name (first project if multiple)
                    content_project = row.projects[0] if row.projects else 'unknown'.strip()
                    self.audit_summary.increment_total_checked(content_project)
                    
                    # Check source files
                    if row.is_downloaded:
                        source_exists = self.check_source_files(row.content_id)
                        if not source_exists:
                            self.audit_summary.add_missing_file(content_project, 'source files')
                            self.audit_summary.add_inconsistent_state(content_project, 'is_downloaded_without_source')
                    
                    # Check diarization files
                    if row.is_diarized:
                        if not self.check_diarization_files(row.content_id):
                            self.audit_summary.add_missing_file(content_project, 'diarization files')
                            self.audit_summary.add_inconsistent_state(content_project, 'is_diarized_without_files')
                    
                    # Check stitched transcript
                    if row.is_stitched:
                        if not self.check_stitch_files(row.content_id):
                            self.audit_summary.add_missing_file(content_project, 'stitched_transcript.json')
                            self.audit_summary.add_inconsistent_state(content_project, 'is_stitched_without_file')
                    
                    # Check chunk files
                    if row.is_converted:
                        for chunk_index in range(row.total_chunks):
                            if not self.check_chunk_files(row.content_id, chunk_index):
                                self.audit_summary.add_missing_file(content_project, f'chunk/{chunk_index}/audio.wav')
                                self.audit_summary.add_missing_file(content_project, f'chunk/{chunk_index}/transcription_words.json')
                                self.audit_summary.add_inconsistent_state(content_project, 'is_converted_with_missing_chunk_audio')
                                self.audit_summary.add_inconsistent_state(content_project, 'is_transcribed_with_missing_chunk_transcript')
                
                # Print summary
                self.audit_summary.print_summary()
                
                return {
                    'status': 'success',
                    'summary': self.audit_summary.projects
                }
                
        except Exception as e:
            self.logger.error(f"Error during audit: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
            
    async def fix_inconsistencies(self) -> None:
        """Fix inconsistencies between S3 and database"""
        try:
            self.logger.info("Fixing inconsistencies between S3 and database...")
            
            # Get content with inconsistencies
            with get_session() as session:
                query = text("""
                    SELECT 
                        c.id, 
                        c.content_id,
                        c.is_downloaded, 
                        c.is_converted, 
                        c.is_diarized, 
                        c.is_transcribed, 
                        c.is_stitched,
                        c.total_chunks
                    FROM content c
                    WHERE c.is_downloaded = true
                    AND (
                        (c.is_converted = false)
                        OR
                        (c.is_diarized = false AND c.is_converted = true)
                        OR
                        (c.is_transcribed = false AND c.is_diarized = true)
                        OR
                        (c.is_stitched = false AND c.is_transcribed = true)
                    )
                """)
                
                content_list = session.execute(query).fetchall()
                
                if not content_list:
                    self.logger.info("No inconsistencies found")
                    return
                
                self.logger.info(f"Found {len(content_list)} content items with inconsistencies")
                
                # Process in batches
                batch_size = 100
                updates = []
                
                for i in range(0, len(content_list), batch_size):
                    batch = content_list[i:i + batch_size]
                    batch_updates = []
                    
                    for row in batch:
                        content_id = row.content_id
                        s3_content = self.content_map.get(content_id, {'files': set(), 'chunks': defaultdict(set)})
                        
                        # Check if main audio.wav exists
                        has_main_audio = 'audio.wav' in s3_content['files']
                        
                        # Check if all chunks have audio.wav
                        has_chunk_audio = len(s3_content['chunks']) > 0
                        for chunk_idx, files in s3_content['chunks'].items():
                            if 'audio.wav' not in files:
                                has_chunk_audio = False
                                break
                        
                        # Check if audio conversion is complete but not flagged in DB
                        if has_main_audio and has_chunk_audio and not row.is_converted:
                            self.logger.info(f"Updating is_converted to true for content {content_id}")
                            update_query = text("""
                                UPDATE content
                                SET is_converted = true, last_updated = NOW()
                                WHERE content_id = :content_id
                            """)
                            session.execute(update_query, {'content_id': content_id})
                        
                        # Check if diarization files exist but not flagged
                        if row.is_converted and not row.is_diarized:
                            has_diarization = 'diarization.json' in s3_content['files']
                            if has_diarization:
                                self.logger.info(f"Updating is_diarized to true for content {content_id}")
                                update_query = text("""
                                    UPDATE content
                                    SET is_diarized = true, last_updated = NOW()
                                    WHERE content_id = :content_id
                                """)
                                session.execute(update_query, {'content_id': content_id})
                        
                        # Check if all chunks are transcribed but not flagged
                        if row.is_converted and not row.is_transcribed:
                            all_chunks_transcribed = True
                            for chunk_idx, files in s3_content['chunks'].items():
                                if 'transcript.json' not in files:
                                    all_chunks_transcribed = False
                                    break
                            
                            if all_chunks_transcribed and len(s3_content['chunks']) > 0:
                                self.logger.info(f"Updating is_transcribed to true for content {content_id}")
                                update_query = text("""
                                    UPDATE content
                                    SET is_transcribed = true, last_updated = NOW()
                                    WHERE content_id = :content_id
                                """)
                                session.execute(update_query, {'content_id': content_id})
                        
                        # Check if stitched transcript exists but not flagged
                        if row.is_transcribed and not row.is_stitched:
                            has_stitched = 'stitched_transcript.json' in s3_content['files']
                            if has_stitched:
                                self.logger.info(f"Updating is_stitched to true for content {content_id}")
                                update_query = text("""
                                    UPDATE content
                                    SET is_stitched = true, last_updated = NOW()
                                    WHERE content_id = :content_id
                                """)
                                session.execute(update_query, {'content_id': content_id})
                    
                    # Commit batch updates
                    session.commit()
            
            self.logger.info("Finished fixing inconsistencies")
            
        except Exception as e:
            self.logger.error(f"Error fixing inconsistencies: {str(e)}")
            raise

    async def audit_single_content(self, content_id: str) -> Dict:
        """Audit a single content item with detailed logging"""
        try:
            self.logger.info(f"\n🔍 Auditing content: {content_id}")
            
            # Only map this content's folder
            content_path = f"content/{content_id}/"
            content_files = self.s3_storage.list_files(content_path)
            
            # Build content map for just this content
            self.content_map[content_id] = {
                'files': set(),
                'chunks': defaultdict(set)
            }
            
            # Process files to build content map
            for file_path in content_files:
                # Get the relative path after content/{content_id}/
                parts = file_path.split('/')
                relative_path = '/'.join(parts[2:])
                
                # Check if it's a chunk file
                if relative_path.startswith("chunks/"):
                    chunk_parts = relative_path.split('/')
                    if len(chunk_parts) >= 3:  # chunks/{index}/file
                        chunk_index = chunk_parts[1]
                        self.content_map[content_id]['chunks'][chunk_index].add('/'.join(chunk_parts[2:]))
                else:
                    self.content_map[content_id]['files'].add(relative_path)
            
            # Check source files
            source_exists = self.check_source_files(content_id)
            self.logger.info(f"Source files check: {'✅' if source_exists else '❌'}")
            
            # Check main audio.wav
            main_audio = 'audio.wav' in self.content_map[content_id]['files']
            self.logger.info(f"Main audio.wav: {'✅' if main_audio else '❌'}")
            
            # Check chunks
            chunks_exist = bool(self.content_map[content_id]['chunks'])
            self.logger.info(f"Chunk directories exist: {'✅' if chunks_exist else '❌'}")
            
            if chunks_exist:
                # Check each chunk's audio.wav
                for chunk_id, files in self.content_map[content_id]['chunks'].items():
                    chunk_audio = 'audio.wav' in files
                    self.logger.info(f"  Chunk {chunk_id} audio.wav: {'✅' if chunk_audio else '❌'}")
            
            # Check diarization files
            diarization_exists = self.check_diarization_files(content_id)
            self.logger.info(f"Diarization files: {'✅' if diarization_exists else '❌'}")
            
            # Check stitched transcript
            stitch_exists = self.check_stitch_files(content_id)
            self.logger.info(f"Stitched transcript: {'✅' if stitch_exists else '❌'}")
            
            # Get database state and check duration coverage
            duration_check = None
            with get_session() as session:
                query = text("""
                    SELECT
                        c.id,
                        c.duration as expected_duration,
                        c.is_downloaded,
                        c.is_converted,
                        c.is_transcribed,
                        c.is_diarized,
                        c.is_stitched,
                        COUNT(cc.id) as chunk_count,
                        COALESCE(MAX(cc.end_time), 0) as max_chunk_end
                    FROM content c
                    LEFT JOIN content_chunks cc ON c.id = cc.content_id
                    WHERE c.content_id = :content_id
                    GROUP BY c.id, c.duration, c.is_downloaded, c.is_converted,
                             c.is_transcribed, c.is_diarized, c.is_stitched
                """)
                result = session.execute(query, {'content_id': content_id}).fetchone()

                if result:
                    self.logger.info("\n📊 Database State:")
                    self.logger.info(f"is_downloaded: {'✅' if result.is_downloaded else '❌'}")
                    self.logger.info(f"is_converted: {'✅' if result.is_converted else '❌'}")
                    self.logger.info(f"is_transcribed: {'✅' if result.is_transcribed else '❌'}")
                    self.logger.info(f"is_diarized: {'✅' if result.is_diarized else '❌'}")
                    self.logger.info(f"is_stitched: {'✅' if result.is_stitched else '❌'}")

                    # Duration coverage check
                    if result.expected_duration and result.expected_duration > 0 and result.chunk_count > 0:
                        coverage_pct = (result.max_chunk_end / result.expected_duration) * 100
                        self.logger.info(f"\n📏 Duration Coverage:")
                        self.logger.info(f"Expected duration: {result.expected_duration:.1f}s")
                        self.logger.info(f"Chunk coverage: {result.max_chunk_end:.1f}s ({result.chunk_count} chunks)")
                        self.logger.info(f"Coverage: {coverage_pct:.1f}%")

                        if coverage_pct < 90:
                            self.logger.warning(f"⚠️  TRUNCATED AUDIO: Only {coverage_pct:.1f}% of expected duration covered!")
                            duration_check = {
                                'expected': result.expected_duration,
                                'actual': result.max_chunk_end,
                                'coverage_pct': coverage_pct,
                                'truncated': True
                            }
                        else:
                            self.logger.info("✅ Duration coverage OK")
                            duration_check = {
                                'expected': result.expected_duration,
                                'actual': result.max_chunk_end,
                                'coverage_pct': coverage_pct,
                                'truncated': False
                            }

            return {
                'status': 'success',
                'content_id': content_id,
                'source_exists': source_exists,
                'main_audio': main_audio,
                'chunks_exist': chunks_exist,
                'diarization_exists': diarization_exists,
                'stitch_exists': stitch_exists,
                'duration_check': duration_check
            }
            
        except Exception as e:
            self.logger.error(f"Error auditing content {content_id}: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }

    async def initialize(self) -> None:
        """Initialize the auditor with async operations"""
        await self.build_content_map()

    async def audit_truncated_content(self, coverage_threshold: float = 0.9, reset: bool = False) -> Dict:
        """
        Find and optionally reset content with truncated audio extraction.

        Args:
            coverage_threshold: Minimum ratio of chunk coverage to expected duration (default 0.9 = 90%)
            reset: If True, reset truncated items for reprocessing

        Returns:
            Dict with truncated content info and reset counts
        """
        try:
            self.logger.info(f"\n🔍 Auditing for truncated audio (coverage < {coverage_threshold*100:.0f}%)")

            with get_session() as session:
                # Find content where chunk coverage is less than threshold of expected duration
                query = text("""
                    WITH chunk_coverage AS (
                        SELECT
                            c.id,
                            c.content_id,
                            c.title,
                            c.duration as expected_duration,
                            c.platform,
                            c.is_transcribed,
                            COUNT(cc.id) as chunk_count,
                            COALESCE(MAX(cc.end_time), 0) as max_chunk_end
                        FROM content c
                        LEFT JOIN content_chunks cc ON c.id = cc.content_id
                        WHERE c.is_converted = true
                          AND c.duration > 60
                        GROUP BY c.id, c.content_id, c.title, c.duration, c.platform, c.is_transcribed
                        HAVING COUNT(cc.id) > 0
                    )
                    SELECT
                        id,
                        content_id,
                        title,
                        platform,
                        expected_duration,
                        max_chunk_end,
                        chunk_count,
                        (max_chunk_end / expected_duration) as coverage_ratio
                    FROM chunk_coverage
                    WHERE max_chunk_end / expected_duration < :threshold
                    ORDER BY coverage_ratio ASC
                """)

                results = session.execute(query, {'threshold': coverage_threshold}).fetchall()

                if not results:
                    self.logger.info("✅ No truncated content found!")
                    return {'status': 'success', 'truncated_count': 0, 'reset_count': 0}

                self.logger.info(f"\n⚠️  Found {len(results)} content items with truncated audio:\n")

                # Group by coverage bucket for summary
                buckets = {'< 10%': [], '10-50%': [], '50-90%': []}
                for row in results:
                    pct = row.coverage_ratio * 100
                    if pct < 10:
                        buckets['< 10%'].append(row)
                    elif pct < 50:
                        buckets['10-50%'].append(row)
                    else:
                        buckets['50-90%'].append(row)

                for bucket, items in buckets.items():
                    if items:
                        self.logger.info(f"  {bucket} coverage: {len(items)} items")

                # Show some examples
                self.logger.info(f"\nSample truncated items:")
                for row in results[:10]:
                    pct = row.coverage_ratio * 100
                    self.logger.info(
                        f"  {row.content_id} ({row.platform}): "
                        f"{row.max_chunk_end:.0f}s / {row.expected_duration:.0f}s ({pct:.1f}%) "
                        f"- {row.title[:50]}"
                    )

                if len(results) > 10:
                    self.logger.info(f"  ... and {len(results) - 10} more")

                reset_count = 0
                if reset:
                    self.logger.info(f"\n🔄 Resetting {len(results)} truncated items for reprocessing...")

                    for row in results:
                        # Delete theme classifications first (FK constraint)
                        delete_theme_classifications = text("""
                            DELETE FROM theme_classifications
                            WHERE segment_id IN (
                                SELECT id FROM embedding_segments WHERE content_id = :content_id
                            )
                        """)
                        session.execute(delete_theme_classifications, {'content_id': row.id})

                        # Delete existing embedding segments
                        delete_segments = text("""
                            DELETE FROM embedding_segments WHERE content_id = :content_id
                        """)
                        session.execute(delete_segments, {'content_id': row.id})

                        # Delete existing sentences
                        delete_sentences = text("""
                            DELETE FROM sentences WHERE content_id = :content_id
                        """)
                        session.execute(delete_sentences, {'content_id': row.id})

                        # Delete existing chunks
                        delete_chunks = text("""
                            DELETE FROM content_chunks WHERE content_id = :content_id
                        """)
                        session.execute(delete_chunks, {'content_id': row.id})

                        # Reset content flags
                        reset_content = text("""
                            UPDATE content
                            SET is_converted = false,
                                is_transcribed = false,
                                is_diarized = false,
                                is_stitched = false,
                                processing_state = 0,
                                last_updated = NOW()
                            WHERE id = :content_id
                        """)
                        session.execute(reset_content, {'content_id': row.id})
                        reset_count += 1

                        if reset_count % 100 == 0:
                            self.logger.info(f"  Reset {reset_count}/{len(results)} items...")
                            session.commit()

                    session.commit()
                    self.logger.info(f"✅ Reset {reset_count} items for reprocessing")

                return {
                    'status': 'success',
                    'truncated_count': len(results),
                    'reset_count': reset_count,
                    'by_bucket': {k: len(v) for k, v in buckets.items()}
                }

        except Exception as e:
            self.logger.error(f"Error auditing truncated content: {str(e)}")
            return {'status': 'error', 'error': str(e)}

def load_config() -> Dict:
    """Load configuration from yaml file with env var substitution"""
    from src.utils.config import load_config as load_config_with_env
    return load_config_with_env()

async def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(description='Audit content processing state')
    parser.add_argument('--project', help='Specific project to audit')
    parser.add_argument('--start-date', help='Start date for audit (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date for audit (YYYY-MM-DD)')
    parser.add_argument('--fix', action='store_true', help='Fix inconsistencies found during audit')
    parser.add_argument('--content-id', help='Audit a single content item')
    parser.add_argument('--audit-truncated', action='store_true',
                       help='Find content with truncated audio extraction (coverage < 90%%)')
    parser.add_argument('--reset-truncated', action='store_true',
                       help='Reset truncated items for reprocessing (use with --audit-truncated)')
    parser.add_argument('--coverage-threshold', type=float, default=0.9,
                       help='Coverage threshold for truncation detection (default: 0.9 = 90%%)')
    args = parser.parse_args()

    # Setup console logging for visibility
    import logging
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logging.getLogger().addHandler(console_handler)
    logging.getLogger().setLevel(logging.INFO)

    config = load_config()
    auditor = StateAuditor(config)
    # Ensure auditor logger outputs to console
    auditor.logger.addHandler(console_handler)

    try:
        if args.audit_truncated:
            # Audit for truncated audio
            result = await auditor.audit_truncated_content(
                coverage_threshold=args.coverage_threshold,
                reset=args.reset_truncated
            )
            print(f"\nResult: {result}")
            return

        if args.content_id:
            # Single content mode - only audit that content item
            result = await auditor.audit_single_content(args.content_id)
            if args.fix and result['status'] == 'success':
                # We need to first initialize the content map to fix inconsistencies
                await auditor.initialize()
                await auditor.fix_inconsistencies()
            return
            
        # Full audit mode - initialize auditor with content map
        await auditor.initialize()
        
        # Audit content
        await auditor.audit_content(
            project=args.project,
            start_date=datetime.strptime(args.start_date, '%Y-%m-%d') if args.start_date else None,
            end_date=datetime.strptime(args.end_date, '%Y-%m-%d') if args.end_date else None
        )
        
        # Print summary
        auditor.audit_summary.print_summary()
        
        # Fix inconsistencies if requested
        if args.fix:
            await auditor.fix_inconsistencies()
            
    except Exception as e:
        auditor.logger.error(f"Error during audit: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    asyncio.run(main()) 