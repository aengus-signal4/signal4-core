"""
Task Creation Manager
=====================

Orchestrator-driven task creation for content processing pipeline.
Creates tasks for: indexing, download, convert, transcribe, diarize, stitch, segment, cleanup.

Usage (as module):
    python -m src.automation.task_creation_manager --steps index_podcast download_podcast
    python -m src.automation.task_creation_manager --project CPRMV --steps download convert transcribe

Called by scheduled tasks defined in config/config.yaml under scheduled_tasks.
"""

import sys
import asyncio
from pathlib import Path
import argparse
from datetime import datetime, timedelta, timezone
import logging
import yaml
import time
from typing import Dict, Optional, List
from sqlalchemy.exc import OperationalError
from sqlalchemy import and_
from collections import defaultdict
from sqlalchemy.sql import text
from sqlalchemy.sql import func
import json
from dataclasses import dataclass, field
from typing import DefaultDict

from src.utils.logger import setup_worker_logger
from src.utils.status import get_project_status, print_status_report
from src.utils.priority import calculate_priority_by_date
from src.processing.content import process_content_pipeline
from src.distributed.task_queue import TaskQueueManager
from src.database.manager import DatabaseManager
from src.database.session import get_session
from src.database.models import TaskQueue, WorkerConfig, Content, ContentChunk, SpeakerTranscription, Channel, ChannelProject, ChannelSource
from src.ingestion.youtube_indexer import YouTubeIndexer
from src.ingestion.podcast_indexer import PodcastIndexer
from src.ingestion.rumble_indexer import RumbleIndexer
from src.storage.s3_utils import S3Storage, S3StorageConfig
from src.storage.content_storage import ContentStorageManager
from src.storage.config import get_storage_config
from src.utils.version_utils import should_recreate_stitch_task, format_version_comparison_log

# Set up logger
logger = setup_worker_logger('create_tasks')

@dataclass
class TaskSummary:
    """Tracks task creation statistics across projects"""
    projects: DefaultDict[str, DefaultDict[str, int]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    
    def add_tasks(self, project: str, task_type: str, count: int):
        """Record tasks created for a project/type combination"""
        self.projects[project][task_type] += count
        
    def print_summary(self):
        """Print a formatted summary of all tasks created"""
        if not self.projects:
            logger.info("\n📊 No tasks were created")
            return
            
        logger.info("\n📊 Task Creation Summary")
        logger.info("=" * 60)
        
        # Calculate column widths
        project_width = max(len("Project"), max(len(p) for p in self.projects.keys()))
        type_width = max(len("Task Type"), max(len(t) for p in self.projects.values() for t in p.keys()))
        count_width = max(len("Count"), max(len(str(c)) for p in self.projects.values() for c in p.values()))
        
        # Print header
        header = f"{'Project':<{project_width}} | {'Task Type':<{type_width}} | {'Count':>{count_width}}"
        logger.info(header)
        logger.info("-" * len(header))
        
        # Print each project's tasks
        total_tasks = 0
        for project, task_types in sorted(self.projects.items()):
            project_total = sum(task_types.values())
            total_tasks += project_total
            
            # Print each task type
            for task_type, count in sorted(task_types.items()):
                # Format task type for display
                display_type = task_type.replace('_', ' ').title()
                if task_type == 'download_youtube':
                    display_type = 'YouTube Download'
                elif task_type == 'download_podcast':
                    display_type = 'Podcast Download'
                elif task_type == 'download_rumble':
                    display_type = 'Rumble Download'
                logger.info(f"{project:<{project_width}} | {display_type:<{type_width}} | {count:>{count_width}}")
            
            # Print project subtotal if multiple task types
            if len(task_types) > 1:
                logger.info("-" * len(header))
                logger.info(f"{project:<{project_width}} | {'SUBTOTAL':<{type_width}} | {project_total:>{count_width}}")
                logger.info("-" * len(header))
        
        # Print grand total if multiple projects
        if len(self.projects) > 1:
            logger.info("=" * len(header))
            logger.info(f"{'TOTAL':<{project_width}} | {'(all projects)':<{type_width}} | {total_tasks:>{count_width}}")
        
        logger.info("=" * 60)

class TaskCreator:
    """Creates and manages tasks for content processing."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = setup_worker_logger('task_creator')
        self.task_queue = TaskQueueManager(config['database'])
        from src.database.state_manager import StateManager
        self.state_manager = StateManager(logger=self.logger)
        self.task_summary = TaskSummary()  # Initialize task_summary as instance variable
        
    async def create_download_tasks(self, project: str, clean: bool = False, date_range: Optional[Dict] = None, download_type: Optional[str] = None) -> int:
        """Create download tasks for content that needs to be downloaded.
        
        Args:
            project: Project name
            clean: Whether to clean existing tasks first
            date_range: Date range for content filtering
            download_type: Optional, either 'download_podcast', 'download_youtube', or 'download_rumble' to filter by platform
        """
        try:
            with get_session() as session:
                # Get project date range from config if not provided
                if not date_range:
                    config = load_config()
                    project_config = config.get('active_projects', {}).get(project, {})
                    if project_config:
                        date_range = {
                            'start': parse_date(project_config.get('start_date')),
                            'end': parse_date(project_config.get('end_date'))
                        }
                
                # Get diagnostic counts first
                diagnostic_query = text("""
                    SELECT
                        COUNT(*) as total_content,
                        COUNT(*) FILTER (WHERE is_downloaded = true) as downloaded,
                        COUNT(*) FILTER (WHERE blocked_download = true) as blocked,
                        COUNT(*) FILTER (WHERE meta_data->>'permanent_block' = 'true') as permanently_blocked,
                        COUNT(*) FILTER (WHERE blocked_download = true AND meta_data->>'retry_after' IS NOT NULL) as blocked_with_retry,
                        COUNT(*) FILTER (WHERE blocked_download = true AND meta_data->>'retry_after' IS NULL) as blocked_without_retry,
                        COUNT(*) FILTER (WHERE blocked_download = true AND (meta_data->>'retry_after')::timestamp < NOW()) as blocked_retry_expired,
                        COUNT(*) FILTER (WHERE platform = 'youtube' AND is_downloaded = false) as youtube_pending,
                        COUNT(*) FILTER (WHERE platform = 'podcast' AND is_downloaded = false) as podcast_pending,
                        COUNT(*) FILTER (WHERE platform = 'rumble' AND is_downloaded = false) as rumble_pending,
                        COUNT(*) FILTER (WHERE publish_date >= :start_date AND publish_date <= :end_date) as content_in_date_range
                    FROM content
                    WHERE :project = ANY(projects)
                """)
                
                diagnostics = session.execute(diagnostic_query, {
                    'project': project,
                    'start_date': date_range['start'] if date_range else None,
                    'end_date': date_range['end'] if date_range else None
                }).fetchone()
                
                logger.info("\nDownload Task Diagnostics:")
                logger.info(f"  Total content items: {diagnostics.total_content}")
                logger.info(f"  Content in date range: {diagnostics.content_in_date_range}")
                logger.info(f"  Downloaded: {diagnostics.downloaded}")
                logger.info(f"  Blocked: {diagnostics.blocked}")
                logger.info(f"  Permanently blocked: {diagnostics.permanently_blocked}")
                logger.info(f"  Blocked with retry: {diagnostics.blocked_with_retry}")
                logger.info(f"  Blocked without retry: {diagnostics.blocked_without_retry}")
                logger.info(f"  Blocked retry expired: {diagnostics.blocked_retry_expired}")
                logger.info(f"  YouTube pending downloads: {diagnostics.youtube_pending}")
                logger.info(f"  Podcast pending downloads: {diagnostics.podcast_pending}")
                logger.info(f"  Rumble pending downloads: {diagnostics.rumble_pending}")
                
                # Base query for content needing download, with platform-specific filters if needed
                query = text("""
                    WITH ranked_content AS (
                        SELECT
                            c.id,
                            c.content_id,
                            c.platform,
                            c.meta_data,
                            c.blocked_download,
                            c.publish_date,
                            c.duration,
                            c.channel_url,
                            c.projects,
                            ROW_NUMBER() OVER (
                                PARTITION BY c.platform
                                ORDER BY c.publish_date DESC
                            ) as platform_rank
                        FROM content c
                        WHERE :project = ANY(c.projects)
                        AND c.is_downloaded = false
                        AND c.is_converted = false  -- Don't create download tasks if already converted
                        AND (
                            -- Not blocked at all
                            c.blocked_download = false
                            OR
                            -- Or blocked but retry period has passed and not permanently blocked
                            (
                                c.blocked_download = true 
                                AND c.meta_data->>'permanent_block' IS DISTINCT FROM 'true'
                                AND (
                                    c.meta_data->>'retry_after' IS NULL
                                    OR
                                    (c.meta_data->>'retry_after')::timestamp < NOW()
                                )
                            )
                        )
                        -- Apply platform filter if specified
                        AND (:platform IS NULL OR c.platform = :platform)
                        -- Skip short content only for podcasts (not for YouTube/Rumble)
                        AND (
                            c.platform IN ('youtube', 'rumble')
                            OR c.duration >= 180 
                            OR c.duration IS NULL
                        )
                        AND (:start_date IS NULL OR c.publish_date >= :start_date)
                        AND (:end_date IS NULL OR c.publish_date <= :end_date)
                        -- Add check for existing tasks
                        AND NOT EXISTS (
                            SELECT 1 FROM tasks.task_queue tq
                            WHERE tq.content_id = c.content_id
                            AND tq.task_type IN ('download_youtube', 'download_podcast', 'download_rumble')
                            AND tq.status IN ('pending', 'processing')
                        )
                        -- Skip content with permanent download failures (SSL errors, etc)
                        AND NOT EXISTS (
                            SELECT 1 FROM tasks.task_queue tq
                            WHERE tq.content_id = c.content_id
                            AND tq.task_type IN ('download_youtube', 'download_podcast', 'download_rumble')
                            AND tq.status = 'failed'
                            AND tq.result->>'permanent' = 'true'
                        )
                    )
                    SELECT *
                    FROM ranked_content
                    ORDER BY publish_date DESC, platform_rank ASC
                """)
                
                # Determine platform filter based on download_type
                platform = None
                if download_type == 'download_podcast':
                    platform = 'podcast'
                elif download_type == 'download_youtube':
                    platform = 'youtube'
                elif download_type == 'download_rumble':
                    platform = 'rumble'
                
                results = session.execute(query, {
                    'project': project,
                    'platform': platform,
                    'start_date': date_range['start'] if date_range else None,
                    'end_date': date_range['end'] if date_range else None
                })
                rows = list(results)

                if not rows:
                    logger.info(f"No content found requiring download tasks for project {project} " + (f"(type: {download_type})" if download_type else ""))
                    return 0

                logger.info(f"Found {len(rows)} content items requiring download tasks for project {project}")

                # Process in batches
                batch_size = 1000
                tasks_created = 0
                youtube_tasks_created = 0
                podcast_tasks_created = 0
                rumble_tasks_created = 0

                for i in range(0, len(rows), batch_size):
                    batch_rows = rows[i:i + batch_size]
                    batch_data = []
                    skipped_permanently_blocked = 0
                    skipped_missing_feed_url = 0

                    # Prepare batch data
                    # Load config once per batch
                    config = load_config()
                    active_projects_config = config.get('active_projects', {})

                    for row in batch_rows:
                        # Skip if permanently blocked
                        meta = row.meta_data or {}
                        if meta.get('permanent_block'):
                            skipped_permanently_blocked += 1
                            continue

                        if row.platform == 'youtube':
                            task_type = 'download_youtube'
                        elif row.platform == 'podcast':
                            task_type = 'download_podcast'
                        elif row.platform == 'rumble':
                            task_type = 'download_rumble'
                        else:
                            logger.warning(f"Unknown platform {row.platform} for content {row.content_id}")
                            continue
                        # Use the project passed to the script as the primary project for input_data
                        task_input_data = {'project': project}

                        if row.platform == 'podcast':
                            if not row.channel_url:
                                skipped_missing_feed_url += 1
                                logger.warning(f"Skipping podcast download task for {row.content_id}: Missing channel_url in database.")
                                continue
                            task_input_data['feed_url'] = row.channel_url

                        # ---- START PRIORITY CALCULATION UPDATE ----
                        # Get all projects for this content item
                        projects_str = row.projects
                        project_list = projects_str if isinstance(projects_str, list) else [projects_str] if projects_str else []

                        # Find the maximum priority among the associated projects
                        max_project_priority = 1 # Default priority
                        for proj_name in project_list:
                            proj_name = proj_name.strip() # Clean up whitespace
                            if proj_name in active_projects_config:
                                proj_config = active_projects_config[proj_name]
                                # Only consider enabled projects for priority calculation
                                if proj_config.get('enabled', False):
                                    priority_val = proj_config.get('priority', 1)
                                    max_project_priority = max(max_project_priority, priority_val)

                        # Calculate priority using max project priority and publish date
                        priority = calculate_priority_by_date(row.publish_date, max_project_priority) if row.publish_date else (max_project_priority * 1000000)
                        # ---- END PRIORITY CALCULATION UPDATE ----

                        batch_data.append({
                            'content_id': row.content_id,
                            'task_type': task_type,
                            'input_data': task_input_data,
                            'priority': priority,
                            'platform': row.platform # Keep platform for counting later
                        })

                    # Log skipped items for this batch
                    if skipped_permanently_blocked > 0:
                        logger.info(f"Skipped {skipped_permanently_blocked} permanently blocked items in this batch.")
                    if skipped_missing_feed_url > 0:
                        logger.warning(f"Skipped {skipped_missing_feed_url} podcast items due to missing feed_url in this batch.")

                    if not batch_data:
                        logger.debug("Batch is empty after filtering, skipping insert.")
                        continue

                    # Insert tasks using bulk insert method
                    # Note: input_data needs to be serialized to JSON string here
                    # We also need to define the structure for jsonb_to_recordset
                    insert_query = text("""
                        INSERT INTO tasks.task_queue (
                            task_type, content_id, status, priority, input_data
                        )
                        SELECT
                            x.task_type,
                            x.content_id,
                            'pending',
                            x.priority,
                            x.input_data::jsonb -- Ensure input_data is stored as jsonb
                        FROM jsonb_to_recordset(:batch_data) AS x(
                            content_id text,
                            task_type text,
                            input_data jsonb, -- Use jsonb type here
                            priority int
                        )
                        WHERE NOT EXISTS (
                            SELECT 1 FROM tasks.task_queue tq
                            WHERE tq.content_id = x.content_id
                            AND tq.task_type = x.task_type
                            AND tq.status IN ('pending', 'processing')
                        )
                    """)

                    # Prepare batch data for JSON serialization
                    serializable_batch_data = [
                        {
                            'content_id': task['content_id'],
                            'task_type': task['task_type'],
                            'input_data': json.dumps(task['input_data']), # Serialize input_data
                            'priority': task['priority']
                        } for task in batch_data
                    ]

                    try:
                        result = session.execute(insert_query, {
                            'batch_data': json.dumps(serializable_batch_data) # Pass the serialized list as JSON
                        })
                        session.commit()
                        tasks_created_this_batch = result.rowcount
                        tasks_created += tasks_created_this_batch

                        # Update platform-specific counts and summary
                        batch_yt_count = sum(1 for task in batch_data if task['platform'] == 'youtube')
                        batch_podcast_count = sum(1 for task in batch_data if task['platform'] == 'podcast')
                        batch_rumble_count = sum(1 for task in batch_data if task['platform'] == 'rumble')

                        youtube_tasks_created += batch_yt_count
                        podcast_tasks_created += batch_podcast_count
                        rumble_tasks_created += batch_rumble_count

                        if batch_yt_count > 0:
                            self.task_summary.add_tasks(project, 'download_youtube', batch_yt_count)
                        if batch_podcast_count > 0:
                            self.task_summary.add_tasks(project, 'download_podcast', batch_podcast_count)
                        if batch_rumble_count > 0:
                            self.task_summary.add_tasks(project, 'download_rumble', batch_rumble_count)

                        if tasks_created_this_batch > 0:
                             logger.debug(f"Successfully created {tasks_created_this_batch} download tasks in this batch ({batch_yt_count} YT, {batch_podcast_count} Podcast, {batch_rumble_count} Rumble).")

                    except Exception as insert_error:
                        logger.error(f"Error inserting batch of download tasks: {insert_error}")
                        session.rollback() # Rollback failed batch

                if tasks_created > 0:
                    logger.info(f"\nCreated {tasks_created} download tasks for project {project}:")
                    logger.info(f"  YouTube: {youtube_tasks_created}")
                    logger.info(f"  Podcast: {podcast_tasks_created}")
                    logger.info(f"  Rumble: {rumble_tasks_created}")
                else:
                    logger.info(f"No new download tasks were created for project {project} (either already exist or failed).")

                return tasks_created

        except Exception as e:
            logger.error(f"Error creating download tasks for {project}: {str(e)}", exc_info=True) # Add exc_info
            return 0
            
    async def create_convert_tasks(self, project: str, clean: bool = False, date_range: Optional[Dict] = None) -> int:
        """Create audio extraction tasks for content that needs it"""
        try:
            # Load config for project settings
            config = load_config()
            # Get project date range from config if not provided
            if not date_range:
                config = load_config()
                project_config = config.get('active_projects', {}).get(project, {})
                if project_config:
                    date_range = {
                        'start': parse_date(project_config.get('start_date')),
                        'end': parse_date(project_config.get('end_date'))
                    }
            
            # Get content needing audio extraction
            with get_session() as session:
                # Clean existing tasks if requested
                if clean:
                    await cleanup_task_queue(project)
                
                # Simplified query that only relies on database flags
                query = text("""
                    SELECT 
                        c.content_id,
                        c.duration,
                        c.platform,
                        c.publish_date
                    FROM content c
                    WHERE :project = ANY(c.projects)
                    AND c.is_downloaded = true
                    AND c.is_converted = false
                    AND NOT EXISTS (
                        SELECT 1 FROM tasks.task_queue t
                        WHERE t.content_id = c.content_id
                        AND t.task_type = 'convert'
                        AND t.status IN ('pending', 'processing')
                    )
                    AND (:start_date IS NULL OR c.publish_date >= :start_date)
                    AND (:end_date   IS NULL OR c.publish_date <= :end_date)
                    ORDER BY c.publish_date DESC NULLS LAST, c.content_id ASC
                """)
                
                results = session.execute(query, {
                    'project': f"%{project}%",
                    'start_date': date_range['start'] if date_range else None,
                    'end_date': date_range['end'] if date_range else None
                }).fetchall()
                
                if not results:
                    return 0
                    
                # Process in batches
                batch_size = 1000
                tasks_created = 0
                
                for i in range(0, len(results), batch_size):
                    batch = results[i:i + batch_size]
                    
                    # Prepare batch data with priority based on publish date
                    batch_data = []
                    for row in batch:
                        project_priority = config['active_projects'][project].get('priority', 1)
                        priority = calculate_priority_by_date(row.publish_date, project_priority) if row.publish_date else (project_priority * 1000000)
                        batch_data.append({
                            'content_id': row.content_id,
                            'duration': float(row.duration),
                            'platform': row.platform,
                            'priority': priority
                        })
                    
                    # Insert tasks
                    insert_query = text("""
                        INSERT INTO tasks.task_queue (
                            task_type, content_id, status, priority, input_data
                        )
                        SELECT 
                            'convert',
                            x.content_id,
                            'pending',
                            x.priority,
                            jsonb_build_object(
                                'project', :project,
                                'duration', x.duration::float,
                                'platform', x.platform
                            )
                        FROM jsonb_to_recordset(:batch_data) AS x(
                            content_id text,
                            duration float,
                            platform text,
                            priority int
                        )
                        WHERE NOT EXISTS (
                            SELECT 1 FROM tasks.task_queue t
                            WHERE t.content_id = x.content_id
                            AND t.task_type = 'convert'
                            AND t.status IN ('pending', 'processing')
                        )
                    """)
                    
                    result = session.execute(insert_query, {
                        'project': project,
                        'batch_data': json.dumps(batch_data)
                    })
                    
                    session.commit()
                    tasks_created += result.rowcount
                
                logger.info(f"\nCreated {tasks_created} audio extraction tasks")
                return tasks_created
                
        except Exception as e:
            logger.error(f"Error creating audio tasks: {str(e)}")
            return 0
            
    async def create_transcription_tasks(self, project: str, clean: bool = False, date_range: Optional[Dict] = None) -> int:
        """Create transcription tasks for chunks ready for transcription"""
        try:
            # Get project date range from config if not provided
            if not date_range:
                config = load_config()
                project_config = config.get('active_projects', {}).get(project, {})
                if project_config:
                    date_range = {
                        'start': parse_date(project_config.get('start_date')),
                        'end': parse_date(project_config.get('end_date'))
                    }
            
            with get_session() as session:
                # Clean existing tasks if requested
                if clean:
                    await cleanup_task_queue(project)
                
                # Find all chunks needing transcription - improved query to group by content
                query = text("""
                WITH content_chunks AS (
                    SELECT 
                        c.id as content_id,
                        c.content_id as external_id,
                        c.publish_date,
                        cc.chunk_index,
                        cc.start_time,
                        cc.end_time,
                        cc.duration,
                        cc.extraction_status,
                        cc.transcription_status,
                        cc.transcription_attempts,
                        -- Add row number within each content item to order chunks
                        ROW_NUMBER() OVER (
                            PARTITION BY c.content_id 
                            ORDER BY cc.chunk_index ASC
                        ) as chunk_sequence
                    FROM content c
                    JOIN content_chunks cc ON c.id = cc.content_id
                    WHERE 
                        :project = ANY(c.projects)
                        AND (:start_date IS NULL OR c.publish_date >= :start_date)
                        AND (:end_date IS NULL OR c.publish_date <= :end_date)
                        AND c.is_converted = true  -- Ensure source.wav exists
                        AND cc.extraction_status = 'completed'  -- Chunk must be extracted
                        AND c.is_transcribed = false  -- Only create tasks if content isn't fully transcribed
                        AND (
                            -- Never attempted transcription
                            cc.transcription_status = 'pending'
                            OR
                            -- Failed but under retry limit
                            (
                                cc.transcription_status = 'failed'
                                AND cc.transcription_attempts < 3
                            )
                            OR
                            -- Processing for too long (stuck)
                            (
                                cc.transcription_status = 'processing'
                                AND cc.last_updated < NOW() - INTERVAL '1 hour'
                            )
                        )
                )
                SELECT *
                FROM content_chunks cc
                WHERE NOT EXISTS (
                    SELECT 1 FROM tasks.task_queue tq
                    WHERE tq.content_id = cc.external_id
                    AND tq.task_type = 'transcribe'
                    AND tq.status IN ('pending', 'processing')
                    AND CAST(tq.input_data->>'chunk_index' AS INTEGER) = cc.chunk_index
                )
                ORDER BY 
                    -- Order by publish date DESC to prioritize newer content
                    publish_date DESC NULLS LAST,
                    -- Then by content_id to group chunks together
                    external_id ASC,
                    -- Then by chunk sequence to maintain order within content
                    chunk_sequence ASC
                """)
                
                results = session.execute(query, {
                    'project': f"%{project}%",
                    'start_date': date_range['start'] if date_range else None,
                    'end_date': date_range['end'] if date_range else None
                })
                rows = list(results)
                
                if not rows:
                    return 0
                    
                # Process chunks in batches
                batch_size = 1000
                tasks_created = 0
                
                for i in range(0, len(rows), batch_size):
                    batch = rows[i:i + batch_size]
                    
                    # Prepare batch data with priority based on content order and chunk sequence
                    batch_data = []
                    for row in batch:
                        # Base priority on publish date with much higher weight
                        project_priority = self.config['active_projects'][project].get('priority', 1)
                        base_priority = calculate_priority_by_date(row.publish_date, project_priority) if row.publish_date else (project_priority * 1000000)
                        # Adjust priority to keep chunks from same content together
                        # Subtract chunk_sequence to prioritize earlier chunks but maintain content grouping
                        # Convert to integer by rounding
                        priority = max(1, int(base_priority - (row.chunk_sequence / 1000)))
                        
                        batch_data.append({
                            'content_id': row.external_id,
                            'chunk_index': row.chunk_index,
                            'start_time': row.start_time,
                            'end_time': row.end_time,
                            'duration': row.duration,
                            'priority': priority
                        })
                    
                    # Insert tasks
                    insert_query = text("""
                        INSERT INTO tasks.task_queue (
                            task_type, content_id, status, priority, input_data
                        )
                        SELECT 
                            'transcribe',
                            x.content_id,
                            'pending',
                            x.priority,
                            jsonb_build_object(
                                'project', :project,
                                'chunk_index', x.chunk_index,
                                'start_time', x.start_time,
                                'end_time', x.end_time,
                                'duration', x.duration
                            )
                        FROM jsonb_to_recordset(:batch_data) AS x(
                            content_id text,
                            chunk_index int,
                            start_time float,
                            end_time float,
                            duration float,
                            priority int
                        )
                        WHERE NOT EXISTS (
                            SELECT 1 FROM tasks.task_queue tq
                            WHERE tq.content_id = x.content_id
                            AND tq.task_type = 'transcribe'
                            AND CAST(tq.input_data->>'chunk_index' AS INTEGER) = x.chunk_index
                            AND tq.status IN ('pending', 'processing')
                        )
                    """)
                    
                    result = session.execute(insert_query, {
                        'project': project,
                        'batch_data': json.dumps(batch_data)
                    })
                    
                    session.commit()
                    tasks_created += result.rowcount
                
                logger.info(f"\nCreated {tasks_created} transcription tasks")
                return tasks_created
                
        except Exception as e:
            logger.error(f"Error creating transcription tasks: {str(e)}")
            return 0
            
    async def create_diarization_tasks(self, project: str, clean: bool = False, date_range: Optional[Dict] = None) -> int:
        """Create diarization tasks for content that has been converted (audio ready) but not yet diarized"""
        try:
            # Get project date range from config if not provided
            if not date_range:
                config = load_config()
                project_config = config.get('active_projects', {}).get(project, {})
                if project_config:
                    date_range = {
                        'start': parse_date(project_config.get('start_date')),
                        'end': parse_date(project_config.get('end_date'))
                    }
            
            with get_session() as session:
                # Clean existing tasks if requested
                if clean:
                    await cleanup_task_queue(project)
                
                # Find content ready for diarization (only needs audio conversion, not transcription)
                # The new diarize approach just runs pyannote on the audio file
                query = text("""
                    SELECT 
                        c.content_id,
                        c.publish_date
                    FROM content c
                    WHERE :project = ANY(c.projects)
                    AND (:start_date IS NULL OR c.publish_date >= :start_date)
                    AND (:end_date IS NULL OR c.publish_date <= :end_date)
                    AND c.is_converted = true  -- Must have audio.wav
                    AND c.is_diarized = false  -- Not yet diarized
                    AND NOT EXISTS (
                        SELECT 1 FROM tasks.task_queue tq
                        WHERE tq.content_id = c.content_id
                        AND tq.task_type = 'diarize'
                        AND tq.status IN ('pending', 'processing')
                    )
                    ORDER BY c.publish_date DESC NULLS LAST, c.content_id ASC
                """)
                
                results = session.execute(query, {
                    'project': f"%{project}%",
                    'start_date': date_range['start'] if date_range else None,
                    'end_date': date_range['end'] if date_range else None
                })
                rows = list(results)
                
                if not rows:
                    return 0
                    
                # Process in batches
                batch_size = 1000
                tasks_created = 0
                
                for i in range(0, len(rows), batch_size):
                    batch = rows[i:i + batch_size]
                    
                    # Prepare batch data with priority based on publish date
                    batch_data = []
                    for row in batch:
                        project_priority = self.config['active_projects'][project].get('priority', 1)
                        priority = calculate_priority_by_date(row.publish_date, project_priority) if row.publish_date else (project_priority * 1000000)
                        batch_data.append({
                            'content_id': row.content_id,
                            'priority': priority
                        })
                    
                    # Insert tasks
                    insert_query = text("""
                        INSERT INTO tasks.task_queue (
                            task_type, content_id, status, priority, input_data
                        )
                        SELECT 
                            'diarize',
                            x.content_id,
                            'pending',
                            x.priority,
                            jsonb_build_object(
                                'project', :project
                            )
                        FROM jsonb_to_recordset(:batch_data) AS x(
                            content_id text,
                            priority int
                        )
                        WHERE NOT EXISTS (
                            SELECT 1 FROM tasks.task_queue tq
                            WHERE tq.content_id = x.content_id
                            AND tq.task_type = 'diarize'
                            AND tq.status IN ('pending', 'processing')
                        )
                    """)
                    
                    result = session.execute(insert_query, {
                        'project': project,
                        'batch_data': json.dumps(batch_data)
                    })
                    
                    session.commit()
                    tasks_created += result.rowcount
                
                logger.info(f"\nCreated {tasks_created} diarization tasks")
                return tasks_created
                
        except Exception as e:
            logger.error(f"Error creating diarization tasks: {str(e)}")
            return 0

    async def create_identify_speakers_tasks(self, project: str, clean: bool = False, date_range: Optional[Dict] = None) -> int:
        """DEPRECATED: Speaker identification is now integrated into the stitch step."""
        logger.warning(f"create_identify_speakers_tasks is deprecated - speaker identification is now handled by the stitch step")
        return 0  # Return 0 tasks created since this step is now integrated into stitch

    async def create_stitch_tasks(self, project: str, clean: bool = False, date_range: Optional[Dict] = None) -> int:
        """Create stitching tasks for content with transcription AND diarization complete"""
        task_type = 'stitch'
        tasks_created = 0
        try:
            # Get project date range from config if not provided
            if not date_range:
                config = load_config()
                project_config = config.get('active_projects', {}).get(project, {})
                if project_config:
                    date_range = {
                        'start': parse_date(project_config.get('start_date')),
                        'end': parse_date(project_config.get('end_date'))
                    }

            # Get current stitch version from config
            current_version = self.config.get('processing', {}).get('stitch', {}).get('current_version', 'stitch_v1')

            with get_session() as session:
                # Find content that needs stitching - requires both transcription AND diarization
                # The stitch step now also handles speaker identification internally
                query = session.query(Content).filter(
                    Content.is_converted == True,
                    Content.is_transcribed == True,
                    Content.is_diarized == True,  # Requires diarization completion
                    Content.projects.any(project)
                )

                # Add date range filter if provided
                if date_range:
                    if date_range.get('start'):
                        query = query.filter(Content.publish_date >= date_range['start'])
                    if date_range.get('end'):
                        query = query.filter(Content.publish_date <= date_range['end'])

                # Get all potential candidates for stitching
                candidates = query.order_by(Content.publish_date.desc().nullslast(), Content.content_id.asc()).all()
                
                # Filter for content that either:
                # 1. Has never been stitched (is_stitched = False)
                # 2. Was stitched with an incompatible version (using version compatibility logic)
                needs_stitching = []
                for content in candidates:
                    if not content.is_stitched:
                        needs_stitching.append(content)
                    elif should_recreate_stitch_task(current_version, content.stitch_version):
                        needs_stitching.append(content)
                        logger.debug(f"Content {content.content_id} needs re-stitching: {format_version_comparison_log(current_version, content.stitch_version)}")

                if not needs_stitching:
                    logger.info(f"No content found that needs stitching for project {project}")
                    return 0

                logger.info(f"Found {len(needs_stitching)} content items that need stitching for project {project}")

                # Process in batches
                batch_size = 1000 # Use batching
                for i in range(0, len(needs_stitching), batch_size):
                    batch_content = needs_stitching[i:i + batch_size]
                    batch_data = []

                    # Prepare batch data
                    for content in batch_content:
                        # Calculate priority based on publish date
                        project_priority = self.config['active_projects'][project].get('priority', 1)
                        priority = calculate_priority_by_date(content.publish_date, project_priority) if content.publish_date else (project_priority * 1000000)

                        batch_data.append({
                            'content_id': content.content_id,
                            'priority': priority
                        })

                    # Insert tasks using bulk insert method
                    insert_query = text(f"""
                        INSERT INTO tasks.task_queue (
                            task_type, content_id, status, priority, input_data
                        )
                        SELECT
                            '{task_type}',
                            x.content_id,
                            'pending',
                            x.priority,
                            jsonb_build_object('project', :project) -- Basic input data
                        FROM jsonb_to_recordset(:batch_data) AS x(
                            content_id text,
                            priority int
                        )
                        WHERE NOT EXISTS (
                            SELECT 1 FROM tasks.task_queue tq
                            WHERE tq.content_id = x.content_id
                            AND tq.task_type = '{task_type}'
                            AND tq.status IN ('pending', 'processing')
                        )
                    """)

                    try:
                        result = session.execute(insert_query, {
                            'project': project,
                            'batch_data': json.dumps(batch_data)
                        })
                        session.commit()
                        tasks_created_this_batch = result.rowcount
                        tasks_created += tasks_created_this_batch
                        if tasks_created_this_batch > 0:
                           self.task_summary.add_tasks(project, task_type, tasks_created_this_batch) # Update summary
                           logger.debug(f"Successfully created {tasks_created_this_batch} stitch tasks in this batch.")
                    except Exception as insert_error:
                        logger.error(f"Error inserting batch of stitch tasks: {insert_error}")
                        session.rollback() # Rollback failed batch

                if tasks_created > 0:
                    logger.info(f"\nCreated {tasks_created} {task_type} tasks for project {project}")
                else:
                    logger.info(f"No new {task_type} tasks were created for project {project} (either already exist or failed).")

                return tasks_created

        except Exception as e:
            logger.error(f"Error creating {task_type} tasks for {project}: {str(e)}", exc_info=True) # Add exc_info for more detail
            return 0

    async def create_segment_tasks(self, project: str, clean: bool = False, date_range: Optional[Dict] = None) -> int:
        """Create segment tasks for content with stitching complete"""
        task_type = 'segment'
        tasks_created = 0
        try:
            # Get project date range from config if not provided
            if not date_range:
                config = load_config()
                project_config = config.get('active_projects', {}).get(project, {})
                if project_config:
                    date_range = {
                        'start': parse_date(project_config.get('start_date')),
                        'end': parse_date(project_config.get('end_date'))
                    }

            # Get current segment version from config
            current_version = self.config.get('processing', {}).get('segment', {}).get('current_version', 'segment_v1')

            with get_session() as session:
                # Find content that needs semantic segmentation - requires stitching to be complete
                query = session.query(Content).filter(
                    Content.is_stitched == True,  # Must have speaker turns from stitch
                    Content.projects.any(project)
                )

                # Add date range filter if provided
                if date_range:
                    if date_range.get('start'):
                        query = query.filter(Content.publish_date >= date_range['start'])
                    if date_range.get('end'):
                        query = query.filter(Content.publish_date <= date_range['end'])

                # Filter for content that either:
                # 1. Has never been segmented (is_embedded = False)
                # 2. Was segmented with an old version (stored in meta_data)
                # Note: segment_version is tracked in meta_data['segment_version'] until DB schema is updated
                query = query.filter(
                    (Content.is_embedded == False) |
                    (func.coalesce(Content.meta_data.op('->>')('segment_version'), '') != current_version)
                )

                # Get content that needs segmentation
                needs_segmentation = query.order_by(Content.publish_date.desc().nullslast(), Content.content_id.asc()).all()

                if not needs_segmentation:
                    logger.info(f"No content found that needs segment tasks for project {project}")
                    return 0

                logger.info(f"Found {len(needs_segmentation)} content items that need segment tasks for project {project}")

                # Process in batches
                batch_size = 1000
                for i in range(0, len(needs_segmentation), batch_size):
                    batch_content = needs_segmentation[i:i + batch_size]
                    batch_data = []

                    # Prepare batch data
                    for content in batch_content:
                        # Calculate priority based on publish date
                        project_priority = self.config['active_projects'][project].get('priority', 1)
                        priority = calculate_priority_by_date(content.publish_date, project_priority) if content.publish_date else (project_priority * 1000000)

                        batch_data.append({
                            'content_id': content.content_id,
                            'priority': priority
                        })

                    # Insert tasks using bulk insert method
                    insert_query = text(f"""
                        INSERT INTO tasks.task_queue (
                            task_type, content_id, status, priority, input_data
                        )
                        SELECT
                            '{task_type}',
                            x.content_id,
                            'pending',
                            x.priority,
                            jsonb_build_object('project', :project)
                        FROM jsonb_to_recordset(:batch_data) AS x(
                            content_id text,
                            priority int
                        )
                        WHERE NOT EXISTS (
                            SELECT 1 FROM tasks.task_queue tq
                            WHERE tq.content_id = x.content_id
                            AND tq.task_type = '{task_type}'
                            AND tq.status IN ('pending', 'processing')
                        )
                    """)

                    try:
                        result = session.execute(insert_query, {
                            'project': project,
                            'batch_data': json.dumps(batch_data)
                        })
                        session.commit()
                        tasks_created_this_batch = result.rowcount
                        tasks_created += tasks_created_this_batch
                        if tasks_created_this_batch > 0:
                           self.task_summary.add_tasks(project, task_type, tasks_created_this_batch)
                           logger.debug(f"Successfully created {tasks_created_this_batch} segment tasks in this batch.")
                    except Exception as insert_error:
                        logger.error(f"Error inserting batch of segment tasks: {insert_error}")
                        session.rollback()

                if tasks_created > 0:
                    logger.info(f"\nCreated {tasks_created} {task_type} tasks for project {project}")
                else:
                    logger.info(f"No new {task_type} tasks were created for project {project} (either already exist or failed).")

                return tasks_created

        except Exception as e:
            logger.error(f"Error creating {task_type} tasks for {project}: {str(e)}", exc_info=True)
            return 0

    async def create_cleanup_tasks(self, project: str, clean: bool = False, date_range: Optional[Dict] = None) -> int:
        """Create cleanup tasks for content with stitching complete"""
        task_type = 'cleanup'
        tasks_created = 0
        try:
            # Get project date range from config if not provided
            if not date_range:
                config = load_config()
                project_config = config.get('active_projects', {}).get(project, {})
                if project_config:
                    date_range = {
                        'start': parse_date(project_config.get('start_date')),
                        'end': parse_date(project_config.get('end_date'))
                    }

            with get_session() as session:
                # Clean existing tasks if requested
                if clean:
                    await cleanup_task_queue(project, task_type)

                # Find content that needs cleanup - requires stitching to be complete
                query = session.query(Content).filter(
                    Content.is_stitched == True,  # Must have stitching complete
                    Content.projects.any(project),
                    (Content.is_compressed == False) | (Content.is_compressed.is_(None))  # Not yet compressed
                )

                # Add date range filter if provided
                if date_range:
                    if date_range.get('start'):
                        query = query.filter(Content.publish_date >= date_range['start'])
                    if date_range.get('end'):
                        query = query.filter(Content.publish_date <= date_range['end'])

                # Get content that needs cleanup
                needs_cleanup = query.order_by(Content.publish_date.desc().nullslast(), Content.content_id.asc()).all()

                if not needs_cleanup:
                    logger.info(f"No content found that needs cleanup tasks for project {project}")
                    return 0

                logger.info(f"Found {len(needs_cleanup)} content items that need cleanup tasks for project {project}")

                # Process in batches
                batch_size = 1000
                for i in range(0, len(needs_cleanup), batch_size):
                    batch_content = needs_cleanup[i:i + batch_size]
                    batch_data = []

                    # Prepare batch data
                    for content in batch_content:
                        # Calculate priority based on publish date
                        project_priority = self.config['active_projects'][project].get('priority', 1)
                        priority = calculate_priority_by_date(content.publish_date, project_priority) if content.publish_date else (project_priority * 1000000)

                        batch_data.append({
                            'content_id': content.content_id,
                            'priority': priority
                        })

                    # Insert tasks using bulk insert method
                    insert_query = text(f"""
                        INSERT INTO tasks.task_queue (
                            task_type, content_id, status, priority, input_data
                        )
                        SELECT
                            '{task_type}',
                            x.content_id,
                            'pending',
                            x.priority,
                            jsonb_build_object('project', :project)
                        FROM jsonb_to_recordset(:batch_data) AS x(
                            content_id text,
                            priority int
                        )
                        WHERE NOT EXISTS (
                            SELECT 1 FROM tasks.task_queue tq
                            WHERE tq.content_id = x.content_id
                            AND tq.task_type = '{task_type}'
                            AND tq.status IN ('pending', 'processing')
                        )
                    """)

                    try:
                        result = session.execute(insert_query, {
                            'project': project,
                            'batch_data': json.dumps(batch_data)
                        })
                        session.commit()
                        tasks_created_this_batch = result.rowcount
                        tasks_created += tasks_created_this_batch
                        if tasks_created_this_batch > 0:
                           self.task_summary.add_tasks(project, task_type, tasks_created_this_batch)
                           logger.debug(f"Successfully created {tasks_created_this_batch} cleanup tasks in this batch.")
                    except Exception as insert_error:
                        logger.error(f"Error inserting batch of cleanup tasks: {insert_error}")
                        session.rollback()

                if tasks_created > 0:
                    logger.info(f"\nCreated {tasks_created} {task_type} tasks for project {project}")
                else:
                    logger.info(f"No new {task_type} tasks were created for project {project} (either already exist or failed).")

                return tasks_created

        except Exception as e:
            logger.error(f"Error creating {task_type} tasks for {project}: {str(e)}", exc_info=True)
            return 0

def load_config() -> Dict:
    """Load configuration from yaml file"""
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)

def parse_date(date_str: str) -> datetime:
    """Parse date string in YYYY-MM-DD format and return UTC datetime"""
    if not date_str:
        return None
    try:
        # Parse the date and make it timezone-aware (UTC)
        parsed_date = datetime.strptime(date_str, '%Y-%m-%d')
        return parsed_date.replace(tzinfo=timezone.utc)
    except ValueError as e:
        logger.error(f"Error parsing date {date_str}: {str(e)}")
        return None

def get_channels_by_platform(project_name: str) -> Dict[str, List[Dict]]:
    """Get active channels grouped by platform for a project.

    Returns:
        Dict with platform keys containing list of channel dicts:
        {
            'youtube': [{'url': '...', 'language': 'en', 'channel_id': 123}, ...],
            'podcast': [...],
            'rumble': [...]
        }
    """
    channels_by_platform = defaultdict(list)

    logger.info(f"Loading active channels from database for project: {project_name}")

    try:
        with get_session() as session:
            # Query all active channels with their additional sources
            query = text("""
                SELECT DISTINCT
                    c.id as channel_id,
                    c.platform,
                    c.primary_url as url,
                    c.display_name,
                    c.language,
                    cs.platform as source_platform,
                    cs.source_url
                FROM channels c
                JOIN channel_projects cp ON c.id = cp.channel_id
                LEFT JOIN channel_sources cs ON c.id = cs.channel_id
                WHERE cp.project_name = :project_name
                  AND c.status = 'active'
                ORDER BY c.platform, c.display_name
            """)

            results = session.execute(query, {'project_name': project_name}).fetchall()

            if not results:
                logger.warning(f"No active channels found for project {project_name}")
                return channels_by_platform

            # Group channels by platform
            seen_channels = set()  # Track (platform, url) to avoid duplicates

            for row in results:
                # Add primary channel URL
                channel_key = (row.platform, row.url)
                if channel_key not in seen_channels:
                    channels_by_platform[row.platform].append({
                        'url': row.url,
                        'language': row.language or 'en',
                        'channel_id': row.channel_id,
                        'display_name': row.display_name
                    })
                    seen_channels.add(channel_key)

                # Add additional source URL if exists
                if row.source_platform and row.source_url:
                    source_key = (row.source_platform, row.source_url)
                    if source_key not in seen_channels:
                        channels_by_platform[row.source_platform].append({
                            'url': row.source_url,
                            'language': row.language or 'en',
                            'channel_id': row.channel_id,
                            'display_name': row.display_name
                        })
                        seen_channels.add(source_key)

            # Log summary
            for platform, channels in channels_by_platform.items():
                logger.info(f"  {platform}: {len(channels)} channels")

            return dict(channels_by_platform)

    except Exception as e:
        logger.error(f"Error loading channels from database: {e}", exc_info=True)
        return {}

async def create_download_tasks_for_content(content_ids: List[str], platform: str, project: str, config: Dict) -> int:
    """Bulk create download tasks for a list of content IDs.

    Args:
        content_ids: List of content_id strings
        platform: Platform type ('youtube', 'podcast', 'rumble')
        project: Project name
        config: Config dict for priority calculation

    Returns:
        Number of tasks created
    """
    if not content_ids:
        return 0

    try:
        with get_session() as session:
            # Determine task type based on platform
            task_type_map = {
                'youtube': 'download_youtube',
                'podcast': 'download_podcast',
                'rumble': 'download_rumble'
            }
            task_type = task_type_map.get(platform, f'download_{platform}')

            # Get project priority
            project_priority = config.get('active_projects', {}).get(project, {}).get('priority', 1)

            # Fetch content metadata for priority calculation
            content_query = text("""
                SELECT content_id, publish_date, channel_url
                FROM content
                WHERE content_id = ANY(:content_ids)
            """)
            content_data = session.execute(content_query, {'content_ids': content_ids}).fetchall()

            # Prepare batch data
            batch_data = []
            for row in content_data:
                # Calculate priority based on publish date
                priority = calculate_priority_by_date(row.publish_date, project_priority) if row.publish_date else (project_priority * 1000000)

                task_input = {'project': project}
                if platform == 'podcast' and row.channel_url:
                    task_input['feed_url'] = row.channel_url

                batch_data.append({
                    'content_id': row.content_id,
                    'task_type': task_type,
                    'input_data': json.dumps(task_input),
                    'priority': priority
                })

            if not batch_data:
                return 0

            # Bulk insert tasks
            insert_query = text("""
                INSERT INTO tasks.task_queue (
                    task_type, content_id, status, priority, input_data
                )
                SELECT
                    x.task_type,
                    x.content_id,
                    'pending',
                    x.priority,
                    x.input_data::jsonb
                FROM jsonb_to_recordset(:batch_data) AS x(
                    content_id text,
                    task_type text,
                    input_data text,
                    priority int
                )
                WHERE NOT EXISTS (
                    SELECT 1 FROM tasks.task_queue tq
                    WHERE tq.content_id = x.content_id
                    AND tq.task_type = x.task_type
                    AND tq.status IN ('pending', 'processing')
                )
            """)

            result = session.execute(insert_query, {'batch_data': json.dumps(batch_data)})
            session.commit()

            return result.rowcount

    except Exception as e:
        logger.error(f"Error creating download tasks: {e}", exc_info=True)
        return 0


async def process_platform_channels(platform: str, channels: List[Dict], project: str, config: Dict, date_range: Optional[Dict] = None) -> Dict:
    """Process all channels for a platform: index + create download tasks immediately.

    Args:
        platform: Platform name ('youtube', 'podcast', 'rumble')
        channels: List of channel dicts with 'url', 'language', 'display_name'
        project: Project name
        config: Config dict
        date_range: Optional date range filter

    Returns:
        Dict with 'indexed', 'tasks_created', 'failed' counts
    """
    results = {
        'indexed': 0,
        'tasks_created': 0,
        'failed': 0,
        'errors': []
    }

    if not channels:
        logger.info(f"[{platform.upper()}] No channels to process")
        return results

    logger.info(f"[{platform.upper()}] Processing {len(channels)} channels...")

    try:
        # Initialize appropriate indexer
        if platform == 'youtube':
            from src.ingestion.youtube_indexer import YouTubeIndexer
            indexer = YouTubeIndexer(test_mode=False)
        elif platform == 'podcast':
            from src.ingestion.podcast_indexer import PodcastIndexer
            indexer = PodcastIndexer(test_mode=False)
        elif platform == 'rumble':
            from src.ingestion.rumble_indexer import RumbleIndexer
            indexer = RumbleIndexer(test_mode=False)
        else:
            logger.error(f"[{platform.upper()}] Unknown platform type")
            return results

        # Build project_sources mapping from channels data
        from src.utils.project_utils import normalize_language_code
        project_sources = {
            'url_to_language': {}
        }
        for channel in channels:
            channel_url = channel['url']
            language = normalize_language_code(channel.get('language', 'en'))
            project_sources['url_to_language'][channel_url] = language

        # Process each channel sequentially within platform
        for idx, channel in enumerate(channels, 1):
            channel_url = channel['url']
            channel_name = channel.get('display_name', channel_url)

            try:
                logger.info(f"[{platform.upper()}] ({idx}/{len(channels)}) Indexing: {channel_name}")

                # Index the channel
                if platform == 'podcast':
                    index_result = await indexer.index_feed(channel_url, project, None, project_sources)
                else:
                    index_result = await indexer.index_channel(channel_url, project, None, project_sources)

                if index_result.get('status') == 'success':
                    results['indexed'] += 1
                    # Get the count of newly indexed content (field name varies by indexer)
                    new_content = index_result.get('indexed_count', index_result.get('video_count', 0))

                    # If new content was indexed, create download tasks immediately
                    if new_content > 0:
                        # Get the newly indexed content IDs
                        with get_session() as session:
                            query = text("""
                                SELECT content_id
                                FROM content
                                WHERE channel_url = :channel_url
                                  AND :project = ANY(projects)
                                  AND is_downloaded = false
                                  AND (:start_date IS NULL OR publish_date >= :start_date)
                                  AND (:end_date IS NULL OR publish_date <= :end_date)
                                ORDER BY publish_date DESC
                                LIMIT :limit
                            """)

                            content_ids = session.execute(query, {
                                'channel_url': channel_url,
                                'project': project,
                                'start_date': date_range.get('start') if date_range else None,
                                'end_date': date_range.get('end') if date_range else None,
                                'limit': new_content
                            }).fetchall()

                            content_id_list = [row.content_id for row in content_ids]

                        # Create download tasks for new content
                        if content_id_list:
                            tasks_created = await create_download_tasks_for_content(
                                content_id_list, platform, project, config
                            )
                            results['tasks_created'] += tasks_created
                            logger.info(f"[{platform.upper()}] ✓ {channel_name}: {new_content} new, {tasks_created} tasks created")
                    else:
                        logger.info(f"[{platform.upper()}] ✓ {channel_name}: No new content")
                else:
                    results['failed'] += 1
                    error_msg = index_result.get('error', 'Unknown error')
                    logger.warning(f"[{platform.upper()}] ✗ {channel_name}: {error_msg}")
                    results['errors'].append(f"{channel_name}: {error_msg}")

            except Exception as e:
                results['failed'] += 1
                error_msg = str(e)
                logger.error(f"[{platform.upper()}] ✗ {channel_name}: {error_msg}")
                results['errors'].append(f"{channel_name}: {error_msg}")

        # Summary
        logger.info(f"[{platform.upper()}] Complete: {results['indexed']} indexed, {results['tasks_created']} tasks created, {results['failed']} failed")

        return results

    except Exception as e:
        logger.error(f"[{platform.upper()}] Fatal error: {e}", exc_info=True)
        return results


async def create_task_with_retry(session, task_data: Dict, priority: int = 0, max_retries: int = 3) -> bool:
    """Create a task with retry logic"""
    for attempt in range(max_retries):
        try:
            # Convert type to task_type
            task_type = task_data.pop('type', None)
            task_data['task_type'] = task_type
            
            # Get project priority from config
            config = load_config()
            project = task_data.get('input_data', {}).get('project')
            project_priority = 1  # Default priority
            if project and project in config.get('active_projects', {}):
                project_priority = config['active_projects'][project].get('priority', 1)
            
            # Calculate priority using both date and project priority
            if priority == 0:  # Only calculate if not explicitly provided
                content = session.query(Content).filter_by(content_id=task_data['content_id']).first()
                if content and content.publish_date:
                    priority = calculate_priority_by_date(content.publish_date, project_priority)
                else:
                    priority = project_priority * 1000000  # Use project priority only if no date
            
            task_data['priority'] = priority
            
            # Check for existing task
            existing_task = session.query(TaskQueue).filter(
                TaskQueue.content_id == task_data['content_id'],
                TaskQueue.task_type == task_data['task_type'],
                TaskQueue.status.in_(['pending', 'processing'])
            ).first()
            
            if existing_task:
                # Task already exists and is pending/processing
                return False
                
            # Create new task
            task = TaskQueue(**task_data)
            session.add(task)
            session.commit()
            return True
            
        except OperationalError as e:
            if attempt < max_retries - 1:
                try:
                    logger.warning(f"Database error creating task (attempt {attempt + 1}/{max_retries}): {str(e)}")
                except Exception as log_error:
                    print(f"Logging error: {log_error}")
                await asyncio.sleep(2)
                continue
            else:
                try:
                    logger.error(f"Failed to create task after {max_retries} attempts: {str(e)}")
                except Exception as log_error:
                    print(f"Logging error: {log_error}")
                return False
        except Exception as e:
            try:
                logger.error(f"Unexpected error creating task: {str(e)}")
            except Exception as log_error:
                print(f"Logging error: {log_error}")
            return False

async def cleanup_task_queue(project: str, task_type: Optional[str] = None) -> None:
    """Clean up task queue by deleting tasks for the specified project and optionally task_type."""
    try:
        with get_session() as session:
            # Build base WHERE clause
            where_clause = text("""
                WHERE
                    -- For download tasks, check project in input_data
                    (tq.task_type IN ('download_youtube', 'download_podcast', 'download_rumble')
                     AND tq.input_data->>'project' = :project)
                    OR
                    -- For other tasks, join with content to check project
                    (tq.task_type NOT IN ('download_youtube', 'download_podcast', 'download_rumble')
                     AND EXISTS (
                         SELECT 1 FROM content c
                         WHERE c.content_id = tq.content_id
                         AND :project_like = ANY(c.projects)
                     ))
            """)
            params = {'project': project, 'project_like': f'%{project}%'}

            # Add task_type filter if provided
            if task_type:
                where_clause = text(f"({where_clause.text}) AND tq.task_type = :task_type")
                params['task_type'] = task_type

            # Count tasks before cleanup
            count_query_str = f"SELECT COUNT(*) FROM tasks.task_queue tq {where_clause.text}"
            count_query = text(count_query_str)
            tasks_to_delete = session.execute(count_query, params).scalar()

            if tasks_to_delete == 0:
                logger.info(f"No tasks found to clean for project '{project}'" + (f" and type '{task_type}'" if task_type else ""))
                return

            logger.info(f"Cleaning up {tasks_to_delete} tasks for project '{project}'" + (f" of type '{task_type}'" if task_type else ""))

            # Delete tasks
            delete_query_str = f"DELETE FROM tasks.task_queue tq {where_clause.text}"
            delete_query = text(delete_query_str)
            result = session.execute(delete_query, params)
            session.commit()

            logger.info(f"Deleted {result.rowcount} tasks.")

    except Exception as e:
        logger.error(f"Error cleaning task queue for project {project}" + (f", type {task_type}" if task_type else "") + f": {str(e)}")
        session.rollback() # Rollback on error
        raise

async def verify_nas_access(config: Dict) -> bool:
    """Verify NAS access by checking mount point"""
    try:
        nas_path = Path(config['storage']['nas']['mount_point'])
        return nas_path.exists() and nas_path.is_mount()
    except Exception as e:
        logger.error(f"Error checking NAS access: {str(e)}")
        return False

async def display_task_queue_summary(project: str) -> None:
    """Display a summary of tasks in the queue for a project with emojis."""
    try:
        with get_session() as session:
            # Get project date range from config
            config = load_config()
            project_config = config.get('active_projects', {}).get(project, {})
            start_date = project_config.get('start_date', 'ongoing')
            end_date = project_config.get('end_date', 'ongoing')
            
            # Query task counts by type
            query = text("""
                SELECT 
                    task_type,
                    COUNT(*) as task_count
                FROM tasks.task_queue
                WHERE status = 'pending'
                AND (
                    -- For download tasks, check input_data->>'project'
                    (task_type IN ('download_youtube', 'download_podcast', 'download_rumble') 
                     AND input_data->>'project' = :project)
                    OR
                    -- For other tasks, join with content to check project
                    (task_type NOT IN ('download_youtube', 'download_podcast', 'download_rumble')
                     AND EXISTS (
                         SELECT 1 FROM content c
                         WHERE c.content_id = tasks.task_queue.content_id
                         AND :project_like = ANY(c.projects)
                     ))
                )
                GROUP BY task_type
                ORDER BY task_type
            """)
            
            results = session.execute(query, {
                'project': project,
                'project_like': f'%{project}%'
            })
            
            # Define emoji mappings for task types
            emoji_map = {
                'download_youtube': '📺',
                'download_podcast': '🎙️',
                'download_rumble': '🎥',
                'convert': '🔊',
                'transcribe': '📝',
                'stitch': '🔄'
            }
            
            # Collect task counts
            task_counts = {row.task_type: row.task_count for row in results}
            
            # Print summary header
            logger.info(f"\n📊 Task Queue Summary for {project} ({start_date} → {end_date})")
            
            # Print counts with emojis, using thousands separator
            if task_counts.get('download_youtube', 0) > 0:
                logger.info(f"  {emoji_map['download_youtube']} YouTube Downloads:    {task_counts['download_youtube']:,}")
            if task_counts.get('download_podcast', 0) > 0:
                logger.info(f"  {emoji_map['download_podcast']} Podcast Downloads:    {task_counts['download_podcast']:,}")
            if task_counts.get('download_rumble', 0) > 0:
                logger.info(f"  {emoji_map['download_rumble']} Rumble Downloads:     {task_counts['download_rumble']:,}")
            if task_counts.get('convert', 0) > 0:
                logger.info(f"  {emoji_map['convert']} Audio Conversion:     {task_counts['convert']:,}")
            if task_counts.get('transcribe', 0) > 0:
                logger.info(f"  {emoji_map['transcribe']} Transcription:       {task_counts['transcribe']:,}")
            if task_counts.get('stitch', 0) > 0:
                logger.info(f"  {emoji_map['stitch']} Stitching:           {task_counts['stitch']:,}")
            
            # Print total
            total = sum(task_counts.values())
            logger.info(f"  📈 Total Tasks:          {total:,}\n")
            
    except Exception as e:
        logger.error(f"Error displaying task queue summary: {str(e)}")

def get_active_projects(config: Dict) -> List[Dict]:
    """Get list of active projects and their date ranges from config"""
    active_projects = []
    for project, settings in config.get('active_projects', {}).items():
        if settings.get('enabled', False):
            active_projects.append({
                'name': project,
                'start_date': parse_date(settings.get('start_date')),
                'end_date': parse_date(settings.get('end_date'))
            })
    return active_projects

async def remove_duplicate_tasks():
    """Remove any duplicate tasks from the task queue, keeping only the highest priority task for each unique task combination."""
    try:
        with get_session() as session:
            # Find and remove duplicates using a CTE, handling chunk_index for transcription tasks
            cleanup_query = text("""
                WITH duplicates AS (
                    SELECT id
                    FROM (
                        SELECT 
                            id,
                            content_id,
                            task_type,
                            status,
                            priority,
                            -- For transcribe tasks, include chunk_index in partition
                            CASE 
                                WHEN task_type = 'transcribe' THEN 
                                    CAST(input_data->>'chunk_index' AS INTEGER)
                                ELSE 0
                            END as chunk_index,
                            ROW_NUMBER() OVER (
                                PARTITION BY 
                                    content_id, 
                                    task_type,
                                    status,
                                    -- Include chunk_index in partition for transcribe tasks
                                    CASE 
                                        WHEN task_type = 'transcribe' THEN 
                                            CAST(input_data->>'chunk_index' AS INTEGER)
                                        ELSE 0
                                    END
                                ORDER BY priority DESC, created_at ASC
                            ) as rn
                        FROM tasks.task_queue
                        WHERE status IN ('pending', 'processing')
                    ) t
                    WHERE rn > 1
                )
                DELETE FROM tasks.task_queue
                WHERE id IN (SELECT id FROM duplicates)
                RETURNING content_id, task_type, 
                    CASE 
                        WHEN task_type = 'transcribe' THEN 
                            CAST(input_data->>'chunk_index' AS INTEGER)::text
                        ELSE NULL
                    END as chunk_index;
            """)
            
            result = session.execute(cleanup_query)
            session.commit()
            
            # Count removed duplicates by task type
            removed = defaultdict(int)
            chunk_details = defaultdict(list)
            for row in result:
                removed[row.task_type] += 1
                if row.task_type == 'transcribe' and row.chunk_index:
                    chunk_details[row.content_id].append(row.chunk_index)
            
            if sum(removed.values()) > 0:
                logger.info("\n🧹 Removed duplicate tasks:")
                for task_type, count in removed.items():
                    if task_type == 'transcribe':
                        logger.info(f"  {task_type}: {count:,} (across {len(chunk_details)} content items)")
                        # Log detailed chunk information if there were transcribe duplicates
                        if chunk_details:
                            logger.debug("Duplicate transcription chunks removed:")
                            for content_id, chunks in chunk_details.items():
                                logger.debug(f"  Content {content_id}: chunks {sorted(chunks)}")
                    else:
                        logger.info(f"  {task_type}: {count:,}")
            
    except Exception as e:
        logger.error(f"Error removing duplicate tasks: {str(e)}")

async def unblock_content_by_platform(project: str, platform: str) -> int:
    """Unblock all blocked content for a specific platform.
    
    Args:
        project: Project name to filter content
        platform: Platform type ('podcast' or 'youtube')
        
    Returns:
        Number of records unblocked
    """
    try:
        with get_session() as session:
            # Get count of blocked content before update
            blocked_count_query = text("""
                SELECT COUNT(*) as count
                FROM content
                WHERE platform = :platform
                AND blocked_download = true
                AND :project = ANY(projects)
            """)
            
            blocked_count_result = session.execute(blocked_count_query, {
                'platform': platform,
                'project': project
            })
            blocked_count = blocked_count_result.scalar()
            
            logger.info(f"Found {blocked_count} blocked {platform} downloads for project {project}")
            
            if blocked_count == 0:
                logger.info(f"No blocked {platform} content found. Nothing to do.")
                return 0
            
            # Update blocked_download status - fixed JSON syntax with jsonb_build_object
            update_query = text("""
                UPDATE content
                SET
                    blocked_download = false,
                    meta_data = CASE
                        WHEN meta_data IS NULL THEN jsonb_build_object('unblocked_at', now()::text, 'unblocked_by', 'create_tasks.py')
                        ELSE meta_data || jsonb_build_object('unblocked_at', now()::text, 'unblocked_by', 'create_tasks.py')
                    END
                WHERE platform = :platform
                AND blocked_download = true
                AND :project = ANY(projects)
                RETURNING content_id
            """)
            
            result = session.execute(update_query, {
                'platform': platform,
                'project': project
            })
            unblocked_ids = result.fetchall()
            
            # Commit the transaction
            session.commit()
            
            logger.info(f"Successfully unblocked {len(unblocked_ids)} {platform} downloads.")
            
            # Log some of the unblocked content IDs for verification
            if unblocked_ids:
                sample_size = min(10, len(unblocked_ids))
                sample_ids = [row[0] for row in unblocked_ids[:sample_size]]
                logger.info(f"Sample of unblocked content IDs: {', '.join(sample_ids)}")
                
                if len(unblocked_ids) > sample_size:
                    logger.info(f"... and {len(unblocked_ids) - sample_size} more")
            
            return len(unblocked_ids)
            
    except Exception as e:
        logger.error(f"Error unblocking {platform} content: {str(e)}")
        return 0

async def reset_failed_tasks(project: str) -> None:
    """Reset failed tasks for a project by creating new tasks for failed ones."""
    try:
        with get_session() as session:
            # Count failed tasks before reset
            count_query = text("""
                SELECT task_type, COUNT(*) 
                FROM tasks.task_queue 
                WHERE status = 'error'
                AND (
                    -- For download tasks, check project in input_data
                    (task_type IN ('download_youtube', 'download_podcast', 'download_rumble') 
                     AND input_data->>'project' = :project)
                    OR
                    -- For other tasks, join with content to check project
                    (task_type NOT IN ('download_youtube', 'download_podcast', 'download_rumble')
                     AND EXISTS (
                         SELECT 1 FROM content c 
                         WHERE c.content_id = tasks.task_queue.content_id 
                         AND :project_like = ANY(c.projects)
                     ))
                )
                GROUP BY task_type
            """)
            results = session.execute(count_query, {
                'project': project,
                'project_like': f'%{project}%'
            })
            
            logger.info("Failed tasks for project %s:", project)
            for row in results:
                logger.info(f"  {row.task_type}: {row.count}")

            # Get failed tasks and create new ones
            failed_tasks_query = text("""
                SELECT 
                    task_type,
                    content_id,
                    input_data,
                    priority
                FROM tasks.task_queue
                WHERE status = 'error'
                AND (
                    -- For download tasks, check project in input_data
                    (task_type IN ('download_youtube', 'download_podcast', 'download_rumble') 
                     AND input_data->>'project' = :project)
                    OR
                    -- For other tasks, join with content to check project
                    (task_type NOT IN ('download_youtube', 'download_podcast', 'download_rumble')
                     AND EXISTS (
                         SELECT 1 FROM content c 
                         WHERE c.content_id = tasks.task_queue.content_id 
                         AND :project_like = ANY(c.projects)
                     ))
                )
            """)
            
            failed_tasks = session.execute(failed_tasks_query, {
                'project': project,
                'project_like': f'%{project}%'
            }).fetchall()
            
            if not failed_tasks:
                logger.info("No failed tasks found to reset")
                return
                
            # Create new tasks for failed ones
            tasks_created = 0
            for task in failed_tasks:
                # Create new task with same data but reset status
                new_task = TaskQueue(
                    task_type=task.task_type,
                    content_id=task.content_id,
                    status='pending',
                    priority=task.priority,
                    input_data=task.input_data,
                    created_at=datetime.now(timezone.utc)
                )
                session.add(new_task)
                tasks_created += 1
                
            session.commit()
            logger.info(f"Created {tasks_created} new tasks for failed ones")
            
    except Exception as e:
        logger.error(f"Error resetting failed tasks: {str(e)}")
        raise

async def main():
    """Main entry point for task creation"""
    parser = argparse.ArgumentParser(description='Create tasks for content processing')
    parser.add_argument('--project', help='Project name to create tasks for. If not specified, creates tasks for all active projects.')
    parser.add_argument('--clean', action='store_true', help='Clean existing tasks before creating new ones')
    parser.add_argument('--reset', action='store_true', help='Reset failed tasks by creating new ones')
    parser.add_argument('--complete', action='store_true', help='Run all steps in sequence')
    parser.add_argument('--steps', nargs='+', default=['index', 'download', 'convert', 'transcribe', 'diarize', 'stitch', 'segment', 'cleanup'],
                      help='Steps to run. Available steps: index, index_youtube, index_podcast, index_rumble, download, download_youtube, download_podcast, download_rumble, convert, transcribe, diarize, stitch, segment, cleanup (default: all steps).')
    parser.add_argument('--start-date', type=str, help='Start date for content filtering (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date for content filtering (YYYY-MM-DD)')
    parser.add_argument('--unblock', action='store_true', help='Unblock content before creating tasks. Works with download_podcast and download_youtube steps.')
    parser.add_argument('--content-id', help='Debug task creation for a single content item')
    args = parser.parse_args()

    # Load config
    config = load_config()
    
    # --- Truncate Task Queue if --clean specified ---
    if args.clean:
        logger.warning("!!! TRUNCATING ENTIRE tasks.task_queue TABLE (--clean specified) !!!")
        try:
            with get_session() as session:
                # Using TRUNCATE for speed, RESTART IDENTITY resets any sequences if needed
                session.execute(text("TRUNCATE TABLE tasks.task_queue RESTART IDENTITY"))
                session.commit()
                logger.info("Successfully truncated tasks.task_queue.")
        except Exception as e:
            logger.error(f"Failed to truncate task queue: {e}")
            return # Stop processing if truncation fails
        # No need to call project-specific cleanup later if we truncated here

    # Get date range if specified
    date_range = None
    if args.start_date or args.end_date:
        date_range = {
            'start': parse_date(args.start_date) if args.start_date else None,
            'end': parse_date(args.end_date) if args.end_date else None
        }

    # Create task creator instance and summary tracker
    creator = TaskCreator(config)
    task_summary = TaskSummary()

    # If content-id is specified, debug task creation for that content
    if args.content_id:
        with get_session() as session:
            # Get content details
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
                    c.blocked_download,
                    c.meta_data,
                    COUNT(cc.id) as total_chunks,
                    COUNT(cc.id) FILTER (WHERE cc.extraction_status = 'completed') as extracted_chunks,
                    COUNT(cc.id) FILTER (WHERE cc.transcription_status = 'completed') as transcribed_chunks
                FROM content c
                LEFT JOIN content_chunks cc ON c.id = cc.content_id
                WHERE c.content_id = :content_id
                GROUP BY c.id, c.content_id, c.projects, c.platform, c.is_downloaded,
                         c.is_converted, c.is_transcribed, c.is_diarized, c.is_stitched, 
                         c.publish_date, c.blocked_download, c.meta_data
            """)
            result = session.execute(query, {'content_id': args.content_id}).fetchone()
            
            if not result:
                logger.error(f"Content {args.content_id} not found")
                return
                
            logger.info(f"\n🔍 Debugging task creation for content: {args.content_id}")
            logger.info(f"Project: {result.projects}")
            logger.info(f"Platform: {result.platform}")
            logger.info(f"Publish date: {result.publish_date}")
            logger.info(f"Database flags:")
            logger.info(f"  is_downloaded: {result.is_downloaded}")
            logger.info(f"  is_converted: {result.is_converted}")
            logger.info(f"  is_transcribed: {result.is_transcribed}")
            logger.info(f"  is_diarized: {result.is_diarized}")
            logger.info(f"  is_stitched: {result.is_stitched}")
            logger.info(f"Chunk stats:")
            logger.info(f"  Total chunks: {result.total_chunks}")
            logger.info(f"  Extracted chunks: {result.extracted_chunks}")
            logger.info(f"  Transcribed chunks: {result.transcribed_chunks}")
            
            # Check existing tasks
            task_query = text("""
                SELECT task_type, status, priority, created_at
                FROM tasks.task_queue
                WHERE content_id = :content_id
                ORDER BY created_at DESC
            """)
            tasks = session.execute(task_query, {'content_id': args.content_id}).fetchall()
            
            if tasks:
                logger.info("\nExisting tasks:")
                for task in tasks:
                    logger.info(f"  {task.task_type}: {task.status} (priority: {task.priority}, created: {task.created_at})")
            else:
                logger.info("\nNo existing tasks found")
            
            # Check audio conversion
            if result.is_downloaded and not result.is_converted:
                logger.info("  Would create convert task")
            
            # Check transcription - only if not all chunks are transcribed
            if result.is_converted and not result.is_transcribed and result.total_chunks > result.transcribed_chunks:
                logger.info("  Would create transcribe task")
            
            # Check diarization
            if result.is_converted and not result.is_diarized:
                logger.info("  Would create diarize task")
            
            # Check stitching - only if all chunks are transcribed and there's no pending stitch task
            if (result.is_converted and 
                not result.is_stitched and 
                result.total_chunks == result.transcribed_chunks and
                not any(task.task_type == 'stitch' and task.status in ['pending', 'processing'] for task in tasks)):
                logger.info("  Would create stitch task")
            
            return

    # Determine which projects to process
    projects_to_process = []
    if args.project:
        # Single project mode
        project_config = config.get('active_projects', {}).get(args.project, {})
        projects_to_process.append({
            'name': args.project,
            'start_date': date_range['start'] if date_range else parse_date(project_config.get('start_date')),
            'end_date': date_range['end'] if date_range else parse_date(project_config.get('end_date'))
        })
    else:
        # All active projects mode
        active_projects = get_active_projects(config)
        if not active_projects:
            logger.error("No active projects found in config")
            return

        for project in active_projects:
            projects_to_process.append({
                'name': project['name'],
                'start_date': project['start_date'] if not date_range else date_range['start'],
                'end_date': project['end_date'] if not date_range else date_range['end']
            })

    if not projects_to_process:
        logger.error("No projects to process")
        return

    # Process each project
    for project_info in projects_to_process:
        project_name = project_info['name']
        project_date_range = {
            'start': project_info['start_date'],
            'end': project_info['end_date']
        }

        date_str = f"({project_date_range['start'].strftime('%Y-%m-%d') if project_date_range['start'] else 'start'} to {project_date_range['end'].strftime('%Y-%m-%d') if project_date_range['end'] else 'now'})"
        logger.info(f"\n📂 Processing project: {project_name} {date_str}")

        # Reset failed tasks if requested
        if args.reset:
            await reset_failed_tasks(project_name)
            if not args.steps and not args.complete:
                continue

        # Determine steps to run
        steps_to_run = args.steps
        if args.complete:
            steps_to_run = ['index', 'download', 'convert', 'transcribe', 'diarize', 'stitch', 'segment', 'cleanup']

        # Handle index/download steps with new parallel architecture
        needs_indexing = any(s in steps_to_run for s in ['index', 'download', 'index_youtube', 'index_podcast', 'index_rumble',
                                                           'download_youtube', 'download_podcast', 'download_rumble'])

        if needs_indexing:
            logger.info(f"\n🔍 Loading channels for {project_name}...")
            channels_by_platform = get_channels_by_platform(project_name)

            if not channels_by_platform:
                logger.warning(f"No active channels found for project {project_name}")
            else:
                # Determine which platforms to process based on steps
                platforms_to_process = []
                if 'index' in steps_to_run or 'download' in steps_to_run:
                    # Process all platforms
                    platforms_to_process = ['youtube', 'podcast', 'rumble']
                else:
                    # Process only specific platforms requested
                    if 'index_youtube' in steps_to_run or 'download_youtube' in steps_to_run:
                        platforms_to_process.append('youtube')
                    if 'index_podcast' in steps_to_run or 'download_podcast' in steps_to_run:
                        platforms_to_process.append('podcast')
                    if 'index_rumble' in steps_to_run or 'download_rumble' in steps_to_run:
                        platforms_to_process.append('rumble')

                # Process platforms in parallel
                logger.info(f"🚀 Processing {len(platforms_to_process)} platforms in parallel...")

                platform_tasks = []
                for platform in platforms_to_process:
                    if platform in channels_by_platform and channels_by_platform[platform]:
                        platform_tasks.append(
                            process_platform_channels(
                                platform,
                                channels_by_platform[platform],
                                project_name,
                                config,
                                project_date_range
                            )
                        )

                # Run all platform processing in parallel
                if platform_tasks:
                    platform_results = await asyncio.gather(*platform_tasks)

                    # Aggregate results and update summary
                    for platform, result in zip(platforms_to_process, platform_results):
                        if result['tasks_created'] > 0:
                            task_type = f'download_{platform}'
                            task_summary.add_tasks(project_name, task_type, result['tasks_created'])

                logger.info(f"✅ Parallel indexing + task creation complete for {project_name}")

        # Process remaining pipeline steps (convert, transcribe, etc.)
        for step in steps_to_run:
            # Skip index-only steps - already handled above
            if step in ['index', 'index_youtube', 'index_podcast', 'index_rumble']:
                continue

            # Handle download steps - create tasks for existing content that needs downloading
            if step == 'download':
                # Create download tasks for all platforms
                for download_type in ['download_youtube', 'download_podcast', 'download_rumble']:
                    tasks = await creator.create_download_tasks(project_name, False, project_date_range, download_type)
                    if tasks > 0:
                        task_summary.add_tasks(project_name, download_type, tasks)
                continue

            if step == 'download_youtube':
                tasks = await creator.create_download_tasks(project_name, False, project_date_range, 'download_youtube')
                if tasks > 0:
                    task_summary.add_tasks(project_name, 'download_youtube', tasks)
                continue

            if step == 'download_podcast':
                tasks = await creator.create_download_tasks(project_name, False, project_date_range, 'download_podcast')
                if tasks > 0:
                    task_summary.add_tasks(project_name, 'download_podcast', tasks)
                continue

            if step == 'download_rumble':
                tasks = await creator.create_download_tasks(project_name, False, project_date_range, 'download_rumble')
                if tasks > 0:
                    task_summary.add_tasks(project_name, 'download_rumble', tasks)
                continue

            if step == 'convert':
                tasks = await creator.create_convert_tasks(project_name, False, project_date_range)
                task_summary.add_tasks(project_name, 'convert', tasks)

            elif step == 'diarize':
                tasks = await creator.create_diarization_tasks(project_name, False, project_date_range)
                task_summary.add_tasks(project_name, 'diarize', tasks)

            elif step == 'identify_speakers':
                logger.info(f"⚠️ identify_speakers step is deprecated - now handled by diarize step")
                continue

            elif step == 'transcribe':
                tasks = await creator.create_transcription_tasks(project_name, False, project_date_range)
                task_summary.add_tasks(project_name, 'transcribe', tasks)

            elif step == 'stitch':
                tasks = await creator.create_stitch_tasks(project_name, False, project_date_range)
                task_summary.add_tasks(project_name, 'stitch', tasks)

            elif step == 'segment':
                tasks = await creator.create_segment_tasks(project_name, False, project_date_range)
                task_summary.add_tasks(project_name, 'segment', tasks)

            elif step == 'cleanup':
                tasks = await creator.create_cleanup_tasks(project_name, False, project_date_range)
                task_summary.add_tasks(project_name, 'cleanup', tasks)

            else:
                logger.warning(f"⚠️ Unknown step: {step}")
                continue

    # Print final summary
    task_summary.print_summary()
    
    # Final cleanup of any duplicate tasks
    await remove_duplicate_tasks()

if __name__ == '__main__':
    asyncio.run(main()) 