import warnings

warnings.warn(
    "DEPRECATED: This module is deprecated and will be removed in a future version. "
    "Use StateManager (src/database/state_manager.py) for state management and "
    "PipelineManager (src/processing/pipeline_manager.py) for task management instead.",
    DeprecationWarning,
    stacklevel=2
)

from typing import List, Dict, Any, Callable, Optional
import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time
import json
import socket
import yaml
import psycopg2
from psycopg2.extras import Json

from ..database.session import Session, get_connection, get_session
from ..database.manager import DatabaseManager
from ..ingestion.youtube_indexer import YouTubeIndexer
from ..ingestion.podcast_indexer import PodcastIndexer
from ..distributed.task_queue import TaskQueueManager
from ..utils.logger import setup_indexer_logger, load_config
from ..utils.status import get_project_status, print_status_report

# Set up component logger
logger = setup_indexer_logger('content')

# Load configuration
config = load_config()
base_path = Path(config['storage']['local']['base_path'])

def get_db_connection_from_config(config):
    """Get database connection using config"""
    # Use the global get_connection function without arguments
    return get_connection()

async def process_content_pipeline(
    project_name: str,
    steps: List[str],
    sources: Dict[str, List[str]],
    config: Dict,
    date_range: Optional[Dict] = None,
    progress_callback: Optional[Callable] = None
) -> Dict:
    """Process content through the pipeline"""
    results = {
        'status': 'success',
        'steps': {},
        'errors': []
    }
    
    try:
        # Initialize task queue manager
        task_queue = TaskQueueManager(config['database'])
        
        # Run indexing if requested (this still runs locally as it's a head node task)
        if 'index' in steps:
            logger.info("Starting indexing step...")
            index_results = await _run_indexing_step(
                project_name=project_name,
                sources=sources,
                progress_callback=progress_callback
            )
            results['steps']['index'] = index_results
        
        # Create tasks for downloads if requested
        if 'download' in steps:
            logger.info("Creating download tasks...")
            with get_session() as session:
                db = DatabaseManager(session)
                # Get YouTube content
                youtube_content = db.get_content_to_download(
                    project_name,
                    platform='youtube',
                    date_range=date_range
                )
                # Get podcast content
                podcast_content = db.get_content_to_download(
                    project_name,
                    platform='podcast',
                    date_range=date_range
                )
                
                logger.info(f"Found {len(youtube_content)} YouTube videos and {len(podcast_content)} podcasts to download")
                
                # Create all download tasks at once
                download_tasks = []
                
                # Add YouTube tasks with higher priority
                download_tasks.extend([
                    {
                        'task_type': 'download_youtube',
                        'content_id': content.content_id,
                        'input_data': {
                            'project': project_name,
                            'batch_index': idx
                        },
                        'priority': 2  # Higher priority for YouTube
                    }
                    for idx, content in enumerate(youtube_content)
                ])
                
                # Add podcast tasks with normal priority
                download_tasks.extend([
                    {
                        'task_type': 'download_podcast',
                        'content_id': content.content_id,
                        'input_data': {
                            'project': project_name,
                            'batch_index': idx + len(youtube_content)  # Continue indexing from YouTube
                        },
                        'priority': 1  # Normal priority for podcasts
                    }
                    for idx, content in enumerate(podcast_content)
                ])
                
                if download_tasks:
                    logger.info(f"Adding {len(download_tasks)} download tasks in single transaction")
                    await task_queue.add_tasks(download_tasks)
                    
                    if progress_callback:
                        progress_callback({
                            'step': 'download',
                            'message': f"Added {len(download_tasks)} download tasks"
                        })
                else:
                    logger.info("No download tasks to add")
        
        # Create tasks for transcription if requested
        if 'transcribe' in steps:
            logger.info("Creating transcription tasks...")
            with get_session() as session:
                db = DatabaseManager(session)
                content_to_transcribe = db.get_content_to_transcribe(
                    project_name,
                    date_range=date_range
                )
                
                if content_to_transcribe:
                    # Sort content by duration (shortest first)
                    sorted_content = sorted(content_to_transcribe, key=lambda x: x.duration or float('inf'))
                    logger.info(f"Found {len(sorted_content)} items to transcribe")
                    logger.info(f"Duration range: {sorted_content[0].duration/60:.1f}min to {sorted_content[-1].duration/60:.1f}min")
                    
                    # Calculate priority based on duration
                    # Shorter duration = higher priority (max 100)
                    max_duration = max(c.duration or 0 for c in sorted_content)
                    
                    # Create all transcription tasks in single transaction
                    transcribe_tasks = [
                        {
                            'task_type': 'transcribe',
                            'content_id': content.content_id,
                            'input_data': {
                                'project': project_name,
                                'batch_index': idx,
                                'duration': content.duration
                            },
                            # Priority 100 (highest) for shortest, down to 1 for longest
                            'priority': max(1, int(100 * (1 - (content.duration or max_duration) / max_duration)))
                        }
                        for idx, content in enumerate(sorted_content)
                    ]
                    
                    # Log some example priorities
                    examples = [
                        (t['content_id'], t['input_data']['duration']/60, t['priority']) 
                        for t in transcribe_tasks[:3] + transcribe_tasks[-3:]
                    ]
                    logger.info("Priority examples (content_id, duration_mins, priority):")
                    for ex in examples[:3]:
                        logger.info(f"  Shortest: {ex[0]}, {ex[1]:.1f}min -> priority {ex[2]}")
                    for ex in examples[3:]:
                        logger.info(f"  Longest: {ex[0]}, {ex[1]:.1f}min -> priority {ex[2]}")
                    
                    logger.info(f"Adding {len(transcribe_tasks)} transcription tasks in single transaction")
                    await task_queue.add_tasks(transcribe_tasks)
                    
                    if progress_callback:
                        progress_callback({
                            'step': 'transcribe',
                            'message': f"Added {len(transcribe_tasks)} transcription tasks (prioritized by duration)"
                        })
                else:
                    logger.info("No content to transcribe")
                    results['steps']['transcribe'] = {
                        'successful': [],
                        'failed': [],
                        'skipped': []
                    }
        
        # Monitor task progress using direct psycopg2 connection
        while True:
            # Get task counts using psycopg2
            conn = get_db_connection_from_config(config)
            try:
                with conn.cursor() as cur:
                    # Get task counts by type and status
                    cur.execute("""
                        SELECT 
                            task_type, 
                            status, 
                            COUNT(*),
                            COUNT(CASE WHEN error IS NOT NULL THEN 1 END) as error_count
                        FROM tasks.task_queue
                        WHERE task_type = ANY(%s)
                        GROUP BY task_type, status
                    """, ([f"download_{platform}" for platform in ['youtube', 'podcast']] + ['transcribe'],))
                    
                    task_counts = {}
                    for task_type, status, count, error_count in cur.fetchall():
                        if task_type not in task_counts:
                            task_counts[task_type] = {
                                'pending': 0, 
                                'in_progress': 0, 
                                'done': 0, 
                                'error': 0,
                                'error_details': 0
                            }
                        task_counts[task_type][status] = count
                        if status == 'error':
                            task_counts[task_type]['error_details'] = error_count
                    
                    # Get worker status
                    cur.execute("""
                        SELECT 
                            hostname,
                            enabled_tasks,
                            status,
                            last_heartbeat
                        FROM tasks.worker_config
                        WHERE status = 'active'
                    """)
                    
                    active_workers = {}
                    for hostname, enabled_tasks, status, last_heartbeat in cur.fetchall():
                        active_workers[hostname] = {
                            'enabled_tasks': enabled_tasks,
                            'status': status,
                            'last_heartbeat': last_heartbeat
                        }
            finally:
                conn.close()
            
            # Check if all tasks are complete
            all_complete = True
            for task_type, counts in task_counts.items():
                if counts['pending'] > 0 or counts['in_progress'] > 0:
                    all_complete = False
                    break
            
            if all_complete:
                break
            
            # Update progress with detailed status
            if progress_callback:
                status_msg = "\n=== Task Processing Status ===\n"
                
                # Task status
                for task_type, counts in task_counts.items():
                    total = sum(counts.values())
                    done = counts['done']
                    in_progress = counts['in_progress']
                    pending = counts['pending']
                    errors = counts['error']
                    error_details = counts['error_details']
                    
                    status_msg += f"\n{task_type}:\n"
                    status_msg += f"  Total: {total} | Complete: {done} | In Progress: {in_progress} | Pending: {pending}\n"
                    if errors > 0:
                        status_msg += f"  Errors: {errors} (with details: {error_details})\n"
                
                # Worker status
                status_msg += "\nActive Workers:\n"
                for hostname, worker in active_workers.items():
                    tasks = ", ".join(worker['enabled_tasks'])
                    last_seen = (datetime.now(timezone.utc) - worker['last_heartbeat']).total_seconds()
                    status_msg += f"  {hostname}: Tasks: [{tasks}] | Last seen: {last_seen:.1f}s ago\n"
                
                progress_callback(status_msg)
            
            # Wait before checking again
            await asyncio.sleep(10)
        
        # Collect results using psycopg2
        conn = get_db_connection_from_config(config)
        try:
            with conn.cursor() as cur:
                for step in steps:
                    if step == 'index':
                        continue  # Index results already collected
                        
                    task_types = []
                    if step == 'download':
                        task_types = [f"download_{platform}" for platform in ['youtube', 'podcast']]
                    elif step == 'transcribe':
                        task_types = ['transcribe']
                    
                    cur.execute("""
                        SELECT content_id, status, result, error
                        FROM tasks.task_queue
                        WHERE task_type = ANY(%s)
                    """, (task_types,))
                    
                    step_results = {
                        'successful': [],
                        'failed': [],
                        'skipped': []
                    }
                    
                    for content_id, status, result, error in cur.fetchall():
                        if status == 'done':
                            step_results['successful'].append(content_id)
                        elif status == 'error':
                            step_results['failed'].append({
                                'content_id': content_id,
                                'error': error.get('message') if error else 'Unknown error'
                            })
                    
                    results['steps'][step] = step_results
        finally:
            conn.close()
        
        return results
        
    except Exception as e:
        logger.error(f"Error in content pipeline: {str(e)}")
        results['status'] = 'error'
        results['error'] = str(e)
        return results

async def _run_indexing_step(
    project_name: str,
    sources: Dict[str, List[str]],
    progress_callback: Optional[Callable] = None
) -> Dict:
    """Run the indexing step for all content sources"""
    results = {
        'successful': [],
        'failed': [],
        'skipped': []
    }
    
    try:
        # Initialize indexers
        youtube_indexer = YouTubeIndexer()
        podcast_indexer = PodcastIndexer()
        
        # Process YouTube sources
        if 'youtube' in sources:
            logger.info(f"Indexing {len(sources['youtube'])} YouTube sources")
            for url in sources['youtube']:
                try:
                    await youtube_indexer.index_channel(url, project_name)
                    results['successful'].append(f"YouTube: {url}")
                except Exception as e:
                    logger.error(f"Failed to index YouTube channel {url}: {str(e)}")
                    results['failed'].append(f"YouTube: {url}")
                
                if progress_callback:
                    await progress_callback('index', 'youtube', url)
        
        # Process podcast sources
        if 'podcast' in sources:
            logger.info(f"Indexing {len(sources['podcast'])} podcast sources")
            for url in sources['podcast']:
                try:
                    await podcast_indexer.index_feed(url, project_name)
                    results['successful'].append(f"Podcast: {url}")
                except Exception as e:
                    logger.error(f"Failed to index podcast feed {url}: {str(e)}")
                    results['failed'].append(f"Podcast: {url}")
                
                if progress_callback:
                    await progress_callback('index', 'podcast', url)
        
        return results
        
    except Exception as e:
        logger.error(f"Error in indexing step: {str(e)}")
        results['failed'].append(f"General indexing error: {str(e)}")
        return results