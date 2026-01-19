# src/processing/pipeline_manager.py

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, DefaultDict, Tuple, Set
import re
from collections import defaultdict

from sqlalchemy.orm import Session
from sqlalchemy import text, update
from sqlalchemy.exc import IntegrityError

# Add project root to Python path if necessary, or ensure proper packaging
import sys
from pathlib import Path
from src.utils.paths import get_project_root, get_config_path
sys.path.append(str(get_project_root()))

from src.database.session import get_session # Use the existing session context
from src.database.models import TaskQueue, Content, ContentChunk, EmbeddingSegment, Sentence
# from src.database.models import TASK_STATUS_PENDING, TASK_STATUS_PROCESSING, TASK_STATUS_COMPLETED, TASK_STATUS_FAILED # Import constants (Removed - Define locally)
from src.utils.human_behavior import HumanBehaviorManager # Import behavior manager
from src.utils.priority import calculate_priority_by_date  # Add this import
from src.utils.logger import setup_worker_logger
from src.storage.s3_utils import S3Storage, S3StorageConfig  # Add S3StorageConfig import
from src.utils.chunk_utils import get_chunk_plan, store_chunk_plan, create_chunk_plan
from src.utils.error_codes import ErrorCode, get_error_category
from src.processing.error_handler import ErrorHandler
from src.utils.version_utils import should_recreate_stitch_task, format_version_comparison_log
from src.processing_steps.stitch_steps.stage14_segment import get_current_segment_version
from minio import Minio

# Assuming logger setup similar to other modules
logger = setup_worker_logger('pipeline_manager')
# Configure logger further if needed (e.g., level, handlers)
# For now, basic config:
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Define Task Status Constants Locally
TASK_STATUS_PENDING = 'pending'
TASK_STATUS_PROCESSING = 'processing'
TASK_STATUS_COMPLETED = 'completed'
TASK_STATUS_FAILED = 'failed'
# TASK_STATUS_CANCELLED = 'cancelled' # Add if needed by PipelineManager

class PipelineManager:
    """
    Manages the workflow logic after a task completes or fails.
    Determines the next steps in the content processing pipeline.
    Also handles calculating worker wait times based on behavior.
    """

    # Add version identifier
    VERSION = "2.0.0"  # State evaluation based version

    # --- Error-based Task Recreation Policy ---
    # Map: task_type -> list of error keywords that trigger recreation
    RECREATE_ON_ERROR = {
        # For all task types, S3 connection errors should trigger recreation
        "*": [
            "Could not connect to the endpoint URL",
            "Connect timeout on endpoint URL",
            "S3 connection failed",
            "Failed to connect to S3",
            "endpoint URL"
        ],
        # Network errors for download tasks should trigger recreation
        "download_youtube": [
            "Network is unreachable",
            "Connection reset by peer",
            "Connection timed out",
            "Connection refused",
            "No route to host",
            "Name or service not known",
            "Temporary failure in name resolution",
            "Network error",
            "URLError",
            "ConnectionError",
            "TimeoutError",
            "OSError",
            "socket.gaierror"
        ],
        "download_rumble": [
            "Network is unreachable",
            "Connection reset by peer", 
            "Connection timed out",
            "Connection refused",
            "No route to host",
            "Name or service not known",
            "Temporary failure in name resolution",
            "Network error",
            "URLError",
            "ConnectionError",
            "TimeoutError",
            "OSError",
            "socket.gaierror"
        ],
        "download_podcast": [
            "Network is unreachable",
            "Connection reset by peer",
            "Connection timed out", 
            "Connection refused",
            "No route to host",
            "Name or service not known",
            "Temporary failure in name resolution",
            "Network error",
            "URLError",
            "ConnectionError", 
            "TimeoutError",
            "OSError",
            "socket.gaierror"
        ]
    }

    # Add helper method for consistent stitch version format
    def get_current_stitch_version(self):
        """Get the current stitch version with consistent format.
        
        Returns:
            Current stitch version string (e.g., 'stitch_v13' or 'stitch_v13.1')
            
        Note:
            This function supports both major versions (stitch_v13) and sub-versions (stitch_v13.1).
            For version compatibility checking, use should_recreate_stitch_task() or other
            functions from src.utils.version_utils instead of direct string comparison.
        """
        # Get version from config to match what stitch process uses
        try:
            return self.config.get('processing', {}).get('stitch', {}).get('current_version', 'stitch_v1')
        except Exception as e:
            logger.warning(f"Failed to get stitch version from config: {e}, using default 'stitch_v1'")
            return 'stitch_v1'

    def _is_content_within_project_date_range(self, content: Content, project: str) -> bool:
        """Check if content's publish_date falls within the project's configured date range."""
        if not content.publish_date:
            return True  # Allow content without publish_date

        try:
            # Get project configuration
            project_config = self.config.get('active_projects', {}).get(project, {})
            if not project_config:
                return True  # Allow if project not found in config

            # Check if project is enabled
            if not project_config.get('enabled', True):
                logger.debug(f"Project {project} is disabled - skipping task creation for content {content.content_id}")
                return False

            start_date_str = project_config.get('start_date')
            end_date_str = project_config.get('end_date')
            
            # Parse dates
            from datetime import datetime, timezone
            
            # Check start date
            if start_date_str:
                start_date = datetime.strptime(start_date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
                if content.publish_date < start_date:
                    return False
            
            # Check end date (inclusive - content published ON end_date should be included)
            if end_date_str:
                from datetime import timedelta
                end_date = datetime.strptime(end_date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
                # Add 1 day to make end_date inclusive (e.g., end_date=2025-10-31 includes all of Oct 31)
                end_date_exclusive = end_date + timedelta(days=1)
                if content.publish_date >= end_date_exclusive:
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error checking date range for content {content.content_id} in project {project}: {e}")
            return True  # Allow on error to be safe

    @classmethod
    def should_recreate_task(cls, task_type: str, error_message: str) -> bool:
        """Return True if the error for this task_type should trigger recreation."""
        # First check if this is an error we should NOT recreate for
        if "Failed to download audio chunk" in error_message:
            return False
        
        # Don't recreate for permanent YouTube/content errors
        permanent_error_patterns = [
            'Join this channel to get access to members-only content',
            'Sign in to confirm your age',
            'Video unavailable',
            'This video is private',
            'Private video',
            'This live event will begin in',
            'This live stream has ended',
            'Premiere will begin shortly',
            'Bad URL',
            'HTTP Error 400: Bad Request',
            'HTTP Error 403: Forbidden',
            'Failed to extract audio from source file'
        ]
        
        if any(pattern in error_message for pattern in permanent_error_patterns):
            logger.info(f"Not recreating {task_type} task due to permanent error: {error_message[:100]}")
            return False
        
        # Don't recreate conda run failures - these are usually permanent issues
        if "conda run" in error_message and "failed" in error_message:
            # For stitch tasks, also log the specific content ID for debugging
            if task_type == "stitch":
                import re
                content_match = re.search(r'--content=([a-f0-9-]+)', error_message)
                if content_match:
                    logger.warning(f"Stitch task failed with conda run error for content {content_match.group(1)} - not recreating")
            return False
        
        # Don't recreate stitch tasks - they typically fail due to missing prerequisites (audio, transcripts, etc.)
        # Exception: If it's a missing audio file error or S3 connection error, we'll handle it specially in handle_task_result
        if task_type == "stitch" and "No audio file found" not in error_message:
            # Check if this is an S3 connection error (which should be retried)
            s3_connection_errors = [
                "Could not connect to the endpoint URL",
                "Connect timeout on endpoint URL", 
                "S3 connection failed",
                "Failed to connect to S3",
                "endpoint URL"
            ]
            
            # Allow recreation for S3 connection errors
            if any(s3_error in error_message for s3_error in s3_connection_errors):
                logger.info(f"Stitch task failed with S3 connection error - allowing recreation")
                return True
            
            logger.warning(f"Stitch task failed - not recreating (likely missing prerequisites)")
            return False
        
        
        # Check task-specific first, then wildcard
        error_keywords = cls.RECREATE_ON_ERROR.get(task_type, []) + cls.RECREATE_ON_ERROR.get("*", [])
        return any(keyword in error_message for keyword in error_keywords)

    def __init__(self, behavior_manager: HumanBehaviorManager, config: Dict, worker_task_failures: DefaultDict[str, Dict[str, Any]]):
        """Initialize the PipelineManager.

        Args:
            behavior_manager: Instance of HumanBehaviorManager.
            config: The main application configuration dictionary.
            worker_task_failures: Reference to the orchestrator's failure tracking dict.
        """
        self.behavior_manager = behavior_manager
        self.config = config
        self.worker_task_failures = worker_task_failures # Store reference
        # Extract specific config values needed for convenience
        self.youtube_auth_pause_timeout = self.config.get('processing', {}).get('youtube_auth_pause_duration_seconds', 50400)  # Default 14 hours
        self.max_consecutive_failures = self.config.get('processing', {}).get('max_consecutive_failures', 3)

        # Initialize MinIO client
        self.minio_client = Minio(
            endpoint=self.config['storage']['s3']['endpoint_url'].replace('https://', '').replace('http://', ''),
            access_key=self.config['storage']['s3']['access_key'],
            secret_key=self.config['storage']['s3']['secret_key'],
            secure=self.config['storage']['s3']['use_ssl']
        )
        self.bucket_name = self.config['storage']['s3']['bucket_name']

        # Initialize S3 storage wrapper
        s3_config = S3StorageConfig(
            endpoint_url=self.config['storage']['s3']['endpoint_url'],
            access_key=self.config['storage']['s3']['access_key'],
            secret_key=self.config['storage']['s3']['secret_key'],
            bucket_name=self.config['storage']['s3']['bucket_name'],
            use_ssl=self.config['storage']['s3']['use_ssl']
        )
        self.s3_storage = S3Storage(s3_config)

        # Initialize error handler
        self.error_handler = ErrorHandler(config, worker_task_failures)

        # Load safe transcription models from config (language-aware)
        self.safe_transcription_models_english = self.config.get('processing', {}).get('transcription', {}).get('safe_models_english', [])
        self.safe_transcription_models_other = self.config.get('processing', {}).get('transcription', {}).get('safe_models_other', [])
        logger.info(f"Safe transcription models (English): {self.safe_transcription_models_english}")
        logger.info(f"Safe transcription models (Other): {self.safe_transcription_models_other}")

        logger.info(f"PipelineManager v{self.VERSION} initialized with ErrorHandler, BehaviorManager and config.")

    async def handle_task_result(self,
                                 session: Session,
                                 content_id: str,
                                 task_type: str,
                                 status: str,
                                 result: Dict[str, Any],
                                 db_task: TaskQueue,
                                 content: Content,
                                 worker_id: str,
                                 chunk_index: Optional[int] = None) -> Dict[str, Any]:
        """
        Handle task completion and determine next steps using state evaluation.
        
        This method:
        1. Handles special cases (auth errors, permanent failures)
        2. Evaluates actual content state
        3. Updates database flags to match reality
        4. Creates appropriate next tasks based on actual state
        
        Args:
            session: Active database session
            content_id: Content ID
            task_type: Type of task that completed
            status: Task status (completed, failed, skipped)
            result: Task result dictionary
            db_task: Database task object
            content: Content object
            worker_id: Worker ID
            chunk_index: Optional chunk index for chunk-specific tasks
            
        Returns:
            Dict containing:
                - available_after: When worker should be available again
                - outcome_message: Description of actions taken
        """
        try:
            # Debug logging for download_youtube tasks
            if task_type == "download_youtube":
                import json
                from pathlib import Path
                debug_log_path = Path("/tmp/download_youtube_debug.log")
                with open(debug_log_path, "a") as f:
                    f.write(f"\n{'='*80}\n")
                    f.write(f"Time: {datetime.now().isoformat()}\n")
                    f.write(f"Content ID: {content_id}\n")
                    f.write(f"Task Type: {task_type}\n")
                    f.write(f"Status: {status}\n")
                    f.write(f"Result: {json.dumps(result, indent=2)}\n")
                    f.write(f"Content blocked_download: {content.blocked_download}\n")
                    f.write(f"Content meta_data: {json.dumps(dict(content.meta_data) if content.meta_data else {}, indent=2)}\n")
                    f.write(f"Worker ID: {worker_id}\n")
                    f.write(f"{'='*80}\n")
            
            # Initialize outcome at the very beginning
            outcome = {
                'available_after': None,
                'outcome_message': "No outcome message set"
            }

            # EARLY CHECK: If content is blocked, skip ALL processing immediately
            # This prevents race conditions where content is blocked while a task is in progress
            if content.blocked_download:
                logger.info(f"Content {content_id} is blocked for download. Skipping all pipeline processing.")
                outcome['outcome_message'] = "Content blocked - no further processing"
                return outcome

            # Initialize error variables early to avoid scoping issues
            error_code = result.get('error_code') if isinstance(result, dict) else None
            error_message = str(result.get('error', '')) if isinstance(result, dict) else str(result)
            error_details = result.get('error_details', {}) if isinstance(result, dict) else {}

            # --- New Unified Error Handling ---
            # Handle both 'failed' and 'failed_permanent' statuses
            if status == TASK_STATUS_FAILED or status == 'failed_permanent':
                # Handle structured errors using ErrorHandler
                if error_code:
                    error_result = self.error_handler.handle_error(
                        task_type=task_type,
                        error_code=error_code,
                        error_message=error_message,
                        error_details=error_details,
                        content=content,
                        session=session,
                        worker_id=worker_id
                    )
                    
                    # Handle different error actions
                    if error_result['action'] == 'recreate_task':
                        # Recreate the task
                        db_task.status = TASK_STATUS_PENDING
                        db_task.worker_id = None
                        db_task.started_at = None
                        db_task.completed_at = None
                        db_task.error = None
                        session.add(db_task)
                        session.commit()
                        outcome['outcome_message'] = error_result['outcome_message']
                        return outcome
                    
                    elif error_result['action'] == 'block_content':
                        # Content was blocked by error handler
                        outcome['outcome_message'] = error_result['outcome_message']
                        return outcome
                    
                    elif error_result['action'] == 'pause_worker':
                        # Worker was paused
                        outcome['outcome_message'] = error_result['outcome_message']
                        outcome['available_after'] = error_result.get('available_after')
                        return outcome
                    
                    elif error_result['action'] == 'mark_ignored':
                        # Content was marked as ignored
                        outcome['outcome_message'] = error_result['outcome_message']
                        return outcome
                    
                    elif error_result.get('skip_state_audit', False):
                        # Skip state evaluation
                        outcome['outcome_message'] = error_result['outcome_message']
                        return outcome
                    
                    # For 'trigger_prerequisites' and 'continue', fall through to state evaluation
                
                # Legacy error handling for tasks not yet updated to use error codes
                elif self.should_recreate_task(task_type, error_message):
                    # Check if there are already 3+ failed tasks for the same content/task combination
                    # This prevents endless loops of task recreation
                    failed_task_count = session.query(TaskQueue).filter(
                        TaskQueue.content_id == content_id,
                        TaskQueue.task_type == task_type,
                        TaskQueue.status == TASK_STATUS_FAILED
                    ).count()
                    
                    if failed_task_count >= 3:
                        logger.warning(f"Task {task_type} for {content_id} has {failed_task_count} failed attempts. Not recreating to prevent endless loop.")
                        outcome['outcome_message'] = f"Task {task_type} for {content_id} failed permanently after {failed_task_count} attempts."
                        return outcome
                    
                    logger.warning(f"Task {task_type} for {content_id} failed with error '{error_message}'. Policy: Will recreate task (attempt {failed_task_count + 1}).")
                    # Recreate the task by setting status to pending and clearing error
                    db_task.status = TASK_STATUS_PENDING
                    db_task.worker_id = None
                    db_task.started_at = None
                    db_task.completed_at = None
                    db_task.error = None
                    session.add(db_task)
                    session.commit()
                    outcome['outcome_message'] = f"Task {task_type} for {content_id} failed. Task reset to pending for retry (attempt {failed_task_count + 1})."
                    return outcome
            # --- End New Unified Error Handling ---

            # === LEGACY ERROR HANDLING (Kept for backward compatibility) ===
            # These will be removed once all processing steps are updated to use error codes
            
            # Handle YouTube auth errors (will be replaced by YOUTUBE_AUTH error code)
            if task_type == "download_youtube" and result.get('error_type') == 'youtube_auth' and not error_code:
                logger.warning("Legacy YouTube auth error handling - task should use ErrorCode.YOUTUBE_AUTH")
                # Use error handler with synthetic error code
                error_result = self.error_handler.handle_error(
                    task_type=task_type,
                    error_code=ErrorCode.YOUTUBE_AUTH.value,
                    error_message=result.get('error', 'Unknown error'),
                    error_details={},
                    content=content,
                    session=session,
                    worker_id=worker_id
                )
                outcome['outcome_message'] = error_result['outcome_message']
                return outcome

            # Handle corrupt media (will be replaced by CORRUPT_MEDIA error code)
            if task_type == "convert" and status == "failed" and result.get('corrupt_media_detected', False) and not error_code:
                logger.warning("Legacy corrupt media handling - task should use ErrorCode.CORRUPT_MEDIA")
                # Use error handler with synthetic error code
                error_result = self.error_handler.handle_error(
                    task_type=task_type,
                    error_code=ErrorCode.CORRUPT_MEDIA.value,
                    error_message=result.get('error', 'Corrupt media detected'),
                    error_details={},
                    content=content,
                    session=session,
                    worker_id=worker_id
                )
                # DON'T recreate download task automatically - let state evaluation handle it
                # This prevents endless loops when convert consistently fails
                content.is_converted = False
                content.last_updated = datetime.now(timezone.utc)
                session.add(content)
                session.commit()
                outcome['outcome_message'] = error_result['outcome_message'] + " - Convert will be retried, download task creation blocked to prevent endless loop"
                return outcome

            # Handle stitch missing audio (will be replaced by MISSING_AUDIO error code)
            if task_type == "stitch" and status == "failed" and "No audio file found" in str(result.get('error', '')) and not error_code:
                logger.warning("Legacy stitch missing audio handling - task should use ErrorCode.MISSING_AUDIO")
                # Don't recreate task for missing audio - this is likely a transient issue
                # Skip state evaluation to prevent task recreation
                outcome['outcome_message'] = f"Stitch task failed due to missing audio file - not recreating task"
                return outcome
                
            # Handle permanent failures (kept for backward compatibility with 'permanent' flag)
            # Note: orchestrator may pass status as 'failed_permanent' instead of 'failed' with permanent flag
            if (status == TASK_STATUS_FAILED and result.get('permanent', False) and not error_code) or \
               (status == 'failed_permanent' and not error_code):
                logger.warning(f"Legacy permanent failure handling for {task_type} - task should use appropriate ErrorCode")
                logger.warning(f"Status: {status}, Result has permanent flag: {result.get('permanent', 'not set')}")
                
                # For download tasks with permanent failures, block the content
                if task_type in ["download_youtube", "download_podcast", "download_rumble"]:
                    logger.warning(f"BLOCKING content {content_id} due to permanent download failure: {result.get('error', 'Unknown error')}")
                    content.blocked_download = True
                    
                    # Update metadata
                    meta_data = dict(content.meta_data) if content.meta_data else {}
                    meta_data.update({
                        'block_reason': result.get('error', 'Permanent download failure'),
                        'permanent_block': 'true',
                        'blocked_at': datetime.now(timezone.utc).isoformat(),
                        'error_details': {'permanent': True}
                    })
                    content.meta_data = meta_data
                    content.last_updated = datetime.now(timezone.utc)
                    
                    session.add(content)
                    session.commit()
                    
                    outcome['outcome_message'] = f"Content blocked: {result.get('error', 'Permanent download failure')}"
                    outcome['skip_state_audit'] = True
                else:
                    # For non-download tasks, use the error handler
                    error_result = self.error_handler.handle_error(
                        task_type=task_type,
                        error_code=ErrorCode.UNKNOWN_ERROR.value,
                        error_message=result.get('error', 'Unknown permanent error'),
                        error_details={'permanent': True},
                        content=content,
                        session=session,
                        worker_id=worker_id
                    )
                    outcome['outcome_message'] = error_result['outcome_message']
                    
                    # Apply the action from error handler if needed
                    if error_result.get('skip_state_audit'):
                        outcome['skip_state_audit'] = True
                
                return outcome

            # Handle diarization empty result (will be replaced by EMPTY_RESULT error code)
            if task_type == "diarize" and status == "failed" and "Diarization pipeline returned empty result" in str(result.get('error', '')) and not error_code:
                logger.warning("Legacy diarization empty result handling - task should use ErrorCode.EMPTY_RESULT")
                error_result = self.error_handler.handle_error(
                    task_type=task_type,
                    error_code=ErrorCode.EMPTY_RESULT.value,
                    error_message="Diarization pipeline returned empty result",
                    error_details={},
                    content=content,
                    session=session,
                    worker_id=worker_id
                )
                outcome['outcome_message'] = error_result['outcome_message']
                return outcome

            # Handle bad URL (will be replaced by BAD_URL error code)
            if task_type == "download_podcast" and status == "failed" and "Bad URL" in str(result.get('error', '')) and not error_code:
                logger.warning("Legacy bad URL handling - task should use ErrorCode.BAD_URL")
                error_result = self.error_handler.handle_error(
                    task_type=task_type,
                    error_code=ErrorCode.BAD_URL.value,
                    error_message=result.get('error', ''),
                    error_details={},
                    content=content,
                    session=session,
                    worker_id=worker_id
                )
                outcome['outcome_message'] = error_result['outcome_message']
                return outcome

            # Handle members-only content (will be replaced by MEMBERS_ONLY error code)
            if task_type == "download_youtube" and status == "failed" and "Join this channel to get access to members-only content" in str(result.get('error', '')) and not error_code:
                logger.warning("Legacy members-only handling - task should use ErrorCode.MEMBERS_ONLY")
                error_result = self.error_handler.handle_error(
                    task_type=task_type,
                    error_code=ErrorCode.MEMBERS_ONLY.value,
                    error_message="Members-only content",
                    error_details={},
                    content=content,
                    session=session,
                    worker_id=worker_id
                )
                outcome['outcome_message'] = error_result['outcome_message']
                return outcome

            # Handle age-restricted content (will be replaced by AGE_RESTRICTED error code)
            if task_type == "download_youtube" and status == "failed" and "Sign in to confirm your age" in str(result.get('error', '')) and not error_code:
                logger.warning("Legacy age-restricted handling - task should use ErrorCode.AGE_RESTRICTED")
                error_result = self.error_handler.handle_error(
                    task_type=task_type,
                    error_code=ErrorCode.AGE_RESTRICTED.value,
                    error_message="Age-restricted content",
                    error_details={},
                    content=content,
                    session=session,
                    worker_id=worker_id
                )
                outcome['outcome_message'] = error_result['outcome_message']
                return outcome
            
            # === END LEGACY ERROR HANDLING ===

            # Handle successful task completions before state evaluation
            if status == TASK_STATUS_COMPLETED:
                # For convert task completion, store chunk plan in database
                if task_type == "convert" and isinstance(result, dict):
                    chunk_plan = result.get('chunk_plan', [])
                    if chunk_plan:
                        logger.info(f"Convert completed for {content_id} with {len(chunk_plan)} chunks - storing in database")
                        # Transform chunk_plan format from convert output to what store_chunk_plan expects
                        formatted_chunks = []
                        for chunk in chunk_plan:
                            formatted_chunks.append({
                                'index': chunk.get('index'),
                                'start_time': chunk.get('start_time'),
                                'end_time': chunk.get('end_time'),
                                'duration': chunk.get('duration')
                            })
                        
                        if store_chunk_plan(session, content, formatted_chunks):
                            logger.info(f"Successfully stored {len(formatted_chunks)} chunks for {content_id}")
                            content.last_updated = datetime.now(timezone.utc)
                            session.add(content)
                            session.commit()
                            session.refresh(content)
                        else:
                            logger.error(f"Failed to store chunk plan for {content_id}")
                    else:
                        logger.warning(f"Convert task completed but no chunk_plan in result for {content_id}")
                
                # For stitch task completion, immediately update flags to prevent duplicate task creation
                elif task_type == "stitch":
                    current_version = self.get_current_stitch_version()
                    
                    # CRITICAL: Commit the current transaction to ensure we see changes from the stitch process
                    # The stitch task runs in a separate process/transaction, so we need to commit
                    # to see its database changes
                    session.commit()
                    
                    # Force a fresh read from database to see any updates from the stitch process
                    session.expire_all()  # Expire all objects to force fresh reads
                    content = session.query(Content).filter_by(content_id=content_id).first()
                    
                    if not content:
                        logger.error(f"Content {content_id} not found after refresh")
                        outcome['outcome_message'] = f"Content {content_id} not found after refresh"
                        return outcome
                    
                    # Check if stitch process already updated the flags
                    logger.info(f"After stitch completion and refresh: is_stitched={content.is_stitched}, stitch_version={content.stitch_version}")

                    # Check for sentences (primary output) - sentences table is now the atomic unit
                    # Note: speaker_transcriptions table is deprecated, sentences replaced it
                    sentence_count = session.query(Sentence).filter(
                        Sentence.content_id == content.id,
                        Sentence.stitch_version == current_version
                    ).count()

                    logger.info(f"Found {sentence_count} sentences with version {current_version} for {content_id}")

                    if sentence_count > 0:
                        # Sentences exist - ensure flags are set correctly
                        if not content.is_stitched or content.stitch_version != current_version:
                            logger.info(f"Stitch task completed - found {sentence_count} sentences, updating is_stitched=True and stitch_version={current_version} for {content_id}")
                            content.is_stitched = True
                            content.stitch_version = current_version
                            content.last_updated = datetime.now(timezone.utc)
                            session.add(content)
                            session.commit()
                            session.refresh(content)
                            logger.info(f"Successfully updated flags for {content_id}: is_stitched={content.is_stitched}, stitch_version={content.stitch_version}")
                        else:
                            logger.info(f"Flags already correct for {content_id}: is_stitched={content.is_stitched}, stitch_version={content.stitch_version}")
                    else:
                        logger.warning(f"Stitch task completed but no Sentence records found for {content_id} with version {current_version}")
                        logger.warning(f"This indicates the stitch task failed to save to database - will NOT create another stitch task immediately")
                        # Set a flag to prevent immediate recreation
                        outcome['outcome_message'] = f"Stitch task completed but database save failed - preventing immediate recreation"
                        return outcome
                
                # NOTE: segment tasks are deprecated - segmentation is now integrated into stitch Stage 13

                # For cleanup task completion, skip state evaluation and pipeline processing
                elif task_type == "cleanup":
                    logger.info(f"Cleanup task completed for {content_id} - marking as compressed and skipping further pipeline processing")
                    content.is_compressed = True
                    content.last_updated = datetime.now(timezone.utc)
                    session.add(content)
                    session.commit()
                    outcome['outcome_message'] = "Cleanup completed - marked as compressed"
                    return outcome

            # For all other cases, evaluate current state and create necessary tasks
            # Note: blocked_download check is now done at the start of handle_task_result

            # Debug log before state evaluation
            if task_type == "download_youtube":
                with open("/tmp/download_youtube_debug.log", "a") as f:
                    f.write(f"\nPROCEEDING TO STATE EVALUATION\n")
                    f.write(f"  Content {content_id} blocked_download: {content.blocked_download}\n")
            
            eval_results = await self.evaluate_content_state(session, content, db_task)
            
            # Debug log evaluation results
            if task_type == "download_youtube":
                with open("/tmp/download_youtube_debug.log", "a") as f:
                    f.write(f"\nSTATE EVALUATION RESULTS:\n")
                    f.write(f"  flags_updated: {eval_results.get('flags_updated', [])}\n")
                    f.write(f"  tasks_created: {eval_results.get('tasks_created', [])}\n")
                    f.write(f"  errors: {eval_results.get('errors', [])}\n")
            
            # Calculate worker availability based on task type and result
            if task_type in ["download_youtube", "download_rumble"]:
                logger.info(f"🎯 Processing {task_type} task completion for {worker_id} - checking behavior wait requirements")
            
            skip_behavior_wait = (
                (task_type in ["download_youtube", "download_rumble"] and status == "skipped" and result.get('skipped_existing', False)) or
                # If download task was skipped due to already existing in S3, skip behavior wait
                (task_type in ["download_youtube", "download_rumble"] and status == "skipped" and result.get('reason') == 'already_exists') or
                # Also skip if skip_wait_time flag is explicitly set
                (task_type in ["download_youtube", "download_rumble"] and result.get('skip_wait_time', False))
            )
            outcome['available_after'] = self._calculate_worker_available_after(
                worker_id, task_type, skip_behavior_wait
            )

            # Construct outcome message from evaluation results
            if eval_results['errors']:
                outcome['outcome_message'] = f"Errors during evaluation: {'; '.join(eval_results['errors'])}"
            else:
                updates = []
                # Always show flag updates first
                if eval_results['flags_updated']:
                    updates.append(f"Updated {', '.join(eval_results['flags_updated'])}")
                # Then show task creations
                if eval_results['tasks_created']:
                    updates.append(f"Created {', '.join(eval_results['tasks_created'])}")
                # Then show blocked tasks (but skip transcribe blocks as they're too verbose)
                if eval_results['tasks_blocked'] and task_type != 'transcribe':
                    updates.append(f"Blocked {', '.join(eval_results['tasks_blocked'])}")
                # If no updates but task completed, show appropriate message
                if not updates:
                    if task_type == 'transcribe':
                        # Get list of pending chunks
                        chunks = session.query(ContentChunk).filter_by(content_id=content.id).all()
                        pending_chunks = [str(c.chunk_index) for c in chunks if c.transcription_status != 'completed']
                        if pending_chunks:
                            updates.append(f"Verified chunk {chunk_index}, waiting for chunks {','.join(pending_chunks)}")
                        else:
                            updates.append(f"Verified chunk {chunk_index}, all chunks complete")
                    else:
                        updates.append(f"Verified {task_type} completion")
                
                outcome['outcome_message'] = '. '.join(updates)

            return outcome

        except Exception as e:
            logger.error(f"Error handling task result: {str(e)}", exc_info=True)
            return {
                'available_after': None,
                'outcome_message': f"Error handling task result: {str(e)}"
            }

    def _calculate_worker_available_after(self, worker_id: str, task_type: str, skip_behavior_wait: bool) -> Optional[datetime]:
        """Calculates the next available time for a worker based on behavior rules."""
        # --- Optimization: Only apply behavior management to YouTube and Rumble download tasks ---
        if task_type not in ["download_youtube", "download_rumble"]:
             logger.debug(f"Skipping behavior management calculation for non-behavior-managed task: {task_type}")
             return None # Immediately available for other task types

        # --- Existing logic for download tasks ---
        logger.info(f"🤖 Calculating behavior wait time for {worker_id} after {task_type} task")
        
        if skip_behavior_wait:
            # Add more detailed logging
            logger.info(f"Worker {worker_id} skipping behavior wait time for {task_type} task based on 'skip_behavior_wait=True' flag (this typically means the download was skipped due to content already existing)")
            return None # Immediately available

        if not self.behavior_manager:
            logger.warning(f"BehaviorManager not available in PipelineManager. Cannot calculate wait time for {task_type}.")
            return None

        logger.info(f"Calculating post-completion wait time for {worker_id} after {task_type}...")
        if worker_id not in self.behavior_manager.worker_states:
             logger.warning(f"Cannot apply post-completion behavior management: Worker {worker_id} not found in behavior manager states.")
             return None # Make available if state not found

        state = self.behavior_manager.worker_states[worker_id]
        now_utc = datetime.now(timezone.utc)

        # Update behavior state *first* (only for download_youtube tasks now)
        self.behavior_manager.handle_task_completion(worker_id, task_type, current_time_override=now_utc)

        # Calculate wait times based on updated state
        behavior_wait_time, reason = self.behavior_manager.calculate_next_task_wait_time(worker_id, task_type) # Unpack tuple
        post_task_delay = self.behavior_manager.calculate_post_task_delay(worker_id, task_type) # Random inter-task delay

        # Final wait time is the MAX
        wait_duration_seconds = max(behavior_wait_time, post_task_delay)
        logger.info(f"Worker {worker_id} post-completion delays: behavior_wait={behavior_wait_time:.1f}s ({reason}), post_task={post_task_delay:.1f}s => Max Wait: {wait_duration_seconds:.1f}s")

        # Determine when the worker is available
        available_after_time: Optional[datetime] = None
        if wait_duration_seconds > 0:
             # This block now only runs if task_type was already a download task
             available_after_time = now_utc + timedelta(seconds=wait_duration_seconds)
             logger.info(f"🕐 Worker {worker_id} post-completion wait for {task_type}: {wait_duration_seconds:.1f}s --> AvailableAfter={available_after_time}")
        else:
             logger.info(f"✅ No post-completion wait time required for {worker_id} (after {task_type}).")
             available_after_time = None # Immediately available

        return available_after_time

    async def _check_and_create_stitch_task(self, session: Session, content: Content) -> Tuple[bool, Optional[int]]:
        """Checks if content is ready for stitching and creates the task. Returns (task_created, task_id).
        
        Note: The stitch task now integrates speaker identification (Step 6) and will:
        - Create speaker_mapping.json and update is_identified flag
        - Create speaker_turns.json and update is_stitched flag
        """
        logger.debug(f"Checking stitch readiness for {content.content_id}")
        # Content state should already be up to date since we commit before calling this
        logger.debug(f"Checking stitch readiness with state: is_diarized={content.is_diarized}, is_stitched={content.is_stitched}, is_transcribed={content.is_transcribed}")

        # Get current stitch version using the new helper method
        current_version = self.get_current_stitch_version()

        # Check if already stitched with current version
        if content.is_stitched and not should_recreate_stitch_task(current_version, content.stitch_version):
            logger.debug(f"Already stitched with compatible version. {format_version_comparison_log(current_version, content.stitch_version)}")
            return False, None

        # Check if all chunks are transcribed
        all_transcribed = False
        total_chunks = session.query(ContentChunk).filter_by(content_id=content.id).count()
        if total_chunks > 0: # Avoid division by zero or checks if no chunks exist
            completed_chunks = session.query(ContentChunk).filter(
                ContentChunk.content_id == content.id,
                ContentChunk.transcription_status == 'completed'
            ).count()
            if total_chunks == completed_chunks:
                all_transcribed = True

        # Check conditions for stitch task creation (requires transcription AND diarization)
        # The stitch task now handles speaker identification internally
        if all_transcribed and content.is_diarized:
            logger.info(f"Content {content.content_id} is ready for stitching (all {total_chunks} chunks transcribed AND diarization complete).")
            new_task_id, _ = await self._create_task_if_not_exists(
                session,
                content_id=content.content_id,
                task_type='stitch',
                input_data={'project': content.projects[0] if content.projects else 'unknown'},
                content=content
            )
            return bool(new_task_id), new_task_id

        return False, None

    async def _create_task_if_not_exists(self, session: Session, content_id: str, task_type: str, input_data: Dict[str, Any], priority: int = None, content: Content = None) -> Tuple[Optional[int], Optional[str]]:
        """Creates a new task if a pending/processing one doesn't already exist. 
        
        Returns:
            Tuple of (task_id, block_reason):
            - task_id: New task ID if created, None if not created
            - block_reason: Reason task was not created (None if task was created)
        """
        logger.debug(f"Checking for existing task {task_type} for {content_id}")
        
        # First check if content has a permanent error in meta_data
        content_obj = content if content else session.query(Content).filter_by(content_id=content_id).first()
        if content_obj and content_obj.meta_data:
            if content_obj.meta_data.get('permanent_error'):
                error_code = content_obj.meta_data.get('error_code', 'UNKNOWN')
                error_msg = content_obj.meta_data.get('error_message', 'Permanent error')
                logger.warning(f"Content {content_id} has permanent error ({error_code}): {error_msg}")
                return None, f"Content has permanent error: {error_code}"
        
        # Universal task limit check: prevent creating new tasks if 3+ tasks (including failed) exist for this content/task combination
        # Count pending, processing, failed tasks to prevent retry loops
        task_count_query = session.query(TaskQueue).filter(
            TaskQueue.content_id == content_id,
            TaskQueue.task_type == task_type,
            TaskQueue.status.in_([TASK_STATUS_PENDING, TASK_STATUS_PROCESSING, TASK_STATUS_FAILED])
        )
        
        # For transcribe tasks, also filter by chunk_index to count per-chunk
        if task_type == 'transcribe':
            chunk_index = input_data.get('chunk_index')
            if chunk_index is not None:
                task_count_query = task_count_query.filter(
                    text("input_data->>'chunk_index' = :chunk_index").params(chunk_index=str(chunk_index))
                )
        
        active_task_count = task_count_query.count()
        
        if active_task_count >= 3:
            # Get task status breakdown for logging (check all statuses for the full picture)
            all_tasks_query = session.query(TaskQueue).filter(
                TaskQueue.content_id == content_id,
                TaskQueue.task_type == task_type
            )
            if task_type == 'transcribe' and chunk_index is not None:
                all_tasks_query = all_tasks_query.filter(
                    text("input_data->>'chunk_index' = :chunk_index").params(chunk_index=str(chunk_index))
                )
            
            status_counts = {}
            for status in ['pending', 'processing', 'completed', 'failed']:
                count = all_tasks_query.filter(TaskQueue.status == status).count()
                if count > 0:
                    status_counts[status] = count
            
            status_summary = ', '.join([f"{status}: {count}" for status, count in status_counts.items()])
            
            if task_type == 'transcribe' and chunk_index is not None:
                block_reason = f"Already have {active_task_count} {task_type} tasks (including failed) for chunk {chunk_index} ({status_summary})"
                logger.warning(f"Skipping creation of {task_type} task for {content_id} chunk {chunk_index} - {block_reason.lower()}")
            else:
                block_reason = f"Already have {active_task_count} {task_type} tasks (including failed) ({status_summary})"
                logger.warning(f"Skipping creation of {task_type} task for {content_id} - {block_reason.lower()}")
            return None, block_reason
        
        # Special check for stitch tasks - if content is already stitched with current version, don't create
        if task_type == 'stitch':
            # Use passed content object if available, otherwise query
            content_obj = content if content else session.query(Content).filter_by(content_id=content_id).first()
            if content_obj and content_obj.is_stitched and not should_recreate_stitch_task(self.get_current_stitch_version(), content_obj.stitch_version):
                logger.debug(f"Content {content_id} already stitched with compatible version. {format_version_comparison_log(self.get_current_stitch_version(), content_obj.stitch_version)}")
                return None, f"Already stitched with compatible version {content_obj.stitch_version}"
        
        # First check if this task has previously failed with a permanent error
        failed_task = session.query(TaskQueue).filter(
            TaskQueue.content_id == content_id,
            TaskQueue.task_type == task_type,
            TaskQueue.status == 'failed'
        ).first()

        if failed_task:
            # Check if it's a permanent failure that shouldn't be retried
            error_str = str(failed_task.error or '')
            permanent_errors = [
                'Join this channel to get access to members-only content',
                'Sign in to confirm your age',
                'Bad URL',
                'Permanent failure',
                'Video unavailable',
                'This video is private',
                'Content blocked',
                'This live event will begin in',
                'This live stream has ended',
                'Premiere will begin shortly',
                'Private video',
                'HTTP Error 400: Bad Request',
                'HTTP Error 403: Forbidden',
                'Failed to extract audio from source file'
            ]
            
            # Add task-specific permanent errors
            # NOTE: We specifically exclude 'No audio file found' for stitch tasks since that's a recoverable error
            
            # For stitch tasks, only treat specific errors as permanent (not unknown/transient errors)
            # Note: exit code -15 is SIGTERM (process killed), -9 is SIGKILL - both are retryable
            if task_type == 'stitch':
                stitch_permanent_errors = [
                    'No diarization data',
                    'No transcription data',
                    'Content is blocked',
                ]
                if 'No audio file found' in error_str:
                    logger.info(f"Stitch failed due to missing audio for {content_id} - this is recoverable, not permanent")
                    is_permanent = False
                elif any(keyword in error_str for keyword in stitch_permanent_errors):
                    logger.info(f"Treating stitch failure as permanent for {content_id}: {error_str[:100]}")
                    is_permanent = True
                else:
                    # Unknown errors, timeouts, etc. should be retryable
                    logger.info(f"Stitch failure for {content_id} is retryable (not a known permanent error): {error_str[:100]}")
                    is_permanent = False
            else:
                is_permanent = any(keyword in error_str for keyword in permanent_errors)
            
            if is_permanent:
                logger.info(f"Not creating new {task_type} task for {content_id} due to permanent failure: {error_str[:100]}")
                return None, f"Permanent failure: {error_str[:100]}"
            else:
                # For non-permanent failures, allow task recreation
                logger.debug(f"Allowing recreation of {task_type} task for {content_id} (previous failure was not permanent)")

        # Check for existing pending/processing task, or failed task that can be reset
        existing_task_query = session.query(TaskQueue).filter(
            TaskQueue.content_id == content_id,
            TaskQueue.task_type == task_type,
            TaskQueue.status.in_([TASK_STATUS_PENDING, TASK_STATUS_PROCESSING, TASK_STATUS_FAILED])
        )

        # Add chunk_index check specifically for transcribe tasks
        if task_type == 'transcribe':
            chunk_index = input_data.get('chunk_index')
            if chunk_index is not None:
                 existing_task_query = existing_task_query.filter(
                      text("input_data->>'chunk_index' = :chunk_index").params(chunk_index=str(chunk_index))
                 )
            else:
                 logger.warning(f"Attempted to create transcribe task for {content_id} without chunk_index in input_data. Cannot guarantee uniqueness.")

        existing_task = existing_task_query.first()

        if existing_task and existing_task.status == TASK_STATUS_FAILED:
            # Check if this was a permanent failure - don't reset those
            task_result = existing_task.result or {}
            if task_result.get('permanent', False):
                error_code = task_result.get('error_code', 'unknown')
                logger.info(f"Not resetting permanently failed task {existing_task.id} ({task_type}) for {content_id} "
                           f"(error_code: {error_code})")
                return None, f"Task permanently failed with {error_code}"

            # Reset failed task to pending for retry
            existing_task.status = TASK_STATUS_PENDING
            existing_task.error = None
            existing_task.worker_id = None
            existing_task.processor_task_id = None
            existing_task.started_at = None
            existing_task.completed_at = None
            session.flush()
            logger.info(f"Reset failed task {existing_task.id} ({task_type}) for {content_id} " +
                       (f"(chunk {input_data.get('chunk_index')}) " if task_type == 'transcribe' else "") +
                       f"to pending (attempt {existing_task.attempts + 1}/{existing_task.max_attempts})")
            return existing_task.id, None
        elif existing_task:
            logger.debug(f"Task {task_type} already exists (pending/processing) for {content_id}. Skipping creation.")
            return None, f"Task already exists (pending/processing)"
        else:
            # Get project priority from config - use highest priority if multiple projects
            project_priority = 1  # Default priority

            active_projects_config = self.config.get('active_projects', {})

            # Get project from input_data or from content object
            project_str = input_data.get('project')
            if not project_str and content:
                project_str = content.projects
            elif not project_str:
                content_obj = session.query(Content).filter_by(content_id=content_id).first()
                project_str = content_obj.projects if content_obj else None

            # Handle multiple projects - use highest priority
            if project_str and isinstance(active_projects_config, dict):
                projects = project_str if isinstance(project_str, list) else [project_str]
                for project in projects:
                    if project in active_projects_config:
                        project_specific_config = active_projects_config.get(project)
                        if isinstance(project_specific_config, dict):
                            proj_priority = project_specific_config.get('priority', 1)
                            project_priority = max(project_priority, proj_priority)
            
            # Calculate priority using both date and project priority
            if priority is None:
                content_obj = content if content else session.query(Content).filter_by(content_id=content_id).first()
                publish_date = getattr(content_obj, 'publish_date', None) if content_obj else None
                priority = calculate_priority_by_date(publish_date, project_priority)
            else:
                # If priority was explicitly provided, still add project priority
                priority = priority + (project_priority * 1000000)

            new_task = TaskQueue(
                content_id=content_id,
                task_type=task_type,
                status=TASK_STATUS_PENDING,
                priority=priority,
                input_data=input_data,
                created_at=datetime.now(timezone.utc)
            )
            session.add(new_task)
            try:
                session.flush() # Flush to get the ID before returning
                task_id = new_task.id # Get the ID
                logger.info(f"Created new task: {task_type} (ID: {task_id}) for {content_id} " +
                           (f"(chunk {input_data.get('chunk_index')})" if task_type == 'transcribe' else "") +
                           f" with priority {priority}")
                return task_id, None # Return the ID and no block reason
            except IntegrityError as e:
                session.rollback()
                logger.warning(f"Task {task_type} for {content_id} " +
                             (f"(chunk {input_data.get('chunk_index')}) " if task_type == 'transcribe' else "") +
                             f"already exists (race condition caught): {e}")
                return None, "Task already exists (race condition)"

    async def _update_content_chunks(self, session: Session, content: Content, chunk_plan: List[Dict[str, Any]]):
        """Updates ContentChunk records based on the plan from convert task."""
        logger.debug(f"Updating chunks for {content.content_id} based on convert plan ({len(chunk_plan)} chunks)")
        # Clear existing chunks for this content? Or update? Let's delete and recreate for simplicity.
        session.query(ContentChunk).filter_by(content_id=content.id).delete()
        session.flush() # Ensure delete happens before add

        chunks_processed_count = 0
        for chunk_info in chunk_plan:
            chunk = ContentChunk(
                content_id=content.id, # Use internal integer ID
                chunk_index=chunk_info['index'],
                start_time=chunk_info['start_time'],
                end_time=chunk_info['end_time'],
                duration=chunk_info['duration'],
                extraction_status=chunk_info['extraction_status'],
                transcription_status='pending' # New chunks always start as pending transcription
            )
            session.add(chunk)
            if chunk_info['extraction_status'] == 'completed':
                chunks_processed_count += 1

        # Update content chunk counts
        content.total_chunks = len(chunk_plan)
        content.chunks_processed = chunks_processed_count # This might not be needed if only based on transcription
        session.add(content)
        logger.debug(f"Updated chunks for {content.content_id}: total={content.total_chunks}")

    async def evaluate_content_state(self, session: Session, content: Content, db_task: Optional[TaskQueue] = None) -> Dict[str, Any]:
        """Reconcile database state with S3 reality and create any missing tasks.
        
        This method acts as the cleanup/reconciliation step after task completion:
        1. Check what files actually exist in S3
        2. Update database flags and chunk status to match reality
        3. Create any tasks needed based on actual state
        
        This ensures the database stays in sync with S3, which is the source of truth.
        create_tasks.py can safely use database state to create initial tasks, and this
        method will clean up any inconsistencies during task processing.
        
        Args:
            session: Active database session
            content: Content object to evaluate
            db_task: Optional TaskQueue object that triggered this evaluation
            
        Returns:
            Dict containing evaluation results and actions taken
        """
        logger.debug(f"Reconciling state for content {content.content_id}")
        
        # CRITICAL: Refresh content object to ensure we have the latest state
        # This prevents race conditions when called after task completion handlers
        # that may have updated the content flags in a separate transaction
        content_id = content.content_id  # Store before refresh
        session.expire_all()  # Expire all objects to force fresh reads
        content = session.query(Content).filter_by(content_id=content_id).first()
        
        if not content:
            logger.error(f"Content {content_id} not found after refresh in evaluate_content_state")
            return {
                'content_id': content_id,
                'flags_updated': [],
                'tasks_created': [],
                'tasks_blocked': [],
                'errors': ['Content not found after refresh']
            }
        
        results = {
            'content_id': content.content_id,
            'flags_updated': [],
            'tasks_created': [],
            'tasks_blocked': [],
            'errors': []
        }
        
        # Early check: if content is fully processed, be extra careful about state changes
        if content.is_downloaded and content.is_stitched:
            logger.debug(f"Content {content.content_id} is fully processed (downloaded & stitched) - being careful with state changes")


        try:
            # Initialize S3 storage
            s3_config = S3StorageConfig(
                endpoint_url=self.config['storage']['s3']['endpoint_url'],
                access_key=self.config['storage']['s3']['access_key'],
                secret_key=self.config['storage']['s3']['secret_key'],
                bucket_name=self.config['storage']['s3']['bucket_name'],
                use_ssl=self.config['storage']['s3']['use_ssl']
            )
            s3_storage = S3Storage(s3_config)
            
            # Get all files in content directory in one operation
            content_prefix = f"content/{content.content_id}/"
            try:
                all_files = set(s3_storage.list_files(content_prefix))
                logger.debug(f"Found {len(all_files)} files for content {content.content_id}")
            except Exception as s3_error:
                logger.error(f"S3 error listing files for {content.content_id}: {s3_error}")
                results['errors'].append(f"S3 error: {str(s3_error)}")
                # Don't modify any state if we can't access S3
                logger.warning(f"Skipping state reconciliation for {content.content_id} due to S3 error")
                return results

            # Check existence of all relevant files (considering compression)
            storage_manifest_exists = f"{content_prefix}storage_manifest.json" in all_files
            
            # For audio, check both original WAV and compressed formats
            audio_exists = (
                f"{content_prefix}audio.wav" in all_files or 
                f"{content_prefix}audio.opus" in all_files or
                f"{content_prefix}audio.mp3" in all_files
            )
            
            # For source files, check both original and compressed formats
            source_files_exist = (
                any(f"{content_prefix}source{ext}" in all_files 
                    for ext in ['.mp4', '.mp3', '.wav', '.m4a']) or
                any(f"{content_prefix}video{ext}" in all_files 
                    for ext in ['.mp4', '.webm'])
            )
            
            # Debug logging for download task completion
            if not source_files_exist and not audio_exists:
                logger.debug(f"No source or audio files found for {content.content_id}. Files found: {sorted(all_files)}")
            elif source_files_exist and not audio_exists:
                logger.debug(f"Source files exist but no audio for {content.content_id}. Should create convert task.")
            # For JSON files, check both original and compressed formats
            diarization_exists = (
                f"{content_prefix}diarization.json" in all_files or
                f"{content_prefix}diarization.json.gz" in all_files
            )
            
            # Note: speaker_embeddings.json is now generated by stitch, not diarize
            embeddings_exist = (
                f"{content_prefix}speaker_embeddings.json" in all_files or
                f"{content_prefix}speaker_embeddings.json.gz" in all_files
            )
            
            speaker_mapping_exists = (
                f"{content_prefix}speaker_mapping.json" in all_files or
                f"{content_prefix}speaker_mapping.json.gz" in all_files
            )
            
            # Stitch creates transcript_diarized.json
            stitched_exists = (
                f"{content_prefix}transcript_diarized.json" in all_files or
                f"{content_prefix}transcript_diarized.json.gz" in all_files
            )
            
            semantic_segments_exist = (
                f"{content_prefix}semantic_segments.json" in all_files or
                f"{content_prefix}semantic_segments.json.gz" in all_files
            )

            # Get project for task creation
            project = content.projects[0] if content.projects else 'unknown' if content.projects else 'unknown'

            # Update database flags based on file existence
            updates_needed = False

            # Update is_downloaded flag based on source files, audio existence, or storage manifest
            # If storage_manifest exists, content was compressed and source files were deleted intentionally
            should_be_downloaded = source_files_exist or audio_exists or storage_manifest_exists
            if should_be_downloaded != content.is_downloaded:
                # Log detailed state change information
                logger.info(f"State change for {content.content_id}: is_downloaded {content.is_downloaded} -> {should_be_downloaded}")
                logger.info(f"  Reason: source_files={source_files_exist}, audio={audio_exists}, manifest={storage_manifest_exists}")
                
                content.is_downloaded = should_be_downloaded
                results['flags_updated'].append('is_downloaded')
                updates_needed = True
                
                # Immediately commit and refresh for download state changes to ensure task creation works
                if should_be_downloaded:
                    content.last_updated = datetime.now(timezone.utc)
                    session.add(content)
                    session.commit()
                    session.refresh(content)
                    logger.debug(f"Immediately committed is_downloaded=True for {content.content_id}")
                else:
                    # Log warning when setting to false - this is unusual and worth investigating
                    logger.warning(f"Setting is_downloaded=False for {content.content_id} - this may trigger re-download!")

            # Update is_converted flag based on audio AND chunk audio existence
            # is_converted should be true if:
            # 1. Root audio exists AND chunk audio files exist, OR
            # 2. Root audio exists AND transcripts exist (chunks were cleaned up)
            should_be_converted = False
            if audio_exists:
                # Check if chunk audio files exist (sample first chunk)
                chunk_audio_exists = any(f"{content_prefix}chunks/0/audio.wav" in all_files for _ in [1])
                if chunk_audio_exists:
                    should_be_converted = True
                else:
                    # No chunk audio - check if transcript FILES actually exist in S3 (post-cleanup case)
                    # We check actual files, not just database status, because transcripts should survive compression
                    chunks = session.query(ContentChunk).filter_by(content_id=content.id).all()
                    if chunks:
                        # Check if any transcript files actually exist in S3
                        transcript_files_exist = False
                        for chunk in chunks[:3]:  # Sample first 3 chunks
                            transcript_path = f"{content_prefix}chunks/{chunk.chunk_index}/transcript_words.json"
                            transcript_path_gz = f"{content_prefix}chunks/{chunk.chunk_index}/transcript_words.json.gz"
                            if transcript_path in all_files or transcript_path_gz in all_files:
                                transcript_files_exist = True
                                break

                        if transcript_files_exist:
                            should_be_converted = True
                            logger.debug(f"No chunk audio but transcript files exist in S3 for {content.content_id} - assuming post-cleanup")
                        else:
                            logger.warning(f"Audio exists but no chunk audio or transcript files for {content.content_id} - conversion incomplete (transcripts lost)")
                    else:
                        logger.warning(f"Audio exists but no chunks in database for {content.content_id} - conversion incomplete")

            if should_be_converted != content.is_converted:
                # Log state change
                logger.info(f"State change for {content.content_id}: is_converted {content.is_converted} -> {should_be_converted}")
                logger.info(f"  Audio files found: {[f for f in all_files if 'audio.' in f][:5]}")

                content.is_converted = should_be_converted
                results['flags_updated'].append('is_converted')
                updates_needed = True
                # Immediately commit and refresh for convert state changes to ensure transcribe task creation works
                if should_be_converted:
                    content.last_updated = datetime.now(timezone.utc)
                    session.add(content)
                    session.commit()
                    session.refresh(content)
                    logger.debug(f"Immediately committed is_converted=True for {content.content_id}")

            # Update chunk status based on actual files
            if should_be_converted:
                # Get the existing chunk plan - we only check status, don't create
                chunk_plan = get_chunk_plan(session, content)
                
                # Update status of existing chunks based on files
                if chunk_plan:
                    for chunk in chunk_plan:
                        chunk_index = chunk['index']
                        chunk_audio = f"{content_prefix}chunks/{chunk_index}/audio.wav"
                        chunk_transcript = f"{content_prefix}chunks/{chunk_index}/transcript_words.json"
                        
                        try:
                            db_chunk = session.query(ContentChunk).filter_by(
                                content_id=content.id,
                                chunk_index=chunk_index
                            ).first()
                            
                            if db_chunk:
                                # Check transcript file existence first (used by both extraction and transcription status)
                                # Note: Transcript files are preserved during compression but may be compressed to .gz
                                chunk_transcript_gz = f"{content_prefix}chunks/{chunk_index}/transcript_words.json.gz"
                                transcript_exists = chunk_transcript in all_files or chunk_transcript_gz in all_files

                                # Update extraction status based on audio file existence
                                # IMPORTANT: Don't downgrade extraction status if content is compressed
                                # (chunk audio files are intentionally deleted during compression)
                                # EXCEPTION: If transcripts are also missing, conversion was incomplete
                                new_extraction_status = 'completed' if chunk_audio in all_files else 'pending'
                                if db_chunk.extraction_status != new_extraction_status:
                                    # Allow status changes if:
                                    # 1. Content is not compressed, OR
                                    # 2. We're upgrading status from pending to completed, OR
                                    # 3. Content is compressed BUT transcripts are missing (conversion was incomplete)
                                    allow_downgrade = (
                                        not storage_manifest_exists or
                                        (db_chunk.extraction_status == 'pending' and new_extraction_status == 'completed') or
                                        (storage_manifest_exists and not transcript_exists)
                                    )

                                    if allow_downgrade:
                                        if storage_manifest_exists and not transcript_exists and new_extraction_status == 'pending':
                                            logger.warning(f"Resetting chunk {chunk_index} extraction status to pending - " +
                                                         f"transcripts missing despite compression (conversion was incomplete)")
                                        db_chunk.extraction_status = new_extraction_status
                                        updates_needed = True
                                        logger.debug(f"Updated chunk {chunk_index} extraction status to {new_extraction_status}")
                                    else:
                                        logger.debug(f"Skipping chunk {chunk_index} extraction status downgrade " +
                                                   f"(content is compressed - chunk files intentionally deleted)")

                                # Update transcription status based on transcript file existence AND safe model check
                                # Language-aware safe model check
                                is_safe_model = False

                                # Handle legacy transcripts with no model information
                                if not db_chunk.transcribed_with and (transcript_exists or self.s3_storage.file_exists(chunk_transcript_gz)):
                                    # Try to read transcript file to verify it has valid content
                                    try:
                                        # Determine which file exists
                                        if chunk_transcript in all_files:
                                            transcript_key = chunk_transcript
                                        elif chunk_transcript_gz in all_files:
                                            transcript_key = chunk_transcript_gz
                                        elif self.s3_storage.file_exists(chunk_transcript_gz):
                                            transcript_key = chunk_transcript_gz
                                            transcript_exists = True  # Update flag for later logic
                                        elif self.s3_storage.file_exists(chunk_transcript):
                                            transcript_key = chunk_transcript
                                            transcript_exists = True  # Update flag for later logic
                                        else:
                                            transcript_key = None

                                        if not transcript_key:
                                            continue

                                        transcript_data = self.s3_storage.read_json_flexible(transcript_key)

                                        if transcript_data:
                                            # Check for valid content (either words or segments)
                                            has_words = transcript_data.get('words') and len(transcript_data.get('words', [])) > 0
                                            has_segments = transcript_data.get('segments') and len(transcript_data.get('segments', [])) > 0

                                            if has_words or has_segments:
                                                # Valid legacy transcript - populate model field
                                                db_chunk.transcribed_with = 'legacy_whisper'
                                                is_safe_model = True  # Accept legacy transcripts
                                                logger.info(f"Found legacy transcript for {content.content_id} chunk {chunk_index} - setting transcribed_with='legacy_whisper'")
                                            else:
                                                logger.warning(f"Transcript file exists for {content.content_id} chunk {chunk_index} but has no words or segments")
                                    except Exception as e:
                                        logger.error(f"Error reading transcript for {content.content_id} chunk {chunk_index}: {e}")

                                elif db_chunk.transcribed_with:
                                    # Determine language (treat main_language starting with 'en' as English)
                                    is_english = content.main_language and content.main_language.lower().startswith('en')

                                    # Helper to check if model matches safe list (handles language suffixes like _fr, _es, _de)
                                    def is_model_safe(model_name: str, safe_list: list) -> bool:
                                        if model_name in safe_list or model_name == 'legacy_whisper':
                                            return True
                                        # Strip language suffix (e.g., whisper_mlx_turbo_fr -> whisper_mlx_turbo)
                                        # Language suffixes are typically 2-3 char codes after final underscore
                                        if '_' in model_name:
                                            base_model = '_'.join(model_name.rsplit('_', 1)[:-1])
                                            suffix = model_name.rsplit('_', 1)[-1]
                                            # Only strip if suffix looks like a language code (2-3 lowercase letters)
                                            if len(suffix) in (2, 3) and suffix.isalpha() and suffix.islower():
                                                return base_model in safe_list
                                        return False

                                    # Check appropriate safe model list based on language
                                    if is_english:
                                        is_safe_model = is_model_safe(db_chunk.transcribed_with, self.safe_transcription_models_english)
                                        if not is_safe_model:
                                            logger.info(f"Model '{db_chunk.transcribed_with}' not in English safe models, marking {content.content_id} chunk {chunk_index} as pending")
                                    else:
                                        is_safe_model = is_model_safe(db_chunk.transcribed_with, self.safe_transcription_models_other)
                                        if not is_safe_model:
                                            logger.info(f"Model '{db_chunk.transcribed_with}' not in non-English safe models for {content.main_language} content, marking {content.content_id} chunk {chunk_index} as pending")

                                new_status = 'completed' if (transcript_exists and is_safe_model) else 'pending'

                                if db_chunk.transcription_status != new_status:
                                    # Log when we detect a missing transcript that was marked as completed
                                    if db_chunk.transcription_status == 'completed' and new_status == 'pending':
                                        if not transcript_exists:
                                            logger.warning(f"Transcript file missing in S3 for {content.content_id} chunk {chunk_index} " +
                                                         f"but was marked as completed in database. Resetting to pending.")
                                        elif not is_safe_model:
                                            logger.info(f"Transcript exists but model '{db_chunk.transcribed_with}' not in safe models list " +
                                                       f"for {content.content_id} chunk {chunk_index}. Keeping status as pending for re-transcription.")

                                    db_chunk.transcription_status = new_status
                                    updates_needed = True
                                    found_file = "original" if chunk_transcript in all_files else ("compressed" if chunk_transcript_gz in all_files else "missing")
                                    logger.debug(f"Updated chunk {chunk_index} transcription status to {new_status} " +
                                               f"(transcript file {found_file} in S3, model: {db_chunk.transcribed_with}, safe: {is_safe_model})")
                                    
                        except Exception as e:
                            error_msg = f"Error updating chunk {chunk_index} status: {str(e)}"
                            logger.error(error_msg)
                            results['errors'].append(error_msg)

            # Update is_diarized flag (only check diarization.json now)
            if diarization_exists != content.is_diarized:
                content.is_diarized = diarization_exists
                results['flags_updated'].append('is_diarized')
                updates_needed = True


            # Update is_transcribed flag based on chunk transcripts
            # Get all chunks and their transcription status
            chunks = session.query(ContentChunk).filter_by(content_id=content.id).all()
            all_chunks_transcribed = False
            if chunks:  # Only check if we have chunks
                all_chunks_transcribed = all(chunk.transcription_status == 'completed' for chunk in chunks)
                logger.debug(f"Checking transcription status for {content.content_id}: " +
                           f"{sum(1 for c in chunks if c.transcription_status == 'completed')} completed out of {len(chunks)} chunks")

            if all_chunks_transcribed != content.is_transcribed:
                content.is_transcribed = all_chunks_transcribed
                results['flags_updated'].append('is_transcribed')
                updates_needed = True
                logger.info(f"Updated is_transcribed to {all_chunks_transcribed} for {content.content_id} " +
                          f"(all {len(chunks)} chunks have transcripts)")
                # Immediately commit and refresh for transcribe state changes to ensure diarize/stitch task creation works
                if all_chunks_transcribed:
                    content.last_updated = datetime.now(timezone.utc)
                    session.add(content)
                    session.commit()
                    session.refresh(content)
                    logger.debug(f"Immediately committed is_transcribed=True for {content.content_id}")

            # Update is_stitched flag
            # NOTE: Stitch now produces Sentence records (not SpeakerTranscription which is deprecated)
            current_version = self.get_current_stitch_version()

            if stitched_exists or content.is_stitched:  # Check either condition to trigger verification
                # Initialize variables for debugging
                sentences_exist = False
                version_list = []

                # Log detailed stitch check
                logger.debug(f"Checking stitch state for {content.content_id}: file_exists={stitched_exists}, is_stitched={content.is_stitched}, current_version={current_version}")

                # Verify both file and database records exist - check Sentence table (primary output of stitch)
                sentence_count = session.query(Sentence).filter(
                    Sentence.content_id == content.id,
                    Sentence.stitch_version == current_version
                ).count()
                sentences_exist = sentence_count > 0

                logger.debug(f"Sentence records for {content.content_id} with version {current_version}: {sentence_count}")

                # Check what versions exist in the database
                all_versions = session.query(Sentence.stitch_version).filter(
                    Sentence.content_id == content.id
                ).distinct().all()
                version_list = [v[0] for v in all_versions] if all_versions else []

                # Only consider stitched if both conditions are met
                # However, if we just found sentences with ANY version, update to be stitched
                any_sentences_exist = len(version_list) > 0

                if stitched_exists and any_sentences_exist and not sentences_exist:
                    # File exists and there are sentences, but with a different version
                    logger.info(f"Found sentences for {content.content_id} with version(s) {version_list}, " +
                              f"but current version is '{current_version}'. Content needs re-stitching.")
                    should_be_stitched = False  # Need to re-stitch with current version
                else:
                    # Only consider stitched if BOTH file exists AND database records exist with current version
                    should_be_stitched = stitched_exists and sentences_exist

                # Log detailed stitch state for debugging
                logger.debug(f"Stitch state for {content.content_id}: " +
                           f"file_exists={stitched_exists}, db_records={sentences_exist}, " +
                           f"current_version='{current_version}', db_versions={version_list}, " +
                           f"should_be_stitched={should_be_stitched}, current_is_stitched={content.is_stitched}")

                # Update stitch_version if content is stitched and version is outdated
                if should_be_stitched and should_recreate_stitch_task(current_version, content.stitch_version):
                    content.stitch_version = current_version
                    results['flags_updated'].append('stitch_version')
                    updates_needed = True
                    # Immediately commit version update to prevent recreating stitch task
                    content.last_updated = datetime.now(timezone.utc)
                    session.add(content)
                    session.commit()
                    session.refresh(content)
                    logger.debug(f"Immediately committed stitch_version={current_version} for {content.content_id}")

                if should_be_stitched != content.is_stitched:
                    content.is_stitched = should_be_stitched
                    content.stitch_version = current_version if should_be_stitched else 'none'
                    results['flags_updated'].append('is_stitched')
                    updates_needed = True
                    # Immediately commit and refresh for stitch state changes to ensure segment task creation works
                    if should_be_stitched:
                        content.last_updated = datetime.now(timezone.utc)
                        session.add(content)
                        session.commit()
                        session.refresh(content)
                        logger.debug(f"Immediately committed is_stitched=True for {content.content_id}")

                    if stitched_exists and not sentences_exist:
                        logger.warning(f"Found speaker_turns.json but no Sentence records in database for {content.content_id} " +
                                     f"with version {current_version}. Will create stitch task.")
                        # Create stitch task to fix this if content is ready
                        # The stitch task now handles speaker identification internally
                        # Only requires transcription and diarization.json
                        if content.is_transcribed and diarization_exists:
                            task_id, block_reason = await self._create_task_if_not_exists(
                                session,
                                content_id=content.content_id,
                                task_type='stitch',
                                input_data={'project': project},
                                content=content
                            )
                            if task_id:
                                results['tasks_created'].append('stitch')
                            elif block_reason:
                                results['tasks_blocked'].append(f"stitch: {block_reason}")

            # Update is_embedded flag based on semantic_segments.json existence
            # Check current segment version from meta_data
            current_segment_version = get_current_segment_version()
            content_segment_version = content.meta_data.get('segment_version') if content.meta_data else None
            
            # For segment task completion, also verify database records exist
            if semantic_segments_exist:
                # Verify EmbeddingSegment records exist with correct version
                embedding_segments_exist = session.query(EmbeddingSegment).filter(
                    EmbeddingSegment.content_id == content.id
                ).count() > 0
                
                # Content is embedded if file exists AND database records exist
                should_be_embedded = semantic_segments_exist and embedding_segments_exist
            else:
                should_be_embedded = False
            
            if should_be_embedded != content.is_embedded:
                content.is_embedded = should_be_embedded
                # Update segment_version in meta_data if embedded
                if should_be_embedded:
                    meta_data = dict(content.meta_data) if content.meta_data else {}
                    meta_data['segment_version'] = current_segment_version
                    content.meta_data = meta_data
                results['flags_updated'].append('is_embedded')
                updates_needed = True
                # Immediately commit and refresh for segment state changes to prevent duplicate task creation
                if semantic_segments_exist:
                    content.last_updated = datetime.now(timezone.utc)
                    session.add(content)
                    session.commit()
                    session.refresh(content)
                    logger.debug(f"Immediately committed is_embedded=True and segment_version={current_segment_version} for {content.content_id}")

            # Update is_compressed flag based on storage_manifest.json existence
            if storage_manifest_exists != content.is_compressed:
                content.is_compressed = storage_manifest_exists
                results['flags_updated'].append('is_compressed')
                updates_needed = True

            # Commit all updates
            if updates_needed:
                content.last_updated = datetime.now(timezone.utc)
                session.add(content)
                session.commit()
                session.refresh(content)
                logger.info(f"Updated content state for {content.content_id}: {', '.join(results['flags_updated'])}")

            # Ensure we have the latest content state after all updates
            session.refresh(content)
            
            # Create any missing tasks based on reconciled state
            logger.debug(f"Evaluating next tasks for {content.content_id} - State: " + 
                        f"downloaded={content.is_downloaded}, converted={content.is_converted}, " +
                        f"transcribed={content.is_transcribed}, diarized={content.is_diarized}, " +
                        f"stitched={content.is_stitched}, stitch_version={content.stitch_version}")

            # Prepare detailed status message
            if not updates_needed:
                if db_task and hasattr(db_task, 'task_type') and db_task.task_type == 'transcribe':
                    chunk_index = db_task.input_data.get('chunk_index') if db_task.input_data else None
                    # Get list of pending chunks
                    pending_chunks = [str(c.chunk_index) for c in chunks if c.transcription_status != 'completed']
                    if pending_chunks:
                        results['outcome_message'] = f"Updated status of chunk {chunk_index}, waiting for chunks {','.join(pending_chunks)}."
                    else:
                        results['outcome_message'] = f"Updated status of chunk {chunk_index}, all chunks complete."
                else:
                    results['outcome_message'] = "No changes needed"
            else:
                updates = []
                if results['flags_updated']:
                    updates.append(f"Updated flags: {', '.join(results['flags_updated'])}")
                if results['tasks_created']:
                    updates.append(f"Created tasks: {', '.join(results['tasks_created'])}")
                if results['tasks_blocked']:
                    updates.append(f"Blocked tasks: {', '.join(results['tasks_blocked'])}")
                results['outcome_message'] = '. '.join(updates) if updates else "No changes needed"

            # Create any missing tasks based on reconciled state
            # First check if content is within project date range
            if not self._is_content_within_project_date_range(content, project):
                logger.debug(f"Skipping task creation for {content.content_id} - outside project {project} date range (publish_date: {content.publish_date})")
                return results
            
            if not source_files_exist and not audio_exists and not content.blocked_download:
                # Check if there are multiple completed download tasks but no files
                # This indicates the download script is failing silently
                completed_download_tasks = session.query(TaskQueue).filter(
                    TaskQueue.content_id == content.content_id,
                    TaskQueue.task_type == f"download_{content.platform}",
                    TaskQueue.status == 'completed'
                ).count()
                
                # If we have 3+ completed download tasks but no files, block further attempts
                if completed_download_tasks >= 3:
                    logger.warning(f"Content {content.content_id} has {completed_download_tasks} completed download tasks but no files in S3. "
                                 f"This indicates silent download failures. Blocking further download attempts.")
                    # Mark content as blocked to prevent infinite download loops
                    content.blocked_download = True
                    content.last_updated = datetime.now(timezone.utc)
                    session.add(content)
                    session.commit()
                    results['flags_updated'].append('blocked_download')
                    results['tasks_blocked'].append(f"download_{content.platform}: Silent failure after {completed_download_tasks} attempts")
                else:
                    # Check if there's a failed download task with conda run error
                    failed_download_task = session.query(TaskQueue).filter(
                        TaskQueue.content_id == content.content_id,
                        TaskQueue.task_type == f"download_{content.platform}",
                        TaskQueue.status == 'failed'
                    ).first()
                    
                    # Skip creating new task if it failed with conda run error
                    if failed_download_task and failed_download_task.error and "conda run" in str(failed_download_task.error) and "failed" in str(failed_download_task.error):
                        logger.debug(f"Skipping recreation of download task for {content.content_id} due to conda run failure")
                    else:
                        task_id, block_reason = await self._create_task_if_not_exists(
                            session,
                            content_id=content.content_id,
                            task_type=f"download_{content.platform}",
                            input_data={'project': project},
                            content=content
                        )
                        if task_id:
                            results['tasks_created'].append(f"download_{content.platform}")
                        elif block_reason:
                            results['tasks_blocked'].append(f"download_{content.platform}: {block_reason}")

            if source_files_exist and not audio_exists:
                # Check if there's a failed convert task - don't recreate download if convert failed
                failed_convert_task = session.query(TaskQueue).filter(
                    TaskQueue.content_id == content.content_id,
                    TaskQueue.task_type == 'convert',
                    TaskQueue.status == 'failed'
                ).first()

                # If convert failed, only recreate convert task, not download task
                if failed_convert_task:
                    logger.debug(f"Convert task failed for {content.content_id}, will recreate convert task only")

                task_id, block_reason = await self._create_task_if_not_exists(
                    session,
                    content_id=content.content_id,
                    task_type='convert',
                    input_data={'project': project},
                    content=content
                )
                if task_id:
                    results['tasks_created'].append('convert')
                elif block_reason:
                    results['tasks_blocked'].append(f"convert: {block_reason}")

            # Also create convert task if audio exists but conversion is incomplete
            # (e.g., audio.opus exists but chunks weren't created)
            if audio_exists and not should_be_converted:
                logger.info(f"Audio exists but conversion incomplete for {content.content_id} - creating convert task")
                task_id, block_reason = await self._create_task_if_not_exists(
                    session,
                    content_id=content.content_id,
                    task_type='convert',
                    input_data={'project': project},
                    content=content
                )
                if task_id:
                    results['tasks_created'].append('convert')
                elif block_reason:
                    results['tasks_blocked'].append(f"convert: {block_reason}")

            if should_be_converted:
                # Create diarize task first if needed (runs as soon as audio is converted)
                if not diarization_exists:
                    # Immediate fix: skip diarize if diarization_ignored is set in meta_data
                    if content.meta_data and content.meta_data.get('diarization_ignored'):
                        logger.info(f"Skipping diarize task for {content.content_id} due to diarization_ignored flag.")
                    else:
                        task_id, block_reason = await self._create_task_if_not_exists(
                            session,
                            content_id=content.content_id,
                            task_type='diarize',
                            input_data={'project': project},
                            content=content
                        )
                        if task_id:
                            results['tasks_created'].append('diarize')
                        elif block_reason:
                            results['tasks_blocked'].append(f"diarize: {block_reason}")

                # Create transcribe tasks for chunks without transcripts (requires diarize to be complete)
                if diarization_exists or (content.meta_data and content.meta_data.get('diarization_ignored')):
                    chunk_files = {f for f in all_files if '/chunks/' in f}
                    chunk_pattern = re.compile(f"{content_prefix}chunks/(\\d+)/audio\\.wav")

                    for chunk_file in chunk_files:
                        if match := chunk_pattern.match(chunk_file):
                            chunk_index = int(match.group(1))
                            chunk_transcript = f"{content_prefix}chunks/{chunk_index}/transcript_words.json"
                            chunk_transcript_gz = f"{content_prefix}chunks/{chunk_index}/transcript_words.json.gz"

                            if chunk_transcript not in all_files and chunk_transcript_gz not in all_files:
                                # Check if there's a failed transcribe task with "Failed to download audio chunk" error
                                failed_chunk_task = session.query(TaskQueue).filter(
                                    TaskQueue.content_id == content.content_id,
                                    TaskQueue.task_type == 'transcribe',
                                    TaskQueue.status == 'failed',
                                    text("input_data->>'chunk_index' = :chunk_index").params(chunk_index=str(chunk_index))
                                ).first()

                                # Skip creating new task if it failed with audio chunk download error
                                if failed_chunk_task and failed_chunk_task.error and "Failed to download audio chunk" in str(failed_chunk_task.error):
                                    logger.debug(f"Skipping recreation of transcribe task for chunk {chunk_index} due to audio download failure")
                                    continue

                                task_id, block_reason = await self._create_task_if_not_exists(
                                    session,
                                    content_id=content.content_id,
                                    task_type='transcribe',
                                    input_data={
                                        'project': project,
                                        'chunk_index': chunk_index
                                    },
                                    content=content
                                )
                                if task_id:
                                    results['tasks_created'].append(f'transcribe_{chunk_index}')
                                elif block_reason:
                                    results['tasks_blocked'].append(f"transcribe_{chunk_index}: {block_reason}")
                else:
                    logger.debug(f"Skipping transcribe task creation for {content.content_id} - waiting for diarize to complete")

                # Check for incomplete chunk extraction - if is_transcribed=False but audio exists,
                # we may have missing chunks that weren't extracted properly
                if not content.is_transcribed:
                    # Get chunk plan from database to see expected chunks
                    chunks = session.query(ContentChunk).filter_by(content_id=content.id).all()
                    if chunks:
                        # Check if any chunks are missing extraction
                        missing_extraction = [c for c in chunks if c.extraction_status != 'completed']
                        if missing_extraction:
                            chunk_indices = [c.chunk_index for c in missing_extraction]
                            logger.info(f"Content {content.content_id} has missing chunk extractions: {chunk_indices} - will recreate convert task")

                            # Create convert task to re-extract all chunks properly
                            task_id, block_reason = await self._create_task_if_not_exists(
                                session,
                                content_id=content.content_id,
                                task_type='convert',
                                input_data={'project': project},
                                content=content
                            )
                            if task_id:
                                results['tasks_created'].append('convert')
                                logger.info(f"Created convert task {task_id} to re-extract missing chunks for {content.content_id}")
                            elif block_reason:
                                results['tasks_blocked'].append(f"convert: {block_reason}")

                # Create stitch task if needed
                # The stitch task now integrates identify_speakers (Step 6) and creates speaker_mapping.json
                # It also generates speaker_embeddings.json from the diarization data
                # It requires diarization to be complete (diarization.json exists)
                needs_stitch = (
                    all_chunks_transcribed and 
                    diarization_exists and  # Only requires diarization.json now
                    (not content.is_stitched or should_recreate_stitch_task(current_version, content.stitch_version))
                )
                
                # Log detailed stitch task decision
                logger.debug(f"Stitch task decision for {content.content_id}: " +
                           f"all_chunks_transcribed={all_chunks_transcribed}, " +
                           f"diarization_exists={diarization_exists}, " +
                           f"is_stitched={content.is_stitched}, " +
                           f"stitch_version={content.stitch_version}, " +
                           f"current_version={current_version}, " +
                           f"needs_stitch={needs_stitch}")
                
                if needs_stitch:
                    task_id, block_reason = await self._create_task_if_not_exists(
                        session,
                        content_id=content.content_id,
                        task_type='stitch',
                        input_data={'project': project},
                        content=content
                    )
                    if task_id:
                        results['tasks_created'].append('stitch')
                    elif block_reason:
                        results['tasks_blocked'].append(f"stitch: {block_reason}")
                
                # NOTE: Segmentation is now integrated into stitch pipeline as Stage 13 (stitch_v13+)
                # Standalone segment tasks are NO LONGER CREATED - segments are created by stitch
                # If content is missing segments but is stitched, re-run stitch task instead
                
                # Create cleanup task if needed
                # Cleanup task runs after stitch is complete to compress files and clean up storage
                # Use database flags instead of storage_manifest check since manifest persists after resets
                needs_cleanup = (
                    content.is_stitched and
                    not content.is_compressed
                )

                logger.debug(f"Cleanup task check for {content.content_id}: " +
                           f"is_stitched={content.is_stitched}, " +
                           f"is_compressed={content.is_compressed}, " +
                           f"needs_cleanup={needs_cleanup}")

                if needs_cleanup:
                    task_id, block_reason = await self._create_task_if_not_exists(
                        session,
                        content_id=content.content_id,
                        task_type='cleanup',
                        input_data={'project': project},
                        content=content
                    )
                    if task_id:
                        results['tasks_created'].append('cleanup')
                    elif block_reason:
                        results['tasks_blocked'].append(f"cleanup: {block_reason}")
                
            # Commit any tasks that were created
            if results['tasks_created']:
                session.commit()
                logger.debug(f"Committed {len(results['tasks_created'])} tasks for {content.content_id}")

            return results

        except Exception as e:
            import traceback
            error_msg = f"Error reconciling content state: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
            results['errors'].append(error_msg)
            return results

    async def bulk_reconcile_content_states(self, session: Session) -> Dict[str, Any]:
        """
        Efficiently reconcile database state with S3/MinIO reality for all content.
        This method skips task creation and focuses on state reconciliation.
        
        Args:
            session: Active database session
            
        Returns:
            Dict containing summary of updates made
        """
        logger.info("Starting bulk content state reconciliation")
        
        results = {
            'total_content': 0,
            'updated_content': 0,
            'flag_updates': defaultdict(int),
            'errors': [],
            'processing_time': 0
        }
        
        start_time = datetime.now()
        
        try:
            # Step 1: Build file index from MinIO
            logger.info("Building file index from MinIO storage...")
            file_index = defaultdict(set)  # content_id -> set of files
            
            # List all objects in content/ prefix
            objects = self.minio_client.list_objects(
                self.bucket_name,
                prefix='content/',
                recursive=True
            )
            
            # Index files by content_id
            for obj in objects:
                path_parts = obj.object_name.split('/')
                if len(path_parts) >= 2:
                    content_id = path_parts[1]
                    file_index[content_id].add(obj.object_name)
            
            logger.info(f"Found files for {len(file_index)} content items")
            
            # Step 2: Get all content from database
            all_content = session.query(Content).all()
            results['total_content'] = len(all_content)
            
            # Step 3: Batch update content flags
            updates = []
            current_version = self.get_current_stitch_version()
            
            for content in all_content:
                try:
                    content_files = file_index.get(content.content_id, set())
                    content_prefix = f"content/{content.content_id}/"
                    
                    # Check file existence (considering compression)
                    storage_manifest_exists = f"{content_prefix}storage_manifest.json" in content_files
                    
                    # For audio, check both original WAV and compressed formats
                    audio_exists = (
                        f"{content_prefix}audio.wav" in content_files or 
                        f"{content_prefix}audio.opus" in content_files or
                        f"{content_prefix}audio.mp3" in content_files
                    )
                    
                    # For source files, check both original and compressed formats
                    source_files_exist = (
                        any(f"{content_prefix}source{ext}" in content_files 
                            for ext in ['.mp4', '.mp3', '.wav', '.m4a']) or
                        any(f"{content_prefix}video{ext}" in content_files 
                            for ext in ['.mp4', '.webm'])
                    )
                    
                    # For JSON files, check both original and compressed formats
                    diarization_exists = (
                        f"{content_prefix}diarization.json" in content_files or
                        f"{content_prefix}diarization.json.gz" in content_files
                    )
                    
                    # Note: speaker_embeddings.json is now generated by stitch, not diarize
                    embeddings_exist = (
                        f"{content_prefix}speaker_embeddings.json" in content_files or
                        f"{content_prefix}speaker_embeddings.json.gz" in content_files
                    )
                    
                    speaker_mapping_exists = (
                        f"{content_prefix}speaker_mapping.json" in content_files or
                        f"{content_prefix}speaker_mapping.json.gz" in content_files
                    )
                    
                    # Stitch creates transcript_diarized.json
                    stitched_exists = (
                        f"{content_prefix}transcript_diarized.json" in content_files or
                        f"{content_prefix}transcript_diarized.json.gz" in content_files
                    )
                    
                    semantic_segments_exist = (
                        f"{content_prefix}semantic_segments.json" in content_files or
                        f"{content_prefix}semantic_segments.json.gz" in content_files
                    )
                    
                    # Calculate chunk transcription status
                    chunk_files = {f for f in content_files if '/chunks/' in f}
                    chunk_pattern = re.compile(f"{content_prefix}chunks/(\\d+)/transcript_words\\.json(\\.gz)?")
                    completed_chunks = set()
                    
                    for file in chunk_files:
                        if match := chunk_pattern.match(file):
                            completed_chunks.add(int(match.group(1)))
                    
                    # Get total chunks from database
                    total_chunks = session.query(ContentChunk).filter_by(content_id=content.id).count()
                    all_chunks_transcribed = (
                        total_chunks > 0 and 
                        len(completed_chunks) == total_chunks
                    )
                    
                    # Check if updates needed
                    updates_needed = False
                    new_flags = {
                        'is_downloaded': source_files_exist or audio_exists,
                        'is_converted': audio_exists,
                        'is_diarized': diarization_exists,  # Only check diarization.json now
                        'is_transcribed': all_chunks_transcribed,
                        'is_stitched': stitched_exists,
                        'is_embedded': semantic_segments_exist,
                        'is_compressed': storage_manifest_exists
                    }
                    
                    # Compare with current flags
                    current_flags = {
                        'is_downloaded': content.is_downloaded,
                        'is_converted': content.is_converted,
                        'is_diarized': content.is_diarized,
                        'is_transcribed': content.is_transcribed,
                        'is_stitched': content.is_stitched,
                        'is_embedded': content.is_embedded,
                        'is_compressed': content.is_compressed
                    }
                    
                    # Track changes
                    for flag, new_value in new_flags.items():
                        if current_flags[flag] != new_value:
                            updates_needed = True
                            results['flag_updates'][flag] += 1
                    
                    if updates_needed:
                        # Update content object
                        for flag, value in new_flags.items():
                            setattr(content, flag, value)
                        
                        if content.is_stitched:
                            content.stitch_version = current_version
                            
                        content.last_updated = datetime.now(timezone.utc)
                        updates.append(content)
                        results['updated_content'] += 1
                        
                        if len(updates) >= 1000:  # Batch size
                            session.bulk_save_objects(updates)
                            session.commit()
                            
                except Exception as e:
                    error_msg = f"Error processing content {content.content_id}: {str(e)}"
                    logger.error(error_msg)
                    results['errors'].append(error_msg)
            
            # Commit any remaining updates
            if updates:
                session.bulk_save_objects(updates)
                session.commit()
            
            end_time = datetime.now()
            results['processing_time'] = (end_time - start_time).total_seconds()
            
            logger.info(
                f"Bulk reconciliation complete: "
                f"Processed {results['total_content']} content items, "
                f"Updated {results['updated_content']} items in {results['processing_time']:.2f} seconds"
            )
            
            return results
            
        except Exception as e:
            error_msg = f"Error during bulk reconciliation: {str(e)}"
            logger.error(error_msg, exc_info=True)
            results['errors'].append(error_msg)
            return results

