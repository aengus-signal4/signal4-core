"""
Centralized error handling for the content processing pipeline.

This module provides the ErrorHandler class that implements policy-based
error handling based on error codes and task types.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session

from src.database.models import Content, TaskQueue, ContentChunk
from src.utils.error_codes import ErrorCode, get_error_category
from src.utils.logger import setup_worker_logger

logger = setup_worker_logger('error_handler')


class ErrorHandler:
    """Centralized error handler for processing pipeline."""
    
    def __init__(self, config: Dict[str, Any], worker_task_failures: Dict = None):
        """
        Initialize the error handler.
        
        Args:
            config: Application configuration dictionary
            worker_task_failures: Reference to orchestrator's failure tracking
        """
        self.config = config
        self.worker_task_failures = worker_task_failures or {}
        
        # Load error handling configuration
        self.error_config = config.get('error_handling', {})
        self.policies = self.error_config.get('policies', {})
        self.task_overrides = self.error_config.get('task_overrides', {})
        
        # Get timeouts and limits from config
        self.youtube_auth_pause_timeout = config.get('processing', {}).get(
            'youtube_auth_pause_duration_seconds', 50400  # 14 hours
        )
        self.max_consecutive_failures = config.get('processing', {}).get(
            'max_consecutive_failures', 3
        )
        
        logger.info("ErrorHandler initialized with policy-based error handling")
    
    def handle_error(self,
                     task_type: str,
                     error_code: str,
                     error_message: str,
                     error_details: Dict[str, Any],
                     content: Content,
                     session: Session,
                     worker_id: str = None) -> Dict[str, Any]:
        """
        Handle an error based on error code and task type.
        
        Args:
            task_type: Type of task that failed
            error_code: Error code from ErrorCode enum
            error_message: Human-readable error message
            error_details: Additional error context
            content: Content object
            session: Database session
            worker_id: Worker ID (for auth errors)
            
        Returns:
            Dict containing:
                - action: What action to take
                - skip_state_audit: Whether to skip state reconciliation
                - outcome_message: Message describing action taken
                - available_after: When worker is available (for auth errors)
                - additional task-specific data
        """
        logger.info(f"Handling error for {task_type} task: {error_code} - {error_message}")
        
        # Get policy for this error (with task-specific override)
        policy = self._get_policy(task_type, error_code)
        
        # If no specific policy, use category-based defaults
        if not policy:
            policy = self._get_default_policy(error_code)
        
        # Execute action based on policy
        action = policy.get('action', 'continue')
        
        if action == 'recreate_task':
            return self._handle_recreate_task(policy, error_message)
            
        elif action == 'block_content':
            return self._handle_block_content(
                content, session, policy, error_message, error_details
            )
            
        elif action == 'trigger_prerequisites':
            return self._handle_trigger_prerequisites(
                content, session, policy, error_code, error_message
            )
            
        elif action == 'pause_worker':
            return self._handle_pause_worker(
                worker_id, task_type, policy, error_message
            )
            
        elif action == 'mark_ignored':
            return self._handle_mark_ignored(
                content, session, policy, error_message, error_details
            )
            
        elif action == 'skip_state_audit':
            return {
                'action': 'continue',
                'skip_state_audit': True,
                'outcome_message': policy.get('message', error_message)
            }
            
        else:
            # Default: continue with normal processing
            return {
                'action': 'continue',
                'skip_state_audit': False,
                'outcome_message': f"Error handled: {error_message}"
            }
    
    def _get_policy(self, task_type: str, error_code: str) -> Dict[str, Any]:
        """Get error handling policy with task-specific overrides."""
        # Check for task-specific override first
        task_policies = self.task_overrides.get(task_type, {})
        if error_code in task_policies:
            return task_policies[error_code]
        
        # Fall back to general policy
        return self.policies.get(error_code, {})
    
    def _get_default_policy(self, error_code: str) -> Dict[str, Any]:
        """Get default policy based on error category."""
        try:
            error_enum = ErrorCode(error_code)
            category = get_error_category(error_enum)
            
            if category == 'transient':
                return {'action': 'recreate_task', 'max_retries': 3}
            elif category == 'permanent':
                return {'action': 'skip_state_audit', 'permanent': True}
            elif category == 'auth':
                return {'action': 'pause_worker'}
            elif category == 'missing_deps':
                return {'action': 'trigger_prerequisites'}
            elif category == 'content_blocked':
                return {'action': 'block_content'}
            elif category == 'special':
                return {'action': 'mark_ignored'}
            else:
                return {'action': 'continue'}
                
        except ValueError:
            # Unknown error code
            logger.warning(f"Unknown error code: {error_code}")
            return {'action': 'continue'}
    
    def _handle_recreate_task(self, policy: Dict, error_message: str) -> Dict[str, Any]:
        """Handle transient errors by marking task for recreation."""
        max_retries = policy.get('max_retries', 3)
        
        return {
            'action': 'recreate_task',
            'skip_state_audit': True,
            'outcome_message': f"Transient error, will retry: {error_message}",
            'max_retries': max_retries
        }
    
    def _handle_block_content(self,
                             content: Content,
                             session: Session,
                             policy: Dict,
                             error_message: str,
                             error_details: Dict) -> Dict[str, Any]:
        """Handle permanent failures by blocking content."""
        # Block the content
        content.blocked_download = True
        
        # Update metadata
        meta_data = dict(content.meta_data) if content.meta_data else {}
        meta_data.update({
            'block_reason': policy.get('block_reason', error_message),
            'permanent_block': 'true',
            'blocked_at': datetime.now(timezone.utc).isoformat(),
            'error_details': error_details
        })
        content.meta_data = meta_data
        content.last_updated = datetime.now(timezone.utc)
        
        session.add(content)
        session.commit()
        
        logger.warning(f"Content {content.content_id} blocked: {error_message}")
        
        return {
            'action': 'block_content',
            'skip_state_audit': True,
            'outcome_message': f"Content blocked: {policy.get('block_reason', error_message)}"
        }
    
    def _handle_trigger_prerequisites(self,
                                    content: Content,
                                    session: Session,
                                    policy: Dict,
                                    error_code: str,
                                    error_message: str) -> Dict[str, Any]:
        """Handle missing dependencies by triggering prerequisite tasks."""
        # Map error codes to state flags that should be reset
        state_resets = {
            'missing_audio': ['is_converted', 'is_transcribed', 'is_stitched'],
            'missing_source': ['is_downloaded', 'is_converted', 'is_transcribed', 'is_stitched'],
            'missing_transcript': ['is_transcribed', 'is_stitched'],
            'missing_diarization': ['is_diarized', 'is_stitched'],
            'missing_chunks': ['is_transcribed', 'is_stitched']
        }

        # Special handling for missing_audio: only reset if transcripts don't exist
        # (since compress/cleanup deletes audio after transcription)
        flags_to_reset = state_resets.get(error_code, [])
        if error_code == 'missing_audio':
            # Check if transcripts exist in database
            chunks = session.query(ContentChunk).filter_by(content_id=content.id).all()
            transcripts_exist = any(chunk.transcription_status == 'completed' for chunk in chunks) if chunks else False

            if transcripts_exist:
                logger.info(f"Missing audio for {content.content_id} but transcripts exist - skipping state reset (likely post-cleanup)")
                return {
                    'action': 'continue',
                    'skip_state_audit': True,
                    'outcome_message': f"Audio missing but transcripts exist (post-cleanup) - skipping task recreation"
                }
            else:
                logger.info(f"Missing audio for {content.content_id} and no transcripts - will reset state to trigger convert")

        # Reset appropriate state flags
        for flag in flags_to_reset:
            setattr(content, flag, False)

        content.last_updated = datetime.now(timezone.utc)
        session.add(content)
        session.commit()

        logger.info(f"Reset state flags {flags_to_reset} for {content.content_id} due to {error_code}")

        return {
            'action': 'trigger_prerequisites',
            'skip_state_audit': False,  # Let state evaluation create missing tasks
            'outcome_message': f"Missing prerequisites detected: {error_message}. State reset for task creation.",
            'prerequisites': policy.get('prerequisites', [])
        }
    
    def _handle_pause_worker(self,
                           worker_id: str,
                           task_type: str,
                           policy: Dict,
                           error_message: str) -> Dict[str, Any]:
        """Handle authentication errors by pausing worker."""
        if not worker_id:
            logger.warning("Cannot pause worker - no worker_id provided")
            return {
                'action': 'continue',
                'skip_state_audit': True,
                'outcome_message': "Auth error but no worker to pause"
            }
        
        pause_duration = policy.get('pause_duration', self.youtube_auth_pause_timeout)
        
        # Update worker failure tracking
        now = datetime.now()
        if worker_id not in self.worker_task_failures:
            self.worker_task_failures[worker_id] = {}
        if task_type not in self.worker_task_failures[worker_id]:
            self.worker_task_failures[worker_id][task_type] = {}
        
        self.worker_task_failures[worker_id][task_type].update({
            'count': self.max_consecutive_failures,
            'last_failure_time': now,
            'paused': True,
            'pause_timeout': pause_duration
        })
        
        logger.error(f"Worker {worker_id} paused for {task_type} tasks for {pause_duration}s: {error_message}")
        
        return {
            'action': 'pause_worker',
            'skip_state_audit': True,
            'outcome_message': f"Auth error. Worker paused for {pause_duration}s",
            'available_after': None  # Pipeline manager will calculate this
        }
    
    def _handle_mark_ignored(self,
                           content: Content,
                           session: Session,
                           policy: Dict,
                           error_message: str,
                           error_details: Dict) -> Dict[str, Any]:
        """Handle special results by marking content as ignored."""
        # Update metadata
        meta_data = dict(content.meta_data) if content.meta_data else {}
        
        ignore_key = policy.get('metadata_key', 'processing_ignored')
        ignore_reason = policy.get('ignored_reason', error_message)
        
        meta_data[ignore_key] = True
        meta_data[f'{ignore_key}_reason'] = ignore_reason
        meta_data[f'{ignore_key}_at'] = datetime.now(timezone.utc).isoformat()
        
        if error_details:
            meta_data[f'{ignore_key}_details'] = error_details
        
        content.meta_data = meta_data
        content.last_updated = datetime.now(timezone.utc)
        
        session.add(content)
        session.commit()
        
        logger.info(f"Content {content.content_id} marked as ignored: {ignore_reason}")
        
        return {
            'action': 'mark_ignored',
            'skip_state_audit': True,
            'outcome_message': f"Content marked as ignored: {ignore_reason}"
        }
    
    def should_recreate_task(self, task_type: str, error_code: str) -> bool:
        """
        Determine if a task should be recreated based on error.
        
        This is used by pipeline manager for backward compatibility.
        """
        policy = self._get_policy(task_type, error_code)
        if not policy:
            policy = self._get_default_policy(error_code)
        
        return policy.get('action') == 'recreate_task'