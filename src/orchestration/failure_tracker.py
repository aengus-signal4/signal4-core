"""
Failure Tracker - Tracks worker failures by task type and manages pausing
"""
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

class FailureTracker:
    """Tracks worker failures and manages task type pausing"""
    
    def __init__(self):
        # Failure tracking: worker_id -> task_type -> failure_info
        self.worker_task_failures: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(
            lambda: defaultdict(lambda: {
                'count': 0,
                'last_failure_time': None,
                'paused': False,
                'pause_until': None
            })
        )
        
        # Configuration
        self.max_consecutive_failures = 3  # Failures before pausing
        self.failure_reset_timeout = 600  # 10 minutes before resetting failures
        self.pause_duration = 1800  # 30 minutes pause duration
        self.youtube_auth_pause_duration = 50400  # 14 hours for YouTube auth failures (was 86400 * 10 = 10 days)
        
    def record_failure(self, worker_id: str, task_type: str, error_message: str = "",
                       error_code: Optional[str] = None, is_permanent: bool = False) -> bool:
        """
        Record a task failure for a worker/task_type combination.
        Returns True if task type should be paused for this worker.

        Args:
            worker_id: The worker that failed
            task_type: The type of task that failed
            error_message: Human-readable error message
            error_code: Optional error code from ErrorCode enum
            is_permanent: Whether this is marked as a permanent content failure
        """
        now = datetime.now(timezone.utc)

        # Check if this is a permanent content-level error (not a worker issue)
        # These should NOT count toward worker pausing
        if is_permanent or self._is_permanent_content_error(error_message, error_code):
            logger.info(f"Permanent content error for worker {worker_id} task type {task_type} "
                       f"(not counting toward worker pause): {error_message}")
            return False

        failure_info = self.worker_task_failures[worker_id][task_type]

        # Reset count if enough time has passed since last failure
        if (failure_info['last_failure_time'] and
            (now - failure_info['last_failure_time']).total_seconds() > self.failure_reset_timeout):
            failure_info['count'] = 0
            logger.debug(f"Reset failure count for worker {worker_id} task type {task_type}")

        # Record the failure
        failure_info['count'] += 1
        failure_info['last_failure_time'] = now

        logger.warning(f"Recorded failure #{failure_info['count']} for worker {worker_id} task type {task_type}: {error_message}")
        
        # Check if we should pause this task type
        if failure_info['count'] >= self.max_consecutive_failures:
            # Determine pause duration based on error type
            pause_duration = self.pause_duration
            
            # Special handling for YouTube authentication errors
            if self._is_youtube_auth_error(error_message):
                pause_duration = self.youtube_auth_pause_duration
                logger.error(f"YouTube authentication error detected for worker {worker_id}. "
                           f"Pausing download_youtube tasks for {pause_duration/3600:.1f} hours")
            
            failure_info['paused'] = True
            failure_info['pause_until'] = now + timedelta(seconds=pause_duration)
            
            logger.warning(f"Pausing task type {task_type} for worker {worker_id} "
                         f"until {failure_info['pause_until']} due to {failure_info['count']} consecutive failures")
            
            return True
        
        return False
    
    def record_success(self, worker_id: str, task_type: str):
        """Record a successful task completion, resetting failure count"""
        if worker_id in self.worker_task_failures and task_type in self.worker_task_failures[worker_id]:
            failure_info = self.worker_task_failures[worker_id][task_type]
            if failure_info['count'] > 0:
                logger.debug(f"Resetting failure count for worker {worker_id} task type {task_type} after success")
                failure_info['count'] = 0
                failure_info['last_failure_time'] = None
                failure_info['paused'] = False
                failure_info['pause_until'] = None
    
    def is_task_type_paused(self, worker_id: str, task_type: str) -> bool:
        """Check if a task type is currently paused for a worker"""
        if worker_id not in self.worker_task_failures:
            return False
        
        failure_info = self.worker_task_failures[worker_id].get(task_type, {})
        
        if not failure_info.get('paused', False):
            return False
        
        # Check if pause has expired
        pause_until = failure_info.get('pause_until')
        if pause_until and datetime.now(timezone.utc) >= pause_until:
            # Pause has expired, reset
            failure_info['paused'] = False
            failure_info['pause_until'] = None
            failure_info['count'] = 0
            logger.info(f"Pause expired for worker {worker_id} task type {task_type}")
            return False
        
        return True
    
    def reset_failures(self, worker_id: Optional[str] = None, task_type: Optional[str] = None):
        """
        Reset failure tracking.
        If worker_id is None, reset for all workers.
        If task_type is None, reset for all task types for the specified worker(s).
        """
        if worker_id is None:
            # Reset all failures
            self.worker_task_failures.clear()
            logger.info("Reset all worker-task failure tracking")
            return
        
        if worker_id not in self.worker_task_failures:
            logger.debug(f"No failure tracking data found for worker {worker_id}")
            return
        
        if task_type is None:
            # Reset all task types for the worker
            self.worker_task_failures[worker_id].clear()
            logger.info(f"Reset all task type failures for worker {worker_id}")
            return
        
        # Reset specific worker-task combination
        if task_type in self.worker_task_failures[worker_id]:
            self.worker_task_failures[worker_id][task_type] = {
                'count': 0,
                'last_failure_time': None,
                'paused': False,
                'pause_until': None
            }
            logger.info(f"Reset failure tracking for worker {worker_id} task type {task_type}")
        else:
            logger.debug(f"No failure tracking data found for worker {worker_id}, task type {task_type}")
    
    def get_failure_info(self, worker_id: str, task_type: str) -> Dict[str, Any]:
        """Get failure information for a worker/task_type combination"""
        if worker_id not in self.worker_task_failures:
            return {'count': 0, 'paused': False}
        
        failure_info = self.worker_task_failures[worker_id].get(task_type, {})
        return {
            'count': failure_info.get('count', 0),
            'last_failure_time': failure_info.get('last_failure_time'),
            'paused': failure_info.get('paused', False),
            'pause_until': failure_info.get('pause_until')
        }
    
    def get_all_failures(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Get all failure tracking data"""
        return dict(self.worker_task_failures)
    
    def cleanup_expired_pauses(self):
        """Clean up expired pauses"""
        now = datetime.now(timezone.utc)
        cleaned_count = 0
        
        for worker_id, task_types in self.worker_task_failures.items():
            for task_type, failure_info in task_types.items():
                if (failure_info.get('paused', False) and 
                    failure_info.get('pause_until') and
                    now >= failure_info['pause_until']):
                    
                    failure_info['paused'] = False
                    failure_info['pause_until'] = None
                    failure_info['count'] = 0
                    cleaned_count += 1
                    logger.info(f"Cleaned up expired pause for worker {worker_id} task type {task_type}")
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} expired pauses")
    
    def _is_youtube_auth_error(self, error_message: str) -> bool:
        """Check if error message indicates YouTube authentication failure"""
        auth_error_keywords = [
            "Sign in to confirm you're not a bot",
            "This helps protect our community",
            "authentication",
            "sign in",
            "login required",
            "access denied",
            "forbidden",
            "401",
            "403"
        ]

        error_lower = error_message.lower()
        return any(keyword in error_lower for keyword in auth_error_keywords)

    def _is_permanent_content_error(self, error_message: str, error_code: Optional[str] = None) -> bool:
        """
        Check if error indicates a permanent content-level failure.

        These errors are specific to individual content items (bad URLs, deleted content, etc.)
        and should NOT trigger worker-level pausing since other content can still be processed.
        """
        # Check by error code first (most reliable)
        permanent_error_codes = [
            'not_found',
            'bad_url',
            'content_gone',
            'feed_disabled',
            'video_unavailable',
            'access_denied',
            'age_restricted',
            'members_only',
            'private_content',
            'corrupt_media',
            'invalid_format',
            'unsupported_format',
        ]

        if error_code and error_code in permanent_error_codes:
            return True

        # Fallback: check error message for HTTP status codes indicating content issues
        permanent_error_patterns = [
            "HTTP Error 400",
            "HTTP Error 404",
            "HTTP Error 410",
            "400: Bad Request",
            "404: Not Found",
            "410: Gone",
            "not found",
            "content no longer available",
            "content unavailable",
            "video unavailable",
            "bad url",
        ]

        error_lower = error_message.lower()
        return any(pattern.lower() in error_lower for pattern in permanent_error_patterns)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get failure tracking statistics"""
        now = datetime.now(timezone.utc)
        
        total_failures = 0
        paused_workers = {}
        active_pauses = 0
        
        for worker_id, task_types in self.worker_task_failures.items():
            worker_pauses = []
            for task_type, failure_info in task_types.items():
                total_failures += failure_info.get('count', 0)
                
                if failure_info.get('paused', False):
                    pause_until = failure_info.get('pause_until')
                    if pause_until and now < pause_until:
                        worker_pauses.append({
                            'task_type': task_type,
                            'pause_until': pause_until.isoformat(),
                            'remaining_seconds': (pause_until - now).total_seconds()
                        })
                        active_pauses += 1
            
            if worker_pauses:
                paused_workers[worker_id] = worker_pauses
        
        return {
            'total_tracked_failures': total_failures,
            'active_pauses': active_pauses,
            'paused_workers': paused_workers,
            'max_consecutive_failures': self.max_consecutive_failures,
            'failure_reset_timeout': self.failure_reset_timeout,
            'pause_duration': self.pause_duration
        }