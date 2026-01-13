"""
Timeout Manager - Handles task timeouts and cancellation
"""
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

class TaskTimeoutManager:
    """Manages task timeouts and automatic cancellation"""
    
    def __init__(self, worker_manager=None):
        # Default timeout thresholds in seconds
        # Note: stitch tasks excluded - they legitimately take a long time
        self.timeouts = {
            'download_youtube': 15*60,  # 15 minutes
            'download_podcast': 6*60*60,  # 6 hours (includes exponential backoff for 429 rate limiting)
            'download_rumble': 10*60,   # 10 minutes (large files)
            'convert': 20*60,          # 20 minutes
            'transcribe': 5*60,        # 5 minutes
            'diarize': 35*60,          # 35 minutes
            'segment': 10*60           # 10 minutes
            # 'stitch' intentionally excluded - can run 1+ hours for long content
        }
        self.check_interval = 60  # Check every minute
        self.worker_manager = worker_manager
        self.should_stop = False
        self._monitor_task = None

    def get_timeout(self, task_type: str) -> int:
        """Get timeout for a task type."""
        return self.timeouts.get(task_type, 1800)  # Default 30 minutes

    def update_timeout(self, task_type: str, timeout: int):
        """Update timeout for a task type."""
        self.timeouts[task_type] = timeout
        logger.info(f"Updated timeout for {task_type} to {timeout} seconds")

    async def check_timed_out_tasks(self, session: Session) -> List[Dict[str, Any]]:
        """Check for and identify timed out tasks."""
        try:
            now = datetime.now(timezone.utc)
            timed_out_tasks = []
            
            for task_type, timeout in self.timeouts.items():
                timeout_threshold = now - timedelta(seconds=timeout)
                
                # Find tasks that have been processing too long
                query = text("""
                    SELECT id, content_id, task_type, worker_id, started_at
                    FROM tasks.task_queue 
                    WHERE status = 'processing' 
                    AND task_type = :task_type
                    AND started_at < :timeout_threshold
                """)
                
                results = session.execute(query, {
                    'task_type': task_type,
                    'timeout_threshold': timeout_threshold
                }).fetchall()
                
                for task in results:
                    timed_out_tasks.append({
                        'id': task.id,
                        'content_id': task.content_id,
                        'task_type': task.task_type,
                        'worker_id': task.worker_id,
                        'started_at': task.started_at,
                        'timeout': timeout
                    })
            
            if timed_out_tasks:
                logger.warning(f"Found {len(timed_out_tasks)} timed out tasks")
            
            return timed_out_tasks
            
        except Exception as e:
            logger.error(f"Error checking timed out tasks: {str(e)}")
            return []

    async def handle_timed_out_task(self, session: Session, task_info: Dict[str, Any]) -> bool:
        """Handle a single timed out task by attempting cancellation"""
        task_id = task_info['id']
        worker_id = task_info['worker_id']
        task_type = task_info['task_type']
        
        try:
            logger.warning(f"Handling timed out task {task_id} on worker {worker_id}")
            
            # Attempt to cancel the task via worker manager
            cancel_success = False
            if self.worker_manager:
                cancel_success = await self.worker_manager.cancel_task(worker_id, str(task_id))
            
            if cancel_success:
                # Update task status to failed with timeout reason
                session.execute(
                    text("""
                        UPDATE tasks.task_queue
                        SET status = 'failed',
                            error = 'Task timed out',
                            completed_at = NOW()
                        WHERE id = :task_id
                    """),
                    {"task_id": task_id}
                )
                logger.info(f"Successfully cancelled timed out task {task_id}")
                return True
            else:
                # If cancel fails, reset task to pending
                session.execute(
                    text("""
                        UPDATE tasks.task_queue
                        SET status = 'pending',
                            worker_id = NULL,
                            error = 'Task reset after timeout',
                            started_at = NULL
                        WHERE id = :task_id
                    """),
                    {"task_id": task_id}
                )
                logger.info(f"Reset timed out task {task_id} to pending")
                return False
                
        except Exception as e:
            logger.error(f"Error handling timed out task {task_id}: {str(e)}")
            return False

    async def start_monitoring(self, get_session_func):
        """Start the timeout monitoring loop"""
        self.should_stop = False
        self._monitor_task = asyncio.create_task(self._monitor_loop(get_session_func))
        logger.info("Started timeout monitoring")

    async def stop_monitoring(self):
        """Stop the timeout monitoring loop"""
        self.should_stop = True
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Stopped timeout monitoring")

    async def _monitor_loop(self, get_session_func):
        """Main monitoring loop"""
        while not self.should_stop:
            try:
                with get_session_func() as session:
                    # Get timed out tasks
                    timed_out_tasks = await self.check_timed_out_tasks(session)
                    
                    # Handle each timed out task
                    for task_info in timed_out_tasks:
                        try:
                            await self.handle_timed_out_task(session, task_info)
                            session.commit()
                        except Exception as e:
                            session.rollback()
                            logger.error(f"Error handling timed out task {task_info['id']}: {str(e)}")
                            continue
                            
            except Exception as e:
                logger.error(f"Error in timeout monitoring loop: {str(e)}")
                
            # Wait before next check
            try:
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break

    def get_stats(self) -> Dict[str, Any]:
        """Get timeout manager statistics"""
        return {
            'timeouts_configured': dict(self.timeouts),
            'check_interval': self.check_interval,
            'monitoring_active': self._monitor_task is not None and not self._monitor_task.done()
        }