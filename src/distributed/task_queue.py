import asyncio
import socket
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Any
import psycopg2
from psycopg2.extras import Json
from ..utils.logger import setup_worker_logger, get_worker_ip
from ..utils.node_utils import is_head_node
from ..database.session import Session, get_connection
from ..database.manager import DatabaseManager
from ..database.models import TaskQueue, WorkerConfig
from sqlalchemy import or_, func, and_
import os
import time
import yaml
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
import math

def sanitize_json(data: Any) -> Any:
    """Sanitize JSON data by replacing NaN values with null"""
    if isinstance(data, float) and math.isnan(data):
        return None
    elif isinstance(data, dict):
        return {k: sanitize_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_json(item) for item in data]
    return data

class TaskQueueManager:
    def __init__(self, db_config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """Initialize task queue manager with database configuration"""
        # Ensure proper environment setup
        self._setup_environment()
        
        self.db_config = db_config.copy()  # Make a copy to avoid modifying the original
        self.logger = logger or setup_worker_logger('task_queue')
        
        # Get worker IP and determine if we're on head node
        self.worker_ip = get_worker_ip()
        self._is_head_node = is_head_node()

        # For head node, always use localhost
        if self._is_head_node:
            self.db_config['host'] = 'localhost'
        else:
            # When not on head node, use head node IP from config
            self.db_config['host'] = '10.0.0.4'  # Head node IP
            # Add connection timeout and retry settings
            self.db_config['connect_timeout'] = 10
            self.db_config['keepalives'] = 1
            self.db_config['keepalives_idle'] = 30
            self.db_config['keepalives_interval'] = 10
            self.db_config['keepalives_count'] = 5
            
        self.logger.info(f"Task queue manager connecting to database at {self.db_config['host']}")
        
        # Use the database session module instead of creating our own engine
        try:
            self.Session = Session()._session_factory
            self.logger.info("Using shared database session factory")
        except Exception as e:
            self.logger.error(f"Error initializing database session: {str(e)}")
            # Try to establish connection directly to verify connectivity
            self._test_db_connection()
            # Re-raise the exception after logging
            raise

        self.schema = db_config['schemas']['task_queue']
        self.worker_id = f"{self.worker_ip}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.heartbeat_interval = 30  # seconds
        self.task_timeout = db_config['cleanup']['stale_task_timeout']
        self._stop_requested = False
        self._heartbeat_task = None

    def _setup_environment(self):
        """Set up the environment variables and paths"""
        import os
        import sys
        
        # Set up base paths
        conda_path = "/opt/homebrew/Caskroom/miniforge/base"
        conda_env = "content-processing"
        remote_dir = "/Users/signal4/content_processing"
        
        # Set environment variables
        os.environ['CONDA_PATH'] = conda_path
        os.environ['CONDA_ENV'] = conda_env
        os.environ['REMOTE_DIR'] = remote_dir
        
        # Update PYTHONPATH
        if remote_dir not in sys.path:
            sys.path.insert(0, remote_dir)
            os.environ['PYTHONPATH'] = f"{remote_dir}:{os.environ.get('PYTHONPATH', '')}"

    def connect(self):
        """Create a new database connection using the shared session factory"""
        return get_connection()  # Use the shared connection function

    async def register_worker(self, enabled_tasks: List[str], max_concurrent: int = 1) -> bool:
        """Register this worker in the worker_config table"""
        try:
            self.logger.info(f"Attempting to register worker {self.worker_ip} with tasks: {enabled_tasks}")
            
            # Get a session from the shared factory
            session = self.Session()
            
            try:
                # Log connection details
                self.logger.info(f"Using database connection: {self.db_config['host']}:{self.db_config['port']}")
                
                # Test the connection first
                try:
                    session.execute(text("SELECT 1"))
                    self.logger.info("Database connection test successful")
                except Exception as test_e:
                    self.logger.error(f"Database connection test failed: {str(test_e)}")
                    raise
                
                # Upsert worker configuration
                worker_config = session.query(WorkerConfig).filter(WorkerConfig.hostname == self.worker_ip).one_or_none()
                if worker_config:
                    self.logger.info(f"Updating existing worker configuration for {self.worker_ip}")
                    worker_config.enabled_tasks = enabled_tasks
                    worker_config.max_concurrent_tasks = max_concurrent
                    worker_config.last_heartbeat = datetime.now(timezone.utc)
                    worker_config.status = 'active'
                else:
                    self.logger.info(f"Creating new worker configuration for {self.worker_ip}")
                    worker_config = WorkerConfig(
                        hostname=self.worker_ip,
                        enabled_tasks=enabled_tasks,
                        max_concurrent_tasks=max_concurrent,
                        last_heartbeat=datetime.now(timezone.utc),
                        status='active'
                    )
                    session.add(worker_config)
                
                self.logger.info("Committing worker configuration to database...")
                session.commit()
                self.logger.info(f"Successfully registered worker {self.worker_ip}")
                return True
            except Exception as inner_e:
                self.logger.error(f"Database error during worker registration: {str(inner_e)}")
                raise
            finally:
                session.close()
                
        except Exception as e:
            self.logger.error(f"Error registering worker: {str(e)}")
            self.logger.error(f"Worker IP: {self.worker_ip}")
            self.logger.error(f"Enabled tasks: {enabled_tasks}")
            self.logger.error(f"Database config: {self.db_config}")
            return False

    async def start_heartbeat(self):
        """Start the heartbeat task"""
        async def heartbeat_loop():
            while not self._stop_requested:
                try:
                    session = self.Session()
                    try:
                        worker_config = session.query(WorkerConfig).filter(WorkerConfig.hostname == self.worker_ip).one_or_none()
                        if worker_config:
                            worker_config.last_heartbeat = datetime.now(timezone.utc)
                            session.commit()
                    finally:
                        session.close()
                    await asyncio.sleep(self.heartbeat_interval)
                except Exception as e:
                    self.logger.error(f"Error in heartbeat: {str(e)}")
                    await asyncio.sleep(5)  # Short sleep on error

        self._heartbeat_task = asyncio.create_task(heartbeat_loop())

    async def stop_heartbeat(self):
        """Stop the heartbeat task"""
        if self._heartbeat_task:
            self._stop_requested = True
            await self._heartbeat_task
            self._heartbeat_task = None

    async def claim_task(self, worker_id: str, enabled_tasks: List[str]) -> Optional[Dict]:
        """Attempt to claim an available task. Tasks are ordered by priority, content_id, and chunk_index."""
        try:
            session = self.Session()
            try:
                self.logger.debug(f"Worker {worker_id} attempting to claim task for types: {enabled_tasks}")
                
                # Get highest priority unclaimed task that matches our capabilities
                task = session.query(TaskQueue).filter(
                    and_(
                        TaskQueue.status == 'pending',
                        TaskQueue.task_type.in_(enabled_tasks),
                        or_(
                            TaskQueue.worker_id.is_(None),
                            TaskQueue.worker_id == worker_id
                        )
                    )
                ).order_by(
                    TaskQueue.priority.desc(),  # Highest priority first
                    TaskQueue.content_id.asc(),  # Group by content
                    text("CAST(input_data->>'chunk_index' AS INTEGER) ASC")  # Order chunks within content
                ).first()
                
                if task:
                    # Update task status and worker info
                    task.status = 'processing'
                    task.worker_id = worker_id
                    task.started_at = datetime.now(timezone.utc)
                    session.commit()
                    
                    # Convert task to dict for return
                    task_dict = {
                        'id': task.id,
                        'content_id': task.content_id,
                        'type': task.task_type,
                        'input_data': task.input_data or {}
                    }
                    
                    self.logger.info(f"Claimed task {task.id} ({task.task_type})")
                    return task_dict
                else:
                    self.logger.debug(f"No pending tasks found for types: {enabled_tasks}")
                    return None
                    
            finally:
                session.close()
                
        except Exception as e:
            self.logger.error(f"Error claiming task: {str(e)}", exc_info=True)
            return None

    async def complete_task(self, task_id: int, result: Dict, status: str = 'done') -> bool:
        """Mark a task as complete with results"""
        try:
            session = self.Session()
            try:
                task = session.query(TaskQueue).filter_by(id=task_id).first()
                if task:
                    task.status = status
                    task.result = result
                    task.completed_at = datetime.now(timezone.utc)
                    session.commit()
                    self.logger.info(f"Completed task {task_id} with status {status}")
                    return True
                return False
            finally:
                session.close()
                
        except Exception as e:
            self.logger.error(f"Error completing task: {str(e)}")
            return False

    async def fail_task(self, task_id: int, error: str) -> bool:
        """Mark a task as failed with error message"""
        try:
            session = self.Session()
            try:
                task = session.query(TaskQueue).filter_by(id=task_id).first()
                if task:
                    task.status = 'error'
                    task.result = {'error': error}
                    task.completed_at = datetime.now(timezone.utc)
                    session.commit()
                    self.logger.error(f"Task {task_id} failed: {error}")
                    return True
                return False
            finally:
                session.close()
                
        except Exception as e:
            self.logger.error(f"Error marking task as failed: {str(e)}")
            return False

    async def update_task_status(self, task_id: str, status: str, result: Optional[Dict] = None, error: Optional[str] = None) -> None:
        """Update the status of a task in the queue."""
        try:
            session = self.Session()
            try:
                task = session.query(TaskQueue).filter(TaskQueue.id == task_id).one_or_none()
                if not task:
                    self.logger.error(f"Task {task_id} not found")
                    return

                # Update task status
                task.status = status
                task.completed_at = datetime.now(timezone.utc) if status in ('done', 'error') else None
                # Sanitize result data before storing
                task.result = sanitize_json(result) if result else None
                task.error = error
                task.last_heartbeat = datetime.now(timezone.utc)

                session.commit()
            finally:
                session.close()
        except Exception as e:
            self.logger.error(f"Error updating task status: {e}")
            raise

    async def cleanup_stale_tasks(self):
        """Reset stale in_progress tasks to pending"""
        try:
            session = self.Session()
            try:
                # Get stale tasks
                stale_tasks = session.query(TaskQueue) \
                    .filter(TaskQueue.status == 'in_progress') \
                    .filter(TaskQueue.last_heartbeat < datetime.now(timezone.utc) - timedelta(seconds=self.task_timeout)) \
                    .filter(or_(
                        TaskQueue.max_attempts.is_(None),
                        TaskQueue.attempts < TaskQueue.max_attempts
                    )) \
                    .all()
                
                for task in stale_tasks:
                    # Reset task status without modifying content status
                    task.status = 'pending'
                    task.worker_id = None
                    task.error = Json({
                        'message': 'Task reset due to stale worker',
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'attempt': task.attempts
                    })
                
                session.commit()
                
                if stale_tasks:
                    self.logger.info(f"Reset {len(stale_tasks)} stale tasks to pending")
            finally:
                session.close()
        except Exception as e:
            self.logger.error(f"Error cleaning up stale tasks: {str(e)}")

    async def add_tasks(self, tasks: List[Dict]) -> List[int]:
        """Add multiple tasks to the queue in a single transaction"""
        try:
            session = self.Session()
            try:
                # Build lists of task identifiers
                self.logger.debug(f"Processing {len(tasks)} tasks")
                task_identifiers = [(t['task_type'], t['content_id']) for t in tasks]
                task_types = [t[0] for t in task_identifiers]
                content_ids = [t[1] for t in task_identifiers]

                # First clean up stale tasks
                stale_timeout = datetime.now(timezone.utc) - timedelta(seconds=self.task_timeout)
                stale_count = session.query(TaskQueue) \
                    .filter(and_(
                        TaskQueue.task_type.in_(task_types),
                        TaskQueue.content_id.in_(content_ids),
                        TaskQueue.status == 'processing',
                        or_(
                            TaskQueue.last_heartbeat < stale_timeout,
                            TaskQueue.last_heartbeat.is_(None)
                        )
                    )).update({
                        'status': 'pending',
                        'worker_id': None,
                        'attempts': 0,
                        'error': Json({
                            'message': 'Task reset due to stale worker',
                            'timestamp': datetime.now(timezone.utc).isoformat()
                        })
                    }, synchronize_session=False)

                if stale_count > 0:
                    self.logger.info(f"Reset {stale_count} stale tasks to pending")

                # Clean up completed/failed tasks
                cleanup_count = session.query(TaskQueue) \
                    .filter(and_(
                        TaskQueue.task_type.in_(task_types),
                        TaskQueue.content_id.in_(content_ids),
                        TaskQueue.status.in_(('done', 'error'))
                    )).delete(synchronize_session=False)
                
                if cleanup_count > 0:
                    self.logger.debug(f"Cleaned up {cleanup_count} completed/failed tasks")
                
                # Get all existing tasks in a single query
                existing_tasks = session.query(TaskQueue) \
                    .filter(and_(
                        TaskQueue.task_type.in_(task_types),
                        TaskQueue.content_id.in_(content_ids)
                    )).all()
                
                # Reset failed tasks and track existing ones
                reset_count = 0
                existing_pairs = set()
                for task in existing_tasks:
                    task_pair = (task.task_type, task.content_id)
                    existing_pairs.add(task_pair)
                    
                    # Reset task if it's failed or stale
                    if (task.status == 'error' and task.attempts < task.max_attempts) or \
                       (task.status == 'processing' and (
                           task.last_heartbeat is None or 
                           task.last_heartbeat < stale_timeout
                       )):
                        task.status = 'pending'
                        task.worker_id = None
                        task.attempts = 0
                        task.error = None
                        reset_count += 1
                
                if reset_count > 0:
                    self.logger.debug(f"Reset {reset_count} failed/stale tasks to pending")
                
                # Create new tasks only for non-existing ones
                new_tasks = []
                task_type_counts = {}
                for task_data in tasks:
                    try:
                        if (task_data['task_type'], task_data['content_id']) not in existing_pairs:
                            # Ensure input_data is a dict
                            input_data = task_data.get('input_data', {})
                            if input_data is None:
                                input_data = {}
                                
                            # For transcription tasks, ensure project is set
                            if task_data['task_type'] == 'transcribe' and 'project' not in input_data:
                                input_data['project'] = 'default'
                                
                            self.logger.debug(f"Creating task: type={task_data['task_type']}, content_id={task_data['content_id']}, input_data={input_data}")
                            task = TaskQueue(
                                task_type=task_data['task_type'],
                                content_id=task_data['content_id'],
                                input_data=input_data,
                                priority=task_data.get('priority', 0),
                                max_attempts=3,
                                status='pending',
                                attempts=0,
                                created_at=datetime.now(timezone.utc)
                            )
                            new_tasks.append(task)
                            
                            # Track counts by task type
                            task_type = task_data['task_type']
                            task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1
                    except Exception as task_error:
                        self.logger.error(f"Error creating task for {task_data.get('content_id')}: {str(task_error)}")
                        continue
                
                # Log a single summary for all new tasks
                if task_type_counts:
                    summary_parts = []
                    for task_type, count in task_type_counts.items():
                        summary_parts.append(f"{count} {task_type}")
                    summary = ", ".join(summary_parts)
                    self.logger.info(f"Adding new tasks: {summary}")
                    
                    try:
                        # Bulk insert new tasks with ON CONFLICT DO NOTHING
                        session.execute(
                            text("""
                                INSERT INTO tasks.task_queue 
                                (task_type, content_id, input_data, priority, max_attempts, status, attempts, created_at)
                                VALUES (:task_type, :content_id, :input_data, :priority, :max_attempts, :status, :attempts, :created_at)
                                ON CONFLICT (content_id, task_type) DO NOTHING
                            """),
                            [{
                                'task_type': task.task_type,
                                'content_id': task.content_id,
                                'input_data': task.input_data,
                                'priority': task.priority,
                                'max_attempts': task.max_attempts,
                                'status': task.status,
                                'attempts': task.attempts,
                                'created_at': task.created_at
                            } for task in new_tasks]
                        )
                        session.commit()
                        
                        # Refresh tasks to get their IDs
                        for task in new_tasks:
                            session.refresh(task)
                            
                        self.logger.info(f"Successfully added {len(new_tasks)} new tasks")
                    except Exception as bulk_error:
                        self.logger.error(f"Error during bulk insert: {str(bulk_error)}")
                        session.rollback()
                        raise
                
                # Return all task IDs (new and existing)
                task_ids = [task.id for task in new_tasks]
                task_ids.extend(task.id for task in existing_tasks)
                
                self.logger.debug(f"Returning {len(task_ids)} task IDs")
                return task_ids
            finally:
                session.close()
        except Exception as e:
            self.logger.error(f"Error adding tasks in batch: {str(e)}")
            if hasattr(e, '__cause__'):
                self.logger.error(f"Caused by: {str(e.__cause__)}")
            return []

    async def add_task(self, task_type: str, content_id: str, input_data: Dict, priority: int = 0) -> Optional[int]:
        """Add a single task to the queue"""
        task_ids = await self.add_tasks([{
            'task_type': task_type,
            'content_id': content_id,
            'input_data': input_data,
            'priority': priority
        }])
        return task_ids[0] if task_ids else None

    async def get_task_status(self, task_id: int) -> Optional[Dict]:
        """Get current status of a task"""
        try:
            session = self.Session()
            try:
                task = session.query(TaskQueue).filter(TaskQueue.id == task_id).one_or_none()
                if task:
                    return {
                        'status': task.status,
                        'result': task.result,
                        'error': task.error,
                        'created_at': task.created_at,
                        'started_at': task.started_at,
                        'completed_at': task.completed_at
                    }
                return None
            finally:
                session.close()
        except Exception as e:
            self.logger.error(f"Error getting task status: {str(e)}")
            return None

    async def get_worker_stats(self) -> Dict:
        """Get statistics for worker tasks"""
        try:
            with self.Session() as session:
                results = session.query(
                    TaskQueue.worker_id,
                    func.count().label('total_tasks'),
                    func.count().filter(TaskQueue.status == 'done').label('completed_tasks'),
                    func.count().filter(TaskQueue.status == 'error').label('failed_tasks'),
                    func.avg(func.extract('EPOCH', TaskQueue.completed_at - TaskQueue.started_at)) \
                        .filter(TaskQueue.status == 'done') \
                        .label('avg_processing_time')
                ).filter(TaskQueue.worker_id.isnot(None)) \
                .group_by(TaskQueue.worker_id) \
                .all()
                
                stats = {}
                for row in results:
                    worker_id = row[0]
                    stats[worker_id] = {
                        'total_tasks': row[1],
                        'completed_tasks': row[2],
                        'failed_tasks': row[3],
                        'avg_processing_time': row[4]
                    }
                return stats
                
        except Exception as e:
            self.logger.error(f"Error getting worker stats: {str(e)}")
            return {}

    async def deregister_worker(self, worker_id: str) -> bool:
        """Deregister a worker from the worker_config table"""
        try:
            session = self.Session()
            try:
                # Delete worker configuration
                result = session.query(WorkerConfig).filter(
                    WorkerConfig.hostname == worker_id.split('-')[0]  # Extract hostname from worker_id
                ).delete()
                
                session.commit()
                
                if result > 0:
                    self.logger.info(f"Deregistered worker {worker_id}")
                    return True
                else:
                    self.logger.warning(f"No worker found to deregister: {worker_id}")
                    return False
            finally:
                session.close()
                
        except Exception as e:
            self.logger.error(f"Error deregistering worker: {str(e)}")
            return False

    async def get_pending_tasks(self) -> List[Dict]:
        """Get all pending tasks ordered by priority"""
        try:
            session = self.Session()
            try:
                # Query pending tasks ordered by priority and creation time
                tasks = session.query(TaskQueue)\
                    .filter(TaskQueue.status == 'pending')\
                    .order_by(TaskQueue.priority.desc(), TaskQueue.created_at.asc())\
                    .all()
                
                # Convert tasks to dictionaries
                return [{
                    'id': task.id,
                    'type': task.task_type,
                    'content_id': task.content_id,
                    'input_data': task.input_data,
                    'priority': task.priority
                } for task in tasks]
                
            finally:
                session.close()
                
        except Exception as e:
            self.logger.error(f"Error getting pending tasks: {str(e)}")
            return []

    async def add_transcription_tasks(self, content_ids: List[str], priority: int = 0) -> List[int]:
        """Add transcription tasks for content IDs"""
        tasks = []
        for content_id in content_ids:
            tasks.append({
                'task_type': 'transcribe',
                'content_id': content_id,
                'input_data': {},
                'priority': priority
            })
        return await self.add_tasks(tasks)

    async def add_stitching_task(self, content_id: str, priority: int = 0) -> Optional[int]:
        """Add a stitching task for a content ID"""
        task = {
            'task_type': 'stitch_transcripts',
            'content_id': content_id,
            'input_data': {},
            'priority': priority
        }
        task_ids = await self.add_tasks([task])
        return task_ids[0] if task_ids else None

    async def add_stitching_tasks(self, content_ids: List[str], priority: int = 0) -> List[int]:
        """Add stitching tasks for content IDs"""
        tasks = []
        for content_id in content_ids:
            tasks.append({
                'task_type': 'stitch_transcripts',
                'content_id': content_id,
                'input_data': {},
                'priority': priority
            })
        return await self.add_tasks(tasks)

    async def return_task(self, task_id: str) -> bool:
        """Return a task to the queue when it can't be processed yet"""
        try:
            with get_session() as session:
                task = session.query(TaskQueue).filter_by(id=task_id).first()
                if task:
                    task.status = 'pending'
                    task.worker_id = None
                    task.started_at = None
                    task.last_heartbeat = None
                    session.commit()
                    return True
                return False
        except Exception as e:
            self.logger.error(f"Error returning task {task_id} to queue: {str(e)}")
            return False

    def _test_db_connection(self):
        """Test database connectivity and log detailed information"""
        try:
            import psycopg2
            self.logger.info("Testing direct database connection...")
            
            conn = psycopg2.connect(
                dbname=self.db_config['database'],
                user=self.db_config['user'],
                password=self.db_config['password'],
                host=self.db_config['host'],
                port=self.db_config['port'],
                connect_timeout=10
            )
            
            with conn.cursor() as cur:
                cur.execute("SELECT version();")
                version = cur.fetchone()[0]
                self.logger.info(f"Successfully connected to PostgreSQL. Version: {version}")
                
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Database connection test failed: {str(e)}")
            self.logger.error(f"Connection details: host={self.db_config['host']}, port={self.db_config['port']}, database={self.db_config['database']}, user={self.db_config['user']}")
            # Try to get network information
            try:
                import subprocess
                result = subprocess.run(['ping', '-c', '1', self.db_config['host']], capture_output=True, text=True)
                self.logger.error(f"Ping test to {self.db_config['host']}: {result.stdout}")
            except Exception as ping_e:
                self.logger.error(f"Ping test failed: {str(ping_e)}") 