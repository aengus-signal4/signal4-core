"""
Task management models for the distributed processing system.

Contains:
- TaskQueue: Distributed task queue for processing
- WorkerConfig: Worker node configuration
"""

from sqlalchemy import (
    Column, Integer, String, DateTime, Text, Index, ARRAY, text, UniqueConstraint, JSON
)
from datetime import datetime

from .base import Base


class TaskQueue(Base):
    """
    Distributed task queue for coordinating processing across workers.

    Tasks represent units of work (download, transcribe, etc.) that are
    claimed and executed by worker nodes. The queue ensures each task
    is processed exactly once and tracks execution status.

    Attributes:
        id: Primary key
        task_type: Type of task ('download', 'transcribe', 'diarize', etc.)
        content_id: Content ID this task operates on
        status: Task status ('pending', 'processing', 'completed', 'failed')
        worker_id: ID of worker currently processing this task
        processor_task_id: Task ID from the processor API
        input_data: JSON input parameters for the task
        result: JSON result data from task execution
        error: Error message if task failed
        priority: Task priority (higher = more urgent)
        created_at: When the task was created
        started_at: When processing started
        completed_at: When processing completed
        last_heartbeat: Last heartbeat from processing worker
        attempts: Number of execution attempts
        max_attempts: Maximum attempts before giving up

    Unique Constraint:
        (content_id, task_type) - Only one task per type per content

    Task Lifecycle:
        1. Created as 'pending' with input_data
        2. Worker claims task, status -> 'processing'
        3. Worker sends heartbeats during execution
        4. On success: status -> 'completed', result populated
        5. On failure: status -> 'failed', error populated
    """
    __tablename__ = 'task_queue'

    id = Column(Integer, primary_key=True)
    task_type = Column(String(50), nullable=False)
    content_id = Column(String(255), nullable=False)
    status = Column(String(20), default='pending', nullable=False)
    worker_id = Column(String(100))
    processor_task_id = Column(String(255), nullable=True)
    input_data = Column(JSON, nullable=False, default=dict)  # Ensure never NULL
    result = Column(JSON)
    error = Column(Text)
    priority = Column(Integer, default=0)  # Added priority field
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow)
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    last_heartbeat = Column(DateTime(timezone=True))
    attempts = Column(Integer, default=0)
    max_attempts = Column(Integer, default=3)

    # Indexes
    __table_args__ = (
        Index('idx_task_queue_status', 'status'),
        Index('idx_task_queue_content_id', 'content_id'),
        Index('idx_task_queue_task_type', 'task_type'),
        # Composite index for task claiming with priority ordering
        Index('idx_task_queue_claim',
              'status', 'task_type', 'priority', 'created_at',
              postgresql_where=text("status = 'pending'")),
        # Add unique constraint for content_id and task_type
        UniqueConstraint('content_id', 'task_type', name='uq_task_queue_content_id_task_type'), # Ensure unique constraint name
        {'schema': 'tasks'}  # Use tasks schema
    )


class WorkerConfig(Base):
    """
    Configuration for worker nodes in the distributed system.

    Each worker machine registers its capabilities and the orchestrator
    uses this information to assign appropriate tasks.

    Attributes:
        id: Primary key
        hostname: Unique hostname of the worker
        enabled_tasks: List of task types this worker can handle
        max_concurrent_tasks: Maximum parallel tasks
        last_heartbeat: Last time worker checked in
        status: Worker status ('active', 'inactive', 'disabled')

    Note: This table uses the 'tasks' schema, not the default schema.
    """
    __tablename__ = 'worker_config'
    __table_args__ = (
        {'schema': 'tasks'}  # Use tasks schema
    )

    id = Column(Integer, primary_key=True)
    hostname = Column(String(255), nullable=False, unique=True)
    enabled_tasks = Column(ARRAY(String), nullable=False)
    max_concurrent_tasks = Column(Integer, default=1)
    last_heartbeat = Column(DateTime(timezone=True))
    status = Column(String(20), default='active')

    # Indexes
    __table_args__ = (
        Index('idx_worker_config_hostname', 'hostname'),
        Index('idx_worker_config_status', 'status'),
        {'schema': 'tasks'}  # Use tasks schema
    )
