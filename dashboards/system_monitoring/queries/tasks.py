"""
Task queue database queries for the system monitoring dashboard.

This module contains optimized queries for task queue status and throughput.
Key optimization: get_completed_tasks_with_duration uses SQL aggregation
to pre-compute chunk durations instead of N+1 ORM queries.
"""

from datetime import datetime, timezone, timedelta

import pandas as pd
import streamlit as st
from sqlalchemy import text, func, and_

from src.database.session import get_session
from src.database.models import TaskQueue
from src.utils.logger import setup_worker_logger

logger = setup_worker_logger('system_monitoring')


@st.cache_data(ttl=30)
def get_task_stats() -> dict:
    """Get statistics about tasks including content hours"""
    with get_session() as session:
        result = session.execute(text("""
            SELECT
                tq.task_type,
                CASE
                    WHEN tq.status = 'done' THEN 'completed'
                    WHEN tq.status = 'error' THEN 'failed'
                    ELSE tq.status
                END as status,
                COUNT(*) as count,
                COALESCE(SUM(c.duration), 0) / 3600.0 as total_hours
            FROM tasks.task_queue tq
            LEFT JOIN content c ON tq.content_id = c.content_id
            GROUP BY
                tq.task_type,
                CASE
                    WHEN tq.status = 'done' THEN 'completed'
                    WHEN tq.status = 'error' THEN 'failed'
                    ELSE tq.status
                END
        """)).fetchall()

        stats = {}
        for row in result:
            if row.task_type == 'transcribe':
                estimated_hours = (row.count * 5) / 60.0
            else:
                estimated_hours = float(row.total_hours) if row.total_hours else 0.0

            if row.task_type not in stats:
                stats[row.task_type] = {}
            stats[row.task_type][row.status.lower()] = {
                'count': row.count,
                'hours': estimated_hours
            }

        return stats


@st.cache_data(ttl=30)
def get_recent_throughput(minutes=60) -> dict:
    """Get content hours processed by task type in the last N minutes"""
    with get_session() as session:
        result = session.execute(text(f"""
            SELECT
                tq.task_type,
                COALESCE(SUM(c.duration), 0) / 3600.0 as content_hours
            FROM tasks.task_queue tq
            LEFT JOIN content c ON tq.content_id = c.content_id
            WHERE tq.status = 'completed'
                AND tq.completed_at >= NOW() - INTERVAL '{minutes} minutes'
            GROUP BY tq.task_type
        """)).fetchall()

        throughput = {}
        for row in result:
            throughput[row.task_type] = float(row.content_hours) if row.content_hours else 0.0

        return throughput


@st.cache_data(ttl=30)
def get_task_queue_status() -> list[dict]:
    """Get comprehensive task queue status with hourly rates - matches Worker Monitoring format."""
    with get_session() as session:
        result = session.execute(text("""
            WITH task_stats AS (
                SELECT
                    tq.task_type,
                    COUNT(*) as total_tasks,
                    COALESCE(SUM(c.duration), 0) / 3600.0 as total_hours,
                    COUNT(*) FILTER (WHERE tq.status = 'pending') as pending_count,
                    COALESCE(SUM(c.duration) FILTER (WHERE tq.status = 'pending'), 0) / 3600.0 as pending_hours,
                    COUNT(*) FILTER (WHERE tq.status = 'processing') as processing_count,
                    COALESCE(SUM(c.duration) FILTER (WHERE tq.status = 'processing'), 0) / 3600.0 as processing_hours,
                    COUNT(*) FILTER (WHERE tq.status = 'completed') as completed_count,
                    COALESCE(SUM(c.duration) FILTER (WHERE tq.status = 'completed'), 0) / 3600.0 as completed_hours,
                    COUNT(*) FILTER (WHERE tq.status = 'error') as failed_count,
                    COALESCE(SUM(c.duration) FILTER (WHERE tq.status = 'error'), 0) / 3600.0 as failed_hours,
                    COUNT(*) FILTER (WHERE tq.status = 'completed' AND tq.completed_at >= NOW() - INTERVAL '1 hour') as last_hour_count,
                    COALESCE(SUM(c.duration) FILTER (WHERE tq.status = 'completed' AND tq.completed_at >= NOW() - INTERVAL '1 hour'), 0) / 3600.0 as last_hour_hours
                FROM tasks.task_queue tq
                LEFT JOIN content c ON tq.content_id = c.content_id
                GROUP BY tq.task_type
            )
            SELECT * FROM task_stats
            ORDER BY task_type
        """)).fetchall()

        stats = []
        for row in result:
            # For transcribe tasks, estimate hours based on 5 min average chunk
            if row.task_type == 'transcribe':
                total_hours = (row.total_tasks * 5) / 60.0
                pending_hours = (row.pending_count * 5) / 60.0
                processing_hours = (row.processing_count * 5) / 60.0
                completed_hours = (row.completed_count * 5) / 60.0
                failed_hours = (row.failed_count * 5) / 60.0
                last_hour_hours = (row.last_hour_count * 5) / 60.0
            else:
                total_hours = float(row.total_hours) if row.total_hours else 0.0
                pending_hours = float(row.pending_hours) if row.pending_hours else 0.0
                processing_hours = float(row.processing_hours) if row.processing_hours else 0.0
                completed_hours = float(row.completed_hours) if row.completed_hours else 0.0
                failed_hours = float(row.failed_hours) if row.failed_hours else 0.0
                last_hour_hours = float(row.last_hour_hours) if row.last_hour_hours else 0.0

            stats.append({
                'task_type': row.task_type,
                'total_tasks': row.total_tasks,
                'total_hours': total_hours,
                'last_hour_rate': last_hour_hours,
                'pending_count': row.pending_count,
                'pending_hours': pending_hours,
                'processing_count': row.processing_count,
                'processing_hours': processing_hours,
                'completed_count': row.completed_count,
                'completed_hours': completed_hours,
                'failed_count': row.failed_count,
                'failed_hours': failed_hours,
            })

        return stats


@st.cache_data(ttl=60)
def get_hourly_throughput(hours: int = 24) -> pd.DataFrame:
    """Calculate hourly task completion throughput"""
    with get_session() as session:
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)

        results = session.query(
            func.date_trunc('hour', TaskQueue.completed_at).label('hour'),
            TaskQueue.task_type,
            func.count(TaskQueue.id).label('count')
        ).filter(
            and_(
                TaskQueue.status == 'completed',
                TaskQueue.completed_at >= cutoff_time
            )
        ).group_by(
            func.date_trunc('hour', TaskQueue.completed_at),
            TaskQueue.task_type
        ).order_by('hour').all()

        if not results:
            return pd.DataFrame()

        data = [{
            'hour': r.hour,
            'task_type': r.task_type,
            'count': r.count
        } for r in results]

        return pd.DataFrame(data)


@st.cache_data(ttl=60)
def get_completed_tasks_with_duration(start_date=None, end_date=None, task_types=None, worker_ids=None):
    """Get completed tasks with content duration information.

    OPTIMIZED: Uses a single SQL query with subquery for chunk duration
    instead of N+1 ORM queries for each transcribe task.
    """
    with get_session() as session:
        try:
            # Build parameter dict
            params = {}

            # Build WHERE clauses
            where_clauses = ["tq.status = 'completed'", "tq.completed_at IS NOT NULL"]

            if start_date:
                where_clauses.append("tq.completed_at >= :start_date")
                params['start_date'] = start_date
            if end_date:
                where_clauses.append("tq.completed_at <= :end_date")
                params['end_date'] = end_date

            # Handle task_types filter
            if task_types and len(task_types) > 0:
                # Create parameter placeholders for each task type
                type_placeholders = []
                for i, tt in enumerate(task_types):
                    param_name = f"task_type_{i}"
                    type_placeholders.append(f":{param_name}")
                    params[param_name] = tt
                where_clauses.append(f"tq.task_type IN ({', '.join(type_placeholders)})")

            # Handle worker_ids filter
            if worker_ids and len(worker_ids) > 0:
                worker_placeholders = []
                for i, wid in enumerate(worker_ids):
                    param_name = f"worker_id_{i}"
                    worker_placeholders.append(f":{param_name}")
                    params[param_name] = wid
                where_clauses.append(f"tq.worker_id IN ({', '.join(worker_placeholders)})")

            where_sql = " AND ".join(where_clauses)

            # Single optimized query with pre-computed chunk duration
            sql = text(f"""
                SELECT
                    tq.id as task_id,
                    tq.task_type,
                    tq.content_id,
                    tq.worker_id,
                    tq.completed_at,
                    tq.started_at,
                    tq.created_at,
                    tq.processor_task_id,
                    tq.input_data,
                    c.duration as content_duration,
                    c.title as content_title,
                    c.id as content_table_id,
                    -- Pre-compute chunk duration for transcribe tasks via subquery
                    CASE
                        WHEN tq.task_type = 'transcribe' AND tq.input_data->>'chunk_index' IS NOT NULL
                        THEN COALESCE(
                            (SELECT cc.duration FROM content_chunks cc
                             WHERE cc.content_id = c.id
                             AND cc.chunk_index = (tq.input_data->>'chunk_index')::int
                             LIMIT 1),
                            300
                        )
                        ELSE c.duration
                    END as content_duration_processed
                FROM tasks.task_queue tq
                JOIN content c ON tq.content_id = c.content_id
                WHERE {where_sql}
                ORDER BY tq.completed_at DESC
                LIMIT 50000
            """)

            result = session.execute(sql, params).fetchall()

            task_data = []
            for row in result:
                # Calculate execution duration
                if row.started_at and row.completed_at:
                    execution_duration = (row.completed_at - row.started_at).total_seconds()
                elif row.created_at and row.completed_at:
                    execution_duration = (row.completed_at - row.created_at).total_seconds()
                else:
                    execution_duration = None

                task_data.append({
                    'timestamp': row.completed_at,
                    'worker_id': row.worker_id,
                    'task_type': row.task_type,
                    'content_id': row.content_id,
                    'content_title': row.content_title,
                    'execution_duration': execution_duration,
                    'content_duration_processed': row.content_duration_processed,
                    'created_at': row.created_at,
                    'started_at': row.started_at,
                    'processor_task_id': row.processor_task_id
                })

            return task_data
        except Exception as e:
            logger.error(f"Error getting completed tasks from database: {e}")
            return []


@st.cache_data(ttl=30)
def get_worker_performance_stats(hours: int = 2) -> list[dict]:
    """Get worker performance aggregated in SQL.

    Returns aggregated stats per worker/task_type for performance analysis.
    """
    with get_session() as session:
        try:
            sql = text("""
                SELECT
                    tq.worker_id,
                    tq.task_type,
                    COUNT(*) as tasks_completed,
                    -- For transcribe, use 5min estimate; otherwise use content duration
                    SUM(CASE
                        WHEN tq.task_type = 'transcribe' THEN 300
                        ELSE COALESCE(c.duration, 0)
                    END) / 3600.0 as content_hours,
                    SUM(EXTRACT(EPOCH FROM (tq.completed_at - tq.started_at))) / 3600.0 as exec_hours,
                    AVG(EXTRACT(EPOCH FROM (tq.completed_at - tq.started_at))) as avg_exec_seconds
                FROM tasks.task_queue tq
                JOIN content c ON tq.content_id = c.content_id
                WHERE tq.status = 'completed'
                  AND tq.completed_at >= NOW() - INTERVAL :hours_interval
                  AND tq.worker_id IS NOT NULL
                  AND tq.started_at IS NOT NULL
                GROUP BY tq.worker_id, tq.task_type
                ORDER BY tq.worker_id, tasks_completed DESC
            """)

            result = session.execute(sql, {'hours_interval': f'{hours} hours'}).fetchall()

            stats = []
            for row in result:
                rate = None
                if row.exec_hours and row.exec_hours > 0:
                    rate = row.content_hours / row.exec_hours

                stats.append({
                    'worker_id': row.worker_id,
                    'task_type': row.task_type,
                    'tasks_completed': row.tasks_completed,
                    'content_hours': row.content_hours or 0,
                    'exec_hours': row.exec_hours or 0,
                    'avg_exec_seconds': row.avg_exec_seconds or 0,
                    'rate': rate
                })

            return stats
        except Exception as e:
            logger.error(f"Error getting worker performance stats: {e}")
            return []
