#!/usr/bin/env python3
"""
System Monitoring Dashboard - Tabbed Interface
==============================================

Four-tab layout:
- Quick Status: Color-coded grid of all critical services
- Services: System health, model servers, workers, LLM balancer
- Tasks: Scheduled tasks, task queue, throughput charts
- Projects: Pipeline progress, embeddings, speaker identification
"""
import sys
from pathlib import Path
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import yaml
from sqlalchemy import text, func, and_, or_

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.database.session import get_session
from src.database.models import Content, TaskQueue, ContentChunk
from src.utils.logger import setup_worker_logger

logger = setup_worker_logger('system_monitoring')

# Configuration
ORCHESTRATOR_API_URL = "http://localhost:8001"
LLM_BALANCER_URL = "http://localhost:8002"

# Model servers configuration (from config.yaml)
MODEL_SERVERS = {
    'worker0': {'host': '10.0.0.34', 'port': 8004, 'tiers': ['tier_1', 'tier_2']},
    'head': {'host': '10.0.0.4', 'port': 8004, 'tiers': ['tier_2']},
    'worker5': {'host': '10.0.0.209', 'port': 8004, 'tiers': ['tier_3']},
}

# Worker processors configuration
WORKER_PROCESSORS = {
    'worker0': {'host': '10.0.0.34', 'port': 8000},
    'worker1': {'host': '10.0.0.101', 'port': 8000},
    'worker2': {'host': '10.0.0.181', 'port': 8000},
    'worker3': {'host': '10.0.0.159', 'port': 8000},
    'worker4': {'host': '10.0.0.213', 'port': 8000},
    'worker5': {'host': '10.0.0.209', 'port': 8000},
    'worker6': {'host': '10.0.0.189', 'port': 8000},
}

# Log sources configuration for the Logs tab
LOG_SOURCES = {
    'Orchestrator': {
        'orchestrator': {
            'name': 'Orchestrator V2',
            'command': 'tail -f /Users/signal4/logs/content_processing/worker6/worker6_orchestrator_v2.log',
            'local': True,
        },
        'task_orchestrator': {
            'name': 'Task Orchestrator',
            'command': 'tail -f /Users/signal4/logs/content_processing/worker6/worker6_task_orchestrator.log',
            'local': True,
        },
        'transcribe_orchestrator': {
            'name': 'Transcribe Orchestrator',
            'command': 'tail -f /Users/signal4/logs/content_processing/worker6/worker6_transcribe_orchestrator.log',
            'local': True,
        },
    },
    'Model Servers': {
        'worker6_model_server': {
            'name': 'Worker6 Model Server',
            'command': 'tail -f /Users/signal4/logs/content_processing/worker6/worker6_model_server.log',
            'local': True,
        },
        'worker0_model_server': {
            'name': 'Worker0 Model Server',
            'command': 'ssh signal4@10.0.0.34 "tail -f /Users/signal4/logs/content_processing/worker0/worker0_model_server.log"',
            'local': False,
            'host': '10.0.0.34',
        },
        'worker5_model_server': {
            'name': 'Worker5 Model Server',
            'command': 'ssh signal4@10.0.0.209 "tail -f /Users/signal4/logs/content_processing/worker5/worker5_model_server.log"',
            'local': False,
            'host': '10.0.0.209',
        },
    },
    'Task Processors': {
        'worker6_processor': {
            'name': 'Worker6 Processor',
            'command': 'tail -f /Users/signal4/logs/content_processing/worker6/worker6_task_processor.log',
            'local': True,
        },
        'worker0_processor': {
            'name': 'Worker0 Processor',
            'command': 'ssh signal4@10.0.0.34 "tail -f /Users/signal4/logs/content_processing/worker0/worker0_task_processor.log"',
            'local': False,
            'host': '10.0.0.34',
        },
        'worker1_processor': {
            'name': 'Worker1 Processor',
            'command': 'ssh signal4@10.0.0.101 "tail -f /Users/signal4/logs/content_processing/worker1/worker1_task_processor.log"',
            'local': False,
            'host': '10.0.0.101',
        },
        'worker2_processor': {
            'name': 'Worker2 Processor',
            'command': 'ssh signal4@10.0.0.181 "tail -f /Users/signal4/logs/content_processing/worker2/worker2_task_processor.log"',
            'local': False,
            'host': '10.0.0.181',
        },
        'worker3_processor': {
            'name': 'Worker3 Processor',
            'command': 'ssh signal4@10.0.0.159 "tail -f /Users/signal4/logs/content_processing/worker3/worker3_task_processor.log"',
            'local': False,
            'host': '10.0.0.159',
        },
        'worker4_processor': {
            'name': 'Worker4 Processor',
            'command': 'ssh signal4@10.0.0.213 "tail -f /Users/signal4/logs/content_processing/worker4/worker4_task_processor.log"',
            'local': False,
            'host': '10.0.0.213',
        },
        'worker5_processor': {
            'name': 'Worker5 Processor',
            'command': 'ssh signal4@10.0.0.209 "tail -f /Users/signal4/logs/content_processing/worker5/worker5_task_processor.log"',
            'local': False,
            'host': '10.0.0.209',
        },
    },
    'Other': {
        'embedding_hydrator': {
            'name': 'Embedding Hydrator',
            'command': 'tail -f /Users/signal4/logs/content_processing/worker6/worker6_hydrate_embeddings.log',
            'local': True,
        },
        'speaker_id_orchestrator': {
            'name': 'Speaker ID Orchestrator',
            'command': 'tail -f /Users/signal4/logs/content_processing/worker6/worker6_speaker_identification.orchestrator.log',
            'local': True,
        },
    },
}

# Quick status grid configuration - organized by category
QUICK_STATUS_GROUPS = {
    'Orchestrator': [
        {'name': 'Orchestrator', 'url': 'http://localhost:8001', 'endpoint': '/api/health',
         'log_cmd': 'tail -f /Users/signal4/logs/content_processing/worker6/worker6_orchestrator_v2.log'},
    ],
    'LLMs': [
        {'name': 'Worker6', 'url': 'http://10.0.0.189:8004', 'endpoint': '/health',
         'log_cmd': 'tail -f /Users/signal4/logs/content_processing/worker6/worker6_model_server.log'},
        {'name': 'Worker0', 'url': 'http://10.0.0.34:8004', 'endpoint': '/health',
         'log_cmd': 'ssh signal4@10.0.0.34 "tail -f /Users/signal4/logs/content_processing/worker0/worker0_model_server.log"'},
        {'name': 'Worker5', 'url': 'http://10.0.0.209:8004', 'endpoint': '/health',
         'log_cmd': 'ssh signal4@10.0.0.209 "tail -f /Users/signal4/logs/content_processing/worker5/worker5_model_server.log"'},
    ],
    'Workers': [
        {'name': 'W0', 'url': 'http://10.0.0.34:8000', 'endpoint': '/tasks',
         'log_cmd': 'ssh signal4@10.0.0.34 "tail -f /Users/signal4/logs/content_processing/worker0/worker0_task_processor.log"'},
        {'name': 'W1', 'url': 'http://10.0.0.101:8000', 'endpoint': '/tasks',
         'log_cmd': 'ssh signal4@10.0.0.101 "tail -f /Users/signal4/logs/content_processing/worker1/worker1_task_processor.log"'},
        {'name': 'W2', 'url': 'http://10.0.0.181:8000', 'endpoint': '/tasks',
         'log_cmd': 'ssh signal4@10.0.0.181 "tail -f /Users/signal4/logs/content_processing/worker2/worker2_task_processor.log"'},
        {'name': 'W3', 'url': 'http://10.0.0.159:8000', 'endpoint': '/tasks',
         'log_cmd': 'ssh signal4@10.0.0.159 "tail -f /Users/signal4/logs/content_processing/worker3/worker3_task_processor.log"'},
        {'name': 'W4', 'url': 'http://10.0.0.213:8000', 'endpoint': '/tasks',
         'log_cmd': 'ssh signal4@10.0.0.213 "tail -f /Users/signal4/logs/content_processing/worker4/worker4_task_processor.log"'},
        {'name': 'W5', 'url': 'http://10.0.0.209:8000', 'endpoint': '/tasks',
         'log_cmd': 'ssh signal4@10.0.0.209 "tail -f /Users/signal4/logs/content_processing/worker5/worker5_task_processor.log"'},
        {'name': 'W6', 'url': 'http://10.0.0.189:8000', 'endpoint': '/tasks',
         'log_cmd': 'tail -f /Users/signal4/logs/content_processing/worker6/worker6_task_processor.log'},
    ],
    'Scheduled': [
        {'name': 'Podcast Index', 'task_id': 'podcast_index_download'},
        {'name': 'YouTube Index', 'task_id': 'youtube_index_download'},
        {'name': 'Embeddings', 'task_id': 'embedding_hydration'},
        {'name': 'Speaker ID 1', 'task_id': 'speaker_id_phase1'},
        {'name': 'Speaker ID 2', 'task_id': 'speaker_id_phase2'},
        {'name': 'Tone', 'task_id': 'tone_hydration'},
        {'name': 'Cache 30d', 'task_id': 'cache_refresh_30d'},
        {'name': 'Cache 7d', 'task_id': 'cache_refresh_7d'},
    ],
}

# Pipeline stage colors
PIPELINE_COLORS = {
    'Pending Download': '#e74c3c',
    'Downloaded': '#95a5a6',
    'Audio Extracted': '#e67e22',
    'Transcribed': '#3498db',
    'Diarized & Transcribed': '#9b59b6',
    'Stitched': '#2ecc71',
    'Segment Embeddings': '#155724',
    'Blocked': '#c0392b',
    'Skipped (Short)': '#7f8c8d',
}

# Task type colors
TASK_COLORS = {
    'download': '#1f77b4',
    'convert': '#ff7f0e',
    'diarize': '#d62728',
    'transcribe': '#2ca02c',
    'stitch': '#9467bd',
    'cleanup_and_compress': '#e377c2',
    'segment_embeddings': '#17becf',
}


def load_config() -> dict:
    """Load configuration from yaml file"""
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    try:
        with open(config_path) as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return {}


# =============================================================================
# API Functions
# =============================================================================

def fetch_api(endpoint: str, base_url: str = None, timeout: int = 15) -> dict:
    """Fetch data from API with error handling"""
    url = base_url or ORCHESTRATOR_API_URL
    try:
        response = requests.get(f"{url}{endpoint}", timeout=timeout)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Connection refused - Service not running"}
    except requests.exceptions.Timeout:
        return {"error": "Request timeout"}
    except Exception as e:
        return {"error": str(e)}


def post_api(endpoint: str, data: dict = None, base_url: str = None) -> dict:
    """POST to API with error handling"""
    url = base_url or ORCHESTRATOR_API_URL
    try:
        response = requests.post(f"{url}{endpoint}", json=data or {}, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Connection refused - Service not running"}
    except requests.exceptions.Timeout:
        return {"error": "Request timeout"}
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=30)
def check_service_health(url: str, endpoint: str = "/health") -> dict:
    """Check if a service is healthy"""
    try:
        response = requests.get(f"{url}{endpoint}", timeout=3)
        if response.status_code == 200:
            return {"status": "running", "response": response.json() if response.text else {}}
        else:
            return {"status": "unhealthy", "code": response.status_code}
    except requests.exceptions.ConnectionError:
        return {"status": "stopped", "error": "Connection refused"}
    except requests.exceptions.Timeout:
        return {"status": "timeout", "error": "Request timeout"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# =============================================================================
# Database Query Functions
# =============================================================================

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
def get_available_filters():
    """Get available task types and workers from database"""
    with get_session() as session:
        try:
            task_types = session.query(TaskQueue.task_type).distinct().order_by(TaskQueue.task_type).all()
            task_types = [t[0] for t in task_types if t[0]]

            workers = session.query(TaskQueue.worker_id).filter(
                TaskQueue.worker_id.isnot(None)
            ).distinct().order_by(TaskQueue.worker_id).all()
            workers = [w[0] for w in workers if w[0]]

            return task_types, workers
        except Exception as e:
            logger.error(f"Error getting available filters: {e}")
            return [], []


def get_completed_tasks_with_duration(start_date=None, end_date=None, task_types=None, worker_ids=None):
    """Get completed tasks with content duration information."""
    with get_session() as session:
        try:
            query = session.query(
                TaskQueue.id.label('task_id'),
                TaskQueue.task_type,
                TaskQueue.content_id,
                TaskQueue.worker_id,
                TaskQueue.completed_at,
                TaskQueue.started_at,
                TaskQueue.created_at,
                TaskQueue.processor_task_id,
                TaskQueue.input_data,
                Content.duration.label('content_duration'),
                Content.title.label('content_title'),
                Content.id.label('content_table_id')
            ).join(
                Content, TaskQueue.content_id == Content.content_id
            ).filter(
                TaskQueue.status == 'completed',
                TaskQueue.completed_at.isnot(None)
            )

            if start_date:
                query = query.filter(TaskQueue.completed_at >= start_date)
            if end_date:
                query = query.filter(TaskQueue.completed_at <= end_date)
            if task_types and len(task_types) > 0:
                query = query.filter(TaskQueue.task_type.in_(task_types))
            if worker_ids and len(worker_ids) > 0:
                query = query.filter(TaskQueue.worker_id.in_(worker_ids))

            query = query.order_by(TaskQueue.completed_at.desc())
            tasks = query.all()

            task_data = []
            for task in tasks:
                if task.started_at and task.completed_at:
                    execution_duration = (task.completed_at - task.started_at).total_seconds()
                elif task.created_at and task.completed_at:
                    execution_duration = (task.completed_at - task.created_at).total_seconds()
                else:
                    execution_duration = None

                content_duration_processed = None

                if task.task_type == 'transcribe' and task.input_data:
                    chunk_index = task.input_data.get('chunk_index')
                    if chunk_index is not None and hasattr(task, 'content_table_id'):
                        chunk = session.query(ContentChunk).filter(
                            ContentChunk.content_id == task.content_table_id,
                            ContentChunk.chunk_index == chunk_index
                        ).first()
                        if chunk and chunk.duration:
                            content_duration_processed = chunk.duration
                        else:
                            content_duration_processed = 300

                if content_duration_processed is None and task.content_duration:
                    content_duration_processed = task.content_duration

                task_data.append({
                    'timestamp': task.completed_at,
                    'worker_id': task.worker_id,
                    'task_type': task.task_type,
                    'content_id': task.content_id,
                    'content_title': task.content_title,
                    'execution_duration': execution_duration,
                    'content_duration_processed': content_duration_processed,
                    'created_at': task.created_at,
                    'started_at': task.started_at,
                    'processor_task_id': task.processor_task_id
                })

            return task_data
        except Exception as e:
            logger.error(f"Error getting completed tasks from database: {e}")
            return []


@st.cache_data(ttl=60)
def get_pipeline_progress_from_db() -> dict:
    """Query pipeline progress directly from database"""
    try:
        with get_session() as session:
            emb_result = session.execute(text("""
                SELECT
                    COUNT(*) as total_segments,
                    COUNT(embedding) FILTER (WHERE embedding IS NOT NULL) as with_primary,
                    COUNT(embedding_alt) FILTER (WHERE embedding_alt IS NOT NULL) as with_alternative
                FROM embedding_segments
            """)).fetchone()

            spk_result = session.execute(text("""
                SELECT
                    COUNT(*) as total_speakers,
                    COUNT(speaker_identity_id) FILTER (WHERE speaker_identity_id IS NOT NULL) as with_identity,
                    COUNT(*) FILTER (WHERE text_evidence_status IS NOT NULL AND text_evidence_status NOT IN ('not_processed', 'unprocessed')) as phase2_processed,
                    COUNT(*) FILTER (WHERE text_evidence_status = 'certain') as phase2_certain,
                    COUNT(*) FILTER (WHERE text_evidence_status = 'none') as phase2_no_evidence,
                    COUNT(*) FILTER (WHERE duration > 60) as significant_duration
                FROM speakers
            """)).fetchone()

            identity_result = session.execute(text("""
                SELECT
                    COUNT(*) as total_identities,
                    COUNT(CASE WHEN primary_name IS NOT NULL AND primary_name != '' THEN 1 END) as named_identities
                FROM speaker_identities
            """)).fetchone()

            content_result = session.execute(text("""
                SELECT
                    COUNT(*) as total_content,
                    COUNT(*) FILTER (WHERE is_stitched = true) as stitched,
                    COUNT(*) FILTER (WHERE is_embedded = true) as embedded,
                    COUNT(*) FILTER (WHERE is_stitched = true AND is_embedded = false) as needs_embedding
                FROM content
                WHERE blocked_download = false AND is_duplicate = false AND is_short = false
            """)).fetchone()

        total_segments = emb_result.total_segments or 1
        total_speakers = spk_result.total_speakers or 1
        significant_speakers = spk_result.significant_duration or 1
        total_content = content_result.total_content or 1

        return {
            "embedding": {
                "total_segments": emb_result.total_segments,
                "primary": {
                    "completed": emb_result.with_primary,
                    "percent": round(100 * emb_result.with_primary / total_segments, 2)
                },
                "alternative": {
                    "completed": emb_result.with_alternative,
                    "percent": round(100 * emb_result.with_alternative / total_segments, 2)
                }
            },
            "speaker_identification": {
                "total_speakers": spk_result.total_speakers,
                "significant_duration_speakers": spk_result.significant_duration,
                "with_identity": {
                    "count": spk_result.with_identity,
                    "percent": round(100 * spk_result.with_identity / total_speakers, 2)
                },
                "phase2_text_evidence": {
                    "processed": spk_result.phase2_processed,
                    "certain": spk_result.phase2_certain,
                    "no_evidence": spk_result.phase2_no_evidence,
                    "percent_of_significant": round(100 * spk_result.phase2_processed / significant_speakers, 2) if spk_result.phase2_processed else 0
                },
                "identities": {
                    "total": identity_result.total_identities,
                    "named": identity_result.named_identities
                }
            },
            "content": {
                "total": content_result.total_content,
                "stitched": content_result.stitched,
                "embedded": content_result.embedded,
                "needs_embedding": content_result.needs_embedding,
                "percent_embedded": round(100 * content_result.embedded / total_content, 2)
            }
        }
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=60)
def get_cache_table_status(time_window: str = '30d') -> dict:
    """Get status of embedding cache tables (7d or 30d).

    These cache tables contain segments ready for analysis with HNSW indexes.
    Shows content/segment counts per project from the maintained cache.

    Args:
        time_window: '7d' or '30d' to select the cache table
    """
    cache_table = f"embedding_cache_{time_window}"

    with get_session() as session:
        try:
            config = load_config()
            active_projects = [project for project, settings in config.get('active_projects', {}).items()
                             if settings.get('enabled', False)]

            if not active_projects:
                return {'error': 'No active projects'}

            # Get cache table stats per project
            project_stats = {}

            # Generate date range for the cache window
            days = 30 if time_window == '30d' else 7
            end_date = datetime.now(timezone.utc).date()
            start_date = end_date - timedelta(days=days - 1)

            for project in active_projects:
                # Get segment and content counts from cache table
                result = session.execute(text(f"""
                    SELECT
                        COUNT(*) as segment_count,
                        COUNT(DISTINCT content_id) as content_count,
                        COUNT(*) FILTER (WHERE embedding IS NOT NULL) as with_main_embedding,
                        COUNT(*) FILTER (WHERE embedding_alt IS NOT NULL) as with_alt_embedding,
                        MIN(publish_date) as earliest_date,
                        MAX(publish_date) as latest_date
                    FROM {cache_table}
                    WHERE :project = ANY(projects)
                """), {'project': project}).fetchone()

                # Get daily embedded content counts (by publish_date) from cache
                daily_results = session.execute(text(f"""
                    SELECT
                        DATE(publish_date AT TIME ZONE 'UTC') as publish_day,
                        COUNT(DISTINCT content_id) as content_count,
                        COUNT(*) as segment_count
                    FROM {cache_table}
                    WHERE :project = ANY(projects)
                    GROUP BY DATE(publish_date AT TIME ZONE 'UTC')
                    ORDER BY publish_day
                """), {'project': project}).fetchall()

                daily_embedded = {row.publish_day: row.content_count for row in daily_results}
                daily_segments = {row.publish_day: row.segment_count for row in daily_results}

                # Get daily total content counts from content table (stitched content in date range)
                total_daily_results = session.execute(text("""
                    SELECT
                        DATE(publish_date AT TIME ZONE 'UTC') as publish_day,
                        COUNT(*) as total_content,
                        COUNT(*) FILTER (WHERE is_stitched = true) as stitched_content,
                        COUNT(*) FILTER (WHERE is_embedded = true) as embedded_content
                    FROM content
                    WHERE :project = ANY(projects)
                      AND blocked_download = false
                      AND is_duplicate = false
                      AND is_short = false
                      AND publish_date >= :start_date
                      AND publish_date < :end_date + interval '1 day'
                    GROUP BY DATE(publish_date AT TIME ZONE 'UTC')
                    ORDER BY publish_day
                """), {
                    'project': project,
                    'start_date': start_date,
                    'end_date': end_date
                }).fetchall()

                daily_total = {row.publish_day: row.total_content for row in total_daily_results}
                daily_stitched = {row.publish_day: row.stitched_content for row in total_daily_results}
                daily_embedded_content = {row.publish_day: row.embedded_content for row in total_daily_results}

                project_stats[project] = {
                    'segment_count': result.segment_count or 0,
                    'content_count': result.content_count or 0,
                    'with_main_embedding': result.with_main_embedding or 0,
                    'with_alt_embedding': result.with_alt_embedding or 0,
                    'earliest_date': result.earliest_date.date().isoformat() if result.earliest_date else None,
                    'latest_date': result.latest_date.date().isoformat() if result.latest_date else None,
                    'daily_embedded': daily_embedded,
                    'daily_segments': daily_segments,
                    'daily_total': daily_total,
                    'daily_stitched': daily_stitched,
                    'daily_embedded_content': daily_embedded_content,
                }

            # Get overall cache stats
            total_result = session.execute(text(f"""
                SELECT
                    COUNT(*) as total_segments,
                    COUNT(DISTINCT content_id) as total_content,
                    MIN(publish_date) as cache_start,
                    MAX(publish_date) as cache_end
                FROM {cache_table}
            """)).fetchone()

            # Get total content in the date range for percentage calculation
            window_totals = session.execute(text("""
                SELECT
                    COUNT(*) as total_content,
                    COUNT(*) FILTER (WHERE is_stitched = true) as stitched_content,
                    COUNT(*) FILTER (WHERE is_embedded = true) as embedded_content
                FROM content
                WHERE blocked_download = false
                  AND is_duplicate = false
                  AND is_short = false
                  AND publish_date >= :start_date
                  AND publish_date < :end_date + interval '1 day'
            """), {'start_date': start_date, 'end_date': end_date}).fetchone()

            all_dates = []
            current = start_date
            while current <= end_date:
                all_dates.append(current)
                current += timedelta(days=1)

            return {
                'time_window': time_window,
                'cache_table': cache_table,
                'projects': project_stats,
                'total_segments': total_result.total_segments or 0,
                'total_content_in_cache': total_result.total_content or 0,
                'cache_start': total_result.cache_start.date().isoformat() if total_result.cache_start else None,
                'cache_end': total_result.cache_end.date().isoformat() if total_result.cache_end else None,
                'window_total_content': window_totals.total_content or 0,
                'window_stitched_content': window_totals.stitched_content or 0,
                'window_embedded_content': window_totals.embedded_content or 0,
                'date_range': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat(),
                    'dates': [d.isoformat() for d in all_dates]
                }
            }

        except Exception as e:
            logger.error(f"Error getting {time_window} cache status: {e}")
            return {'error': str(e)}


@st.cache_data(ttl=60)
def get_global_content_status() -> dict:
    """Get content processing status breakdown by project for top-level summary."""
    with get_session() as session:
        try:
            config = load_config()
            active_projects = [project for project, settings in config.get('active_projects', {}).items()
                             if settings.get('enabled', False)]

            if not active_projects:
                return {}

            project_data = {}
            total_content = 0

            for project in active_projects:
                project_settings = config.get('active_projects', {}).get(project, {})
                project_start_str = project_settings.get('start_date')
                project_end_str = project_settings.get('end_date')

                project_filters = [Content.projects.any(project)]

                if project_start_str:
                    project_start = datetime.strptime(project_start_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                    project_filters.append(Content.publish_date >= project_start)

                if project_end_str:
                    project_end = datetime.strptime(project_end_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                    project_filters.append(Content.publish_date <= project_end)

                if project_start_str or project_end_str:
                    project_filters.append(Content.publish_date.isnot(None))

                project_total = session.query(func.count(Content.content_id)).filter(*project_filters).scalar() or 0

                if project_total == 0:
                    continue

                total_content += project_total

                segment_embeddings = session.query(func.count()).filter(
                    *project_filters,
                    Content.is_embedded == True
                ).scalar() or 0

                stitched = session.query(func.count()).filter(
                    *project_filters,
                    Content.is_stitched == True,
                    or_(Content.is_embedded != True, Content.is_embedded.is_(None))
                ).scalar() or 0

                diarized_transcribed = session.query(func.count()).filter(
                    *project_filters,
                    Content.is_diarized == True,
                    Content.is_transcribed == True,
                    Content.is_stitched == False,
                    or_(Content.is_embedded != True, Content.is_embedded.is_(None))
                ).scalar() or 0

                audio_extracted = session.query(func.count()).filter(
                    *project_filters,
                    Content.is_converted == True,
                    Content.is_transcribed == False,
                    Content.is_diarized == False,
                    Content.is_stitched == False,
                    or_(Content.is_embedded != True, Content.is_embedded.is_(None))
                ).scalar() or 0

                downloaded_only = session.query(func.count()).filter(
                    *project_filters,
                    Content.is_downloaded == True,
                    Content.is_converted == False,
                    Content.is_transcribed == False,
                    Content.is_diarized == False,
                    Content.is_stitched == False,
                    or_(Content.is_embedded != True, Content.is_embedded.is_(None))
                ).scalar() or 0

                pending_download = session.query(func.count()).filter(
                    *project_filters,
                    Content.is_downloaded == False,
                    Content.blocked_download == False,
                    Content.is_converted == False,
                    Content.is_transcribed == False,
                    Content.is_diarized == False,
                    Content.is_stitched == False,
                    Content.is_embedded == False,
                    Content.is_compressed == False,
                    or_(
                        Content.duration >= 180,
                        Content.duration.is_(None),
                        and_(Content.platform != 'podcast', Content.duration < 180)
                    )
                ).scalar() or 0

                status_counts = {
                    'Segment Embeddings': segment_embeddings,
                    'Stitched': stitched,
                    'Diarized & Transcribed': diarized_transcribed,
                    'Audio Extracted': audio_extracted,
                    'Downloaded Only': downloaded_only,
                    'Pending Download': pending_download,
                }

                project_data[project] = {
                    'status_counts': status_counts,
                    'total_content': project_total,
                    'start_date': project_start_str,
                    'end_date': project_end_str
                }

            return {
                'project_data': project_data,
                'total_content': total_content,
                'active_projects': list(project_data.keys())
            }

        except Exception as e:
            logger.error(f"Database error in get_global_content_status: {str(e)}")
            session.rollback()
            return {}


# =============================================================================
# Visualization Functions
# =============================================================================

def create_cache_heatmap(cache_status: dict) -> go.Figure | None:
    """Create a heatmap showing daily embedded percentage per project.

    Color intensity indicates what percentage of content published that day
    has been fully processed (embedded). 100% = all green.

    Args:
        cache_status: Output from get_cache_table_status()
    """
    try:
        if not cache_status or 'error' in cache_status:
            return None

        projects = cache_status.get('projects', {})
        date_range = cache_status.get('date_range', {})
        dates = date_range.get('dates', [])
        time_window = cache_status.get('time_window', '30d')

        if not projects or not dates:
            return None

        # Build heatmap data matrix
        sorted_projects = sorted(projects.keys())
        z_data = []
        hover_text = []

        for project in sorted_projects:
            project_data = projects[project]
            daily_total = project_data.get('daily_total', {})
            daily_embedded = project_data.get('daily_embedded_content', {})
            daily_stitched = project_data.get('daily_stitched', {})
            row_data = []
            row_hover = []

            for date_str in dates:
                date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
                total = daily_total.get(date_obj, 0)
                embedded = daily_embedded.get(date_obj, 0)
                stitched = daily_stitched.get(date_obj, 0)

                if total > 0:
                    pct = (embedded / total) * 100
                    stitched_pct = (stitched / total) * 100
                else:
                    pct = 0
                    stitched_pct = 0

                row_data.append(pct)
                row_hover.append(
                    f"{project}<br>{date_str}<br>"
                    f"Embedded: {embedded:,}/{total:,} ({pct:.0f}%)<br>"
                    f"Stitched: {stitched:,}/{total:,} ({stitched_pct:.0f}%)"
                )

            z_data.append(row_data)
            hover_text.append(row_hover)

        # Create heatmap with 0-100% scale
        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=[datetime.strptime(d, "%Y-%m-%d").strftime("%m/%d") for d in dates],
            y=sorted_projects,
            zmin=0,
            zmax=100,
            colorscale=[
                [0, '#f0f0f0'],      # 0% - light gray
                [0.01, '#ffcccc'],   # 1% - light red
                [0.25, '#ffeb99'],   # 25% - yellow
                [0.5, '#c6e48b'],    # 50% - light green
                [0.75, '#7bc96f'],   # 75% - medium green
                [1, '#239a3b']       # 100% - full green
            ],
            showscale=True,
            colorbar=dict(
                title="Embedded %",
                titleside="right",
                ticksuffix="%",
            ),
            hovertemplate='%{text}<extra></extra>',
            text=hover_text,
        ))

        # Adjust tick display based on window size
        dtick = 1 if time_window == '7d' else 2

        fig.update_layout(
            xaxis=dict(
                title='Publish Date',
                tickangle=-45,
                tickfont=dict(size=10),
                dtick=dtick,
            ),
            yaxis=dict(
                title='',
                tickfont=dict(size=11),
            ),
            height=max(200, len(sorted_projects) * 35 + 80),
            margin=dict(t=10, b=60, l=100, r=60),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )

        return fig

    except Exception as e:
        logger.error(f"Error creating cache heatmap: {e}")
        return None


def create_project_progress_bars(global_status: dict):
    """Create horizontal progress bars showing pipeline progress for each project."""
    try:
        if not global_status or 'project_data' not in global_status:
            return None

        project_data = global_status['project_data']

        if not project_data:
            return None

        pipeline_stages = [
            ('Pending Download', 5, '#e74c3c'),
            ('Downloaded Only', 15, '#95a5a6'),
            ('Audio Extracted', 30, '#e67e22'),
            ('Diarized & Transcribed', 75, '#9b59b6'),
            ('Stitched', 85, '#2ecc71'),
            ('Segment Embeddings', 100, '#155724')
        ]

        fig = go.Figure()
        sorted_projects = sorted(project_data.keys())

        for project in sorted_projects:
            project_info = project_data[project]
            status_counts = project_info['status_counts']
            total_content = project_info['total_content']

            if total_content == 0:
                continue

            x_offset = 0
            for status, progress_pct, color in pipeline_stages:
                count = status_counts.get(status, 0)
                if count > 0:
                    percentage = (count / total_content) * 100

                    fig.add_trace(go.Bar(
                        name=f"{project} - {status}",
                        x=[percentage],
                        y=[project],
                        orientation='h',
                        marker_color=color,
                        text=f"{count}" if percentage >= 8 else "",
                        textposition='inside',
                        textfont=dict(color='white', size=11),
                        hovertemplate=f"<b>{project}</b><br>" +
                                    f"Status: {status}<br>" +
                                    f"Count: {count:,}<br>" +
                                    f"Percentage: {percentage:.1f}%<extra></extra>",
                        base=x_offset,
                        showlegend=False
                    ))
                    x_offset += percentage

            total_progress = 0
            total_items = 0
            for status, progress_pct, color in pipeline_stages:
                count = status_counts.get(status, 0)
                total_progress += count * progress_pct
                total_items += count

            project_progress = total_progress / total_items if total_items > 0 else 0

            fig.add_annotation(
                x=102,
                y=project,
                text=f"{project_progress:.0f}% ({total_content:,})",
                showarrow=False,
                xanchor="left",
                font=dict(size=11, color="black")
            )

        fig.update_layout(
            xaxis=dict(
                title='Percentage (%)',
                range=[0, 115],
                ticksuffix='%',
                showgrid=True,
                gridcolor='lightgray'
            ),
            yaxis=dict(
                title='',
                categoryorder='array',
                categoryarray=sorted_projects[::-1],
                tickfont=dict(size=11)
            ),
            barmode='stack',
            showlegend=False,
            height=max(200, len(sorted_projects) * 40),
            margin=dict(t=10, b=30, l=100, r=80),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            bargap=0.3
        )

        return fig

    except Exception as e:
        logger.error(f"Error creating project progress bars: {str(e)}")
        return None


def plot_throughput_over_time(task_data, time_interval='hour'):
    """Create a stacked bar chart showing content hours processed per time interval."""
    if not task_data:
        return None

    try:
        df = pd.DataFrame(task_data)
        df = df[df['content_duration_processed'].notna()].copy()

        if df.empty:
            return None

        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df['content_hours'] = df['content_duration_processed'] / 3600

        now = datetime.now(timezone.utc)
        min_time = df['timestamp'].min()

        if time_interval == 'hour':
            df['time_bucket'] = df['timestamp'].dt.floor('h')
            title = 'Content Hours Processed Per Hour'
            start_bucket = min_time.replace(minute=0, second=0, microsecond=0)
            end_bucket = now.replace(minute=0, second=0, microsecond=0)
            all_buckets = pd.date_range(start=start_bucket, end=end_bucket, freq='h', tz=timezone.utc)
        elif time_interval == 'day':
            df['time_bucket'] = df['timestamp'].dt.floor('D')
            title = 'Content Hours Processed Per Day'
            start_bucket = min_time.replace(hour=0, minute=0, second=0, microsecond=0)
            end_bucket = now.replace(hour=0, minute=0, second=0, microsecond=0)
            all_buckets = pd.date_range(start=start_bucket, end=end_bucket, freq='D', tz=timezone.utc)
        else:
            df['time_bucket'] = df['timestamp'].dt.floor('h')
            title = 'Content Hours Processed Per Hour'
            start_bucket = min_time.replace(minute=0, second=0, microsecond=0)
            end_bucket = now.replace(minute=0, second=0, microsecond=0)
            all_buckets = pd.date_range(start=start_bucket, end=end_bucket, freq='h', tz=timezone.utc)

        throughput_rates = df.groupby(['time_bucket', 'task_type'])['content_hours'].sum().reset_index()
        all_task_types = df['task_type'].unique().tolist()
        pivot_df = throughput_rates.pivot(index='time_bucket', columns='task_type', values='content_hours').reset_index()
        complete_index = pd.DataFrame({'time_bucket': all_buckets})
        pivot_df = complete_index.merge(pivot_df, on='time_bucket', how='left')
        pivot_df = pivot_df.fillna(0)

        for task_type in all_task_types:
            if task_type not in pivot_df.columns:
                pivot_df[task_type] = 0

        if len(pivot_df) == 0:
            return None

        fig = go.Figure()

        for task_type in sorted([col for col in pivot_df.columns if col != 'time_bucket']):
            fig.add_trace(go.Bar(
                x=pivot_df['time_bucket'],
                y=pivot_df[task_type],
                name=task_type,
                marker_color=TASK_COLORS.get(task_type),
                hovertemplate=f'<b>{task_type}</b>: %{{y:.1f}}h<extra></extra>'
            ))

        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='Content Hours',
            height=300,
            barmode='stack',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode='x unified',
            yaxis=dict(tickformat=',d', rangemode='tozero'),
            margin=dict(t=50, b=30, l=60, r=20)
        )

        return fig

    except Exception as e:
        logger.error(f"Error creating throughput plot: {str(e)}")
        return None


def display_worker_throughput_table(task_data):
    """Display worker throughput by content duration processed"""
    if not task_data:
        return None

    df = pd.DataFrame(task_data)
    df_with_duration = df[df['content_duration_processed'].notna()]

    if df_with_duration.empty:
        return None

    throughput = df_with_duration.groupby(['worker_id', 'task_type']).agg(
        tasks_completed=('content_id', 'count'),
        total_content_hours=('content_duration_processed', lambda x: x.sum() / 3600),
        total_execution_time=('execution_duration', lambda x: x[x.notna()].sum() / 3600 if x[x.notna()].size > 0 else None)
    ).reset_index()

    throughput['throughput_per_hour'] = throughput.apply(
        lambda row: row['total_content_hours'] / row['total_execution_time']
        if row['total_execution_time'] and row['total_execution_time'] > 0
        else None,
        axis=1
    )

    throughput['tasks_completed'] = throughput['tasks_completed'].apply(lambda x: f"{int(x):,}")
    throughput['total_content_hours'] = throughput['total_content_hours'].apply(lambda x: f"{round(x):,}h")
    throughput['total_execution_time'] = throughput['total_execution_time'].apply(
        lambda x: f"{round(x):,}h" if pd.notna(x) else "N/A"
    )
    throughput['throughput_per_hour'] = throughput['throughput_per_hour'].apply(
        lambda x: f"{round(x, 1)}x" if pd.notna(x) else "N/A"
    )

    throughput.columns = ['Worker', 'Task Type', 'Tasks', 'Content Hours', 'Exec Time', 'Rate']
    throughput = throughput.sort_values(['Worker', 'Tasks'], ascending=[True, False])

    return throughput


# =============================================================================
# Helper Functions
# =============================================================================

def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string"""
    if seconds is None:
        return "N/A"
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_time_until(target_time_str: str) -> str:
    """Format time until a future datetime"""
    if not target_time_str:
        return "N/A"
    try:
        target = datetime.fromisoformat(target_time_str.replace('Z', '+00:00'))
        now = datetime.now(timezone.utc)
        delta = target - now

        if delta.total_seconds() < 0:
            return "overdue"

        total_seconds = delta.total_seconds()
        if total_seconds < 60:
            return f"in {int(total_seconds)}s"
        elif total_seconds < 3600:
            return f"in {int(total_seconds / 60)}m"
        elif total_seconds < 86400:
            hours = total_seconds / 3600
            return f"in {hours:.1f}h"
        else:
            days = total_seconds / 86400
            return f"in {days:.1f}d"
    except Exception:
        return "N/A"


def format_time_ago(timestamp_str: str) -> str:
    """Format time since a past datetime"""
    if not timestamp_str:
        return "Never"
    try:
        ts = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        now = datetime.now(timezone.utc)
        delta = now - ts

        total_seconds = delta.total_seconds()
        if total_seconds < 60:
            return f"{int(total_seconds)}s ago"
        elif total_seconds < 3600:
            return f"{int(total_seconds / 60)}m ago"
        elif total_seconds < 86400:
            hours = total_seconds / 3600
            return f"{hours:.1f}h ago"
        else:
            days = total_seconds / 86400
            return f"{days:.1f}d ago"
    except Exception:
        return timestamp_str[:16] if timestamp_str else "N/A"


def format_schedule_description(schedule_config: dict) -> str:
    """Format schedule configuration to human-readable description"""
    schedule_type = schedule_config.get('type', 'interval')

    if schedule_type == 'time_of_day':
        hours = schedule_config.get('hours', [0])
        minutes = schedule_config.get('minutes', [0])
        days_interval = schedule_config.get('days_interval', 1)

        times = []
        for h in hours:
            for m in minutes:
                times.append(f"{h:02d}:{m:02d}")

        times_str = ', '.join(times)

        if days_interval == 1:
            return f"Daily at {times_str}"
        else:
            return f"Every {days_interval}d at {times_str}"

    elif schedule_type == 'interval':
        interval = schedule_config.get('interval_seconds', 3600)
        if interval < 3600:
            return f"Every {interval // 60}m"
        else:
            return f"Every {interval // 3600}h"

    elif schedule_type == 'run_then_wait':
        wait = schedule_config.get('wait_seconds', 3600)
        if wait < 3600:
            return f"Continuous (wait {wait // 60}m)"
        else:
            return f"Continuous (wait {wait // 3600}h)"

    return schedule_type


# =============================================================================
# Quick Status Tab Render Functions
# =============================================================================

def check_service_health_fast(url: str, endpoint: str = "/health") -> dict:
    """Check service health with short timeout - no caching"""
    try:
        response = requests.get(f"{url}{endpoint}", timeout=5.0)
        if response.status_code == 200:
            return {"status": "running"}
        else:
            return {"status": "unhealthy"}
    except requests.exceptions.ConnectionError:
        return {"status": "stopped"}
    except requests.exceptions.Timeout:
        return {"status": "timeout"}
    except Exception:
        return {"status": "unknown"}


def get_status_grid_css() -> str:
    """Return CSS for status grid"""
    return """
    <style>
        .status-row {
            display: flex;
            align-items: center;
            margin-bottom: 8px;
            gap: 8px;
        }
        .status-row-label {
            font-weight: 600;
            font-size: 12px;
            color: #666;
            min-width: 80px;
            text-align: right;
        }
        .status-row-items {
            display: flex;
            gap: 6px;
            flex-wrap: wrap;
        }
        .status-box {
            padding: 8px 14px;
            border-radius: 6px;
            text-align: center;
            font-weight: 600;
            font-size: 12px;
            color: white;
            cursor: pointer;
            transition: transform 0.1s, box-shadow 0.1s;
            user-select: none;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .status-box:hover {
            transform: translateY(-1px);
            box-shadow: 0 2px 6px rgba(0,0,0,0.2);
        }
        .status-box:active {
            transform: translateY(0);
        }
        .status-running { background-color: #2ecc71; }
        .status-idle { background-color: #3498db; }
        .status-stopped { background-color: #e74c3c; }
        .status-timeout { background-color: #f39c12; }
        .status-unhealthy { background-color: #e67e22; }
        .status-unknown { background-color: #95a5a6; }
        .status-checking { background-color: #bdc3c7; }
        .status-disabled { background-color: #7f8c8d; }
        .copy-toast {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: #333;
            color: white;
            padding: 10px 20px;
            border-radius: 6px;
            opacity: 0;
            transition: opacity 0.3s;
            z-index: 1000;
            pointer-events: none;
        }
        .copy-toast.show {
            opacity: 1;
        }
    </style>
    <script>
        function copyToClipboard(text, serviceName) {
            var textarea = document.createElement('textarea');
            textarea.value = text;
            textarea.style.position = 'fixed';
            textarea.style.left = '-9999px';
            document.body.appendChild(textarea);
            textarea.select();
            textarea.setSelectionRange(0, 99999);
            try {
                document.execCommand('copy');
                var toast = document.getElementById('copy-toast');
                toast.textContent = 'Copied: ' + serviceName;
                toast.classList.add('show');
                setTimeout(function() { toast.classList.remove('show'); }, 1500);
            } catch (err) {
                console.error('Copy failed', err);
            }
            document.body.removeChild(textarea);
        }
    </script>
    """


def build_status_row_html(label: str, services: list) -> str:
    """Build HTML for a single row of status boxes"""
    html = f'<div class="status-row"><div class="status-row-label">{label}</div><div class="status-row-items">'
    for svc in services:
        status_class = f"status-{svc.get('status', 'checking')}"
        log_cmd = svc.get('log_cmd', '')
        if log_cmd:
            escaped_cmd = log_cmd.replace('\\', '\\\\').replace("'", "\\'").replace('"', '\\"')
            onclick = f'''onclick="copyToClipboard('{escaped_cmd}', '{svc['name']}')"'''
        else:
            onclick = ''
        html += f'<div class="status-box {status_class}" {onclick}>{svc["name"]}</div>'
    html += '</div></div>'
    return html


def render_quick_status_tab():
    """Render the Quick Status tab - color-coded grid organized by category"""

    # Initialize status tracking for each group
    group_statuses = {}
    for group_name, services in QUICK_STATUS_GROUPS.items():
        group_statuses[group_name] = [
            {**svc, 'status': 'checking'}
            for svc in services
        ]

    # Create placeholder for the grid
    grid_placeholder = st.empty()

    # Render CSS and initial checking state
    def render_grid():
        html = get_status_grid_css()
        for group_name in ['Orchestrator', 'LLMs', 'Workers', 'Scheduled']:
            if group_name in group_statuses:
                html += build_status_row_html(group_name, group_statuses[group_name])
        html += '<div id="copy-toast" class="copy-toast"></div>'
        return html

    grid_placeholder.markdown(render_grid(), unsafe_allow_html=True)

    # Check services progressively (skip scheduled tasks - they use API)
    for group_name in ['Orchestrator', 'LLMs', 'Workers']:
        for i, svc in enumerate(group_statuses[group_name]):
            if 'url' in svc:
                result = check_service_health_fast(svc['url'], svc['endpoint'])
                group_statuses[group_name][i]['status'] = result.get('status', 'unknown')
                grid_placeholder.markdown(render_grid(), unsafe_allow_html=True)

    # Check scheduled tasks via API
    scheduled_tasks_data = fetch_api("/api/scheduled_tasks")
    if "error" not in scheduled_tasks_data:
        tasks = scheduled_tasks_data.get('tasks', {})
        for i, svc in enumerate(group_statuses['Scheduled']):
            task_id = svc.get('task_id')
            if task_id and task_id in tasks:
                task_info = tasks[task_id]
                if task_info.get('is_running'):
                    group_statuses['Scheduled'][i]['status'] = 'running'
                elif not task_info.get('enabled'):
                    group_statuses['Scheduled'][i]['status'] = 'disabled'
                elif task_info.get('last_run', {}).get('result') == 'success':
                    group_statuses['Scheduled'][i]['status'] = 'idle'
                elif task_info.get('last_run', {}).get('result') in ['failed', 'error']:
                    group_statuses['Scheduled'][i]['status'] = 'stopped'
                else:
                    group_statuses['Scheduled'][i]['status'] = 'unknown'
            else:
                group_statuses['Scheduled'][i]['status'] = 'unknown'
    else:
        for i in range(len(group_statuses['Scheduled'])):
            group_statuses['Scheduled'][i]['status'] = 'unknown'

    grid_placeholder.markdown(render_grid(), unsafe_allow_html=True)

    # Summary counts (only for HTTP services)
    all_http_services = []
    for group_name in ['Orchestrator', 'LLMs', 'Workers']:
        all_http_services.extend(group_statuses[group_name])

    running = sum(1 for s in all_http_services if s['status'] == 'running')
    stopped = sum(1 for s in all_http_services if s['status'] == 'stopped')

    st.caption(f"Services: {running} running, {stopped} stopped | Click to copy log command")


# =============================================================================
# Services Tab Render Functions
# =============================================================================

def render_scheduled_tasks_summary():
    """Render a summary of scheduled tasks at the top of Services tab"""
    st.subheader("Scheduled Tasks")

    tasks_data = fetch_api("/api/scheduled_tasks/status")

    if "error" in tasks_data:
        st.error(f"Failed to fetch scheduled tasks: {tasks_data['error']}")
        return

    tasks = tasks_data.get('tasks', {})
    if not tasks:
        st.info("No scheduled tasks configured")
        return

    # Build task status rows
    rows = []
    for task_id, task in tasks.items():
        last_run = task.get('last_run', {})
        last_time = last_run.get('time')
        result = last_run.get('result')
        duration = last_run.get('duration_seconds')
        summary = last_run.get('summary')
        error = last_run.get('error')
        is_running = task.get('is_running', False)
        screen_session = task.get('screen_session')
        exec_start = task.get('execution_start_time')

        # Format last run time or running duration
        if is_running and exec_start:
            try:
                dt = datetime.fromisoformat(exec_start.replace('Z', '+00:00'))
                running_for = datetime.now(timezone.utc) - dt
                if running_for.total_seconds() < 3600:
                    time_str = f"Running {int(running_for.total_seconds() / 60)}m"
                else:
                    time_str = f"Running {running_for.total_seconds() / 3600:.1f}h"
            except:
                time_str = "Running"
        elif last_time:
            try:
                dt = datetime.fromisoformat(last_time.replace('Z', '+00:00'))
                time_ago = datetime.now(timezone.utc) - dt
                if time_ago.total_seconds() < 3600:
                    time_str = f"{int(time_ago.total_seconds() / 60)}m ago"
                elif time_ago.total_seconds() < 86400:
                    time_str = f"{int(time_ago.total_seconds() / 3600)}h ago"
                else:
                    time_str = f"{int(time_ago.total_seconds() / 86400)}d ago"
            except:
                time_str = "Unknown"
        else:
            time_str = "Never"

        # Format duration
        if duration and not is_running:
            if duration < 60:
                dur_str = f"{duration:.0f}s"
            else:
                dur_str = f"{duration/60:.1f}m"
        else:
            dur_str = "-"

        # Status indicator with screen session info
        if is_running:
            if screen_session:
                status = f"🟢 {screen_session}"
            else:
                status = "🟢 Running"
        elif not task.get('enabled'):
            status = "⚫ Disabled"
        elif result == 'success':
            status = "✅ Success"
        elif result == 'failed':
            status = "❌ Failed"
        elif result == 'error':
            status = "⚠️ Error"
        elif result == 'unknown':
            status = "❓ Unknown"
        else:
            status = "⏳ Pending"

        # Build summary string from the summary dict
        summary_str = ""
        if is_running:
            summary_str = f"screen -r {screen_session}" if screen_session else "Running..."
        elif summary:
            # Extract key metrics based on task type
            if 'total_tasks_created' in summary:
                summary_str = f"{summary['total_tasks_created']} tasks created"
            elif 'total_segments_processed' in summary:
                summary_str = f"{summary['total_segments_processed']} segments"
            elif 'phase1_speakers_identified' in summary:
                summary_str = f"{summary.get('phase1_speakers_identified', 0)} speakers"
            elif 'phase2_evidence_certain' in summary:
                summary_str = f"{summary.get('phase2_evidence_certain', 0)} with evidence"
            else:
                # Generic: show first numeric value
                for k, v in summary.items():
                    if isinstance(v, (int, float)) and v > 0:
                        summary_str = f"{v} {k.replace('_', ' ')}"
                        break
        elif error:
            summary_str = f"Error: {error[:50]}..." if len(str(error)) > 50 else f"Error: {error}"

        rows.append({
            'Task': task.get('name', task_id),
            'Status': status,
            'Last Run': time_str,
            'Duration': dur_str,
            'Summary': summary_str or "-"
        })

    # Sort: running first, then by status
    df = pd.DataFrame(rows)
    # Custom sort: Running tasks first, then by status
    status_order = {'🟢': 0, '❌': 1, '⚠️': 2, '❓': 3, '✅': 4, '⏳': 5, '⚫': 6}
    df['_sort'] = df['Status'].apply(lambda x: status_order.get(x[:1] if x else '', 99))
    df = df.sort_values('_sort').drop('_sort', axis=1)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()


def render_services_tab():
    """Render the Services tab - system health and service status"""

    # Scheduled Tasks Summary at top
    render_scheduled_tasks_summary()

    # System Health Section
    st.subheader("System Health")

    health_data = fetch_api("/api/health")
    monitoring_stats = fetch_api("/api/monitoring/stats")

    col1, col2, col3 = st.columns(3)

    with col1:
        if "error" in health_data:
            st.error("OFFLINE")
            st.caption("Orchestrator")
        elif health_data.get('status') == 'healthy':
            st.success("HEALTHY")
            st.caption("Orchestrator")
        else:
            st.warning("DEGRADED")
            st.caption("Orchestrator")

    with col2:
        global_pause = monitoring_stats.get('global_pause', {})
        if "error" in monitoring_stats:
            st.warning("UNKNOWN")
            st.caption("Task Assignment")
        elif global_pause.get('is_paused'):
            st.warning("PAUSED")
            st.caption(f"Until: {global_pause.get('pause_until', 'N/A')[:19]}")
        else:
            st.success("ACTIVE")
            st.caption("Task Assignment")

    with col3:
        # LLM Balancer status
        balancer_health = check_service_health(LLM_BALANCER_URL)
        if balancer_health.get('status') == 'running':
            st.success("RUNNING")
        elif balancer_health.get('status') == 'stopped':
            st.error("STOPPED")
        else:
            st.warning(balancer_health.get('status', 'UNKNOWN').upper())
        st.caption("LLM Balancer")

    st.divider()

    # Core Infrastructure Services
    st.subheader("Infrastructure Services")

    services_data = []

    # Orchestrator
    services_data.append({
        'Service': 'Orchestrator',
        'Host': 'localhost',
        'Port': '8001',
        'Status': ':large_green_circle: Running' if 'error' not in health_data else ':red_circle: Stopped',
    })

    # LLM Balancer
    balancer_status = ':large_green_circle: Running' if balancer_health.get('status') == 'running' else ':red_circle: Stopped'
    services_data.append({
        'Service': 'LLM Balancer',
        'Host': 'localhost',
        'Port': '8002',
        'Status': balancer_status,
    })

    # Backend API
    backend_health = check_service_health("http://localhost:7999")
    backend_status = ':large_green_circle: Running' if backend_health.get('status') == 'running' else ':red_circle: Stopped'
    services_data.append({
        'Service': 'Backend API',
        'Host': 'localhost',
        'Port': '7999',
        'Status': backend_status,
    })

    services_df = pd.DataFrame(services_data)
    st.dataframe(services_df, use_container_width=True, hide_index=True)

    st.divider()

    # Model Servers
    st.subheader("Model Servers")

    model_server_data = []
    for name, config in MODEL_SERVERS.items():
        url = f"http://{config['host']}:{config['port']}"
        health = check_service_health(url)
        status = ':large_green_circle: Running' if health.get('status') == 'running' else ':red_circle: Stopped'
        model_server_data.append({
            'Server': name,
            'Host': config['host'],
            'Port': str(config['port']),
            'Tiers': ', '.join(config['tiers']),
            'Status': status,
        })

    model_df = pd.DataFrame(model_server_data)
    st.dataframe(model_df, use_container_width=True, hide_index=True)

    st.divider()

    # Worker Status
    st.subheader("Worker Status")

    workers_data = fetch_api("/api/workers")

    if "error" in workers_data:
        st.warning(f"Unable to fetch worker data: {workers_data.get('error')}")
    else:
        workers = workers_data.get('workers', {})

        if workers:
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Active Workers", workers_data.get('active_workers', 0))
            with col2:
                st.metric("Total Workers", workers_data.get('total_workers', 0))
            with col3:
                failed_count = sum(1 for w in workers.values() if w.get('status') == 'failed')
                unhealthy_count = sum(1 for w in workers.values() if w.get('status') == 'unhealthy')
                if failed_count > 0:
                    st.metric("Failed", failed_count, delta_color="inverse")
                elif unhealthy_count > 0:
                    st.metric("Unhealthy", unhealthy_count, delta_color="inverse")
                else:
                    st.metric("Issues", 0)
            with col4:
                processor_running = sum(
                    1 for w in workers.values()
                    if w.get('services', {}).get('task_processor', {}).get('status') == 'running'
                )
                st.metric("Processors Running", processor_running)

            # Worker table
            worker_rows = []
            for worker_id, worker_info in sorted(workers.items()):
                status = worker_info.get('status', 'unknown')
                services = worker_info.get('services', {})
                processor_info = services.get('task_processor', {})
                processor_status = processor_info.get('status', 'unknown')

                if status == 'active' and processor_status == 'running':
                    status_display = ":large_green_circle: Active"
                elif status == 'active':
                    status_display = ":yellow_circle: Active (no proc)"
                elif status == 'starting':
                    status_display = ":hourglass: Starting"
                elif status == 'failed':
                    status_display = ":red_circle: Failed"
                elif status == 'unhealthy':
                    status_display = ":orange_circle: Unhealthy"
                else:
                    status_display = ":white_circle: Unknown"

                if processor_status == 'running':
                    proc_display = f":white_check_mark: Port {processor_info.get('port', 8000)}"
                elif processor_status == 'stopped':
                    proc_display = ":octagonal_sign: Stopped"
                elif processor_status == 'starting':
                    proc_display = ":hourglass: Starting"
                else:
                    proc_display = ":question: Unknown"

                task_counts = worker_info.get('task_counts_by_type', {})
                task_summary = ', '.join([f"{k}:{v}" for k, v in task_counts.items()]) if task_counts else '-'

                last_heartbeat = worker_info.get('last_heartbeat')
                hb_display = format_time_ago(last_heartbeat) if last_heartbeat else "Never"

                worker_rows.append({
                    'Worker': worker_id,
                    'Status': status_display,
                    'Processor': proc_display,
                    'Tasks': f"{worker_info.get('active_tasks', 0)}/{worker_info.get('max_concurrent_tasks', 0)}",
                    'Running': task_summary[:30] + ('...' if len(task_summary) > 30 else ''),
                    'Heartbeat': hb_display
                })

            if worker_rows:
                df = pd.DataFrame(worker_rows)
                st.dataframe(df, use_container_width=True, hide_index=True)

            # Expandable details
            with st.expander("Worker Details", expanded=False):
                for worker_id, worker_info in sorted(workers.items()):
                    col1, col2 = st.columns([1, 3])

                    with col1:
                        st.markdown(f"**{worker_id}**")
                        status = worker_info.get('status', 'unknown')
                        if status == 'active':
                            st.success("ACTIVE")
                        elif status == 'starting':
                            st.info("STARTING")
                        elif status == 'failed':
                            st.error("FAILED")
                        elif status == 'unhealthy':
                            st.warning("UNHEALTHY")
                        else:
                            st.info("UNKNOWN")

                    with col2:
                        task_types = worker_info.get('task_types', [])
                        if task_types:
                            st.caption(f"Task types: {', '.join(task_types)}")

                        detail_cols = st.columns(4)

                        with detail_cols[0]:
                            st.markdown("**Active Tasks**")
                            st.text(f"{worker_info.get('active_tasks', 0)}/{worker_info.get('max_concurrent_tasks', 0)}")

                        with detail_cols[1]:
                            st.markdown("**Processor**")
                            services = worker_info.get('services', {})
                            proc = services.get('task_processor', {})
                            proc_status = proc.get('status', 'unknown')
                            st.text(f"{proc_status} (:{proc.get('port', 8000)})")

                        with detail_cols[2]:
                            st.markdown("**Model Server**")
                            model_srv = services.get('model_server', {})
                            if model_srv:
                                models = model_srv.get('models', [])
                                st.text(f"{model_srv.get('status', 'N/A')} ({len(models)} models)")
                            else:
                                st.text("Not configured")

                        with detail_cols[3]:
                            st.markdown("**Network**")
                            network = worker_info.get('network_monitoring', {})
                            if network:
                                st.text(f"Latency: {network.get('avg_latency_ms', 'N/A')}ms")
                            else:
                                st.text("N/A")

                        failure_info = worker_info.get('failure_info', {})
                        if failure_info:
                            failures = [f"{k}: {v.get('count', 0)}" for k, v in failure_info.items() if v.get('count', 0) > 0]
                            if failures:
                                st.caption(f":warning: Failures: {', '.join(failures)}")

                    st.markdown("---")
        else:
            st.info("No worker data available")


# =============================================================================
# Tasks Tab Render Functions
# =============================================================================

def render_tasks_tab():
    """Render the Tasks tab - throughput and task queue status"""

    # Throughput Chart (moved to top)
    st.subheader("Content Throughput (Last 24 Hours)")

    now = datetime.now(timezone.utc)
    start_dt = now - timedelta(hours=24)
    end_dt = now

    task_data = get_completed_tasks_with_duration(start_date=start_dt, end_date=end_dt)

    if task_data:
        fig = plot_throughput_over_time(task_data, 'hour')
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data to display throughput chart")
    else:
        st.info("No throughput data available for the last 24 hours")

    st.divider()

    # Task Queue Section
    st.subheader("Task Queue Status")

    task_queue_stats = get_task_queue_status()

    if task_queue_stats:
        # Build the table data matching the Worker Monitoring format
        task_queue_data = []

        for stats in task_queue_stats:
            # Format hours helper
            def fmt_hours(h):
                return f"{int(round(h)):,}h" if h >= 1 else f"{int(round(h * 60))}m" if h > 0 else "0h"

            task_queue_data.append({
                'Task Type': stats['task_type'],
                'Last Hour Rate': f"{fmt_hours(stats['last_hour_rate'])}/h",
                'Pending': f"{stats['pending_count']:,} ({fmt_hours(stats['pending_hours'])})",
                'Processing': f"{stats['processing_count']:,} ({fmt_hours(stats['processing_hours'])})",
                'Completed': f"{stats['completed_count']:,} ({fmt_hours(stats['completed_hours'])})",
                'Failed': f"{stats['failed_count']:,} ({fmt_hours(stats['failed_hours'])})",
            })

        task_df = pd.DataFrame(task_queue_data)
        st.dataframe(task_df, hide_index=True, use_container_width=True)
    else:
        st.info("No task queue data available")


# =============================================================================
# Logs Tab Render Functions
# =============================================================================

def get_log_file_path(log_config: dict) -> str:
    """Extract the log file path from the command."""
    command = log_config.get('command', '')
    # Extract path from "tail -f /path/to/file" or ssh command
    if 'tail -f ' in command:
        # Find the path after 'tail -f '
        parts = command.split('tail -f ')
        if len(parts) > 1:
            # Remove any trailing quotes
            path = parts[-1].strip().rstrip('"')
            return path
    return ''


def fetch_log_lines(log_config: dict, num_lines: int = 100) -> tuple[list[str], str | None]:
    """Fetch the last N lines from a log file."""
    import subprocess

    log_path = get_log_file_path(log_config)
    if not log_path:
        return [], "Could not determine log file path"

    try:
        if log_config.get('local', True):
            # Local file - use tail directly
            result = subprocess.run(
                ['tail', '-n', str(num_lines), log_path],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                return [], f"Error reading log: {result.stderr}"
            lines = result.stdout.strip().split('\n') if result.stdout.strip() else []
            return lines, None
        else:
            # Remote file - use ssh
            host = log_config.get('host', '')
            if not host:
                return [], "No host configured for remote log"

            result = subprocess.run(
                ['ssh', f'signal4@{host}', f'tail -n {num_lines} {log_path}'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                return [], f"Error reading remote log: {result.stderr}"
            lines = result.stdout.strip().split('\n') if result.stdout.strip() else []
            return lines, None

    except subprocess.TimeoutExpired:
        return [], "Timeout fetching log"
    except Exception as e:
        return [], f"Error: {str(e)}"


def render_logs_tab():
    """Render the Logs tab - log viewer with selectable sources"""

    # Build flat list of log sources for selection
    log_options = {}
    for category, sources in LOG_SOURCES.items():
        for log_id, log_config in sources.items():
            log_options[log_id] = {
                'category': category,
                'name': log_config['name'],
                'config': log_config
            }

    # Two-column layout: sidebar for selection, main area for log display
    col_select, col_logs = st.columns([1, 4])

    with col_select:
        st.subheader("Log Sources")

        # Group logs by category
        selected_log = st.session_state.get('selected_log', 'orchestrator')

        for category, sources in LOG_SOURCES.items():
            st.markdown(f"**{category}**")
            for log_id, log_config in sources.items():
                is_selected = selected_log == log_id
                button_type = "primary" if is_selected else "secondary"
                if st.button(
                    log_config['name'],
                    key=f"log_btn_{log_id}",
                    use_container_width=True,
                    type=button_type
                ):
                    st.session_state.selected_log = log_id
                    st.rerun()
            st.markdown("")  # Spacing between categories

    with col_logs:
        if selected_log and selected_log in log_options:
            log_info = log_options[selected_log]
            log_config = log_info['config']

            # Header with log name and controls
            header_col1, header_col2, header_col3 = st.columns([3, 1, 1])

            with header_col1:
                st.subheader(f"{log_info['name']}")
                location = "Local" if log_config.get('local', True) else f"Remote ({log_config.get('host', 'unknown')})"
                st.caption(f"{location} | {get_log_file_path(log_config)}")

            with header_col2:
                num_lines = st.selectbox(
                    "Lines",
                    options=[50, 100, 200, 500, 1000],
                    index=1,
                    key="log_num_lines",
                    label_visibility="collapsed"
                )

            with header_col3:
                load_clicked = st.button("Load Logs", key="load_logs", use_container_width=True)

            # Copy command button
            st.code(log_config['command'], language="bash")

            # Track loaded logs in session state
            log_state_key = f"loaded_log_{selected_log}"

            # Load logs only when button is clicked
            if load_clicked:
                with st.spinner("Loading logs..."):
                    lines, error = fetch_log_lines(log_config, num_lines)
                    st.session_state[log_state_key] = {'lines': lines, 'error': error, 'num_lines': num_lines}

            # Display logs if they've been loaded
            if log_state_key in st.session_state:
                cached = st.session_state[log_state_key]
                lines = cached['lines']
                error = cached['error']

                if error:
                    st.error(error)
                elif not lines:
                    st.info("No log entries found")
                else:
                    # Display logs in a scrollable container
                    # Use text_area for easy copying and scrolling
                    log_content = '\n'.join(lines)
                    st.text_area(
                        "Log Output",
                        value=log_content,
                        height=600,
                        key="log_display",
                        label_visibility="collapsed"
                    )

                    st.caption(f"Showing last {len(lines)} lines")
            else:
                st.info("Click 'Load Logs' to fetch log entries")
        else:
            st.info("Select a log source from the left panel")


# =============================================================================
# Projects Tab Render Functions
# =============================================================================

def render_projects_tab():
    """Render the Projects tab - pipeline progress and project status"""

    # Analysis Cache Status (primary view - shows what's ready for analysis)
    st.subheader("Analysis Cache Status")
    st.caption("Content available for semantic search and analysis (from embedding cache tables)")

    # Time window selector
    col_toggle, col_spacer = st.columns([1, 3])
    with col_toggle:
        time_window = st.radio(
            "Time Window",
            options=['30d', '7d'],
            horizontal=True,
            key='cache_time_window',
            label_visibility='collapsed'
        )

    # Fetch cache status for selected window
    cache_status = get_cache_table_status(time_window)

    if cache_status and 'error' not in cache_status:
        projects = cache_status.get('projects', {})

        # Get totals for the time window
        window_total = cache_status.get('window_total_content', 0)
        window_stitched = cache_status.get('window_stitched_content', 0)
        window_embedded = cache_status.get('window_embedded_content', 0)
        total_segments = cache_status.get('total_segments', 0)

        # Calculate percentages for content in the time window
        stitched_pct = (window_stitched / window_total * 100) if window_total > 0 else 0
        embedded_pct = (window_embedded / window_total * 100) if window_total > 0 else 0

        # Summary metrics row
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Stitched",
                f"{stitched_pct:.1f}%",
                delta=f"{window_stitched:,} of {window_total:,}",
                delta_color="off"
            )
        with col2:
            st.metric(
                "Embedded",
                f"{embedded_pct:.1f}%",
                delta=f"{window_embedded:,} of {window_total:,}",
                delta_color="off"
            )
        with col3:
            total_with_main = sum(p.get('with_main_embedding', 0) for p in projects.values())
            main_pct = (total_with_main / total_segments * 100) if total_segments > 0 else 0
            st.metric(
                "Main Embeddings",
                f"{main_pct:.1f}%",
                delta=f"{total_with_main:,} segments",
                delta_color="off"
            )
        with col4:
            total_with_alt = sum(p.get('with_alt_embedding', 0) for p in projects.values())
            alt_pct = (total_with_alt / total_segments * 100) if total_segments > 0 else 0
            st.metric(
                "Alt Embeddings",
                f"{alt_pct:.1f}%",
                delta=f"{total_with_alt:,} segments",
                delta_color="off"
            )

        # Show cache date range
        date_range = cache_status.get('date_range', {})
        cache_start = date_range.get('start', 'N/A')
        cache_end = date_range.get('end', 'N/A')
        st.caption(f"Date range: {cache_start} to {cache_end} | Cache table: `{cache_status.get('cache_table')}`")

        # Heatmap visualization (show_segments parameter no longer used)
        heatmap_fig = create_cache_heatmap(cache_status)
        if heatmap_fig:
            st.plotly_chart(heatmap_fig, use_container_width=True)
        else:
            st.info(f"No data available in {time_window} cache")

        # Per-project summary table
        with st.expander("Project Details", expanded=False):
            project_rows = []
            for project_name, pdata in sorted(projects.items()):
                project_rows.append({
                    'Project': project_name,
                    'Content': f"{pdata.get('content_count', 0):,}",
                    'Segments': f"{pdata.get('segment_count', 0):,}",
                    'Main Emb.': f"{pdata.get('with_main_embedding', 0):,}",
                    'Alt Emb.': f"{pdata.get('with_alt_embedding', 0):,}",
                    'Date Range': f"{pdata.get('earliest_date', 'N/A')} - {pdata.get('latest_date', 'N/A')}"
                })
            if project_rows:
                st.dataframe(pd.DataFrame(project_rows), hide_index=True, use_container_width=True)
    else:
        st.warning(f"Unable to load {time_window} cache status: {cache_status.get('error', 'Unknown error')}")

    st.divider()

    # Project Progress Section (existing pipeline view)
    st.subheader("Project Pipeline Progress")

    global_status = get_global_content_status()

    if not global_status or global_status.get('total_content', 0) == 0:
        st.info("No project data available")
        return

    project_data = global_status['project_data']
    total_content = global_status['total_content']

    # Summary metrics
    completed_items = sum(p['status_counts'].get('Segment Embeddings', 0) for p in project_data.values())
    in_progress = sum(
        p['status_counts'].get('Stitched', 0) +
        p['status_counts'].get('Diarized & Transcribed', 0) +
        p['status_counts'].get('Audio Extracted', 0) +
        p['status_counts'].get('Downloaded Only', 0)
        for p in project_data.values()
    )
    pending = sum(p['status_counts'].get('Pending Download', 0) for p in project_data.values())

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Content", f"{total_content:,}")
    with col2:
        completion_rate = (completed_items / total_content * 100) if total_content > 0 else 0
        st.metric("Completed", f"{completed_items:,} ({completion_rate:.1f}%)")
    with col3:
        st.metric("In Progress", f"{in_progress:,}")
    with col4:
        st.metric("Pending", f"{pending:,}")

    # Progress bars
    progress_fig = create_project_progress_bars(global_status)
    if progress_fig:
        st.plotly_chart(progress_fig, use_container_width=True)

    # Legend
    st.caption(
        "Legend: "
        "<span style='color:#e74c3c;'>Pending Download</span> | "
        "<span style='color:#95a5a6;'>Downloaded</span> | "
        "<span style='color:#e67e22;'>Audio Extracted</span> | "
        "<span style='color:#9b59b6;'>Diarized & Transcribed</span> | "
        "<span style='color:#2ecc71;'>Stitched</span> | "
        "<span style='color:#155724;'>Embedded</span>",
        unsafe_allow_html=True
    )

    st.divider()

    # Pipeline Progress Details
    st.subheader("Pipeline Progress Details")

    progress = get_pipeline_progress_from_db()

    if not progress or 'error' in progress:
        st.warning(f"No progress data available: {progress.get('error', 'Unknown error')}")
    else:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Embeddings**")
            embedding = progress.get('embedding', {})
            primary = embedding.get('primary', {})
            alt = embedding.get('alternative', {})

            primary_pct = primary.get('percent', 0)
            st.metric(
                "Primary (0.6B)",
                f"{primary.get('completed', 0):,} / {embedding.get('total_segments', 0):,}",
                f"{primary_pct:.1f}%"
            )
            st.progress(min(primary_pct / 100, 1.0))

            alt_pct = alt.get('percent', 0)
            st.metric(
                "Alternative (4B)",
                f"{alt.get('completed', 0):,} / {embedding.get('total_segments', 0):,}",
                f"{alt_pct:.1f}%"
            )
            st.progress(min(alt_pct / 100, 1.0))

        with col2:
            st.markdown("**Speaker Identification**")
            speaker = progress.get('speaker_identification', {})
            identity = speaker.get('with_identity', {})
            identities = speaker.get('identities', {})

            identity_pct = identity.get('percent', 0)
            st.metric(
                "Speakers Identified",
                f"{identity.get('count', 0):,} / {speaker.get('total_speakers', 0):,}",
                f"{identity_pct:.2f}%"
            )
            st.progress(min(identity_pct / 100, 1.0))

            st.metric(
                "Speaker Identities",
                f"{identities.get('total', 0):,}",
                f"Named: {identities.get('named', 0):,}"
            )

        with col3:
            st.markdown("**Content**")
            content = progress.get('content', {})

            embedded_pct = content.get('percent_embedded', 0)
            st.metric(
                "Content Embedded",
                f"{content.get('embedded', 0):,} / {content.get('total', 0):,}",
                f"{embedded_pct:.1f}%"
            )
            st.progress(min(embedded_pct / 100, 1.0))

            st.metric("Stitched", f"{content.get('stitched', 0):,}")
            st.metric("Needs Embedding", f"{content.get('needs_embedding', 0):,}")

    st.divider()

    # Worker Throughput Details
    st.subheader("Worker Throughput (Last 24 Hours)")

    now = datetime.now(timezone.utc)
    start_dt = now - timedelta(hours=24)
    end_dt = now

    task_data = get_completed_tasks_with_duration(start_date=start_dt, end_date=end_dt)

    if task_data:
        throughput_df = display_worker_throughput_table(task_data)
        if throughput_df is not None:
            st.dataframe(throughput_df, hide_index=True, use_container_width=True)
        else:
            st.info("No throughput data with duration information available")
    else:
        st.info("No worker throughput data available")


# =============================================================================
# Sidebar & Header
# =============================================================================

def render_header():
    """Render dashboard header with refresh controls"""
    col1, col2 = st.columns([3, 1])

    with col1:
        st.caption(f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")

    with col2:
        if st.button("Refresh Now", use_container_width=True):
            st.cache_data.clear()
            st.rerun()




# =============================================================================
# Main Application
# =============================================================================

def main():
    st.set_page_config(
        page_title="System Monitoring Dashboard",
        page_icon=":",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Custom CSS
    st.markdown("""
    <style>
        .stMetric {
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 5px;
        }
        div[data-testid="stExpander"] details summary p {
            font-size: 1rem;
            font-weight: 600;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding-left: 20px;
            padding-right: 20px;
        }
    </style>
    """, unsafe_allow_html=True)

    # Header
    render_header()

    st.divider()

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Tasks", "Services", "Projects", "Logs"])

    with tab1:
        render_tasks_tab()

    with tab2:
        render_services_tab()

    with tab3:
        render_projects_tab()

    with tab4:
        render_logs_tab()


if __name__ == "__main__":
    main()
