#!/usr/bin/env python3
"""
System Monitoring Dashboard - Unified View
==========================================

A single-pane-of-glass view combining:
- System health and orchestrator status (from orchestrator_monitoring.py)
- Worker throughput by content duration (from worker_monitoring_v2.py)
- Project pipeline progress (from project_monitoring.py)
- Automation services status

This dashboard provides a comprehensive view of the entire Signal4 processing system.
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
import time

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.database.session import get_session
from src.database.models import Content, TaskQueue, ContentChunk
from src.utils.logger import setup_worker_logger

logger = setup_worker_logger('system_monitoring')

# Configuration
ORCHESTRATOR_API_URL = "http://localhost:8001"
DEFAULT_REFRESH_INTERVAL = 60  # seconds

# Pipeline stage colors (from project_monitoring)
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

# Task type colors (from worker_monitoring_v2)
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
# API Functions (from orchestrator_monitoring.py)
# =============================================================================

def fetch_api(endpoint: str) -> dict:
    """Fetch data from orchestrator API with error handling"""
    try:
        response = requests.get(f"{ORCHESTRATOR_API_URL}{endpoint}", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Connection refused - Orchestrator not running"}
    except requests.exceptions.Timeout:
        return {"error": "Request timeout"}
    except Exception as e:
        return {"error": str(e)}


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
                WHERE status = 'active'
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

                # Calculate counts for each status
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

            # Calculate overall progress
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
# Render Functions
# =============================================================================

def render_header():
    """Render dashboard header with title and refresh controls"""
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.title("System Monitoring Dashboard")

    with col2:
        st.caption(f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")

    with col3:
        if st.button("Refresh Now", use_container_width=True):
            st.cache_data.clear()
            st.rerun()


def render_system_health():
    """Render system health overview section"""
    st.subheader("System Health")

    health_data = fetch_api("/api/health")
    monitoring_stats = fetch_api("/api/monitoring/stats")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if "error" in health_data:
            st.error("OFFLINE")
            st.caption(health_data.get("error", "Unknown error"))
        elif health_data.get('status') == 'healthy':
            st.success("HEALTHY")
            st.caption("Orchestrator Status")
        else:
            st.warning("DEGRADED")
            st.caption("Orchestrator Status")

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
        services = monitoring_stats.get('services', {})
        if "error" in monitoring_stats:
            st.warning("UNKNOWN")
            st.caption("Services")
        elif services:
            service_statuses = []
            for name, status in services.items():
                if isinstance(status, dict):
                    service_statuses.append(status.get('status', 'unknown'))

            if all(s == 'running' for s in service_statuses):
                st.success("ALL RUNNING")
            else:
                st.warning("ISSUES")
            st.caption(f"{len(service_statuses)} Services")
        else:
            st.info("NO DATA")
            st.caption("Services")

    with col4:
        timestamp = health_data.get('timestamp', '')
        if timestamp and "error" not in health_data:
            st.metric("Uptime Check", timestamp[:19].replace('T', ' '))
        else:
            st.caption("No uptime data")


def render_worker_status():
    """Render worker status grid with processor details"""
    st.subheader("Worker Status")

    workers_data = fetch_api("/api/workers")

    if "error" in workers_data:
        st.warning(f"Unable to fetch worker data: {workers_data.get('error')}")
        return

    workers = workers_data.get('workers', {})

    if not workers:
        st.info("No worker data available")
        return

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
        # Count workers with running processors
        processor_running = sum(
            1 for w in workers.values()
            if w.get('services', {}).get('task_processor', {}).get('status') == 'running'
        )
        st.metric("Processors Running", processor_running)

    # Worker table with processor status
    worker_rows = []
    for worker_id, worker_info in sorted(workers.items()):
        status = worker_info.get('status', 'unknown')

        # Get processor status from services
        services = worker_info.get('services', {})
        processor_info = services.get('task_processor', {})
        processor_status = processor_info.get('status', 'unknown')

        # Determine status display
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

        # Processor status
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

        # Format last heartbeat
        last_heartbeat = worker_info.get('last_heartbeat')
        if last_heartbeat:
            try:
                hb_dt = datetime.fromisoformat(last_heartbeat.replace('Z', '+00:00'))
                age = (datetime.now(timezone.utc) - hb_dt).total_seconds()
                if age < 60:
                    hb_display = f"{int(age)}s ago"
                elif age < 3600:
                    hb_display = f"{int(age/60)}m ago"
                else:
                    hb_display = f"{age/3600:.1f}h ago"
            except Exception:
                hb_display = last_heartbeat[:19] if len(last_heartbeat) >= 19 else last_heartbeat
        else:
            hb_display = "Never"

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

    # Expandable details for each worker
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
                # Task types
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

                # Show failure info if any
                failure_info = worker_info.get('failure_info', {})
                if failure_info:
                    failures = [f"{k}: {v.get('count', 0)}" for k, v in failure_info.items() if v.get('count', 0) > 0]
                    if failures:
                        st.caption(f":warning: Failures: {', '.join(failures)}")

            st.markdown("---")


def render_task_queue_overview():
    """Render task queue overview section"""
    st.subheader("Task Queue Overview")

    task_stats = get_task_stats()
    recent_throughput = get_recent_throughput(minutes=60)

    # Summary metrics
    total_pending = sum(stats.get('pending', {}).get('count', 0) for stats in task_stats.values())
    total_processing = sum(stats.get('processing', {}).get('count', 0) for stats in task_stats.values())
    total_completed = sum(stats.get('completed', {}).get('count', 0) for stats in task_stats.values())
    total_failed = sum(stats.get('failed', {}).get('count', 0) for stats in task_stats.values())

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Pending", f"{total_pending:,}")
    with col2:
        st.metric("Processing", f"{total_processing:,}")
    with col3:
        st.metric("Completed", f"{total_completed:,}")
    with col4:
        st.metric("Failed", f"{total_failed:,}")

    # Detailed breakdown by task type
    task_queue_data = []
    default_stat = {'count': 0, 'hours': 0.0}

    for task_type in sorted(task_stats.keys()):
        stats = task_stats.get(task_type, {})
        total_count = sum(s['count'] for s in stats.values())

        if total_count > 0:
            pending = stats.get('pending', default_stat)
            processing = stats.get('processing', default_stat)
            completed = stats.get('completed', default_stat)
            failed = stats.get('failed', default_stat)
            throughput_60min = recent_throughput.get(task_type, 0.0)

            task_queue_data.append({
                'Task Type': task_type,
                'Pending': f"{pending['count']:,}",
                'Processing': f"{processing['count']:,}",
                'Completed': f"{completed['count']:,}",
                'Failed': f"{failed['count']:,}",
                'Last Hour': f"{round(throughput_60min, 1)}h"
            })

    if task_queue_data:
        task_df = pd.DataFrame(task_queue_data)
        st.dataframe(task_df, hide_index=True, use_container_width=True)


def render_project_progress():
    """Render project pipeline progress section"""
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


def render_throughput_charts(task_data, time_interval):
    """Render throughput charts section"""
    st.subheader("Content Throughput")

    if not task_data:
        st.info("No throughput data available for the selected period")
        return

    fig = plot_throughput_over_time(task_data, time_interval)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data to display throughput chart")


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


def render_scheduled_tasks():
    """Render comprehensive scheduled tasks monitoring section"""
    st.subheader("Scheduled Tasks")

    scheduled_tasks_data = fetch_api("/api/scheduled_tasks")

    if "error" in scheduled_tasks_data:
        st.warning(f"Unable to fetch scheduled tasks: {scheduled_tasks_data.get('error')}")
        return

    tasks = scheduled_tasks_data.get('tasks', {})
    if not tasks:
        st.info("No scheduled tasks configured")
        return

    # Summary metrics
    total_tasks = len(tasks)
    enabled_tasks = sum(1 for t in tasks.values() if t and t.get('enabled', False))
    running_tasks = sum(1 for t in tasks.values() if t and t.get('is_running', False))

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Tasks", total_tasks)
    with col2:
        st.metric("Enabled", enabled_tasks)
    with col3:
        if running_tasks > 0:
            st.metric("Running", running_tasks)
        else:
            st.metric("Running", 0)
    with col4:
        # Count tasks with recent failures
        failed_recently = sum(
            1 for t in tasks.values()
            if t and t.get('last_run', {}).get('result') in ['failed', 'error']
        )
        if failed_recently > 0:
            st.metric("Recent Failures", failed_recently, delta_color="inverse")
        else:
            st.metric("Recent Failures", 0)

    # Detailed task table
    st.markdown("---")

    task_rows = []
    for task_id, task_info in sorted(tasks.items(), key=lambda x: x[1].get('name', x[0]) if x[1] else x[0]):
        if not task_info:
            continue

        last_run = task_info.get('last_run', {})
        schedule = task_info.get('schedule', {})
        schedule_config = schedule.get('config', {})

        # Determine status
        if task_info.get('is_running'):
            status = "Running"
            status_icon = ":large_green_circle:"
        elif not task_info.get('enabled'):
            status = "Disabled"
            status_icon = ":white_circle:"
        elif last_run.get('result') == 'success':
            status = "Success"
            status_icon = ":white_check_mark:"
        elif last_run.get('result') in ['failed', 'error']:
            status = "Failed"
            status_icon = ":x:"
        else:
            status = "Pending"
            status_icon = ":hourglass:"

        # Format last run time
        last_run_time = last_run.get('time')
        if last_run_time:
            try:
                last_dt = datetime.fromisoformat(last_run_time.replace('Z', '+00:00'))
                last_run_display = last_dt.strftime('%Y-%m-%d %H:%M')
            except Exception:
                last_run_display = last_run_time[:16] if len(last_run_time) >= 16 else last_run_time
        else:
            last_run_display = "Never"

        # Format next run
        next_run = task_info.get('next_run_time')
        if next_run and task_info.get('enabled'):
            next_run_display = format_time_until(next_run)
        else:
            next_run_display = "N/A" if not task_info.get('enabled') else "Pending"

        task_rows.append({
            'Status': f"{status_icon} {status}",
            'Task': task_info.get('name', task_id),
            'Schedule': format_schedule_description(schedule_config),
            'Last Run': last_run_display,
            'Duration': format_duration(last_run.get('duration_seconds')),
            'Next Run': next_run_display,
        })

    if task_rows:
        df = pd.DataFrame(task_rows)
        st.dataframe(df, hide_index=True, use_container_width=True)

    # Expandable details for each task
    with st.expander("Task Details", expanded=False):
        for task_id, task_info in sorted(tasks.items(), key=lambda x: x[1].get('name', x[0]) if x[1] else x[0]):
            if not task_info:
                continue

            last_run = task_info.get('last_run', {})
            schedule = task_info.get('schedule', {})
            executor = task_info.get('executor', {})

            col1, col2 = st.columns([1, 3])

            with col1:
                st.markdown(f"**{task_info.get('name', task_id)}**")
                if task_info.get('is_running'):
                    st.success("RUNNING")
                elif not task_info.get('enabled'):
                    st.warning("DISABLED")
                elif last_run.get('result') == 'success':
                    st.info("IDLE")
                elif last_run.get('result') in ['failed', 'error']:
                    st.error("FAILED")
                else:
                    st.info("PENDING")

            with col2:
                st.caption(task_info.get('description', ''))

                detail_cols = st.columns(4)
                with detail_cols[0]:
                    st.markdown("**Schedule**")
                    st.text(format_schedule_description(schedule.get('config', {})))

                with detail_cols[1]:
                    st.markdown("**Last Run**")
                    last_time = last_run.get('time')
                    if last_time:
                        try:
                            last_dt = datetime.fromisoformat(last_time.replace('Z', '+00:00'))
                            st.text(last_dt.strftime('%Y-%m-%d %H:%M'))
                        except Exception:
                            st.text(last_time[:16] if last_time else "Never")
                    else:
                        st.text("Never")

                with detail_cols[2]:
                    st.markdown("**Duration**")
                    st.text(format_duration(last_run.get('duration_seconds')))

                with detail_cols[3]:
                    st.markdown("**Next Run**")
                    next_run = task_info.get('next_run_time')
                    if next_run and task_info.get('enabled'):
                        st.text(format_time_until(next_run))
                    else:
                        st.text("N/A")

                # Show executor info
                executor_type = executor.get('type', 'cli')
                if executor_type == 'cli':
                    cmd = executor.get('command', '')
                    args = executor.get('args', [])
                    st.caption(f"Executor: `{cmd} {' '.join(args)}`")
                elif executor_type == 'sql':
                    func = executor.get('function', '')
                    st.caption(f"Executor: SQL function `{func}()`")

            st.markdown("---")


def render_automation_services():
    """Render automation services status section (legacy summary view)"""
    # This section is now replaced by render_scheduled_tasks()
    # Keeping for backward compatibility but can be removed
    pass


def render_pipeline_progress():
    """Render embedding and speaker identification progress"""
    with st.expander("Pipeline Progress Details", expanded=False):
        progress = get_pipeline_progress_from_db()

        if not progress or 'error' in progress:
            st.warning(f"No progress data available: {progress.get('error', 'Unknown error')}")
            return

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


def render_worker_throughput_details(task_data):
    """Render detailed worker throughput in an expander"""
    with st.expander("Worker Throughput Details", expanded=False):
        if not task_data:
            st.info("No worker throughput data available")
            return

        throughput_df = display_worker_throughput_table(task_data)
        if throughput_df is not None:
            st.dataframe(throughput_df, hide_index=True, use_container_width=True)
        else:
            st.info("No throughput data with duration information available")


def render_sidebar():
    """Render sidebar with filters and controls"""
    with st.sidebar:
        st.header("Filters & Controls")

        # Auto-refresh toggle
        auto_refresh = st.toggle("Auto-refresh", value=True)
        refresh_interval = st.slider("Refresh interval (seconds)", 30, 300, DEFAULT_REFRESH_INTERVAL, 30)

        st.divider()

        # Quick date presets
        st.subheader("Date Range")

        if "filter_start_date" not in st.session_state:
            st.session_state["filter_start_date"] = (datetime.now(timezone.utc) - timedelta(hours=24)).date()
        if "filter_end_date" not in st.session_state:
            st.session_state["filter_end_date"] = datetime.now(timezone.utc).date()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Today", use_container_width=True):
                st.session_state["filter_start_date"] = datetime.now(timezone.utc).date()
                st.session_state["filter_end_date"] = datetime.now(timezone.utc).date()
                st.rerun()
        with col2:
            if st.button("24h", use_container_width=True):
                st.session_state["filter_start_date"] = (datetime.now(timezone.utc) - timedelta(days=1)).date()
                st.session_state["filter_end_date"] = datetime.now(timezone.utc).date()
                st.rerun()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("7 days", use_container_width=True):
                st.session_state["filter_start_date"] = (datetime.now(timezone.utc) - timedelta(days=7)).date()
                st.session_state["filter_end_date"] = datetime.now(timezone.utc).date()
                st.rerun()
        with col2:
            if st.button("30 days", use_container_width=True):
                st.session_state["filter_start_date"] = (datetime.now(timezone.utc) - timedelta(days=30)).date()
                st.session_state["filter_end_date"] = datetime.now(timezone.utc).date()
                st.rerun()

        start_date = st.date_input("Start", value=st.session_state["filter_start_date"])
        end_date = st.date_input("End", value=st.session_state["filter_end_date"])

        st.session_state["filter_start_date"] = start_date
        st.session_state["filter_end_date"] = end_date

        st.divider()

        # Project filter
        st.subheader("Project Filter")
        config = load_config()
        active_projects = [p for p, s in config.get('active_projects', {}).items() if s.get('enabled', False)]

        selected_projects = st.multiselect(
            "Projects",
            active_projects,
            default=active_projects,
            help="Filter project progress view"
        )

        st.divider()

        # Time interval for charts
        st.subheader("Chart Settings")
        time_interval = st.radio(
            "Throughput interval",
            ["Hour", "Day"],
            horizontal=True,
            index=0
        )

        st.divider()

        # Clear cache button
        if st.button("Clear Cache", use_container_width=True):
            st.cache_data.clear()
            st.success("Cache cleared!")
            st.rerun()

        return {
            'auto_refresh': auto_refresh,
            'refresh_interval': refresh_interval,
            'start_date': start_date,
            'end_date': end_date,
            'selected_projects': selected_projects,
            'time_interval': time_interval.lower()
        }


# =============================================================================
# Main Application
# =============================================================================

def main():
    st.set_page_config(
        page_title="System Monitoring Dashboard",
        page_icon=":",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for cleaner appearance
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
    </style>
    """, unsafe_allow_html=True)

    # Render sidebar and get settings
    settings = render_sidebar()

    # Header
    render_header()

    st.divider()

    # System Health Summary
    render_system_health()

    st.divider()

    # Worker Status Grid
    render_worker_status()

    st.divider()

    # Task Queue Overview
    render_task_queue_overview()

    st.divider()

    # Project Progress
    render_project_progress()

    st.divider()

    # Throughput Charts
    now = datetime.now(timezone.utc)
    start_dt = datetime.combine(settings['start_date'], datetime.min.time()).replace(tzinfo=timezone.utc)
    end_dt = datetime.combine(settings['end_date'], datetime.max.time()).replace(tzinfo=timezone.utc)

    task_data = get_completed_tasks_with_duration(
        start_date=start_dt,
        end_date=end_dt
    )

    render_throughput_charts(task_data, settings['time_interval'])

    st.divider()

    # Scheduled Tasks Status (replaces legacy automation services)
    render_scheduled_tasks()

    # Expandable detailed sections
    st.divider()

    render_pipeline_progress()
    render_worker_throughput_details(task_data)

    # Auto-refresh
    if settings['auto_refresh']:
        time.sleep(settings['refresh_interval'])
        st.rerun()


if __name__ == "__main__":
    main()
