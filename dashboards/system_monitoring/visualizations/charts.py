"""
Chart visualization functions for the system monitoring dashboard.
"""

from datetime import datetime, timezone, timedelta

import pandas as pd
import plotly.graph_objects as go

from src.utils.logger import setup_worker_logger
from ..config import TASK_COLORS

logger = setup_worker_logger('system_monitoring')


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
