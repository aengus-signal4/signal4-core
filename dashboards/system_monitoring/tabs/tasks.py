"""
Tasks tab for the system monitoring dashboard.

Shows throughput charts and task queue status.
"""

from datetime import datetime, timezone, timedelta

import pandas as pd
import streamlit as st

from ..queries.tasks import (
    get_task_queue_status,
    get_completed_tasks_with_duration,
    get_worker_performance_stats,
)
from ..visualizations.charts import plot_throughput_over_time
from ..utils.formatters import format_duration, format_hours


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
            task_queue_data.append({
                'Task Type': stats['task_type'],
                'Last Hour Rate': f"{format_hours(stats['last_hour_rate'])}/h",
                'Pending': f"{stats['pending_count']:,} ({format_hours(stats['pending_hours'])})",
                'Processing': f"{stats['processing_count']:,} ({format_hours(stats['processing_hours'])})",
                'Completed': f"{stats['completed_count']:,} ({format_hours(stats['completed_hours'])})",
                'Failed': f"{stats['failed_count']:,} ({format_hours(stats['failed_hours'])})",
            })

        task_df = pd.DataFrame(task_queue_data)
        st.dataframe(task_df, hide_index=True, use_container_width=True)
    else:
        st.info("No task queue data available")

    st.divider()

    # Worker Performance Assessment (optional)
    show_worker_perf = st.checkbox("Assess Performance by Worker", value=False, key="show_worker_performance")

    if show_worker_perf:
        st.subheader("Worker Performance (Last 2 Hours)")

        # Use the new optimized SQL aggregation function
        worker_stats = get_worker_performance_stats(hours=2)

        if worker_stats:
            perf_rows = []
            for stat in worker_stats:
                perf_rows.append({
                    'Worker': stat['worker_id'],
                    'Task Type': stat['task_type'],
                    'Tasks': stat['tasks_completed'],
                    'Content Hours': f"{stat['content_hours']:.1f}h" if stat['content_hours'] else "0h",
                    'Exec Time': f"{stat['exec_hours']:.2f}h" if stat['exec_hours'] else "N/A",
                    'Avg Task Time': format_duration(stat['avg_exec_seconds']) if stat['avg_exec_seconds'] else "N/A",
                    'Rate': f"{stat['rate']:.1f}x" if stat['rate'] else "N/A"
                })

            if perf_rows:
                perf_df = pd.DataFrame(perf_rows)
                st.dataframe(perf_df, hide_index=True, use_container_width=True)

                # Summary by worker (collapsed)
                with st.expander("Worker Summary", expanded=False):
                    # Aggregate by worker
                    worker_summary = {}
                    for stat in worker_stats:
                        wid = stat['worker_id']
                        if wid not in worker_summary:
                            worker_summary[wid] = {
                                'total_tasks': 0,
                                'total_content_hours': 0,
                                'total_exec_hours': 0
                            }
                        worker_summary[wid]['total_tasks'] += stat['tasks_completed']
                        worker_summary[wid]['total_content_hours'] += stat['content_hours'] or 0
                        worker_summary[wid]['total_exec_hours'] += stat['exec_hours'] or 0

                    summary_rows = []
                    for wid in sorted(worker_summary.keys()):
                        ws = worker_summary[wid]
                        overall_rate = None
                        if ws['total_exec_hours'] > 0:
                            overall_rate = ws['total_content_hours'] / ws['total_exec_hours']

                        summary_rows.append({
                            'Worker': wid,
                            'Total Tasks': ws['total_tasks'],
                            'Total Content Hours': f"{ws['total_content_hours']:.1f}h",
                            'Total Exec Time': f"{ws['total_exec_hours']:.2f}h" if ws['total_exec_hours'] > 0 else "N/A",
                            'Overall Rate': f"{overall_rate:.1f}x" if overall_rate else "N/A"
                        })

                    if summary_rows:
                        summary_df = pd.DataFrame(summary_rows)
                        st.dataframe(summary_df, hide_index=True, use_container_width=True)
            else:
                st.info("No worker performance data available for the last 2 hours")
        else:
            st.info("No completed tasks in the last 2 hours")
