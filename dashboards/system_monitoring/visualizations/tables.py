"""
Table visualization functions for the system monitoring dashboard.
"""

import pandas as pd


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
