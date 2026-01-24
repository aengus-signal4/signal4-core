"""
Services tab for the system monitoring dashboard.

Shows system health, scheduled tasks, model servers, and worker status.
"""

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from ..config import MODEL_SERVERS, LLM_BALANCER_URL
from ..utils.api import fetch_api, check_service_health
from ..utils.formatters import format_time_ago


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
