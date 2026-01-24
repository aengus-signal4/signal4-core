"""
Quick Status tab for the system monitoring dashboard.

Shows a color-coded grid of all critical services organized by category.
"""

import requests
import streamlit as st

from ..config import QUICK_STATUS_GROUPS
from ..utils.api import fetch_api


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
