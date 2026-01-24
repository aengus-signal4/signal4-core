"""
Logs tab for the system monitoring dashboard.

Provides a log viewer with selectable sources from local and remote machines.
"""

import subprocess

import streamlit as st

from ..config import LOG_SOURCES


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
