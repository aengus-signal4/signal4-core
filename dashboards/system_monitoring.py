#!/usr/bin/env python3
"""
System Monitoring Dashboard - Tabbed Interface
==============================================

A modular dashboard for monitoring the Signal4 content processing system.

Four tabs:
- Tasks: Content throughput charts and task queue status
- Services: System health, model servers, workers, scheduled tasks
- Projects: Pipeline progress, embeddings, speaker identification
- Logs: Log viewer with selectable sources

The dashboard is organized into modules under the `system_monitoring/` package:
- config.py: Configuration constants and loaders
- queries/: Database query functions (tasks, content, cache)
- tabs/: Tab rendering functions
- visualizations/: Charts and tables
- utils/: Helper functions (formatters, API)
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

import streamlit as st

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

# Import tab renderers from the modular package
from system_monitoring.tabs import (
    render_tasks_tab,
    render_services_tab,
    render_projects_tab,
    render_logs_tab,
)


def render_header():
    """Render dashboard header with refresh controls"""
    col1, col2 = st.columns([3, 1])

    with col1:
        st.caption(f"Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")

    with col2:
        if st.button("Refresh Now", use_container_width=True):
            st.cache_data.clear()
            st.rerun()


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
