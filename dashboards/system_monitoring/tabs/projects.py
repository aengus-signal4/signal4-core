"""
Projects tab for the system monitoring dashboard.

Shows pipeline progress and project status including embedding cache.
"""

from datetime import datetime, timezone, timedelta

import pandas as pd
import streamlit as st

from ..queries.content import get_global_content_status, get_pipeline_progress_from_db
from ..queries.cache import get_cache_table_status
from ..queries.tasks import get_completed_tasks_with_duration
from ..visualizations.charts import create_cache_heatmap, create_project_progress_bars
from ..visualizations.tables import display_worker_throughput_table


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

        # Heatmap visualization
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
