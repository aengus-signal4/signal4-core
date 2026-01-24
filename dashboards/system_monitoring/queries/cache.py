"""
Embedding cache database queries for the system monitoring dashboard.
"""

from datetime import datetime, timezone, timedelta

import streamlit as st
from sqlalchemy import text

from src.database.session import get_session
from src.utils.logger import setup_worker_logger
from ..config import load_config

logger = setup_worker_logger('system_monitoring')


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
