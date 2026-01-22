"""
Report Service
==============

Service for generating public daily report data.
Provides read-only statistics and content summaries for the public report page.
"""

import time
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy import func, desc, text

from ..utils.backend_logger import get_logger
logger = get_logger("report_service")

from src.database.session import get_session
from src.database.models import Content, Channel, Speaker, SpeakerIdentity, EmbeddingSegment


class ReportService:
    """Service for generating public report data (read-only)"""

    def get_platform_stats(
        self,
        projects: Optional[List[str]] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get platform statistics by project.

        Args:
            projects: List of project names to include (None = all)
            days: Time window in days (default: 30)

        Returns:
            Dict with stats per project and totals
        """
        start_time = time.time()

        with get_session() as session:
            cutoff_date = datetime.utcnow() - timedelta(days=days)

            # Build project filter
            project_filter = text("c.is_embedded = true AND c.publish_date >= :cutoff")
            params = {"cutoff": cutoff_date}

            if projects:
                # Filter to specific projects
                project_conditions = " OR ".join([
                    f"c.projects @> ARRAY['{p}']::varchar[]" for p in projects
                ])
                project_filter = text(f"c.is_embedded = true AND c.publish_date >= :cutoff AND ({project_conditions})")

            # Query stats grouped by platform
            stats_query = text(f"""
                SELECT
                    c.platform,
                    COUNT(DISTINCT c.id) as episode_count,
                    COUNT(DISTINCT c.channel_id) as channel_count,
                    COALESCE(SUM(c.duration), 0) as total_duration_seconds
                FROM content c
                WHERE {project_filter.text}
                GROUP BY c.platform
                ORDER BY episode_count DESC
            """)

            results = session.execute(stats_query, params).fetchall()

            # Format results
            stats_by_platform = {}
            total_episodes = 0
            total_channels = 0
            total_duration = 0

            for row in results:
                platform = row.platform or "unknown"
                stats_by_platform[platform] = {
                    "episodes": row.episode_count,
                    "channels": row.channel_count,
                    "duration_hours": round(row.total_duration_seconds / 3600, 1)
                }
                total_episodes += row.episode_count
                total_channels += row.channel_count
                total_duration += row.total_duration_seconds

            # Get total unique speakers (those with identities)
            speaker_query = text("""
                SELECT COUNT(DISTINCT si.id) as speaker_count
                FROM speaker_identities si
                WHERE si.is_active = true
            """)
            speaker_result = session.execute(speaker_query).fetchone()
            total_speakers = speaker_result.speaker_count if speaker_result else 0

            processing_time = (time.time() - start_time) * 1000

            return {
                "time_window_days": days,
                "projects_filter": projects,
                "by_platform": stats_by_platform,
                "totals": {
                    "episodes": total_episodes,
                    "channels": total_channels,
                    "speakers": total_speakers,
                    "duration_hours": round(total_duration / 3600, 1)
                },
                "processing_time_ms": round(processing_time, 1)
            }

    def get_recent_content(
        self,
        projects: Optional[List[str]] = None,
        days: int = 7,
        limit: int = 20
    ) -> Dict[str, Any]:
        """
        Get recent content/episodes.

        Args:
            projects: List of project names to include (None = all)
            days: Time window in days (default: 7)
            limit: Maximum number of episodes to return

        Returns:
            Dict with recent episodes
        """
        start_time = time.time()

        with get_session() as session:
            cutoff_date = datetime.utcnow() - timedelta(days=days)

            # Build project filter for SQL
            if projects:
                project_conditions = " OR ".join([
                    f"c.projects @> ARRAY['{p}']::varchar[]" for p in projects
                ])
                project_clause = f"AND ({project_conditions})"
            else:
                project_clause = ""

            # Query recent content with channel info
            recent_query = text(f"""
                SELECT
                    c.id,
                    c.content_id,
                    c.title,
                    c.channel_name,
                    c.channel_id,
                    c.platform,
                    c.publish_date,
                    c.duration,
                    c.description,
                    ch.importance_score,
                    ch.channel_key,
                    CASE WHEN ch.platform_metadata->>'image_url' IS NOT NULL
                         AND ch.platform_metadata->>'image_url' <> ''
                         THEN true ELSE false END as has_channel_image
                FROM content c
                LEFT JOIN channels ch ON c.channel_id = ch.id
                WHERE c.is_embedded = true
                    AND c.publish_date >= :cutoff
                    {project_clause}
                ORDER BY c.publish_date DESC
                LIMIT :limit
            """)

            results = session.execute(
                recent_query,
                {"cutoff": cutoff_date, "limit": limit}
            ).fetchall()

            episodes = []
            for row in results:
                # Generate thumbnail URL
                thumbnail_url = self._get_thumbnail_url(
                    row.content_id,
                    row.platform,
                    row.channel_key,
                    row.has_channel_image if hasattr(row, 'has_channel_image') else False
                )

                episodes.append({
                    "content_id": row.content_id,
                    "title": row.title or "Untitled",
                    "channel_name": row.channel_name or "Unknown Channel",
                    "platform": row.platform or "unknown",
                    "publish_date": row.publish_date.isoformat() if row.publish_date else None,
                    "duration_minutes": round(row.duration / 60, 1) if row.duration else None,
                    "description_short": self._truncate_description(row.description),
                    "thumbnail_url": thumbnail_url
                })

            processing_time = (time.time() - start_time) * 1000

            return {
                "time_window_days": days,
                "projects_filter": projects,
                "total_returned": len(episodes),
                "episodes": episodes,
                "processing_time_ms": round(processing_time, 1)
            }

    def get_top_channels(
        self,
        projects: Optional[List[str]] = None,
        days: int = 30,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Get top channels by content volume.

        Args:
            projects: List of project names to include (None = all)
            days: Time window in days (default: 30)
            limit: Maximum number of channels to return

        Returns:
            Dict with top channels
        """
        start_time = time.time()

        with get_session() as session:
            cutoff_date = datetime.utcnow() - timedelta(days=days)

            # Build project filter
            if projects:
                project_conditions = " OR ".join([
                    f"c.projects @> ARRAY['{p}']::varchar[]" for p in projects
                ])
                project_clause = f"AND ({project_conditions})"
            else:
                project_clause = ""

            # Query top channels by episode count
            channels_query = text(f"""
                SELECT
                    ch.id,
                    ch.display_name,
                    ch.channel_key,
                    ch.platform,
                    ch.importance_score,
                    COUNT(c.id) as episode_count,
                    COALESCE(SUM(c.duration), 0) as total_duration
                FROM channels ch
                JOIN content c ON ch.id = c.channel_id
                WHERE c.is_embedded = true
                    AND c.publish_date >= :cutoff
                    {project_clause}
                GROUP BY ch.id, ch.display_name, ch.channel_key, ch.platform, ch.importance_score
                ORDER BY episode_count DESC
                LIMIT :limit
            """)

            results = session.execute(
                channels_query,
                {"cutoff": cutoff_date, "limit": limit}
            ).fetchall()

            channels = []
            for row in results:
                channels.append({
                    "channel_id": row.id,
                    "name": row.display_name,
                    "platform": row.platform,
                    "episode_count": row.episode_count,
                    "total_duration_hours": round(row.total_duration / 3600, 1),
                    "importance_score": float(row.importance_score) if row.importance_score else None
                })

            processing_time = (time.time() - start_time) * 1000

            return {
                "time_window_days": days,
                "projects_filter": projects,
                "channels": channels,
                "processing_time_ms": round(processing_time, 1)
            }

    def get_top_speakers(
        self,
        projects: Optional[List[str]] = None,
        days: int = 30,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Get top speakers by appearances.

        Args:
            projects: List of project names to include (None = all)
            days: Time window in days (default: 30)
            limit: Maximum number of speakers to return

        Returns:
            Dict with top speakers
        """
        start_time = time.time()

        with get_session() as session:
            cutoff_date = datetime.utcnow() - timedelta(days=days)

            # Build project filter
            if projects:
                project_conditions = " OR ".join([
                    f"c.projects @> ARRAY['{p}']::varchar[]" for p in projects
                ])
                project_clause = f"AND ({project_conditions})"
            else:
                project_clause = ""

            # Query top speakers by episode appearances
            speakers_query = text(f"""
                SELECT
                    si.id,
                    si.primary_name,
                    si.occupation,
                    si.role,
                    si.total_episodes,
                    COUNT(DISTINCT c.id) as recent_episodes
                FROM speaker_identities si
                JOIN speakers s ON s.speaker_identity_id = si.id
                JOIN content c ON s.content_id = c.content_id
                WHERE si.is_active = true
                    AND c.is_embedded = true
                    AND c.publish_date >= :cutoff
                    {project_clause}
                GROUP BY si.id, si.primary_name, si.occupation, si.role, si.total_episodes
                ORDER BY recent_episodes DESC
                LIMIT :limit
            """)

            results = session.execute(
                speakers_query,
                {"cutoff": cutoff_date, "limit": limit}
            ).fetchall()

            speakers = []
            for row in results:
                speakers.append({
                    "speaker_id": row.id,
                    "name": row.primary_name or "Unknown Speaker",
                    "occupation": row.occupation,
                    "role": row.role,
                    "recent_appearances": row.recent_episodes,
                    "total_appearances": row.total_episodes
                })

            processing_time = (time.time() - start_time) * 1000

            return {
                "time_window_days": days,
                "projects_filter": projects,
                "speakers": speakers,
                "processing_time_ms": round(processing_time, 1)
            }

    def get_daily_report(
        self,
        projects: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get combined daily report data.

        Args:
            projects: List of project names to include (None = all)

        Returns:
            Combined report with stats, recent content, top channels, and speakers
        """
        start_time = time.time()

        # Gather all report sections
        stats = self.get_platform_stats(projects=projects, days=30)
        recent = self.get_recent_content(projects=projects, days=7, limit=10)
        channels = self.get_top_channels(projects=projects, days=30, limit=5)
        speakers = self.get_top_speakers(projects=projects, days=30, limit=5)

        total_time = (time.time() - start_time) * 1000

        return {
            "report_date": datetime.utcnow().isoformat(),
            "projects_filter": projects,
            "stats": stats,
            "recent_content": recent,
            "top_channels": channels,
            "top_speakers": speakers,
            "total_processing_time_ms": round(total_time, 1)
        }

    def _get_thumbnail_url(
        self,
        content_id: str,
        platform: str,
        channel_key: Optional[str] = None,
        has_channel_image: bool = False
    ) -> Optional[str]:
        """Generate thumbnail URL based on platform."""
        if platform == 'youtube' and content_id and not content_id.startswith('pod_'):
            return f"https://img.youtube.com/vi/{content_id}/mqdefault.jpg"
        if platform == 'podcast' and channel_key and has_channel_image:
            from urllib.parse import quote
            encoded_key = quote(channel_key, safe='')
            return f"/api/images/channels/{encoded_key}.jpg"
        return None

    def _truncate_description(
        self,
        description: Optional[str],
        max_length: int = 150
    ) -> Optional[str]:
        """Truncate description to max length, ending at word boundary."""
        if not description:
            return None
        if len(description) <= max_length:
            return description
        truncated = description[:max_length]
        last_space = truncated.rfind(' ')
        if last_space > max_length * 0.7:
            truncated = truncated[:last_space]
        return truncated.rstrip() + '...'


# Singleton instance
_report_service = None

def get_report_service() -> ReportService:
    """Get or create the report service singleton."""
    global _report_service
    if _report_service is None:
        _report_service = ReportService()
    return _report_service
