"""
SegmentRetriever
================

Unified interface for fetching segments with flexible filtering.

This module provides a single source of truth for segment retrieval,
replacing the scattered fetching logic across multiple services.
"""

from typing import List, Optional, Tuple, Set
from datetime import datetime
from sqlalchemy import and_, or_
from sqlalchemy.orm import Session, joinedload

from ...database.connection import get_db
from src.database.models import EmbeddingSegment, Content, Speaker
from ...utils.backend_logger import get_logger

logger = get_logger("segment_retriever")

# Type alias for consistency with architecture doc
Segment = EmbeddingSegment


class SegmentRetriever:
    """Unified segment retrieval with flexible filtering."""

    def __init__(self, db: Optional[Session] = None):
        """
        Initialize retriever.

        Args:
            db: Optional database session. If not provided, will use get_db()
        """
        self._db = db
        self._session = None

    def _get_session(self) -> Session:
        """Get or create database session."""
        if self._session is None:
            if self._db is not None:
                self._session = self._db
            else:
                self._session = next(get_db())
        return self._session

    def fetch_by_filter(
        self,
        projects: Optional[List[str]] = None,
        languages: Optional[List[str]] = None,
        channels: Optional[List[str]] = None,
        channel_urls: Optional[List[str]] = None,
        speakers: Optional[List[str]] = None,
        content_ids: Optional[List[str]] = None,
        segment_ids: Optional[List[int]] = None,
        date_range: Optional[Tuple[datetime, datetime]] = None,
        stitch_versions: Optional[List[str]] = None,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        must_be_stitched: bool = True,
        must_be_embedded: bool = True,
        limit: Optional[int] = None
    ) -> List[Segment]:
        """
        Fetch segments matching all specified filters.

        All filters are combined with AND logic. List filters use OR within the list.

        Args:
            projects: Filter by project names (e.g., ["CPRMV", "Europe"])
            languages: Filter by language codes (e.g., ["en", "fr", "de"])
            channels: Filter by channel names
            channel_urls: Filter by channel URLs
            speakers: Filter by speaker names
            content_ids: Filter by content IDs (YouTube IDs, etc.)
            segment_ids: Filter by specific segment IDs
            date_range: Tuple of (start_date, end_date)
            stitch_versions: Filter by stitch versions (e.g., ["stitch_v14"])
            min_duration: Minimum segment duration in seconds
            max_duration: Maximum segment duration in seconds
            must_be_stitched: Only return stitched segments (default True)
            must_be_embedded: Only return embedded segments (default True)
            limit: Maximum number of segments to return

        Returns:
            List of Segment objects matching filters
        """
        session = self._get_session()

        # Build query with eager loading
        query = (
            session.query(Segment)
            .join(Content, Segment.content_id == Content.id)
            .options(joinedload(Segment.content))
        )

        # Apply filters
        filters = []

        # Content-level filters
        if projects:
            # Projects stored as array, use overlap operator
            # SQLAlchemy doesn't have native support for &&, use op() for PostgreSQL array overlap
            filters.append(Content.projects.op('&&')(projects))

        if languages:
            filters.append(Content.main_language.in_(languages))

        if channels:
            filters.append(Content.channel_name.in_(channels))

        if channel_urls:
            filters.append(Content.channel_url.in_(channel_urls))

        if content_ids:
            filters.append(Content.content_id.in_(content_ids))

        if date_range:
            start_date, end_date = date_range
            if start_date:
                filters.append(Content.publish_date >= start_date)
            if end_date:
                filters.append(Content.publish_date <= end_date)

        if stitch_versions:
            filters.append(Content.stitch_version.in_(stitch_versions))

        # Note: is_stitched and is_embedded not in minimal backend models
        # All content in embedding_segments table is already stitched and embedded
        if must_be_stitched or must_be_embedded:
            # Just ensure we have embeddings if must_be_embedded
            if must_be_embedded:
                filters.append(Segment.embedding != None)

        # Segment-level filters
        if segment_ids:
            filters.append(Segment.id.in_(segment_ids))

        # Note: speakers filter not supported on EmbeddingSegment
        # (it has source_speaker_hashes instead of speaker_name)
        if speakers:
            logger.warning("Speaker filtering not supported on EmbeddingSegment model")

        if min_duration is not None:
            # duration = end_time - start_time
            filters.append((Segment.end_time - Segment.start_time) >= min_duration)

        if max_duration is not None:
            filters.append((Segment.end_time - Segment.start_time) <= max_duration)

        # Apply all filters
        if filters:
            query = query.filter(and_(*filters))

        # Apply limit
        if limit:
            query = query.limit(limit)

        # Execute query
        segments = query.all()

        return segments

    def fetch_by_ids(self, segment_ids: List[int]) -> List[Segment]:
        """
        Direct fetch by segment IDs.

        Args:
            segment_ids: List of segment IDs

        Returns:
            List of Segment objects
        """
        return self.fetch_by_filter(segment_ids=segment_ids, must_be_stitched=False, must_be_embedded=False)

    def get_segment_embeddings(self, segments: List[Segment]) -> List[Optional[List[float]]]:
        """
        Extract embeddings from segments.

        Args:
            segments: List of Segment objects

        Returns:
            List of embeddings (or None if segment has no embedding)
        """
        return [seg.embedding for seg in segments]

    def count_by_filter(
        self,
        projects: Optional[List[str]] = None,
        languages: Optional[List[str]] = None,
        channels: Optional[List[str]] = None,
        date_range: Optional[Tuple[datetime, datetime]] = None,
        must_be_stitched: bool = True,
        must_be_embedded: bool = True
    ) -> int:
        """
        Count segments matching filters (without loading them).

        Args:
            Same as fetch_by_filter

        Returns:
            Count of matching segments
        """
        session = self._get_session()

        query = (
            session.query(Segment.id)
            .join(Content, Segment.content_id == Content.id)
        )

        # Apply same filters as fetch_by_filter
        filters = []

        if projects:
            project_filters = [Content.projects.any(project) for project in projects]
            filters.append(or_(*project_filters))

        if languages:
            filters.append(Content.main_language.in_(languages))

        if channels:
            filters.append(Content.channel_name.in_(channels))

        if date_range:
            start_date, end_date = date_range
            if start_date:
                filters.append(Content.publish_date >= start_date)
            if end_date:
                filters.append(Content.publish_date <= end_date)

        # Note: Backend models don't have is_stitched/is_embedded flags
        if must_be_embedded:
            filters.append(Segment.embedding != None)

        if filters:
            query = query.filter(and_(*filters))

        count = query.count()
        logger.debug(f"Count: {count} segments matching filters")
        return count

    def get_baseline_stats(
        self,
        projects: Optional[List[str]] = None,
        languages: Optional[List[str]] = None,
        channels: Optional[List[str]] = None,
        date_range: Optional[Tuple[datetime, datetime]] = None,
        must_be_stitched: bool = True,
        must_be_embedded: bool = True,
        use_cache: bool = True
    ) -> dict:
        """
        Get aggregated baseline statistics using SQL (much faster than loading all segments).

        Returns counts of unique videos, channels, and total segments matching filters.

        Args:
            Same as fetch_by_filter
            use_cache: If True and date_range matches a cache window, use embedding_cache tables

        Returns:
            Dict with:
                - total_segments: int
                - unique_videos: int
                - unique_channels: int
        """
        from sqlalchemy import func, distinct, text, table, column

        session = self._get_session()

        # DISABLED: Cache tables are stale snapshots and should be ignored
        # TODO: Implement cache table freshness checks (6 hours for 7d, 24 hours for 30d/180d)
        # For now, always use main tables with optimized indexes
        cache_table = None
        use_cache = False  # Force disable cache tables

        if cache_table:
            logger.info(f"Using cache table: {cache_table}")

            # Build query on cache table (no join needed - denormalized!)
            # Use table() construct to create a proper FROM clause
            cache_tbl = table(cache_table,
                            column('id'),
                            column('content_id_string'),
                            column('channel_name'),
                            column('projects'),
                            column('main_language'))

            query = session.query(
                func.count(cache_tbl.c.id).label('total_segments'),
                func.count(distinct(cache_tbl.c.content_id_string)).label('unique_videos'),
                func.count(distinct(cache_tbl.c.channel_name)).label('unique_channels')
            ).select_from(cache_tbl)

            # Apply filters directly on cache table
            filters = []
            if projects:
                for project in projects:
                    filters.append(text(f"'{project}' = ANY(projects)"))
            if languages:
                lang_list = "','".join(languages)
                filters.append(text(f"main_language IN ('{lang_list}')"))
            if channels:
                chan_list = "','".join(channels)
                filters.append(text(f"channel_name IN ('{chan_list}')"))

            if filters:
                for f in filters:
                    query = query.filter(f)
        else:
            # Standard query on main tables
            query = (
                session.query(
                    func.count(Segment.id).label('total_segments'),
                    func.count(distinct(Segment.content_id_string)).label('unique_videos'),
                    func.count(distinct(Content.channel_name)).label('unique_channels')
                )
                .join(Content, Segment.content_id == Content.id)
            )

            # Apply same filters as fetch_by_filter (only for standard query)
            filters = []

            if projects:
                project_filters = [Content.projects.any(project) for project in projects]
                filters.append(or_(*project_filters))

            if languages:
                filters.append(Content.main_language.in_(languages))

            if channels:
                filters.append(Content.channel_name.in_(channels))

            if date_range:
                start_date, end_date = date_range
                if start_date:
                    filters.append(Content.publish_date >= start_date)
                if end_date:
                    filters.append(Content.publish_date <= end_date)

            if must_be_embedded:
                filters.append(Segment.embedding != None)

            if filters:
                query = query.filter(and_(*filters))

        # Log the actual SQL query for debugging
        result = query.one()

        stats = {
            'total_segments': result.total_segments or 0,
            'unique_videos': result.unique_videos or 0,
            'unique_channels': result.unique_channels or 0
        }

        return stats

    def get_unique_values(
        self,
        field: str,
        projects: Optional[List[str]] = None,
        languages: Optional[List[str]] = None
    ) -> List[str]:
        """
        Get unique values for a field (e.g., all channels, speakers, etc.).

        Args:
            field: Field name ("channel_name", "speaker_name", "main_language", etc.)
            projects: Optional project filter
            languages: Optional language filter

        Returns:
            List of unique values
        """
        session = self._get_session()

        # Map field to table/column
        field_map = {
            "channel_name": Content.channel_name,
            "channel_url": Content.channel_url,
            "main_language": Content.main_language,
            "stitch_version": Content.stitch_version,
        }

        if field not in field_map:
            raise ValueError(f"Unknown field: {field}. Valid fields: {list(field_map.keys())}")

        column = field_map[field]

        # Build query
        query = session.query(column).distinct()

        # Join if needed
        if field.startswith("content_") or field in ["channel_name", "channel_url", "main_language", "stitch_version"]:
            # Already on Content table
            pass
        else:
            # Need to join through Segment
            query = query.select_from(Segment).join(Content)

        # Apply filters
        filters = []
        if projects:
            project_filters = [Content.projects.any(project) for project in projects]
            filters.append(or_(*project_filters))

        if languages:
            filters.append(Content.main_language.in_(languages))

        if filters:
            query = query.filter(and_(*filters))

        # Execute
        results = [row[0] for row in query.all() if row[0] is not None]
        return results
