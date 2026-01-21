"""
Dynamic Grouper
===============

Dynamically filter and group segments based on user-provided criteria.
Supports filtering by channel URLs, keywords, metadata, projects, and more.
"""

import logging
from typing import List, Dict, Set, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func

from ...utils.backend_logger import get_logger
logger = get_logger("dynamic_grouper")

# Import database models
from src.database.models import EmbeddingSegment, Content


@dataclass
class GroupFilter:
    """Filter criteria for a group"""
    channel_urls: Optional[List[str]] = None
    keywords: Optional[List[str]] = None  # Search in meta_data
    language: Optional[str] = None
    projects: Optional[List[str]] = None
    meta_data_query: Optional[Dict[str, Any]] = None  # JSONPath-style queries
    start_date: Optional[datetime] = None  # Replaces min_publish_date
    end_date: Optional[datetime] = None    # Replaces max_publish_date
    # Deprecated - kept for backward compatibility
    min_publish_date: Optional[datetime] = None
    max_publish_date: Optional[datetime] = None


@dataclass
class GroupResult:
    """Result of group filtering with metadata"""
    group_id: str
    group_name: str
    segment_ids: List[int] = field(default_factory=list)
    channel_urls: Set[str] = field(default_factory=set)
    segment_count: int = 0
    date_range: Optional[tuple] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'group_id': self.group_id,
            'group_name': self.group_name,
            'segment_ids': self.segment_ids,
            'channel_urls': list(self.channel_urls),
            'segment_count': self.segment_count,
            'date_range': [
                self.date_range[0].isoformat() if self.date_range and self.date_range[0] else None,
                self.date_range[1].isoformat() if self.date_range and self.date_range[1] else None
            ] if self.date_range else None,
            'metadata': self.metadata
        }


class DynamicGrouper:
    """
    Dynamically groups segments based on flexible user-defined filters.

    No hardcoded groups - everything is driven by API request parameters.
    """

    def __init__(self, session: Session):
        """
        Initialize grouper.

        Args:
            session: SQLAlchemy database session
        """
        self.session = session
        logger.info("DynamicGrouper initialized")

    def build_group(
        self,
        group_id: str,
        group_name: str,
        filter_config: GroupFilter,
        time_window_days: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> GroupResult:
        """
        Build a group by applying filters to retrieve segments.

        Args:
            group_id: Unique identifier for this group
            group_name: Human-readable group name
            filter_config: GroupFilter with criteria
            time_window_days: Optional time window in days (convenient shorthand)
            start_date: Optional explicit start date (overrides time_window_days)
            end_date: Optional explicit end date (defaults to now)

        Returns:
            GroupResult with matching segments and metadata
        """
        logger.info(f"Building group '{group_id}' with filters: {filter_config}")

        # Start with base query
        query = self.session.query(EmbeddingSegment).join(
            Content, EmbeddingSegment.content_id == Content.id
        )

        # Determine time filtering approach (precedence: explicit dates > filter dates > time_window_days)
        effective_start = start_date or filter_config.start_date or filter_config.min_publish_date
        effective_end = end_date or filter_config.end_date or filter_config.max_publish_date

        # If no explicit dates but time_window_days specified, calculate from window
        if not effective_start and time_window_days:
            # Use timezone-aware datetime to match database timestamps
            effective_start = datetime.now(timezone.utc) - timedelta(days=time_window_days)

        # Apply time filters
        if effective_start:
            query = query.filter(Content.publish_date >= effective_start)
            logger.debug(f"  Filtering: publish_date >= {effective_start}")

        if effective_end:
            query = query.filter(Content.publish_date <= effective_end)
            logger.debug(f"  Filtering: publish_date <= {effective_end}")

        # Apply channel URL filter
        if filter_config.channel_urls:
            logger.debug(f"Filtering by {len(filter_config.channel_urls)} channel URLs")
            query = query.filter(Content.channel_url.in_(filter_config.channel_urls))

        # Apply project filter
        if filter_config.projects:
            logger.debug(f"Filtering by projects: {filter_config.projects}")
            project_conditions = [
                Content.projects.any(proj)
                for proj in filter_config.projects
            ]
            query = query.filter(or_(*project_conditions))

        # Apply keyword filter (search in meta_data JSONB)
        if filter_config.keywords:
            logger.debug(f"Filtering by keywords: {filter_config.keywords}")
            query = self._apply_keyword_filter(query, filter_config.keywords)

        # Apply language filter using main_language field
        if filter_config.language:
            logger.debug(f"Filtering by language: {filter_config.language}")
            query = query.filter(Content.main_language == filter_config.language)

        # Apply custom meta_data queries
        if filter_config.meta_data_query:
            logger.warning(f"Meta_data query requested but Content table has no meta_data field - skipping filter")
            # TODO: Add meta_data JSONB field to Content table
            pass

        # Only include segments with embeddings
        query = query.filter(EmbeddingSegment.embedding.isnot(None))

        # Execute query
        segments = query.all()

        # Build result
        result = self._build_result(group_id, group_name, segments)

        logger.info(
            f"✓ Group '{group_id}' built: {result.segment_count} segments "
            f"from {len(result.channel_urls)} channels"
        )

        return result

    def build_multiple_groups(
        self,
        group_configs: List[Dict[str, Any]],
        time_window_days: Optional[int] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[GroupResult]:
        """
        Build multiple groups in batch.

        Args:
            group_configs: List of dicts with {group_id, group_name, filter}
            time_window_days: Optional time window for all groups
            start_date: Optional explicit start date for all groups
            end_date: Optional explicit end date for all groups

        Returns:
            List of GroupResult objects
        """
        results = []

        for config in group_configs:
            # Convert filter dict to GroupFilter object
            filter_config = self._dict_to_filter(config.get('filter', {}))

            result = self.build_group(
                group_id=config['group_id'],
                group_name=config['group_name'],
                filter_config=filter_config,
                time_window_days=time_window_days,
                start_date=start_date,
                end_date=end_date
            )
            results.append(result)

        logger.info(f"Built {len(results)} groups")
        return results

    def _apply_keyword_filter(self, query, keywords: List[str]):
        """
        Apply keyword filter to search in meta_data JSONB.

        Searches for keywords in meta_data['keywords'] array.
        """
        keyword_conditions = []

        for kw in keywords:
            # Search in meta_data['keywords'] array using PostgreSQL JSONB operators
            # This checks if any keyword dict contains the term
            keyword_conditions.append(
                Content.meta_data.op('?')('keywords') &
                func.jsonb_path_exists(
                    Content.meta_data,
                    f'$.keywords[*].term ? (@ like_regex "{kw}" flag "i")'
                )
            )

        if keyword_conditions:
            query = query.filter(or_(*keyword_conditions))

        return query

    def _apply_metadata_filter(self, query, meta_data_query: Dict[str, Any]):
        """
        Apply custom metadata filters using JSONB containment.

        Args:
            query: SQLAlchemy query
            meta_data_query: Dict with key-value pairs to match in meta_data

        Example:
            {"explicit": true} -> Filters for content with explicit=true in meta_data
        """
        for key, value in meta_data_query.items():
            # Simple containment check
            query = query.filter(
                Content.meta_data.contains({key: value})
            )

        return query

    def _dict_to_filter(self, filter_dict: Dict[str, Any]) -> GroupFilter:
        """Convert dictionary to GroupFilter object"""
        return GroupFilter(
            channel_urls=filter_dict.get('channel_urls'),
            keywords=filter_dict.get('keywords'),
            language=filter_dict.get('language'),
            projects=filter_dict.get('projects'),
            meta_data_query=filter_dict.get('meta_data_query'),
            start_date=filter_dict.get('start_date'),
            end_date=filter_dict.get('end_date'),
            # Backward compatibility
            min_publish_date=filter_dict.get('min_publish_date'),
            max_publish_date=filter_dict.get('max_publish_date')
        )

    def _build_result(
        self,
        group_id: str,
        group_name: str,
        segments: List[EmbeddingSegment]
    ) -> GroupResult:
        """
        Build GroupResult from query results.

        Args:
            group_id: Group identifier
            group_name: Group name
            segments: List of EmbeddingSegment objects

        Returns:
            GroupResult with metadata
        """
        if not segments:
            logger.warning(f"No segments found for group '{group_id}'")
            return GroupResult(
                group_id=group_id,
                group_name=group_name,
                segment_count=0
            )

        # Extract segment IDs
        segment_ids = [seg.id for seg in segments]

        # Extract unique channels
        channel_urls = set()
        for seg in segments:
            if seg.content and seg.content.channel_url:
                channel_urls.add(seg.content.channel_url)

        # Calculate date range
        dates = [
            seg.content.publish_date
            for seg in segments
            if seg.content and seg.content.publish_date
        ]
        date_range = (min(dates), max(dates)) if dates else None

        # Build metadata
        metadata = {
            'unique_content_items': len(set(seg.content_id for seg in segments)),
            'unique_channels': len(channel_urls),
            'avg_segment_length': sum(seg.end_time - seg.start_time for seg in segments) / len(segments),
            'total_duration': sum(seg.end_time - seg.start_time for seg in segments)
        }

        return GroupResult(
            group_id=group_id,
            group_name=group_name,
            segment_ids=segment_ids,
            channel_urls=channel_urls,
            segment_count=len(segment_ids),
            date_range=date_range,
            metadata=metadata
        )

    def get_segments_for_group(
        self,
        group_result: GroupResult,
        limit: Optional[int] = None
    ) -> List[EmbeddingSegment]:
        """
        Retrieve full EmbeddingSegment objects for a group.

        Args:
            group_result: GroupResult with segment IDs
            limit: Optional limit on number of segments to retrieve

        Returns:
            List of EmbeddingSegment objects
        """
        query = self.session.query(EmbeddingSegment).filter(
            EmbeddingSegment.id.in_(group_result.segment_ids)
        )

        if limit:
            query = query.limit(limit)

        segments = query.all()
        logger.debug(f"Retrieved {len(segments)} segments for group '{group_result.group_id}'")
        return segments

    def validate_group_filters(self, filter_config: GroupFilter) -> Dict[str, Any]:
        """
        Validate group filters before applying.

        Args:
            filter_config: GroupFilter to validate

        Returns:
            Dict with validation results: {'valid': bool, 'errors': List[str]}
        """
        errors = []

        # Check channel URLs exist
        if filter_config.channel_urls:
            existing_channels = self.session.query(Content.channel_url).filter(
                Content.channel_url.in_(filter_config.channel_urls)
            ).distinct().all()
            existing_urls = set(row[0] for row in existing_channels)
            missing = set(filter_config.channel_urls) - existing_urls
            if missing:
                errors.append(f"Channel URLs not found: {missing}")

        # Check projects exist
        if filter_config.projects:
            # Simple check - could be more sophisticated
            for proj in filter_config.projects:
                count = self.session.query(Content).filter(
                    Content.projects.any(proj)
                ).count()
                if count == 0:
                    errors.append(f"No content found for project: {proj}")

        # Check date range validity
        if filter_config.min_publish_date and filter_config.max_publish_date:
            if filter_config.min_publish_date > filter_config.max_publish_date:
                errors.append("min_publish_date cannot be after max_publish_date")

        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
