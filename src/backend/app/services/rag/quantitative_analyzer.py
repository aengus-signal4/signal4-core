"""
QuantitativeAnalyzer
====================

Analyzes segment distributions and discourse centrality metrics.

Provides quantitative metrics about search results:
- Number of relevant segments
- Main channels/episodes focused on the issue
- Temporal distribution
- Discourse centrality (how central is this topic)
"""

from typing import List, Dict, Any, Optional
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from ...utils.backend_logger import get_logger
logger = get_logger("quantitative_analyzer")


class QuantitativeAnalyzer:
    """
    Analyzes segment distributions to provide quantitative metrics.

    Provides insights into:
    - Segment counts and distribution
    - Channel/episode concentration
    - Temporal patterns
    - Discourse centrality metrics
    """

    def __init__(self, db_session: Optional[Session] = None):
        """
        Initialize analyzer.

        Args:
            db_session: Optional database session for additional queries
        """
        self.db_session = db_session
        logger.info("QuantitativeAnalyzer initialized")

    def analyze(
        self,
        segments: List[Any],
        baseline_segments: Optional[List[Any]] = None,
        time_window_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Perform quantitative analysis on segments.

        Args:
            segments: List of segment objects to analyze
            baseline_segments: Optional baseline segments for comparison (e.g., all segments in time window)
            time_window_days: Time window for context

        Returns:
            Dict with quantitative metrics:
                - total_segments: Total number of segments
                - unique_videos: Number of unique videos/episodes
                - unique_channels: Number of unique channels
                - channel_distribution: Top channels by segment count
                - video_distribution: Top videos by segment count
                - temporal_distribution: Segments over time
                - discourse_centrality: How central is this topic (0-1 scale)
                - concentration_metrics: How concentrated is the discussion
        """
        if not segments:
            return self._empty_analysis()

        logger.info(f"Analyzing {len(segments)} segments")

        # Extract metadata
        channel_counter = Counter()
        video_counter = Counter()
        video_metadata = {}  # content_id -> {title, channel, date, segment_count}
        dates = []

        for i, seg in enumerate(segments):
            # Handle both dict and object segments
            if isinstance(seg, dict):
                channel_name = seg.get('channel_name', 'Unknown')
                content_id_str = seg.get('content_id_string') or seg.get('content_id')
                title = seg.get('title', 'Unknown')
                channel_url = seg.get('channel_url')
                publish_date = seg.get('publish_date')
                start_time = seg.get('start_time', 0)
                end_time = seg.get('end_time', 0)

                # Debug first segment
                if i == 0:
                    logger.info(f"[DEBUG] First segment (dict): keys={list(seg.keys())}, channel_name={channel_name}, content_id={content_id_str}")
            else:
                # Legacy object format
                channel_name = getattr(seg.content, 'channel_name', None) or getattr(seg, 'channel_name', 'Unknown')
                content_id_str = getattr(seg, 'content_id_string', None) or getattr(seg.content, 'content_id', None)
                title = getattr(seg.content, 'title', 'Unknown')
                channel_url = getattr(seg.content, 'channel_url', None)
                publish_date = getattr(seg.content, 'publish_date', None)
                start_time = getattr(seg, 'start_time', 0)
                end_time = getattr(seg, 'end_time', 0)

                # Debug first segment
                if i == 0:
                    logger.info(f"[DEBUG] First segment (object): type={type(seg)}, has_content={hasattr(seg, 'content')}, channel_name={channel_name}, content_id={content_id_str}")

            # Channel info
            channel_counter[channel_name] += 1

            # Video info
            if content_id_str:
                video_counter[content_id_str] += 1

                # Store metadata
                if content_id_str not in video_metadata:
                    video_metadata[content_id_str] = {
                        'content_id': content_id_str,
                        'title': title,
                        'channel_name': channel_name,
                        'channel_url': channel_url,
                        'publish_date': publish_date,
                        'segment_count': 0,
                        'total_duration': 0.0
                    }

                video_metadata[content_id_str]['segment_count'] += 1
                duration = end_time - start_time
                video_metadata[content_id_str]['total_duration'] += duration

            # Temporal data
            if publish_date:
                dates.append(publish_date)

        # Calculate metrics
        total_segments = len(segments)
        unique_videos = len(video_counter)
        unique_channels = len(channel_counter)

        # Channel distribution (top 10) - enrich with metadata
        channel_distribution = []
        for channel, count in channel_counter.most_common(10):
            channel_data = {
                'channel_name': channel,
                'segment_count': count,
                'percentage': round(100 * count / total_segments, 2)
            }
            # Enrich with metadata if db_session available
            if self.db_session:
                metadata = self._get_channel_metadata(channel)
                channel_data.update(metadata)

            channel_distribution.append(channel_data)

        # Video distribution (top 10 most focused episodes) - enrich with metadata
        video_distribution = []
        for content_id, count in video_counter.most_common(10):
            meta = video_metadata[content_id]
            video_data = {
                'content_id': content_id,
                'title': meta['title'],
                'channel_name': meta['channel_name'],
                'channel_url': meta['channel_url'],
                'publish_date': meta['publish_date'].isoformat() if meta['publish_date'] and hasattr(meta['publish_date'], 'isoformat') else meta['publish_date'],
                'segment_count': count,
                'total_duration_seconds': round(meta['total_duration'], 2),
                'percentage': round(100 * count / total_segments, 2)
            }

            # Enrich with metadata if db_session available
            if self.db_session:
                content_metadata = self._get_content_metadata(content_id)
                video_data.update(content_metadata)

            video_distribution.append(video_data)

        # Temporal distribution
        temporal_distribution = self._analyze_temporal_distribution(dates, time_window_days)

        # Concentration metrics
        concentration = self._calculate_concentration(
            channel_counter,
            video_counter,
            total_segments,
            unique_channels,
            unique_videos
        )

        # Discourse centrality (how central is this topic) - only if baseline provided
        discourse_centrality = None
        if baseline_segments is not None:
            discourse_centrality = self._calculate_discourse_centrality(
                segments=segments,
                baseline_segments=baseline_segments,
                unique_videos=unique_videos,
                unique_channels=unique_channels,
                time_window_days=time_window_days
            )

        # Calculate total duration and episode count if db_session available
        total_duration_hours = None
        episode_count = unique_videos  # Default to unique_videos count

        if self.db_session and video_counter:
            duration_data = self._calculate_total_duration(list(video_counter.keys()))
            total_duration_hours = duration_data.get('total_duration_hours')
            episode_count = duration_data.get('episode_count', unique_videos)

        result = {
            'total_segments': total_segments,
            'unique_videos': unique_videos,
            'unique_channels': unique_channels,
            'episode_count': episode_count,
            'total_duration_hours': total_duration_hours,
            'channel_distribution': channel_distribution,
            'video_distribution': video_distribution,
            'temporal_distribution': temporal_distribution,
            'concentration_metrics': concentration,
            'discourse_centrality': discourse_centrality
        }

        if discourse_centrality:
            logger.info(
                f"Analysis complete: {total_segments} segments, "
                f"{unique_videos} videos, {unique_channels} channels, "
                f"centrality={discourse_centrality['level']}/5 ({discourse_centrality['label']})"
            )
        else:
            logger.info(
                f"Analysis complete: {total_segments} segments, "
                f"{unique_videos} videos, {unique_channels} channels "
                f"(no centrality - baseline not provided)"
            )

        return result

    def _empty_analysis(self) -> Dict[str, Any]:
        """Return empty analysis structure."""
        return {
            'total_segments': 0,
            'unique_videos': 0,
            'unique_channels': 0,
            'episode_count': 0,
            'total_duration_hours': None,
            'channel_distribution': [],
            'video_distribution': [],
            'temporal_distribution': {},
            'concentration_metrics': {},
            'discourse_centrality': None
        }

    def _calculate_total_duration(self, content_ids: List[str]) -> Dict[str, Any]:
        """
        Calculate total duration in hours for a list of content IDs.

        Args:
            content_ids: List of content ID strings

        Returns:
            Dict with total_duration_hours and episode_count
        """
        if not content_ids or not self.db_session:
            return {'total_duration_hours': None, 'episode_count': len(content_ids)}

        try:
            from src.database.models import Content
            from sqlalchemy import func

            # Query for total duration and count
            # Note: Content.duration is in seconds
            result = self.db_session.query(
                func.count(Content.content_id).label('count'),
                func.sum(Content.duration).label('total_seconds')
            ).filter(
                Content.content_id.in_(content_ids)
            ).first()

            if result and result.total_seconds:
                total_hours = round(result.total_seconds / 3600.0, 2)
                episode_count = result.count or len(content_ids)
                logger.debug(f"Calculated duration: {total_hours}h across {episode_count} episodes")
                return {
                    'total_duration_hours': total_hours,
                    'episode_count': episode_count
                }
            else:
                logger.debug(f"No duration data found for {len(content_ids)} content IDs")
                return {
                    'total_duration_hours': None,
                    'episode_count': len(content_ids)
                }

        except Exception as e:
            logger.error(f"Error calculating total duration: {e}", exc_info=True)
            return {
                'total_duration_hours': None,
                'episode_count': len(content_ids)
            }

    def _analyze_temporal_distribution(
        self,
        dates: List[datetime],
        time_window_days: Optional[int]
    ) -> Dict[str, Any]:
        """
        Analyze temporal distribution of segments.

        Returns distribution by day/week/month depending on time window.
        """
        if not dates:
            return {}

        # Convert string dates to datetime if needed
        from datetime import datetime as dt
        parsed_dates = []
        for date in dates:
            if isinstance(date, str):
                # Parse ISO format string
                try:
                    parsed_dates.append(dt.fromisoformat(date.replace('Z', '+00:00')))
                except:
                    continue
            else:
                parsed_dates.append(date)

        if not parsed_dates:
            return {}

        parsed_dates = sorted(parsed_dates)
        earliest = parsed_dates[0]
        latest = parsed_dates[-1]
        span_days = (latest - earliest).days + 1
        dates = parsed_dates  # Use parsed dates for bucketing

        # Choose granularity based on span
        if span_days <= 14:
            # Daily granularity
            granularity = 'daily'
            bucket_format = '%Y-%m-%d'
        elif span_days <= 90:
            # Weekly granularity
            granularity = 'weekly'
            bucket_format = '%Y-W%U'  # Year-Week
        else:
            # Monthly granularity
            granularity = 'monthly'
            bucket_format = '%Y-%m'

        # Count by bucket
        bucket_counter = Counter()
        for date in dates:
            bucket = date.strftime(bucket_format)
            bucket_counter[bucket] += 1

        # Build time series
        time_series = [
            {'period': bucket, 'count': count}
            for bucket, count in sorted(bucket_counter.items())
        ]

        return {
            'granularity': granularity,
            'earliest_date': earliest.isoformat(),
            'latest_date': latest.isoformat(),
            'span_days': span_days,
            'time_series': time_series
        }

    def _calculate_concentration(
        self,
        channel_counter: Counter,
        video_counter: Counter,
        total_segments: int,
        unique_channels: int,
        unique_videos: int
    ) -> Dict[str, Any]:
        """
        Calculate concentration metrics (how focused vs distributed is the discussion).

        Returns Herfindahl-Hirschman Index (HHI) and interpretation.
        """
        # HHI for channels (0 = perfectly distributed, 10000 = monopoly)
        channel_hhi = sum((count / total_segments * 100) ** 2 for count in channel_counter.values())

        # HHI for videos
        video_hhi = sum((count / total_segments * 100) ** 2 for count in video_counter.values())

        # Top-K concentration (% of segments in top N channels/videos)
        top_3_channels_pct = sum(count for _, count in channel_counter.most_common(3)) / total_segments * 100
        top_10_videos_pct = sum(count for _, count in video_counter.most_common(10)) / total_segments * 100

        # Interpret channel HHI
        if channel_hhi < 1500:
            channel_interpretation = 'highly distributed across many channels'
        elif channel_hhi < 2500:
            channel_interpretation = 'moderately distributed'
        else:
            channel_interpretation = 'concentrated in few channels'

        # Interpret video HHI
        if video_hhi < 1500:
            video_interpretation = 'discussed across many episodes'
        elif video_hhi < 2500:
            video_interpretation = 'moderate episode concentration'
        else:
            video_interpretation = 'concentrated in few episodes'

        return {
            'channel_hhi': round(channel_hhi, 2),
            'channel_interpretation': channel_interpretation,
            'video_hhi': round(video_hhi, 2),
            'video_interpretation': video_interpretation,
            'top_3_channels_percentage': round(top_3_channels_pct, 2),
            'top_10_videos_percentage': round(top_10_videos_pct, 2)
        }

    def _calculate_discourse_centrality(
        self,
        segments: List[Any],
        baseline_segments: Any,  # Can be list of segments OR dict with stats
        unique_videos: int,
        unique_channels: int,
        time_window_days: Optional[int]
    ) -> Dict[str, Any]:
        """
        Calculate discourse centrality - how central is this topic to the discourse.

        Returns a simple 0-5 level scale based on channel coverage:
        - 5: Very common (>50% of channels)
        - 4: Common (30-50% of channels)
        - 3: Moderate (15-30% of channels)
        - 2: Uncommon (5-15% of channels)
        - 1: Rare (1-5% of channels)
        - 0: Very rare (<1% of channels)
        """
        # Check if baseline_segments is already aggregated stats (dict)
        if isinstance(baseline_segments, dict):
            baseline_channels = baseline_segments.get('unique_channels', 0)
        else:
            # Legacy path: iterate through segment objects (slow)
            logger.warning("Using legacy segment iteration for baseline stats")
            baseline_channels = len(set(
                getattr(seg.content, 'channel_name', None) or getattr(seg, 'channel_name', 'Unknown')
                for seg in baseline_segments
            ))

        # Calculate channel coverage percentage (capped at 100%)
        if baseline_channels > 0:
            channel_coverage_pct = min((unique_channels / baseline_channels) * 100, 100.0)
        else:
            channel_coverage_pct = 0

        logger.info(
            f"Discourse centrality: {unique_channels}/{baseline_channels} channels = {channel_coverage_pct:.1f}%"
        )

        # Map to 0-5 level based on channel coverage
        if channel_coverage_pct >= 50:
            level = 5
            label = 'Very common'
        elif channel_coverage_pct >= 30:
            level = 4
            label = 'Common'
        elif channel_coverage_pct >= 15:
            level = 3
            label = 'Moderate'
        elif channel_coverage_pct >= 5:
            level = 2
            label = 'Uncommon'
        elif channel_coverage_pct >= 1:
            level = 1
            label = 'Rare'
        else:
            level = 0
            label = 'Very rare'

        return {
            'level': level,
            'label': label
        }

    def _get_channel_metadata(self, channel_name: str) -> Dict[str, Any]:
        """
        Fetch additional metadata for a channel from the database.

        Args:
            channel_name: Channel name to look up

        Returns:
            Dict with channel metadata (description, first_episode_date, etc.)

        TODO: Implement proper channel metadata enrichment
            - Query channels table joined with content table
            - Fetch channel description from channels.description
            - Calculate first_episode_date as MIN(content.publish_date)
            - Handle missing channel_id properly
            - Add error handling for DB query failures
            See: BACKEND_METADATA_SPEC.md for requirements
        """
        # Stubbed out - return empty metadata for now
        return {}

        # COMMENTED OUT - Implementation needs testing and refinement
        # try:
        #     from ...database.models import Content, Channel
        #     from sqlalchemy import func
        #
        #     # Query for channel metadata via join
        #     result = self.db_session.query(
        #         Channel.description,
        #         func.min(Content.publish_date).label('first_episode_date')
        #     ).join(
        #         Content, Content.channel_id == Channel.id
        #     ).filter(
        #         Content.channel_name == channel_name
        #     ).group_by(
        #         Channel.description
        #     ).first()
        #
        #     if result:
        #         description, first_date = result
        #         metadata = {
        #             'channel_description': description or "No description available",
        #             'first_episode_date': first_date.isoformat() if first_date and hasattr(first_date, 'isoformat') else None
        #         }
        #         return metadata
        #     else:
        #         logger.warning(f"No metadata found for channel: {channel_name}")
        #         return {
        #             'channel_description': "No description available",
        #             'first_episode_date': None
        #         }
        #
        # except Exception as e:
        #     logger.error(f"Error fetching channel metadata for {channel_name}: {e}")
        #     return {
        #         'channel_description': "No description available",
        #         'first_episode_date': None
        #     }

    def _get_content_metadata(self, content_id_string: str) -> Dict[str, Any]:
        """
        Fetch additional metadata for a specific content/episode from the database.

        Args:
            content_id_string: Content ID string to look up (e.g., YouTube ID)

        Returns:
            Dict with content metadata (description, guests, content_url)

        TODO: Implement proper content metadata enrichment
            - Query content table by content_id
            - Fetch description from content.description
            - Extract guests from description (or dedicated field if available)
            - Build content_url from content_id (YouTube: youtube.com/watch?v=ID)
            - Handle different platforms (YouTube, Rumble, etc.)
            - Add error handling for missing content
            See: BACKEND_METADATA_SPEC.md for requirements
        """
        # Stubbed out - return empty metadata for now
        return {}

        # COMMENTED OUT - Implementation needs testing and refinement
        # try:
        #     from ...database.models import Content
        #
        #     # Query for content metadata using content_id (which stores the YouTube ID)
        #     content = self.db_session.query(Content).filter(
        #         Content.content_id == content_id_string
        #     ).first()
        #
        #     if content:
        #         # Extract guests from description if guests field doesn't exist
        #         guests = []
        #         if hasattr(content, 'guests') and content.guests:
        #             # If guests is stored as array/JSON
        #             guests = content.guests if isinstance(content.guests, list) else []
        #         elif content.description:
        #             # Try to extract guests from description
        #             guests = self._extract_guests_from_description(content.description)
        #
        #         # Build content URL
        #         content_url = None
        #         if content.content_id:
        #             # Assuming YouTube URLs for now
        #             content_url = f"https://www.youtube.com/watch?v={content.content_id}"
        #
        #         metadata = {
        #             'description': content.description or "No description available",
        #             'guests': guests,
        #             'content_url': content_url
        #         }
        #         return metadata
        #     else:
        #         logger.warning(f"No metadata found for content: {content_id_string}")
        #         return {
        #             'description': "No description available",
        #             'guests': [],
        #             'content_url': None
        #         }
        #
        # except Exception as e:
        #     logger.error(f"Error fetching content metadata for {content_id_string}: {e}")
        #     return {
        #         'description': "No description available",
        #         'guests': [],
        #         'content_url': None
        #     }

    def _extract_guests_from_description(self, description: str) -> List[str]:
        """
        Extract guest names from episode description using common patterns.

        Args:
            description: Episode description text

        Returns:
            List of guest names (can be empty)

        TODO: Implement robust guest extraction
            - Test regex patterns against real episode descriptions
            - Handle edge cases (false positives, formatting variations)
            - Consider NER (Named Entity Recognition) for better accuracy
            - Handle French language patterns as well
            - Validate extracted names (remove non-name text)
            See: BACKEND_METADATA_SPEC.md for requirements
        """
        # Stubbed out - return empty list for now
        return []

        # COMMENTED OUT - Implementation needs testing and refinement
        # import re
        #
        # if not description:
        #     return []
        #
        # guests = []
        #
        # # Common patterns for guest mentions
        # patterns = [
        #     r"Guest[s]?:\s*([^.\n]+)",  # "Guest: Name" or "Guests: Name1, Name2"
        #     r"Featuring[s]?:\s*([^.\n]+)",  # "Featuring: Name"
        #     r"With[s]?:\s*([^.\n]+)",  # "With: Name"
        #     r"Interview with\s+([^.\n]+)",  # "Interview with Name"
        #     r"Joined by\s+([^.\n]+)",  # "Joined by Name"
        # ]
        #
        # for pattern in patterns:
        #     matches = re.findall(pattern, description, re.IGNORECASE)
        #     for match in matches:
        #         # Split on commas and 'and' to get individual names
        #         names = re.split(r',|\sand\s', match)
        #         guests.extend([name.strip() for name in names if name.strip()])
        #
        # # Remove duplicates while preserving order
        # seen = set()
        # unique_guests = []
        # for guest in guests:
        #     if guest.lower() not in seen:
        #         seen.add(guest.lower())
        #         unique_guests.append(guest)
        #
        # return unique_guests[:5]  # Limit to 5 guests max

