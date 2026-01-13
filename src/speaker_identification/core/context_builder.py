#!/usr/bin/env python3
"""
Context Builder for Speaker Identification
===========================================

Builds rich context from database for LLM speaker identification.
Queries channels, episodes, speakers, and transcripts efficiently.
"""

import json
from typing import Dict, List, Optional, Union
from difflib import SequenceMatcher
from sqlalchemy import text
from datetime import datetime

from src.database.session import get_session
from src.utils.logger import setup_worker_logger

logger = setup_worker_logger('speaker_identification.context_builder')


def normalize_host_name(name: str) -> str:
    """
    Normalize a host name for matching.

    Removes titles, normalizes whitespace, and lowercases.

    Args:
        name: Raw host name

    Returns:
        Normalized name for comparison
    """
    name = name.lower().strip()
    # Remove common titles
    for title in ['dr.', 'dr ', 'prof.', 'prof ', 'mr.', 'mr ', 'mrs.', 'mrs ', 'ms.', 'ms ']:
        name = name.replace(title, '')
    # Normalize whitespace
    return ' '.join(name.split())


def fuzzy_name_match(name1: str, name2: str) -> float:
    """
    Calculate similarity score between two names.

    Uses multiple strategies: exact match, SequenceMatcher,
    containment, and last-name matching.

    Args:
        name1: First name (normalized)
        name2: Second name (normalized)

    Returns:
        Similarity score 0.0-1.0
    """
    # Exact match
    if name1 == name2:
        return 1.0

    # SequenceMatcher ratio
    score = SequenceMatcher(None, name1, name2).ratio()

    # Containment boost (e.g., "huberman" in "andrew huberman")
    if name1 in name2 or name2 in name1:
        score = max(score, 0.90)

    # Last name match boost (for "First Last" formats)
    parts1, parts2 = name1.split(), name2.split()
    if len(parts1) >= 1 and len(parts2) >= 1:
        # Compare last words (likely surnames)
        if parts1[-1] == parts2[-1] and len(parts1[-1]) > 2:
            score = max(score, 0.85)

    return score


def merge_similar_names(
    host_list: List[Dict],
    threshold: float = 0.85
) -> List[Dict]:
    """
    Merge name variations in a host list using fuzzy matching.

    Groups similar names together, keeping the most common variant
    as the canonical name.

    Args:
        host_list: List of dicts with 'name', 'count', 'episode_ids'
        threshold: Minimum similarity score to merge (default 0.85)

    Returns:
        Merged list with deduplicated names
    """
    if not host_list:
        return []

    # Sort by count descending - most frequent names become canonical
    sorted_hosts = sorted(host_list, key=lambda x: x['count'], reverse=True)

    merged = []
    used_indices = set()

    for i, host in enumerate(sorted_hosts):
        if i in used_indices:
            continue

        # Start a new group with this host
        group = {
            'name': host['name'],  # Keep original casing of most frequent
            'normalized_name': normalize_host_name(host['name']),
            'count': host['count'],
            'episode_ids': set(host['episode_ids']),
            'variants': [host['name']]
        }
        used_indices.add(i)

        # Find similar names to merge
        for j, other in enumerate(sorted_hosts):
            if j in used_indices:
                continue

            other_norm = normalize_host_name(other['name'])
            similarity = fuzzy_name_match(group['normalized_name'], other_norm)

            if similarity >= threshold:
                group['count'] += other['count']
                group['episode_ids'].update(other['episode_ids'])
                group['variants'].append(other['name'])
                used_indices.add(j)

        # Convert episode_ids back to list
        group['episode_ids'] = list(group['episode_ids'])
        merged.append(group)

    # Sort by count descending
    return sorted(merged, key=lambda x: x['count'], reverse=True)


class ContextBuilder:
    """Build rich context from database for speaker identification."""

    def get_channel_context(self, channel_id: int) -> Optional[Dict]:
        """
        Get channel metadata for Phase 1A host identification.

        Args:
            channel_id: Channel ID

        Returns:
            {
                'id': channel_id,
                'name': 'Huberman Lab',
                'platform': 'podcast',
                'description': 'The Huberman Lab podcast is hosted...',
                'metadata': {...}  # platform_metadata JSON
            }
        """
        with get_session() as session:
            query = text("""
                SELECT
                    id,
                    display_name as name,
                    platform,
                    description,
                    platform_metadata as metadata
                FROM channels
                WHERE id = :channel_id
            """)

            result = session.execute(query, {'channel_id': channel_id}).fetchone()

            if not result:
                return None

            return {
                'id': result.id,
                'name': result.name,
                'platform': result.platform,
                'description': result.description or '',
                'metadata': result.metadata or {}
            }

    def get_channels_with_unassigned_speakers(
        self,
        project: Optional[str] = None,
        min_unassigned: int = 5,
        limit: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict]:
        """
        Get channels with unassigned speakers for processing.

        Args:
            project: Filter to specific project (e.g., 'CPRMV')
            min_unassigned: Minimum number of unassigned high-quality speakers
            limit: Maximum number of channels to return
            start_date: Filter content published on or after this date (YYYY-MM-DD)
            end_date: Filter content published before this date (YYYY-MM-DD)

        Returns:
            List of channel dicts with unassigned speaker counts
        """
        with get_session() as session:
            # Build date filter clause
            date_filters = []
            if start_date:
                date_filters.append("co.publish_date >= :start_date")
            if end_date:
                date_filters.append("co.publish_date < :end_date")
            date_clause = " AND ".join(date_filters) if date_filters else "TRUE"

            if project:
                query = text(f"""
                    SELECT DISTINCT
                        c.id,
                        c.display_name as name,
                        c.platform,
                        c.description,
                        COUNT(DISTINCT s.id) FILTER (
                            WHERE s.speaker_identity_id IS NULL
                            AND s.embedding_quality_score >= 0.65
                        ) as unassigned_count,
                        COUNT(DISTINCT co.content_id) as episode_count
                    FROM channels c
                    JOIN content co ON c.id = co.channel_id
                    JOIN speakers s ON s.content_id = co.content_id
                    WHERE :project = ANY(co.projects)
                      AND s.embedding IS NOT NULL
                      AND {date_clause}
                    GROUP BY c.id, c.display_name, c.platform, c.description
                    HAVING COUNT(DISTINCT s.id) FILTER (
                        WHERE s.speaker_identity_id IS NULL
                        AND s.embedding_quality_score >= 0.65
                    ) >= :min_unassigned
                    ORDER BY unassigned_count DESC, episode_count DESC
                    LIMIT :limit
                """)
                params = {
                    'project': project,
                    'min_unassigned': min_unassigned,
                    'limit': limit or 999999,
                    'start_date': start_date,
                    'end_date': end_date
                }
            else:
                query = text(f"""
                    SELECT
                        c.id,
                        c.display_name as name,
                        c.platform,
                        c.description,
                        COUNT(DISTINCT s.id) FILTER (
                            WHERE s.speaker_identity_id IS NULL
                            AND s.embedding_quality_score >= 0.65
                        ) as unassigned_count,
                        COUNT(DISTINCT co.content_id) as episode_count
                    FROM channels c
                    JOIN content co ON c.id = co.channel_id
                    JOIN speakers s ON s.content_id = co.content_id
                    WHERE s.embedding IS NOT NULL
                      AND {date_clause}
                    GROUP BY c.id, c.display_name, c.platform, c.description
                    HAVING COUNT(DISTINCT s.id) FILTER (
                        WHERE s.speaker_identity_id IS NULL
                        AND s.embedding_quality_score >= 0.65
                    ) >= :min_unassigned
                    ORDER BY unassigned_count DESC, episode_count DESC
                    LIMIT :limit
                """)
                params = {
                    'min_unassigned': min_unassigned,
                    'limit': limit or 999999,
                    'start_date': start_date,
                    'end_date': end_date
                }

            results = session.execute(query, params).fetchall()

            return [
                {
                    'id': row.id,
                    'name': row.name,
                    'platform': row.platform,
                    'description': row.description or '',
                    'unassigned_count': row.unassigned_count,
                    'episode_count': row.episode_count
                }
                for row in results
            ]

    def get_cached_channel_hosts(self, channel_id: int) -> List[Dict]:
        """
        Get cached host identifications for a channel.

        Args:
            channel_id: Channel ID

        Returns:
            List of cached hosts:
            [
                {
                    'name': 'Andrew Huberman',
                    'confidence': 0.95,
                    'reasoning': 'Explicitly stated',
                    'method': 'llm_channel_description',
                    'identified_at': datetime
                }
            ]
        """
        with get_session() as session:
            query = text("""
                SELECT
                    host_name as name,
                    confidence,
                    reasoning,
                    method,
                    identified_at,
                    metadata
                FROM channel_host_cache
                WHERE channel_id = :channel_id
                ORDER BY confidence DESC
            """)

            results = session.execute(query, {'channel_id': channel_id}).fetchall()

            return [
                {
                    'name': row.name,
                    'confidence': row.confidence,
                    'reasoning': row.reasoning,
                    'method': row.method,
                    'identified_at': row.identified_at,
                    'metadata': row.metadata or {}
                }
                for row in results
            ]

    def get_episodes_with_unassigned_speakers(
        self,
        channel_id: int,
        min_quality: float = 0.65,
        limit: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict]:
        """
        Get episodes from a channel that don't have hosts/guests populated yet.

        Phase 1 is metadata-only - it extracts speaker info from episode titles
        and descriptions. It doesn't require embeddings.

        Args:
            channel_id: Channel ID
            min_quality: Not used (kept for API compatibility)
            limit: Maximum number of episodes
            start_date: Filter content published on or after this date (YYYY-MM-DD)
            end_date: Filter content published before this date (YYYY-MM-DD)

        Returns:
            List of episode dicts with metadata
        """
        with get_session() as session:
            # Build date filter clause
            date_filters = []
            if start_date:
                date_filters.append("c.publish_date >= :start_date")
            if end_date:
                date_filters.append("c.publish_date < :end_date")
            date_clause = " AND ".join(date_filters) if date_filters else "TRUE"

            # Get episodes that haven't been processed by Phase 1 yet
            # Uses metadata_speakers_extracted flag to track processing status
            query = text(f"""
                SELECT
                    c.content_id,
                    c.title,
                    c.description,
                    c.publish_date,
                    c.duration as episode_duration
                FROM content c
                WHERE c.channel_id = :channel_id
                  AND (c.metadata_speakers_extracted IS NULL OR c.metadata_speakers_extracted = false)
                  AND {date_clause}
                ORDER BY c.publish_date DESC
                LIMIT :limit
            """)

            results = session.execute(query, {
                'channel_id': channel_id,
                'limit': limit or 999999,
                'start_date': start_date,
                'end_date': end_date
            }).fetchall()

            return [
                {
                    'content_id': row.content_id,
                    'title': row.title,
                    'description': row.description or '',
                    'publish_date': row.publish_date.isoformat() if row.publish_date else '',
                    'episode_duration': row.episode_duration
                }
                for row in results
            ]

    def get_episodes_by_priority(
        self,
        projects: Optional[List[str]] = None,
        project_priorities: Optional[Dict[str, int]] = None,
        limit: int = 50,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict]:
        """
        Get episodes needing Phase 1 processing, ordered by priority.

        Prioritization (matches calculate_priority_by_date):
        1. Recent content (last 30 days) - gets 10M priority boost
        2. Then by project priority (higher priority projects first)
        3. Then by publish date (newer content first)

        Args:
            projects: List of project names to filter to
            project_priorities: Dict mapping project name to priority (1-3)
            limit: Maximum number of episodes to return
            start_date: Filter content published on or after this date (YYYY-MM-DD)
            end_date: Filter content published before this date (YYYY-MM-DD)

        Returns:
            List of episode dicts with channel info, ordered by priority
        """
        with get_session() as session:
            filters = [
                "(c.metadata_speakers_extracted IS NULL OR c.metadata_speakers_extracted = false)"
            ]
            params = {'limit': limit}

            if projects:
                projects_literal = "ARRAY[" + ",".join(f"'{p}'" for p in projects) + "]::varchar[]"
                filters.append(f"c.projects && {projects_literal}")

            if start_date:
                filters.append("c.publish_date >= :start_date")
                params['start_date'] = start_date
            if end_date:
                filters.append("c.publish_date < :end_date")
                params['end_date'] = end_date

            filter_clause = " AND ".join(filters)

            # Build project priority CASE expression
            if project_priorities:
                priority_cases = " ".join(
                    f"WHEN '{proj}' = ANY(c.projects) THEN {priority}"
                    for proj, priority in project_priorities.items()
                )
                project_priority_expr = f"CASE {priority_cases} ELSE 1 END"
            else:
                project_priority_expr = "1"

            query = text(f"""
                SELECT
                    c.content_id,
                    c.title,
                    c.description,
                    c.publish_date,
                    c.duration as episode_duration,
                    c.channel_id,
                    ch.display_name as channel_name,
                    ch.platform,
                    -- Calculate priority matching calculate_priority_by_date
                    (
                        -- Base date priority
                        (EXTRACT(EPOCH FROM c.publish_date)::bigint / 86400 - 20000) * 1000
                        -- Add project priority band
                        + ({project_priority_expr}) * 1000000
                        -- Add recency boost for content < 30 days old
                        + CASE
                            WHEN c.publish_date >= (NOW() - INTERVAL '30 days') THEN 10000000
                            ELSE 0
                          END
                    ) as calculated_priority
                FROM content c
                JOIN channels ch ON c.channel_id = ch.id
                WHERE {filter_clause}
                ORDER BY calculated_priority DESC
                LIMIT :limit
            """)

            results = session.execute(query, params).fetchall()

            return [
                {
                    'content_id': row.content_id,
                    'title': row.title,
                    'description': row.description or '',
                    'publish_date': row.publish_date,
                    'episode_duration': row.episode_duration,
                    'channel_id': row.channel_id,
                    'channel_name': row.channel_name,
                    'platform': row.platform,
                    'calculated_priority': row.calculated_priority
                }
                for row in results
            ]

    def count_episodes_by_priority(
        self,
        projects: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> int:
        """
        Count episodes needing Phase 1 processing.

        Args:
            projects: List of project names to filter to
            start_date: Filter content published on or after this date (YYYY-MM-DD)
            end_date: Filter content published before this date (YYYY-MM-DD)

        Returns:
            Total count of episodes needing processing
        """
        with get_session() as session:
            filters = [
                "(c.metadata_speakers_extracted IS NULL OR c.metadata_speakers_extracted = false)"
            ]
            params = {}

            if projects:
                projects_literal = "ARRAY[" + ",".join(f"'{p}'" for p in projects) + "]::varchar[]"
                filters.append(f"c.projects && {projects_literal}")

            if start_date:
                filters.append("c.publish_date >= :start_date")
                params['start_date'] = start_date
            if end_date:
                filters.append("c.publish_date < :end_date")
                params['end_date'] = end_date

            filter_clause = " AND ".join(filters)

            query = text(f"""
                SELECT COUNT(*) as total
                FROM content c
                WHERE {filter_clause}
            """)

            result = session.execute(query, params).fetchone()
            return result.total if result else 0

    def get_episode_speakers(
        self,
        content_id: str,
        min_quality: float = 0.65,
        only_unassigned: bool = True
    ) -> List[Dict]:
        """
        Get speakers from an episode with metrics for Phase 1B.

        Args:
            content_id: Content ID
            min_quality: Minimum speaker quality score
            only_unassigned: Only return unassigned speakers

        Returns:
            List of speaker dicts sorted by duration:
            [
                {
                    'speaker_id': 123,
                    'local_speaker_id': 'SPEAKER_2',
                    'duration': 1981.1,
                    'duration_pct': 66.4,
                    'segments': 21,
                    'quality': 0.80,
                    'speaker_identity_id': None
                }
            ]
        """
        with get_session() as session:
            query = text("""
                SELECT
                    s.id as speaker_id,
                    s.local_speaker_id,
                    s.duration,
                    CASE
                        WHEN c.duration > 0 THEN (s.duration / c.duration * 100)
                        ELSE 0
                    END as duration_pct,
                    s.segment_count as segments,
                    s.embedding_quality_score as quality,
                    s.speaker_identity_id
                FROM speakers s
                JOIN content c ON s.content_id = c.content_id
                WHERE s.content_id = :content_id
                  AND s.embedding IS NOT NULL
                  AND s.embedding_quality_score >= :min_quality
                  AND (:only_unassigned = FALSE OR s.speaker_identity_id IS NULL)
                ORDER BY s.duration DESC
            """)

            results = session.execute(query, {
                'content_id': content_id,
                'min_quality': min_quality,
                'only_unassigned': only_unassigned
            }).fetchall()

            return [
                {
                    'speaker_id': row.speaker_id,
                    'local_speaker_id': row.local_speaker_id,
                    'duration': float(row.duration),
                    'duration_pct': float(row.duration_pct),
                    'segments': row.segments,
                    'quality': float(row.quality),
                    'speaker_identity_id': row.speaker_identity_id
                }
                for row in results
            ]

    def save_channel_hosts(
        self,
        channel_id: int,
        hosts: List[Dict]
    ):
        """
        Save hosts to channels.hosts field.

        Args:
            channel_id: Channel ID
            hosts: List of host dicts with keys: name, confidence, reasoning
        """
        with get_session() as session:
            import json
            # Use CAST instead of :: to avoid bind parameter conflicts
            query = text("""
                UPDATE channels
                SET hosts = CAST(:hosts_json AS jsonb)
                WHERE id = :channel_id
            """)

            session.execute(query, {
                'channel_id': channel_id,
                'hosts_json': json.dumps(hosts)
            })
            session.commit()

            logger.info(f"Saved {len(hosts)} host(s) to channel {channel_id}")

            # Also cache in channel_host_cache for legacy compatibility
            for host in hosts:
                cache_query = text("""
                    INSERT INTO channel_host_cache (
                        channel_id, host_name, confidence, reasoning, method, metadata
                    )
                    VALUES (
                        :channel_id, :host_name, :confidence, :reasoning, :method, :metadata
                    )
                    ON CONFLICT (channel_id, host_name) DO UPDATE SET
                        confidence = EXCLUDED.confidence,
                        reasoning = EXCLUDED.reasoning,
                        identified_at = NOW()
                """)

                # Convert categorical confidence to numeric for cache table
                conf_map = {"certain": 0.95, "very_likely": 0.80, "somewhat_likely": 0.65}
                numeric_conf = conf_map.get(host.get('confidence', 'certain'), 0.80)

                session.execute(cache_query, {
                    'channel_id': channel_id,
                    'host_name': host['name'],
                    'confidence': numeric_conf,
                    'reasoning': host.get('reasoning', ''),
                    'method': 'llm_channel_description',
                    'metadata': json.dumps(host)
                })
            session.commit()

    def sync_channel_hosts_from_episodes(
        self,
        channel_id: int,
        min_episodes: int = 10
    ) -> List[str]:
        """
        Promote frequent episode-level hosts to channel-level hosts.

        For multi-host channels, the channel description may only mention the
        primary host. This method looks at content.hosts frequency and promotes
        any host appearing in >= min_episodes to channels.hosts and channel_host_cache.

        Args:
            channel_id: Channel ID
            min_episodes: Minimum episode appearances to qualify (default 10)

        Returns:
            List of newly promoted host names
        """
        import json

        # Get hosts from episode frequency
        frequent_hosts = self.get_host_frequency_for_channel(
            channel_id=channel_id,
            min_count=min_episodes,
            merge_names=True
        )

        if not frequent_hosts:
            return []

        # Get existing channel-level hosts
        existing_hosts = self.get_cached_channel_hosts(channel_id)
        existing_names = {normalize_host_name(h['name']) for h in existing_hosts}

        # Find new hosts to promote
        new_hosts = []
        for host in frequent_hosts:
            normalized = normalize_host_name(host['name'])
            if normalized not in existing_names:
                new_hosts.append({
                    'name': host['name'],
                    'confidence': 'very_likely',
                    'reasoning': f"Appears as host in {host['count']} episodes",
                    'episode_count': host['count']
                })

        if not new_hosts:
            logger.info(f"All {len(frequent_hosts)} frequent hosts already in channel cache")
            return []

        logger.info(f"Promoting {len(new_hosts)} new hosts to channel level:")
        for h in new_hosts:
            logger.info(f"  - {h['name']} ({h['episode_count']} episodes)")

        # Add to channel_host_cache
        with get_session() as session:
            for host in new_hosts:
                cache_query = text("""
                    INSERT INTO channel_host_cache (
                        channel_id, host_name, confidence, reasoning, method, metadata
                    )
                    VALUES (
                        :channel_id, :host_name, :confidence, :reasoning, :method, :metadata
                    )
                    ON CONFLICT (channel_id, host_name) DO UPDATE SET
                        confidence = EXCLUDED.confidence,
                        reasoning = EXCLUDED.reasoning,
                        identified_at = NOW()
                """)

                session.execute(cache_query, {
                    'channel_id': channel_id,
                    'host_name': host['name'],
                    'confidence': 0.85,  # High confidence from frequency
                    'reasoning': host['reasoning'],
                    'method': 'episode_frequency_promotion',
                    'metadata': json.dumps(host)
                })

            # Also update channels.hosts to include all hosts
            all_hosts = existing_hosts + new_hosts
            hosts_json = [
                {
                    'name': h.get('name'),
                    'confidence': h.get('confidence', 'very_likely'),
                    'reasoning': h.get('reasoning', '')
                }
                for h in all_hosts
            ]

            update_query = text("""
                UPDATE channels
                SET hosts = CAST(:hosts_json AS jsonb)
                WHERE id = :channel_id
            """)
            session.execute(update_query, {
                'channel_id': channel_id,
                'hosts_json': json.dumps(hosts_json)
            })

            session.commit()

        return [h['name'] for h in new_hosts]

    def save_content_speakers(
        self,
        content_id: str,
        speakers: Dict,
        channel_id: Optional[int] = None
    ):
        """
        Save speakers to content.hosts, content.guests, and content.mentioned columns.

        If channel_id is provided, normalizes speaker names using channel aliases.

        Args:
            content_id: Content ID
            speakers: Dict with keys:
                - 'speakers': List of speaker dicts with name, role, confidence, reasoning
                - 'mentioned': List of mentioned people dicts with name, reasoning
            channel_id: Optional channel ID for name normalization
        """
        with get_session() as session:
            import json

            # Extract hosts, guests, and mentioned from input
            speakers_list = speakers.get('speakers', [])
            mentioned_list = speakers.get('mentioned', [])

            # Normalize names using channel aliases if available
            if channel_id:
                alias_mapping = self.get_host_alias_mapping(channel_id)
                if alias_mapping:
                    for speaker in speakers_list:
                        if speaker.get('name') in alias_mapping:
                            speaker['name'] = alias_mapping[speaker['name']]
                    for mentioned in mentioned_list:
                        if mentioned.get('name') in alias_mapping:
                            mentioned['name'] = alias_mapping[mentioned['name']]

            hosts = [s for s in speakers_list if s.get('role') == 'host']
            guests = [s for s in speakers_list if s.get('role') == 'guest']

            query = text("""
                UPDATE content
                SET hosts = CAST(:hosts_json AS jsonb),
                    guests = CAST(:guests_json AS jsonb),
                    mentioned = CAST(:mentioned_json AS jsonb),
                    metadata_speakers_extracted = true
                WHERE content_id = :content_id
            """)

            session.execute(query, {
                'content_id': content_id,
                'hosts_json': json.dumps(hosts),
                'guests_json': json.dumps(guests),
                'mentioned_json': json.dumps(mentioned_list)
            })
            session.commit()

            logger.debug(f"Saved {len(hosts)} host(s), {len(guests)} guest(s), {len(mentioned_list)} mentioned to content {content_id}")

    def mark_metadata_speakers_extracted(self, content_id: str):
        """
        Mark an episode as having been processed by Phase 1, even if no speakers were found.

        Args:
            content_id: Content ID
        """
        with get_session() as session:
            query = text("""
                UPDATE content
                SET metadata_speakers_extracted = true
                WHERE content_id = :content_id
            """)
            session.execute(query, {'content_id': content_id})
            session.commit()
            logger.debug(f"Marked content {content_id} as metadata_speakers_extracted")

    def get_content_speakers(self, content_id: str) -> Dict:
        """
        Get hosts, guests, and mentioned people for a content item.

        Args:
            content_id: Content ID (string)

        Returns:
            {
                'hosts': [{"name": "...", "confidence": "...", "reasoning": "..."}],
                'guests': [{"name": "...", "confidence": "...", "reasoning": "..."}],
                'mentioned': [{"name": "...", "reasoning": "..."}]
            }
        """
        with get_session() as session:
            query = text("""
                SELECT hosts, guests, mentioned
                FROM content
                WHERE content_id = :content_id
            """)

            result = session.execute(query, {'content_id': content_id}).fetchone()

            if not result:
                return {'hosts': [], 'guests': [], 'mentioned': []}

            return {
                'hosts': result.hosts or [],
                'guests': result.guests or [],
                'mentioned': result.mentioned or []
            }

    def get_content_host_names(self, content_id: str) -> List[str]:
        """
        Get just the host names for a content item (convenience method for Phase 2).

        Args:
            content_id: Content ID (string)

        Returns:
            List of host names
        """
        speakers = self.get_content_speakers(content_id)
        return [h.get('name', '').strip() for h in speakers['hosts'] if h.get('name')]

    def get_host_frequency_for_channel(
        self,
        channel_id: int,
        min_count: int = 10,
        merge_names: bool = True
    ) -> List[Dict]:
        """
        Aggregate host names from content.hosts across a channel.

        Queries all episodes in the channel, extracts host names from
        the content.hosts JSONB column, and aggregates by frequency.
        Optionally merges similar name variations.

        Args:
            channel_id: Channel ID
            min_count: Minimum occurrences to include (default 10)
            merge_names: Whether to merge similar names (default True)

        Returns:
            List of dicts sorted by count descending:
            [
                {
                    'name': 'Andrew Huberman',
                    'normalized_name': 'andrew huberman',
                    'count': 450,
                    'episode_ids': ['yt_abc123', ...],
                    'variants': ['Dr. Andrew Huberman', 'Andrew Huberman']
                }
            ]
        """
        with get_session() as session:
            # Query content.hosts for all episodes in channel
            # Use jsonb_array_elements to unnest the hosts array
            query = text("""
                SELECT
                    host_entry->>'name' as host_name,
                    COUNT(*) as occurrence_count,
                    ARRAY_AGG(DISTINCT c.content_id) as episode_ids
                FROM content c,
                     jsonb_array_elements(c.hosts) as host_entry
                WHERE c.channel_id = :channel_id
                  AND c.hosts IS NOT NULL
                  AND jsonb_array_length(c.hosts) > 0
                  AND host_entry->>'name' IS NOT NULL
                  AND TRIM(host_entry->>'name') != ''
                GROUP BY host_entry->>'name'
                ORDER BY occurrence_count DESC
            """)

            results = session.execute(query, {'channel_id': channel_id}).fetchall()

            if not results:
                logger.info(f"No host data found in content.hosts for channel {channel_id}")
                return []

            # Build initial host list
            host_list = [
                {
                    'name': row.host_name,
                    'count': row.occurrence_count,
                    'episode_ids': list(row.episode_ids) if row.episode_ids else []
                }
                for row in results
            ]

            logger.info(f"Found {len(host_list)} unique host names in channel {channel_id}")

            # Merge similar names if requested
            if merge_names:
                host_list = merge_similar_names(host_list)
                logger.info(f"After merging similar names: {len(host_list)} hosts")

            # Filter by minimum count
            qualified_hosts = [h for h in host_list if h['count'] >= min_count]

            logger.info(
                f"Qualified hosts with >={min_count} occurrences: {len(qualified_hosts)}"
            )
            for host in qualified_hosts:
                variants_str = f" (variants: {host.get('variants', [])})" if len(host.get('variants', [])) > 1 else ""
                logger.info(f"  - {host['name']}: {host['count']} episodes{variants_str}")

            return qualified_hosts

    def get_speaker_transcript_context(
        self,
        speaker_id: int,
        max_utterance_chars: int = 500,
        use_sentences: bool = True
    ) -> Optional[Dict]:
        """
        Get transcript context for a speaker (for Phase 2 LLM verification).

        Args:
            speaker_id: Speaker ID
            max_utterance_chars: Maximum characters per utterance
            use_sentences: If True, query sentences table (default).
                          If False, use legacy speaker_transcriptions table.

        Returns:
            {
                'speaker_id': int,
                'content_id': str,
                'local_speaker_id': str,
                'episode_title': str,
                'episode_description': str,
                'known_hosts': List[str],
                'first_utterance': str,
                'last_utterance': str,
                'turn_before': str or None,
                'turn_after': str or None,
                'total_turns': int,
                'duration': float,
                'duration_pct': float
            }
        """
        with get_session() as session:
            # Get speaker and episode info
            speaker_query = text("""
                SELECT
                    s.id,
                    s.content_id,
                    s.local_speaker_id,
                    s.duration,
                    c.title,
                    c.description,
                    c.duration as episode_duration,
                    c.stitch_version,
                    ch.hosts,
                    ch.id as channel_id
                FROM speakers s
                JOIN content c ON s.content_id = c.content_id
                JOIN channels ch ON c.channel_id = ch.id
                WHERE s.id = :speaker_id
            """)

            speaker_result = session.execute(
                speaker_query,
                {'speaker_id': speaker_id}
            ).fetchone()

            if not speaker_result:
                logger.warning(f"Speaker {speaker_id} not found")
                return None

            content_id = speaker_result.content_id
            local_speaker_id = speaker_result.local_speaker_id

            # Get known hosts
            known_hosts = []
            if speaker_result.hosts:
                known_hosts = [h.get('name', '') for h in speaker_result.hosts]

            # Calculate duration percentage
            duration_pct = 0.0
            if speaker_result.episode_duration and speaker_result.episode_duration > 0:
                duration_pct = (speaker_result.duration / speaker_result.episode_duration) * 100

            # Get the content.id (integer) from content_id (string) for speaker_transcriptions join
            content_db_id_query = text("""
                SELECT id FROM content WHERE content_id = :content_id
            """)

            content_db_id_result = session.execute(
                content_db_id_query,
                {'content_id': content_id}
            ).fetchone()

            if not content_db_id_result:
                logger.warning(f"Content not found: {content_id}")
                return None

            content_db_id = content_db_id_result.id

            # Get speaker turns - use sentences table (aggregated) or legacy speaker_transcriptions
            if use_sentences:
                # Query sentences table, aggregate back into turns
                turns_query = text("""
                    SELECT
                        turn_index,
                        string_agg(text, ' ' ORDER BY sentence_in_turn) as text
                    FROM sentences
                    WHERE speaker_id = :speaker_id
                      AND content_id = :content_db_id
                      AND stitch_version = :stitch_version
                    GROUP BY turn_index
                    ORDER BY turn_index
                """)
            else:
                # Legacy: query speaker_transcriptions directly
                turns_query = text("""
                    SELECT
                        turn_index,
                        text
                    FROM speaker_transcriptions
                    WHERE speaker_id = :speaker_id
                      AND content_id = :content_db_id
                      AND stitch_version = :stitch_version
                    ORDER BY turn_index
                """)

            speaker_turns = session.execute(
                turns_query,
                {
                    'speaker_id': speaker_id,
                    'content_db_id': content_db_id,
                    'stitch_version': speaker_result.stitch_version
                }
            ).fetchall()

            # Fallback: try without stitch_version constraint if no turns found
            if not speaker_turns:
                if use_sentences:
                    turns_query_fallback = text("""
                        SELECT
                            turn_index,
                            string_agg(text, ' ' ORDER BY sentence_in_turn) as text
                        FROM sentences
                        WHERE speaker_id = :speaker_id
                          AND content_id = :content_db_id
                        GROUP BY turn_index
                        ORDER BY turn_index
                    """)
                else:
                    turns_query_fallback = text("""
                        SELECT
                            turn_index,
                            text
                        FROM speaker_transcriptions
                        WHERE speaker_id = :speaker_id
                          AND content_id = :content_db_id
                        ORDER BY turn_index
                    """)
                speaker_turns = session.execute(
                    turns_query_fallback,
                    {
                        'speaker_id': speaker_id,
                        'content_db_id': content_db_id
                    }
                ).fetchall()

            if not speaker_turns:
                logger.warning(f"No turns found for speaker {speaker_id}")
                return None

            # Get first and last utterances
            first_turn = speaker_turns[0]
            last_turn = speaker_turns[-1]

            first_utterance = (first_turn.text or '')[:max_utterance_chars]
            last_utterance = (last_turn.text or '')[:max_utterance_chars]

            first_turn_idx = first_turn.turn_index
            last_turn_idx = last_turn.turn_index

            # Get surrounding context for FIRST utterance
            turn_before_first = None
            turn_before_first_speaker_name = None
            turn_after_first = None
            turn_after_first_speaker_name = None

            # Turn before first - also get speaker name if assigned
            # NOTE: Don't filter by stitch_version here - content.stitch_version may not match
            # sentences.stitch_version due to re-stitching
            if use_sentences:
                before_first_query = text("""
                    SELECT
                        string_agg(sen.text, ' ' ORDER BY sen.sentence_in_turn) as text,
                        si.primary_name as speaker_name
                    FROM sentences sen
                    LEFT JOIN speakers s ON sen.speaker_id = s.id
                    LEFT JOIN speaker_identities si ON s.speaker_identity_id = si.id
                    WHERE sen.content_id = :content_db_id
                      AND sen.turn_index < :first_turn_idx
                    GROUP BY sen.turn_index, si.primary_name
                    ORDER BY sen.turn_index DESC
                    LIMIT 1
                """)
            else:
                before_first_query = text("""
                    SELECT st.text, si.primary_name as speaker_name
                    FROM speaker_transcriptions st
                    LEFT JOIN speakers s ON st.speaker_id = s.id
                    LEFT JOIN speaker_identities si ON s.speaker_identity_id = si.id
                    WHERE st.content_id = :content_db_id
                      AND st.turn_index < :first_turn_idx
                    ORDER BY st.turn_index DESC
                    LIMIT 1
                """)
            before_first_result = session.execute(
                before_first_query,
                {
                    'content_db_id': content_db_id,
                    'first_turn_idx': first_turn_idx
                }
            ).fetchone()
            if before_first_result:
                turn_before_first = (before_first_result.text or '')[:max_utterance_chars]
                turn_before_first_speaker_name = before_first_result.speaker_name

            # Turn after first - also get speaker name if assigned
            if use_sentences:
                after_first_query = text("""
                    SELECT
                        string_agg(sen.text, ' ' ORDER BY sen.sentence_in_turn) as text,
                        si.primary_name as speaker_name
                    FROM sentences sen
                    LEFT JOIN speakers s ON sen.speaker_id = s.id
                    LEFT JOIN speaker_identities si ON s.speaker_identity_id = si.id
                    WHERE sen.content_id = :content_db_id
                      AND sen.turn_index > :first_turn_idx
                      AND sen.speaker_id != :speaker_id
                    GROUP BY sen.turn_index, si.primary_name
                    ORDER BY sen.turn_index ASC
                    LIMIT 1
                """)
            else:
                after_first_query = text("""
                    SELECT st.text, si.primary_name as speaker_name
                    FROM speaker_transcriptions st
                    LEFT JOIN speakers s ON st.speaker_id = s.id
                    LEFT JOIN speaker_identities si ON s.speaker_identity_id = si.id
                    WHERE st.content_id = :content_db_id
                      AND st.turn_index > :first_turn_idx
                      AND st.speaker_id != :speaker_id
                    ORDER BY st.turn_index ASC
                    LIMIT 1
                """)
            after_first_result = session.execute(
                after_first_query,
                {
                    'content_db_id': content_db_id,
                    'first_turn_idx': first_turn_idx,
                    'speaker_id': speaker_id
                }
            ).fetchone()
            if after_first_result:
                turn_after_first = (after_first_result.text or '')[:max_utterance_chars]
                turn_after_first_speaker_name = after_first_result.speaker_name

            # Get surrounding context for LAST utterance
            turn_before_last = None
            turn_before_last_speaker_name = None
            turn_after_last = None
            turn_after_last_speaker_name = None

            # Turn before last - also get speaker name if assigned
            if use_sentences:
                before_last_query = text("""
                    SELECT
                        string_agg(sen.text, ' ' ORDER BY sen.sentence_in_turn) as text,
                        si.primary_name as speaker_name
                    FROM sentences sen
                    LEFT JOIN speakers s ON sen.speaker_id = s.id
                    LEFT JOIN speaker_identities si ON s.speaker_identity_id = si.id
                    WHERE sen.content_id = :content_db_id
                      AND sen.turn_index < :last_turn_idx
                      AND sen.speaker_id != :speaker_id
                    GROUP BY sen.turn_index, si.primary_name
                    ORDER BY sen.turn_index DESC
                    LIMIT 1
                """)
            else:
                before_last_query = text("""
                    SELECT st.text, si.primary_name as speaker_name
                    FROM speaker_transcriptions st
                    LEFT JOIN speakers s ON st.speaker_id = s.id
                    LEFT JOIN speaker_identities si ON s.speaker_identity_id = si.id
                    WHERE st.content_id = :content_db_id
                      AND st.turn_index < :last_turn_idx
                      AND st.speaker_id != :speaker_id
                    ORDER BY st.turn_index DESC
                    LIMIT 1
                """)
            before_last_result = session.execute(
                before_last_query,
                {
                    'content_db_id': content_db_id,
                    'last_turn_idx': last_turn_idx,
                    'speaker_id': speaker_id
                }
            ).fetchone()
            if before_last_result:
                turn_before_last = (before_last_result.text or '')[:max_utterance_chars]
                turn_before_last_speaker_name = before_last_result.speaker_name

            # Turn after last - also get speaker name if assigned
            if use_sentences:
                after_last_query = text("""
                    SELECT
                        string_agg(sen.text, ' ' ORDER BY sen.sentence_in_turn) as text,
                        si.primary_name as speaker_name
                    FROM sentences sen
                    LEFT JOIN speakers s ON sen.speaker_id = s.id
                    LEFT JOIN speaker_identities si ON s.speaker_identity_id = si.id
                    WHERE sen.content_id = :content_db_id
                      AND sen.turn_index > :last_turn_idx
                    GROUP BY sen.turn_index, si.primary_name
                    ORDER BY sen.turn_index ASC
                    LIMIT 1
                """)
            else:
                after_last_query = text("""
                    SELECT st.text, si.primary_name as speaker_name
                    FROM speaker_transcriptions st
                    LEFT JOIN speakers s ON st.speaker_id = s.id
                    LEFT JOIN speaker_identities si ON s.speaker_identity_id = si.id
                    WHERE st.content_id = :content_db_id
                      AND st.turn_index > :last_turn_idx
                    ORDER BY st.turn_index ASC
                    LIMIT 1
                """)
            after_last_result = session.execute(
                after_last_query,
                {
                    'content_db_id': content_db_id,
                    'last_turn_idx': last_turn_idx
                }
            ).fetchone()
            if after_last_result:
                turn_after_last = (after_last_result.text or '')[:max_utterance_chars]
                turn_after_last_speaker_name = after_last_result.speaker_name

            return {
                'speaker_id': speaker_id,
                'content_id': content_id,
                'local_speaker_id': local_speaker_id,
                'episode_title': speaker_result.title or '',
                'episode_description': speaker_result.description or '',
                'known_hosts': known_hosts,
                'first_utterance': first_utterance,
                'last_utterance': last_utterance,
                'turn_before_first': turn_before_first,
                'turn_before_first_speaker_name': turn_before_first_speaker_name,
                'turn_after_first': turn_after_first,
                'turn_after_first_speaker_name': turn_after_first_speaker_name,
                'turn_before_last': turn_before_last,
                'turn_before_last_speaker_name': turn_before_last_speaker_name,
                'turn_after_last': turn_after_last,
                'turn_after_last_speaker_name': turn_after_last_speaker_name,
                # Backwards compat
                'turn_before': turn_before_first,
                'turn_after': turn_after_last,
                'total_turns': len(speaker_turns),
                'duration': float(speaker_result.duration),
                'duration_pct': float(duration_pct)
            }

    def get_host_name_distribution(self, channel_id: int) -> Dict[str, int]:
        """
        Get distribution of host names across all episodes for a channel.

        Args:
            channel_id: Channel ID

        Returns:
            Dict mapping host name -> episode count
        """
        with get_session() as session:
            query = text("""
                SELECT
                    host_entry->>'name' as host_name,
                    COUNT(DISTINCT content_id) as episode_count
                FROM content,
                     jsonb_array_elements(hosts) as host_entry
                WHERE channel_id = :channel_id
                  AND hosts IS NOT NULL
                  AND jsonb_array_length(hosts) > 0
                GROUP BY host_entry->>'name'
                ORDER BY episode_count DESC
            """)

            results = session.execute(query, {'channel_id': channel_id}).fetchall()

            return {row.host_name: row.episode_count for row in results if row.host_name}

    def apply_host_name_consolidations(
        self,
        channel_id: int,
        name_mapping: Dict[str, str]
    ) -> int:
        """
        Apply name consolidations to content.hosts for a channel.

        Updates each content record's hosts JSON, replacing variation names
        with their canonical forms and marking hosts_consolidated = true.

        Args:
            channel_id: Channel ID to update
            name_mapping: Dict mapping old_name -> canonical_name

        Returns:
            Number of content records updated
        """
        if not name_mapping:
            return 0

        with get_session() as session:
            # Get all content with hosts for this channel
            query = text("""
                SELECT id, hosts
                FROM content
                WHERE channel_id = :channel_id
                  AND hosts IS NOT NULL
                  AND jsonb_array_length(hosts) > 0
                  AND (hosts_consolidated IS NULL OR hosts_consolidated = false)
            """)

            results = session.execute(query, {'channel_id': channel_id}).fetchall()
            updated_count = 0

            for row in results:
                hosts = row.hosts
                changed = False

                # Update each host entry
                for host in hosts:
                    old_name = host.get('name')
                    if old_name in name_mapping:
                        host['name'] = name_mapping[old_name]
                        changed = True

                if changed:
                    # Update the content record
                    update_query = text("""
                        UPDATE content
                        SET hosts = :hosts,
                            hosts_consolidated = true,
                            last_updated = NOW()
                        WHERE id = :content_id
                    """)
                    session.execute(update_query, {
                        'hosts': json.dumps(hosts),
                        'content_id': row.id
                    })
                    updated_count += 1
                else:
                    # Mark as consolidated even if no changes (already canonical)
                    mark_query = text("""
                        UPDATE content
                        SET hosts_consolidated = true
                        WHERE id = :content_id
                    """)
                    session.execute(mark_query, {'content_id': row.id})

            session.commit()
            return updated_count

    def get_work_summary(
        self,
        projects: Optional[List[str]] = None
    ) -> Dict:
        """
        Get summary of work to be done across projects.

        Args:
            projects: List of project names to include. If None, includes all.

        Returns:
            Dict with channels, episodes, and per-channel breakdown
        """
        with get_session() as session:
            if projects:
                # Build project filter
                project_placeholders = ', '.join([f':p{i}' for i in range(len(projects))])
                project_params = {f'p{i}': p for i, p in enumerate(projects)}

                query = text(f"""
                    SELECT
                        ch.id as channel_id,
                        ch.display_name as channel_name,
                        ch.platform,
                        COUNT(DISTINCT c.content_id) as episode_count
                    FROM channels ch
                    JOIN content c ON ch.id = c.channel_id
                    WHERE (c.metadata_speakers_extracted IS NULL OR c.metadata_speakers_extracted = false)
                      AND EXISTS (
                          SELECT 1 FROM unnest(c.projects) p WHERE p IN ({project_placeholders})
                      )
                    GROUP BY ch.id, ch.display_name, ch.platform
                    ORDER BY episode_count DESC
                """)
                results = session.execute(query, project_params).fetchall()
            else:
                query = text("""
                    SELECT
                        ch.id as channel_id,
                        ch.display_name as channel_name,
                        ch.platform,
                        COUNT(DISTINCT c.content_id) as episode_count
                    FROM channels ch
                    JOIN content c ON ch.id = c.channel_id
                    WHERE (c.metadata_speakers_extracted IS NULL OR c.metadata_speakers_extracted = false)
                    GROUP BY ch.id, ch.display_name, ch.platform
                    ORDER BY episode_count DESC
                """)
                results = session.execute(query).fetchall()

            channels = []
            total_episodes = 0
            for row in results:
                channels.append({
                    'channel_id': row.channel_id,
                    'channel_name': row.channel_name,
                    'platform': row.platform,
                    'episode_count': row.episode_count
                })
                total_episodes += row.episode_count

            return {
                'total_channels': len(channels),
                'total_episodes': total_episodes,
                'channels': channels
            }

    def get_channels_needing_consolidation(
        self,
        project: Optional[str] = None,
        min_unique_names: int = 3
    ) -> List[Dict]:
        """
        Get channels that have hosts but no aliases configured yet.

        Args:
            project: Filter to specific project
            min_unique_names: Minimum unique host names to consider

        Returns:
            List of channel dicts with channel_id, channel_name, unique_names
        """
        with get_session() as session:
            if project:
                query = text("""
                    WITH channel_hosts AS (
                        SELECT
                            c.channel_id,
                            ch.display_name as channel_name,
                            COUNT(DISTINCT host_entry->>'name') as unique_names
                        FROM content c
                        JOIN channels ch ON ch.id = c.channel_id
                        CROSS JOIN jsonb_array_elements(c.hosts) as host_entry
                        WHERE :project = ANY(c.projects)
                          AND c.hosts IS NOT NULL
                          AND jsonb_array_length(c.hosts) > 0
                        GROUP BY c.channel_id, ch.display_name
                    ),
                    channels_with_aliases AS (
                        SELECT channel_id
                        FROM channel_host_cache
                        WHERE aliases IS NOT NULL AND array_length(aliases, 1) > 1
                        GROUP BY channel_id
                    )
                    SELECT
                        ch.channel_id,
                        ch.channel_name,
                        ch.unique_names
                    FROM channel_hosts ch
                    LEFT JOIN channels_with_aliases cwa ON cwa.channel_id = ch.channel_id
                    WHERE ch.unique_names >= :min_names
                      AND cwa.channel_id IS NULL
                    ORDER BY ch.unique_names DESC
                """)
                results = session.execute(query, {
                    'project': project,
                    'min_names': min_unique_names
                }).fetchall()
            else:
                query = text("""
                    WITH channel_hosts AS (
                        SELECT
                            c.channel_id,
                            ch.display_name as channel_name,
                            COUNT(DISTINCT host_entry->>'name') as unique_names
                        FROM content c
                        JOIN channels ch ON ch.id = c.channel_id
                        CROSS JOIN jsonb_array_elements(c.hosts) as host_entry
                        WHERE c.hosts IS NOT NULL
                          AND jsonb_array_length(c.hosts) > 0
                        GROUP BY c.channel_id, ch.display_name
                    ),
                    channels_with_aliases AS (
                        SELECT channel_id
                        FROM channel_host_cache
                        WHERE aliases IS NOT NULL AND array_length(aliases, 1) > 1
                        GROUP BY channel_id
                    )
                    SELECT
                        ch.channel_id,
                        ch.channel_name,
                        ch.unique_names
                    FROM channel_hosts ch
                    LEFT JOIN channels_with_aliases cwa ON cwa.channel_id = ch.channel_id
                    WHERE ch.unique_names >= :min_names
                      AND cwa.channel_id IS NULL
                    ORDER BY ch.unique_names DESC
                """)
                results = session.execute(query, {'min_names': min_unique_names}).fetchall()

            return [
                {
                    'channel_id': row.channel_id,
                    'channel_name': row.channel_name,
                    'unique_names': row.unique_names
                }
                for row in results
            ]

    def save_host_aliases(
        self,
        channel_id: int,
        canonical_name: str,
        aliases: List[str]
    ) -> bool:
        """
        Save name aliases for a host in channel_host_cache.

        Args:
            channel_id: Channel ID
            canonical_name: The canonical host name
            aliases: List of name variations that map to this host

        Returns:
            True if updated successfully
        """
        if not aliases:
            return False

        with get_session() as session:
            # Update the host entry with aliases
            query = text("""
                UPDATE channel_host_cache
                SET aliases = :aliases
                WHERE channel_id = :channel_id
                  AND host_name = :canonical_name
            """)

            result = session.execute(query, {
                'channel_id': channel_id,
                'canonical_name': canonical_name,
                'aliases': aliases
            })
            session.commit()

            return result.rowcount > 0

    def get_host_alias_mapping(self, channel_id: int) -> Dict[str, str]:
        """
        Get a mapping of all aliases to canonical names for a channel.

        Args:
            channel_id: Channel ID

        Returns:
            Dict mapping alias -> canonical_name
        """
        with get_session() as session:
            query = text("""
                SELECT host_name, aliases
                FROM channel_host_cache
                WHERE channel_id = :channel_id
                  AND aliases IS NOT NULL
                  AND array_length(aliases, 1) > 0
            """)

            results = session.execute(query, {'channel_id': channel_id}).fetchall()

            mapping = {}
            for row in results:
                for alias in row.aliases:
                    if alias != row.host_name:
                        mapping[alias] = row.host_name

            return mapping

    def normalize_speaker_name(self, channel_id: int, name: str) -> str:
        """
        Normalize a speaker name using channel aliases.

        Args:
            channel_id: Channel ID
            name: Speaker name to normalize

        Returns:
            Canonical name if alias found, otherwise original name
        """
        with get_session() as session:
            # Check if this name is an alias for any host
            query = text("""
                SELECT host_name
                FROM channel_host_cache
                WHERE channel_id = :channel_id
                  AND :name = ANY(aliases)
                LIMIT 1
            """)

            result = session.execute(query, {
                'channel_id': channel_id,
                'name': name
            }).fetchone()

            if result:
                return result.host_name

            return name
