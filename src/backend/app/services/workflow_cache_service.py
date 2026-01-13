"""
Workflow Cache Service
======================

Two-level caching strategy for RAG workflows:

Level 1: Query Expansion Cache (Long TTL - 30 days)
- Cache Key: query only (time-agnostic)
- Cached Data: expanded_queries, keywords
- Rationale: Query expansion doesn't depend on time window

Level 2: Workflow Result Cache (Dynamic TTL - 6-24 hours)
- Cache Key: query + time_window + filters
- Cached Data: selected_segments, summary, segment_ids
- Rationale: Time-sensitive results, dynamic TTL based on time window
"""

import hashlib
from typing import Optional, List, Dict, Any
from .pg_cache_service import PgLLMCache

import sys
from pathlib import Path
from src.utils.paths import get_project_root, get_config_path
sys.path.insert(0, str(get_project_root()))
from src.utils.logger import setup_worker_logger
logger = setup_worker_logger("backend.workflow_cache")


class ExpandQueryCache:
    """
    Level 1: Query expansion cache with long TTL.

    Caches query expansions independently of time filters since
    query expansion is time-agnostic (e.g., "Mark Carney" expands
    the same way regardless of time window).
    """

    def __init__(self, dashboard_id: str):
        """
        Initialize expand query cache.

        Args:
            dashboard_id: Dashboard identifier for logging
        """
        self.dashboard_id = dashboard_id
        self.cache = PgLLMCache(
            dashboard_id=dashboard_id,
            similarity_threshold=0.85,
            ttl_hours=720  # 30 days
        )
        logger.debug(f"[{dashboard_id}] ExpandQueryCache initialized (ttl=30d)")

    def build_cache_key(self, query: str) -> str:
        """
        Build cache key from query only.

        Args:
            query: User query string

        Returns:
            Cache key string
        """
        query_hash = hashlib.md5(query.encode()).hexdigest()[:12]
        return f"expand_query:{query_hash}"

    def get_sync(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Check cache for query expansion result.

        Args:
            query: User query string

        Returns:
            Cached result dict or None if not found
        """
        cache_key = self.build_cache_key(query)
        result = self.cache.get(
            cache_key,
            query_embedding=None,
            cache_type='expand_query'
        )

        if result:
            age_minutes = result.get('_cache_age_seconds', 0) / 60
            logger.info(f"[{self.dashboard_id}] ExpandQueryCache HIT (age={age_minutes:.1f}m)")

        return result

    def put(self, query: str, data: Dict[str, Any]) -> None:
        """
        Cache query expansion result.

        Args:
            query: User query string
            data: Dict with 'expanded_queries' and 'keywords'
        """
        cache_key = self.build_cache_key(query)
        self.cache.put(
            cache_key,
            query_embedding=None,
            response=data,
            cache_type='expand_query',
            ttl_hours=720  # 30 days
        )
        logger.debug(f"[{self.dashboard_id}] ExpandQueryCache PUT (ttl=30d)")


class WorkflowResultCache:
    """
    Level 2: Complete workflow result cache with dynamic TTL.

    Caches complete analysis results including segments and summary.
    TTL varies based on time window (shorter windows = shorter cache).
    """

    def __init__(self, dashboard_id: str):
        """
        Initialize workflow result cache.

        Args:
            dashboard_id: Dashboard identifier for logging
        """
        self.dashboard_id = dashboard_id
        self.cache = PgLLMCache(
            dashboard_id=dashboard_id,
            similarity_threshold=0.85,
            ttl_hours=None  # Dynamic per-request
        )
        logger.debug(f"[{dashboard_id}] WorkflowResultCache initialized")

    def build_cache_key(
        self,
        query: str,
        time_window: int,
        projects: Optional[List[str]],
        languages: Optional[List[str]]
    ) -> str:
        """
        Build cache key from query + time_window + filters.

        Args:
            query: User query string
            time_window: Time window in days
            projects: Project filter list
            languages: Language filter list

        Returns:
            Cache key string
        """
        # Sort lists for consistent hashing
        projects_str = ",".join(sorted(projects or []))
        languages_str = ",".join(sorted(languages or []))

        key_data = f"{query}:{time_window}:{projects_str}:{languages_str}"
        cache_hash = hashlib.md5(key_data.encode()).hexdigest()[:12]
        return f"workflow_result:{cache_hash}"

    def get_ttl_for_time_window(self, time_window_days: int) -> int:
        """
        Determine cache TTL based on time window.

        Shorter windows = more dynamic data = shorter cache
        Longer windows = more stable data = longer cache

        Args:
            time_window_days: Analysis time window in days

        Returns:
            TTL in hours
        """
        if time_window_days <= 7:
            return 6   # 6 hours for 7-day queries (breaking topics)
        elif time_window_days <= 30:
            return 12  # 12 hours for 30-day queries (balanced)
        else:
            return 24  # 24 hours for longer queries (historical)

    def get_sync(
        self,
        query: str,
        time_window: int,
        projects: Optional[List[str]],
        languages: Optional[List[str]]
    ) -> Optional[Dict[str, Any]]:
        """
        Check cache for complete workflow result.

        Args:
            query: User query string
            time_window: Time window in days
            projects: Project filter list
            languages: Language filter list

        Returns:
            Cached result dict or None if not found
        """
        cache_key = self.build_cache_key(query, time_window, projects, languages)
        result = self.cache.get(
            cache_key,
            query_embedding=None,
            cache_type='workflow_result'
        )

        if result:
            age_minutes = result.get('_cache_age_seconds', 0) / 60
            ttl_hours = self.get_ttl_for_time_window(time_window)
            logger.info(
                f"[{self.dashboard_id}] WorkflowResultCache HIT "
                f"(age={age_minutes:.1f}m, ttl={ttl_hours}h)"
            )

        return result

    def put(
        self,
        query: str,
        time_window: int,
        projects: Optional[List[str]],
        languages: Optional[List[str]],
        data: Dict[str, Any]
    ) -> None:
        """
        Cache complete workflow result with dynamic TTL.

        Args:
            query: User query string
            time_window: Time window in days
            projects: Project filter list
            languages: Language filter list
            data: Result dict with 'selected_segments', 'summary', 'segment_ids'
        """
        cache_key = self.build_cache_key(query, time_window, projects, languages)
        ttl_hours = self.get_ttl_for_time_window(time_window)

        self.cache.put(
            cache_key,
            query_embedding=None,
            response=data,
            cache_type='workflow_result',
            ttl_hours=ttl_hours
        )

        logger.info(
            f"[{self.dashboard_id}] WorkflowResultCache PUT (ttl={ttl_hours}h, {time_window}d query)"
        )


class LandingPageCache:
    """
    Landing page corpus analysis cache with fixed 24-hour TTL.

    Caches complete landing page results (no query, just filters).
    Cache key: time_window + projects + languages
    """

    TTL_HOURS = 24  # 24-hour TTL for landing page cache

    def __init__(self, dashboard_id: str):
        """
        Initialize landing page cache.

        Args:
            dashboard_id: Dashboard identifier for logging
        """
        self.dashboard_id = dashboard_id
        self.cache = PgLLMCache(
            dashboard_id=dashboard_id,
            similarity_threshold=0.85,
            ttl_hours=self.TTL_HOURS
        )
        logger.debug(f"[{dashboard_id}] LandingPageCache initialized (ttl={self.TTL_HOURS}h)")

    def build_cache_key(
        self,
        time_window: int,
        projects: Optional[List[str]],
        languages: Optional[List[str]]
    ) -> str:
        """
        Build cache key from filters only (no query).

        Args:
            time_window: Time window in days
            projects: Project filter list
            languages: Language filter list

        Returns:
            Cache key string
        """
        # Sort lists for consistent hashing
        projects_str = ",".join(sorted(projects or []))
        languages_str = ",".join(sorted(languages or []))

        key_data = f"landing:{time_window}:{projects_str}:{languages_str}"
        cache_hash = hashlib.md5(key_data.encode()).hexdigest()[:12]
        return f"landing_page:{cache_hash}"

    def get_sync(
        self,
        time_window: int,
        projects: Optional[List[str]],
        languages: Optional[List[str]]
    ) -> Optional[Dict[str, Any]]:
        """
        Check cache for landing page result.

        Args:
            time_window: Time window in days
            projects: Project filter list
            languages: Language filter list

        Returns:
            Cached result dict or None if not found
        """
        cache_key = self.build_cache_key(time_window, projects, languages)
        result = self.cache.get(
            cache_key,
            query_embedding=None,
            cache_type='landing_page'
        )

        if result:
            age_seconds = result.get('_cache_age_seconds', 0)
            age_minutes = age_seconds / 60
            age_hours = age_seconds / 3600
            logger.info(
                f"[{self.dashboard_id}] LandingPageCache HIT "
                f"(age={age_hours:.1f}h, ttl={self.TTL_HOURS}h)"
            )
            # Add age to result for frontend display
            result['_cache_age_hours'] = age_hours

        return result

    def put(
        self,
        time_window: int,
        projects: Optional[List[str]],
        languages: Optional[List[str]],
        data: Dict[str, Any]
    ) -> None:
        """
        Cache landing page result with 20-hour TTL.

        Args:
            time_window: Time window in days
            projects: Project filter list
            languages: Language filter list
            data: Result dict with themes, corpus_stats, summaries, etc.
        """
        cache_key = self.build_cache_key(time_window, projects, languages)

        self.cache.put(
            cache_key,
            query_embedding=None,
            response=data,
            cache_type='landing_page',
            ttl_hours=self.TTL_HOURS
        )

        logger.info(
            f"[{self.dashboard_id}] LandingPageCache PUT (ttl={self.TTL_HOURS}h, {time_window}d window)"
        )


def check_caches_sequential(
    query: str,
    time_window_days: int,
    projects: Optional[List[str]],
    languages: Optional[List[str]],
    dashboard_id: str
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Check both cache levels sequentially.

    Note: Sequential (not parallel) to avoid thread-safety issues with
    embedding models and database connections. The performance difference
    is negligible (~10ms) compared to the time saved by cache hits (~15s).

    Args:
        query: User query string
        time_window_days: Time window in days
        projects: Project filter list
        languages: Language filter list
        dashboard_id: Dashboard identifier

    Returns:
        Dict with 'expand_query' and 'workflow' cache results
    """
    expand_cache = ExpandQueryCache(dashboard_id)
    workflow_cache = WorkflowResultCache(dashboard_id)

    # Check caches sequentially (safe for psycopg2 + embedding models)
    expand_result = expand_cache.get_sync(query)
    workflow_result = workflow_cache.get_sync(
        query,
        time_window_days,
        projects,
        languages
    )

    return {
        'expand_query': expand_result,
        'workflow': workflow_result
    }
