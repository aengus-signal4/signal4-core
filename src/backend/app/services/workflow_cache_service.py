"""
Workflow Cache Service
======================

Multi-level caching for RAG workflows. See workflows.py for full architecture docs.

Cache Levels:
-------------

1. **ExpandQueryCache** (via QueryVariationService)
   - Uses normalized query_variations + query_expansions tables
   - Stores: expanded_queries + embeddings (reusable across queries)
   - Time-agnostic (same expansion regardless of time window)
   - Permanent storage (variations grow over time)

2. **WorkflowResultCache** (llm_cache table)
   - Stores: selected_segments, summaries, segment_ids
   - Cache key: query + time_window + projects + languages
   - TTL: 6h (7d queries), 12h (30d), 24h (longer)

3. **LandingPageCache** (llm_cache table)
   - Stores: themes, corpus_stats, summaries (no query)
   - Cache key: time_window + projects + languages
   - TTL: 24h fixed

4. **SemanticQueryCache** (future)
   - Will match semantically similar queries
   - TTL: 1h (7d), 8h (30d)

Performance:
------------
- Full cache hit (quick_summary): ~740ms (instant)
- Partial cache (expand only): ~10s (skip LLM expansion)
- Cold (no cache): ~14s (full pipeline)

Usage:
------
    # Check caches before running pipeline
    cache_results = check_caches_sequential(query, time_window, projects, languages, dashboard_id)

    if cache_results['workflow'] and cache_results['expand_query']:
        # Full cache hit - return immediately for quick_summary
        return cached_result

    if cache_results['expand_query']:
        # Partial hit - use cached embeddings, skip expand_query step
        initial_context = {'query_embeddings': cached_embeddings, ...}

    # After pipeline completes, cache results
    expand_cache.put(query, {'expanded_queries': ..., 'query_embeddings': ...})
    workflow_cache.put(query, time_window, projects, languages, {'selected_segments': ...})
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


class SemanticQueryCache:
    """
    Level 0: Semantic query similarity cache (fastest path).

    Uses query embedding to find semantically similar cached results.
    Returns complete workflow result immediately if found.

    Examples of matches:
    - "What is being said about Mark Carney?" ≈ "Mark Carney discussion"
    - "Climate policy debate" ≈ "What are people saying about climate policy?"

    TTL varies by time window:
    - 7d queries: 1 hour (breaking/current topics change quickly)
    - 30d queries: 8 hours (more stable)
    """

    # Similarity threshold for semantic cache hits (higher = more strict)
    SIMILARITY_THRESHOLD = 0.92  # 92% similar required

    def __init__(self, dashboard_id: str):
        """
        Initialize semantic query cache.

        Args:
            dashboard_id: Dashboard identifier for logging
        """
        self.dashboard_id = dashboard_id
        self.cache = PgLLMCache(
            dashboard_id=dashboard_id,
            similarity_threshold=self.SIMILARITY_THRESHOLD,
            ttl_hours=None  # Dynamic per-request
        )
        logger.debug(f"[{dashboard_id}] SemanticQueryCache initialized (threshold={self.SIMILARITY_THRESHOLD})")

    def get_ttl_for_time_window(self, time_window_days: int) -> int:
        """
        Determine cache TTL based on time window.

        Args:
            time_window_days: Analysis time window in days

        Returns:
            TTL in hours
        """
        if time_window_days <= 7:
            return 1   # 1 hour for 7-day queries (current events)
        elif time_window_days <= 30:
            return 8   # 8 hours for 30-day queries
        else:
            return 24  # 24 hours for historical queries

    def build_cache_key(
        self,
        time_window: int,
        projects: list,
        languages: list
    ) -> str:
        """
        Build cache key from filters (query is matched via embedding).

        Args:
            time_window: Time window in days
            projects: Project filter list
            languages: Language filter list

        Returns:
            Cache key string
        """
        projects_str = ",".join(sorted(projects or []))
        languages_str = ",".join(sorted(languages or []))
        return f"semantic_query:{time_window}d:{projects_str}:{languages_str}"

    def get(
        self,
        query: str,
        query_embedding,
        time_window: int,
        projects: list,
        languages: list
    ):
        """
        Check cache for semantically similar query result.

        Args:
            query: User query string
            query_embedding: Query embedding vector (numpy array)
            time_window: Time window in days
            projects: Project filter list
            languages: Language filter list

        Returns:
            Cached result dict or None if not found
        """
        if query_embedding is None:
            return None

        # Build filter-specific cache key
        cache_key = self.build_cache_key(time_window, projects, languages)
        ttl_hours = self.get_ttl_for_time_window(time_window)

        result = self.cache.get(
            cache_key,
            query_embedding=query_embedding,
            cache_type='semantic_query'
        )

        if result:
            age_seconds = result.get('_cache_age_seconds', 0)
            similarity = result.get('_cache_similarity', 0)
            original_query = result.get('_original_query', 'unknown')

            # Check if within TTL
            if age_seconds / 3600 > ttl_hours:
                logger.debug(
                    f"[{self.dashboard_id}] SemanticQueryCache EXPIRED "
                    f"(age={age_seconds/3600:.1f}h > ttl={ttl_hours}h)"
                )
                return None

            logger.info(
                f"[{self.dashboard_id}] SemanticQueryCache HIT "
                f"(similarity={similarity:.3f}, age={age_seconds/60:.1f}m, ttl={ttl_hours}h) "
                f"'{query[:40]}...' ≈ '{original_query[:40]}...'"
            )

        return result

    def put(
        self,
        query: str,
        query_embedding,
        time_window: int,
        projects: list,
        languages: list,
        data: dict
    ) -> None:
        """
        Cache workflow result with query embedding for semantic matching.

        Args:
            query: User query string
            query_embedding: Query embedding vector
            time_window: Time window in days
            projects: Project filter list
            languages: Language filter list
            data: Result dict with segments, summary, etc.
        """
        if query_embedding is None:
            logger.warning(f"[{self.dashboard_id}] Cannot cache without query embedding")
            return

        cache_key = self.build_cache_key(time_window, projects, languages)
        ttl_hours = self.get_ttl_for_time_window(time_window)

        # Add original query for logging
        data_with_query = {**data, '_original_query': query}

        self.cache.put(
            cache_key,
            query_embedding=query_embedding,
            response=data_with_query,
            cache_type='semantic_query',
            ttl_hours=ttl_hours
        )

        logger.info(
            f"[{self.dashboard_id}] SemanticQueryCache PUT "
            f"(ttl={ttl_hours}h, {time_window}d query) '{query[:50]}...'"
        )


class ExpandQueryCache:
    """
    Level 1: Query expansion cache using normalized variation library.

    Uses query_variations and query_expansions tables to:
    - Deduplicate query texts (same text = same embedding)
    - Reuse embeddings across different original queries
    - Build a growing library of embedded variations over time

    Query expansion is time-agnostic (e.g., "Mark Carney" expands
    the same way regardless of time window).
    """

    def __init__(self, dashboard_id: str):
        """
        Initialize expand query cache.

        Args:
            dashboard_id: Dashboard identifier for logging
        """
        self.dashboard_id = dashboard_id
        from .query_variation_service import QueryVariationService
        self.variation_service = QueryVariationService(dashboard_id)
        logger.debug(f"[{dashboard_id}] ExpandQueryCache initialized (using variation library)")

    def get_sync(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Check cache for query expansion result.

        Looks up the original query in query_expansions table
        and returns all linked variations with their embeddings.

        Args:
            query: User query string

        Returns:
            Cached result dict with 'expanded_queries' and 'query_embeddings',
            or None if not found
        """
        result = self.variation_service.get_expansion(query)

        if result and result.get('all_embedded'):
            # Convert to expected format
            import numpy as np
            return {
                'expanded_queries': result['variations'],
                'query_embeddings': [emb.tolist() for emb in result['embeddings']],
                '_cache_hit': True
            }
        elif result:
            # Found variations but missing some embeddings
            logger.info(
                f"[{self.dashboard_id}] ExpandQueryCache partial HIT "
                f"({len(result['variations'])} variations, needs embedding)"
            )
            return {
                'expanded_queries': result['variations'],
                'query_embeddings': [],  # Need to regenerate
                '_cache_hit': False,
                '_needs_embedding': True
            }

        return None

    def put(self, query: str, data: Dict[str, Any]) -> None:
        """
        Cache query expansion result with embeddings.

        Stores the expansion mapping and all variation embeddings
        in the normalized query_variations library.

        Args:
            query: User query string
            data: Dict with 'expanded_queries' and 'query_embeddings' (list of lists)
        """
        expanded_queries = data.get('expanded_queries', [])
        query_embeddings = data.get('query_embeddings', [])

        if not expanded_queries:
            logger.warning(f"[{self.dashboard_id}] No expanded queries to cache")
            return

        # Convert embeddings from lists to numpy arrays if needed
        import numpy as np
        embeddings = None
        if query_embeddings:
            embeddings = [
                np.array(emb, dtype=np.float32) if not isinstance(emb, np.ndarray) else emb
                for emb in query_embeddings
            ]

        # Store in normalized library
        self.variation_service.store_expansion(
            original_query=query,
            variations=expanded_queries,
            embeddings=embeddings
        )


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

    def _build_cache_query(
        self,
        query: str,
        time_window: int,
        projects: Optional[List[str]],
        languages: Optional[List[str]]
    ) -> str:
        """
        Build the query string used for cache key computation.

        This returns the raw string that will be hashed by PgLLMCache.
        """
        projects_str = ",".join(sorted(projects or []))
        languages_str = ",".join(sorted(languages or []))
        return f"{query}:{time_window}:{projects_str}:{languages_str}"

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
        # Build the full query string for cache lookup
        # PgLLMCache will hash this to create the cache key
        cache_query = self._build_cache_query(query, time_window, projects, languages)
        result = self.cache.get(
            cache_query,
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
        # Build the full query string for cache storage
        # PgLLMCache will hash this to create the cache key
        cache_query = self._build_cache_query(query, time_window, projects, languages)
        ttl_hours = self.get_ttl_for_time_window(time_window)

        self.cache.put(
            cache_query,
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

    def _build_cache_query(
        self,
        time_window: int,
        projects: Optional[List[str]],
        languages: Optional[List[str]]
    ) -> str:
        """
        Build the query string used for cache key computation.

        This returns the raw string that will be hashed by PgLLMCache.
        """
        projects_str = ",".join(sorted(projects or []))
        languages_str = ",".join(sorted(languages or []))
        return f"landing:{time_window}:{projects_str}:{languages_str}"

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
        # Build the full query string for cache lookup
        # PgLLMCache will hash this to create the cache key
        cache_query = self._build_cache_query(time_window, projects, languages)
        result = self.cache.get(
            cache_query,
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
        # Build the full query string for cache storage
        # PgLLMCache will hash this to create the cache key
        cache_query = self._build_cache_query(time_window, projects, languages)

        self.cache.put(
            cache_query,
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
