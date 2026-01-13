"""
PostgreSQL LLM Response Cache
==============================

Replaces pickle-based LLM cache with PostgreSQL table for better:
- Concurrent access (no file locking issues)
- Semantic similarity lookup via pgvector
- Automatic cleanup via TTL
- Query-able cache statistics
- Standard backup/replication

Cache types supported:
- optimize_query: Query optimization/expansion results
- theme_summary: Theme summarization results
- search_results: Search result caching
"""

import logging
import time
import json
import hashlib
import psycopg2
from typing import Optional, Dict, Any
from datetime import datetime

# Setup logger
import sys
from pathlib import Path
from src.utils.paths import get_project_root, get_config_path
sys.path.insert(0, str(get_project_root()))
from src.utils.logger import setup_worker_logger
logger = setup_worker_logger("backend.pg_cache")


class PgLLMCache:
    """PostgreSQL-based LLM response cache with semantic similarity matching"""

    def __init__(self, dashboard_id: str, similarity_threshold: float = 0.85, ttl_hours: int = None):
        """
        Initialize PostgreSQL LLM cache

        Args:
            dashboard_id: Dashboard identifier
            similarity_threshold: Minimum cosine similarity for cache hit (default 0.85)
            ttl_hours: Time-to-live in hours (None = permanent cache)
        """
        self.dashboard_id = dashboard_id
        self.similarity_threshold = similarity_threshold
        self.ttl_hours = ttl_hours

        # PostgreSQL connection parameters
        self.db_config = {
            'host': '10.0.0.4',
            'database': 'av_content',
            'user': 'signal4',
            'password': 'signal4'
        }

        ttl_str = f"{ttl_hours}h" if ttl_hours is not None else "permanent"
        logger.info(f"PgLLMCache initialized: dashboard={dashboard_id}, threshold={similarity_threshold}, ttl={ttl_str}")

    def _get_connection(self):
        """Get PostgreSQL connection"""
        return psycopg2.connect(**self.db_config)

    def _compute_cache_key(self, query: str, cache_type: str, embedding_dim: Optional[int] = None) -> str:
        """
        Compute cache key from query text

        Args:
            query: Query string
            cache_type: Type of cache
            embedding_dim: Embedding dimension (to separate different model caches)

        Returns:
            Cache key string
        """
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]

        if embedding_dim is not None:
            return f"{cache_type}_{embedding_dim}d_{query_hash}"
        else:
            return f"{cache_type}_{query_hash}"

    def get(self, query: str, query_embedding: Optional[Any], cache_type: str = "rag_summary") -> Optional[Any]:
        """
        Get cached response - exact match or semantic similarity

        Args:
            query: Query string
            query_embedding: Query embedding vector (None for text-based lookup)
            cache_type: Type of cache ('rag_summary', 'optimize_query', etc.)

        Returns:
            Cached response (dict, list, or any type) or None if no match
        """
        start_time = time.time()

        # Compute cache key
        embedding_dim = query_embedding.shape[0] if query_embedding is not None else None
        cache_key = self._compute_cache_key(query, cache_type, embedding_dim)

        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # Try exact cache_key match first (fastest path)
            cursor.execute("""
                SELECT response, created_at, access_count
                FROM llm_cache
                WHERE cache_key = %s AND cache_type = %s
                  AND (ttl_hours IS NULL OR created_at > NOW() - (ttl_hours || ' hours')::INTERVAL)
            """, (cache_key, cache_type))

            result = cursor.fetchone()

            if result:
                response, created_at, access_count = result
                age_seconds = (datetime.now() - created_at).total_seconds()
                age_minutes = age_seconds / 60

                # Update access statistics
                cursor.execute("""
                    UPDATE llm_cache
                    SET accessed_at = NOW(), access_count = access_count + 1
                    WHERE cache_key = %s AND cache_type = %s
                """, (cache_key, cache_type))
                conn.commit()

                logger.info(
                    f"[{cache_type}] Cache HIT (exact): age={age_minutes:.1f}m, "
                    f"accesses={access_count+1}, query='{query[:50]}...'"
                )

                # Add cache metadata if response is dict
                if isinstance(response, dict):
                    response['_cache_hit'] = True
                    response['_cache_age_seconds'] = age_seconds
                    return response
                else:
                    return response

            # Exact match not found - try semantic similarity if embedding provided
            if query_embedding is not None:
                # Convert embedding to PostgreSQL format
                embedding_list = query_embedding.flatten().tolist()
                embedding_str = '[' + ','.join(map(str, embedding_list)) + ']'

                # Set IVFFlat probes
                cursor.execute("SET ivfflat.probes = 5;")

                # Search for similar cached entries
                cursor.execute("""
                    SELECT
                        cache_key,
                        response,
                        created_at,
                        access_count,
                        1 - (query_embedding <=> %s::vector) as similarity
                    FROM llm_cache
                    WHERE cache_type = %s
                      AND query_embedding IS NOT NULL
                      AND (ttl_hours IS NULL OR created_at > NOW() - (ttl_hours || ' hours')::INTERVAL)
                      AND (query_embedding <=> %s::vector) < %s
                    ORDER BY query_embedding <=> %s::vector
                    LIMIT 1;
                """, (embedding_str, cache_type, embedding_str, 1.0 - self.similarity_threshold, embedding_str))

                result = cursor.fetchone()

                if result:
                    matched_key, response, created_at, access_count, similarity = result
                    age_seconds = (datetime.now() - created_at).total_seconds()
                    age_minutes = age_seconds / 60

                    # Update access statistics
                    cursor.execute("""
                        UPDATE llm_cache
                        SET accessed_at = NOW(), access_count = access_count + 1
                        WHERE cache_key = %s AND cache_type = %s
                    """, (matched_key, cache_type))
                    conn.commit()

                    logger.info(
                        f"[{cache_type}] Cache HIT (semantic): similarity={similarity:.3f}, "
                        f"age={age_minutes:.1f}m, accesses={access_count+1}, query='{query[:50]}...'"
                    )

                    # Add cache metadata
                    if isinstance(response, dict):
                        response['_cache_hit'] = True
                        response['_cache_similarity'] = float(similarity)
                        response['_cache_age_seconds'] = age_seconds
                        return response
                    else:
                        return response

            # No match found
            lookup_time = (time.time() - start_time) * 1000
            logger.debug(f"[{cache_type}] Cache MISS: lookup={lookup_time:.0f}ms, query='{query[:50]}...'")
            cursor.close()
            return None

        finally:
            conn.close()

    def put(self, query: str, query_embedding: Optional[Any], response: Any, cache_type: str = "rag_summary", ttl_hours: Optional[int] = None):
        """
        Cache an LLM response

        Args:
            query: Query string
            query_embedding: Query embedding vector (None for text-based caching)
            response: Response from LLM (can be dict, list, or any type)
            cache_type: Type of cache ('rag_summary', 'optimize_query', etc.)
            ttl_hours: Time-to-live in hours (overrides instance default if provided)
        """
        start_time = time.time()

        # Use provided TTL or fall back to instance default
        effective_ttl = ttl_hours if ttl_hours is not None else self.ttl_hours

        # Compute cache key
        embedding_dim = query_embedding.shape[0] if query_embedding is not None else None
        cache_key = self._compute_cache_key(query, cache_type, embedding_dim)

        # Convert embedding to PostgreSQL format if provided
        embedding_str = None
        if query_embedding is not None:
            embedding_list = query_embedding.flatten().tolist()
            embedding_str = '[' + ','.join(map(str, embedding_list)) + ']'

        # Serialize response to JSON
        response_json = json.dumps(response)

        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # Upsert (insert or update on conflict)
            cursor.execute("""
                INSERT INTO llm_cache
                  (cache_key, cache_type, query_text, query_embedding, response, dashboard_id, ttl_hours)
                VALUES (%s, %s, %s, %s::vector, %s::jsonb, %s, %s)
                ON CONFLICT (cache_key, cache_type)
                DO UPDATE SET
                  response = EXCLUDED.response,
                  accessed_at = NOW(),
                  access_count = llm_cache.access_count + 1
            """, (cache_key, cache_type, query, embedding_str, response_json, self.dashboard_id, effective_ttl))

            conn.commit()
            cursor.close()

            put_time = (time.time() - start_time) * 1000
            ttl_str = f"{effective_ttl}h" if effective_ttl is not None else "permanent"
            logger.debug(f"[{cache_type}] Cached response: key={cache_key}, ttl={ttl_str}, time={put_time:.0f}ms, query='{query[:50]}...'")

        finally:
            conn.close()

    def cleanup_expired(self):
        """Remove expired entries from cache (only if TTL is enabled)"""
        if self.ttl_hours is None:
            logger.debug("Cache is permanent, skipping cleanup")
            return

        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # Delete expired entries
            cursor.execute("""
                DELETE FROM llm_cache
                WHERE ttl_hours IS NOT NULL
                  AND created_at < NOW() - (ttl_hours || ' hours')::INTERVAL
            """)

            deleted_count = cursor.rowcount
            conn.commit()
            cursor.close()

            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} expired cache entries")

        finally:
            conn.close()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # Get stats
            cursor.execute("""
                SELECT
                    COUNT(*) as total_entries,
                    COUNT(*) FILTER (WHERE ttl_hours IS NULL OR created_at > NOW() - (ttl_hours || ' hours')::INTERVAL) as valid_entries,
                    COUNT(*) FILTER (WHERE ttl_hours IS NOT NULL AND created_at <= NOW() - (ttl_hours || ' hours')::INTERVAL) as expired_entries,
                    AVG(access_count) as avg_accesses,
                    MAX(accessed_at) as last_access
                FROM llm_cache
                WHERE dashboard_id = %s
            """, (self.dashboard_id,))

            result = cursor.fetchone()
            cursor.close()

            if result:
                return {
                    'total_entries': result[0],
                    'valid_entries': result[1],
                    'expired_entries': result[2],
                    'avg_accesses': float(result[3]) if result[3] else 0,
                    'last_access': result[4],
                    'similarity_threshold': self.similarity_threshold,
                    'ttl_hours': self.ttl_hours,
                    'permanent': self.ttl_hours is None
                }
            else:
                return {
                    'total_entries': 0,
                    'valid_entries': 0,
                    'expired_entries': 0,
                    'avg_accesses': 0,
                    'last_access': None,
                    'similarity_threshold': self.similarity_threshold,
                    'ttl_hours': self.ttl_hours,
                    'permanent': self.ttl_hours is None
                }

        finally:
            conn.close()

    def clear(self):
        """Clear all cache entries for this dashboard"""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            cursor.execute("""
                DELETE FROM llm_cache WHERE dashboard_id = %s
            """, (self.dashboard_id,))

            deleted_count = cursor.rowcount
            conn.commit()
            cursor.close()

            logger.info(f"Cleared {deleted_count} cache entries for dashboard {self.dashboard_id}")

        finally:
            conn.close()
