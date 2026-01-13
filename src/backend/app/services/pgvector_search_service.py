"""
PostgreSQL pgvector Search Service
===================================

PostgreSQL pgvector-based semantic search using cache tables.
Uses HNSW indexes for fast similarity search on cached embeddings.

Cache tables (each has both embedding columns):
- embedding_cache_7d: Hot cache (hourly refresh) for recent content (≤7 days)
- embedding_cache_30d: Main cache (6-hour refresh, incremental updates) for all queries (8+ days)

Embedding columns:
- embedding: vector(1024) - 0.6B model embeddings
- embedding_alt: vector(2000) - 4B model embeddings (higher quality, truncated from 2560)

HNSW Index Settings:
- m=16, ef_construction=64 (build-time parameters)
- hnsw.ef_search=100 (query-time parameter, set per-connection)

Refresh Strategy (incremental, not truncate+rebuild):
- DELETE aged rows (publish_date outside window)
- INSERT new rows from content.last_updated >= 12 hours
- ON CONFLICT (id) DO NOTHING
- Typical refresh: ~50ms (no changes) to ~3-4min (with new content)
"""

import logging
import time
import psycopg2
from psycopg2.extras import execute_values, RealDictCursor
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

# Setup logger
import sys
from pathlib import Path
from src.utils.paths import get_project_root, get_config_path
sys.path.insert(0, str(get_project_root()))
from src.utils.logger import setup_worker_logger
logger = setup_worker_logger("backend.pgvector_search")


class PgVectorSearchService:
    """PostgreSQL pgvector-based semantic search service"""

    def __init__(self, dashboard_id: str, config):
        """
        Initialize pgvector search service

        Args:
            dashboard_id: Dashboard identifier
            config: DashboardConfig object
        """
        self.dashboard_id = dashboard_id
        self.config = config

        # PostgreSQL connection parameters
        self.db_config = {
            'host': '10.0.0.4',
            'database': 'av_content',
            'user': 'signal4',
            'password': 'signal4'
        }

        logger.info(f"PgVectorSearchService initialized for {dashboard_id} (project: {config.project})")

    def _get_connection(self):
        """Get PostgreSQL connection"""
        return psycopg2.connect(**self.db_config)

    def _select_view(self, time_window_days: int) -> str:
        """
        Select appropriate cache table based on time window

        Args:
            time_window_days: Time window in days

        Returns:
            Table name to query

        Selection strategy:
        - ≤7 days: Use 7d cache (hourly refresh, freshest data)
        - 8+ days: Use 30d cache (6-hour refresh, incremental updates)

        Note: 180d cache was removed - 30d cache now covers all longer windows.
        For queries beyond 30 days, the date filter in the WHERE clause handles it.
        """
        if time_window_days <= 7:
            return 'embedding_cache_7d'
        else:
            return 'embedding_cache_30d'

    def _get_embedding_column(self, use_alt_model: bool = False) -> str:
        """
        Get the embedding column name based on model type

        Args:
            use_alt_model: Whether to use alt embeddings (4B model)

        Returns:
            Column name: 'embedding' (1024-dim, 0.6B) or 'embedding_alt' (2000-dim, 4B)
        """
        return 'embedding_alt' if use_alt_model else 'embedding'

    def _build_where_clause(self, filters: Dict[str, Any], time_window_days: int = None) -> tuple:
        """
        Build WHERE clause and parameters from filters

        Args:
            filters: Dictionary of filter conditions
            time_window_days: Time window in days (adds date filter automatically)

        Returns:
            (where_clause_string, parameters_list)
        """
        conditions = []
        params = []

        # Project filter (required)
        if self.config.allowed_projects:
            # Projects is an array column (varchar[]), use && (array overlap) operator
            conditions.append("projects && %s::varchar[]")
            params.append(self.config.allowed_projects)

        # Automatic date filter based on time window (if not using 7-day hot cache)
        if time_window_days and time_window_days > 7 and 'date_from' not in filters:
            conditions.append("publish_date >= NOW() - INTERVAL '%s days'")
            params.append(time_window_days)

        # Date range filter (for custom time windows)
        if 'date_from' in filters:
            conditions.append("publish_date >= %s")
            params.append(filters['date_from'])

        if 'date_to' in filters:
            conditions.append("publish_date <= %s")
            params.append(filters['date_to'])

        # Content ID filter
        if 'filter_content_ids' in filters and filters['filter_content_ids']:
            conditions.append(f"content_id = ANY(%s::int[])")
            params.append(filters['filter_content_ids'])

        # Speaker filter (check array overlap)
        if 'filter_speakers' in filters and filters['filter_speakers']:
            conditions.append(f"source_speaker_hashes && %s::text[]")
            params.append(filters['filter_speakers'])

        # Stitch version filter
        if 'filter_stitch_versions' in filters and filters['filter_stitch_versions']:
            conditions.append(f"stitch_version = ANY(%s::text[])")
            params.append(filters['filter_stitch_versions'])

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        return where_clause, params

    def search(self, query_embedding, time_window_days: int = 7,
               k: int = 200, threshold: float = 0.43,
               use_alt_model: bool = False,
               **filters) -> List[Dict]:
        """
        Semantic search using pgvector

        Args:
            query_embedding: Query embedding vector (numpy array)
            time_window_days: Time window in days (7, 30, 180)
            k: Number of results
            threshold: Similarity threshold (0.0-1.0)
            use_alt_model: Use alt embedding model
            **filters: Additional filters (filter_speakers, filter_content_ids, etc.)

        Returns:
            List of result dictionaries with segment info and similarity scores
        """
        start_time = time.time()

        # Select table and embedding column
        view_name = self._select_view(time_window_days)
        use_alt = use_alt_model or self.config.use_alt_embeddings
        emb_col = self._get_embedding_column(use_alt)

        # Build WHERE clause with automatic date filtering
        where_clause, params = self._build_where_clause(filters, time_window_days)

        # Convert embedding to PostgreSQL format
        embedding_list = query_embedding.flatten().tolist()
        embedding_str = '[' + ','.join(map(str, embedding_list)) + ']'

        # Calculate max distance from similarity threshold
        # pgvector cosine distance: 0 = identical, 2 = opposite
        # Our similarity: 1.0 = identical, 0.0 = orthogonal
        # Conversion: distance = 1 - similarity, so max_distance = 1 - threshold
        max_distance = 1.0 - threshold

        # Build query - use dynamic embedding column
        query = f"""
        SELECT
            id as segment_id,
            1 - ({emb_col} <=> %s::vector) as similarity,
            content_id,
            content_id_string,
            channel_name,
            title,
            publish_date,
            text,
            start_time,
            end_time,
            source_speaker_hashes as speaker_hashes,
            segment_index,
            stitch_version
        FROM {view_name}
        WHERE {where_clause}
          AND ({emb_col} <=> %s::vector) < %s
        ORDER BY {emb_col} <=> %s::vector
        LIMIT %s;
        """

        # Execute query
        conn = self._get_connection()
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Set HNSW ef_search (affects speed/accuracy tradeoff)
            # Higher values = better recall but slower queries
            cursor.execute("SET hnsw.ef_search = 100;")

            # Execute search
            search_start = time.time()
            query_params = [embedding_str] + params + [embedding_str, max_distance, embedding_str, k]
            cursor.execute(query, query_params)
            results = cursor.fetchall()
            search_time = (time.time() - search_start) * 1000

            # Format results
            formatted = []
            for row in results:
                # Convert publish_date to ISO string if datetime
                publish_date = row['publish_date']
                if publish_date and hasattr(publish_date, 'isoformat'):
                    publish_date = publish_date.isoformat()

                formatted.append({
                    'segment_id': row['segment_id'],
                    'similarity': float(row['similarity']),
                    'content_id': row['content_id'],
                    'content_id_string': row['content_id_string'],
                    'channel_url': None,  # Not available in materialized views
                    'channel_name': row['channel_name'],
                    'title': row['title'],
                    'publish_date': publish_date,
                    'text': row['text'],
                    'start_time': float(row['start_time']),
                    'end_time': float(row['end_time']),
                    'speaker_hashes': row['speaker_hashes'] or [],
                    'segment_index': row['segment_index'],
                    'stitch_version': row['stitch_version']
                })

            cursor.close()

            total_time = (time.time() - start_time) * 1000
            logger.info(
                f"[{self.dashboard_id}] pgvector search: {len(formatted)} results in {total_time:.0f}ms "
                f"(query={search_time:.0f}ms, table={view_name}, col={emb_col}, threshold={threshold})"
            )

            return formatted

        finally:
            conn.close()

    def batch_search(self, query_embeddings: List, time_window_days: int = 7,
                     k: int = 200, threshold: float = 0.43,
                     use_alt_model: bool = False,
                     **filters) -> List[List[Dict]]:
        """
        Batch semantic search (multiple queries in parallel)

        Args:
            query_embeddings: List of query embedding vectors
            time_window_days: Time window in days
            k: Number of results per query
            threshold: Similarity threshold
            use_alt_model: Use alt embedding model
            **filters: Additional filters

        Returns:
            List of result lists (one per query)
        """
        start_time = time.time()

        if not query_embeddings:
            return []

        # Select table and embedding column
        view_name = self._select_view(time_window_days)
        use_alt = use_alt_model or self.config.use_alt_embeddings
        emb_col = self._get_embedding_column(use_alt)

        # Build WHERE clause with automatic date filtering
        where_clause, params = self._build_where_clause(filters, time_window_days)

        # Convert threshold
        max_distance = 1.0 - threshold

        logger.info(f"[{self.dashboard_id}] batch_search: table={view_name}, col={emb_col}, where_clause='{where_clause}', params={params}, max_distance={max_distance}, k={k}, num_queries={len(query_embeddings)}")

        # Build batch query
        conn = self._get_connection()
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Set HNSW ef_search for better recall
            cursor.execute("SET hnsw.ef_search = 100;")

            all_results = []

            # Execute each query - use dynamic embedding column
            query_template = f"""
            SELECT
                id as segment_id,
                1 - ({emb_col} <=> %s::vector) as similarity,
                content_id,
                content_id_string,
                channel_name,
                title,
                publish_date,
                text,
                start_time,
                end_time,
                source_speaker_hashes as speaker_hashes,
                segment_index,
                stitch_version,
                {emb_col} as embedding
            FROM {view_name}
            WHERE {where_clause}
              AND ({emb_col} <=> %s::vector) < %s
            ORDER BY {emb_col} <=> %s::vector
            LIMIT %s;
            """

            batch_start = time.time()
            for i, emb in enumerate(query_embeddings):
                embedding_list = emb.flatten().tolist()
                embedding_str = '[' + ','.join(map(str, embedding_list)) + ']'

                if i == 0:
                    logger.info(f"[{self.dashboard_id}] First query embedding: dim={len(embedding_list)}, first_vals={embedding_list[:5]}")

                query_params = [embedding_str] + params + [embedding_str, max_distance, embedding_str, k]
                cursor.execute(query_template, query_params)
                results = cursor.fetchall()

                logger.debug(f"[{self.dashboard_id}]   Query {i+1} returned {len(results)} raw results")

                # Format results for this query
                formatted = []
                for row in results:
                    publish_date = row['publish_date']
                    if publish_date and hasattr(publish_date, 'isoformat'):
                        publish_date = publish_date.isoformat()

                    # Parse embedding from pgvector string format to numpy array
                    # Store as _embedding (transient, not for caching/frontend)
                    embedding = None
                    if row.get('embedding'):
                        import numpy as np
                        # pgvector returns embedding as string like "[0.1,0.2,...]"
                        emb_str = row['embedding']
                        if isinstance(emb_str, str):
                            emb_str = emb_str.strip('[]')
                            embedding = np.array([float(x) for x in emb_str.split(',')])
                        else:
                            embedding = np.array(emb_str)

                    formatted.append({
                        'segment_id': row['segment_id'],
                        'similarity': float(row['similarity']),
                        'content_id': row['content_id'],
                        'content_id_string': row['content_id_string'],
                        'channel_url': None,  # Not available in materialized views
                        'channel_name': row['channel_name'],
                        'title': row['title'],
                        'publish_date': publish_date,
                        'text': row['text'],
                        'start_time': float(row['start_time']),
                        'end_time': float(row['end_time']),
                        'speaker_hashes': row['speaker_hashes'] or [],
                        'segment_index': row['segment_index'],
                        'stitch_version': row['stitch_version'],
                        '_embedding': embedding  # Transient: NOT for caching/frontend, stripped before response
                    })

                all_results.append(formatted)

            batch_time = (time.time() - batch_start) * 1000
            cursor.close()

            total_time = (time.time() - start_time) * 1000
            total_results = sum(len(r) for r in all_results)
            logger.info(
                f"[{self.dashboard_id}] pgvector batch_search: {total_results} results from {len(query_embeddings)} queries "
                f"in {total_time:.0f}ms (queries={batch_time:.0f}ms, table={view_name}, col={emb_col})"
            )

            return all_results

        finally:
            conn.close()

    def batch_search_unified(self, query_embeddings: List, time_window_days: int = 7,
                             k: int = None, threshold: float = 0.43,
                             use_alt_model: bool = False,
                             **filters) -> List[Dict]:
        """
        Unified batch search - single query for multiple embeddings with deduplication.

        Returns ALL segments that match ANY of the query embeddings within threshold.
        Each segment appears only once with its best (highest) similarity score.

        This is more efficient than batch_search when you want a combined result set
        because it:
        1. Executes a single database query instead of N queries
        2. Automatically deduplicates segments that match multiple queries
        3. Reduces network round-trips

        Args:
            query_embeddings: List of query embedding vectors
            time_window_days: Time window in days
            k: Optional max results (safety cap, default None = no limit)
            threshold: Similarity threshold (segment must match at least one query above this)
            use_alt_model: Use alt embedding model
            **filters: Additional filters

        Returns:
            Single list of unique segments above threshold, sorted by best similarity
        """
        import numpy as np
        start_time = time.time()

        if not query_embeddings:
            return []

        # Select table and embedding column
        view_name = self._select_view(time_window_days)
        use_alt = use_alt_model or self.config.use_alt_embeddings
        emb_col = self._get_embedding_column(use_alt)

        # Build WHERE clause with automatic date filtering
        where_clause, params = self._build_where_clause(filters, time_window_days)

        # Convert threshold to max distance
        max_distance = 1.0 - threshold

        num_queries = len(query_embeddings)
        logger.info(f"[{self.dashboard_id}] batch_search_unified: table={view_name}, col={emb_col}, "
                   f"num_queries={num_queries}, threshold={threshold}, k={k or 'unlimited'}")

        # Convert all embeddings to pgvector string format
        embedding_strs = []
        for emb in query_embeddings:
            embedding_list = emb.flatten().tolist()
            embedding_str = '[' + ','.join(map(str, embedding_list)) + ']'
            embedding_strs.append(embedding_str)

        conn = self._get_connection()
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("SET hnsw.ef_search = 100;")

            # Build UNION ALL query - one subquery per embedding, all within threshold
            # Then deduplicate by segment_id keeping the best similarity
            union_parts = []
            all_params = []

            for i, emb_str in enumerate(embedding_strs):
                # Each subquery finds ALL matches within threshold for one embedding
                # No LIMIT here - we want all segments above similarity threshold
                union_parts.append(f"""(
                    SELECT
                        id as segment_id,
                        1 - ({emb_col} <=> %s::vector) as similarity,
                        content_id,
                        content_id_string,
                        channel_name,
                        title,
                        publish_date,
                        text,
                        start_time,
                        end_time,
                        source_speaker_hashes as speaker_hashes,
                        segment_index,
                        stitch_version,
                        {emb_col} as embedding,
                        {i} as query_idx
                    FROM {view_name}
                    WHERE {where_clause}
                      AND ({emb_col} <=> %s::vector) < %s
                )""")
                # Parameters: embedding for similarity calc, where_clause params,
                # embedding for distance filter, max_distance
                all_params.extend([emb_str] + params + [emb_str, max_distance])

            # Combine with UNION ALL, then deduplicate keeping best similarity
            union_query = " UNION ALL ".join(union_parts)

            full_query = f"""
                SELECT DISTINCT ON (segment_id)
                    segment_id,
                    similarity,
                    content_id,
                    content_id_string,
                    channel_name,
                    title,
                    publish_date,
                    text,
                    start_time,
                    end_time,
                    speaker_hashes,
                    segment_index,
                    stitch_version,
                    embedding
                FROM ({union_query}) combined
                ORDER BY segment_id, similarity DESC
            """

            # Wrap to get final ordering by similarity, with optional limit
            if k:
                final_query = f"""
                    SELECT * FROM ({full_query}) deduped
                    ORDER BY similarity DESC
                    LIMIT %s
                """
                all_params.append(k)
            else:
                final_query = f"""
                    SELECT * FROM ({full_query}) deduped
                    ORDER BY similarity DESC
                """

            query_start = time.time()
            cursor.execute(final_query, all_params)
            results = cursor.fetchall()
            query_time = (time.time() - query_start) * 1000

            cursor.close()

            # Format results
            formatted = []
            for row in results:
                publish_date = row['publish_date']
                if publish_date and hasattr(publish_date, 'isoformat'):
                    publish_date = publish_date.isoformat()

                # Parse embedding
                embedding = None
                if row.get('embedding'):
                    emb_str = row['embedding']
                    if isinstance(emb_str, str):
                        emb_str = emb_str.strip('[]')
                        embedding = np.array([float(x) for x in emb_str.split(',')])
                    else:
                        embedding = np.array(emb_str)

                formatted.append({
                    'segment_id': row['segment_id'],
                    'similarity': float(row['similarity']),
                    'content_id': row['content_id'],
                    'content_id_string': row['content_id_string'],
                    'channel_url': None,
                    'channel_name': row['channel_name'],
                    'title': row['title'],
                    'publish_date': publish_date,
                    'text': row['text'],
                    'start_time': float(row['start_time']),
                    'end_time': float(row['end_time']),
                    'speaker_hashes': row['speaker_hashes'] or [],
                    'segment_index': row['segment_index'],
                    'stitch_version': row['stitch_version'],
                    '_embedding': embedding
                })

            total_time = (time.time() - start_time) * 1000
            logger.info(
                f"[{self.dashboard_id}] pgvector batch_search_unified: {len(formatted)} unique results "
                f"from {num_queries} queries in {total_time:.0f}ms (query={query_time:.0f}ms, table={view_name})"
            )

            return formatted

        finally:
            conn.close()

    def keyword_search(self, keywords: List[str], time_window_days: int = 7,
                       limit: int = 200, query_embedding=None,
                       similarity_threshold: float = 0.40,
                       use_alt_model: bool = False) -> List[Dict]:
        """
        Keyword search using PostgreSQL text matching

        Args:
            keywords: List of keywords/phrases to search
            time_window_days: Time window
            limit: Max results
            query_embedding: Optional embedding for re-ranking
            similarity_threshold: Min similarity if re-ranking
            use_alt_model: Use alt embedding model

        Returns:
            List of matching segments
        """
        # Select table and embedding column
        view_name = self._select_view(time_window_days)
        use_alt = use_alt_model or self.config.use_alt_embeddings
        emb_col = self._get_embedding_column(use_alt)

        # Build project filter with date filtering
        where_clause, params = self._build_where_clause({}, time_window_days)

        # Build keyword conditions (case-insensitive LIKE)
        keyword_conditions = " OR ".join(["text ILIKE %s"] * len(keywords))
        keyword_params = [f'%{kw}%' for kw in keywords]

        query = f"""
        SELECT
            id as segment_id,
            NULL as similarity,
            content_id,
            content_id_string,
            channel_name,
            title,
            publish_date,
            text,
            start_time,
            end_time,
            source_speaker_hashes as speaker_hashes,
            segment_index,
            stitch_version,
            {emb_col} as embedding
        FROM {view_name}
        WHERE {where_clause}
          AND ({keyword_conditions})
        LIMIT %s;
        """

        conn = self._get_connection()
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            query_params = params + keyword_params + [limit]
            cursor.execute(query, query_params)
            results = cursor.fetchall()
            cursor.close()

            logger.info(f"[{self.dashboard_id}] Keyword search found {len(results)} text matches")

            # If query_embedding provided, re-rank by similarity
            if query_embedding is not None and len(results) > 0:
                import numpy as np

                # Calculate similarities
                segment_embeddings = []
                valid_results = []
                for row in results:
                    if row[14] is not None:  # embedding field
                        segment_embeddings.append(np.array(row[14]))
                        valid_results.append(row)

                if not segment_embeddings:
                    logger.warning("No embeddings found for keyword matches")
                    return self._format_keyword_results(results)

                # Compute similarities
                segment_embeddings = np.array(segment_embeddings)
                query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
                seg_norms = segment_embeddings / (np.linalg.norm(segment_embeddings, axis=1, keepdims=True) + 1e-8)
                similarities = (query_norm.reshape(1, -1) @ seg_norms.T).flatten()

                # Filter and format
                formatted = []
                for row, sim in zip(valid_results, similarities):
                    if sim >= similarity_threshold:
                        formatted.append(self._format_result_row(row, float(sim)))

                formatted.sort(key=lambda x: x['similarity'], reverse=True)
                logger.info(f"Keyword search with semantic filtering: {len(formatted)}/{len(results)} above threshold")
                return formatted
            else:
                return self._format_keyword_results(results)

        finally:
            conn.close()

    def _format_result_row(self, row, similarity=None):
        """Format a result row (RealDictRow) into dictionary"""
        publish_date = row['publish_date']
        if publish_date and hasattr(publish_date, 'isoformat'):
            publish_date = publish_date.isoformat()

        return {
            'segment_id': row['segment_id'],
            'similarity': similarity,
            'content_id': row['content_id'],
            'content_id_string': row['content_id_string'],
            'channel_url': None,  # Not available in materialized views
            'channel_name': row['channel_name'],
            'title': row['title'],
            'publish_date': publish_date,
            'text': row['text'],
            'start_time': float(row['start_time']),
            'end_time': float(row['end_time']),
            'speaker_hashes': row['speaker_hashes'] or [],
            'segment_index': row['segment_index'],
            'stitch_version': row['stitch_version']
        }

    def _format_keyword_results(self, results):
        """Format keyword results without similarity"""
        formatted = []
        for row in results:
            formatted.append(self._format_result_row(row))
        return formatted
