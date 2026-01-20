"""
PostgreSQL pgvector Search Service
===================================

PostgreSQL pgvector-based semantic search using per-project HNSW indexes.

Architecture:
- Each project has its own partial HNSW index for fast (~8ms) queries
- Queries with multiple projects run against each project's index separately
- Results are combined, deduplicated, and returned sorted by similarity

Cache tables:
- embedding_cache_7d: Hot cache (hourly refresh) for recent content (≤7 days)
- embedding_cache_30d: Main cache (6-hour refresh) for all queries (8+ days)

Embedding model:
- embedding: vector(1024) - Qwen 0.6B model (only model used)

Per-project HNSW indexes (on both 7d and 30d tables):
- embedding_cache_Xd_hnsw_canadian
- embedding_cache_Xd_hnsw_big_channels
- embedding_cache_Xd_hnsw_health
- embedding_cache_Xd_hnsw_finance
- embedding_cache_Xd_hnsw_europe
- embedding_cache_Xd_hnsw_cprmv
- embedding_cache_Xd_hnsw_anglosphere

HNSW Index Settings:
- m=16, ef_construction=64 (build-time parameters)
- hnsw.ef_search=100 (query-time parameter, set per-connection)

Query Strategy:
1. For each project in allowed_projects, query its partial index
2. Each query uses pure HNSW nearest-neighbor (~8ms) - no filter overhead
3. Combine results, deduplicate by segment_id (keep best similarity)
4. Apply threshold filter in Python
5. Return sorted results
"""

import logging
import time
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Any, Optional, Set
from datetime import datetime, timedelta

from ..utils.backend_logger import get_logger
logger = get_logger("pgvector_search")

# Projects that have per-project HNSW indexes
INDEXED_PROJECTS = {
    'Canadian', 'Big_Channels', 'Health', 'Finance',
    'Europe', 'CPRMV', 'Anglosphere'
}


class PgVectorSearchService:
    """PostgreSQL pgvector-based semantic search service using per-project indexes"""

    def __init__(self, dashboard_id: str, config):
        """
        Initialize pgvector search service

        Args:
            dashboard_id: Dashboard identifier
            config: DashboardConfig object with allowed_projects
        """
        self.dashboard_id = dashboard_id
        self.config = config

        # PostgreSQL connection parameters
        from src.backend.app.config.database import get_db_config
        self.db_config = get_db_config()

        # Determine which projects to query
        self.query_projects = self._get_query_projects()

        logger.info(
            f"PgVectorSearchService initialized for {dashboard_id} "
            f"(projects: {self.query_projects})"
        )

    def _get_query_projects(self) -> List[str]:
        """
        Get list of projects to query based on config.

        Returns:
            List of project names that have HNSW indexes
        """
        if not self.config.allowed_projects:
            # No filter = query all indexed projects
            return list(INDEXED_PROJECTS)

        # Filter to only projects that have indexes
        projects = [p for p in self.config.allowed_projects if p in INDEXED_PROJECTS]

        if not projects:
            logger.warning(
                f"No indexed projects found in allowed_projects: {self.config.allowed_projects}. "
                f"Available: {INDEXED_PROJECTS}"
            )
            # Fallback to querying all
            return list(INDEXED_PROJECTS)

        return projects

    def _get_connection(self):
        """Get PostgreSQL connection"""
        return psycopg2.connect(**self.db_config)

    def _select_table(self, time_window_days: int) -> str:
        """
        Select appropriate cache table based on time window

        Args:
            time_window_days: Time window in days

        Returns:
            Table name to query
        """
        if time_window_days <= 7:
            return 'embedding_cache_7d'
        else:
            return 'embedding_cache_30d'

    def _get_index_name(self, table: str, project: str) -> str:
        """
        Get the HNSW index name for a project

        Args:
            table: Cache table name (embedding_cache_7d or embedding_cache_30d)
            project: Project name

        Returns:
            Index name like 'embedding_cache_30d_hnsw_canadian'
        """
        return f"{table}_hnsw_{project.lower()}"

    def search(
        self,
        query_embedding,
        time_window_days: int = 7,
        k: int = 200,
        threshold: float = 0.43,
        must_contain: List[str] = None,
        must_contain_any: List[str] = None,
        **filters
    ) -> List[Dict]:
        """
        Semantic search using per-project HNSW indexes.

        Queries each project's index separately for fast (~8ms each) results,
        then combines and deduplicates.

        Args:
            query_embedding: Query embedding vector (numpy array, 1024-dim)
            time_window_days: Time window in days (7 or 30)
            k: Number of results per project (final results may be more after combining)
            threshold: Similarity threshold (0.0-1.0)
            must_contain: List of keywords that ALL must appear in text (AND logic)
            must_contain_any: List of keywords where AT LEAST ONE must appear (OR logic)
            **filters: Additional filters (filter_speakers, filter_content_ids, etc.)

        Returns:
            List of result dictionaries with segment info and similarity scores,
            sorted by similarity descending
        """
        start_time = time.time()

        table = self._select_table(time_window_days)
        projects = self.query_projects

        # Convert embedding to PostgreSQL format
        embedding_list = query_embedding.flatten().tolist()
        embedding_str = '[' + ','.join(map(str, embedding_list)) + ']'

        # Build additional WHERE conditions (excluding project filter)
        extra_conditions, extra_params = self._build_extra_conditions(
            filters, time_window_days, must_contain, must_contain_any
        )

        conn = self._get_connection()
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("SET hnsw.ef_search = 100;")

            # Query each project's index and collect results
            all_results: Dict[int, Dict] = {}  # segment_id -> best result
            query_times = []

            for project in projects:
                index_name = self._get_index_name(table, project)

                # Build query with index hint
                # The WHERE clause with project filter makes PostgreSQL use the partial index
                query = f"""
                SELECT
                    id as segment_id,
                    1 - (embedding <=> %s::vector) as similarity,
                    content_id,
                    content_id_string,
                    channel_url,
                    channel_name,
                    title,
                    publish_date,
                    text,
                    start_time,
                    end_time,
                    source_speaker_hashes as speaker_hashes,
                    segment_index,
                    stitch_version
                FROM {table}
                WHERE projects && ARRAY[%s]::varchar[]
                  {extra_conditions}
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """

                query_start = time.time()
                params = [embedding_str, project] + extra_params + [embedding_str, k]
                cursor.execute(query, params)
                results = cursor.fetchall()
                query_time = (time.time() - query_start) * 1000
                query_times.append((project, query_time, len(results)))

                # Merge results, keeping best similarity per segment
                for row in results:
                    seg_id = row['segment_id']
                    similarity = float(row['similarity'])

                    if seg_id not in all_results or similarity > all_results[seg_id]['similarity']:
                        # Format the result
                        publish_date = row['publish_date']
                        if publish_date and hasattr(publish_date, 'isoformat'):
                            publish_date = publish_date.isoformat()

                        all_results[seg_id] = {
                            'segment_id': seg_id,
                            'similarity': similarity,
                            'content_id': row['content_id'],
                            'content_id_string': row['content_id_string'],
                            'channel_url': row['channel_url'],
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

            cursor.close()

            # Apply threshold filter and sort by similarity
            formatted = [
                r for r in all_results.values()
                if r['similarity'] >= threshold
            ]
            formatted.sort(key=lambda x: x['similarity'], reverse=True)

            total_time = (time.time() - start_time) * 1000
            total_query_time = sum(qt[1] for qt in query_times)

            logger.info(
                f"[{self.dashboard_id}] pgvector search: {len(formatted)} results "
                f"(threshold={threshold}) in {total_time:.0f}ms "
                f"(queries={total_query_time:.0f}ms across {len(projects)} projects, "
                f"table={table})"
            )
            for proj, qt, count in query_times:
                logger.debug(f"  {proj}: {qt:.0f}ms, {count} results")

            return formatted

        finally:
            conn.close()

    def _build_extra_conditions(
        self,
        filters: Dict[str, Any],
        time_window_days: int,
        must_contain: List[str] = None,
        must_contain_any: List[str] = None
    ) -> tuple:
        """
        Build additional WHERE conditions (excluding project filter).

        Returns:
            (conditions_string, params_list) - conditions string starts with AND if non-empty
        """
        conditions = []
        params = []

        # Date filter for 30d table
        if time_window_days > 7 and 'date_from' not in filters:
            conditions.append("publish_date >= NOW() - INTERVAL '%s days'")
            params.append(time_window_days)

        # Custom date range
        if 'date_from' in filters:
            conditions.append("publish_date >= %s")
            params.append(filters['date_from'])

        if 'date_to' in filters:
            conditions.append("publish_date <= %s")
            params.append(filters['date_to'])

        # Content ID filter
        if filters.get('filter_content_ids'):
            conditions.append("content_id = ANY(%s::int[])")
            params.append(filters['filter_content_ids'])

        # Speaker filter
        if filters.get('filter_speakers'):
            conditions.append("source_speaker_hashes && %s::text[]")
            params.append(filters['filter_speakers'])

        # Stitch version filter
        if filters.get('filter_stitch_versions'):
            conditions.append("stitch_version = ANY(%s::text[])")
            params.append(filters['filter_stitch_versions'])

        # Keyword filters
        if must_contain:
            for kw in must_contain:
                conditions.append("text ILIKE %s")
                params.append(f'%{kw}%')

        if must_contain_any:
            or_conditions = " OR ".join(["text ILIKE %s"] * len(must_contain_any))
            conditions.append(f"({or_conditions})")
            params.extend([f'%{kw}%' for kw in must_contain_any])

        if conditions:
            return "AND " + " AND ".join(conditions), params
        return "", []

    def batch_search(
        self,
        query_embeddings: List,
        time_window_days: int = 7,
        k: int = 200,
        threshold: float = 0.43,
        **filters
    ) -> List[List[Dict]]:
        """
        Batch semantic search (multiple queries).

        Each query is run using the per-project index strategy.

        Args:
            query_embeddings: List of query embedding vectors
            time_window_days: Time window in days
            k: Number of results per query
            threshold: Similarity threshold
            **filters: Additional filters

        Returns:
            List of result lists (one per query)
        """
        return [
            self.search(
                query_embedding=emb,
                time_window_days=time_window_days,
                k=k,
                threshold=threshold,
                **filters
            )
            for emb in query_embeddings
        ]

    def batch_search_unified(
        self,
        query_embeddings: List,
        time_window_days: int = 7,
        k: int = None,
        threshold: float = 0.43,
        must_contain: List[str] = None,
        must_contain_any: List[str] = None,
        **filters
    ) -> List[Dict]:
        """
        Unified batch search - combines results from multiple query embeddings.

        Returns segments that match ANY of the query embeddings above threshold,
        deduplicated by segment_id (keeping best similarity).

        Args:
            query_embeddings: List of query embedding vectors
            time_window_days: Time window in days
            k: Optional max total results
            threshold: Similarity threshold
            must_contain: Keywords that ALL must appear (AND logic)
            must_contain_any: Keywords where AT LEAST ONE must appear (OR logic)
            **filters: Additional filters

        Returns:
            Single list of unique segments sorted by best similarity
        """
        start_time = time.time()

        if not query_embeddings:
            return []

        # Run each query and collect all results
        all_results: Dict[int, Dict] = {}  # segment_id -> best result

        for emb in query_embeddings:
            results = self.search(
                query_embedding=emb,
                time_window_days=time_window_days,
                k=k or 500,  # Fetch more per query to ensure good coverage
                threshold=threshold,
                must_contain=must_contain,
                must_contain_any=must_contain_any,
                **filters
            )

            for r in results:
                seg_id = r['segment_id']
                if seg_id not in all_results or r['similarity'] > all_results[seg_id]['similarity']:
                    all_results[seg_id] = r

        # Sort by similarity and apply k limit
        formatted = sorted(all_results.values(), key=lambda x: x['similarity'], reverse=True)

        if k:
            formatted = formatted[:k]

        total_time = (time.time() - start_time) * 1000
        logger.info(
            f"[{self.dashboard_id}] pgvector batch_search_unified: {len(formatted)} unique results "
            f"from {len(query_embeddings)} queries in {total_time:.0f}ms"
        )

        return formatted

    def keyword_search(
        self,
        keywords: List[str],
        time_window_days: int = 7,
        limit: int = 200,
        query_embedding=None,
        similarity_threshold: float = 0.40
    ) -> List[Dict]:
        """
        Keyword search using PostgreSQL text matching.

        Searches across all projects (uses the table directly, not per-project indexes).

        Args:
            keywords: List of keywords/phrases to search
            time_window_days: Time window
            limit: Max results
            query_embedding: Optional embedding for re-ranking
            similarity_threshold: Min similarity if re-ranking

        Returns:
            List of matching segments
        """
        import numpy as np

        table = self._select_table(time_window_days)

        # Build project filter
        if self.config.allowed_projects:
            project_condition = "projects && %s::varchar[]"
            project_params = [self.config.allowed_projects]
        else:
            project_condition = "1=1"
            project_params = []

        # Build keyword conditions
        keyword_conditions = " OR ".join(["text ILIKE %s"] * len(keywords))
        keyword_params = [f'%{kw}%' for kw in keywords]

        query = f"""
        SELECT
            id as segment_id,
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
            embedding
        FROM {table}
        WHERE {project_condition}
          AND ({keyword_conditions})
        LIMIT %s;
        """

        conn = self._get_connection()
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            params = project_params + keyword_params + [limit]
            cursor.execute(query, params)
            results = cursor.fetchall()
            cursor.close()

            logger.info(f"[{self.dashboard_id}] Keyword search found {len(results)} text matches")

            # If query_embedding provided, re-rank by similarity
            if query_embedding is not None and len(results) > 0:
                segment_embeddings = []
                valid_results = []

                for row in results:
                    if row.get('embedding'):
                        emb_str = row['embedding']
                        if isinstance(emb_str, str):
                            emb_str = emb_str.strip('[]')
                            embedding = np.array([float(x) for x in emb_str.split(',')])
                        else:
                            embedding = np.array(emb_str)
                        segment_embeddings.append(embedding)
                        valid_results.append(row)

                if not segment_embeddings:
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
        """Format a result row into dictionary"""
        publish_date = row['publish_date']
        if publish_date and hasattr(publish_date, 'isoformat'):
            publish_date = publish_date.isoformat()

        return {
            'segment_id': row['segment_id'],
            'similarity': similarity,
            'content_id': row['content_id'],
            'content_id_string': row['content_id_string'],
            'channel_url': row.get('channel_url'),
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
        return [self._format_result_row(row) for row in results]
