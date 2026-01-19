"""
Query Service
=============

Service for executing read-only segment queries with filtering and search.
Replaces direct database access in dashboards.
"""

import logging
import time
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
from src.utils.paths import get_project_root, get_config_path
import sys

# Setup logger
sys.path.insert(0, str(get_project_root()))
from src.utils.logger import setup_worker_logger
logger = setup_worker_logger("backend.query_service")


class QueryService:
    """Service for querying segments with filters and search"""

    def __init__(self):
        """Initialize query service"""
        from src.backend.app.config.database import get_db_config
        self.db_config = get_db_config()

    def _get_connection(self):
        """Get PostgreSQL connection"""
        return psycopg2.connect(**self.db_config)

    async def query_segments(
        self,
        project: str,
        filters: Optional[Dict[str, Any]] = None,
        search: Optional[Dict[str, str]] = None,
        pagination: Optional[Dict[str, int]] = None,
        sort: Optional[Dict[str, str]] = None
    ) -> Tuple[List[Dict], int, Dict]:
        """
        Query segments with filtering and optional search

        Args:
            project: Project ID
            filters: Optional filters (dates, language, confidence, etc.)
            search: Optional search config {"mode": "semantic|keyword", "query": "..."}
            pagination: Optional pagination {"limit": 1000, "offset": 0}
            sort: Optional sort config {"field": "publish_date", "order": "desc"}

        Returns:
            (results, total_count, filters_applied_info)
        """
        start_time = time.time()

        filters = filters or {}
        pagination = pagination or {}
        sort = sort or {}

        # Parse pagination
        limit = min(pagination.get('limit', 1000), 5000)
        offset = pagination.get('offset', 0)

        # Parse sort
        sort_field = sort.get('field', 'publish_date')
        sort_order = sort.get('order', 'desc').upper()

        # Map sort field to SQL
        sort_field_map = {
            'publish_date': 'c.publish_date',
            'start_time': 'es.start_time',
            'confidence': "(es.meta_data->>'avg_confidence')::float"
        }
        sql_sort_field = sort_field_map.get(sort_field, 'c.publish_date')

        # Execute query based on search mode
        if search and search.get('mode'):
            if search['mode'] == 'semantic':
                results, total_count, filters_applied = await self._semantic_search_query(
                    project, filters, search['query'], limit, offset, sql_sort_field, sort_order
                )
            elif search['mode'] == 'keyword':
                results, total_count, filters_applied = self._keyword_search_query(
                    project, filters, search['query'], limit, offset, sql_sort_field, sort_order
                )
            else:
                raise ValueError(f"Invalid search mode: {search['mode']}")
        else:
            # No search - just filter and browse
            results, total_count, filters_applied = self._filter_query(
                project, filters, limit, offset, sql_sort_field, sort_order
            )

        execution_time = (time.time() - start_time) * 1000
        logger.info(
            f"Query completed: project={project}, total={total_count}, "
            f"returned={len(results)}, time={execution_time:.0f}ms"
        )

        return results, total_count, filters_applied

    def _build_where_clause(self, project: str, filters: Dict[str, Any]) -> Tuple[List[str], List[Any], Dict]:
        """
        Build WHERE clause conditions from filters

        Returns:
            (conditions_list, parameters_list, filters_applied_info)
        """
        conditions = []
        params = []
        filters_applied = {
            'date_range': False,
            'language': None,
            'min_confidence': None,
            'content_ids': False,
            'channel_names': False
        }

        # Project filter (required)
        conditions.append("%s = ANY(c.projects)")
        params.append(project)

        # is_embedded filter (only show embedded content)
        conditions.append("c.is_embedded = true")

        # Date range
        if 'start_date' in filters and filters['start_date']:
            try:
                start_date = datetime.fromisoformat(filters['start_date'])
                conditions.append("c.publish_date >= %s")
                params.append(start_date)
                filters_applied['date_range'] = True
            except ValueError:
                raise ValueError(f"Invalid start_date format: {filters['start_date']}")

        if 'end_date' in filters and filters['end_date']:
            try:
                end_date = datetime.fromisoformat(filters['end_date'])
                conditions.append("c.publish_date <= %s")
                params.append(end_date)
                filters_applied['date_range'] = True
            except ValueError:
                raise ValueError(f"Invalid end_date format: {filters['end_date']}")

        # Language filter
        if 'language' in filters and filters['language']:
            conditions.append("c.main_language = %s")
            params.append(filters['language'])
            filters_applied['language'] = filters['language']

        # Confidence filter
        if 'min_confidence' in filters and filters['min_confidence'] is not None:
            conditions.append("(es.meta_data->>'avg_confidence')::float >= %s")
            params.append(filters['min_confidence'])
            filters_applied['min_confidence'] = filters['min_confidence']

        # Content IDs filter
        if 'content_ids' in filters and filters['content_ids']:
            conditions.append("es.content_id = ANY(%s)")
            params.append(filters['content_ids'])
            filters_applied['content_ids'] = True

        # Channel names filter
        if 'channel_names' in filters and filters['channel_names']:
            conditions.append("c.channel_name = ANY(%s)")
            params.append(filters['channel_names'])
            filters_applied['channel_names'] = True

        return conditions, params, filters_applied

    def _filter_query(
        self,
        project: str,
        filters: Dict[str, Any],
        limit: int,
        offset: int,
        sort_field: str,
        sort_order: str
    ) -> Tuple[List[Dict], int, Dict]:
        """Execute query with filters only (no search)"""

        conditions, params, filters_applied = self._build_where_clause(project, filters)
        where_clause = " AND ".join(conditions)

        # Count query
        count_query = f"""
        SELECT COUNT(*) as total
        FROM embedding_segments es
        JOIN content c ON es.content_id = c.id
        WHERE {where_clause}
        """

        # Main query
        main_query = f"""
        SELECT
            es.id,
            es.content_id,
            es.content_id_string,
            es.text,
            es.start_time,
            es.end_time,
            es.meta_data,
            es.source_speaker_hashes,
            c.title,
            c.channel_name,
            c.platform,
            c.publish_date,
            c.main_language
        FROM embedding_segments es
        JOIN content c ON es.content_id = c.id
        WHERE {where_clause}
        ORDER BY {sort_field} {sort_order}, es.start_time ASC
        LIMIT %s OFFSET %s
        """

        conn = self._get_connection()
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Get total count
            cursor.execute(count_query, params)
            total_count = cursor.fetchone()['total']

            # Get results
            cursor.execute(main_query, params + [limit, offset])
            rows = cursor.fetchall()

            cursor.close()

            # Format results
            results = [self._format_row(row) for row in rows]

            return results, total_count, filters_applied

        finally:
            conn.close()

    def _keyword_search_query(
        self,
        project: str,
        filters: Dict[str, Any],
        query: str,
        limit: int,
        offset: int,
        sort_field: str,
        sort_order: str
    ) -> Tuple[List[Dict], int, Dict]:
        """Execute query with keyword search"""

        conditions, params, filters_applied = self._build_where_clause(project, filters)

        # Add keyword search condition
        conditions.append("es.text ILIKE %s")
        params.append(f'%{query}%')

        where_clause = " AND ".join(conditions)

        # Count query
        count_query = f"""
        SELECT COUNT(*) as total
        FROM embedding_segments es
        JOIN content c ON es.content_id = c.id
        WHERE {where_clause}
        """

        # Main query
        main_query = f"""
        SELECT
            es.id,
            es.content_id,
            es.content_id_string,
            es.text,
            es.start_time,
            es.end_time,
            es.meta_data,
            es.source_speaker_hashes,
            c.title,
            c.channel_name,
            c.platform,
            c.publish_date,
            c.main_language
        FROM embedding_segments es
        JOIN content c ON es.content_id = c.id
        WHERE {where_clause}
        ORDER BY {sort_field} {sort_order}, es.start_time ASC
        LIMIT %s OFFSET %s
        """

        conn = self._get_connection()
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Get total count
            cursor.execute(count_query, params)
            total_count = cursor.fetchone()['total']

            # Get results
            cursor.execute(main_query, params + [limit, offset])
            rows = cursor.fetchall()

            cursor.close()

            # Format results
            results = [self._format_row(row) for row in rows]

            return results, total_count, filters_applied

        finally:
            conn.close()

    async def _semantic_search_query(
        self,
        project: str,
        filters: Dict[str, Any],
        query: str,
        limit: int,
        offset: int,
        sort_field: str,
        sort_order: str
    ) -> Tuple[List[Dict], int, Dict]:
        """Execute query with semantic search"""

        # Get embedding model from global cache
        try:
            # Import from app.main (not src.backend.main)
            from ..main import _embedding_models, wait_for_models
            import numpy as np
            import asyncio

            # Wait for models to be loaded (non-blocking async wait)
            await wait_for_models()

            if '0.6B' not in _embedding_models:
                raise RuntimeError("Embedding model not loaded")

            model = _embedding_models['0.6B']

            # Run model encoding in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            query_embedding = await loop.run_in_executor(
                None,  # Use default executor
                lambda: model.encode([query], convert_to_numpy=True)[0]
            )

        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}", exc_info=True)
            raise RuntimeError(f"Semantic search unavailable: {e}")

        # Convert to PostgreSQL vector format
        embedding_list = query_embedding.flatten().tolist()
        embedding_str = '[' + ','.join(map(str, embedding_list)) + ']'

        # Build WHERE clause
        conditions, params, filters_applied = self._build_where_clause(project, filters)

        # Add similarity threshold (0.4 default, converted to distance)
        threshold = 0.4
        max_distance = 1.0 - threshold
        conditions.append("(es.embedding <=> %s::vector) < %s")

        where_clause = " AND ".join(conditions)

        # Count query (segments matching filters AND within similarity threshold)
        count_query = f"""
        SELECT COUNT(*) as total
        FROM embedding_segments es
        JOIN content c ON es.content_id = c.id
        WHERE {where_clause}
        """

        # Main query with similarity ranking
        main_query = f"""
        SELECT
            es.id,
            es.content_id,
            es.content_id_string,
            es.text,
            es.start_time,
            es.end_time,
            es.meta_data,
            es.source_speaker_hashes,
            1 - (es.embedding <=> %s::vector) as similarity,
            c.title,
            c.channel_name,
            c.platform,
            c.publish_date,
            c.main_language
        FROM embedding_segments es
        JOIN content c ON es.content_id = c.id
        WHERE {where_clause}
        ORDER BY es.embedding <=> %s::vector
        LIMIT %s OFFSET %s
        """

        conn = self._get_connection()
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)

            # Set IVFFlat probes for performance
            cursor.execute("SET ivfflat.probes = 10;")

            # Get total count
            count_params = params + [embedding_str, max_distance]
            cursor.execute(count_query, count_params)
            total_count = cursor.fetchone()['total']

            # Get results
            main_params = [embedding_str] + params + [embedding_str, max_distance] + [embedding_str, embedding_str, limit, offset]
            cursor.execute(main_query, main_params)
            rows = cursor.fetchall()

            cursor.close()

            # Format results
            results = [self._format_row(row, include_similarity=True) for row in rows]

            return results, total_count, filters_applied

        finally:
            conn.close()

    def _format_row(self, row: Dict, include_similarity: bool = False) -> Dict:
        """Format database row to API response format"""

        # Parse meta_data if present
        meta_data = row.get('meta_data') or {}
        avg_confidence = meta_data.get('avg_confidence')

        # Format publish_date
        publish_date = row.get('publish_date')
        if publish_date and hasattr(publish_date, 'isoformat'):
            publish_date = publish_date.isoformat()

        # Calculate duration
        duration = float(row['end_time']) - float(row['start_time'])

        # Create segment_id
        segment_id = f"{row['content_id']}_{int(row['start_time'])}"

        result = {
            'segment_id': segment_id,
            'content_id': row['content_id'],
            'content_id_string': row.get('content_id_string'),
            'title': row.get('title'),
            'channel_name': row.get('channel_name'),
            'platform': row.get('platform'),
            'publish_date': publish_date,
            'text': row['text'],
            'start_time': float(row['start_time']),
            'end_time': float(row['end_time']),
            'duration': duration,
            'language': row.get('main_language'),
            'confidence': avg_confidence,
            'speaker_hashes': row.get('source_speaker_hashes') or [],
            'meta_data': meta_data
        }

        # Add similarity if semantic search
        if include_similarity and 'similarity' in row:
            result['similarity'] = float(row['similarity'])

        return result
