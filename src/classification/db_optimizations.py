#!/usr/bin/env python3
"""
Database optimization utilities for theme classification
"""

import logging
from typing import List, Set, Dict, Any, Optional
from contextlib import contextmanager
import numpy as np
from sqlalchemy import create_engine, text, pool
from sqlalchemy.orm import sessionmaker, scoped_session
from pathlib import Path
from src.utils.paths import get_project_root, get_config_path
import hashlib

logger = logging.getLogger(__name__)

class OptimizedDatabaseManager:
    """Optimized database operations with connection pooling and query batching"""
    
    def __init__(self, connection_string: Optional[str] = None):
        # Use connection pooling for better performance
        self.engine = create_engine(
            connection_string or self._get_connection_string(),
            poolclass=pool.QueuePool,
            pool_size=20,
            max_overflow=40,
            pool_pre_ping=True,  # Verify connections before using
            pool_recycle=3600,  # Recycle connections after 1 hour
        )
        
        # Use scoped sessions for thread safety
        self.SessionFactory = scoped_session(sessionmaker(bind=self.engine))
        
        # Prepared statement cache
        self._prepared_statements = {}
        
    def _get_connection_string(self) -> str:
        """Get database connection string from config"""
        from src.utils.config import load_config
        config = load_config()
        
        db_config = config['database']
        return f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    
    @contextmanager
    def get_session(self):
        """Get a database session with automatic cleanup"""
        session = self.SessionFactory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def batch_keyword_search(self, keywords: List[str], exclude_ids: Set[int], 
                            project: Optional[str] = None, limit: int = 1000) -> List[Dict]:
        """
        Optimized keyword search using full-text search instead of ILIKE
        """
        with self.get_session() as session:
            # Use PostgreSQL's full-text search for better performance
            # Create a tsvector from keywords for efficient searching
            keyword_query = ' | '.join(keywords[:20])  # OR operator in tsquery
            
            # Build exclude condition more efficiently
            exclude_condition = ""
            if exclude_ids:
                # Use ANY array operator instead of IN for large lists
                exclude_condition = "AND es.id != ALL(:exclude_ids)"
            
            project_condition = ""
            if project:
                project_condition = "AND c.projects @> :project_array"
            
            query = text(f"""
                SELECT 
                    es.id as segment_id,
                    es.content_id,
                    es.text,
                    es.start_time,
                    es.end_time,
                    es.segment_index,
                    c.title as episode_title,
                    c.channel_name as episode_channel,
                    ts_rank(to_tsvector('english', es.text), query) as rank
                FROM 
                    embedding_segments es
                    JOIN content c ON es.content_id = c.id,
                    to_tsquery('english', :keyword_query) query
                WHERE 
                    to_tsvector('english', es.text) @@ query
                    {exclude_condition}
                    {project_condition}
                ORDER BY rank DESC
                LIMIT :limit
            """)
            
            params = {
                'keyword_query': keyword_query,
                'limit': limit
            }
            
            if exclude_ids:
                params['exclude_ids'] = list(exclude_ids)
            
            if project:
                params['project_array'] = [project]
            
            result = session.execute(query, params)
            return [dict(row) for row in result]
    
    def batch_get_embeddings(self, segment_ids: List[int]) -> Dict[int, np.ndarray]:
        """
        Batch fetch embeddings for multiple segments
        """
        if not segment_ids:
            return {}
        
        with self.get_session() as session:
            # Use UNNEST for efficient batch fetching
            query = text("""
                SELECT id, embedding::float8[]
                FROM embedding_segments
                WHERE id = ANY(:segment_ids)
            """)
            
            result = session.execute(query, {'segment_ids': segment_ids})
            
            embeddings = {}
            for row in result:
                if row[1]:
                    embeddings[row[0]] = np.array(row[1], dtype=np.float32)
            
            return embeddings
    
    def create_indexes_if_not_exists(self):
        """
        Create performance indexes if they don't exist
        """
        indexes = [
            # Full-text search index
            """
            CREATE INDEX IF NOT EXISTS idx_embedding_segments_text_fts 
            ON embedding_segments 
            USING gin(to_tsvector('english', text))
            """,
            
            # Composite index for common query patterns
            """
            CREATE INDEX IF NOT EXISTS idx_embedding_segments_content_project 
            ON embedding_segments(content_id) 
            WHERE content_id IS NOT NULL
            """,
            
            # BRIN index for time-based queries (very space-efficient)
            """
            CREATE INDEX IF NOT EXISTS idx_embedding_segments_times_brin 
            ON embedding_segments 
            USING brin(start_time, end_time)
            """
        ]
        
        with self.get_session() as session:
            for index_sql in indexes:
                try:
                    session.execute(text(index_sql))
                    session.commit()
                except Exception as e:
                    logger.warning(f"Could not create index: {e}")
                    session.rollback()


class QueryCache:
    """In-memory cache for frequently used queries"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.cache = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.access_times = {}
        
    def _make_key(self, query_type: str, **params) -> str:
        """Generate cache key from query parameters"""
        key_data = f"{query_type}:{sorted(params.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, query_type: str, **params) -> Optional[Any]:
        """Get cached query result"""
        key = self._make_key(query_type, **params)
        
        if key in self.cache:
            # Check TTL
            import time
            if time.time() - self.access_times[key] < self.ttl_seconds:
                return self.cache[key]
            else:
                # Expired
                del self.cache[key]
                del self.access_times[key]
        
        return None
    
    def set(self, query_type: str, result: Any, **params):
        """Cache query result"""
        import time
        
        # Implement LRU if cache is full
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.access_times, key=self.access_times.get)
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        key = self._make_key(query_type, **params)
        self.cache[key] = result
        self.access_times[key] = time.time()