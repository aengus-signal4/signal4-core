"""
Query Variation Service
=======================

Manages a normalized library of query variations with embeddings.
This is the foundation of the embedding caching system.

See workflows.py for full caching architecture documentation.

Key Design Principles:
----------------------
1. **Same text = same embedding**: Never embed the same text twice
2. **Variation reuse**: If "Mark Carney policy" appears in multiple query
   expansions, it's stored and embedded only once
3. **Growing library**: Over time, more variations are pre-embedded,
   reducing embedding latency for common queries

Tables:
-------
- query_variations: Unique texts with embeddings (1024-dim, 0.6B model)
- query_expansions: Maps original queries to their variations

Performance:
------------
- Hash lookup: O(1) via MD5 text_hash index
- Embedding lookup: Direct fetch, no similarity search needed
- Typical cache hit: <10ms for 10 variations

Usage:
------
    service = QueryVariationService(dashboard_id)

    # Check which variations need embedding
    texts = ["Mark Carney PM", "Carney tariffs", ...]
    missing = service.get_missing_variations(texts)

    # Store new variations with embeddings
    service.store_variations(texts, embeddings)

    # Get all variations for an original query (with embeddings)
    result = service.get_expansion("What is Mark Carney saying?")
    # Returns: {'variations': [...], 'embeddings': [...], 'all_embedded': True}

    # Get stats
    stats = service.get_stats()
    # Returns: {'total_variations': 100, 'with_embeddings': 95, ...}
"""

import hashlib
import logging
import numpy as np
import psycopg2
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from ..utils.backend_logger import get_logger

logger = get_logger("query_variations")


class QueryVariationService:
    """Service for managing query variations with embeddings."""

    def __init__(self, dashboard_id: str = "unknown"):
        """
        Initialize query variation service.

        Args:
            dashboard_id: Dashboard identifier for logging
        """
        self.dashboard_id = dashboard_id
        from ..config.database import get_db_config
        self.db_config = get_db_config()

    def _get_connection(self):
        """Get PostgreSQL connection."""
        return psycopg2.connect(**self.db_config)

    @staticmethod
    def _hash_text(text: str) -> str:
        """Compute MD5 hash of normalized text."""
        normalized = text.strip().lower()
        return hashlib.md5(normalized.encode()).hexdigest()

    def get_variations_with_embeddings(
        self,
        texts: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        Get embeddings for texts that already exist in the library.

        Args:
            texts: List of query variation texts

        Returns:
            Dict mapping text -> embedding (only for texts that exist)
        """
        if not texts:
            return {}

        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # Compute hashes for lookup
            hashes = [self._hash_text(t) for t in texts]
            hash_to_text = {h: t for h, t in zip(hashes, texts)}

            # Query for existing embeddings
            placeholders = ','.join(['%s'] * len(hashes))
            cursor.execute(f"""
                SELECT text_hash, embedding
                FROM query_variations
                WHERE text_hash IN ({placeholders})
                AND embedding IS NOT NULL
            """, hashes)

            result = {}
            for row in cursor.fetchall():
                text_hash, embedding_str = row
                if text_hash in hash_to_text and embedding_str:
                    # Convert pgvector string to numpy array
                    embedding = np.array(
                        [float(x) for x in embedding_str.strip('[]').split(',')],
                        dtype=np.float32
                    )
                    result[hash_to_text[text_hash]] = embedding

            # Update usage stats for found variations
            if result:
                found_hashes = [self._hash_text(t) for t in result.keys()]
                placeholders = ','.join(['%s'] * len(found_hashes))
                cursor.execute(f"""
                    UPDATE query_variations
                    SET last_used_at = NOW(), usage_count = usage_count + 1
                    WHERE text_hash IN ({placeholders})
                """, found_hashes)
                conn.commit()

            cursor.close()
            return result

        finally:
            conn.close()

    def get_missing_variations(self, texts: List[str]) -> List[str]:
        """
        Get texts that don't have embeddings in the library.

        Args:
            texts: List of query variation texts

        Returns:
            List of texts that need embedding
        """
        existing = self.get_variations_with_embeddings(texts)
        return [t for t in texts if t not in existing]

    def store_variations(
        self,
        texts: List[str],
        embeddings: List[np.ndarray]
    ) -> int:
        """
        Store query variations with their embeddings.

        Args:
            texts: List of query variation texts
            embeddings: List of corresponding embeddings

        Returns:
            Number of new variations stored
        """
        if not texts or len(texts) != len(embeddings):
            return 0

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            stored = 0

            for text, embedding in zip(texts, embeddings):
                text_hash = self._hash_text(text)

                # Convert embedding to PostgreSQL format
                embedding_list = embedding.flatten().tolist()
                embedding_str = '[' + ','.join(map(str, embedding_list)) + ']'

                # Upsert: insert or update if exists
                cursor.execute("""
                    INSERT INTO query_variations (text_hash, text, embedding)
                    VALUES (%s, %s, %s::vector)
                    ON CONFLICT (text_hash) DO UPDATE SET
                        embedding = COALESCE(query_variations.embedding, EXCLUDED.embedding),
                        last_used_at = NOW(),
                        usage_count = query_variations.usage_count + 1
                    RETURNING (xmax = 0) as is_insert
                """, (text_hash, text, embedding_str))

                is_new = cursor.fetchone()[0]
                if is_new:
                    stored += 1

            conn.commit()
            cursor.close()

            logger.info(
                f"[{self.dashboard_id}] Stored {stored} new variations "
                f"(updated {len(texts) - stored} existing)"
            )
            return stored

        finally:
            conn.close()

    def store_expansion(
        self,
        original_query: str,
        variations: List[str],
        embeddings: Optional[List[np.ndarray]] = None
    ) -> None:
        """
        Store an original query -> variations mapping.

        Also stores embeddings if provided.

        Args:
            original_query: The original user query
            variations: List of expanded query variations
            embeddings: Optional list of embeddings for variations
        """
        if not variations:
            return

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            original_hash = self._hash_text(original_query)

            # First, ensure all variations exist in query_variations
            for i, text in enumerate(variations):
                text_hash = self._hash_text(text)

                # Prepare embedding if provided
                embedding_str = None
                if embeddings and i < len(embeddings):
                    embedding_list = embeddings[i].flatten().tolist()
                    embedding_str = '[' + ','.join(map(str, embedding_list)) + ']'

                # Insert variation (or update embedding if missing)
                if embedding_str:
                    cursor.execute("""
                        INSERT INTO query_variations (text_hash, text, embedding)
                        VALUES (%s, %s, %s::vector)
                        ON CONFLICT (text_hash) DO UPDATE SET
                            embedding = COALESCE(query_variations.embedding, EXCLUDED.embedding),
                            last_used_at = NOW(),
                            usage_count = query_variations.usage_count + 1
                        RETURNING id
                    """, (text_hash, text, embedding_str))
                else:
                    cursor.execute("""
                        INSERT INTO query_variations (text_hash, text)
                        VALUES (%s, %s)
                        ON CONFLICT (text_hash) DO UPDATE SET
                            last_used_at = NOW(),
                            usage_count = query_variations.usage_count + 1
                        RETURNING id
                    """, (text_hash, text))

                variation_id = cursor.fetchone()[0]

                # Link to original query
                cursor.execute("""
                    INSERT INTO query_expansions
                    (original_query_hash, original_query, variation_id, position)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (original_query_hash, variation_id) DO NOTHING
                """, (original_hash, original_query, variation_id, i))

            conn.commit()
            cursor.close()

            num_with_embeddings = len(embeddings) if embeddings else 0
            logger.info(
                f"[{self.dashboard_id}] Stored expansion: '{original_query[:40]}...' -> "
                f"{len(variations)} variations ({num_with_embeddings} with embeddings)"
            )

        finally:
            conn.close()

    def get_expansion(
        self,
        original_query: str
    ) -> Optional[Dict[str, any]]:
        """
        Get cached expansion for an original query.

        Args:
            original_query: The original user query

        Returns:
            Dict with 'variations' (texts) and 'embeddings' (numpy arrays),
            or None if not found
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            original_hash = self._hash_text(original_query)

            cursor.execute("""
                SELECT v.text, v.embedding
                FROM query_expansions e
                JOIN query_variations v ON e.variation_id = v.id
                WHERE e.original_query_hash = %s
                ORDER BY e.position
            """, (original_hash,))

            rows = cursor.fetchall()
            cursor.close()

            if not rows:
                return None

            variations = []
            embeddings = []
            all_have_embeddings = True

            for text, embedding_str in rows:
                variations.append(text)
                if embedding_str:
                    embedding = np.array(
                        [float(x) for x in embedding_str.strip('[]').split(',')],
                        dtype=np.float32
                    )
                    embeddings.append(embedding)
                else:
                    all_have_embeddings = False

            logger.info(
                f"[{self.dashboard_id}] Found cached expansion: '{original_query[:40]}...' -> "
                f"{len(variations)} variations "
                f"({'all embedded' if all_have_embeddings else f'{len(embeddings)} embedded'})"
            )

            return {
                'variations': variations,
                'embeddings': embeddings if all_have_embeddings else None,
                'all_embedded': all_have_embeddings
            }

        finally:
            conn.close()

    def get_stats(self) -> Dict[str, any]:
        """Get statistics about the variation library."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    COUNT(*) as total_variations,
                    COUNT(*) FILTER (WHERE embedding IS NOT NULL) as with_embeddings,
                    SUM(usage_count) as total_uses,
                    COUNT(DISTINCT e.original_query_hash) as unique_queries
                FROM query_variations v
                LEFT JOIN query_expansions e ON v.id = e.variation_id
            """)

            row = cursor.fetchone()
            cursor.close()

            return {
                'total_variations': row[0],
                'with_embeddings': row[1],
                'total_uses': row[2] or 0,
                'unique_queries': row[3] or 0
            }

        finally:
            conn.close()
