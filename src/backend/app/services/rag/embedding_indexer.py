"""
EmbeddingIndexer
================

FAISS index management with project-level caching and runtime merging.

Optimizes FAISS performance by:
1. Caching indexes per project (avoiding 13x data overhead)
2. Merging indexes at runtime for multi-project queries
3. Post-filtering by segment IDs for exact results
"""

import os
import pickle
import hashlib
from typing import List, Tuple, Optional, Set, Dict
from datetime import datetime, timedelta
import numpy as np
import faiss

from .segment_retriever import SegmentRetriever, Segment
from ...utils.backend_logger import get_logger

logger = get_logger("embedding_indexer")


class EmbeddingIndexer:
    """Manage FAISS indexes with project-level caching."""

    def __init__(
        self,
        cache_dir: str = "/tmp/faiss_cache",
        embedding_dim: int = 1024,
        cache_ttl_hours: int = 24
    ):
        """
        Initialize indexer.

        Args:
            cache_dir: Directory to store cached FAISS indexes
            embedding_dim: Embedding dimension (1024 for Jina)
            cache_ttl_hours: Cache time-to-live in hours
        """
        self.cache_dir = cache_dir
        self.embedding_dim = embedding_dim
        self.cache_ttl_hours = cache_ttl_hours

        # In-memory cache
        self._project_indexes: Dict[str, Tuple[faiss.Index, List[int], datetime]] = {}

        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_key(
        self,
        projects: List[str],
        date_range: Optional[Tuple[datetime, datetime]] = None,
        languages: Optional[List[str]] = None
    ) -> str:
        """
        Generate cache key for index.

        Args:
            projects: List of project names
            date_range: Optional date range filter
            languages: Optional language filter

        Returns:
            Cache key string
        """
        # Sort projects for consistent key
        projects_str = "_".join(sorted(projects))

        # Add date range to key
        date_str = ""
        if date_range:
            start, end = date_range
            date_str = f"_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"

        # Add languages to key
        lang_str = ""
        if languages:
            lang_str = f"_{'_'.join(sorted(languages))}"

        key = f"{projects_str}{date_str}{lang_str}"

        # Hash if too long
        if len(key) > 200:
            key = hashlib.md5(key.encode()).hexdigest()

        return key

    def _get_cache_path(self, cache_key: str) -> Tuple[str, str]:
        """Get paths for cached index and metadata."""
        index_path = os.path.join(self.cache_dir, f"{cache_key}.index")
        meta_path = os.path.join(self.cache_dir, f"{cache_key}.meta")
        return index_path, meta_path

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached index exists and is not expired."""
        index_path, meta_path = self._get_cache_path(cache_key)

        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            return False

        # Check TTL
        cache_time = datetime.fromtimestamp(os.path.getmtime(index_path))
        age_hours = (datetime.now() - cache_time).total_seconds() / 3600

        if age_hours > self.cache_ttl_hours:
            logger.debug(f"Cache expired: {cache_key} (age: {age_hours:.1f}h)")
            return False

        return True

    def _save_to_cache(
        self,
        cache_key: str,
        index: faiss.Index,
        segment_ids: List[int]
    ):
        """Save index and metadata to disk."""
        index_path, meta_path = self._get_cache_path(cache_key)

        try:
            # Save FAISS index
            faiss.write_index(index, index_path)

            # Save metadata
            metadata = {
                "segment_ids": segment_ids,
                "embedding_dim": self.embedding_dim,
                "created_at": datetime.now().isoformat()
            }
            with open(meta_path, "wb") as f:
                pickle.dump(metadata, f)

            logger.info(f"Saved index to cache: {cache_key} ({len(segment_ids)} segments)")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def _load_from_cache(self, cache_key: str) -> Optional[Tuple[faiss.Index, List[int]]]:
        """Load index and metadata from disk."""
        index_path, meta_path = self._get_cache_path(cache_key)

        try:
            # Load FAISS index
            index = faiss.read_index(index_path)

            # Load metadata
            with open(meta_path, "rb") as f:
                metadata = pickle.load(f)

            segment_ids = metadata["segment_ids"]
            logger.info(f"Loaded index from cache: {cache_key} ({len(segment_ids)} segments)")

            return index, segment_ids
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            return None

    def _build_index_from_segments(
        self,
        segments: List[Segment]
    ) -> Tuple[faiss.Index, List[int]]:
        """
        Build FAISS index from segments (in subprocess to avoid segfaults).

        Args:
            segments: List of Segment objects with embeddings

        Returns:
            Tuple of (faiss.Index, segment_ids)
        """
        # Extract embeddings and IDs
        embeddings = []
        segment_ids = []

        for seg in segments:
            if seg.embedding is not None:
                embeddings.append(seg.embedding)
                segment_ids.append(seg.id)

        if not embeddings:
            logger.warning("No embeddings found in segments")
            # Return empty index
            index = faiss.IndexFlatIP(self.embedding_dim)
            return index, []

        # Convert to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)

        # Build index in subprocess to avoid segfaults
        from ..faiss_builder import FAISSBuilder

        logger.info(f"Building FAISS index in subprocess: {len(embeddings)} segments")
        # Use empty metadata since we only need segment_ids for this service
        metadata = [{}] * len(segment_ids)
        index, segment_ids, _ = FAISSBuilder.build_index_subprocess(
            embeddings_array, segment_ids, metadata, self.embedding_dim
        )

        logger.info(f"Built FAISS index with {len(segment_ids)} segments")

        return index, segment_ids

    def get_or_build_index(
        self,
        projects: List[str],
        date_range: Optional[Tuple[datetime, datetime]] = None,
        languages: Optional[List[str]] = None,
        force_rebuild: bool = False
    ) -> Tuple[faiss.Index, List[int]]:
        """
        Get or build FAISS index for specified filters.

        Strategy:
        1. Check in-memory cache
        2. Check disk cache
        3. Build from database if needed
        4. Save to cache

        Args:
            projects: List of project names
            date_range: Optional date range filter
            languages: Optional language filter
            force_rebuild: Force rebuild even if cached

        Returns:
            Tuple of (faiss.Index, segment_ids)
        """
        cache_key = self._get_cache_key(projects, date_range, languages)

        # Check in-memory cache
        if not force_rebuild and cache_key in self._project_indexes:
            index, segment_ids, cached_at = self._project_indexes[cache_key]
            age_hours = (datetime.now() - cached_at).total_seconds() / 3600
            if age_hours < self.cache_ttl_hours:
                logger.debug(f"Using in-memory cache: {cache_key}")
                return index, segment_ids
            else:
                logger.debug(f"In-memory cache expired: {cache_key}")
                del self._project_indexes[cache_key]

        # Check disk cache
        if not force_rebuild and self._is_cache_valid(cache_key):
            result = self._load_from_cache(cache_key)
            if result:
                index, segment_ids = result
                # Update in-memory cache
                self._project_indexes[cache_key] = (index, segment_ids, datetime.now())
                return index, segment_ids

        # Build from database
        logger.info(f"Building index from database: {cache_key}")
        retriever = SegmentRetriever()

        segments = retriever.fetch_by_filter(
            projects=projects,
            date_range=date_range,
            languages=languages,
            must_be_stitched=True,
            must_be_embedded=True
        )

        index, segment_ids = self._build_index_from_segments(segments)

        # Save to cache
        self._save_to_cache(cache_key, index, segment_ids)

        # Update in-memory cache
        self._project_indexes[cache_key] = (index, segment_ids, datetime.now())

        return index, segment_ids

    def search(
        self,
        query_embeddings: np.ndarray,
        projects: List[str],
        k: int = 100,
        threshold: float = 0.7,
        segment_filter: Optional[Set[int]] = None,
        date_range: Optional[Tuple[datetime, datetime]] = None,
        languages: Optional[List[str]] = None
    ) -> List[List[Tuple[int, float]]]:
        """
        Search FAISS index with post-filtering.

        Args:
            query_embeddings: Query embeddings (shape: [n_queries, embedding_dim])
            projects: List of project names
            k: Number of results per query
            threshold: Minimum similarity threshold (0-1)
            segment_filter: Optional set of segment IDs to filter by
            date_range: Optional date range filter
            languages: Optional language filter

        Returns:
            List of result lists, one per query.
            Each result is (segment_id, similarity_score)
        """
        # Get or build index
        index, segment_ids = self.get_or_build_index(projects, date_range, languages)

        if len(segment_ids) == 0:
            logger.warning("Empty index, returning no results")
            return [[] for _ in range(len(query_embeddings))]

        # Normalize query embeddings
        query_embeddings = query_embeddings.astype(np.float32)
        faiss.normalize_L2(query_embeddings)

        # Determine k for search (over-retrieve if filtering)
        search_k = k
        if segment_filter:
            # Over-retrieve to account for filtering
            # Assume ~50% will be filtered out
            search_k = min(k * 3, len(segment_ids))

        # FAISS search
        similarities, indices = index.search(query_embeddings, search_k)

        # Post-process results
        all_results = []

        for i in range(len(query_embeddings)):
            query_results = []

            for j in range(search_k):
                idx = indices[i, j]
                similarity = float(similarities[i, j])

                # Skip if below threshold
                if similarity < threshold:
                    continue

                # Map FAISS index to segment ID
                if idx >= 0 and idx < len(segment_ids):
                    segment_id = segment_ids[idx]

                    # Apply segment filter if provided
                    if segment_filter and segment_id not in segment_filter:
                        continue

                    query_results.append((segment_id, similarity))

            # Limit to k results
            query_results = query_results[:k]
            all_results.append(query_results)

        return all_results

    def search_batch(
        self,
        query_embeddings: np.ndarray,
        projects: List[str],
        k: int = 100,
        threshold: float = 0.7,
        segment_filter: Optional[Set[int]] = None,
        date_range: Optional[Tuple[datetime, datetime]] = None,
        languages: Optional[List[str]] = None
    ) -> List[List[Tuple[int, float]]]:
        """
        Batch search (alias for search, which is already vectorized).

        This method exists for API consistency with the architecture doc.
        """
        return self.search(
            query_embeddings=query_embeddings,
            projects=projects,
            k=k,
            threshold=threshold,
            segment_filter=segment_filter,
            date_range=date_range,
            languages=languages
        )

    def clear_cache(self, cache_key: Optional[str] = None):
        """
        Clear cache.

        Args:
            cache_key: Specific cache key to clear, or None to clear all
        """
        if cache_key:
            # Clear specific cache
            if cache_key in self._project_indexes:
                del self._project_indexes[cache_key]

            index_path, meta_path = self._get_cache_path(cache_key)
            if os.path.exists(index_path):
                os.remove(index_path)
            if os.path.exists(meta_path):
                os.remove(meta_path)

            logger.info(f"Cleared cache: {cache_key}")
        else:
            # Clear all caches
            self._project_indexes.clear()

            for filename in os.listdir(self.cache_dir):
                filepath = os.path.join(self.cache_dir, filename)
                if os.path.isfile(filepath):
                    os.remove(filepath)

            logger.info("Cleared all caches")
