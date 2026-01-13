"""
SmartRetriever
==============

High-level retrieval interface with natural language query support.

Combines QueryParser + SegmentRetriever + EmbeddingIndexer for end-to-end
retrieval from natural language queries.

Example:
    retriever = SmartRetriever()

    # Natural language query
    results = retriever.retrieve("Recent French content about immigration", k=50)

    # Structured query
    results = retriever.retrieve(
        projects=["CPRMV"],
        languages=["fr"],
        time_window_days=30,
        k=50
    )
"""

from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import numpy as np

from .query_parser import QueryParser
from .segment_retriever import SegmentRetriever, Segment
from .embedding_indexer import EmbeddingIndexer
from ..llm_service import LLMService
from ..embedding_service import EmbeddingService
from ...utils.backend_logger import get_logger

logger = get_logger("smart_retriever")


class SmartRetriever:
    """High-level retrieval with natural language query support."""

    def __init__(
        self,
        llm_service: LLMService,
        embedding_service: EmbeddingService,
        embedding_indexer: Optional[EmbeddingIndexer] = None
    ):
        """
        Initialize smart retriever.

        Args:
            llm_service: LLMService instance (required for query parsing)
            embedding_service: EmbeddingService instance (required for semantic search)
            embedding_indexer: EmbeddingIndexer instance (optional, will create if not provided)
        """
        if llm_service is None:
            raise ValueError("LLMService is required for SmartRetriever")
        if embedding_service is None:
            raise ValueError("EmbeddingService is required for SmartRetriever")

        self.llm_service = llm_service
        self.embedding_service = embedding_service
        self.query_parser = QueryParser(llm_service)
        self.segment_retriever = SegmentRetriever()
        self.embedding_indexer = embedding_indexer or EmbeddingIndexer()

    def retrieve(
        self,
        query: Optional[str] = None,
        projects: Optional[List[str]] = None,
        languages: Optional[List[str]] = None,
        channels: Optional[List[str]] = None,
        date_range: Optional[Tuple[datetime, datetime]] = None,
        time_window_days: Optional[int] = None,
        k: int = 100,
        threshold: float = 0.7,
        semantic_search: bool = True,
        build_index: bool = True
    ) -> Dict[str, Any]:
        """
        Retrieve segments using natural language or structured filters.

        Args:
            query: Natural language query (e.g., "Recent French content about immigration")
            projects: List of project names (structured filter)
            languages: List of language codes (structured filter)
            channels: List of channel names (structured filter)
            date_range: Date range tuple (structured filter)
            time_window_days: Time window in days (structured filter)
            k: Number of results to return
            threshold: Similarity threshold for semantic search
            semantic_search: Perform semantic search if query provided
            build_index: Build FAISS index for semantic search

        Returns:
            Dictionary with:
            {
                "segments": List[Segment],
                "index": faiss.Index (if build_index=True),
                "segment_ids": List[int] (if build_index=True),
                "filters": Dict (applied filters),
                "count": int
            }
        """
        # Step 1: Parse query if provided
        filters = {}

        if query:
            # Use LLM to parse query into structured filters
            parsed = self.query_parser.parse(
                query,
                default_time_window_days=time_window_days or 30
            )

            # Merge with explicit parameters (explicit params override parsed)
            filters = {
                "projects": projects or parsed.get("projects"),
                "languages": languages or parsed.get("languages"),
                "channels": channels or parsed.get("channels"),
                "date_range": date_range or parsed.get("date_range"),
                "time_window_days": time_window_days or parsed.get("time_window_days"),
                "keywords": parsed.get("keywords", []),
                "intent": parsed.get("intent", "search"),
                "original_query": query
            }
        else:
            # Use only explicit parameters
            filters = {
                "projects": projects,
                "languages": languages,
                "channels": channels,
                "date_range": date_range,
                "time_window_days": time_window_days,
                "keywords": [],
                "intent": "search",
                "original_query": None
            }

            # Build date_range from time_window_days if needed
            if not date_range and time_window_days:
                filters["date_range"] = self.query_parser._time_window_to_date_range(
                    time_window_days
                )

        logger.info(f"Retrieving with filters: {filters}")

        # Step 2: Retrieve segments
        retriever_params = self.query_parser.filters_to_retriever_params(filters)

        segments = self.segment_retriever.fetch_by_filter(
            **retriever_params,
            must_be_embedded=True
        )

        logger.info(f"Retrieved {len(segments)} segments")

        # Step 3: Build FAISS index if requested
        index = None
        segment_ids = None

        if build_index and len(segments) > 0:
            index, segment_ids = self.embedding_indexer.get_or_build_index(
                projects=filters.get("projects"),
                date_range=filters.get("date_range"),
                languages=filters.get("languages")
            )
            logger.info(f"Built/loaded FAISS index with {len(segment_ids)} segments")

        # Step 4: Semantic search if query provided
        if query and semantic_search and index is not None:
            # Generate query embedding
            query_embedding = self.embedding_service.encode_query(query)

            if query_embedding is not None:
                query_embedding = np.array([query_embedding], dtype=np.float32)

                # Search
                segment_filter = {seg.id for seg in segments} if len(segments) < len(segment_ids) else None

                search_results = self.embedding_indexer.search(
                    query_embeddings=query_embedding,
                    projects=filters.get("projects"),
                    k=k,
                    threshold=threshold,
                    segment_filter=segment_filter,
                    date_range=filters.get("date_range"),
                    languages=filters.get("languages")
                )

                # Get top-k segment IDs
                if search_results and len(search_results[0]) > 0:
                    top_segment_ids = [seg_id for seg_id, score in search_results[0]]

                    # Re-fetch segments in order
                    segments = self.segment_retriever.fetch_by_ids(top_segment_ids)

                    # Create segment ID to segment mapping
                    seg_map = {seg.id: seg for seg in segments}

                    # Reorder segments to match search results
                    ordered_segments = []
                    for seg_id in top_segment_ids:
                        if seg_id in seg_map:
                            ordered_segments.append(seg_map[seg_id])

                    segments = ordered_segments
                    logger.info(f"Semantic search: filtered to top {len(segments)} results")

        # Return results
        return {
            "segments": segments[:k],  # Limit to k
            "index": index,
            "segment_ids": segment_ids,
            "filters": filters,
            "count": len(segments)
        }

    def retrieve_and_prepare_index(
        self,
        query: Optional[str] = None,
        projects: Optional[List[str]] = None,
        languages: Optional[List[str]] = None,
        time_window_days: int = 30
    ) -> Tuple[List[Segment], Any, List[int], Dict[str, Any]]:
        """
        Retrieve segments and prepare FAISS index (no semantic search yet).

        Useful for preparing data for analysis workflows.

        Args:
            query: Natural language query for parsing filters
            projects: Explicit project filter
            languages: Explicit language filter
            time_window_days: Time window in days

        Returns:
            Tuple of (segments, index, segment_ids, filters)
        """
        result = self.retrieve(
            query=query,
            projects=projects,
            languages=languages,
            time_window_days=time_window_days,
            build_index=True,
            semantic_search=False  # Don't filter by semantic similarity
        )

        return (
            result["segments"],
            result["index"],
            result["segment_ids"],
            result["filters"]
        )

    def count(
        self,
        query: Optional[str] = None,
        projects: Optional[List[str]] = None,
        languages: Optional[List[str]] = None,
        time_window_days: int = 30
    ) -> int:
        """
        Count segments matching query/filters without retrieving them.

        Args:
            query: Natural language query
            projects: Project filter
            languages: Language filter
            time_window_days: Time window in days

        Returns:
            Number of matching segments
        """
        # Parse query if provided
        if query:
            parsed = self.query_parser.parse(query, time_window_days)
            projects = projects or parsed.get("projects")
            languages = languages or parsed.get("languages")
            date_range = parsed.get("date_range")
        else:
            date_range = self.query_parser._time_window_to_date_range(time_window_days)

        # Count
        count = self.segment_retriever.count_by_filter(
            projects=projects,
            languages=languages,
            date_range=date_range,
            must_be_embedded=True
        )

        return count
