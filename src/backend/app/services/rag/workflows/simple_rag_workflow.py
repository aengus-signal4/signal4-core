"""
Simple RAG Workflow
===================

Direct Q&A workflow with two modes:

1. run(): Accept pre-retrieved segments → sample → generate answer
2. run_with_expansion(): Query expansion → semantic search → sample → generate answer
"""

from typing import List, Dict, Any, Optional
from ..analysis_pipeline import AnalysisPipeline

from ....utils.backend_logger import get_logger
logger = get_logger("simple_rag_workflow")


class SimpleRAGWorkflow:
    """
    Simple RAG workflow for direct Q&A over segments.

    Workflow:
    1. Accept pre-retrieved segments (from search)
    2. Sample diverse subset (diversity-weighted)
    3. Generate LLM answer with citations
    4. Return summary + segment IDs

    Example:
        workflow = SimpleRAGWorkflow(llm_service)
        result = await workflow.run(
            query="What is discussed about climate?",
            segments=search_results,
            n_samples=20
        )
        # Returns: {"summary": "...", "segment_ids": [123, 456, ...]}
    """

    def __init__(self, llm_service, db_session=None):
        """
        Initialize workflow.

        Args:
            llm_service: LLMService instance for text generation
            db_session: Optional database session for query expansion + retrieval
        """
        self.llm_service = llm_service
        self.db_session = db_session
        logger.info("SimpleRAGWorkflow initialized")

    async def run(
        self,
        query: str,
        segments: List,  # List of Segment objects
        n_samples: int = 20,
        diversity_weight: float = 0.8,
        model: str = "grok-4-fast-non-reasoning-latest",
        temperature: float = 0.3,
        max_tokens: int = 800,
        generate_quantitative_metrics: bool = False,
        include_baseline_for_centrality: bool = False,
        time_window_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute simple RAG workflow.

        Args:
            query: User's question
            segments: Pre-retrieved segments (from search)
            n_samples: Number of segments to sample
            diversity_weight: Weight for diversity vs centrality (0-1)
            model: LLM model to use
            temperature: Generation temperature
            max_tokens: Max tokens for answer
            generate_quantitative_metrics: Generate quantitative analysis
            include_baseline_for_centrality: Include baseline for centrality calculation
            time_window_days: Time window for baseline (if include_baseline=True)

        Returns:
            Dict with:
                - summary: Generated answer with citations
                - segment_ids: List of segment IDs used
                - segment_count: Total segments provided
                - samples_used: Number of segments sampled
                - quantitative_metrics: Optional quantitative analysis (if requested)
        """
        logger.info(f"Running SimpleRAG: query='{query[:50]}...', {len(segments)} segments, sample={n_samples}")

        if not segments:
            logger.warning("No segments provided to SimpleRAGWorkflow")
            return {
                "summary": None,
                "segment_ids": [],
                "segment_count": 0,
                "samples_used": 0
            }

        # Build pipeline
        pipeline = AnalysisPipeline("simple_rag", llm_service=self.llm_service)

        # Custom step: prepare segments as a single "theme"
        async def prepare_segments(context, **params):
            segs = params.get("segments", [])
            # Wrap segments in a fake theme structure for compatibility
            from ..theme_extractor import Theme

            theme = Theme(
                theme_id="query_response",
                theme_name=f"Response to: {query}",
                segments=segs,
                representative_segments=segs[:5] if len(segs) > 5 else segs,
                keywords=[],
                embedding=None,
                metadata={}
            )

            context["segments"] = segs
            context["themes"] = [theme]
            return context

        # Build pipeline
        pipeline = (
            pipeline
            .custom_step("prepare_segments", prepare_segments, segments=segments)
        )

        # Add quantitative analysis if requested
        if generate_quantitative_metrics:
            if not self.db_session:
                logger.warning("Cannot generate quantitative metrics without db_session")
            else:
                pipeline = pipeline.quantitative_analysis(
                    include_baseline=include_baseline_for_centrality,
                    time_window_days=time_window_days
                )

        pipeline = (
            pipeline
            .select_segments(
                strategy="diversity" if diversity_weight > 0.5 else "balanced",
                n=n_samples
            )
            .generate_summaries(
                template="rag_answer",
                level="theme",
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
        )

        # Execute with streaming, collect final result
        final_context = {}
        async for event in pipeline.execute():
            if event['type'] == 'result':
                final_context.update(event.get('data', {}))
            elif event['type'] == 'complete':
                break

        # Extract results
        summaries = final_context.get("summaries", {}).get("theme", [])
        summary = summaries[0] if summaries else None

        # Extract segment IDs
        segment_ids_map = final_context.get("segment_ids", {}).get("theme", {})
        task_id = "theme_query_response"
        segment_ids = segment_ids_map.get(task_id, [])

        # Extract quantitative metrics if generated
        quantitative_metrics = final_context.get("quantitative_metrics")

        logger.info(f"SimpleRAG complete: generated {len(summary) if summary else 0} chars, used {len(segment_ids)} segments")

        response = {
            "summary": summary,
            "segment_ids": segment_ids,
            "segment_count": len(segments),
            "samples_used": len(segment_ids)
        }

        if quantitative_metrics:
            response["quantitative_metrics"] = quantitative_metrics

        return response

    async def run_with_expansion(
        self,
        query: str,
        expansion_strategy: str = "multi_query",
        k: int = 100,
        n_samples: int = 20,
        diversity_weight: float = 0.8,
        time_window_days: Optional[int] = 30,
        model: str = "grok-4-fast-non-reasoning-latest",
        temperature: float = 0.3,
        max_tokens: int = 800,
        generate_quantitative_metrics: bool = False,
        include_baseline_for_centrality: bool = False,
        **filters
    ) -> Dict[str, Any]:
        """
        Execute simple RAG workflow WITH query expansion and semantic search.

        This method performs the full pipeline:
        1. Expand query into multiple variations (multi_query or query2doc)
        2. Embed all variations
        3. Search with all embeddings (semantic search)
        4. Deduplicate and merge results
        5. Sample diverse subset
        6. Generate LLM answer with citations

        Args:
            query: User's question
            expansion_strategy: Query expansion strategy
                - "multi_query": 10 diverse variations (5 EN + 5 FR) [default]
                - "query2doc": Single pseudo-document
                - "stance_variation": 3-5 pseudo-docs with different stances/perspectives
            k: Max results per query embedding
            n_samples: Number of segments to sample for answer generation
            diversity_weight: Weight for diversity vs centrality (0-1)
            time_window_days: Time window filter (days)
            model: LLM model to use
            temperature: Generation temperature
            max_tokens: Max tokens for answer
            generate_quantitative_metrics: Generate quantitative analysis
            include_baseline_for_centrality: Include baseline for centrality calculation
            **filters: Additional search filters (projects, languages, channels)
                - n_stances: Number of stances (for stance_variation, default 3)

        Returns:
            Dict with:
                - summary: Generated answer with citations
                - segment_ids: List of segment IDs used in summary
                - segment_count: Total segments retrieved
                - samples_used: Number of segments sampled
                - expanded_queries: List of query variations used
                - expansion_strategy: Strategy used
                - search_results: Full search results with metadata (for UI display)
                - quantitative_metrics: Optional quantitative analysis (if requested)

        Requires:
            - db_session must be provided in __init__
        """
        if not self.db_session:
            raise ValueError("db_session required for run_with_expansion(). Pass it to __init__.")

        logger.info(f"Running SimpleRAG with expansion: query='{query[:50]}...', strategy={expansion_strategy}")

        # Build pipeline with query expansion + semantic search
        pipeline = AnalysisPipeline("simple_rag_expansion", llm_service=self.llm_service, db_session=self.db_session)

        # Custom step: prepare segments as a single "theme"
        async def prepare_segments(context, **params):
            segs = context.get("segments", [])
            from ..theme_extractor import Theme

            theme = Theme(
                theme_id="query_response",
                theme_name=f"Response to: {query}",
                segments=segs,
                representative_segments=segs[:5] if len(segs) > 5 else segs,
                keywords=[],
                embedding=None,
                metadata={}
            )

            context["themes"] = [theme]
            return context

        # Build pipeline
        pipeline = (
            pipeline
            .expand_query(query, strategy=expansion_strategy)
            .retrieve_segments_by_search(k=k, time_window_days=time_window_days, **filters)
            .rerank_segments(
                best_per_episode=True,
                time_window_days=time_window_days or 30
            )
            .custom_step("prepare_segments", prepare_segments)
        )

        # Add quantitative analysis if requested
        if generate_quantitative_metrics:
            pipeline = pipeline.quantitative_analysis(
                include_baseline=include_baseline_for_centrality,
                time_window_days=time_window_days
            )

        pipeline = (
            pipeline
            .select_segments(
                strategy="diversity" if diversity_weight > 0.5 else "balanced",
                n=n_samples
            )
            .generate_summaries(
                template="rag_answer",
                level="theme",
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
        )

        # Execute with streaming, collect final result
        final_context = {}
        async for event in pipeline.execute():
            if event['type'] == 'result':
                final_context.update(event.get('data', {}))
            elif event['type'] == 'complete':
                break

        # Extract results
        summaries = final_context.get("summaries", {}).get("theme", [])
        summary = summaries[0] if summaries else None

        # Extract segment IDs
        segment_ids_map = final_context.get("segment_ids", {}).get("theme", {})
        task_id = "theme_query_response"
        segment_ids = segment_ids_map.get(task_id, [])

        # Get expanded queries from context
        expanded_queries = final_context.get("expanded_queries", [])
        segments = final_context.get("segments", [])
        segments_count = len(segments)

        # Extract quantitative metrics if generated
        quantitative_metrics = final_context.get("quantitative_metrics")

        # Convert segments to search results format (with all metadata for UI)
        search_results = []
        for seg in segments:
            search_results.append({
                "segment_id": seg.id,
                "text": seg.text,
                "similarity": getattr(seg, 'similarity_score', 1.0),
                "content_id": seg.content.id if hasattr(seg, 'content') else 0,
                "content_id_string": seg.content.content_id if hasattr(seg, 'content') else "",
                "channel_url": seg.content.channel_url if hasattr(seg, 'content') else "",
                "channel_name": seg.content.channel_name if hasattr(seg, 'content') else "",
                "title": seg.content.title if hasattr(seg, 'content') else "",
                "publish_date": seg.content.publish_date.isoformat() if hasattr(seg, 'content') and seg.content.publish_date else None,
                "start_time": seg.start_time,
                "end_time": seg.end_time,
                "speaker_hashes": getattr(seg, 'speaker_hashes', []),
                "segment_index": getattr(seg, 'segment_index', 0),
                "stitch_version": getattr(seg, 'stitch_version', None)
            })

        logger.info(f"SimpleRAG with expansion complete: {len(expanded_queries)} queries → {segments_count} segments → {len(segment_ids)} samples → {len(summary) if summary else 0} chars")

        response = {
            "summary": summary,
            "segment_ids": segment_ids,
            "segment_count": segments_count,
            "samples_used": len(segment_ids),
            "expanded_queries": expanded_queries,
            "expansion_strategy": expansion_strategy,
            "search_results": search_results  # NEW: Full search results with metadata
        }

        if quantitative_metrics:
            response["quantitative_metrics"] = quantitative_metrics

        return response

    async def run_with_expansion_stream(
        self,
        query: str,
        expansion_strategy: str = "multi_query",
        k: int = 100,
        n_samples: int = 20,
        diversity_weight: float = 0.8,
        time_window_days: Optional[int] = 30,
        model: str = "grok-4-fast-non-reasoning-latest",
        temperature: float = 0.3,
        max_tokens: int = 800,
        generate_quantitative_metrics: bool = False,
        include_baseline_for_centrality: bool = False,
        **filters
    ):
        """
        Streaming version of run_with_expansion().

        Yields progress events as the pipeline executes each step:
        - step_start: Step beginning
        - step_progress: Progress within step (e.g., generating summaries)
        - step_complete: Step finished with results
        - complete: Final result

        Same args as run_with_expansion(). See that method for details.

        Yields:
            Dict events with structure:
                {"type": "step_start", "step": "expand_query", ...}
                {"type": "step_progress", "step": "generate_summaries", "progress": 5, "total": 20}
                {"type": "step_complete", "step": "expand_query", "data": {...}}
                {"type": "complete", "data": {...}}

        Example:
            async for event in workflow.run_with_expansion_stream(query="..."):
                print(event)
        """
        if not self.db_session:
            raise ValueError("db_session required for run_with_expansion_stream(). Pass it to __init__.")

        logger.info(f"Running SimpleRAG with expansion (streaming): query='{query[:50]}...', strategy={expansion_strategy}")

        # Build pipeline (same as run_with_expansion)
        pipeline = AnalysisPipeline("simple_rag_expansion_stream", llm_service=self.llm_service, db_session=self.db_session)

        # Custom step: prepare segments as a single "theme"
        async def prepare_segments(context, **params):
            segs = context.get("segments", [])
            from ..theme_extractor import Theme

            theme = Theme(
                theme_id="query_response",
                theme_name=f"Response to: {query}",
                segments=segs,
                representative_segments=segs[:5] if len(segs) > 5 else segs,
                keywords=[],
                embedding=None,
                metadata={}
            )

            context["themes"] = [theme]
            return context

        # Build pipeline
        pipeline = (
            pipeline
            .expand_query(query, strategy=expansion_strategy)
            .retrieve_segments_by_search(k=k, time_window_days=time_window_days, **filters)
            .rerank_segments(
                best_per_episode=True,
                time_window_days=time_window_days or 30
            )
            .custom_step("prepare_segments", prepare_segments)
        )

        # Add quantitative analysis if requested
        if generate_quantitative_metrics:
            pipeline = pipeline.quantitative_analysis(
                include_baseline=include_baseline_for_centrality,
                time_window_days=time_window_days
            )

        pipeline = (
            pipeline
            .select_segments(
                strategy="diversity" if diversity_weight > 0.5 else "balanced",
                n=n_samples
            )
            .generate_summaries(
                template="rag_answer",
                level="theme",
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
        )

        # Execute with streaming - AnalysisPipeline handles all event generation
        async for event in pipeline.execute():
            yield event

        logger.info(f"SimpleRAG with expansion (streaming) complete")
