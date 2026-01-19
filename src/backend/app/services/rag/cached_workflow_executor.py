"""
Cached Workflow Executor
========================

Orchestrates RAG workflow execution with multi-level caching.

This module handles all cache scenarios:
1. Landing page cache (no query workflows)
2. Full cache hit (workflow + expand_query cached)
3. Partial cache hit (expand_query only)
4. Cache miss (full pipeline execution)

The executor yields SSE events for streaming responses.

Architecture:
- Router calls executor with request parameters
- Executor checks caches, runs pipeline, caches results
- Executor yields events; router formats as SSE
"""

import json
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Any, AsyncGenerator, Dict, List, Optional

from ..workflow_cache_service import (
    ExpandQueryCache,
    WorkflowResultCache,
    LandingPageCache,
    check_caches_sequential
)
from .event_serializer import clean_event_for_json
from .analysis_pipeline import AnalysisPipeline
from .step_registry import build_pipeline_from_steps

from src.utils.logger import setup_worker_logger
logger = setup_worker_logger("backend.cached_workflow_executor")


class CachedWorkflowExecutor:
    """
    Executes RAG workflows with intelligent caching.

    Handles cache checking, pipeline execution, and result caching
    while yielding SSE events for streaming responses.
    """

    def __init__(
        self,
        dashboard_id: str,
        config: Any,
        llm_service: Any,
        embedding_service: Any,
        db_session: Any
    ):
        """
        Initialize the executor.

        Args:
            dashboard_id: Dashboard identifier
            config: Dashboard configuration
            llm_service: LLM service instance
            embedding_service: Embedding service instance
            db_session: Database session
        """
        self.dashboard_id = dashboard_id
        self.config = config
        self.llm_service = llm_service
        self.embedding_service = embedding_service
        self.db_session = db_session

    async def execute(
        self,
        query: Optional[str],
        workflow: Optional[str],
        steps: List[Dict[str, Any]],
        global_filters: Dict[str, Any],
        verbose: bool = False
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute workflow with caching, yielding SSE events.

        Args:
            query: User query (None for landing page)
            workflow: Workflow name (e.g., "simple_rag", "landing_page_overview")
            steps: Pipeline steps to execute
            global_filters: Filters (time_window_days, projects, languages, channels)
            verbose: Whether to include verbose logging

        Yields:
            Event dicts for SSE streaming
        """
        # Landing page workflow (no query)
        if workflow == "landing_page_overview":
            async for event in self._execute_landing_page(steps, global_filters, verbose):
                yield event
            return

        # Query-based workflows
        if query:
            async for event in self._execute_query_workflow(
                query, workflow, steps, global_filters, verbose
            ):
                yield event

    async def _execute_landing_page(
        self,
        steps: List[Dict[str, Any]],
        global_filters: Dict[str, Any],
        verbose: bool
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute landing page workflow with caching.

        Args:
            steps: Pipeline steps
            global_filters: Filters
            verbose: Verbose mode

        Yields:
            SSE events
        """
        landing_cache = LandingPageCache(self.dashboard_id)
        cached_result = landing_cache.get_sync(
            time_window=global_filters["time_window_days"],
            projects=global_filters.get("projects"),
            languages=global_filters.get("languages")
        )

        if cached_result:
            cache_age_hours = cached_result.get('_cache_age_hours', 0)
            logger.info(
                f"[{self.dashboard_id}] [CACHE] Landing page cached "
                f"(age={cache_age_hours:.1f}h, ttl=24h)"
            )

            yield {'type': 'cache_hit', 'level': 'landing_page', 'cache_age_hours': round(cache_age_hours, 1)}

            complete_event = {
                'type': 'complete',
                'data': cached_result,
                'cache_hit': True,
                'cache_age_hours': round(cache_age_hours, 1)
            }
            yield clean_event_for_json(complete_event)

            logger.info(f"[{self.dashboard_id}] Complete (landing page cached)")
            return

        # Cache miss - run full pipeline
        async for event in self._run_pipeline(
            query=None,
            steps=steps,
            global_filters=global_filters,
            initial_context=None,
            verbose=verbose
        ):
            yield event

    async def _execute_query_workflow(
        self,
        query: str,
        workflow: Optional[str],
        steps: List[Dict[str, Any]],
        global_filters: Dict[str, Any],
        verbose: bool
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute query-based workflow with multi-level caching.

        Args:
            query: User query
            workflow: Workflow name
            steps: Pipeline steps
            global_filters: Filters
            verbose: Verbose mode

        Yields:
            SSE events
        """
        # Check caches
        cache_results = check_caches_sequential(
            query=query,
            time_window_days=global_filters["time_window_days"],
            projects=global_filters.get("projects"),
            languages=global_filters.get("languages"),
            dashboard_id=self.dashboard_id
        )

        # Scenario A: Full cache hit
        if cache_results['workflow'] and cache_results['expand_query']:
            async for event in self._handle_full_cache_hit(
                query, workflow, cache_results, global_filters
            ):
                yield event
            return

        # Scenario B/C: Partial or no cache
        initial_context = None
        skip_expand_query = False

        if cache_results.get('expand_query'):
            initial_context, skip_expand_query = self._build_initial_context(cache_results['expand_query'])
            if skip_expand_query:
                yield {'type': 'cache_hit', 'level': 'partial'}
        else:
            logger.info(f"[{self.dashboard_id}] No cached results")
            yield {'type': 'cache_miss'}

        # Filter steps if we have cached expand_query
        pipeline_steps = steps
        if skip_expand_query:
            pipeline_steps = [s for s in steps if s.get("step") != "expand_query"]
            logger.debug(f"[{self.dashboard_id}] Skipping expand_query step (using cached data)")

        # Run pipeline
        async for event in self._run_pipeline(
            query=query,
            steps=pipeline_steps,
            global_filters=global_filters,
            initial_context=initial_context,
            verbose=verbose
        ):
            yield event

    async def _handle_full_cache_hit(
        self,
        query: str,
        workflow: Optional[str],
        cache_results: Dict[str, Any],
        global_filters: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Handle full cache hit scenario.

        For quick_summary: return cached result immediately.
        For other workflows: refresh retrieval for fresh quantitative stats.

        Args:
            query: User query
            workflow: Workflow name
            cache_results: Cache lookup results
            global_filters: Filters

        Yields:
            SSE events
        """
        workflow_age = cache_results['workflow'].get('_cache_age_seconds', 0) / 60
        expand_age = cache_results['expand_query'].get('_cache_age_seconds', 0) / 60

        logger.info(
            f"[{self.dashboard_id}] [CACHE] Workflow (age={workflow_age:.1f}m) + "
            f"expand_query (age={expand_age:.1f}m) both cached"
        )

        yield {'type': 'cache_hit', 'level': 'full'}

        # Quick summary: instant return
        if workflow == "quick_summary":
            logger.info(f"[{self.dashboard_id}] [CACHE] Returning instant cached result for quick_summary")
            async for event in self._yield_cached_results(cache_results['workflow']):
                yield event
            return

        # Other workflows: refresh retrieval for fresh stats
        async for event in self._refresh_retrieval_with_cached_embeddings(
            cache_results, global_filters
        ):
            yield event

    async def _yield_cached_results(
        self,
        workflow_cache: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Yield cached workflow results as SSE events.

        Args:
            workflow_cache: Cached workflow data

        Yields:
            SSE events
        """
        segments_event = {
            'type': 'result',
            'step': 'select_segments',
            'data': {
                'selected_segments': workflow_cache['selected_segments'],
                'selection_strategy': 'cached'
            }
        }
        yield clean_event_for_json(segments_event)

        summary_event = {
            'type': 'result',
            'step': 'generate_summaries',
            'data': {
                'summaries': workflow_cache['summaries'],
                'segment_ids': workflow_cache.get('selected_segment_ids', [])
            }
        }
        yield clean_event_for_json(summary_event)

        yield {'type': 'complete', 'message': 'Analysis complete (cached)'}
        logger.info(f"[{self.dashboard_id}] Complete (instant cache hit)")

    async def _refresh_retrieval_with_cached_embeddings(
        self,
        cache_results: Dict[str, Any],
        global_filters: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Re-run retrieval with cached embeddings for fresh quantitative stats.

        This saves expensive embedding generation (~2s) and summary generation (~12s)
        while providing up-to-date statistics.

        Args:
            cache_results: Cache lookup results with embeddings
            global_filters: Filters

        Yields:
            SSE events
        """
        from ..pgvector_search_service import PgVectorSearchService
        from .segment_retriever import SegmentRetriever
        from .quantitative_analyzer import QuantitativeAnalyzer

        # Get cached embeddings
        cached_embeddings_list = cache_results['expand_query'].get('query_embeddings', [])

        if not cached_embeddings_list:
            # Fallback: try to generate from cached queries
            expanded_queries = cache_results['expand_query'].get('expanded_queries', [])

            if not expanded_queries:
                logger.warning(f"[{self.dashboard_id}] No cached queries, returning cached results only")
                async for event in self._yield_cached_results(cache_results['workflow']):
                    yield event
                return

            try:
                embeddings = await self.embedding_service.encode_queries(expanded_queries)
                logger.info(f"[{self.dashboard_id}] Generated {len(embeddings)} embeddings from cached queries")
            except Exception as e:
                logger.error(f"[{self.dashboard_id}] Failed to generate embeddings: {e}", exc_info=True)
                async for event in self._yield_cached_results(cache_results['workflow']):
                    yield event
                return
        else:
            embeddings = [np.array(emb, dtype=np.float32) for emb in cached_embeddings_list]
            logger.info(f"[{self.dashboard_id}] [1/3] Using {len(embeddings)} cached embeddings for retrieval")

        # Initialize search service
        search_service = PgVectorSearchService(
            dashboard_id=self.dashboard_id,
            config=self.config
        )

        # Build search filters
        search_filters = {}
        for key in ["projects", "languages", "channels"]:
            if global_filters.get(key):
                search_filters[key] = global_filters[key]

        # Unified batch search
        unique_segments = search_service.batch_search_unified(
            query_embeddings=embeddings,
            time_window_days=global_filters["time_window_days"],
            threshold=0.4,
            **search_filters
        )

        logger.info(f"[{self.dashboard_id}] [2/3] Retrieved {len(unique_segments)} unique segments (fresh)")

        yield {'type': 'progress', 'message': f'Retrieved {len(unique_segments)} segments'}

        # Run fresh quantitative analysis
        retriever = SegmentRetriever(self.db_session)
        analyzer = QuantitativeAnalyzer(db_session=self.db_session)

        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=global_filters["time_window_days"])

        baseline_filters = {
            "date_range": (start_date, end_date),
            "must_be_stitched": True,
            "must_be_embedded": True
        }
        for key in ["projects", "languages", "channels"]:
            if key in global_filters:
                baseline_filters[key] = global_filters[key]

        baseline_segments = retriever.get_baseline_stats(**baseline_filters)

        segment_ids = [seg['segment_id'] for seg in unique_segments]
        segment_objects = retriever.fetch_by_ids(segment_ids)

        quantitative_metrics = analyzer.analyze(
            segments=segment_objects,
            baseline_segments=baseline_segments,
            time_window_days=global_filters["time_window_days"]
        )

        logger.info(
            f"[{self.dashboard_id}] [3/3] quantitative_analysis: "
            f"{quantitative_metrics['total_segments']} segments, "
            f"centrality={quantitative_metrics.get('discourse_centrality', {}).get('score', 0):.2f} (fresh)"
        )

        yield {'type': 'progress', 'message': 'Quantitative analysis complete'}

        # Yield results in correct order for frontend
        retrieve_event = {
            'type': 'result',
            'step': 'retrieve_segments_by_search',
            'data': {
                'segment_count': len(unique_segments),
                'segments': unique_segments
            }
        }
        yield clean_event_for_json(retrieve_event)

        segments_event = {
            'type': 'result',
            'step': 'select_segments',
            'data': {
                'selected_segments': cache_results['workflow']['selected_segments'],
                'selection_strategy': 'cached'
            }
        }
        yield clean_event_for_json(segments_event)

        quant_event = {
            'type': 'result',
            'step': 'quantitative_analysis',
            'data': {
                'quantitative_metrics': quantitative_metrics
            }
        }
        yield clean_event_for_json(quant_event)

        summary_event = {
            'type': 'result',
            'step': 'generate_summaries',
            'data': {
                'summaries': cache_results['workflow']['summaries'],
                'segment_ids': cache_results['workflow'].get('selected_segment_ids', [])
            }
        }
        yield clean_event_for_json(summary_event)

        yield {'type': 'complete', 'message': 'Analysis complete'}
        logger.info(f"[{self.dashboard_id}] Complete (saved ~14s: 2s embed + 12s summary)")

    def _build_initial_context(
        self,
        expand_cache: Dict[str, Any]
    ) -> tuple[Optional[Dict[str, Any]], bool]:
        """
        Build initial context from cached expand_query data.

        Args:
            expand_cache: Cached expand_query result

        Returns:
            Tuple of (initial_context, skip_expand_query)
        """
        expand_age = expand_cache.get('_cache_age_seconds', 0) / 60
        cached_queries = expand_cache.get('expanded_queries', [])
        cached_embeddings = expand_cache.get('query_embeddings', [])

        if not cached_queries:
            return None, False

        initial_context = {
            'expanded_queries': cached_queries,
            'keywords': expand_cache.get('keywords', []),
            'expansion_strategy': 'multi_query'
        }

        if cached_embeddings:
            initial_context['query_embeddings'] = [
                np.array(emb, dtype=np.float32) for emb in cached_embeddings
            ]
            initial_context['query_embeddings_cached'] = cached_embeddings
            logger.info(
                f"[{self.dashboard_id}] [CACHE] Using cached expand_query "
                f"(age={expand_age:.1f}m, {len(cached_queries)} queries, {len(cached_embeddings)} embeddings)"
            )
        else:
            logger.info(
                f"[{self.dashboard_id}] [CACHE] Using cached expand_query "
                f"(age={expand_age:.1f}m, {len(cached_queries)} queries, generating embeddings)"
            )

        return initial_context, True

    async def _run_pipeline(
        self,
        query: Optional[str],
        steps: List[Dict[str, Any]],
        global_filters: Dict[str, Any],
        initial_context: Optional[Dict[str, Any]],
        verbose: bool
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Run the analysis pipeline and cache results.

        Args:
            query: User query (None for landing page)
            steps: Pipeline steps
            global_filters: Filters
            initial_context: Initial context from cache
            verbose: Verbose mode

        Yields:
            SSE events
        """
        pipeline = AnalysisPipeline(
            name=f"analysis_pipeline",
            llm_service=self.llm_service,
            embedding_service=self.embedding_service,
            db_session=self.db_session,
            dashboard_id=self.dashboard_id,
            config=self.config
        )

        # Add steps dynamically
        pipeline = build_pipeline_from_steps(
            pipeline,
            query=query,
            steps=steps,
            global_filters=global_filters
        )

        # Execute pipeline
        step_num = 0
        total_steps = len(steps)
        final_context = {}

        if initial_context:
            final_context.update(initial_context)

        async for event in pipeline.execute(verbose=verbose, initial_context=initial_context):
            if event.get("type") == "result":
                step_num += 1
                step_name = event.get("step", "unknown")
                data = event.get("data", {})

                self._log_step_completion(step_num, total_steps, step_name, data)
                final_context.update(data)

            yield clean_event_for_json(event)

        # Cache results after pipeline completion
        await self._cache_results(query, global_filters, final_context)

        logger.info(f"[{self.dashboard_id}] Complete")

    async def _cache_results(
        self,
        query: Optional[str],
        global_filters: Dict[str, Any],
        final_context: Dict[str, Any]
    ) -> None:
        """
        Cache pipeline results for future requests.

        Args:
            query: User query
            global_filters: Filters
            final_context: Final pipeline context
        """
        try:
            # Landing page cache
            if not query and 'themes' in final_context:
                landing_cache = LandingPageCache(self.dashboard_id)
                cache_data = {
                    'themes': final_context.get('themes', []),
                    'corpus_stats': final_context.get('corpus_stats', {}),
                    'themes_analysis': final_context.get('themes_analysis', {}),
                    'summaries': final_context.get('summaries', []),
                    'time_window_days': global_filters["time_window_days"],
                    'projects': global_filters.get("projects"),
                    'languages': global_filters.get("languages")
                }
                landing_cache.put(
                    time_window=global_filters["time_window_days"],
                    projects=global_filters.get("projects"),
                    languages=global_filters.get("languages"),
                    data=cache_data
                )
                logger.info(f"[{self.dashboard_id}] Cached landing page result (ttl=24h)")
                return

            # Expand query cache
            if query and 'expanded_queries' in final_context:
                expand_cache = ExpandQueryCache(self.dashboard_id)
                expand_cache.put(
                    query=query,
                    data={
                        'expanded_queries': final_context['expanded_queries'],
                        'keywords': final_context.get('keywords', []),
                        'query_embeddings': final_context.get('query_embeddings_cached', [])
                    }
                )
                logger.info(
                    f"[{self.dashboard_id}] Cached expand_query result "
                    f"(with {len(final_context.get('query_embeddings_cached', []))} embeddings)"
                )

            # Workflow result cache
            if query and 'selected_segments' in final_context and 'summaries' in final_context:
                workflow_cache = WorkflowResultCache(self.dashboard_id)

                selected_segments = final_context['selected_segments']
                selected_segment_ids = self._extract_segment_ids(selected_segments)

                all_segments = final_context.get('segments', [])
                all_segment_ids = self._extract_segment_ids(all_segments)

                workflow_cache.put(
                    query=query,
                    time_window=global_filters["time_window_days"],
                    projects=global_filters.get("projects"),
                    languages=global_filters.get("languages"),
                    data={
                        'selected_segments': selected_segments,
                        'summaries': final_context['summaries'],
                        'selected_segment_ids': selected_segment_ids,
                        'all_segment_ids': all_segment_ids
                    }
                )
                ttl = workflow_cache.get_ttl_for_time_window(global_filters['time_window_days'])
                logger.info(f"[{self.dashboard_id}] Cached workflow result (ttl={ttl}h)")

        except Exception as cache_error:
            logger.error(f"[{self.dashboard_id}] Failed to cache results: {cache_error}", exc_info=True)

    def _extract_segment_ids(self, segments: List[Any]) -> List[str]:
        """Extract segment IDs from segment list."""
        if not segments:
            return []
        if isinstance(segments[0], dict):
            return [seg.get('segment_id') for seg in segments if seg.get('segment_id')]
        return [seg.id for seg in segments if hasattr(seg, 'id')]

    def _log_step_completion(
        self,
        step_num: int,
        total_steps: int,
        step_name: str,
        data: Dict[str, Any]
    ) -> None:
        """Log step completion with relevant metrics."""
        if step_name == "retrieve_segments_by_search":
            segment_count = len(data.get("segments", []))
            logger.info(f"[{self.dashboard_id}] [{step_num}/{total_steps}] {step_name}: {segment_count} segments retrieved")

        elif step_name == "quantitative_analysis":
            metrics = data.get("quantitative_metrics") or {}
            seg_count = metrics.get("total_segments", 0)
            centrality = (metrics.get("discourse_centrality") or {}).get("score")
            if centrality is not None:
                logger.info(f"[{self.dashboard_id}] [{step_num}/{total_steps}] {step_name}: {seg_count} segments, centrality={centrality:.2f}")
            else:
                logger.info(f"[{self.dashboard_id}] [{step_num}/{total_steps}] {step_name}: {seg_count} segments")

        elif step_name == "select_segments":
            selected_count = len(data.get("selected_segments", []))
            logger.info(f"[{self.dashboard_id}] [{step_num}/{total_steps}] {step_name}: {selected_count} segments selected")

        elif step_name == "expand_query":
            expanded_count = len(data.get("expanded_queries", []))
            logger.info(f"[{self.dashboard_id}] [{step_num}/{total_steps}] {step_name}: {expanded_count} variations")

        elif step_name == "generate_summaries":
            summaries = data.get("summaries", {})
            summary_count = sum(len(v) if isinstance(v, list) else 1 for v in summaries.values())
            logger.info(f"[{self.dashboard_id}] [{step_num}/{total_steps}] {step_name}: {summary_count} summaries generated")

        elif step_name == "quick_cluster_check":
            self._log_cluster_check(step_num, total_steps, step_name, data)

        else:
            logger.info(f"[{self.dashboard_id}] [{step_num}/{total_steps}] {step_name}: complete")

    def _log_cluster_check(
        self,
        step_num: int,
        total_steps: int,
        step_name: str,
        data: Dict[str, Any]
    ) -> None:
        """Log quick_cluster_check step with detailed validation info."""
        validation = data.get("cluster_validation", {})
        num_clusters = len(data.get("themes", []))
        silhouette = validation.get("silhouette_score")
        fallback_used = data.get("fallback_used", False)

        if validation.get("skipped"):
            reason = validation.get("reason", "unknown")
            if reason == "single_cluster_core_fringe":
                logger.info(f"[{self.dashboard_id}] [{step_num}/{total_steps}] {step_name}: 1 cluster detected (core+fringe structure)")
            else:
                sil_str = f", silhouette={silhouette:.3f}" if silhouette is not None else ""
                logger.info(f"[{self.dashboard_id}] [{step_num}/{total_steps}] {step_name}: skipped ({reason}{sil_str})")

        elif data.get("has_subclusters"):
            sil_str = f", silhouette={silhouette:.3f}" if silhouette is not None else ""
            method_str = " (k-means fallback)" if fallback_used else ""
            logger.info(f"[{self.dashboard_id}] [{step_num}/{total_steps}] {step_name}: {num_clusters} clusters detected{sil_str}{method_str}")

        else:
            if num_clusters == 0:
                sil_str = f", silhouette={silhouette:.3f}" if silhouette is not None else ""
                fallback_str = " (k-means fallback failed)" if fallback_used else " (all noise)"
                logger.info(f"[{self.dashboard_id}] [{step_num}/{total_steps}] {step_name}: no clusters found{fallback_str}{sil_str}")
            else:
                min_required = validation.get("min_silhouette_score", 0.15)
                sil_str = f"silhouette={silhouette:.3f} < {min_required:.2f}" if silhouette is not None else "validation failed"
                logger.info(f"[{self.dashboard_id}] [{step_num}/{total_steps}] {step_name}: {num_clusters} clusters rejected ({sil_str})")
