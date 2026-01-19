"""
Analysis Router
===============

Single unified endpoint for all analysis workflows with declarative pipeline configuration.

Endpoints:
- POST /api/analysis/stream - Streaming SSE (primary, always recommended)
- POST /api/analysis - Batch (convenience wrapper)
- GET /api/analysis/steps - Discover available steps
- GET /api/analysis/workflows - List predefined workflows

Architecture:
- Router: Thin HTTP layer (validate, call pipeline, return)
- Workflows: Predefined step sequences (convenience)
- Pipeline: AnalysisPipeline with dynamic step building
- Steps: Individual components (expand_query, retrieve_segments, etc.)
"""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from ..models.requests import AnalysisRequest
from ..middleware.api_key_auth import validate_project_access
from ..services.rag.analysis_pipeline import AnalysisPipeline
from ..services.rag.step_registry import (
    list_steps,
    build_pipeline_from_steps,
    get_step_metadata
)
from ..config.workflows import (
    get_workflow,
    list_workflows,
    apply_config_overrides
)
from ..services.llm_service import LLMService
from ..config.dashboard_config import load_dashboard_config
from ..database.connection import SessionLocal
import time
import logging
import json

from ..utils.backend_logger import get_logger
logger = get_logger("analysis_router")

router = APIRouter(prefix="/api/analysis", tags=["analysis", "workflows"])


# ============================================================================
# Helper Functions
# ============================================================================

def _clean_event_for_json(event: dict) -> dict:
    """
    Clean event data for JSON serialization by removing non-serializable objects.

    Removes:
    - Embeddings (numpy arrays, _embedding fields)
    - Other non-JSON-serializable data

    Note: Segment objects are already converted to dicts in the pipeline
    """
    import numpy as np

    # Keys to skip entirely
    SKIP_KEYS = {"_embedding", "embedding", "embeddings", "query_embeddings", "query_embedding", "query_embeddings_cached"}

    def clean_value(v):
        """Recursively clean a value."""
        if v is None:
            return None
        elif isinstance(v, np.ndarray):
            return None  # Remove numpy arrays entirely
        elif isinstance(v, dict):
            cleaned_dict = {}
            for k, val in v.items():
                if k in SKIP_KEYS:
                    continue  # Skip embedding keys
                cleaned_val = clean_value(val)
                if cleaned_val is not None or not isinstance(val, np.ndarray):
                    # Keep None values only if original wasn't a numpy array
                    if not isinstance(val, np.ndarray):
                        cleaned_dict[k] = cleaned_val
            return cleaned_dict
        elif isinstance(v, list):
            return [clean_value(item) for item in v if not isinstance(item, np.ndarray)]
        elif isinstance(v, (int, float, str, bool)):
            return v
        else:
            # For other types, try to convert or skip
            try:
                # Check if it's numpy scalar
                if hasattr(v, 'item'):
                    return v.item()
                return v
            except:
                return str(v)  # Fallback to string

    cleaned = {}
    for key, value in event.items():
        if key == "data":
            cleaned["data"] = clean_value(value)
        else:
            cleaned[key] = value

    return cleaned


# ============================================================================
# Discovery Endpoints
# ============================================================================

@router.get("/steps")
async def get_available_steps():
    """
    Discover available pipeline steps.

    Returns step metadata including parameters and descriptions.
    Use this to understand what steps you can use in custom pipelines.

    Returns:
        List of step definitions with parameters
    """
    return {"steps": list_steps()}


@router.get("/workflows")
async def get_available_workflows():
    """
    List predefined workflow shortcuts.

    Workflows are convenient shorthand for common step sequences.
    You can reference them by name: {"workflow": "simple_rag"}

    Returns:
        Dict mapping workflow names to descriptions
    """
    return {"workflows": list_workflows()}


# ============================================================================
# Analysis Endpoints (Streaming Primary)
# ============================================================================

@router.post("/stream")
async def analyze_stream(request: AnalysisRequest, http_request: Request):
    """
    Streaming analysis with real-time progress updates (SSE).

    **This is the primary endpoint** - always use streaming when possible
    for better UX with progress updates.

    Two usage modes:

    1. Workflow shortcut (convenience):
    ```json
    {
      "query": "Pierre Poilievre",
      "dashboard_id": "cprmv-practitioner",
      "workflow": "simple_rag"
    }
    ```

    2. Custom pipeline (advanced):
    ```json
    {
      "query": "carbon tax",
      "dashboard_id": "cprmv-practitioner",
      "pipeline": [
        {"step": "expand_query", "config": {"strategy": "multi_query"}},
        {"step": "retrieve_segments", "config": {"k": 200}},
        {"step": "quantitative_analysis", "config": {}},
        {"step": "select_segments", "config": {"n": 20}},
        {"step": "generate_summary", "config": {}}
      ]
    }
    ```

    Event types (Server-Sent Events):
    - `step_start`: {"type": "step_start", "step": "expand_query"}
    - `step_progress`: {"type": "step_progress", "step": "generate_summary", "progress": 5, "total": 20}
    - `step_complete`: {"type": "step_complete", "step": "expand_query", "data": {...}}
    - `complete`: {"type": "complete", "data": {...}}
    - `error`: {"type": "error", "error": "..."}

    Frontend usage:
    ```javascript
    const eventSource = new EventSource('/api/analysis/stream', {
        method: 'POST',
        body: JSON.stringify({query: "...", workflow: "simple_rag"})
    });
    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === 'step_progress') {
            updateProgressBar(data.progress);
        }
    };
    ```

    Returns:
        StreamingResponse with Server-Sent Events
    """
    # Validate project access before starting the generator
    validate_project_access(http_request, request.projects or [])

    async def event_generator():
        db_session = None
        try:
            # Validate request: must have either workflow or pipeline
            if not request.workflow and not request.pipeline:
                error_event = json.dumps({
                    "type": "error",
                    "error": "Must specify either 'workflow' or 'pipeline'"
                })
                yield f"data: {error_event}\n\n"
                return

            # Load dashboard config
            config = load_dashboard_config(request.dashboard_id)
            llm_service = LLMService(config, request.dashboard_id)

            # Initialize EmbeddingService for query embeddings
            from ..services.embedding_service import EmbeddingService
            embedding_service = EmbeddingService(config, request.dashboard_id)

            query_preview = f"'{request.query[:60]}...'" if request.query else "(no query - discovery mode)"
            logger.info(
                f"[{request.dashboard_id}] Request: {query_preview} "
                f"[workflow={request.workflow or 'custom'}]"
            )

            # Get pipeline steps
            if request.workflow:
                # Mode 1: Workflow shortcut
                try:
                    steps = get_workflow(request.workflow)

                    # Apply config overrides if provided
                    if request.config_overrides:
                        steps = apply_config_overrides(steps, request.config_overrides)
                except ValueError as e:
                    error_event = json.dumps({
                        "type": "error",
                        "error": str(e)
                    })
                    yield f"data: {error_event}\n\n"
                    return
            else:
                # Mode 2: Custom pipeline
                steps = [
                    {"step": s.step, "config": s.config}
                    for s in request.pipeline
                ]

            # Build global filters
            global_filters = {}
            # Set time_window_days with default of 7 days (uses faster 7d cache table)
            global_filters["time_window_days"] = request.time_window_days or 7

            # Initialize projects and languages to ensure consistent cache keys
            global_filters["projects"] = None
            global_filters["languages"] = None
            global_filters["channels"] = None

            if request.projects:
                global_filters["projects"] = request.projects
            if request.languages:
                global_filters["languages"] = request.languages
            if request.channels:
                global_filters["channels"] = request.channels

            # Also load from dashboard config (use allowed_projects for multi-project dashboards)
            if global_filters["projects"] is None:
                if hasattr(config, 'allowed_projects') and config.allowed_projects:
                    global_filters["projects"] = config.allowed_projects
                elif hasattr(config, 'project') and config.project:
                    global_filters["projects"] = [config.project]
            if hasattr(config, 'languages'):
                if global_filters["languages"] is None:
                    global_filters["languages"] = config.languages

            # ================================================================
            # LANDING PAGE CACHE CHECK (for workflows without query)
            # ================================================================
            if request.workflow == "landing_page_overview":
                from ..services.workflow_cache_service import LandingPageCache

                landing_cache = LandingPageCache(request.dashboard_id)
                cached_result = landing_cache.get_sync(
                    time_window=global_filters["time_window_days"],
                    projects=global_filters.get("projects"),
                    languages=global_filters.get("languages")
                )

                if cached_result:
                    cache_age_hours = cached_result.get('_cache_age_hours', 0)
                    logger.info(
                        f"[{request.dashboard_id}] [CACHE] Landing page cached "
                        f"(age={cache_age_hours:.1f}h, ttl=24h)"
                    )

                    # Yield cache hit event with age
                    yield f"data: {json.dumps({'type': 'cache_hit', 'level': 'landing_page', 'cache_age_hours': round(cache_age_hours, 1)})}\n\n"

                    # Return cached result
                    complete_event = {
                        'type': 'complete',
                        'data': cached_result,
                        'cache_hit': True,
                        'cache_age_hours': round(cache_age_hours, 1)
                    }
                    cleaned_event = _clean_event_for_json(complete_event)
                    yield f"data: {json.dumps(cleaned_event)}\n\n"

                    logger.info(f"[{request.dashboard_id}] Complete (landing page cached)")
                    return

            # ================================================================
            # SEQUENTIAL CACHE CHECK (thread-safe for embeddings + psycopg2)
            # Only for query-based workflows
            # ================================================================
            cache_results = {'workflow': None, 'expand_query': None}  # Default for no-query workflows

            if request.query:
                from ..services.workflow_cache_service import check_caches_sequential
                from ..services.rag.analysis_pipeline import compute_fresh_quantitative_analysis

                cache_results = check_caches_sequential(
                    query=request.query,
                    time_window_days=global_filters["time_window_days"],
                    projects=global_filters.get("projects"),
                    languages=global_filters.get("languages"),
                    dashboard_id=request.dashboard_id
                )

            # ================================================================
            # SCENARIO A: FULL CACHE HIT (workflow + expand_query)
            # ================================================================
            if cache_results['workflow'] and cache_results['expand_query']:
                workflow_age = cache_results['workflow'].get('_cache_age_seconds', 0) / 60
                expand_age = cache_results['expand_query'].get('_cache_age_seconds', 0) / 60

                logger.info(
                    f"[{request.dashboard_id}] [CACHE] Workflow (age={workflow_age:.1f}m) + "
                    f"expand_query (age={expand_age:.1f}m) both cached"
                )

                # Yield cache hit event
                yield f"data: {json.dumps({'type': 'cache_hit', 'level': 'full'})}\n\n"

                # Use cached embeddings to re-run retrieval + quantitative
                # This gives us fresh stats while saving on expensive embedding generation + summary
                db_session = SessionLocal()
                try:
                    from ..services.pgvector_search_service import PgVectorSearchService
                    from ..services.rag.segment_retriever import SegmentRetriever
                    from ..services.rag.quantitative_analyzer import QuantitativeAnalyzer
                    from datetime import datetime, timezone, timedelta
                    import numpy as np

                    # Get cached embeddings
                    cached_embeddings_list = cache_results['expand_query'].get('query_embeddings', [])

                    if not cached_embeddings_list:
                        # Fallback: generate embeddings from cached expanded queries
                        logger.info(f"[{request.dashboard_id}] No cached embeddings, generating from cached queries...")
                        expanded_queries = cache_results['expand_query'].get('expanded_queries', [])

                        if not expanded_queries:
                            # No queries either, can't proceed with retrieval
                            logger.warning(f"[{request.dashboard_id}] No cached queries, returning cached results only")

                            # Send cached results without fresh retrieval
                            segments_event = {
                                'type': 'result',
                                'step': 'select_segments',
                                'data': {
                                    'selected_segments': cache_results['workflow']['selected_segments'],
                                    'selection_strategy': 'cached'
                                }
                            }
                            cleaned_event = _clean_event_for_json(segments_event)
                            yield f"data: {json.dumps(cleaned_event)}\n\n"

                            summary_event = {
                                'type': 'result',
                                'step': 'generate_summaries',
                                'data': {
                                    'summaries': cache_results['workflow']['summaries'],
                                    'segment_ids': cache_results['workflow'].get('selected_segment_ids', [])
                                }
                            }
                            cleaned_event = _clean_event_for_json(summary_event)
                            yield f"data: {json.dumps(cleaned_event)}\n\n"

                            logger.info(f"[{request.dashboard_id}] Complete (cached only)")
                            return

                        # Generate embeddings from cached queries
                        try:
                            embeddings = await embedding_service.encode_queries(expanded_queries)
                            logger.info(f"[{request.dashboard_id}] Generated {len(embeddings)} embeddings from cached queries")
                        except Exception as e:
                            logger.error(f"[{request.dashboard_id}] Failed to generate embeddings: {e}", exc_info=True)
                            # Return cached results only
                            segments_event = {
                                'type': 'result',
                                'step': 'select_segments',
                                'data': {
                                    'selected_segments': cache_results['workflow']['selected_segments'],
                                    'selection_strategy': 'cached'
                                }
                            }
                            cleaned_event = _clean_event_for_json(segments_event)
                            yield f"data: {json.dumps(cleaned_event)}\n\n"

                            summary_event = {
                                'type': 'result',
                                'step': 'generate_summaries',
                                'data': {
                                    'summaries': cache_results['workflow']['summaries'],
                                    'segment_ids': cache_results['workflow'].get('selected_segment_ids', [])
                                }
                            }
                            cleaned_event = _clean_event_for_json(summary_event)
                            yield f"data: {json.dumps(cleaned_event)}\n\n"

                            logger.info(f"[{request.dashboard_id}] Complete (cached only, embedding failed)")
                            return
                    else:
                        # Convert cached embeddings back to numpy arrays
                        embeddings = [np.array(emb, dtype=np.float32) for emb in cached_embeddings_list]
                        logger.info(f"[{request.dashboard_id}] [1/3] Using {len(embeddings)} cached embeddings for retrieval")

                    # Initialize search service
                    search_service = PgVectorSearchService(
                        dashboard_id=request.dashboard_id,
                        config=config
                    )

                    # Build search filters
                    search_filters = {}
                    if global_filters.get("projects"):
                        search_filters["projects"] = global_filters["projects"]
                    if global_filters.get("languages"):
                        search_filters["languages"] = global_filters["languages"]
                    if global_filters.get("channels"):
                        search_filters["channels"] = global_filters["channels"]

                    # Unified batch search - single query with deduplication
                    # Returns ALL segments above threshold (0.4 similarity = 0.6 distance)
                    unique_segments = search_service.batch_search_unified(
                        query_embeddings=embeddings,
                        time_window_days=global_filters["time_window_days"],
                        threshold=0.4,  # Return all segments with similarity >= 0.4
                        **search_filters
                    )

                    logger.info(f"[{request.dashboard_id}] [2/3] Retrieved {len(unique_segments)} unique segments (fresh)")

                    # Send progress event for retrieval completion
                    progress_event = {
                        'type': 'progress',
                        'message': f'Retrieved {len(unique_segments)} segments'
                    }
                    yield f"data: {json.dumps(progress_event)}\n\n"

                    # Run fresh quantitative analysis on retrieved segments
                    retriever = SegmentRetriever(db_session)
                    analyzer = QuantitativeAnalyzer(db_session=db_session)

                    # Get baseline stats
                    end_date = datetime.now(timezone.utc)
                    start_date = end_date - timedelta(days=global_filters["time_window_days"])

                    baseline_filters = {
                        "date_range": (start_date, end_date),
                        "must_be_stitched": True,
                        "must_be_embedded": True
                    }
                    for filter_key in ["projects", "languages", "channels"]:
                        if filter_key in global_filters:
                            baseline_filters[filter_key] = global_filters[filter_key]

                    baseline_segments = retriever.get_baseline_stats(**baseline_filters)

                    # Fetch segment objects for quantitative analysis
                    segment_ids = [seg['segment_id'] for seg in unique_segments]
                    segment_objects = retriever.fetch_by_ids(segment_ids)

                    quantitative_metrics = analyzer.analyze(
                        segments=segment_objects,
                        baseline_segments=baseline_segments,
                        time_window_days=global_filters["time_window_days"]
                    )

                    logger.info(
                        f"[{request.dashboard_id}] [3/3] quantitative_analysis: "
                        f"{quantitative_metrics['total_segments']} segments, "
                        f"centrality={quantitative_metrics.get('discourse_centrality', {}).get('score', 0):.2f} (fresh)"
                    )

                    # Send progress event for quantitative completion
                    progress_event = {
                        'type': 'progress',
                        'message': 'Quantitative analysis complete'
                    }
                    yield f"data: {json.dumps(progress_event)}\n\n"

                    # Send result events in CORRECT ORDER (frontend expects this sequence)
                    # Order matters! Frontend processes these sequentially and updates UI

                    # Event 1: retrieve_segments result (fresh)
                    retrieve_event = {
                        'type': 'result',
                        'step': 'retrieve_segments_by_search',
                        'data': {
                            'segment_count': len(unique_segments),
                            'segments': unique_segments
                        }
                    }
                    cleaned_event = _clean_event_for_json(retrieve_event)
                    yield f"data: {json.dumps(cleaned_event)}\n\n"

                    # Event 2: select_segments result (cached) - MUST come before quantitative!
                    segments_event = {
                        'type': 'result',
                        'step': 'select_segments',
                        'data': {
                            'selected_segments': cache_results['workflow']['selected_segments'],
                            'selection_strategy': 'cached'
                        }
                    }
                    cleaned_event = _clean_event_for_json(segments_event)
                    yield f"data: {json.dumps(cleaned_event)}\n\n"

                    # Event 3: quantitative_analysis result (fresh)
                    quant_event = {
                        'type': 'result',
                        'step': 'quantitative_analysis',
                        'data': {
                            'quantitative_metrics': quantitative_metrics
                        }
                    }
                    cleaned_event = _clean_event_for_json(quant_event)
                    yield f"data: {json.dumps(cleaned_event)}\n\n"

                    # Event 4: generate_summaries result (cached)
                    summary_event = {
                        'type': 'result',
                        'step': 'generate_summaries',
                        'data': {
                            'summaries': cache_results['workflow']['summaries'],
                            'segment_ids': cache_results['workflow'].get('selected_segment_ids', [])
                        }
                    }
                    cleaned_event = _clean_event_for_json(summary_event)
                    yield f"data: {json.dumps(cleaned_event)}\n\n"

                    # Event 5: Complete event (frontend waits for this!)
                    complete_event = {
                        'type': 'complete',
                        'message': 'Analysis complete'
                    }
                    yield f"data: {json.dumps(complete_event)}\n\n"

                    logger.info(f"[{request.dashboard_id}] Complete (saved ~14s: 2s embed + 12s summary)")

                finally:
                    db_session.close()

                return

            # ================================================================
            # SCENARIO B: PARTIAL CACHE HIT (expand_query only)
            # ================================================================
            initial_context = None  # Will be populated if we have cached expand_query
            skip_expand_query = False

            if cache_results.get('expand_query'):
                expand_cache = cache_results['expand_query']
                expand_age = expand_cache.get('_cache_age_seconds', 0) / 60
                cached_queries = expand_cache.get('expanded_queries', [])
                cached_embeddings = expand_cache.get('query_embeddings', [])

                if cached_queries:
                    # Build initial context from cached expand_query data
                    initial_context = {
                        'expanded_queries': cached_queries,
                        'keywords': expand_cache.get('keywords', []),
                        'expansion_strategy': 'multi_query'  # Default
                    }

                    # Add embeddings if available (convert back to numpy arrays)
                    if cached_embeddings:
                        import numpy as np
                        initial_context['query_embeddings'] = [
                            np.array(emb, dtype=np.float32) for emb in cached_embeddings
                        ]
                        initial_context['query_embeddings_cached'] = cached_embeddings
                        skip_expand_query = True
                        logger.info(
                            f"[{request.dashboard_id}] [CACHE] Using cached expand_query "
                            f"(age={expand_age:.1f}m, {len(cached_queries)} queries, {len(cached_embeddings)} embeddings)"
                        )
                    else:
                        # Have queries but no embeddings - still skip LLM call, just need embeddings
                        skip_expand_query = True
                        logger.info(
                            f"[{request.dashboard_id}] [CACHE] Using cached expand_query "
                            f"(age={expand_age:.1f}m, {len(cached_queries)} queries, generating embeddings)"
                        )

                    # Yield partial cache hit event
                    yield f"data: {json.dumps({'type': 'cache_hit', 'level': 'partial'})}\n\n"

            # ================================================================
            # SCENARIO C: FULL CACHE MISS
            # ================================================================
            if not skip_expand_query and not cache_results.get('expand_query'):
                logger.info(f"[{request.dashboard_id}] No cached results")

                # Yield cache miss event
                yield f"data: {json.dumps({'type': 'cache_miss'})}\n\n"

            # Create database session
            db_session = SessionLocal()

            # Filter steps if we have cached expand_query data
            pipeline_steps = steps
            if skip_expand_query:
                # Remove expand_query step since we have cached data
                pipeline_steps = [s for s in steps if s.get("step") != "expand_query"]
                logger.debug(f"[{request.dashboard_id}] Skipping expand_query step (using cached data)")

            # Build pipeline
            pipeline = AnalysisPipeline(
                name=f"analysis_{request.workflow or 'custom'}",
                llm_service=llm_service,
                embedding_service=embedding_service,
                db_session=db_session,
                dashboard_id=request.dashboard_id,
                config=config
            )

            # Add steps dynamically
            try:
                pipeline = build_pipeline_from_steps(
                    pipeline,
                    query=request.query,
                    steps=pipeline_steps,
                    global_filters=global_filters
                )
            except ValueError as e:
                error_event = json.dumps({
                    "type": "error",
                    "error": f"Invalid pipeline configuration: {e}"
                })
                yield f"data: {error_event}\n\n"
                return

            # Execute pipeline with streaming
            step_num = 0
            total_steps = len(pipeline_steps)
            final_context = {}  # Track final pipeline context for caching

            # If we have initial context from cache, also update final_context
            if initial_context:
                final_context.update(initial_context)

            async for event in pipeline.execute_stream(verbose=request.verbose, initial_context=initial_context):
                # Log step completions concisely
                if event.get("type") == "result":
                    step_num += 1
                    step_name = event.get("step", "unknown")

                    # Extract key metrics from step data
                    data = event.get("data", {})
                    if step_name == "retrieve_segments_by_search":
                        segment_count = len(data.get("segments", []))
                        logger.info(f"[{request.dashboard_id}] [{step_num}/{total_steps}] {step_name}: {segment_count} segments retrieved")
                    elif step_name == "quantitative_analysis":
                        metrics = data.get("quantitative_metrics") or {}
                        seg_count = metrics.get("total_segments", 0)
                        centrality = (metrics.get("discourse_centrality") or {}).get("score")
                        if centrality is not None:
                            logger.info(f"[{request.dashboard_id}] [{step_num}/{total_steps}] {step_name}: {seg_count} segments, centrality={centrality:.2f}")
                        else:
                            logger.info(f"[{request.dashboard_id}] [{step_num}/{total_steps}] {step_name}: {seg_count} segments")
                    elif step_name == "select_segments":
                        selected_count = len(data.get("selected_segments", []))
                        logger.info(f"[{request.dashboard_id}] [{step_num}/{total_steps}] {step_name}: {selected_count} segments selected")
                    elif step_name == "expand_query":
                        expanded_count = len(data.get("expanded_queries", []))
                        logger.info(f"[{request.dashboard_id}] [{step_num}/{total_steps}] {step_name}: {expanded_count} variations")
                    elif step_name == "generate_summaries":
                        summaries = data.get("summaries", {})
                        summary_count = sum(len(v) if isinstance(v, list) else 1 for v in summaries.values())
                        logger.info(f"[{request.dashboard_id}] [{step_num}/{total_steps}] {step_name}: {summary_count} summaries generated")
                    elif step_name == "quick_cluster_check":
                        validation = data.get("cluster_validation", {})
                        num_clusters = len(data.get("themes", []))
                        silhouette = validation.get("silhouette_score")
                        fallback_used = data.get("fallback_used", False)

                        if validation.get("skipped"):
                            reason = validation.get("reason", "unknown")
                            # Special case: single_cluster_core_fringe is not really "skipped"
                            if reason == "single_cluster_core_fringe":
                                logger.info(f"[{request.dashboard_id}] [{step_num}/{total_steps}] {step_name}: 1 cluster detected (core+fringe structure)")
                            else:
                                sil_str = f", silhouette={silhouette:.3f}" if silhouette is not None else ""
                                logger.info(f"[{request.dashboard_id}] [{step_num}/{total_steps}] {step_name}: skipped ({reason}{sil_str})")
                        elif data.get("has_subclusters"):
                            sil_str = f", silhouette={silhouette:.3f}" if silhouette is not None else ""
                            method_str = " (k-means fallback)" if fallback_used else ""
                            logger.info(f"[{request.dashboard_id}] [{step_num}/{total_steps}] {step_name}: {num_clusters} clusters detected{sil_str}{method_str}")
                        else:
                            # No valid clusters - show why
                            if num_clusters == 0:
                                sil_str = f", silhouette={silhouette:.3f}" if silhouette is not None else ""
                                fallback_str = " (k-means fallback failed)" if fallback_used else " (all noise)"
                                logger.info(f"[{request.dashboard_id}] [{step_num}/{total_steps}] {step_name}: no clusters found{fallback_str}{sil_str}")
                            else:
                                min_required = validation.get("min_silhouette_score", 0.15)
                                sil_str = f"silhouette={silhouette:.3f} < {min_required:.2f}" if silhouette is not None else "validation failed"
                                logger.info(f"[{request.dashboard_id}] [{step_num}/{total_steps}] {step_name}: {num_clusters} clusters rejected ({sil_str})")
                    else:
                        logger.info(f"[{request.dashboard_id}] [{step_num}/{total_steps}] {step_name}: complete")

                    # Store result data for caching
                    final_context.update(data)

                # Clean event data for JSON serialization (remove embeddings, numpy arrays, etc.)
                cleaned_event = _clean_event_for_json(event)
                event_data = json.dumps(cleaned_event)
                yield f"data: {event_data}\n\n"

            # ================================================================
            # CACHE RESULTS AFTER PIPELINE COMPLETION
            # ================================================================
            try:
                from ..services.workflow_cache_service import ExpandQueryCache, WorkflowResultCache, LandingPageCache

                # Cache landing page result (no query)
                if request.workflow == "landing_page_overview":
                    logger.debug(f"[{request.dashboard_id}] Caching landing page result...")
                    landing_cache = LandingPageCache(request.dashboard_id)

                    # Build cache data from context
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
                    logger.info(f"[{request.dashboard_id}] Cached landing page result (ttl=24h)")

                # Cache expand_query if not already cached (only for query-based workflows)
                if request.query and 'expanded_queries' in final_context and not cache_results.get('expand_query'):
                    logger.debug(f"[{request.dashboard_id}] Caching expand_query result...")
                    expand_cache = ExpandQueryCache(request.dashboard_id)
                    expand_cache.put(
                        query=request.query,
                        data={
                            'expanded_queries': final_context['expanded_queries'],
                            'keywords': final_context.get('keywords', []),
                            'query_embeddings': final_context.get('query_embeddings_cached', [])
                        }
                    )
                    logger.info(f"[{request.dashboard_id}] Cached expand_query result (with {len(final_context.get('query_embeddings_cached', []))} embeddings)")

                # Cache workflow result if not already cached (only for query-based workflows)
                if request.query and 'selected_segments' in final_context and 'summaries' in final_context and not cache_results.get('workflow'):
                    logger.debug(f"[{request.dashboard_id}] Caching workflow result...")
                    workflow_cache = WorkflowResultCache(request.dashboard_id)

                    # Extract selected segment IDs
                    selected_segments = final_context['selected_segments']
                    selected_segment_ids = []
                    if selected_segments:
                        if isinstance(selected_segments[0], dict):
                            selected_segment_ids = [seg.get('segment_id') for seg in selected_segments if seg.get('segment_id')]
                        else:
                            selected_segment_ids = [seg.id for seg in selected_segments if hasattr(seg, 'id')]

                    # Extract ALL retrieved segment IDs (for quantitative analysis)
                    all_segments = final_context.get('segments', [])
                    all_segment_ids = []
                    if all_segments:
                        if isinstance(all_segments[0], dict):
                            all_segment_ids = [seg.get('segment_id') for seg in all_segments if seg.get('segment_id')]
                        else:
                            all_segment_ids = [seg.id for seg in all_segments if hasattr(seg, 'id')]

                    workflow_cache.put(
                        query=request.query,
                        time_window=global_filters["time_window_days"],
                        projects=global_filters.get("projects"),
                        languages=global_filters.get("languages"),
                        data={
                            'selected_segments': selected_segments,
                            'summaries': final_context['summaries'],
                            'selected_segment_ids': selected_segment_ids,
                            'all_segment_ids': all_segment_ids  # For quantitative analysis
                        }
                    )
                    logger.info(f"[{request.dashboard_id}] Cached workflow result (ttl={workflow_cache.get_ttl_for_time_window(global_filters['time_window_days'])}h)")

            except Exception as cache_error:
                # Don't fail the request if caching fails
                logger.error(f"[{request.dashboard_id}] Failed to cache results: {cache_error}", exc_info=True)

            logger.info(f"[{request.dashboard_id}] Complete")

        except Exception as e:
            logger.error(f"[{request.dashboard_id}] Analysis error: {e}", exc_info=True)
            # Don't expose internal error details to client
            error_event = json.dumps({
                "type": "error",
                "error": "Analysis failed. Please try again."
            })
            yield f"data: {error_event}\n\n"

        finally:
            if db_session:
                db_session.close()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


@router.post("")
async def analyze_batch(request: AnalysisRequest, http_request: Request):
    """
    Batch analysis (convenience wrapper around streaming).

    **Recommendation**: Use /stream endpoint instead for better UX with progress updates.

    This endpoint collects all streaming events and returns the final result.
    Useful for simple scripts or when you don't need progress updates.

    Usage:
    ```json
    {
      "query": "Pierre Poilievre",
      "dashboard_id": "cprmv-practitioner",
      "workflow": "simple_rag"
    }
    ```

    Returns:
        Final analysis result (no progress updates)
    """
    start_time = time.time()

    # Validate project access
    validate_project_access(http_request, request.projects or [])

    try:
        # Validate request
        if not request.workflow and not request.pipeline:
            raise HTTPException(
                status_code=400,
                detail="Must specify either 'workflow' or 'pipeline'"
            )

        # Load dashboard config
        config = load_dashboard_config(request.dashboard_id)
        llm_service = LLMService(config, request.dashboard_id)

        # Initialize EmbeddingService for query embeddings
        from ..services.embedding_service import EmbeddingService
        embedding_service = EmbeddingService(config, request.dashboard_id)

        query_preview = f"'{request.query[:50]}...'" if request.query else "(no query - discovery mode)"
        logger.info(
            f"[{request.dashboard_id}] Batch analysis: {query_preview} "
            f"(workflow={request.workflow})"
        )

        # Get pipeline steps
        if request.workflow:
            try:
                steps = get_workflow(request.workflow)
                if request.config_overrides:
                    steps = apply_config_overrides(steps, request.config_overrides)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
        else:
            steps = [
                {"step": s.step, "config": s.config}
                for s in request.pipeline
            ]

        # Build global filters
        global_filters = {}
        # Set time_window_days with default of 7 days (uses faster 7d cache table)
        global_filters["time_window_days"] = request.time_window_days or 7

        # Initialize projects and languages to ensure consistent cache keys
        global_filters["projects"] = None
        global_filters["languages"] = None
        global_filters["channels"] = None

        if request.projects:
            global_filters["projects"] = request.projects
        if request.languages:
            global_filters["languages"] = request.languages
        if request.channels:
            global_filters["channels"] = request.channels

        # Load from dashboard config
        if hasattr(config, 'project'):
            if global_filters["projects"] is None:
                global_filters["projects"] = [config.project]
        if hasattr(config, 'languages'):
            if global_filters["languages"] is None:
                global_filters["languages"] = config.languages

        # Create database session
        db_session = SessionLocal()

        try:
            # Build pipeline
            pipeline = AnalysisPipeline(
                name=f"analysis_{request.workflow or 'custom'}_batch",
                llm_service=llm_service,
                embedding_service=embedding_service,
                db_session=db_session,
                dashboard_id=request.dashboard_id,
                config=config
            )

            # Add steps
            try:
                pipeline = build_pipeline_from_steps(
                    pipeline,
                    query=request.query,
                    steps=steps,
                    global_filters=global_filters
                )
            except ValueError as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid pipeline configuration: {e}"
                )

            # Execute (batch mode)
            result = await pipeline.execute()

            processing_time_ms = (time.time() - start_time) * 1000

            logger.info(
                f"[{request.dashboard_id}] Batch analysis complete ({processing_time_ms:.0f}ms)"
            )

            # Clean result data of non-serializable objects (numpy arrays, embeddings)
            cleaned_result = _clean_event_for_json({"data": result.data})

            # Return result
            return {
                "query": request.query,
                "dashboard_id": request.dashboard_id,
                "workflow": request.workflow,
                "processing_time_ms": processing_time_ms,
                "data": cleaned_result.get("data", result.data)
            }

        finally:
            db_session.close()

    except HTTPException:
        raise
    except Exception as e:
        processing_time_ms = (time.time() - start_time) * 1000
        logger.error(
            f"[{request.dashboard_id}] Batch analysis error after {processing_time_ms:.0f}ms: {e}",
            exc_info=True
        )
        # Don't expose internal error details to client
        raise HTTPException(status_code=500, detail="Internal server error")
