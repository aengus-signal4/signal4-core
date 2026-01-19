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
- Router: Thin HTTP layer (validate, call executor, return SSE)
- CachedWorkflowExecutor: Cache orchestration and pipeline execution
- Workflows: Predefined step sequences (convenience)
- Pipeline: AnalysisPipeline with dynamic step building
- Steps: Individual components (expand_query, retrieve_segments, etc.)
"""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from ..models.requests import AnalysisRequest
from ..middleware.api_key_auth import validate_project_access
from ..services.rag.step_registry import list_steps
from ..config.workflows import (
    get_workflow,
    list_workflows,
    apply_config_overrides
)
from ..services.llm_service import LLMService
from ..config.dashboard_config import load_dashboard_config
from ..database.connection import SessionLocal
import json

from ..utils.backend_logger import get_logger
logger = get_logger("analysis_router")

router = APIRouter(prefix="/api/analysis", tags=["analysis", "workflows"])


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
                yield f"data: {json.dumps({'type': 'error', 'error': 'Must specify either workflow or pipeline'})}\\n\\n"
                return

            # Load dashboard config
            config = load_dashboard_config(request.dashboard_id)
            llm_service = LLMService(config, request.dashboard_id)

            # Initialize EmbeddingService
            from ..services.embedding_service import EmbeddingService
            embedding_service = EmbeddingService(config, request.dashboard_id)

            query_preview = f"'{request.query[:60]}...'" if request.query else "(no query - discovery mode)"
            logger.info(
                f"[{request.dashboard_id}] Request: {query_preview} "
                f"[workflow={request.workflow or 'custom'}]"
            )

            # Get pipeline steps
            if request.workflow:
                try:
                    steps = get_workflow(request.workflow)
                    if request.config_overrides:
                        steps = apply_config_overrides(steps, request.config_overrides)
                except ValueError as e:
                    yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\\n\\n"
                    return
            else:
                steps = [
                    {"step": s.step, "config": s.config}
                    for s in request.pipeline
                ]

            # Build global filters
            global_filters = _build_global_filters(request, config)

            # Create database session
            db_session = SessionLocal()

            # Execute workflow with caching
            from ..services.rag.cached_workflow_executor import CachedWorkflowExecutor

            executor = CachedWorkflowExecutor(
                dashboard_id=request.dashboard_id,
                config=config,
                llm_service=llm_service,
                embedding_service=embedding_service,
                db_session=db_session
            )

            async for event in executor.execute(
                query=request.query,
                workflow=request.workflow,
                steps=steps,
                global_filters=global_filters,
                verbose=request.verbose
            ):
                yield f"data: {json.dumps(event)}\\n\\n"

        except Exception as e:
            logger.error(f"[{request.dashboard_id}] Analysis error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'error': 'Analysis failed. Please try again.'})}\\n\\n"

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


def _build_global_filters(request: AnalysisRequest, config) -> dict:
    """
    Build global filters from request and config.

    Args:
        request: Analysis request
        config: Dashboard configuration

    Returns:
        Global filters dict
    """
    global_filters = {
        "time_window_days": request.time_window_days or 7,
        "projects": None,
        "languages": None,
        "channels": None
    }

    if request.projects:
        global_filters["projects"] = request.projects
    if request.languages:
        global_filters["languages"] = request.languages
    if request.channels:
        global_filters["channels"] = request.channels

    # Load from dashboard config if not specified
    if global_filters["projects"] is None:
        if hasattr(config, 'allowed_projects') and config.allowed_projects:
            global_filters["projects"] = config.allowed_projects
        elif hasattr(config, 'project') and config.project:
            global_filters["projects"] = [config.project]

    if hasattr(config, 'languages') and global_filters["languages"] is None:
        global_filters["languages"] = config.languages

    return global_filters
