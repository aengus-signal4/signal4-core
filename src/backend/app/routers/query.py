"""
Query Router
============

Read-only API endpoint for querying project content with flexible filtering and search.
"""

from fastapi import APIRouter, HTTPException, Request
import time
import logging

from ..utils.backend_logger import get_logger
logger = get_logger("query_router")

from ..middleware.api_key_auth import validate_project_access

from ..models.requests import QueryRequest
from ..models.responses import (
    QueryResponse,
    QueryInfo,
    QueryFiltersApplied,
    QuerySegmentResult,
    ErrorResponse
)
from ..services.query_service import QueryService

router = APIRouter(prefix="/api", tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query_segments(request: QueryRequest, http_request: Request):
    """
    Query segments with filtering and optional search.

    Supports:
    - Project filtering
    - Date range filtering
    - Language filtering
    - Confidence filtering
    - Content ID and channel filtering
    - Semantic and keyword search
    - Pagination and sorting

    Returns segment data with metadata.
    """
    start_time = time.time()

    # Validate project access for API key restrictions
    validate_project_access(http_request, [request.project] if request.project else [])

    try:
        # Initialize query service
        query_service = QueryService()

        # Extract request parameters
        filters = request.filters.model_dump() if request.filters else None
        search = request.search.model_dump() if request.search else None
        pagination = request.pagination.model_dump() if request.pagination else None
        sort = request.sort.model_dump() if request.sort else None

        # Execute query (await if it's a semantic search)
        results, total_count, filters_applied_dict = await query_service.query_segments(
            project=request.project,
            filters=filters,
            search=search,
            pagination=pagination,
            sort=sort
        )

        execution_time = (time.time() - start_time) * 1000

        # Build response
        filters_applied = QueryFiltersApplied(**filters_applied_dict)

        query_info = QueryInfo(
            project=request.project,
            total_results=total_count,
            returned_results=len(results),
            filters_applied=filters_applied,
            search_mode=search.get('mode') if search else None,
            execution_time_ms=execution_time
        )

        # Convert results to response model
        segment_results = [QuerySegmentResult(**r) for r in results]

        return QueryResponse(
            status="success",
            query_info=query_info,
            results=segment_results
        )

    except ValueError as e:
        # Client error (bad input)
        logger.warning(f"Query validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        # Server error - log details but don't expose to client
        logger.error(f"Query execution error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
