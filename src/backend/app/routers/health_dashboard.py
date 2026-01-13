"""
Health Dashboard Router
=======================

Dedicated endpoints for the Health & Wellness analysis dashboard.

Provides:
- Full health dashboard analysis (all domains)
- Single domain exploration
- Health topic search

Uses the health_query_generator and existing RAG pipeline infrastructure.
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Optional, List
from pydantic import BaseModel, Field
import json
import logging
import asyncio

from ..services.rag.health_query_generator import (
    HealthQueryGenerator,
    HealthDomainAnalyzer,
    run_health_analysis_pipeline
)
from ..services.llm_service import LLMService
from ..services.embedding_service import EmbeddingService
from ..config.dashboard_config import load_dashboard_config
from ..database.connection import SessionLocal
from ..utils.backend_logger import get_logger

logger = get_logger("health_dashboard")

router = APIRouter(prefix="/api/health", tags=["health", "wellness"])


# ============================================================================
# Request/Response Models
# ============================================================================

class HealthDashboardRequest(BaseModel):
    """Request for full health dashboard analysis."""
    time_window_days: int = Field(default=90, description="Analysis time window in days")
    max_queries: int = Field(default=100, description="Maximum queries to generate")
    include_llm_expansion: bool = Field(default=True, description="Include LLM-expanded query variations")
    include_cross_domain: bool = Field(default=True, description="Include cross-domain queries")
    include_perspectives: bool = Field(default=True, description="Include perspective variations")
    projects: Optional[List[str]] = Field(default=None, description="Filter by projects")
    domains: Optional[List[str]] = Field(default=None, description="Limit to specific domains")


class HealthTopicRequest(BaseModel):
    """Request for single health topic exploration."""
    topic: str = Field(..., description="Health topic to explore (e.g., 'intermittent fasting')")
    time_window_days: int = Field(default=90, description="Time window in days")
    include_perspectives: bool = Field(default=True, description="Include different perspectives")
    projects: Optional[List[str]] = Field(default=None, description="Filter by projects")


class DomainSummary(BaseModel):
    """Summary for a single health domain."""
    domain: str
    segment_count: int
    theme_count: int
    themes: List[dict]
    subtheme_count: int
    summary: Optional[str] = None


class HealthDashboardResponse(BaseModel):
    """Response from full health dashboard analysis."""
    domains: dict
    total_segments: int
    total_themes: int
    domains_analyzed: int
    synthesis: Optional[str] = None


# ============================================================================
# Discovery Endpoints
# ============================================================================

@router.get("/domains")
async def get_health_domains():
    """
    List available health domains and their topics.

    Returns the configured health domains (nutrition, fitness, mental_health, etc.)
    with their associated query topics.
    """
    try:
        config = load_dashboard_config("health-wellness")
        llm_service = LLMService(config, "health-wellness")

        generator = HealthQueryGenerator(llm_service)
        domains = generator.get_domains()

        domain_info = {}
        for domain in domains:
            topics = generator.get_domain_topics(domain)
            domain_info[domain] = {
                "name": domain.replace("_", " ").title(),
                "topic_count": len(topics),
                "sample_topics": topics[:5]  # First 5 topics as sample
            }

        return {
            "domains": domain_info,
            "total_domains": len(domains)
        }
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Health dashboard config not found. Ensure health_wellness dashboard is configured."
        )
    except Exception as e:
        logger.error(f"Error getting domains: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cross-domain-connections")
async def get_cross_domain_connections():
    """
    List cross-domain topic connections.

    Shows how health domains connect (e.g., nutrition+fitness, sleep+mental_health).
    """
    from ..services.rag.health_query_generator import HealthQueryGenerator

    return {
        "connections": [
            {
                "domains": list(domains),
                "topics": topics
            }
            for domains, topics in HealthQueryGenerator.DOMAIN_CONNECTIONS.items()
        ]
    }


# ============================================================================
# Analysis Endpoints
# ============================================================================

@router.post("/analyze/stream")
async def analyze_health_dashboard_stream(request: HealthDashboardRequest):
    """
    Run full health dashboard analysis with SSE streaming.

    Generates diverse queries across all health domains, retrieves content,
    clusters into themes/sub-themes, and generates hierarchical summaries.

    This is a long-running operation (2-5 minutes depending on content volume).
    Use streaming to get progress updates.

    Event types:
    - `domain_start`: Starting analysis for a domain
    - `domain_progress`: Progress within a domain
    - `domain_complete`: Domain analysis complete
    - `synthesis_start`: Starting cross-domain synthesis
    - `complete`: Full analysis complete with all results
    - `error`: Error occurred
    """
    async def event_generator():
        db_session = None
        try:
            # Load config
            config = load_dashboard_config("health-wellness")
            llm_service = LLMService(config, "health-wellness")
            embedding_service = EmbeddingService(config, "health-wellness")
            db_session = SessionLocal()

            # Initialize analyzer
            analyzer = HealthDomainAnalyzer(
                llm_service=llm_service,
                embedding_service=embedding_service,
                db_session=db_session,
                config=config.raw
            )

            # Generate queries
            yield f"data: {json.dumps({'type': 'status', 'message': 'Generating health queries...'})}\n\n"

            all_queries = await analyzer.query_generator.generate_all_queries(
                include_llm_expansion=request.include_llm_expansion,
                include_cross_domain=request.include_cross_domain,
                include_perspectives=request.include_perspectives
            )

            sampled_queries = analyzer.query_generator.sample_queries(
                all_queries, n=request.max_queries, strategy="stratified"
            )

            yield f"data: {json.dumps({'type': 'queries_generated', 'total': len(sampled_queries)})}\n\n"

            # Group by domain
            queries_by_domain = {}
            for q in sampled_queries:
                primary = q.domain.split("+")[0] if "+" in q.domain else q.domain
                if request.domains and primary not in request.domains:
                    continue
                queries_by_domain.setdefault(primary, []).append(q)

            # Analyze each domain
            domain_results = {}
            for domain, queries in queries_by_domain.items():
                yield f"data: {json.dumps({'type': 'domain_start', 'domain': domain, 'query_count': len(queries)})}\n\n"

                try:
                    result = await analyzer._analyze_single_domain(
                        domain=domain,
                        queries=queries,
                        time_window_days=request.time_window_days,
                        projects=request.projects
                    )

                    # Generate summary for this domain
                    if result.themes:
                        summary = await analyzer.generate_domain_summary(result)
                        result.summary = summary

                    domain_results[domain] = result

                    yield f"data: {json.dumps({'type': 'domain_complete', 'domain': domain, 'segment_count': len(result.segments), 'theme_count': len(result.themes)})}\n\n"

                except Exception as e:
                    logger.error(f"Domain {domain} failed: {e}")
                    yield f"data: {json.dumps({'type': 'domain_error', 'domain': domain, 'error': str(e)})}\n\n"

            # Generate synthesis
            yield f"data: {json.dumps({'type': 'synthesis_start'})}\n\n"

            # Build final response
            response = {
                "domains": {
                    domain: {
                        "segment_count": a.metrics.get("segment_count", 0),
                        "theme_count": len(a.themes),
                        "themes": [
                            {
                                "id": t.id if hasattr(t, 'id') else str(i),
                                "name": t.name if hasattr(t, 'name') else f"Theme {i+1}",
                                "size": len(t.segments) if hasattr(t, 'segments') else 0
                            }
                            for i, t in enumerate(a.themes)
                        ],
                        "subtheme_count": sum(len(subs) for subs in a.subthemes.values()),
                        "summary": a.summary
                    }
                    for domain, a in domain_results.items()
                },
                "total_segments": sum(a.metrics.get("segment_count", 0) for a in domain_results.values()),
                "total_themes": sum(len(a.themes) for a in domain_results.values()),
                "domains_analyzed": len(domain_results)
            }

            yield f"data: {json.dumps({'type': 'complete', 'data': response})}\n\n"

        except FileNotFoundError:
            yield f"data: {json.dumps({'type': 'error', 'error': 'Health dashboard not configured'})}\n\n"
        except Exception as e:
            logger.error(f"Health dashboard analysis failed: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        finally:
            if db_session:
                db_session.close()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.post("/analyze")
async def analyze_health_dashboard(request: HealthDashboardRequest):
    """
    Run full health dashboard analysis (non-streaming).

    Returns complete analysis when finished. For long-running analyses,
    use /analyze/stream instead for progress updates.
    """
    db_session = None
    try:
        config = load_dashboard_config("health-wellness")
        llm_service = LLMService(config, "health-wellness")
        embedding_service = EmbeddingService(config, "health-wellness")
        db_session = SessionLocal()

        result = await run_health_analysis_pipeline(
            llm_service=llm_service,
            embedding_service=embedding_service,
            db_session=db_session,
            time_window_days=request.time_window_days,
            projects=request.projects
        )

        return result

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Health dashboard not configured"
        )
    except Exception as e:
        logger.error(f"Health analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if db_session:
            db_session.close()


@router.post("/topic/stream")
async def explore_health_topic_stream(request: HealthTopicRequest):
    """
    Explore a specific health topic with streaming.

    Uses the health_topic_explorer workflow for quick analysis of a single topic.
    """
    from ..services.rag.analysis_pipeline import AnalysisPipeline
    from ..services.rag.step_registry import build_pipeline_from_steps
    from ..config.workflows import get_workflow

    async def event_generator():
        db_session = None
        try:
            config = load_dashboard_config("health-wellness")
            llm_service = LLMService(config, "health-wellness")
            embedding_service = EmbeddingService(config, "health-wellness")
            db_session = SessionLocal()

            # Use the health_topic_explorer workflow
            steps = get_workflow("health_topic_explorer")

            # Build pipeline
            pipeline = AnalysisPipeline(
                name=f"health_topic_{request.topic[:20]}",
                llm_service=llm_service,
                embedding_service=embedding_service,
                db_session=db_session,
                dashboard_id="health-wellness",
                config=config
            )

            # Build and execute
            global_filters = {
                "time_window_days": request.time_window_days,
                "projects": request.projects
            }

            pipeline = build_pipeline_from_steps(
                pipeline, request.topic, steps, global_filters
            )

            # Stream results
            async for event in pipeline.execute_stream():
                # Clean event for JSON
                cleaned = _clean_health_event(event)
                yield f"data: {json.dumps(cleaned)}\n\n"

        except Exception as e:
            logger.error(f"Topic exploration failed: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
        finally:
            if db_session:
                db_session.close()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )


@router.get("/topic/{topic}")
async def explore_health_topic(
    topic: str,
    time_window_days: int = Query(default=90),
    projects: Optional[str] = Query(default=None)
):
    """
    Quick health topic exploration (non-streaming).

    Simple GET endpoint for exploring a health topic.
    """
    project_list = projects.split(",") if projects else None

    request = HealthTopicRequest(
        topic=topic,
        time_window_days=time_window_days,
        projects=project_list
    )

    # Use the analysis endpoint with health workflow
    from ..routers.analysis import analyze_stream
    from ..models.requests import AnalysisRequest

    analysis_request = AnalysisRequest(
        query=topic,
        dashboard_id="health-wellness",
        workflow="health_topic_explorer",
        time_window_days=time_window_days,
        projects=project_list
    )

    # This returns a streaming response
    return await analyze_stream(analysis_request)


# ============================================================================
# Helper Functions
# ============================================================================

def _clean_health_event(event: dict) -> dict:
    """Clean event for JSON serialization."""
    import numpy as np

    cleaned = {}
    for key, value in event.items():
        if isinstance(value, np.ndarray):
            continue  # Skip numpy arrays
        elif key in ["embedding", "embeddings", "query_embeddings"]:
            continue  # Skip embeddings
        elif isinstance(value, dict):
            cleaned[key] = _clean_health_event(value)
        elif isinstance(value, list):
            cleaned[key] = [
                _clean_health_event(item) if isinstance(item, dict) else item
                for item in value
                if not isinstance(item, np.ndarray)
            ]
        else:
            cleaned[key] = value

    return cleaned
