"""
Health Check Router
===================

Endpoints for monitoring service health and status.
"""

from fastapi import APIRouter, HTTPException
from ..models.responses import HealthResponse
from ..database.connection import engine
from sqlalchemy import text
import time
import logging

from ..utils.backend_logger import get_logger
logger = get_logger("health")

router = APIRouter(prefix="/health", tags=["health"])


@router.get("", response_model=HealthResponse)
async def health_check():
    """Basic health check"""
    return HealthResponse(
        status="healthy",
        service="backend-api",
        timestamp=time.time()
    )


@router.get("/db", response_model=HealthResponse)
async def database_health():
    """Check database connectivity"""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            result.fetchone()

        return HealthResponse(
            status="healthy",
            service="backend-api",
            timestamp=time.time(),
            db_connected=True
        )
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            service="backend-api",
            timestamp=time.time(),
            db_connected=False
        )


@router.get("/models", response_model=HealthResponse)
async def models_health():
    """Check if embedding models are loaded"""
    try:
        # Check if global embedding models are loaded
        from ..main import _embedding_models

        models_loaded = _embedding_models is not None and len(_embedding_models) > 0

        return HealthResponse(
            status="healthy" if models_loaded else "unhealthy",
            service="backend-api",
            timestamp=time.time(),
            models_loaded=models_loaded
        )
    except Exception as e:
        logger.error(f"Error checking models: {e}", exc_info=True)
        return HealthResponse(
            status="error",
            service="backend-api",
            timestamp=time.time(),
            models_loaded=False
        )
