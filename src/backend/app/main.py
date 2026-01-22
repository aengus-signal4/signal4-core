"""
Backend API Service
===================

Simplified FastAPI service providing 3 core endpoints for media and analysis.

Endpoints:
- /health - Service health checks and monitoring
- /api/media/content/{id} - Unified media serving with optional transcription
- /api/analysis - All RAG/search operations with declarative workflows

Note: Embedding generation is handled by the centralized Embedding Server (port 8005).
The backend connects to it as a client with high priority (1) for interactive queries.
"""

import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import time
from pathlib import Path
from src.utils.paths import get_project_root, get_config_path

from src.backend.app.utils.backend_logger import get_logger
logger = get_logger("main")

logger.info("Backend API initializing...")

# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown events"""
    # Startup: Backend now uses the centralized embedding server
    logger.info("=" * 80)
    logger.info("Backend API starting...")
    logger.info("Embedding generation delegated to Embedding Server (port 8005)")
    logger.info("=" * 80)

    yield  # App runs

    # Shutdown
    logger.info("Backend API shutting down...")

# Debug mode from environment (controls docs and CORS)
# SECURITY: Default to false - must explicitly set DEBUG=true for development
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Production safety check - warn if DEBUG is enabled
if DEBUG:
    logger.warning("=" * 80)
    logger.warning("⚠️  DEBUG MODE ENABLED - NOT FOR PRODUCTION USE")
    logger.warning("⚠️  CORS allows localhost origins, API docs are exposed")
    logger.warning("⚠️  Set DEBUG=false in production environments")
    logger.warning("=" * 80)

# Create FastAPI app with lifespan
# Swagger/ReDoc disabled in production (DEBUG=false)
app = FastAPI(
    title="Signal4 Backend API",
    description="LLM, embedding, and search services for website dashboards",
    version="1.0.0",
    docs_url="/docs" if DEBUG else None,
    redoc_url="/redoc" if DEBUG else None,
    lifespan=lifespan
)

# CORS configuration
# Production origins always allowed; development origins only when DEBUG=true
CORS_ORIGINS_PRODUCTION = [
    "https://signal4.ca",
    "https://api.signal4.ca",
]
CORS_ORIGINS_DEVELOPMENT = [
    "http://localhost:5847",  # Flask production
    "http://127.0.0.1:5847",
    "http://localhost:5848",  # Flask staging
    "http://127.0.0.1:5848",
    "http://localhost:3000",  # Next.js development
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS_PRODUCTION + (CORS_ORIGINS_DEVELOPMENT if DEBUG else []),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key", "X-Client-Id", "Authorization"],
)

# API Key authentication middleware
# Enforces API key on all requests except public paths (/health, /docs, etc.)
from .middleware.api_key_auth import ApiKeyMiddleware
app.add_middleware(ApiKeyMiddleware)

# Security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses"""
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing"""
    start_time = time.time()

    response = await call_next(request)

    duration_ms = (time.time() - start_time) * 1000
    logger.info(
        f"{request.method} {request.url.path} - "
        f"{response.status_code} - {duration_ms:.0f}ms"
    )

    return response

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch all unhandled exceptions - log details but don't expose to client"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )

# Include routers
from .routers import health, media, analysis, query, api_keys, entities, bookmarks, explore, images, report

app.include_router(health.router)            # Health monitoring
app.include_router(media.router)             # Media serving + transcription
app.include_router(analysis.router)          # All RAG/search operations
app.include_router(query.router)             # Read-only segment query API
app.include_router(api_keys.router)          # API key management
app.include_router(entities.router)          # Entity details (episodes, speakers, channels)
app.include_router(bookmarks.router)         # User bookmarks
app.include_router(explore.router)           # Project exploration (stats, recent episodes)
app.include_router(images.router)            # Image proxy (thumbnails, avatars)
app.include_router(report.router)            # Public report endpoints (no auth)

# Root endpoint
@app.get("/")
async def root():
    """API root - service info"""
    return {
        "service": "Signal4 Backend API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn

    # Configure uvicorn logging to use our logger
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["loggers"]["uvicorn"]["level"] = "WARNING"
    log_config["loggers"]["uvicorn.error"]["level"] = "WARNING"
    log_config["loggers"]["uvicorn.access"]["level"] = "WARNING"

    logger.info("Starting Backend API on port 7999")
    logger.info("Environment: OMP_NUM_THREADS=1 (prevents threading segfaults)")
    logger.info("Models will be preloaded at startup and kept warm")

    # CRITICAL: Run without reload to avoid process forking issues with PyTorch/MPS
    # Use single worker only to prevent multiprocessing segfaults
    uvicorn.run(
        app,  # Direct app reference, not string
        host="0.0.0.0",
        port=7999,
        log_config=log_config,
        workers=1  # Single worker for macOS compatibility
    )
