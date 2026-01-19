"""
Backend API Service
===================

Simplified FastAPI service providing 3 core endpoints for media and analysis.

Endpoints:
- /health - Service health checks and monitoring
- /api/media/content/{id} - Unified media serving with optional transcription
- /api/analysis - All RAG/search operations with declarative workflows
"""

# CRITICAL: Set environment variables BEFORE any imports
# OMP_NUM_THREADS=1 prevents OpenMP threading conflicts that cause segfaults
# when SentenceTransformer encode() is called from ThreadPoolExecutor
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import time
from pathlib import Path
from src.utils.paths import get_project_root, get_config_path

# Configure centralized backend logging using worker logger
import sys
sys.path.insert(0, str(get_project_root()))
from src.utils.logger import setup_worker_logger
logger = setup_worker_logger("backend")

logger.info("=" * 80)
logger.info("Backend API initializing...")
logger.info("=" * 80)

# Global model instances to keep warm
_embedding_models = {}
_model_loading = False
_models_ready = False

def _load_models_sync():
    """Synchronous model loading to run in thread pool"""
    import torch
    from sentence_transformers import SentenceTransformer

    # Force CPU to avoid MPS segfaults on macOS
    device = torch.device("cpu")
    logger.info(f"Using device: {device} (MPS disabled to prevent segfaults)")

    models = {}

    # Load 0.6B model only (1024-dim) - lightweight model for responsive backend
    logger.info("Loading Qwen3-Embedding-0.6B (1024-dim)...")
    model_start = time.time()
    model_0_6b = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B', trust_remote_code=True)
    model_0_6b = model_0_6b.to(device)
    # Test embedding
    test_emb_0_6b = model_0_6b.encode(["warmup test"], convert_to_numpy=True)
    models['0.6B'] = model_0_6b
    logger.info(f"✓ Loaded 0.6B model in {time.time() - model_start:.1f}s (dim: {test_emb_0_6b.shape[1]})")

    logger.info("=" * 80)
    logger.info("✓ Embedding model (0.6B) loaded and warm - API ready for fast responses")
    logger.info("=" * 80)

    return models

async def _load_models_async():
    """Asynchronously load models in the background using thread pool"""
    global _model_loading, _models_ready, _embedding_models
    _model_loading = True

    try:
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        # Run blocking model loading in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as executor:
            models = await loop.run_in_executor(executor, _load_models_sync)
            _embedding_models.update(models)

    except Exception as e:
        logger.error(f"Background model loading failed: {e}", exc_info=True)
        logger.warning("Models will be lazy-loaded on first request")
    finally:
        _model_loading = False
        _models_ready = True

async def wait_for_models():
    """Wait for models to finish loading if still loading"""
    import asyncio
    while _model_loading:
        await asyncio.sleep(0.1)
    return _embedding_models

# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown events"""
    global _embedding_models, _models_ready

    # Startup: Load models synchronously to keep them warm
    # With OMP_NUM_THREADS=1, this is safe and prevents segfaults
    logger.info("Loading embedding models at startup (OMP_NUM_THREADS=1)...")
    try:
        _embedding_models = _load_models_sync()
        _models_ready = True
        logger.info("Models loaded and warm - API ready for requests")
    except Exception as e:
        logger.error(f"Failed to load models at startup: {e}", exc_info=True)
        logger.warning("Models will be lazy-loaded on first request")
        _models_ready = True  # Allow startup to continue

    yield  # App runs

    # Shutdown: Cleanup models
    logger.info("Backend API shutting down...")
    logger.info("Cleaning up embedding models...")
    _embedding_models.clear()

# Debug mode from environment (controls docs and CORS)
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

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
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key", "Authorization"],
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
from .routers import health, media, analysis, query, health_dashboard, api_keys

app.include_router(health.router)            # Health monitoring
app.include_router(media.router)             # Media serving + transcription
app.include_router(analysis.router)          # All RAG/search operations
app.include_router(query.router)             # Read-only segment query API
app.include_router(health_dashboard.router)  # Health & Wellness dashboard
app.include_router(api_keys.router)          # API key management

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
