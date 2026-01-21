#!/usr/bin/env python3
"""
Embedding Model Server
======================

Centralized embedding service that loads the 0.6B model once and serves
both the backend API (high priority, latency-sensitive) and hydrate_embeddings
(low priority, throughput-oriented).

Architecture:
- Single model instance (Qwen/Qwen3-Embedding-0.6B)
- Priority queue system: backend requests "jump the line"
- FastAPI server running on port 8005

Key features:
1. Priority-based queue (lower number = higher priority)
   - Priority 1: Backend API queries (interactive, latency-sensitive)
   - Priority 5: Hydration batches (background, throughput-oriented)
2. Batch processing support for hydration efficiency
3. Aggressive MPS memory management
4. Health monitoring
"""

# CRITICAL: Set environment variables BEFORE any imports
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import asyncio
import gc
import heapq
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from src.utils.logger import setup_worker_logger

logger = setup_worker_logger('embedding_server')
logger.setLevel(logging.INFO)


# =============================================================================
# Request/Response Models
# =============================================================================

class EmbedRequest(BaseModel):
    """Request for embedding generation."""
    texts: List[str] = Field(..., description="List of texts to embed")
    priority: int = Field(default=5, ge=1, le=10, description="Priority (1-10, lower = higher priority)")
    add_instruction: bool = Field(default=True, description="Add Qwen instruction prefix for queries")


class EmbedResponse(BaseModel):
    """Embedding response."""
    embeddings: List[List[float]] = Field(..., description="List of embedding vectors")
    model: str = Field(..., description="Model used")
    dimension: int = Field(..., description="Embedding dimension")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    request_id: str = Field(..., description="Request ID")
    queue_time_ms: float = Field(..., description="Time spent in queue")


class ServerStatus(BaseModel):
    """Server status information."""
    status: str
    model: str
    model_loaded: bool
    device: str
    embedding_dimension: int
    queue_size: int
    active_requests: int
    total_requests_processed: int
    uptime_seconds: float


# =============================================================================
# Embedding Server Manager
# =============================================================================

class EmbeddingServer:
    """
    Centralized embedding server with priority queue.

    Loads the 0.6B model once and serves requests based on priority.
    Backend requests (priority 1) jump ahead of hydration batches (priority 5).
    """

    def __init__(self, model_name: str = 'Qwen/Qwen3-Embedding-0.6B', max_queue_size: int = 100):
        self.model_name = model_name
        self.max_queue_size = max_queue_size

        # Model state
        self.model: Optional[SentenceTransformer] = None
        self.model_loaded = False
        self.device = None
        self.embedding_dim = 0

        # Priority queue: (priority, counter, texts, add_instruction, future, request_id, enqueue_time)
        self.queue: List[Tuple] = []
        self.queue_lock = asyncio.Lock()
        self.request_counter = 0

        # Tracking
        self.active_requests = 0
        self.total_requests = 0
        self.start_time = time.time()
        self.request_lock = asyncio.Lock()

        # Worker
        self.worker_task: Optional[asyncio.Task] = None
        self.shutdown_event = asyncio.Event()

        # Request expiry
        self.request_max_age_seconds = 300  # 5 minutes max queue time

    async def initialize(self):
        """Initialize the embedding server and load model."""
        logger.info("=" * 60)
        logger.info("Embedding Server Initialization")
        logger.info("=" * 60)
        logger.info(f"Model: {self.model_name}")

        # Load model
        await self._load_model()

        # Start worker
        self.worker_task = asyncio.create_task(self._worker_loop())
        logger.info("Started embedding worker")

        logger.info("Embedding Server ready")

    async def _load_model(self):
        """Load the embedding model."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}...")
            start_time = time.time()

            # Determine device
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                logger.info("Using MPS (Apple Silicon GPU)")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info("Using CUDA GPU")
            else:
                self.device = torch.device("cpu")
                logger.info("Using CPU")

            # Load model
            self.model = SentenceTransformer(
                self.model_name,
                trust_remote_code=True,
                local_files_only=True
            )
            self.model = self.model.to(self.device)

            # Test embedding to get dimension and warm up
            test_embedding = self.model.encode(["warmup test"], convert_to_numpy=True)
            self.embedding_dim = test_embedding.shape[1]

            load_time = time.time() - start_time
            self.model_loaded = True

            logger.info(f"✓ Model loaded in {load_time:.1f}s")
            logger.info(f"  Device: {self.device}")
            logger.info(f"  Embedding dimension: {self.embedding_dim}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            raise

    async def shutdown(self):
        """Shutdown the embedding server."""
        logger.info("Shutting down Embedding Server...")
        self.shutdown_event.set()

        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass

        # Clear model
        if self.model is not None:
            del self.model
            self.model = None
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        logger.info("Embedding Server shutdown complete")

    async def _worker_loop(self):
        """Worker loop that processes embedding requests from the priority queue."""
        logger.info("Embedding worker started")

        while not self.shutdown_event.is_set():
            try:
                # Pull from queue
                async with self.queue_lock:
                    if self.queue:
                        request_item = heapq.heappop(self.queue)
                    else:
                        request_item = None

                if request_item is None:
                    await asyncio.sleep(0.01)  # Brief sleep if queue empty
                    continue

                priority, counter, texts, add_instruction, future, request_id, enqueue_time = request_item

                # Check if request has expired
                age = time.time() - enqueue_time
                if age > self.request_max_age_seconds:
                    logger.warning(f"[{request_id}] Skipping expired request (age={age:.0f}s)")
                    if not future.done():
                        future.set_exception(
                            HTTPException(
                                status_code=504,
                                detail=f"Request expired after {age:.0f}s"
                            )
                        )
                    continue

                # Mark as active
                async with self.request_lock:
                    self.active_requests += 1

                queue_time_ms = (time.time() - enqueue_time) * 1000
                start_time = time.time()

                try:
                    # Process request
                    embeddings = await self._generate_embeddings(texts, add_instruction)

                    processing_time_ms = (time.time() - start_time) * 1000
                    self.total_requests += 1

                    async with self.queue_lock:
                        queue_size = len(self.queue)

                    logger.info(
                        f"[{request_id}] DONE texts={len(texts)} priority={priority} "
                        f"queue_time={queue_time_ms:.0f}ms process_time={processing_time_ms:.0f}ms "
                        f"queue_remaining={queue_size}"
                    )

                    future.set_result({
                        'embeddings': embeddings,
                        'processing_time_ms': processing_time_ms,
                        'queue_time_ms': queue_time_ms
                    })

                except Exception as e:
                    processing_time_ms = (time.time() - start_time) * 1000
                    logger.error(f"[{request_id}] FAILED after {processing_time_ms:.0f}ms: {e}")
                    future.set_exception(e)

                finally:
                    async with self.request_lock:
                        self.active_requests -= 1

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker error: {e}", exc_info=True)
                await asyncio.sleep(0.1)

        logger.info("Embedding worker stopped")

    async def _generate_embeddings(self, texts: List[str], add_instruction: bool) -> List[List[float]]:
        """Generate embeddings for texts."""
        if not self.model_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Add instruction prefix if requested (for query embeddings)
        if add_instruction:
            texts = [f"Instruct: Retrieve relevant passages.\nQuery: {t}" for t in texts]

        # Clear MPS cache before encoding
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.synchronize()

        # Generate embeddings with chunked processing to avoid OOM
        max_chunk_size = 32
        all_embeddings = []

        for chunk_start in range(0, len(texts), max_chunk_size):
            chunk_end = min(chunk_start + max_chunk_size, len(texts))
            chunk_texts = texts[chunk_start:chunk_end]

            with torch.no_grad():
                chunk_embeddings = self.model.encode(
                    chunk_texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=min(32, len(chunk_texts))
                )

            all_embeddings.append(chunk_embeddings)

            # Clean up after each chunk
            del chunk_texts
            gc.collect()

            if torch.backends.mps.is_available():
                torch.mps.synchronize()
                torch.mps.empty_cache()

        # Combine results
        if len(all_embeddings) > 1:
            embeddings = np.vstack(all_embeddings)
        else:
            embeddings = all_embeddings[0]

        # Convert to list for JSON serialization
        return embeddings.astype(np.float32).tolist()

    async def enqueue_request(
        self,
        texts: List[str],
        priority: int = 5,
        add_instruction: bool = True
    ) -> Dict[str, Any]:
        """Enqueue an embedding request and wait for result."""
        timestamp = datetime.now().strftime("%H%M%S%f")[:-3]
        request_id = f"emb_{timestamp}"

        # Create future for result
        future = asyncio.Future()
        enqueue_time = time.time()

        async with self.queue_lock:
            # Check queue capacity
            if len(self.queue) >= self.max_queue_size:
                raise HTTPException(
                    status_code=503,
                    detail=f"Queue full ({len(self.queue)}/{self.max_queue_size} requests)"
                )

            # Add to priority queue
            heapq.heappush(
                self.queue,
                (priority, self.request_counter, texts, add_instruction, future, request_id, enqueue_time)
            )
            self.request_counter += 1
            queue_size = len(self.queue)

        logger.info(
            f"[{request_id}] QUEUED texts={len(texts)} priority={priority} queue_size={queue_size}"
        )

        # Wait for result
        result = await future

        return {
            'embeddings': result['embeddings'],
            'model': self.model_name,
            'dimension': self.embedding_dim,
            'processing_time_ms': result['processing_time_ms'],
            'request_id': request_id,
            'queue_time_ms': result['queue_time_ms']
        }

    def get_status(self) -> Dict[str, Any]:
        """Get server status."""
        return {
            'status': 'healthy' if self.model_loaded else 'loading',
            'model': self.model_name,
            'model_loaded': self.model_loaded,
            'device': str(self.device) if self.device else 'none',
            'embedding_dimension': self.embedding_dim,
            'queue_size': len(self.queue),
            'active_requests': self.active_requests,
            'total_requests_processed': self.total_requests,
            'uptime_seconds': time.time() - self.start_time
        }


# =============================================================================
# FastAPI Application
# =============================================================================

server: Optional[EmbeddingServer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan manager."""
    global server

    logger.info("=" * 60)
    logger.info("Starting Embedding Server")
    logger.info("=" * 60)

    server = EmbeddingServer()
    await server.initialize()

    yield

    if server:
        await server.shutdown()


app = FastAPI(
    title="Embedding Server",
    description="Centralized embedding model service with priority queue",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if server and server.model_loaded:
        return {"status": "healthy", "timestamp": time.time()}
    return {"status": "loading", "timestamp": time.time()}


@app.get("/status", response_model=ServerStatus)
async def get_status():
    """Get server status."""
    if not server:
        raise HTTPException(status_code=503, detail="Server not initialized")
    return server.get_status()


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    """Generate embeddings for texts.

    Priority levels:
    - 1: Backend API queries (interactive, latency-sensitive)
    - 5: Default for hydration batches (background, throughput-oriented)
    """
    if not server:
        raise HTTPException(status_code=503, detail="Server not initialized")

    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided")

    if len(request.texts) > 256:
        raise HTTPException(status_code=400, detail="Maximum 256 texts per request")

    try:
        result = await server.enqueue_request(
            texts=request.texts,
            priority=request.priority,
            add_instruction=request.add_instruction
        )

        return EmbedResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing embed request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Embedding Model Server')
    parser.add_argument('--port', type=int, default=8005, help='Port to listen on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    args = parser.parse_args()

    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["loggers"]["uvicorn"]["level"] = "WARNING"
    log_config["loggers"]["uvicorn.error"]["level"] = "WARNING"
    log_config["loggers"]["uvicorn.access"]["level"] = "WARNING"

    logger.info(f"Starting Embedding Server on {args.host}:{args.port}")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_config=log_config
    )
