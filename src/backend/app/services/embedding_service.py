"""
Embedding Service
=================

Client for the centralized Embedding Server with local fallback.
Sends embedding requests to the server with high priority (1) for interactive queries.

If the embedding server is unavailable, falls back to loading the model locally.
This ensures the backend can still function even if the embedding server isn't running.

Separated from LLMService to maintain single responsibility principle:
- EmbeddingService: Query -> embedding vectors (semantic search)
- LLMService: Prompts -> generated text (LLM responses)
"""

import asyncio
import logging
import os
import time
from typing import List, Optional

import httpx
import numpy as np

from ..utils.backend_logger import get_logger
from src.utils.config import load_config

logger = get_logger("embedding_service")

# Load embedding server configuration from config.yaml
_config = load_config()
_embedding_config = _config.get('services', {}).get('embedding_server', {})
EMBEDDING_SERVER_HOST = _embedding_config.get('host', '127.0.0.1') or '127.0.0.1'
EMBEDDING_SERVER_PORT = _embedding_config.get('port', 8005)
EMBEDDING_SERVER_URL = f"http://{EMBEDDING_SERVER_HOST}:{EMBEDDING_SERVER_PORT}"

# Log the resolved embedding server URL at module load time
logger.info(f"Embedding server configured at: {EMBEDDING_SERVER_URL}")

# Backend requests get highest priority (1) to jump ahead of batch hydration (5)
BACKEND_PRIORITY = 1

# Qwen instruction prefix for query embeddings
QWEN_QUERY_INSTRUCTION = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "

# Global local model instance (shared across all EmbeddingService instances)
_local_model = None
_local_model_lock = asyncio.Lock()


async def _get_local_model(model_name: str = 'Qwen/Qwen3-Embedding-0.6B'):
    """
    Get or initialize the local embedding model (singleton pattern).

    Only loads the model if the embedding server is unavailable.
    """
    global _local_model

    async with _local_model_lock:
        if _local_model is not None:
            return _local_model

        logger.info(f"Loading local embedding model: {model_name}...")
        start_time = time.time()

        try:
            # Set environment variables for optimal performance
            os.environ.setdefault('OMP_NUM_THREADS', '1')
            os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
            os.environ.setdefault('PYTORCH_MPS_HIGH_WATERMARK_RATIO', '0.0')
            os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

            import torch
            from sentence_transformers import SentenceTransformer

            # Determine device
            if torch.backends.mps.is_available():
                device = torch.device("mps")
                logger.info("Using MPS (Apple Silicon GPU) for local model")
            elif torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info("Using CUDA GPU for local model")
            else:
                device = torch.device("cpu")
                logger.info("Using CPU for local model")

            # Load model
            model = SentenceTransformer(
                model_name,
                trust_remote_code=True,
                local_files_only=True
            )
            model = model.to(device)

            # Warm up
            _ = model.encode(["warmup test"], convert_to_numpy=True)

            load_time = time.time() - start_time
            logger.info(f"✓ Local embedding model loaded in {load_time:.1f}s (device: {device})")

            _local_model = model
            return _local_model

        except Exception as e:
            logger.error(f"Failed to load local embedding model: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load local embedding model: {e}") from e


class EmbeddingService:
    """
    Service for converting text queries to embeddings.

    Tries the centralized Embedding Server first, falls back to local model if unavailable.
    """

    def __init__(self, config, dashboard_id: str = 'unknown'):
        """
        Initialize embedding service.

        Args:
            config: DashboardConfig object with embedding_model and embedding_dim
            dashboard_id: Dashboard identifier for logging
        """
        self.config = config
        self.dashboard_id = dashboard_id
        self._client: Optional[httpx.AsyncClient] = None
        self._server_healthy = False
        self._use_local_fallback = False  # Set to True if server is unavailable

        logger.info(f"[{self.dashboard_id}] EmbeddingService initialized (server: {EMBEDDING_SERVER_URL})")

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=EMBEDDING_SERVER_URL,
                timeout=httpx.Timeout(60.0, connect=5.0)  # 60s total, 5s connect
            )
        return self._client

    async def _check_server_health(self) -> bool:
        """Check if embedding server is healthy."""
        try:
            client = await self._get_client()
            response = await client.get("/health")
            if response.status_code == 200:
                data = response.json()
                self._server_healthy = data.get('status') == 'healthy'
                return self._server_healthy
            return False
        except Exception as e:
            logger.warning(f"[{self.dashboard_id}] Embedding server health check failed: {e}")
            self._server_healthy = False
            return False

    async def _wait_for_server(self, timeout: float = 30.0) -> bool:
        """Wait for embedding server to become healthy."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if await self._check_server_health():
                return True
            await asyncio.sleep(0.5)

        logger.error(f"[{self.dashboard_id}] Embedding server not available after {timeout}s")
        return False

    async def _encode_local(self, queries: List[str]) -> List[np.ndarray]:
        """
        Encode queries using the local embedding model (fallback).

        Args:
            queries: List of query strings

        Returns:
            List of embedding vectors (numpy arrays)
        """
        t_start = time.time()

        model = await _get_local_model()

        # Add Qwen instruction prefix for queries
        prefixed_queries = [QWEN_QUERY_INSTRUCTION + q for q in queries]

        # Run encoding in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: model.encode(prefixed_queries, convert_to_numpy=True, normalize_embeddings=True)
        )

        t_end = time.time()
        total_ms = (t_end - t_start) * 1000

        logger.info(
            f"[{self.dashboard_id}] Embedded {len(queries)} queries locally "
            f"in {total_ms:.0f}ms"
        )

        return [np.array(emb, dtype=np.float32) for emb in embeddings]

    async def encode_queries(self, queries: List[str]) -> List[np.ndarray]:
        """
        Convert multiple queries to embeddings.

        Tries the Embedding Server first (priority 1 for backend requests).
        Falls back to local model if server is unavailable.

        Args:
            queries: List of query strings

        Returns:
            List of embedding vectors (numpy arrays)

        Raises:
            RuntimeError: If embedding generation fails
        """
        if not queries:
            return []

        # If we've already determined to use local fallback, skip server check
        if self._use_local_fallback:
            return await self._encode_local(queries)

        try:
            t_start = time.time()

            # Quick health check (5 second timeout)
            if not self._server_healthy:
                server_available = await self._wait_for_server(timeout=5.0)
                if not server_available:
                    # Server unavailable - switch to local fallback
                    logger.warning(
                        f"[{self.dashboard_id}] Embedding server unavailable, "
                        f"falling back to local model"
                    )
                    self._use_local_fallback = True
                    return await self._encode_local(queries)

            client = await self._get_client()

            # Send request with high priority (backend jumps the queue)
            response = await client.post(
                "/embed",
                json={
                    "texts": queries,
                    "priority": BACKEND_PRIORITY,
                    "add_instruction": True  # Server adds Qwen instruction prefix
                }
            )

            if response.status_code != 200:
                error_detail = response.text
                raise RuntimeError(f"Embedding server error {response.status_code}: {error_detail}")

            result = response.json()
            embeddings = result['embeddings']
            processing_time_ms = result.get('processing_time_ms', 0)
            queue_time_ms = result.get('queue_time_ms', 0)

            t_end = time.time()
            total_ms = (t_end - t_start) * 1000

            logger.info(
                f"[{self.dashboard_id}] Embedded {len(queries)} queries "
                f"total={total_ms:.0f}ms queue={queue_time_ms:.0f}ms process={processing_time_ms:.0f}ms"
            )

            # Validate dimensions
            if embeddings and len(embeddings[0]) != self.config.embedding_dim:
                raise RuntimeError(
                    f"Embedding dimension mismatch! Expected {self.config.embedding_dim}, "
                    f"got {len(embeddings[0])}"
                )

            return [np.array(emb, dtype=np.float32) for emb in embeddings]

        except (httpx.TimeoutException, httpx.ConnectError) as e:
            # Server unavailable - switch to local fallback
            logger.warning(
                f"[{self.dashboard_id}] Embedding server connection failed ({type(e).__name__}), "
                f"falling back to local model"
            )
            self._server_healthy = False
            self._use_local_fallback = True
            return await self._encode_local(queries)

        except Exception as e:
            logger.error(f"[{self.dashboard_id}] Error generating embeddings: {e}", exc_info=True)
            raise RuntimeError(f"Failed to generate embeddings: {e}") from e

    async def encode_query_async(self, query: str) -> np.ndarray:
        """
        Convert single query to embedding vector (async version).

        Args:
            query: Natural language query

        Returns:
            Embedding vector (normalized numpy array)

        Raises:
            RuntimeError: If embedding generation fails
        """
        embeddings = await self.encode_queries([query])
        return embeddings[0]

    def encode_query(self, query: str) -> np.ndarray:
        """
        Convert single query to embedding vector (synchronous).

        Note: This creates a new event loop for synchronous callers.
        Prefer encode_query_async() when in async context.

        Args:
            query: Natural language query

        Returns:
            Embedding vector (normalized numpy array)

        Raises:
            RuntimeError: If embedding generation fails
        """
        try:
            # Run async version in new event loop
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(self.encode_query_async(query))
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"[{self.dashboard_id}] Error encoding single query: {e}", exc_info=True)
            raise RuntimeError(f"Failed to encode query: {e}") from e

    async def close(self):
        """Close the HTTP client."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
