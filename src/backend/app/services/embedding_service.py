"""
Embedding Service
=================

Client for the centralized Embedding Server.
Sends embedding requests to the server with high priority (1) for interactive queries.

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

logger = get_logger("embedding_service")

# Embedding server configuration
EMBEDDING_SERVER_HOST = os.getenv('EMBEDDING_SERVER_HOST', '127.0.0.1')
EMBEDDING_SERVER_PORT = int(os.getenv('EMBEDDING_SERVER_PORT', '8005'))
EMBEDDING_SERVER_URL = f"http://{EMBEDDING_SERVER_HOST}:{EMBEDDING_SERVER_PORT}"

# Backend requests get highest priority (1) to jump ahead of batch hydration (5)
BACKEND_PRIORITY = 1


class EmbeddingService:
    """Service for converting text queries to embeddings via the centralized Embedding Server."""

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

    async def encode_queries(self, queries: List[str]) -> List[np.ndarray]:
        """
        Convert multiple queries to embeddings via the Embedding Server.

        Backend requests use priority 1 to jump ahead of batch hydration requests.

        Args:
            queries: List of query strings

        Returns:
            List of embedding vectors (numpy arrays)

        Raises:
            RuntimeError: If embedding generation fails
        """
        if not queries:
            return []

        try:
            t_start = time.time()

            # Ensure server is available
            if not self._server_healthy:
                await self._wait_for_server(timeout=30.0)

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

        except httpx.TimeoutException as e:
            logger.error(f"[{self.dashboard_id}] Embedding server timeout: {e}")
            raise RuntimeError(f"Embedding server timeout: {e}") from e
        except httpx.ConnectError as e:
            logger.error(f"[{self.dashboard_id}] Cannot connect to embedding server: {e}")
            self._server_healthy = False
            raise RuntimeError(f"Cannot connect to embedding server at {EMBEDDING_SERVER_URL}: {e}") from e
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
