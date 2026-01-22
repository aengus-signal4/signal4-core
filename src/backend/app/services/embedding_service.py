"""
Embedding Service
=================

Provides embedding generation for backend API queries.
Loads the embedding model directly in-process (evergreen with the backend).

Model: Qwen/Qwen3-Embedding-0.6B (1024 dimensions)
"""

import asyncio
import gc
import logging
import os
import time
from typing import List, Optional

import numpy as np

from ..utils.backend_logger import get_logger

logger = get_logger("embedding_service")

# Qwen instruction prefix for query embeddings
QWEN_QUERY_INSTRUCTION = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "

# Global model instance (singleton, shared across all EmbeddingService instances)
_model = None
_model_lock = asyncio.Lock()
_model_device = None
_model_dim = 0


async def _get_model(model_name: str = 'Qwen/Qwen3-Embedding-0.6B'):
    """
    Get or initialize the embedding model (singleton pattern).

    Model is loaded once on first use and reused for all subsequent requests.
    """
    global _model, _model_device, _model_dim

    async with _model_lock:
        if _model is not None:
            return _model

        logger.info(f"Loading embedding model: {model_name}...")
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
                _model_device = torch.device("mps")
                logger.info("Using MPS (Apple Silicon GPU) for embeddings")
            elif torch.cuda.is_available():
                _model_device = torch.device("cuda")
                logger.info("Using CUDA GPU for embeddings")
            else:
                _model_device = torch.device("cpu")
                logger.info("Using CPU for embeddings")

            # Load model
            model = SentenceTransformer(
                model_name,
                trust_remote_code=True,
                local_files_only=True
            )
            model = model.to(_model_device)

            # Warm up and get dimension
            test_embedding = model.encode(["warmup test"], convert_to_numpy=True)
            _model_dim = test_embedding.shape[1]

            load_time = time.time() - start_time
            logger.info(f"Embedding model loaded in {load_time:.1f}s (device: {_model_device}, dim: {_model_dim})")

            _model = model
            return _model

        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load embedding model: {e}") from e


class EmbeddingService:
    """
    Service for converting text queries to embeddings.

    Loads the model in-process on first use (evergreen with the backend).
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

        logger.info(f"[{self.dashboard_id}] EmbeddingService initialized")

    async def encode_queries(self, queries: List[str]) -> List[np.ndarray]:
        """
        Convert multiple queries to embeddings.

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
            import torch

            t_start = time.time()

            model = await _get_model()

            # Add Qwen instruction prefix for queries
            prefixed_queries = [QWEN_QUERY_INSTRUCTION + q for q in queries]

            # Clear MPS cache before encoding
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
                torch.mps.synchronize()

            # Run encoding in thread pool to avoid blocking the event loop
            loop = asyncio.get_event_loop()

            def _encode():
                with torch.no_grad():
                    return model.encode(
                        prefixed_queries,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                        show_progress_bar=False,
                        batch_size=min(32, len(prefixed_queries))
                    )

            embeddings = await loop.run_in_executor(None, _encode)

            # Clean up
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
                torch.mps.empty_cache()

            t_end = time.time()
            total_ms = (t_end - t_start) * 1000

            logger.info(
                f"[{self.dashboard_id}] Embedded {len(queries)} queries "
                f"in {total_ms:.0f}ms"
            )

            # Validate dimensions
            if embeddings.shape[1] != self.config.embedding_dim:
                raise RuntimeError(
                    f"Embedding dimension mismatch! Expected {self.config.embedding_dim}, "
                    f"got {embeddings.shape[1]}"
                )

            return [np.array(emb, dtype=np.float32) for emb in embeddings]

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
        """No-op for API compatibility. Model is shared and never closed."""
        pass
