"""
Embedding Service
=================

Dedicated service for query embedding using SentenceTransformer models.
Handles model loading, warm-up, and conversion of text queries to embeddings.

Separated from LLMService to maintain single responsibility principle:
- EmbeddingService: Query → embedding vectors (semantic search)
- LLMService: Prompts → generated text (LLM responses)
"""

import asyncio
import logging
import time
from typing import List, Optional
import numpy as np

from pathlib import Path
from src.utils.paths import get_project_root, get_config_path
import sys
sys.path.insert(0, str(get_project_root()))
from src.utils.logger import setup_worker_logger

logger = setup_worker_logger("backend.embedding")


class EmbeddingService:
    """Service for converting text queries to embeddings using pre-loaded models."""

    def __init__(self, config, dashboard_id: str = 'unknown'):
        """
        Initialize embedding service.

        Args:
            config: DashboardConfig object with embedding_model and embedding_dim
            dashboard_id: Dashboard identifier for logging
        """
        self.config = config
        self.dashboard_id = dashboard_id
        self._embedding_model = None

        logger.info(f"[{self.dashboard_id}] EmbeddingService initialized (model: {config.embedding_model})")

    async def _wait_for_preloaded_models(self, timeout: float = 30.0) -> bool:
        """
        Wait for background model loading to complete.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if models loaded successfully, False if timeout
        """
        try:
            from ..main import wait_for_models

            logger.info(f"[{self.dashboard_id}] Waiting for pre-loaded models...")
            start_time = time.time()

            # Wait with timeout
            while time.time() - start_time < timeout:
                models = await wait_for_models()
                if models:
                    logger.info(f"[{self.dashboard_id}] Pre-loaded models available")
                    return True
                await asyncio.sleep(0.1)

            logger.warning(f"[{self.dashboard_id}] Timeout waiting for pre-loaded models after {timeout}s")
            return False

        except Exception as e:
            logger.error(f"[{self.dashboard_id}] Error waiting for pre-loaded models: {e}", exc_info=True)
            return False

    async def _load_embedding_model(self):
        """
        Load embedding model with wait logic for pre-loaded models.

        Strategy:
        1. Wait for background loading from main.py (up to 30s)
        2. If available, use pre-loaded model
        3. If timeout, fall back to lazy loading
        4. If all fails, raise exception

        Raises:
            RuntimeError: If model fails to load
        """
        if self._embedding_model is not None:
            return

        try:
            # Import global models from main.py
            from ..main import _embedding_models

            model_name = self.config.embedding_model

            # Wait for pre-loaded models (with timeout)
            await self._wait_for_preloaded_models(timeout=30.0)

            # Determine which pre-loaded model to use
            if 'Qwen3-Embedding-4B' in model_name or '4B' in model_name:
                if '4B' in _embedding_models:
                    self._embedding_model = _embedding_models['4B']
                    logger.info(f"[{self.dashboard_id}] Using pre-loaded 4B model (2000-dim)")
                else:
                    logger.warning(f"[{self.dashboard_id}] 4B model not in cache, lazy loading...")
                    await self._lazy_load_model(model_name, truncate_dim=2000)
            else:
                # Default to 0.6B for other configs
                if '0.6B' in _embedding_models:
                    self._embedding_model = _embedding_models['0.6B']
                    logger.info(f"[{self.dashboard_id}] Using pre-loaded 0.6B model (1024-dim)")
                else:
                    logger.warning(f"[{self.dashboard_id}] 0.6B model not in cache, lazy loading...")
                    await self._lazy_load_model(model_name)

            # Verify model works
            if self._embedding_model is None:
                raise RuntimeError("Embedding model failed to load")

            test_embedding = self._embedding_model.encode(["test"], convert_to_numpy=True)
            embedding_dim = test_embedding.shape[1]
            logger.info(f"[{self.dashboard_id}] Embedding model ready, dimension: {embedding_dim}")

        except Exception as e:
            logger.error(f"[{self.dashboard_id}] Failed to load embedding model: {e}", exc_info=True)
            raise RuntimeError(f"Embedding model loading failed: {e}") from e

    async def _lazy_load_model(self, model_name: str, truncate_dim: Optional[int] = None):
        """
        Fallback: lazy load model if not pre-loaded.

        Args:
            model_name: Model identifier
            truncate_dim: Optional dimension truncation

        Raises:
            RuntimeError: If lazy loading fails
        """
        try:
            import torch
            from sentence_transformers import SentenceTransformer

            logger.info(f"[{self.dashboard_id}] Lazy loading embedding model: {model_name}")

            # Run loading in thread pool to avoid blocking
            loop = asyncio.get_event_loop()

            def _load_sync():
                if truncate_dim:
                    model = SentenceTransformer(
                        model_name,
                        trust_remote_code=True,
                        truncate_dim=truncate_dim
                    )
                else:
                    model = SentenceTransformer(
                        model_name,
                        trust_remote_code=True
                    )

                # Move to device
                device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
                model = model.to(device)
                return model

            self._embedding_model = await loop.run_in_executor(None, _load_sync)
            logger.info(f"[{self.dashboard_id}] Lazy loaded model successfully")

        except Exception as e:
            logger.error(f"[{self.dashboard_id}] Failed to lazy load model: {e}", exc_info=True)
            raise RuntimeError(f"Lazy loading failed: {e}") from e

    def _encode_batch_sync(self, query_texts: List[str], batch_size: int) -> np.ndarray:
        """
        Synchronous encoding helper - runs in thread pool executor.

        Args:
            query_texts: Formatted query strings
            batch_size: Batch size for encoding

        Returns:
            Numpy array of embeddings
        """
        import torch

        # Clear MPS cache before encoding to reduce memory contention
        if torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except:
                pass  # Ignore if cache clear fails

        with torch.no_grad():
            result = self._embedding_model.encode(
                query_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=batch_size
            )

            # Synchronize MPS to ensure operation completes
            if torch.backends.mps.is_available():
                try:
                    torch.mps.synchronize()
                except:
                    pass

            return result

    async def encode_queries(self, queries: List[str]) -> List[np.ndarray]:
        """
        Convert multiple queries to embeddings in a single batch (async, non-blocking).

        Args:
            queries: List of query strings

        Returns:
            List of embedding vectors (numpy arrays)

        Raises:
            RuntimeError: If embedding generation fails
        """
        try:
            t_load_start = time.time()
            await self._load_embedding_model()
            t_load_end = time.time()

            if (t_load_end - t_load_start) > 0.1:
                logger.warning(
                    f"[{self.dashboard_id}] Model loading took {(t_load_end-t_load_start)*1000:.0f}ms - "
                    "should be pre-loaded!"
                )

            # Format all queries with instruction prefix (matches hydrate_embeddings.py pattern)
            if 'Qwen' in self.config.embedding_model:
                query_texts = [f"Instruct: Retrieve relevant passages.\nQuery: {q}" for q in queries]
            else:
                query_texts = [f"query: {q}" for q in queries]

            # Use batch_size appropriate for model (matches hydrate_embeddings.py: 4B=16, 0.6B=32)
            if '4B' in self.config.embedding_model:
                batch_size = 16  # 4B model: smaller batches
            else:
                batch_size = 32  # 0.6B/1.5B models: larger batches

            device = self._embedding_model.device

            # Log MPS memory status before encoding
            import torch
            mps_mem_gb = 0.0
            if torch.backends.mps.is_available():
                try:
                    mps_mem_gb = torch.mps.current_allocated_memory() / 1024**3
                    if mps_mem_gb > 10.0:
                        logger.warning(
                            f"[{self.dashboard_id}] High MPS memory usage: {mps_mem_gb:.1f} GB "
                            "(may cause slowdown)"
                        )
                except:
                    pass

            logger.info(
                f"[{self.dashboard_id}] Batch embedding {len(queries)} queries "
                f"(device={device}, batch_size={batch_size}, mps_mem={mps_mem_gb:.1f}GB)..."
            )

            t_encode_start = time.time()
            # Run encoding in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(None, self._encode_batch_sync, query_texts, batch_size)
            t_encode_end = time.time()

            duration_ms = (t_encode_end - t_encode_start) * 1000
            emb_per_sec = len(queries) / (duration_ms / 1000) if duration_ms > 0 else 0
            logger.info(
                f"[{self.dashboard_id}] ✓ Batch embedded {len(queries)} queries "
                f"in {duration_ms:.0f}ms ({emb_per_sec:.1f} emb/sec)"
            )

            # Validate dimensions
            if embeddings.shape[1] != self.config.embedding_dim:
                raise RuntimeError(
                    f"Embedding dimension mismatch! Expected {self.config.embedding_dim}, "
                    f"got {embeddings.shape[1]}"
                )

            return [emb.astype(np.float32) for emb in embeddings]

        except Exception as e:
            logger.error(f"[{self.dashboard_id}] Error generating batch embeddings: {e}", exc_info=True)
            raise RuntimeError(f"Failed to generate embeddings: {e}") from e

    def encode_query(self, query: str) -> np.ndarray:
        """
        Convert single query to embedding vector (synchronous).

        Args:
            query: Natural language query

        Returns:
            Embedding vector (normalized numpy array)

        Raises:
            RuntimeError: If embedding generation fails
        """
        try:
            # Run async method synchronously
            loop = asyncio.get_event_loop()
            embeddings = loop.run_until_complete(self.encode_queries([query]))
            return embeddings[0]

        except Exception as e:
            logger.error(f"[{self.dashboard_id}] Error encoding single query: {e}", exc_info=True)
            raise RuntimeError(f"Failed to encode query: {e}") from e
