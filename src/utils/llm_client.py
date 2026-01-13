#!/usr/bin/env python3
"""
Unified LLM Client
==================

Async client for communicating with the LLM balancer (llm_balancer.py).
Provides standardized access to LLM inference across the codebase.

All LLM requests should go through this client rather than making direct
HTTP calls to model servers. The balancer handles:
- Tier-aware routing (tier_1 to 80B, tier_2 to 30B, tier_3 to 4B)
- Load balancing across endpoints
- Health monitoring and failover
- Priority-based queue management

Priority Scale:
| Priority | Use Case                              |
|----------|---------------------------------------|
| 1        | Real-time/interactive (stitch)        |
| 2        | Important batch (speaker ID)          |
| 3        | Standard batch (classification)       |
| 4        | Background/bulk (rescoring)           |
| 5        | Low priority (enrichment)             |

Tier Defaults:
| Tier   | Default Timeout | Use Case                    |
|--------|-----------------|------------------------------|
| tier_1 | 300s            | Complex reasoning (80B)      |
| tier_2 | 180s            | Standard tasks (30B)         |
| tier_3 | 60s             | Fast/simple tasks (4B/8B)    |
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any

import aiohttp

from src.utils.config import get_llm_server_url
from src.utils.logger import setup_worker_logger

logger = setup_worker_logger('llm_client')


class LLMClient:
    """Unified async client for LLM balancer communication."""

    # Default timeouts by tier (seconds)
    TIER_TIMEOUTS = {
        "tier_1": 300,
        "tier_2": 180,
        "tier_3": 60,
    }

    def __init__(
        self,
        tier: str = "tier_2",
        task_type: str = "text",
        priority: int = 3,
        timeout: Optional[int] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initialize the LLM client.

        Args:
            tier: Model tier to use ("tier_1", "tier_2", "tier_3")
            task_type: Task type for routing ("stitch", "text", "analysis", "embedding")
            priority: Request priority (1-5, lower = higher priority)
            timeout: Request timeout in seconds (auto-set by tier if None)
            base_url: Override base URL (defaults to get_llm_server_url())
        """
        self.base_url = base_url or get_llm_server_url()
        self.tier = tier
        self.task_type = task_type
        self.priority = priority
        self.timeout = timeout or self.TIER_TIMEOUTS.get(tier, 180)

        # Session created lazily with lock to prevent race conditions
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()
        self._connector_limit = 20  # Max simultaneous connections

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create persistent HTTP session with connection pooling."""
        async with self._session_lock:
            if self._session is None or self._session.closed:
                connector = aiohttp.TCPConnector(
                    limit=self._connector_limit,
                    limit_per_host=self._connector_limit,
                    keepalive_timeout=60,
                )
                self._session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                    connector=connector,
                )
            return self._session

    async def close(self):
        """Close the persistent session. Call when done with client."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def call(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 2048,
        top_p: float = 0.9,
        seed: Optional[int] = 42,
        priority: Optional[int] = None,
        tier: Optional[str] = None,
        task_type: Optional[str] = None,
        retries: int = 3,
    ) -> str:
        """
        Make a single LLM call with retry logic.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens in response
            top_p: Top-p sampling parameter
            seed: Random seed for reproducibility
            priority: Override default priority
            tier: Override default tier
            task_type: Override default task type
            retries: Number of retry attempts

        Returns:
            Response text from LLM

        Raises:
            RuntimeError: If all retries fail
        """
        endpoint = f"{self.base_url}/llm-request"
        payload = {
            "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
            "model": tier or self.tier,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "priority": priority or self.priority,
            "task_type": task_type or self.task_type,
        }
        if seed is not None:
            payload["seed"] = seed

        last_error = None
        for attempt in range(retries):
            try:
                session = await self._get_session()
                async with session.post(endpoint, json=payload) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        return result["response"].strip()
                    elif resp.status == 503:
                        # Server at capacity, retry with backoff
                        error_text = await resp.text()
                        logger.warning(f"LLM server at capacity (attempt {attempt + 1}): {error_text}")
                        last_error = RuntimeError(f"Server at capacity: {error_text}")
                    else:
                        error_text = await resp.text()
                        last_error = RuntimeError(f"LLM request failed ({resp.status}): {error_text}")
                        logger.error(f"LLM error (attempt {attempt + 1}): {last_error}")

            except aiohttp.ClientError as e:
                last_error = RuntimeError(f"LLM connection error: {e}")
                logger.warning(f"LLM connection error (attempt {attempt + 1}): {e}")
            except asyncio.TimeoutError:
                last_error = RuntimeError(f"LLM request timed out after {self.timeout}s")
                logger.warning(f"LLM timeout (attempt {attempt + 1})")

            # Exponential backoff before retry
            if attempt < retries - 1:
                await asyncio.sleep(2 ** attempt)

        raise last_error or RuntimeError("LLM request failed after all retries")

    async def call_batch(
        self,
        message_batches: List[List[Dict[str, str]]],
        batch_size: int = 20,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        top_p: float = 0.9,
        seed: Optional[int] = 42,
        priority: Optional[int] = None,
        tier: Optional[str] = None,
        task_type: Optional[str] = None,
    ) -> List[str]:
        """
        Make batch LLM calls using /batch-llm-request endpoint.

        Processes message batches in chunks and returns results in order.

        Args:
            message_batches: List of message lists (each is a conversation)
            batch_size: Maximum requests per batch call
            temperature: Sampling temperature
            max_tokens: Maximum tokens per response
            top_p: Top-p sampling
            seed: Random seed
            priority: Override default priority
            tier: Override default tier
            task_type: Override default task type

        Returns:
            List of response texts (same order as input)
        """
        if not message_batches:
            return []

        endpoint = f"{self.base_url}/batch-llm-request"
        all_results = []

        # Process in chunks
        for i in range(0, len(message_batches), batch_size):
            chunk = message_batches[i:i + batch_size]

            # Build batch request
            requests = []
            for messages in chunk:
                req = {
                    "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
                    "model": tier or self.tier,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": top_p,
                    "priority": priority or self.priority,
                    "task_type": task_type or self.task_type,
                }
                if seed is not None:
                    req["seed"] = seed
                requests.append(req)

            payload = {"requests": requests}

            try:
                session = await self._get_session()
                async with session.post(endpoint, json=payload) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        responses = result.get("responses", [])
                        for r in responses:
                            response_text = r.get("response", "")
                            # Handle error responses
                            if response_text.startswith("Error:"):
                                logger.warning(f"Batch item error: {response_text}")
                                all_results.append("")
                            else:
                                all_results.append(response_text.strip())
                    else:
                        error_text = await resp.text()
                        logger.error(f"Batch request failed ({resp.status}): {error_text}")
                        # Return empty strings for this chunk
                        all_results.extend([""] * len(chunk))

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.error(f"Batch request error: {e}")
                all_results.extend([""] * len(chunk))

        return all_results

    async def call_simple(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Convenience method for simple prompt-response calls.

        Args:
            prompt: User prompt text
            system_prompt: Optional system prompt
            **kwargs: Additional arguments passed to call()

        Returns:
            Response text
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return await self.call(messages, **kwargs)


# Convenience factory functions

def create_stitch_client(**kwargs) -> LLMClient:
    """Create client optimized for stitch tasks (real-time, priority 1)."""
    defaults = {"tier": "tier_2", "priority": 1, "task_type": "stitch"}
    defaults.update(kwargs)
    return LLMClient(**defaults)


def create_speaker_id_client(**kwargs) -> LLMClient:
    """Create client for speaker identification (tier_1, priority 2)."""
    defaults = {"tier": "tier_1", "priority": 2, "task_type": "text"}
    defaults.update(kwargs)
    return LLMClient(**defaults)


def create_classification_client(**kwargs) -> LLMClient:
    """Create client for classification tasks (tier_2, priority 3)."""
    defaults = {"tier": "tier_2", "priority": 3, "task_type": "analysis"}
    defaults.update(kwargs)
    return LLMClient(**defaults)


def create_batch_client(**kwargs) -> LLMClient:
    """Create client for background batch processing (tier_2, priority 4)."""
    defaults = {"tier": "tier_2", "priority": 4, "task_type": "analysis"}
    defaults.update(kwargs)
    return LLMClient(**defaults)


def create_enrichment_client(**kwargs) -> LLMClient:
    """Create client for low-priority enrichment (tier_3, priority 5)."""
    defaults = {"tier": "tier_3", "priority": 5, "task_type": "text"}
    defaults.update(kwargs)
    return LLMClient(**defaults)
