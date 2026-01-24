#!/usr/bin/env python3
"""
LLM Load Balancer
=================

FastAPI server that load-balances LLM requests across multiple MLX model_server
instances. Uses endpoint-affinity queues with spillover for optimal routing.

Architecture:
- Per-endpoint priority queues (each endpoint has its own queue)
- Routing decisions made at ENQUEUE time, not dequeue time (eliminates race conditions)
- Spillover threshold: requests queue to native endpoint until overloaded, then spill to fallback
- Each worker only pulls from its own endpoint's queue

Key features:
1. Endpoint-affinity queues eliminate race conditions between workers
2. Spillover at threshold (default 3) balances affinity with throughput
3. Native endpoint preference with automatic fallback on unhealthy/overloaded
4. Health monitoring and automatic failover
5. Priority-based processing within each endpoint queue
"""

import asyncio
import logging
import os
import time
import heapq
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from enum import Enum

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import httpx
import requests as req_lib
import uvicorn

from src.utils.logger import setup_worker_logger
from src.utils.config import load_config, get_llm_backend_config, get_llm_task_routing

logger = setup_worker_logger('llm_balancer')
logger.setLevel(logging.INFO)


# =============================================================================
# Health Check Functions
# =============================================================================

async def check_mlx_endpoint(endpoint: str, port: int = 8004, timeout: float = 5.0) -> bool:
    """Check if an MLX model_server endpoint is available."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"http://{endpoint}:{port}/health")
            return response.status_code == 200
    except Exception as e:
        logger.debug(f"MLX endpoint {endpoint}:{port} not available: {e}")
        return False


async def check_ollama_endpoint(endpoint: str, timeout: float = 5.0) -> bool:
    """Check if an Ollama endpoint is available."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"http://{endpoint}:11434/api/tags")
            return response.status_code == 200
    except Exception as e:
        logger.debug(f"Ollama endpoint {endpoint} not available: {e}")
        return False


# =============================================================================
# Request/Response Models
# =============================================================================

class TaskType(str, Enum):
    """Supported task types."""
    STITCH = "stitch"
    TEXT = "text"
    ANALYSIS = "analysis"
    EMBEDDING = "embedding"


class LLMMessage(BaseModel):
    """A single message in an LLM conversation."""
    role: str = Field(..., description="Role: 'system', 'user', or 'assistant'")
    content: str = Field(..., description="Message content")


class LLMRequest(BaseModel):
    """Request for LLM processing."""
    messages: List[LLMMessage] = Field(..., description="List of messages")
    model: str = Field(default="tier_3", description="Model tier to use")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    seed: Optional[int] = Field(default=42)
    priority: int = Field(default=2, ge=1, le=99, description="Priority (1-99, lower = higher)")
    task_type: TaskType = Field(default=TaskType.TEXT)


class LLMResponse(BaseModel):
    """LLM response."""
    response: str
    model_used: str
    processing_time: float
    request_id: str
    endpoint_used: str
    priority: int
    task_type: str


class BatchLLMRequest(BaseModel):
    """Batch request containing multiple LLM requests."""
    requests: List[LLMRequest] = Field(..., description="List of LLM requests to process")


class BatchLLMResponse(BaseModel):
    """Batch response containing multiple LLM responses."""
    responses: List[LLMResponse]
    total_processing_time: float
    request_count: int


class ServerStatus(BaseModel):
    """Server status information."""
    status: str
    backend_type: str
    backend_port: int
    backend_endpoints: List[str]
    spillover_threshold: int
    active_requests: int
    queued_requests: int
    queue_by_endpoint: Dict[str, int]
    total_requests_processed: int
    uptime_seconds: float
    endpoint_status: Dict[str, Dict[str, Any]]
    queue_by_priority: Dict[int, int]
    requests_by_task_type: Dict[str, int]
    tier_native_endpoints: Dict[str, List[str]]
    tier_fallback_endpoints: Dict[str, List[str]]


# =============================================================================
# LLM Load Balancer Manager
# =============================================================================

class LLMBalancer:
    """
    Load balancer for LLM requests across multiple MLX endpoints.

    Architecture:
    - Per-endpoint priority queues (no competition between workers)
    - Routing decisions at enqueue time with spillover threshold
    - Each worker only pulls from its own endpoint's queue
    """

    def __init__(self, max_queue_size: int = 200):
        self.max_queue_size = max_queue_size

        # Backend configuration (load first so we know endpoints)
        backend_config = get_llm_backend_config()
        self.backend_type = backend_config['backend']
        self.backend_port = backend_config['port']
        self.backend_endpoints = backend_config['endpoints']
        self.endpoint_tiers = backend_config.get('endpoint_tiers', {})

        # Per-endpoint priority queues: (priority, counter, request, future, request_id, enqueue_time)
        # Each endpoint has its own queue - workers only pull from their assigned queue
        self.endpoint_queues: Dict[str, List[Tuple]] = {
            ep: [] for ep in self.backend_endpoints
        }
        self.queue_lock = asyncio.Lock()
        self.request_counter = 0

        # Spillover threshold: queue to native until this depth, then spill to fallback
        self.spillover_threshold = int(os.environ.get('LLM_SPILLOVER_THRESHOLD', 5))

        # Request tracking
        self.active_requests = 0
        self.total_requests = 0
        self.start_time = time.time()
        self.request_lock = asyncio.Lock()
        self.requests_by_task_type = {t.value: 0 for t in TaskType}

        # Endpoint health tracking
        self.endpoint_status: Dict[str, bool] = {}
        self.endpoint_last_check: Dict[str, float] = {}
        self.endpoint_active: Dict[str, int] = {}  # Active requests per endpoint
        self.health_check_interval = 30

        # Request expiry (drop requests stuck too long)
        self.request_max_age_seconds = 300  # 5 minutes max queue time

        # Worker tasks
        self.workers: List[asyncio.Task] = []
        self.shutdown_event = asyncio.Event()

        # Tier routing maps (built in initialize())
        self.tier_native_endpoints: Dict[str, List[str]] = {}
        self.tier_fallback_endpoints: Dict[str, List[str]] = {}

        # HTTP client for backend calls (blocking, used in executor)
        self.executor = ThreadPoolExecutor(max_workers=len(self.backend_endpoints) * 2)

    async def initialize(self):
        """Initialize the load balancer."""
        logger.info("=" * 60)
        logger.info("LLM Load Balancer Initialization")
        logger.info("=" * 60)
        logger.info(f"Backend: {self.backend_type.upper()} on port {self.backend_port}")
        logger.info(f"Endpoints: {self.backend_endpoints}")
        logger.info(f"Endpoint tiers: {self.endpoint_tiers}")
        logger.info(f"Spillover threshold: {self.spillover_threshold}")

        # Build native endpoint map: which endpoints are dedicated to each tier
        # Native = the lowest tier an endpoint supports (its primary purpose)
        self.tier_native_endpoints = {
            "tier_1": [],
            "tier_2": [],
            "tier_3": []
        }
        for endpoint, tiers in self.endpoint_tiers.items():
            # Find the lowest tier this endpoint supports - that's its native tier
            if "tier_1" in tiers:
                self.tier_native_endpoints["tier_1"].append(endpoint)
            elif "tier_2" in tiers:
                self.tier_native_endpoints["tier_2"].append(endpoint)
            elif "tier_3" in tiers:
                self.tier_native_endpoints["tier_3"].append(endpoint)
        logger.info(f"Native endpoints: {self.tier_native_endpoints}")

        # Build fallback endpoint map: which endpoints can handle overflow for each tier
        # Fallback = higher-tier endpoints that can handle lower-tier work
        self.tier_fallback_endpoints = {
            "tier_1": [],  # No fallback for tier_1 (it's the highest)
            "tier_2": [],  # tier_1 endpoints can handle tier_2 overflow
            "tier_3": []   # tier_1 and tier_2 endpoints can handle tier_3 overflow
        }
        for endpoint, tiers in self.endpoint_tiers.items():
            if "tier_1" in tiers:
                # tier_1 endpoints can be fallback for tier_2 and tier_3
                self.tier_fallback_endpoints["tier_2"].append(endpoint)
                self.tier_fallback_endpoints["tier_3"].append(endpoint)
            elif "tier_2" in tiers:
                # tier_2 endpoints can be fallback for tier_3
                self.tier_fallback_endpoints["tier_3"].append(endpoint)
        logger.info(f"Fallback endpoints: {self.tier_fallback_endpoints}")

        # Check initial endpoint health
        for endpoint in self.backend_endpoints:
            healthy = await self._check_endpoint_health(endpoint)
            self.endpoint_status[endpoint] = healthy
            self.endpoint_active[endpoint] = 0
            status = "OK" if healthy else "FAILED"
            logger.info(f"  {endpoint}: {status}")

        healthy_count = sum(1 for h in self.endpoint_status.values() if h)
        if healthy_count == 0:
            raise RuntimeError("No healthy endpoints available")

        # Start one worker per healthy endpoint
        for endpoint in self.backend_endpoints:
            if self.endpoint_status.get(endpoint):
                worker = asyncio.create_task(self._worker_loop(endpoint))
                self.workers.append(worker)
                logger.info(f"Started worker for endpoint {endpoint}")

        logger.info(f"Started {len(self.workers)} workers")

        # Start health monitor background task
        self.health_monitor_task = asyncio.create_task(self._health_monitor_loop())
        logger.info(f"Started health monitor (interval: {self.health_check_interval}s)")

        logger.info("LLM Load Balancer ready")

    async def shutdown(self):
        """Shutdown the load balancer."""
        logger.info("Shutting down LLM Load Balancer...")
        self.shutdown_event.set()

        # Cancel health monitor
        if hasattr(self, 'health_monitor_task'):
            self.health_monitor_task.cancel()

        for worker in self.workers:
            worker.cancel()

        all_tasks = self.workers + ([self.health_monitor_task] if hasattr(self, 'health_monitor_task') else [])
        if all_tasks:
            await asyncio.gather(*all_tasks, return_exceptions=True)

        self.executor.shutdown(wait=True)
        logger.info("LLM Load Balancer shutdown complete")

    async def _check_endpoint_health(self, endpoint: str) -> bool:
        """Check endpoint health."""
        if self.backend_type == "mlx":
            return await check_mlx_endpoint(endpoint, self.backend_port)
        else:
            return await check_ollama_endpoint(endpoint)

    async def _health_monitor_loop(self):
        """Periodically recheck endpoint health and restart workers if needed."""
        logger.info("Health monitor started")

        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(self.health_check_interval)

                for endpoint in self.backend_endpoints:
                    was_healthy = self.endpoint_status.get(endpoint, False)
                    is_healthy = await self._check_endpoint_health(endpoint)
                    self.endpoint_status[endpoint] = is_healthy
                    self.endpoint_last_check[endpoint] = time.time()

                    if is_healthy and not was_healthy:
                        logger.info(f"Endpoint {endpoint} recovered - starting worker")
                        # Start a new worker for this endpoint
                        worker = asyncio.create_task(self._worker_loop(endpoint))
                        self.workers.append(worker)

                    elif not is_healthy and was_healthy:
                        logger.warning(f"Endpoint {endpoint} became unhealthy")

                # Clean up expired requests from queue
                await self._expire_old_requests()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")

        logger.info("Health monitor stopped")

    async def _expire_old_requests(self):
        """Remove requests that have been queued too long."""
        now = time.time()
        expired_count = 0

        async with self.queue_lock:
            # Process each endpoint queue
            for endpoint, queue in self.endpoint_queues.items():
                new_queue = []
                for item in queue:
                    priority, counter, request, future, request_id, enqueue_time = item
                    age = now - enqueue_time

                    if age > self.request_max_age_seconds:
                        # Expire this request
                        expired_count += 1
                        logger.warning(
                            f"[{request_id}] EXPIRED after {age:.0f}s in queue "
                            f"(endpoint={endpoint}, priority={priority})"
                        )
                        # Set exception on the future so caller gets notified
                        if not future.done():
                            future.set_exception(
                                HTTPException(
                                    status_code=504,
                                    detail=f"Request expired after {age:.0f}s - endpoint unavailable"
                                )
                            )
                    else:
                        new_queue.append(item)

                if len(new_queue) != len(queue):
                    self.endpoint_queues[endpoint] = new_queue
                    heapq.heapify(self.endpoint_queues[endpoint])

            if expired_count > 0:
                logger.info(f"Expired {expired_count} requests from queues")

    def _resolve_model_tier(self, model_name: str) -> str:
        """Resolve model name to tier."""
        model_lower = model_name.lower()
        if any(x in model_lower for x in ["tier_1", "tier1", "80b", "large", "best"]):
            return "tier_1"
        if any(x in model_lower for x in ["tier_2", "tier2", "30b", "medium", "balanced"]):
            return "tier_2"
        # Default to tier_3
        return "tier_3"

    def _select_endpoint_for_request(self, model_tier: str) -> str:
        """Select best endpoint at enqueue time with native preference and load balancing.

        Strategy:
        - Prefer native endpoints when they have low queue depth
        - When native is busy, distribute load across all capable endpoints by queue depth
        - This balances tier affinity with throughput for batch workloads

        Must be called while holding queue_lock.
        """
        native_eps = self.tier_native_endpoints.get(model_tier, [])
        fallback_eps = self.tier_fallback_endpoints.get(model_tier, [])

        # Filter to healthy endpoints
        healthy_native = [ep for ep in native_eps if self.endpoint_status.get(ep, False)]
        healthy_fallback = [ep for ep in fallback_eps if self.endpoint_status.get(ep, False)]

        # 1. If any healthy native has queue below threshold, prefer it
        for ep in healthy_native:
            if len(self.endpoint_queues.get(ep, [])) < self.spillover_threshold:
                return ep

        # 2. Native is busy - load balance across ALL capable endpoints by queue depth
        #    This distributes batch workloads more evenly instead of dumping all overflow to fallback
        all_capable = healthy_native + healthy_fallback
        if all_capable:
            return min(all_capable, key=lambda ep: len(self.endpoint_queues.get(ep, [])))

        # 3. No healthy endpoints at all - raise error
        raise HTTPException(
            status_code=503,
            detail=f"No healthy endpoints available for {model_tier}"
        )

    def _get_capable_endpoints(self, model_tier: str) -> List[str]:
        """Get endpoints capable of handling this tier (including fallback)."""
        if not self.endpoint_tiers:
            return list(self.backend_endpoints)

        capable = []
        for endpoint, tiers in self.endpoint_tiers.items():
            if model_tier in tiers:
                capable.append(endpoint)
            # tier_3 can fall back to tier_2
            elif model_tier == "tier_3" and "tier_2" in tiers:
                capable.append(endpoint)

        return capable

    def _choose_best_endpoint(self, request: LLMRequest) -> Optional[str]:
        """Choose the best endpoint for this request based on tier match and load.

        Strategy: Prefer native tier endpoints when idle, but spill to fallback
        endpoints when native ones are busy. This maximizes throughput while
        keeping the right work on the right hardware when possible.

        For tier_3 requests:
        - If 10.0.0.209 (native tier_3) is idle (0 active), use it
        - If 10.0.0.209 is busy, use tier_2 endpoints as overflow
        - Load balance across all capable endpoints based on active count
        """
        model_tier = self._resolve_model_tier(request.model)

        # Collect all capable endpoints with their type
        native_endpoints = []
        fallback_endpoints = []

        for endpoint, tiers in self.endpoint_tiers.items():
            if not self.endpoint_status.get(endpoint, False):
                continue  # Skip unhealthy

            if model_tier in tiers:
                native_endpoints.append(endpoint)
            elif model_tier == "tier_3" and "tier_2" in tiers:
                fallback_endpoints.append(endpoint)

        # If native endpoint is idle, always prefer it
        for ep in native_endpoints:
            if self.endpoint_active.get(ep, 0) == 0:
                return ep

        # Otherwise, load balance across ALL capable endpoints (native + fallback)
        all_capable = native_endpoints + fallback_endpoints
        if all_capable:
            return min(all_capable, key=lambda ep: self.endpoint_active.get(ep, 0))

        logger.warning(f"No healthy endpoints for tier {model_tier}")
        return None

    async def _worker_loop(self, endpoint: str):
        """
        Worker loop that pulls only from this endpoint's queue.

        Each worker is assigned to exactly one endpoint and only processes
        requests from that endpoint's queue. Routing decisions are made at
        enqueue time, so workers don't compete for requests.
        """
        logger.info(f"Worker for {endpoint} starting")

        while not self.shutdown_event.is_set():
            try:
                # Check if this endpoint is healthy
                if not self.endpoint_status.get(endpoint, False):
                    await asyncio.sleep(1.0)  # Wait and retry health check
                    continue

                # Pull from THIS endpoint's queue only
                async with self.queue_lock:
                    queue = self.endpoint_queues.get(endpoint, [])
                    if queue:
                        request_item = heapq.heappop(queue)
                    else:
                        request_item = None

                if request_item is None:
                    await asyncio.sleep(0.05)  # Brief sleep if queue empty
                    continue

                priority, counter, request, future, request_id, enqueue_time = request_item

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
                    self.endpoint_active[endpoint] = self.endpoint_active.get(endpoint, 0) + 1

                start_time = time.time()
                model_tier = self._resolve_model_tier(request.model)

                try:
                    # Process request
                    response_data = await asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        self._process_request_sync,
                        request,
                        request_id,
                        endpoint
                    )

                    processing_time = time.time() - start_time

                    # Success
                    self.total_requests += 1
                    self.requests_by_task_type[request.task_type.value] += 1

                    total_queued = sum(len(q) for q in self.endpoint_queues.values())
                    logger.info(
                        f"[{request_id}] DONE {endpoint} tier={model_tier} "
                        f"time={processing_time:.2f}s queue={total_queued}"
                    )

                    future.set_result(response_data)

                except Exception as e:
                    processing_time = time.time() - start_time
                    logger.error(f"[{request_id}] FAILED {endpoint} after {processing_time:.2f}s: {e}")

                    # Mark endpoint as potentially unhealthy
                    self.endpoint_status[endpoint] = False

                    future.set_exception(e)

                finally:
                    async with self.request_lock:
                        self.active_requests -= 1
                        self.endpoint_active[endpoint] = max(0, self.endpoint_active.get(endpoint, 1) - 1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker error: {e}")
                await asyncio.sleep(0.5)

    def _process_request_sync(self, request: LLMRequest, request_id: str, endpoint: str) -> Dict[str, Any]:
        """Process request synchronously (runs in executor)."""
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

        if self.backend_type == "mlx":
            return self._call_mlx_endpoint(request, request_id, endpoint, messages)
        else:
            return self._call_ollama_endpoint(request, request_id, endpoint, messages)

    def _call_mlx_endpoint(self, request: LLMRequest, request_id: str, endpoint: str, messages: List[Dict]) -> Dict[str, Any]:
        """Call MLX model_server endpoint."""
        payload = {
            "messages": messages,
            "model": request.model,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "priority": request.priority,
            "task_type": request.task_type.value,
        }
        if request.seed is not None:
            payload["seed"] = request.seed
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens

        response = req_lib.post(
            f"http://{endpoint}:{self.backend_port}/llm-request",
            json=payload,
            timeout=300
        )

        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}: {response.text}")

        data = response.json()
        return {
            "response": data['response'].strip(),
            "endpoint_used": f"{endpoint}:{self.backend_port}",
            "priority": request.priority,
            "task_type": request.task_type.value
        }

    def _call_ollama_endpoint(self, request: LLMRequest, request_id: str, endpoint: str, messages: List[Dict]) -> Dict[str, Any]:
        """Call Ollama endpoint."""
        payload = {
            "model": request.model,
            "messages": messages,
            "options": {
                "temperature": request.temperature,
                "top_p": request.top_p,
            },
            "stream": False
        }
        if request.seed is not None:
            payload["options"]["seed"] = request.seed
        if request.max_tokens:
            payload["options"]["num_predict"] = request.max_tokens

        response = req_lib.post(
            f"http://{endpoint}:11434/api/chat",
            json=payload,
            timeout=120
        )

        if response.status_code != 200:
            raise Exception(f"HTTP {response.status_code}: {response.text}")

        data = response.json()
        return {
            "response": data['message']['content'].strip(),
            "endpoint_used": endpoint,
            "priority": request.priority,
            "task_type": request.task_type.value
        }

    async def enqueue_request(self, request: LLMRequest) -> Dict[str, Any]:
        """Enqueue a request to the selected endpoint's queue and wait for result.

        Routing decision is made at enqueue time (not dequeue time) to eliminate
        race conditions between workers.
        """
        timestamp = datetime.now().strftime("%H%M%S%f")[:-3]
        request_id = f"llm_{timestamp}"

        # Determine tier from model name
        model_tier = self._resolve_model_tier(request.model)

        # Create future for result
        future = asyncio.Future()
        enqueue_time = time.time()

        # Select endpoint and enqueue (all under lock to ensure consistent routing)
        async with self.queue_lock:
            # Check total queue capacity
            total_queued = sum(len(q) for q in self.endpoint_queues.values())
            if total_queued >= self.max_queue_size:
                raise HTTPException(
                    status_code=503,
                    detail=f"Queue full ({total_queued}/{self.max_queue_size} requests)"
                )

            # Select endpoint at enqueue time (eliminates race condition)
            selected_endpoint = self._select_endpoint_for_request(model_tier)

            # Add to endpoint-specific queue
            heapq.heappush(
                self.endpoint_queues[selected_endpoint],
                (request.priority, self.request_counter, request, future, request_id, enqueue_time)
            )
            self.request_counter += 1
            endpoint_queue_size = len(self.endpoint_queues[selected_endpoint])
            total_queued = sum(len(q) for q in self.endpoint_queues.values())

        # Determine if this was native or spillover routing
        is_native = selected_endpoint in self.tier_native_endpoints.get(model_tier, [])
        route_type = "native" if is_native else "spillover"

        logger.info(
            f"[{request_id}] QUEUED tier={model_tier} -> {selected_endpoint} ({route_type}) "
            f"queue={endpoint_queue_size} total={total_queued} active={self.active_requests}"
        )

        # Wait for result
        return await future

    def get_status(self) -> Dict[str, Any]:
        """Get server status."""
        # Queue breakdown by endpoint
        queue_by_endpoint = {ep: len(queue) for ep, queue in self.endpoint_queues.items()}
        total_queued = sum(queue_by_endpoint.values())

        # Queue breakdown by priority (across all endpoints)
        queue_by_priority = {}
        for endpoint, queue in self.endpoint_queues.items():
            for item in queue:
                priority = item[0]  # First element is priority
                queue_by_priority[priority] = queue_by_priority.get(priority, 0) + 1

        # Endpoint status with queue depth
        endpoint_info = {}
        for endpoint in self.backend_endpoints:
            # Determine native tier for this endpoint
            native_tier = None
            for tier, eps in self.tier_native_endpoints.items():
                if endpoint in eps:
                    native_tier = tier
                    break

            endpoint_info[endpoint] = {
                "healthy": self.endpoint_status.get(endpoint, False),
                "active_requests": self.endpoint_active.get(endpoint, 0),
                "queued_requests": len(self.endpoint_queues.get(endpoint, [])),
                "tiers": self.endpoint_tiers.get(endpoint, []),
                "native_tier": native_tier
            }

        return {
            "status": "healthy",
            "backend_type": self.backend_type,
            "backend_port": self.backend_port,
            "backend_endpoints": self.backend_endpoints,
            "spillover_threshold": self.spillover_threshold,
            "active_requests": self.active_requests,
            "queued_requests": total_queued,
            "queue_by_endpoint": queue_by_endpoint,
            "total_requests_processed": self.total_requests,
            "uptime_seconds": time.time() - self.start_time,
            "endpoint_status": endpoint_info,
            "queue_by_priority": queue_by_priority,
            "requests_by_task_type": dict(self.requests_by_task_type),
            "tier_native_endpoints": self.tier_native_endpoints,
            "tier_fallback_endpoints": self.tier_fallback_endpoints
        }


# =============================================================================
# FastAPI Application
# =============================================================================

balancer: Optional[LLMBalancer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan manager."""
    global balancer

    logger.info("=" * 60)
    logger.info("Starting LLM Load Balancer")
    logger.info("=" * 60)

    max_queue = int(os.environ.get('LLM_MAX_QUEUE_SIZE', 200))

    balancer = LLMBalancer(max_queue_size=max_queue)
    await balancer.initialize()

    yield

    if balancer:
        await balancer.shutdown()


app = FastAPI(
    title="LLM Load Balancer",
    description="Load-balanced LLM access across multiple MLX/Ollama endpoints",
    version="2.0.0",
    lifespan=lifespan
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}


@app.get("/status", response_model=ServerStatus)
async def get_status():
    """Get server status."""
    if not balancer:
        raise HTTPException(status_code=503, detail="Server not initialized")
    return balancer.get_status()


@app.post("/llm-request", response_model=LLMResponse)
async def llm_request(request: LLMRequest, req: Request):
    """Process LLM request."""
    if not balancer:
        raise HTTPException(status_code=503, detail="Server not initialized")

    start_time = time.time()
    request_id = f"llm_{datetime.now().strftime('%H%M%S%f')[:-3]}"

    # Get client info
    client_ip = req.client.host if req.client else 'unknown'

    try:
        response_data = await balancer.enqueue_request(request)
        total_time = time.time() - start_time

        return LLMResponse(
            response=response_data['response'],
            model_used=request.model,
            processing_time=total_time,
            request_id=request_id,
            endpoint_used=response_data['endpoint_used'],
            priority=response_data['priority'],
            task_type=response_data['task_type']
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-llm-request", response_model=BatchLLMResponse)
async def batch_llm_request(batch: BatchLLMRequest, req: Request):
    """Process multiple LLM requests as a batch.

    All requests are enqueued atomically with the same priority and processed
    concurrently. Results are returned in the same order as the input requests.
    """
    if not balancer:
        raise HTTPException(status_code=503, detail="Server not initialized")

    start_time = time.time()
    batch_id = f"batch_{datetime.now().strftime('%H%M%S%f')[:-3]}"

    logger.info(f"[{batch_id}] Received batch request with {len(batch.requests)} items")

    try:
        # Enqueue all requests concurrently
        tasks = [balancer.enqueue_request(request) for request in batch.requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Build responses in order
        responses = []
        for i, (request, result) in enumerate(zip(batch.requests, results)):
            if isinstance(result, Exception):
                logger.error(f"[{batch_id}] Request {i} failed: {result}")
                # Return error response for failed request
                responses.append(LLMResponse(
                    response=f"Error: {str(result)}",
                    model_used=request.model,
                    processing_time=0.0,
                    request_id=f"{batch_id}_{i}",
                    endpoint_used="error",
                    priority=request.priority,
                    task_type=request.task_type.value
                ))
            else:
                responses.append(LLMResponse(
                    response=result['response'],
                    model_used=request.model,
                    processing_time=0.0,  # Individual times not tracked in batch
                    request_id=f"{batch_id}_{i}",
                    endpoint_used=result['endpoint_used'],
                    priority=result['priority'],
                    task_type=result['task_type']
                ))

        total_time = time.time() - start_time
        logger.info(f"[{batch_id}] Batch completed in {total_time:.2f}s")

        return BatchLLMResponse(
            responses=responses,
            total_processing_time=total_time,
            request_count=len(batch.requests)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{batch_id}] Batch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["loggers"]["uvicorn"]["level"] = "WARNING"
    log_config["loggers"]["uvicorn.error"]["level"] = "WARNING"
    log_config["loggers"]["uvicorn.access"]["level"] = "WARNING"

    logger.info("Starting LLM Load Balancer on port 8002")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8002,
        log_config=log_config
    )
