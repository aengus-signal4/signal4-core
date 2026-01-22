#!/usr/bin/env python3
"""
MLX-based LLM Server for Stitch Processing
===========================================

FastAPI server that provides LLM access using MLX for Apple Silicon acceleration.
Orchestrates requests across multiple MLX instances on the local network.

PREREQUISITES:
- MLX installed: `pip install mlx-lm`
- Models downloaded locally or will be pulled on first use
- Apple Silicon Mac (M1/M2/M3 series)

Key features:
1. Native MLX acceleration for Apple Silicon
2. Three-tier model system (tier_1/tier_2/tier_3) - automatically upgrades to better model if loaded
3. Priority-based queuing (1-99, lower numbers = higher priority, default=1)
4. Task-type routing (stitch->worker0 first, others->any endpoint)
5. Distributed orchestration across multiple MLX servers
6. Per-worker model restrictions (e.g., 80B only on worker0)
7. Intelligent routing based on model capabilities
8. Health check and status endpoints with queue analytics
9. Keeps models warm in memory (configurable timeout)

Three-tier model system:
- Tier 1 (best):    mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit  (aliases: tier_1, tier1, best, 80b)
- Tier 2 (balanced): mlx-community/Qwen3-4B-Instruct-2507-4bit      (aliases: tier_2, tier2, balanced, 4b)
- Tier 3 (fastest):  mlx-community/LFM2-8B-A1B-4bit                  (aliases: tier_3, tier3, fastest, 8b)

Usage examples:
  request = {"model": "tier_1", "messages": [...]}  # Request tier 1, may get tier 1 if loaded
  request = {"model": "tier_2", "messages": [...]}  # Request tier 2, may get tier 1 if loaded
  request = {"model": "tier_3", "messages": [...]}  # Request tier 3, may get tier 1/2 if loaded

Response includes "model_used" field showing actual model that processed the request.
"""

import asyncio
import logging
import os
import time
import json
import random
import heapq
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from enum import Enum

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import httpx
import uvicorn

from src.utils.logger import setup_worker_logger
from src.utils.config import load_config

logger = setup_worker_logger('llm_server_mlx')
logger.setLevel(logging.INFO)

# MLX imports
try:
    from mlx_lm import load, generate
    from mlx_lm.generate import stream_generate
    from mlx_lm.sample_utils import make_sampler
    MLX_AVAILABLE = True
except ImportError:
    logger.warning("mlx-lm not available - install with: pip install mlx-lm")
    MLX_AVAILABLE = False


# Model configurations with hierarchy (rank 1 = best quality)
MODEL_CONFIGS = {
    "mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit": {
        "rank": 1,  # Tier 1 - Best quality
        "aliases": ["qwen3:80b", "80b", "large", "tier_1", "tier1", "best"],
        "description": "80B parameter model - best quality"
    },
    "mlx-community/Qwen3-4B-Instruct-2507-4bit": {
        "rank": 2,  # Tier 2 - Balanced
        "aliases": ["qwen3:4b-instruct", "qwen3:4b", "4b", "medium", "tier_2", "tier2", "balanced"],
        "description": "4B parameter model - balanced speed/quality"
    },
    "mlx-community/LFM2-8B-A1B-4bit": {
        "rank": 3,  # Tier 3 - Fastest
        "aliases": ["lfm2:8b", "8b", "small", "fast", "tier_3", "tier3", "fastest"],
        "description": "8B parameter model - fastest"
    }
}


async def check_mlx_endpoint(endpoint: str, timeout: float = 5.0) -> bool:
    """Check if an MLX endpoint is available."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"http://{endpoint}:8003/health")
            if response.status_code == 200:
                return True
            else:
                logger.warning(f"MLX endpoint {endpoint} returned status {response.status_code}")
                return False
    except Exception as e:
        logger.warning(f"MLX endpoint {endpoint} not available: {e}")
        return False


async def get_available_models(endpoint: str, timeout: float = 10.0) -> List[str]:
    """Get list of available models from an MLX endpoint."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"http://{endpoint}:8003/status")
            if response.status_code == 200:
                data = response.json()
                return data.get('models_loaded', [])
    except Exception as e:
        logger.warning(f"Failed to get models from {endpoint}: {e}")
    return []


# Request/Response models
class TaskType(str, Enum):
    """Supported task types for routing."""
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
    model: str = Field(
        default="tier_2",
        description="Model to use - supports tier_1 (80B, best), tier_2 (4B, balanced), tier_3 (1.7B, fastest) or full model names"
    )
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, description="Max tokens for response")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p sampling")
    seed: Optional[int] = Field(default=42, description="Random seed for reproducibility")
    priority: int = Field(default=1, ge=1, le=99, description="Request priority (1-99, lower = higher priority)")
    task_type: TaskType = Field(default=TaskType.STITCH, description="Task type for endpoint routing")


class LLMResponse(BaseModel):
    """Generic LLM response."""
    response: str
    model_used: str
    processing_time: float
    request_id: str
    endpoint_used: str
    priority: int
    task_type: str


class ClassificationRequest(BaseModel):
    """Single classification request for batch processing"""
    segment_id: int = Field(..., description="Unique segment identifier")
    text: str = Field(..., description="Text to classify")
    prompt: Optional[str] = Field(default=None, description="Pre-generated prompt to use")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")


class BatchClassifyRequest(BaseModel):
    """Batch classification request optimized for theme classification"""
    phase: str = Field(..., description="Classification phase: 'theme' or 'subtheme'")
    batch_idx: int = Field(..., description="Batch index for tracking")
    requests: List[ClassificationRequest] = Field(..., description="List of classification requests")
    theme_id: Optional[int] = Field(default=None, description="Theme ID for subtheme classification")
    model: str = Field(default="tier_2", description="Model to use for classification")


class ServerStatus(BaseModel):
    """Server status information."""
    status: str
    models_loaded: List[str]
    model_status: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Detailed status for each model")
    active_requests: int
    queued_requests: int
    total_requests_processed: int
    max_concurrent_requests: int
    max_queue_size: int
    model_keep_warm_seconds: int = Field(description="Seconds to keep models warm after last use")
    uptime_seconds: float
    queue_by_priority: Dict[int, int] = Field(default_factory=dict, description="Queue count by priority level")
    queue_by_task_type: Dict[str, int] = Field(default_factory=dict, description="Queue count by task type")
    requests_by_task_type: Dict[str, int] = Field(default_factory=dict, description="Total requests processed by task type")


class MLXManager:
    """Manages MLX models and request processing across multiple MLX endpoints."""

    def __init__(self, max_concurrent: int = 5, max_queue_size: int = 100):
        self.max_concurrent = max_concurrent
        self.max_queue_size = max_queue_size
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)

        # Model management
        self.loaded_models = {}  # model_name -> (model, tokenizer)
        self.models_last_used = {}  # Track last usage time for each model
        self.model_keep_warm_seconds = 300  # 5 minutes
        self.model_lock = asyncio.Lock()

        # Per-worker model restrictions
        self.allowed_models = []  # Models allowed on this worker
        self.default_model = None  # Default model for this worker
        self.local_ip = None  # This worker's IP address

        # Request management
        self.active_requests = 0
        self.queued_requests = 0
        self.total_requests = 0
        self.start_time = time.time()
        self.request_lock = asyncio.Lock()

        # Priority queue: (priority, timestamp, request, future, request_id)
        self.request_queue = []
        self.queue_lock = asyncio.Lock()
        self.request_counter = 0

        # Background tasks
        self.unload_task = None
        self.queue_processors = []

        # Distributed MLX endpoints
        self.mlx_endpoints = ["10.0.0.34", "localhost"]  # Default: worker0 + localhost
        self.endpoint_status = {}
        self.endpoint_last_check = {}
        self.endpoint_models = {}  # Track which models are available on each endpoint
        self.health_check_interval = 30

        # Separate queues per endpoint
        self.endpoint_queues = {}
        self.endpoint_processors = {}
        self.endpoint_active_requests = {}

        # Task type routing configuration
        self.task_routing = {
            TaskType.STITCH: ["localhost", "10.0.0.34"],  # Prefer localhost, fallback to worker0
            TaskType.TEXT: None,  # Any available endpoint
            TaskType.ANALYSIS: None,
            TaskType.EMBEDDING: None,
        }

        # Statistics
        self.requests_by_task_type = {task_type.value: 0 for task_type in TaskType}

        # HTTP client for peer communication
        self.http_client = None

    async def initialize(self):
        """Initialize MLX Manager with lazy loading."""
        logger.info("=== MLX LLM Server Initialization ===")
        logger.info("Using MLX for Apple Silicon acceleration")

        if not MLX_AVAILABLE:
            raise RuntimeError("MLX not available - install with: pip install mlx-lm")

        # Load config to get endpoints and model restrictions
        try:
            config = load_config()
            workers = config.get('processing', {}).get('workers', {})
            mlx_config = config.get('processing', {}).get('llm_server_mlx', {})

            # Detect local IP address
            import socket
            hostname = socket.gethostname()
            try:
                self.local_ip = socket.gethostbyname(hostname)
            except:
                self.local_ip = "127.0.0.1"

            # Try to match local IP to worker config
            for worker_name, worker_info in workers.items():
                eth_ip = worker_info.get('eth')
                wifi_ip = worker_info.get('wifi')
                if self.local_ip in [eth_ip, wifi_ip] or hostname == worker_name:
                    self.local_ip = eth_ip  # Use eth IP as canonical
                    logger.info(f"Detected local worker as {worker_name} with IP {self.local_ip}")
                    break

            # Get model restrictions for this worker from services.model_servers.instances
            model_server_instances = config.get('services', {}).get('model_servers', {}).get('instances', {})
            worker_config_found = False

            for instance_name, instance_config in model_server_instances.items():
                if not instance_config.get('enabled', False):
                    continue
                instance_host = instance_config.get('host')
                if instance_host == self.local_ip:
                    self.allowed_models = instance_config.get('allowed_models', [])
                    self.default_model = instance_config.get('default_model')
                    worker_config_found = True
                    logger.info(f"Found model_server config for {instance_name} ({self.local_ip}):")
                    logger.info(f"  Allowed models: {self.allowed_models}")
                    logger.info(f"  Default model: {self.default_model}")
                    break

            if not worker_config_found:
                # No restrictions - allow all models
                self.allowed_models = list(MODEL_CONFIGS.keys())
                self.default_model = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
                logger.warning(f"No model_server config found for {self.local_ip}, allowing all models")

            # Discover MLX endpoints from model_server instances config
            discovered_endpoints = []
            for instance_name, instance_config in model_server_instances.items():
                if not instance_config.get('enabled', False):
                    continue
                instance_host = instance_config.get('host')
                if instance_host:
                    discovered_endpoints.append(instance_host)
                    # Store allowed models for this endpoint
                    self.endpoint_models[instance_host] = instance_config.get('allowed_models', list(MODEL_CONFIGS.keys()))

            # Add localhost
            self.mlx_endpoints = list(set(discovered_endpoints + ["localhost"]))

            # Set localhost models
            self.endpoint_models["localhost"] = self.allowed_models

            logger.info(f"Discovered MLX endpoints: {self.mlx_endpoints}")
            logger.info(f"Endpoint model capabilities: {self.endpoint_models}")

        except Exception as e:
            logger.warning(f"Could not load endpoint config: {e}, using localhost only")
            self.mlx_endpoints = ["localhost"]
            self.allowed_models = list(MODEL_CONFIGS.keys())
            self.default_model = "mlx-community/Qwen3-4B-Instruct-2507-4bit"

        # Initialize HTTP client
        self.http_client = httpx.AsyncClient(timeout=30.0)

        # Check available MLX endpoints
        logger.info("Checking MLX endpoints...")
        available_endpoints = []
        for endpoint in self.mlx_endpoints:
            if endpoint == "localhost":
                # Local endpoint is always available
                available_endpoints.append(endpoint)
                self.endpoint_status[endpoint] = True
                logger.info(f"MLX endpoint localhost is available (local)")
            else:
                if await check_mlx_endpoint(endpoint):
                    available_endpoints.append(endpoint)
                    self.endpoint_status[endpoint] = True
                    logger.info(f"MLX endpoint {endpoint} is available")
                else:
                    self.endpoint_status[endpoint] = False
                    logger.warning(f"MLX endpoint {endpoint} is not available")

        if not available_endpoints:
            raise RuntimeError("No MLX endpoints are available")

        logger.info(f"Available MLX endpoints: {available_endpoints}")
        logger.info("Models will be loaded on first use and kept warm for {} seconds".format(self.model_keep_warm_seconds))
        logger.info(f"MLX supports up to {self.max_concurrent} concurrent requests with queue size {self.max_queue_size}")

        # Start background tasks
        self.unload_task = asyncio.create_task(self._model_unloader())
        logger.info("Model unloader task started")

        # Initialize endpoint queues and processors
        for endpoint in self.mlx_endpoints:
            if self.endpoint_status.get(endpoint, False):
                self.endpoint_queues[endpoint] = []
                self.endpoint_active_requests[endpoint] = 0

                # Start processors per endpoint
                endpoint_processors = []
                for i in range(5):
                    processor = asyncio.create_task(self._endpoint_queue_processor(endpoint, i))
                    endpoint_processors.append(processor)
                    self.queue_processors.append(processor)

                self.endpoint_processors[endpoint] = endpoint_processors
                logger.info(f"Started 5 queue processors for endpoint {endpoint}")

        total_processors = sum(len(procs) for procs in self.endpoint_processors.values())
        logger.info(f"Started {total_processors} total queue processor tasks")
        logger.info(f"Task routing configuration: {dict(self.task_routing)}")

    async def _model_unloader(self):
        """Background task to unload models that haven't been used recently."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute

                current_time = time.time()
                models_to_unload = []

                async with self.model_lock:
                    for model_name, last_used in self.models_last_used.items():
                        if model_name in self.loaded_models and (current_time - last_used) > self.model_keep_warm_seconds:
                            models_to_unload.append(model_name)

                    for model_name in models_to_unload:
                        logger.info(f"Unloading model {model_name} after {self.model_keep_warm_seconds} seconds of inactivity")
                        try:
                            del self.loaded_models[model_name]
                            del self.models_last_used[model_name]
                            logger.info(f"Model {model_name} unloaded from memory")
                        except Exception as e:
                            logger.error(f"Error unloading model {model_name}: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in model unloader: {e}")

    def _resolve_model_name(self, requested_model: str) -> str:
        """Resolve model alias to full model name."""
        # Check if it's already a full model name
        if requested_model in MODEL_CONFIGS:
            return requested_model

        # Check aliases
        for model_name, config in MODEL_CONFIGS.items():
            if requested_model in config.get('aliases', []):
                return model_name

        # Default to 4B model
        logger.warning(f"Unknown model '{requested_model}', defaulting to 4B model")
        return "mlx-community/Qwen3-4B-Instruct-2507-4bit"

    def _choose_best_model(self, requested_model: str) -> str:
        """Choose the best available model based on hierarchy and restrictions."""
        resolved_model = self._resolve_model_name(requested_model)

        # Check if requested model is allowed on this worker
        if resolved_model not in self.allowed_models:
            logger.warning(f"Requested model {resolved_model} not allowed on this worker ({self.local_ip})")
            # Find best allowed model as fallback
            allowed_by_rank = sorted(
                [m for m in self.allowed_models],
                key=lambda m: MODEL_CONFIGS[m]['rank']
            )
            if allowed_by_rank:
                resolved_model = allowed_by_rank[0]  # Best ranked allowed model
                logger.info(f"Using best allowed model: {resolved_model}")
            else:
                raise RuntimeError(f"No allowed models configured for this worker")

        requested_rank = MODEL_CONFIGS[resolved_model]['rank']

        # If a better model is already loaded AND allowed, use it
        best_model = resolved_model
        best_rank = requested_rank

        for model_name in self.loaded_models.keys():
            # Only consider models that are allowed on this worker
            if model_name not in self.allowed_models:
                continue

            model_rank = MODEL_CONFIGS[model_name]['rank']
            if model_rank < best_rank:  # Lower rank = better model
                best_model = model_name
                best_rank = model_rank
                logger.info(f"Using better loaded model {model_name} instead of requested {requested_model}")

        return best_model

    async def _ensure_model_loaded(self, model_name: str) -> Tuple[Any, Any]:
        """Ensure a model is loaded and return (model, tokenizer)."""
        # Check if model is allowed on this worker
        if model_name not in self.allowed_models:
            raise RuntimeError(
                f"Model {model_name} is not allowed on this worker ({self.local_ip}). "
                f"Allowed models: {self.allowed_models}"
            )

        async with self.model_lock:
            # Update last used time
            self.models_last_used[model_name] = time.time()

            # Return if already loaded
            if model_name in self.loaded_models:
                return self.loaded_models[model_name]

            # Load model
            logger.info(f"Loading model {model_name}...")
            try:
                model, tokenizer = load(model_name)
                self.loaded_models[model_name] = (model, tokenizer)
                logger.info(f"Successfully loaded model {model_name}")
                return (model, tokenizer)
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                raise

    async def _endpoint_queue_processor(self, endpoint: str, processor_id: int):
        """Process requests from a specific endpoint's queue."""
        logger.info(f"Endpoint {endpoint} queue processor {processor_id} started")
        while True:
            try:
                request_item = await self._get_next_endpoint_request(endpoint)
                if not request_item:
                    await asyncio.sleep(0.1)
                    continue

                priority, timestamp, request, future, request_id = request_item

                async with self.request_lock:
                    self.active_requests += 1
                    self.queued_requests -= 1
                    self.endpoint_active_requests[endpoint] += 1

                logger.debug(f"[{request_id}] Processing request (endpoint {endpoint}, processor {processor_id}, priority {priority})")
                start_time = time.time()

                try:
                    # Process locally if this is localhost, otherwise forward
                    if endpoint == "localhost":
                        response_data = await self._process_llm_request_local(request, request_id)
                    else:
                        response_data = await self._process_llm_request_remote(request, request_id, endpoint)

                    processing_time = time.time() - start_time

                    # Clean completion log
                    model_used = response_data.get('model_used', 'unknown')
                    # Extract short model name (e.g., "80B" from full path)
                    if '80B' in model_used:
                        model_short = 'tier_1'
                    elif '30B' in model_used or '4B' in model_used:
                        model_short = 'tier_2'
                    elif '8B' in model_used:
                        model_short = 'tier_3'
                    else:
                        model_short = model_used.split('/')[-1][:20]

                    logger.info(f"[{request_id}] Completed in {processing_time:5.2f}s ({model_short})")

                    self.total_requests += 1
                    self.requests_by_task_type[request.task_type.value] += 1
                    future.set_result(response_data)

                except Exception as e:
                    processing_time = time.time() - start_time
                    logger.error(f"[{request_id}] LLM Request on {endpoint} failed after {processing_time:.2f}s: {e}")
                    future.set_exception(e)

                finally:
                    async with self.request_lock:
                        self.active_requests -= 1
                        self.endpoint_active_requests[endpoint] -= 1

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in endpoint {endpoint} queue processor {processor_id}: {e}")

    async def _get_next_endpoint_request(self, endpoint: str) -> Optional[Tuple]:
        """Get the next highest priority request from an endpoint's queue."""
        async with self.queue_lock:
            endpoint_queue = self.endpoint_queues.get(endpoint, [])
            if endpoint_queue:
                return heapq.heappop(endpoint_queue)
            return None

    async def _process_llm_request_local(self, request: LLMRequest, request_id: str) -> Dict[str, Any]:
        """Process LLM request locally using MLX."""
        # Choose best model based on hierarchy
        requested_model = self._resolve_model_name(request.model)
        best_model = self._choose_best_model(request.model)

        # Ensure model is loaded
        model, tokenizer = await self._ensure_model_loaded(best_model)

        # Update last used time
        self.models_last_used[best_model] = time.time()

        # Build prompt from messages
        if tokenizer.chat_template is not None:
            messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
            prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        else:
            # Fallback to simple concatenation
            prompt = "\n\n".join([f"{msg.role}: {msg.content}" for msg in request.messages])

        # Generate response in executor to avoid blocking
        loop = asyncio.get_event_loop()
        response_data = await loop.run_in_executor(
            self.executor,
            self._generate_mlx_response,
            model,
            tokenizer,
            prompt,
            request
        )

        return {
            "response": response_data['text'],
            "model_used": best_model,
            "endpoint_used": "localhost",
            "priority": request.priority,
            "task_type": request.task_type.value,
            "mlx_stats": response_data.get('stats', {})
        }

    def _generate_mlx_response(self, model, tokenizer, prompt: str, request: LLMRequest) -> Dict[str, Any]:
        """Generate response using MLX (runs in executor)."""
        try:
            # Create sampler with temperature and top_p
            sampler = make_sampler(
                temp=request.temperature,
                top_p=request.top_p
            )

            # Use stream_generate to get stats
            text_parts = []
            last_response = None

            for response in stream_generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=request.max_tokens or 2048,
                sampler=sampler
            ):
                text_parts.append(response.text)
                last_response = response

            full_text = "".join(text_parts).strip()

            # Extract stats from last response
            stats = {}
            if last_response:
                stats = {
                    "prompt_tokens": last_response.prompt_tokens,
                    "generation_tokens": last_response.generation_tokens,
                    "prompt_tps": round(last_response.prompt_tps, 2),
                    "generation_tps": round(last_response.generation_tps, 2),
                    "peak_memory_gb": round(last_response.peak_memory, 3),
                    "finish_reason": last_response.finish_reason
                }

            return {
                "text": full_text,
                "stats": stats
            }
        except Exception as e:
            logger.error(f"MLX generation error: {e}")
            raise

    async def _process_llm_request_remote(self, request: LLMRequest, request_id: str, endpoint: str) -> Dict[str, Any]:
        """Forward request to remote MLX endpoint."""
        try:
            response = await self.http_client.post(
                f"http://{endpoint}:8003/llm-request",
                json=request.dict(),
                timeout=120.0
            )

            if response.status_code == 200:
                data = response.json()
                return {
                    "response": data['response'],
                    "model_used": data.get('model_used', request.model),
                    "endpoint_used": endpoint,
                    "priority": request.priority,
                    "task_type": request.task_type.value,
                    "mlx_stats": data.get('mlx_stats', {})
                }
            else:
                raise Exception(f"Remote endpoint returned {response.status_code}: {response.text}")

        except Exception as e:
            logger.error(f"Error forwarding to {endpoint}: {e}")
            self.endpoint_status[endpoint] = False
            raise

    async def _choose_best_endpoint(self, request: LLMRequest) -> Optional[str]:
        """Choose the best endpoint for this request considering model capabilities."""
        # Resolve the requested model
        requested_model = self._resolve_model_name(request.model)

        # Get healthy endpoints
        healthy_endpoints = [ep for ep in self.mlx_endpoints if self.endpoint_status.get(ep, False)]
        logger.debug(f"Healthy endpoints: {healthy_endpoints}")

        if not healthy_endpoints:
            logger.warning("No healthy endpoints available")
            return None

        # Filter endpoints that support the requested model or better
        requested_rank = MODEL_CONFIGS[requested_model]['rank']
        capable_endpoints = []

        for endpoint in healthy_endpoints:
            endpoint_allowed_models = self.endpoint_models.get(endpoint, [])
            # Check if endpoint has the exact model or a better model
            for model in endpoint_allowed_models:
                if model in MODEL_CONFIGS:
                    model_rank = MODEL_CONFIGS[model]['rank']
                    if model_rank <= requested_rank:  # Can handle this or better
                        capable_endpoints.append(endpoint)
                        break

        logger.debug(f"Capable endpoints for {requested_model}: {capable_endpoints}")

        if not capable_endpoints:
            logger.warning(f"No endpoints capable of handling model {requested_model}, using any available")
            capable_endpoints = healthy_endpoints

        # Apply task-type preferences within capable endpoints
        preferred_endpoints = self.task_routing.get(request.task_type, None)
        logger.debug(f"Task routing preference for {request.task_type}: {preferred_endpoints}")

        if preferred_endpoints:
            healthy_preferred = [ep for ep in preferred_endpoints if ep in capable_endpoints]
            logger.debug(f"Healthy preferred endpoints: {healthy_preferred}")
            if healthy_preferred:
                chosen = self._choose_least_loaded_endpoint(healthy_preferred)
                logger.info(f"Chose endpoint {chosen} for {request.task_type} task (model: {requested_model})")
                return chosen

        chosen = self._choose_least_loaded_endpoint(capable_endpoints)
        logger.info(f"Chose endpoint {chosen} for {request.task_type} task (no preference match)")
        return chosen

    def _choose_least_loaded_endpoint(self, endpoints: List[str]) -> str:
        """Choose the endpoint with the least current load."""
        if not endpoints:
            return None

        load_scores = {}
        for endpoint in endpoints:
            active_requests = self.endpoint_active_requests.get(endpoint, 0)
            queue_length = len(self.endpoint_queues.get(endpoint, []))
            load_scores[endpoint] = active_requests + queue_length

        return min(load_scores, key=load_scores.get)

    async def process_llm_request(self, request: LLMRequest) -> Dict[str, Any]:
        """Process a generic LLM request via priority queue."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        request_id = f"llm_{timestamp.replace(':', '').replace('.', '')}"

        logger.debug(f"[{request_id}] LLM Request - Model: {request.model}, Priority: {request.priority}")

        # Check if queue is full
        async with self.queue_lock:
            current_queue_size = sum(len(q) for q in self.endpoint_queues.values())

        if current_queue_size >= self.max_queue_size:
            logger.warning(f"[{request_id}] Queue full, rejecting request")
            raise HTTPException(
                status_code=503,
                detail=f"Server queue full ({self.max_queue_size} requests queued)"
            )

        # Create future for response
        future = asyncio.Future()

        # Choose best endpoint
        chosen_endpoint = await self._choose_best_endpoint(request)
        if not chosen_endpoint:
            raise HTTPException(status_code=503, detail="No healthy MLX endpoints available")

        # Add to endpoint's priority queue
        async with self.queue_lock:
            priority_item = (request.priority, self.request_counter, request, future, request_id)
            if chosen_endpoint not in self.endpoint_queues:
                self.endpoint_queues[chosen_endpoint] = []
            heapq.heappush(self.endpoint_queues[chosen_endpoint], priority_item)
            self.request_counter += 1

        async with self.request_lock:
            self.queued_requests += 1

        logger.debug(f"[{request_id}] Request queued with priority {request.priority} (active: {self.active_requests}, queued: {self.queued_requests})")

        return await future

    async def process_classification_batch(self, requests: List[ClassificationRequest], phase: str, batch_idx: int) -> List[Dict[str, Any]]:
        """Process a batch of classification requests."""
        logger.info(f"Processing classification batch {batch_idx} with {len(requests)} {phase} requests")

        tasks = []
        for idx, req in enumerate(requests):
            prompt = req.prompt if req.prompt else f"Analyze the following text:\n{req.text[:2000]}\n\nProvide your response:"

            if phase == 'validation':
                system_msg = "You are a precise classification assistant. Respond only with a number between 0.0 and 1.0."
                max_tokens = 10
                priority = 3
            else:
                system_msg = "You are a precise classification assistant. Respond only with the requested numbers."
                max_tokens = 50
                priority = 2

            llm_request = LLMRequest(
                messages=[
                    LLMMessage(role="system", content=system_msg),
                    LLMMessage(role="user", content=prompt)
                ],
                model=request.model,
                temperature=0.1,
                max_tokens=max_tokens,
                priority=priority,
                task_type=TaskType.TEXT
            )

            task = asyncio.create_task(self._process_single_classification(llm_request, req.segment_id, idx))
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        logger.info(f"Completed batch {batch_idx}: processed {len(results)} {phase} classifications")
        return results

    async def _process_single_classification(self, llm_request: LLMRequest, segment_id: int, idx: int) -> Dict[str, Any]:
        """Process a single classification request."""
        try:
            result = await self.process_llm_request(llm_request)

            response_text = result['response'].strip()
            ids = []
            if response_text != "0":
                import re
                numbers = re.findall(r'\d+', response_text)
                ids = [int(num) for num in numbers if num.isdigit()]

            return {
                'segment_id': segment_id,
                'index': idx,
                'ids': ids,
                'response': response_text,
                'processing_time': result.get('processing_time', 0)
            }
        except Exception as e:
            logger.error(f"Error processing classification for segment {segment_id}: {e}")
            return {
                'segment_id': segment_id,
                'index': idx,
                'ids': [],
                'response': 'ERROR',
                'error': str(e)
            }

    def get_status(self) -> Dict[str, Any]:
        """Get current server status."""
        current_time = time.time()
        model_status = {}

        for model_name in self.loaded_models.keys():
            last_used = self.models_last_used.get(model_name, 0)
            time_since_use = current_time - last_used if last_used else 0
            time_until_unload = max(0, self.model_keep_warm_seconds - time_since_use)
            model_status[model_name] = {
                "loaded": True,
                "last_used_seconds_ago": round(time_since_use, 1),
                "unload_in_seconds": round(time_until_unload, 1),
                "rank": MODEL_CONFIGS[model_name]['rank']
            }

        endpoint_info = {}
        for endpoint in self.mlx_endpoints:
            endpoint_info[endpoint] = {
                "healthy": self.endpoint_status.get(endpoint, False),
                "active_requests": self.endpoint_active_requests.get(endpoint, 0),
                "queued_requests": len(self.endpoint_queues.get(endpoint, [])),
            }

        return {
            "status": "healthy",
            "models_loaded": list(self.loaded_models.keys()),
            "model_status": model_status,
            "active_requests": self.active_requests,
            "queued_requests": self.queued_requests,
            "total_requests_processed": self.total_requests,
            "max_concurrent_requests": self.max_concurrent,
            "max_queue_size": self.max_queue_size,
            "model_keep_warm_seconds": self.model_keep_warm_seconds,
            "uptime_seconds": time.time() - self.start_time,
            "mlx_endpoints": self.mlx_endpoints,
            "endpoint_status": endpoint_info,
            "requests_by_task_type": dict(self.requests_by_task_type),
        }


# Global MLX manager instance
mlx_manager: Optional[MLXManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan manager."""
    global mlx_manager

    logger.info("=" * 60)
    logger.info("Starting MLX LLM Server (Apple Silicon acceleration)")
    logger.info("=" * 60)

    max_concurrent = int(os.environ.get('LLM_MAX_CONCURRENT', 5))
    max_queue_size = int(os.environ.get('LLM_MAX_QUEUE_SIZE', 100))
    logger.info(f"Max concurrent requests: {max_concurrent}")
    logger.info(f"Max queue size: {max_queue_size}")

    mlx_manager = MLXManager(max_concurrent=max_concurrent, max_queue_size=max_queue_size)
    await mlx_manager.initialize()
    logger.info("MLX LLM Server ready")

    yield

    logger.info("Shutting down MLX LLM Server...")
    if mlx_manager:
        if mlx_manager.unload_task:
            mlx_manager.unload_task.cancel()
            try:
                await mlx_manager.unload_task
            except asyncio.CancelledError:
                pass
        for processor in mlx_manager.queue_processors:
            processor.cancel()
        await asyncio.gather(*mlx_manager.queue_processors, return_exceptions=True)
        if mlx_manager.http_client:
            await mlx_manager.http_client.aclose()
        mlx_manager.executor.shutdown(wait=True)
    logger.info("MLX LLM Server shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="MLX LLM Server for Stitch Processing",
    description="Provides LLM access using MLX for Apple Silicon acceleration",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}


@app.get("/status", response_model=ServerStatus)
async def get_status():
    """Get server status."""
    if not mlx_manager:
        raise HTTPException(status_code=503, detail="Server not initialized")
    return mlx_manager.get_status()


@app.post("/llm-request", response_model=LLMResponse)
async def llm_request(request: LLMRequest, request_obj: Request):
    """Process LLM request with messages."""
    if not mlx_manager:
        raise HTTPException(status_code=503, detail="Server not initialized")

    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    request_id = f"llm_{timestamp.replace(':', '').replace('.', '')}"
    start_time = time.time()

    try:
        response_data = await mlx_manager.process_llm_request(request)
        total_time = time.time() - start_time

        if mlx_manager.total_requests % 20 == 0:
            logger.info(f"[{timestamp}] BATCH PROGRESS - Completed: {mlx_manager.total_requests}, Time: {total_time:.3f}s")

        return LLMResponse(
            response=response_data['response'],
            model_used=response_data['model_used'],
            processing_time=total_time,
            request_id=request_id,
            endpoint_used=response_data['endpoint_used'],
            priority=response_data['priority'],
            task_type=response_data['task_type']
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing LLM request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-classify")
async def batch_classify(request: BatchClassifyRequest):
    """Process batch of classification requests."""
    if not mlx_manager:
        raise HTTPException(status_code=503, detail="Server not initialized")

    start_time = time.time()
    batch_size = len(request.requests)

    logger.info(f"Received batch {request.batch_idx} with {batch_size} {request.phase} requests")

    try:
        results = await mlx_manager.process_classification_batch(
            request.requests,
            request.phase,
            request.batch_idx
        )

        results.sort(key=lambda x: x['index'])
        total_time = time.time() - start_time
        logger.info(f"Completed batch {request.batch_idx} in {total_time:.2f}s")

        return {
            'batch_idx': request.batch_idx,
            'phase': request.phase,
            'total_requests': batch_size,
            'processing_time': total_time,
            'results': results
        }
    except Exception as e:
        logger.error(f"Error processing batch {request.batch_idx}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["loggers"]["uvicorn"]["level"] = "WARNING"
    log_config["loggers"]["uvicorn.error"]["level"] = "WARNING"
    log_config["loggers"]["uvicorn.access"]["level"] = "WARNING"

    logger.info("Starting MLX LLM Server on port 8003")

    uvicorn.run(
        "llm_server_mlx:app",
        host="0.0.0.0",
        port=8003,
        log_config=log_config
    )
