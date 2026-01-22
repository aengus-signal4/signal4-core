#!/usr/bin/env python3
"""
LLM Server for Stitch Processing
=================================

FastAPI server that provides LLM access for stitch processing stages.
Orchestrates requests across multiple Ollama instances on the local network.

PREREQUISITES:
- Start Ollama service on each target machine: `ollama serve`
- Default Ollama port is 11434
- Ensure models are pulled on target machines: `ollama pull qwen3:4b-instruct`

Key features:
1. Orchestrates requests across multiple Ollama endpoints
2. Priority-based queuing (1-99, lower numbers = higher priority, default=1)
3. Task-type routing (stitch->localhost first, others->any endpoint)
4. Handles up to N concurrent requests (configurable)
5. Load balances across available Ollama instances
6. Supports model fallback and endpoint failover
7. Health check and status endpoints with queue analytics
8. Direct HTTP API calls to Ollama (no Python library dependency)
"""

import asyncio
import logging
import os
import subprocess
import time
import json
import random
import heapq
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from src.utils.paths import get_project_root, get_config_path
from enum import Enum

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import httpx
import requests
import uvicorn
import socket

from src.utils.logger import setup_worker_logger
from src.utils.config import get_llm_config, get_max_concurrent_requests, load_config, get_llm_task_routing, get_llm_backend_config

logger = setup_worker_logger('llm_server')
logger.setLevel(logging.INFO)


async def check_ollama_endpoint(endpoint: str, timeout: float = 5.0) -> bool:
    """Check if an Ollama endpoint is available."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"http://{endpoint}:11434/api/tags")
            return response.status_code == 200
    except Exception as e:
        logger.debug(f"Ollama endpoint {endpoint} not available: {e}")
        return False

async def get_available_models(endpoint: str, timeout: float = 10.0) -> List[str]:
    """Get list of available models from an Ollama endpoint."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"http://{endpoint}:11434/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
    except Exception as e:
        logger.warning(f"Failed to get models from {endpoint}: {e}")
    return []


async def check_mlx_endpoint(endpoint: str, port: int = 8004, timeout: float = 5.0) -> bool:
    """Check if an MLX model_server endpoint is available."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"http://{endpoint}:{port}/health")
            return response.status_code == 200
    except Exception as e:
        logger.debug(f"MLX endpoint {endpoint}:{port} not available: {e}")
        return False


async def get_mlx_status(endpoint: str, port: int = 8004, timeout: float = 10.0) -> Optional[Dict[str, Any]]:
    """Get status from an MLX model_server endpoint."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"http://{endpoint}:{port}/status")
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        logger.warning(f"Failed to get MLX status from {endpoint}:{port}: {e}")
    return None


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
    model: str = Field(default="qwen3:4b-instruct", description="Model to use")
    temperature: float = Field(default=0.1, min=0.0, max=2.0)
    max_tokens: Optional[int] = Field(default=None, description="Max tokens for response")
    top_p: float = Field(default=0.9, min=0.0, max=1.0, description="Top-p sampling")
    seed: Optional[int] = Field(default=42, description="Random seed for reproducibility")
    think: bool = Field(default=False, description="Enable model thinking")
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

class BatchLLMRequest(BaseModel):
    """Batch request for processing multiple prompts at once."""
    requests: List[LLMRequest] = Field(..., description="List of individual LLM requests")
    batch_id: Optional[str] = Field(default=None, description="Optional batch identifier")
    priority: int = Field(default=2, ge=1, le=99, description="Batch priority (1-99, lower = higher priority)")
    task_type: TaskType = Field(default=TaskType.TEXT, description="Task type for endpoint routing")

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

class BatchLLMResponse(BaseModel):
    """Batch response with results for each request."""
    responses: List[LLMResponse] = Field(..., description="List of individual responses")
    batch_id: str = Field(..., description="Batch identifier")
    total_processing_time: float = Field(..., description="Total time for entire batch")
    successful_requests: int = Field(..., description="Number of successful requests")
    failed_requests: int = Field(..., description="Number of failed requests")
    batch_priority: int = Field(..., description="Priority used for batch processing")
    endpoint_used: str = Field(..., description="Primary endpoint used for batch")

class ServerStatus(BaseModel):
    """Server status information."""
    status: str
    backend_type: str = Field(description="Backend type: 'mlx' or 'ollama'")
    backend_port: int = Field(description="Backend port (8004 for mlx, 11434 for ollama)")
    backend_endpoints: List[str] = Field(description="List of backend endpoint IPs")
    models_loaded: List[str]
    model_status: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Detailed status for each model")
    active_requests: int
    queued_requests: int
    total_requests_processed: int
    max_concurrent_requests: int
    max_queue_size: int
    model_keep_warm_seconds: int = Field(description="Seconds to keep models warm after last use")
    uptime_seconds: float
    endpoint_status: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Health and load per endpoint")
    request_distribution: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Request distribution per endpoint")
    queue_by_priority: Dict[int, int] = Field(default_factory=dict, description="Queue count by priority level")
    queue_by_task_type: Dict[str, int] = Field(default_factory=dict, description="Queue count by task type")
    requests_by_task_type: Dict[str, int] = Field(default_factory=dict, description="Total requests processed by task type")
    task_routing_config: Dict[str, Any] = Field(default_factory=dict, description="Task routing configuration")


class LLMManager:
    """Manages LLM models and request processing across multiple backend endpoints (MLX or Ollama)."""

    def __init__(self, max_concurrent: int = 5, max_queue_size: int = 100):
        self.max_concurrent = max_concurrent
        self.max_queue_size = max_queue_size
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self.models_loaded = set()
        self.models_last_used = {}  # Track last usage time for each model
        self.active_requests = 0
        self.queued_requests = 0
        self.total_requests = 0
        self.start_time = time.time()
        self.request_lock = asyncio.Lock()
        # Priority queue: (priority, timestamp, request, future, request_id)
        # Note: heapq is min-heap, lower priority numbers = higher priority
        self.request_queue = []
        self.queue_lock = asyncio.Lock()
        self.request_counter = 0  # For timestamp uniqueness

        self.primary_model = "qwen3:4b-instruct"
        self.fallback_model = "qwen3:4b-instruct"
        self.model_keep_warm_seconds = 300  # 5 minutes
        self.unload_task = None
        self.queue_processors = []

        # Backend configuration - loaded from config
        backend_config = get_llm_backend_config()
        self.backend_type = backend_config['backend']  # "mlx" or "ollama"
        self.backend_port = backend_config['port']  # 8004 for mlx, 11434 for ollama
        self.backend_endpoints = backend_config['endpoints']  # List of endpoint IPs
        self.endpoint_tiers = backend_config.get('endpoint_tiers', {})  # endpoint -> supported tiers

        # For backwards compatibility, keep ollama_endpoints as alias
        self.ollama_endpoints = self.backend_endpoints
        self.endpoint_status = {}  # Track endpoint health
        self.endpoint_last_check = {}  # Track last health check time
        self.health_check_interval = 30  # seconds

        # Separate queues per endpoint for maximum throughput
        self.endpoint_queues = {}  # endpoint -> list of (priority, timestamp, request, future, request_id)
        self.endpoint_processors = {}  # endpoint -> list of processor tasks
        self.endpoint_active_requests = {}  # endpoint -> count of active requests

        # Task type routing configuration - loaded from config
        self.task_routing = {}  # Will be loaded from config
        self.config_last_modified = None  # Track config file modification time
        self.config_check_interval = 5  # Check config every 5 seconds

        # Statistics tracking
        self.requests_by_task_type = {task_type.value: 0 for task_type in TaskType}

        # HTTP client for API calls
        self.http_client = None
        
    async def _load_task_routing(self):
        """Load task routing and endpoint tier configuration from config file."""
        try:
            routing_config = get_llm_task_routing()

            # Convert string keys to TaskType enums
            new_routing = {}
            for task_type_str, endpoints in routing_config.items():
                try:
                    task_type = TaskType(task_type_str)
                    new_routing[task_type] = endpoints
                except ValueError:
                    logger.warning(f"Unknown task type in config: {task_type_str}")

            self.task_routing = new_routing

            # Also reload endpoint tiers
            backend_config = get_llm_backend_config()
            self.endpoint_tiers = backend_config.get('endpoint_tiers', {})
            # Update endpoints list in case it changed
            new_endpoints = backend_config['endpoints']
            if set(new_endpoints) != set(self.backend_endpoints):
                logger.info(f"Endpoints changed: {self.backend_endpoints} -> {new_endpoints}")
                self.backend_endpoints = new_endpoints

            # Update config modification time
            config_path = get_config_path()
            self.config_last_modified = config_path.stat().st_mtime

            logger.info(f"Task routing loaded: {dict((k.value, v) for k, v in self.task_routing.items())}")
            logger.info(f"Endpoint tiers loaded: {self.endpoint_tiers}")

        except Exception as e:
            logger.error(f"Failed to load task routing config: {e}")
            # Fall back to defaults
            self.task_routing = {
                TaskType.STITCH: ["10.0.0.4", "localhost"],
                TaskType.TEXT: None,
                TaskType.ANALYSIS: None,
                TaskType.EMBEDDING: None,
            }

    async def _check_config_reload(self):
        """Check if config file has been modified and reload if needed."""
        try:
            config_path = get_config_path()
            current_mtime = config_path.stat().st_mtime

            if self.config_last_modified is None or current_mtime > self.config_last_modified:
                logger.info("Config file changed, reloading task routing...")
                await self._load_task_routing()
                return True
            return False
        except Exception as e:
            logger.error(f"Error checking config reload: {e}")
            return False

    async def _check_endpoint_health(self, endpoint: str) -> bool:
        """Check if an endpoint is healthy (backend-aware)."""
        if self.backend_type == "mlx":
            return await check_mlx_endpoint(endpoint, self.backend_port)
        else:
            return await check_ollama_endpoint(endpoint)

    async def initialize(self):
        """Initialize LLM Manager with lazy loading."""
        logger.info("=== LLM Server Initialization ===")
        logger.info(f"Backend type: {self.backend_type.upper()}")
        logger.info(f"Backend port: {self.backend_port}")
        logger.info(f"Configured endpoints: {self.backend_endpoints}")

        # Load task routing from config
        await self._load_task_routing()

        # Initialize HTTP client
        self.http_client = httpx.AsyncClient(timeout=30.0)

        # Check available endpoints
        logger.info(f"Checking {self.backend_type.upper()} endpoints...")
        available_endpoints = []
        for endpoint in self.backend_endpoints:
            if await self._check_endpoint_health(endpoint):
                available_endpoints.append(endpoint)
                self.endpoint_status[endpoint] = True
                logger.info(f"{self.backend_type.upper()} endpoint {endpoint}:{self.backend_port} is available")
            else:
                self.endpoint_status[endpoint] = False
                logger.warning(f"{self.backend_type.upper()} endpoint {endpoint}:{self.backend_port} is not available")

        if not available_endpoints:
            raise RuntimeError(f"No {self.backend_type.upper()} endpoints are available - LLM server cannot initialize")

        logger.info(f"Available {self.backend_type.upper()} endpoints: {available_endpoints}")
        
        logger.info("Initializing LLM Manager with lazy loading...")
        logger.info(f"Models will be loaded on first use and kept warm for {self.model_keep_warm_seconds} seconds")
        logger.info(f"Backend supports up to {self.max_concurrent} concurrent requests with queue size {self.max_queue_size}")

        # Start the model unloader task (only relevant for Ollama, MLX manages its own)
        if self.backend_type == "ollama":
            self.unload_task = asyncio.create_task(self._model_unloader())
            logger.info("Model unloader task started")

        # Initialize separate queues and processors for each endpoint
        for endpoint in self.backend_endpoints:
            if self.endpoint_status.get(endpoint, False):  # Only for healthy endpoints
                self.endpoint_queues[endpoint] = []
                self.endpoint_active_requests[endpoint] = 0

                # Start processors per endpoint
                # MLX model_server handles its own concurrency, so we use 1 processor per endpoint
                # Ollama can handle multiple concurrent requests
                processors_per_endpoint = 1 if self.backend_type == "mlx" else 5
                endpoint_processors = []
                for i in range(processors_per_endpoint):
                    processor = asyncio.create_task(self._endpoint_queue_processor(endpoint, i))
                    endpoint_processors.append(processor)
                    self.queue_processors.append(processor)  # Keep in main list for shutdown

                self.endpoint_processors[endpoint] = endpoint_processors
                logger.info(f"Started {processors_per_endpoint} queue processor(s) for endpoint {endpoint}")

        total_processors = sum(len(procs) for procs in self.endpoint_processors.values())
        logger.info(f"Started {total_processors} total queue processor tasks across {len(self.endpoint_processors)} endpoints")
        logger.info(f"Task routing configuration: {dict(self.task_routing)}")
            
    async def _model_unloader(self):
        """Background task to unload models that haven't been used recently."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                current_time = time.time()
                models_to_unload = []
                
                for model, last_used in self.models_last_used.items():
                    if model in self.models_loaded and (current_time - last_used) > self.model_keep_warm_seconds:
                        models_to_unload.append(model)
                
                for model in models_to_unload:
                    logger.info(f"Unloading model {model} after {self.model_keep_warm_seconds} seconds of inactivity")
                    try:
                        # Note: Ollama doesn't have a direct unload command, but we can remove from tracking
                        self.models_loaded.discard(model)
                        del self.models_last_used[model]
                        logger.info(f"Model {model} marked as unloaded")
                    except Exception as e:
                        logger.error(f"Error unloading model {model}: {e}")
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in model unloader: {e}")
    
    async def _get_available_endpoint(self) -> Optional[str]:
        """Get an available endpoint using round-robin with health checks."""
        current_time = time.time()

        # Check endpoint health if needed
        for endpoint in self.backend_endpoints:
            last_check = self.endpoint_last_check.get(endpoint, 0)
            if current_time - last_check > self.health_check_interval:
                is_healthy = await self._check_endpoint_health(endpoint)
                self.endpoint_status[endpoint] = is_healthy
                self.endpoint_last_check[endpoint] = current_time
                if is_healthy:
                    logger.debug(f"Endpoint {endpoint} health check: OK")
                else:
                    logger.warning(f"Endpoint {endpoint} health check: FAILED")

        # Get healthy endpoints
        healthy_endpoints = [ep for ep in self.backend_endpoints if self.endpoint_status.get(ep, False)]

        if not healthy_endpoints:
            logger.error(f"No healthy {self.backend_type.upper()} endpoints available")
            return None

        # Simple round-robin selection
        return random.choice(healthy_endpoints)
    
    async def _ensure_model_loaded(self, model_name: str):
        """Ensure a model is available on at least one endpoint."""
        # Update last used time
        self.models_last_used[model_name] = time.time()

        if model_name in self.models_loaded:
            return

        # For MLX backend, models are managed by model_server.py - just mark as available
        if self.backend_type == "mlx":
            self.models_loaded.add(model_name)
            logger.debug(f"MLX backend - model {model_name} assumed available on model_server")
            return

        # Ollama: Check if model is available on any endpoint
        for endpoint in self.backend_endpoints:
            if not self.endpoint_status.get(endpoint, False):
                continue

            try:
                models = await get_available_models(endpoint)
                if model_name in models:
                    self.models_loaded.add(model_name)
                    logger.info(f"Model {model_name} found on endpoint {endpoint}")
                    return
            except Exception as e:
                logger.warning(f"Failed to check models on {endpoint}: {e}")

        # Ollama: Try to pull model on first available endpoint
        endpoint = await self._get_available_endpoint()
        if not endpoint:
            raise RuntimeError("No healthy endpoints available to pull model")

        try:
            logger.info(f"Attempting to pull model {model_name} on {endpoint}...")
            async with self.http_client as client:
                response = await client.post(
                    f"http://{endpoint}:11434/api/pull",
                    json={"name": model_name},
                    timeout=600  # 10 minutes for model pull
                )
                if response.status_code == 200:
                    self.models_loaded.add(model_name)
                    logger.info(f"Successfully pulled model {model_name} on {endpoint}")
                else:
                    raise Exception(f"Pull failed with status {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            raise
    
    async def _endpoint_queue_processor(self, endpoint: str, processor_id: int):
        """Process requests from a specific endpoint's queue for maximum throughput."""
        logger.info(f"Endpoint {endpoint} queue processor {processor_id} started")
        while True:
            try:
                # Get highest priority request from this endpoint's queue
                request_item = await self._get_next_endpoint_request(endpoint)
                if not request_item:
                    await asyncio.sleep(0.1)  # Brief sleep if no requests
                    continue
                    
                priority, timestamp, request, future, request_id = request_item
                
                async with self.request_lock:
                    self.active_requests += 1
                    self.queued_requests -= 1
                    self.endpoint_active_requests[endpoint] += 1
                
                logger.debug(f"[{request_id}] Processing request (endpoint {endpoint}, processor {processor_id}, priority {priority}, task_type {request.task_type})")
                start_time = time.time()
                
                try:
                    # Ensure model is loaded
                    await self._ensure_model_loaded(request.model)
                    
                    # Process directly on this specific endpoint
                    response_data = await asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        self._process_llm_request_on_endpoint,
                        request,
                        request_id,
                        endpoint
                    )
                    
                    processing_time = time.time() - start_time
                    logger.debug(f"[{request_id}] LLM Response from {endpoint} (took {processing_time:.2f}s): {response_data['response'][:50]}{'...' if len(response_data['response']) > 50 else ''}")
                    
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
        """Get the next highest priority request from a specific endpoint's queue."""
        async with self.queue_lock:
            endpoint_queue = self.endpoint_queues.get(endpoint, [])
            if endpoint_queue:
                return heapq.heappop(endpoint_queue)
            return None
    
    def _process_llm_request_on_endpoint(self, request: LLMRequest, request_id: str, endpoint: str) -> Dict[str, Any]:
        """Process LLM request on a specific endpoint only (backend-aware)."""
        try:
            # Update last used time for the model
            self.models_last_used[request.model] = time.time()

            # Convert messages to list format
            messages = []
            for msg in request.messages:
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })

            if self.backend_type == "mlx":
                return self._process_mlx_request(request, request_id, endpoint, messages)
            else:
                return self._process_ollama_request(request, request_id, endpoint, messages)

        except Exception as e:
            logger.error(f"[{request_id}] Error processing on {endpoint}: {e}")
            # Mark endpoint as unhealthy
            self.endpoint_status[endpoint] = False
            raise

    def _process_mlx_request(self, request: LLMRequest, request_id: str, endpoint: str, messages: List[Dict]) -> Dict[str, Any]:
        """Process request via MLX model_server endpoint."""
        # Build MLX model_server compatible payload
        payload = {
            "messages": messages,
            "model": request.model,  # model_server will resolve aliases
            "temperature": request.temperature,
            "top_p": request.top_p,
            "priority": request.priority,
        }

        if request.seed is not None:
            payload["seed"] = request.seed

        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens

        logger.info(f"[{request_id}] Sending request to MLX endpoint {endpoint}:{self.backend_port} with model {request.model} (task_type: {request.task_type})")

        import requests as req_lib
        response = req_lib.post(
            f"http://{endpoint}:{self.backend_port}/llm-request",
            json=payload,
            timeout=300  # 5 minutes timeout for MLX (large models can be slow)
        )

        if response.status_code == 200:
            data = response.json()
            result = data['response'].strip()
            logger.debug(f"[{request_id}] MLX response from {endpoint}, length: {len(result)} chars")

            return {
                "response": result,
                "endpoint_used": f"{endpoint}:{self.backend_port}",
                "priority": request.priority,
                "task_type": request.task_type.value
            }
        else:
            raise Exception(f"HTTP {response.status_code}: {response.text}")

    def _process_ollama_request(self, request: LLMRequest, request_id: str, endpoint: str, messages: List[Dict]) -> Dict[str, Any]:
        """Process request via Ollama endpoint."""
        # Build Ollama options
        options = {
            "temperature": request.temperature,
            "top_p": request.top_p,
        }

        if request.seed is not None:
            options["seed"] = request.seed

        if request.max_tokens:
            options["num_predict"] = request.max_tokens

        # Prepare Ollama request payload
        payload = {
            "model": request.model,
            "messages": messages,
            "options": options,
            "stream": False
        }

        # Only include think parameter if explicitly enabled
        if request.think:
            payload["think"] = True

        logger.info(f"[{request_id}] Sending request to Ollama endpoint {endpoint} with model {request.model} (task_type: {request.task_type})")

        import requests as req_lib
        response = req_lib.post(
            f"http://{endpoint}:11434/api/chat",
            json=payload,
            timeout=120  # 2 minutes timeout
        )

        if response.status_code == 200:
            data = response.json()
            result = data['message']['content'].strip()
            logger.debug(f"[{request_id}] Ollama response from {endpoint}, length: {len(result)} chars")

            return {
                "response": result,
                "endpoint_used": endpoint,
                "priority": request.priority,
                "task_type": request.task_type.value
            }
        else:
            raise Exception(f"HTTP {response.status_code}: {response.text}")
    
    def _normalize_endpoint(self, endpoint: str) -> str:
        """Normalize endpoint for comparison (localhost -> 10.0.0.4 on head node)."""
        # On head node, treat localhost and 10.0.0.4 as equivalent
        from src.utils.node_utils import is_head_node
        head_node = is_head_node()

        if head_node:
            if endpoint == "10.0.0.4":
                return "localhost"
            elif endpoint == "localhost":
                return "localhost"
        return endpoint

    def _resolve_model_tier(self, model_name: str) -> str:
        """Resolve model name/alias to its tier (tier_1, tier_2, tier_3)."""
        # Direct tier names
        if model_name in ["tier_1", "tier1", "80b", "large", "best"]:
            return "tier_1"
        if model_name in ["tier_2", "tier2", "30b", "medium", "balanced", "default", "qwen3:30b-instruct", "qwen3:30b"]:
            return "tier_2"
        if model_name in ["tier_3", "tier3", "4b", "small", "fast", "qwen3:4b-instruct", "qwen3:4b"]:
            return "tier_3"
        # Full model names
        if "80B" in model_name or "80b" in model_name:
            return "tier_1"
        if "30B" in model_name or "30b" in model_name:
            return "tier_2"
        if "4B" in model_name or "4b" in model_name:
            return "tier_3"
        # Default to tier_2
        return "tier_2"

    def _get_endpoints_for_tier(self, tier: str, include_fallback: bool = False) -> tuple[List[str], List[str]]:
        """Get list of endpoints that support the given tier.

        Args:
            tier: The requested tier (tier_1, tier_2, tier_3)
            include_fallback: If True, also return tier_2 endpoints as fallback for tier_3
                              (tier_1 is reserved and never used as fallback)

        Returns:
            Tuple of (primary_endpoints, fallback_endpoints)
            - primary_endpoints: Endpoints that natively support this tier
            - fallback_endpoints: tier_2 endpoints that can handle tier_3 overflow
        """
        if not self.endpoint_tiers:
            # No tier restrictions configured - all endpoints support all tiers
            return self.backend_endpoints, []

        primary_endpoints = []
        fallback_endpoints = []

        for endpoint, tiers in self.endpoint_tiers.items():
            if tier in tiers:
                # This endpoint natively supports the requested tier
                primary_endpoints.append(endpoint)
            elif include_fallback and tier == "tier_3" and "tier_2" in tiers:
                # Only tier_3 can fall back to tier_2
                # tier_1 is reserved for tier_1 requests only
                fallback_endpoints.append(endpoint)

        return primary_endpoints, fallback_endpoints

    async def _choose_best_endpoint(self, request: LLMRequest) -> Optional[str]:
        """Choose the best endpoint for this request using intelligent load balancing and tier routing.

        Tier fallback logic:
        - Lower tier requests (e.g., tier_3) can use higher tier endpoints (tier_2, tier_1) as fallback
        - Higher tier endpoints are only used if they have zero load (idle)
        - Primary (native tier) endpoints are always preferred
        """
        # Check if config needs reloading (periodic check)
        await self._check_config_reload()

        # Refresh endpoint health status if needed
        current_time = time.time()
        for endpoint in self.backend_endpoints:
            last_check = self.endpoint_last_check.get(endpoint, 0)
            if current_time - last_check > self.health_check_interval:
                is_healthy = await self._check_endpoint_health(endpoint)
                self.endpoint_status[endpoint] = is_healthy
                self.endpoint_last_check[endpoint] = current_time
                if is_healthy:
                    logger.debug(f"Endpoint {endpoint} health check: OK")
                else:
                    logger.warning(f"Endpoint {endpoint} health check: FAILED")

        # Get all healthy endpoints
        healthy_endpoints = [ep for ep in self.backend_endpoints if self.endpoint_status.get(ep, False)]

        if not healthy_endpoints:
            return None

        # For MLX backend, use tier-aware routing with fallback
        if self.backend_type == "mlx" and self.endpoint_tiers:
            model_tier = self._resolve_model_tier(request.model)
            primary_endpoints, fallback_endpoints = self._get_endpoints_for_tier(model_tier, include_fallback=True)

            # Filter to healthy endpoints
            healthy_primary = [ep for ep in healthy_endpoints if ep in primary_endpoints]
            healthy_fallback = [ep for ep in healthy_endpoints if ep in fallback_endpoints]

            logger.debug(f"Model {request.model} -> {model_tier}: primary={healthy_primary}, fallback={healthy_fallback}")

            # Strategy: Combine primary + fallback endpoints and choose least loaded
            # This distributes load across all capable endpoints rather than
            # only using fallback when primary is overloaded
            all_capable = healthy_primary + healthy_fallback

            if all_capable:
                chosen = self._choose_least_loaded_endpoint(all_capable)
                chosen_load = self._get_endpoint_load(chosen)
                is_fallback = chosen in healthy_fallback

                if is_fallback:
                    logger.info(f"ROUTE: model={request.model} tier={model_tier} -> {chosen} (tier_2 fallback) [load={chosen_load}]")
                else:
                    logger.info(f"ROUTE: model={request.model} tier={model_tier} -> {chosen} [primary, load={chosen_load}]")
                return chosen

            elif healthy_fallback:
                # No primary endpoints available, use fallback
                chosen = self._choose_least_loaded_endpoint(healthy_fallback)
                chosen_tier = self.endpoint_tiers.get(chosen, ["unknown"])[0]
                logger.info(f"ROUTE: model={request.model} tier={model_tier} -> {chosen} ({chosen_tier}) [fallback, no primary available]")
                return chosen

            else:
                logger.error(f"ROUTE FAILED: model={request.model} tier={model_tier} - no healthy endpoints")
                return None

        # Get preferred endpoints for this task type
        preferred_endpoints = self.task_routing.get(request.task_type, None)

        # If we have task-specific preferences, try those first
        if preferred_endpoints:
            # Normalize both preferred and healthy endpoints for comparison
            normalized_healthy = {self._normalize_endpoint(ep): ep for ep in healthy_endpoints}

            healthy_preferred = []
            for pref in preferred_endpoints:
                normalized_pref = self._normalize_endpoint(pref)
                if normalized_pref in normalized_healthy:
                    # Use the actual endpoint name (not the normalized one)
                    healthy_preferred.append(normalized_healthy[normalized_pref])

            if healthy_preferred:
                # Choose least loaded among preferred endpoints
                chosen = self._choose_least_loaded_endpoint(healthy_preferred)
                logger.debug(f"Task {request.task_type.value}: Chose {chosen} from preferred endpoints {healthy_preferred}")
                return chosen

        # No preferences or no healthy preferred endpoints - choose least loaded overall
        return self._choose_least_loaded_endpoint(healthy_endpoints)

    def _get_endpoint_load(self, endpoint: str) -> int:
        """Get current load (active + queued) for an endpoint."""
        active = self.endpoint_active_requests.get(endpoint, 0)
        queued = len(self.endpoint_queues.get(endpoint, []))
        return active + queued
    
    def _choose_least_loaded_endpoint(self, endpoints: List[str]) -> str:
        """Choose the endpoint with the least current load."""
        if not endpoints:
            return None

        # Calculate load score for each endpoint
        load_scores = {}
        for endpoint in endpoints:
            active_requests = self.endpoint_active_requests.get(endpoint, 0)
            queue_length = len(self.endpoint_queues.get(endpoint, []))

            # Load score = active requests + queue length
            # Lower is better
            load_scores[endpoint] = active_requests + queue_length

        # Find minimum load
        min_load = min(load_scores.values())

        # Get all endpoints with minimum load
        least_loaded = [ep for ep, load in load_scores.items() if load == min_load]

        # If multiple endpoints have same load, choose randomly for even distribution
        if len(least_loaded) > 1:
            chosen = random.choice(least_loaded)
            logger.debug(f"Load balancing: Multiple endpoints with load {min_load}, chose {chosen} from {least_loaded}")
            return chosen

        logger.debug(f"Load balancing: Chose {least_loaded[0]} with load {min_load} (loads: {load_scores})")
        return least_loaded[0]
                
    async def _queue_processor(self, processor_id: int):
        """Process requests from the priority queue."""
        logger.info(f"Queue processor {processor_id} started")
        while True:
            try:
                # Get highest priority request from queue
                request_item = await self._get_next_request()
                if not request_item:
                    await asyncio.sleep(0.1)  # Brief sleep if no requests
                    continue
                    
                priority, timestamp, request, future, request_id = request_item
                
                async with self.request_lock:
                    self.active_requests += 1
                    self.queued_requests -= 1
                
                logger.debug(f"[{request_id}] Processing request (processor {processor_id}, priority {priority}, task_type {request.task_type})")
                start_time = time.time()
                
                try:
                    # Ensure model is loaded
                    await self._ensure_model_loaded(request.model)
                    
                    # Process in executor to avoid blocking
                    response_data = await asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        self._process_llm_request_sync,
                        request,
                        request_id
                    )
                    
                    processing_time = time.time() - start_time
                    logger.debug(f"[{request_id}] LLM Response (took {processing_time:.2f}s): {response_data['response'][:50]}{'...' if len(response_data['response']) > 50 else ''}")
                    
                    self.total_requests += 1
                    self.requests_by_task_type[request.task_type.value] += 1
                    future.set_result(response_data)
                    
                except Exception as e:
                    processing_time = time.time() - start_time
                    logger.error(f"[{request_id}] LLM Request failed after {processing_time:.2f}s: {e}")
                    future.set_exception(e)
                    
                finally:
                    async with self.request_lock:
                        self.active_requests -= 1
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in queue processor {processor_id}: {e}")
    
    async def _get_next_request(self) -> Optional[Tuple]:
        """Get the next highest priority request from the queue."""
        async with self.queue_lock:
            if self.request_queue:
                return heapq.heappop(self.request_queue)
            return None
                
    async def process_llm_request(self, request: LLMRequest) -> Dict[str, Any]:
        """Process a generic LLM request via priority queue."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        request_id = f"llm_{timestamp.replace(':', '').replace('.', '')}"
        
        # Log the incoming request at debug level for batch processing
        logger.debug(f"[{request_id}] LLM Request - Model: {request.model}, Priority: {request.priority}, Task Type: {request.task_type}, Messages: {len(request.messages)}")
        for i, msg in enumerate(request.messages):
            logger.debug(f"[{request_id}] Message {i+1} ({msg.role}): {msg.content[:200]}{'...' if len(msg.content) > 200 else ''}")
        
        # Check if queue is full
        async with self.queue_lock:
            current_queue_size = len(self.request_queue)
        
        if current_queue_size >= self.max_queue_size:
            logger.warning(f"[{request_id}] Queue full, rejecting request")
            raise HTTPException(
                status_code=503,
                detail=f"Server queue full ({self.max_queue_size} requests queued)"
            )
            
        # Create future for response
        future = asyncio.Future()
        
        # Choose the best endpoint for this request using intelligent load balancing
        chosen_endpoint = await self._choose_best_endpoint(request)
        if not chosen_endpoint:
            raise HTTPException(status_code=503, detail="No healthy Ollama endpoints available")
        
        # Add to chosen endpoint's priority queue
        async with self.queue_lock:
            # heapq is min-heap, so lower priority numbers get processed first
            # Include request counter for stable sorting
            priority_item = (request.priority, self.request_counter, request, future, request_id)
            
            # Add to specific endpoint queue
            if chosen_endpoint not in self.endpoint_queues:
                self.endpoint_queues[chosen_endpoint] = []
            heapq.heappush(self.endpoint_queues[chosen_endpoint], priority_item)
            self.request_counter += 1
        
        async with self.request_lock:
            self.queued_requests += 1
            
        logger.debug(f"[{request_id}] Request queued with priority {request.priority} (active: {self.active_requests}, queued: {self.queued_requests})")
        
        # Wait for result
        return await future
    
    async def process_classification_batch(self, requests: List[ClassificationRequest], phase: str, batch_idx: int) -> List[Dict[str, Any]]:
        """Process a batch of classification requests with queue saturation"""
        logger.info(f"Processing classification batch {batch_idx} with {len(requests)} {phase} requests")
        
        # Create tasks for all requests
        tasks = []
        for idx, req in enumerate(requests):
            # Build LLM request using provided prompt or fallback
            if req.prompt:
                prompt = req.prompt
            else:
                # Simple fallback for any phase
                prompt = f"Analyze the following text:\n{req.text[:2000]}\n\nProvide your response:"
            
            # Determine response parameters based on phase
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
                model="qwen3:4b-instruct",
                temperature=0.1,
                max_tokens=max_tokens,
                priority=priority,
                task_type=TaskType.TEXT
            )
            
            # Create task for this request
            task = asyncio.create_task(
                self._process_single_classification(llm_request, req.segment_id, idx)
            )
            tasks.append(task)
        
        # Process all tasks concurrently (limited by our queue)
        results = await asyncio.gather(*tasks)
        
        logger.info(f"Completed batch {batch_idx}: processed {len(results)} {phase} classifications")
        return results
    async def _process_single_classification(self, llm_request: LLMRequest, segment_id: int, idx: int) -> Dict[str, Any]:
        """Process a single classification request through the queue system"""
        try:
            # Process through existing queue system
            result = await self.process_llm_request(llm_request)
            
            # Parse the response to extract theme/subtheme IDs
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
                
    def _process_llm_request_sync(self, request: LLMRequest, request_id: str) -> Dict[str, Any]:
        """Synchronous processing of LLM request via HTTP with task-based routing (backend-aware)."""
        try:
            # Update last used time for the model
            self.models_last_used[request.model] = time.time()

            # Convert messages to list format
            messages = []
            for msg in request.messages:
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })

            logger.debug(f"[{request_id}] Calling {self.backend_type.upper()} HTTP API for task type {request.task_type}")

            # Get preferred endpoints for this task type
            preferred_endpoints = self.task_routing.get(request.task_type, None)
            endpoints_to_try = []

            if preferred_endpoints:
                # Try preferred endpoints first
                healthy_preferred = [ep for ep in preferred_endpoints if self.endpoint_status.get(ep, False)]
                endpoints_to_try.extend(healthy_preferred)
                # Add other healthy endpoints as fallback
                other_endpoints = [ep for ep in self.backend_endpoints
                                if ep not in preferred_endpoints and self.endpoint_status.get(ep, False)]
                endpoints_to_try.extend(other_endpoints)
            else:
                # No preferences, try all healthy endpoints
                endpoints_to_try = [ep for ep in self.backend_endpoints if self.endpoint_status.get(ep, False)]

            if not endpoints_to_try:
                raise Exception("No healthy endpoints available")

            # Try endpoints until one succeeds
            last_error = None
            for endpoint in endpoints_to_try:
                try:
                    if self.backend_type == "mlx":
                        result = self._process_mlx_request(request, request_id, endpoint, messages)
                    else:
                        result = self._process_ollama_request(request, request_id, endpoint, messages)
                    return result

                except Exception as e:
                    logger.warning(f"[{request_id}] Endpoint {endpoint} failed: {e}")
                    last_error = str(e)
                    # Mark endpoint as unhealthy
                    self.endpoint_status[endpoint] = False

            # If we get here, all endpoints failed
            error_msg = f"All {self.backend_type.upper()} endpoints failed. Last error: {last_error}"
            logger.error(f"[{request_id}] {error_msg}")
            raise Exception(error_msg)

        except Exception as e:
            logger.error(f"[{request_id}] Error in LLM processing: {e}")
            raise
            
    def get_status(self) -> Dict[str, Any]:
        """Get current server status."""
        current_time = time.time()
        model_status = {}

        for model in self.models_loaded:
            last_used = self.models_last_used.get(model, 0)
            time_since_use = current_time - last_used if last_used else 0
            time_until_unload = max(0, self.model_keep_warm_seconds - time_since_use)
            model_status[model] = {
                "loaded": True,
                "last_used_seconds_ago": round(time_since_use, 1),
                "unload_in_seconds": round(time_until_unload, 1)
            }

        # Endpoint status
        endpoint_info = {}
        for endpoint in self.backend_endpoints:
            endpoint_info[endpoint] = {
                "healthy": self.endpoint_status.get(endpoint, False),
                "last_check": self.endpoint_last_check.get(endpoint, 0),
                "active_requests": self.endpoint_active_requests.get(endpoint, 0),
                "queued_requests": len(self.endpoint_queues.get(endpoint, [])),
                "processors": len(self.endpoint_processors.get(endpoint, []))
            }

        # Calculate load distribution
        request_distribution = {}
        for endpoint in self.backend_endpoints:
            active = self.endpoint_active_requests.get(endpoint, 0)
            queued = len(self.endpoint_queues.get(endpoint, []))
            request_distribution[endpoint] = {
                "active": active,
                "queued": queued,
                "total_load": active + queued
            }

        return {
            "status": "healthy",
            "backend_type": self.backend_type,
            "backend_port": self.backend_port,
            "backend_endpoints": self.backend_endpoints,
            "models_loaded": list(self.models_loaded),
            "model_status": model_status,
            "active_requests": self.active_requests,
            "queued_requests": self.queued_requests,
            "total_requests_processed": self.total_requests,
            "max_concurrent_requests": self.max_concurrent,
            "max_queue_size": self.max_queue_size,
            "model_keep_warm_seconds": self.model_keep_warm_seconds,
            "uptime_seconds": time.time() - self.start_time,
            "endpoint_status": endpoint_info,
            "request_distribution": request_distribution,
            "queue_by_priority": self._get_queue_by_priority(),
            "queue_by_task_type": self._get_queue_by_task_type(),
            "requests_by_task_type": dict(self.requests_by_task_type),
            "task_routing_config": {k.value: v for k, v in self.task_routing.items()}
        }
    
    def _get_queue_by_priority(self) -> Dict[int, int]:
        """Get queue count by priority level."""
        priority_counts = {}
        for priority, _, request, _, _ in self.request_queue:
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
        return priority_counts
    
    def _get_queue_by_task_type(self) -> Dict[str, int]:
        """Get queue count by task type."""
        task_counts = {}
        for _, _, request, _, _ in self.request_queue:
            task_type = request.task_type.value
            task_counts[task_type] = task_counts.get(task_type, 0) + 1
        return task_counts


# Global LLM manager instance
llm_manager: Optional[LLMManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan manager."""
    global llm_manager

    # Startup
    logger.info("=" * 60)
    logger.info("Starting LLM Server (Multi-backend load balancer)")
    logger.info("=" * 60)

    # Get backend config for logging
    try:
        backend_config = get_llm_backend_config()
        logger.info(f"Backend type: {backend_config['backend'].upper()}")
        logger.info(f"Backend port: {backend_config['port']}")
        logger.info(f"Backend endpoints: {backend_config['endpoints']}")
    except Exception as e:
        logger.warning(f"Could not load backend config: {e}")

    # Use environment variable if set, otherwise use config (default to 1)
    max_concurrent = int(os.environ.get('LLM_MAX_CONCURRENT', 1))
    max_queue_size = int(os.environ.get('LLM_MAX_QUEUE_SIZE', 100))
    logger.info(f"Max concurrent requests: {max_concurrent}")
    logger.info(f"Max queue size: {max_queue_size}")

    # Get LLM config for logging
    try:
        llm_config = get_llm_config()
        logger.info(f"LLM Server URL configured as: {llm_config.get('server_url')}")
        logger.info(f"Primary model configured in config: {llm_config.get('models', {}).get('primary', {}).get('name', 'qwen3:4b-instruct')}")
    except Exception as e:
        logger.warning(f"Could not load LLM config: {e}")

    logger.info("Note: For non-LLM models (Whisper, PyAnnote, etc.), load directly in each process")

    llm_manager = LLMManager(max_concurrent=max_concurrent, max_queue_size=max_queue_size)
    await llm_manager.initialize()
    logger.info("LLM Server ready")
    
    yield
    
    # Shutdown
    logger.info("Shutting down LLM Server...")
    if llm_manager:
        # Cancel the unloader task
        if llm_manager.unload_task:
            llm_manager.unload_task.cancel()
            try:
                await llm_manager.unload_task
            except asyncio.CancelledError:
                pass
        # Cancel queue processors
        for processor in llm_manager.queue_processors:
            processor.cancel()
        # Wait for all processors to finish
        await asyncio.gather(*llm_manager.queue_processors, return_exceptions=True)
        # Close HTTP client
        if llm_manager.http_client:
            await llm_manager.http_client.aclose()
        llm_manager.executor.shutdown(wait=True)
    logger.info("LLM Server shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="LLM Server (Multi-Backend Load Balancer)",
    description="Load-balanced LLM access supporting MLX and Ollama backends",
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
    if not llm_manager:
        raise HTTPException(status_code=503, detail="Server not initialized")
    return llm_manager.get_status()


@app.post("/reload-config")
async def reload_config():
    """Manually trigger config reload."""
    if not llm_manager:
        raise HTTPException(status_code=503, detail="Server not initialized")

    reloaded = await llm_manager._check_config_reload()

    return {
        "reloaded": reloaded,
        "task_routing": {k.value: v for k, v in llm_manager.task_routing.items()},
        "message": "Config reloaded" if reloaded else "Config unchanged"
    }


@app.post("/llm-request", response_model=LLMResponse)
async def llm_request(request: LLMRequest, request_obj: Request):
    """Process LLM request with messages."""
    if not llm_manager:
        raise HTTPException(status_code=503, detail="Server not initialized")
        
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    request_id = f"llm_{timestamp.replace(':', '').replace('.', '')}"
    start_time = time.time()
    
    # Extract worker ID from request headers or IP
    worker_id = request_obj.headers.get('X-Worker-ID', None)
    client_ip = request_obj.client.host if request_obj.client else 'unknown'
    
    # If no worker ID in headers, try to look it up from IP
    if not worker_id and client_ip != 'unknown':
        try:
            config = load_config()
            workers = config.get('processing', {}).get('workers', {})
            
            # Look for worker by IP (check ip, eth, and wifi fields)
            for worker_name, worker_info in workers.items():
                if (worker_info.get('ip') == client_ip or 
                    worker_info.get('eth') == client_ip or 
                    worker_info.get('wifi') == client_ip):
                    worker_id = worker_name
                    break
            
            # If still not found, check if it's the head node
            if not worker_id:
                head_ip = config.get('network', {}).get('head_node_ip')
                if client_ip == head_ip:
                    worker_id = 'head_node'
        except Exception as e:
            logger.warning(f"Failed to lookup worker from IP {client_ip}: {e}")
    
    # Default to IP if worker not found
    if not worker_id:
        worker_id = f"ip_{client_ip}"
    
    # Log request receipt with worker and queue info
    current_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    async with llm_manager.queue_lock:
        queue_length = len(llm_manager.request_queue)
    request_msg = f"[{current_time}] RECEIVE - Worker: {worker_id}, Priority: {request.priority}, Queue: {queue_length}, Active: {llm_manager.active_requests}"
    logger.info(request_msg)
    print(request_msg)
    
    try:
        response_data = await llm_manager.process_llm_request(request)

        # Log response completion with timing
        total_time = time.time() - start_time
        reply_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        reply_msg = f"[{reply_time}] REPLIED - Worker: {worker_id}, Time: {total_time:.3f}s, Endpoint: {response_data['endpoint_used']}, Queue: {llm_manager.queued_requests}, Active: {llm_manager.active_requests}"
        logger.info(reply_msg)
        print(reply_msg)

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
        logger.error(f"Error processing LLM request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-classify")
async def batch_classify(request: BatchClassifyRequest):
    """Process batch of classification requests with queue saturation"""
    if not llm_manager:
        raise HTTPException(status_code=503, detail="Server not initialized")
    
    start_time = time.time()
    batch_size = len(request.requests)
    
    logger.info(f"Received batch {request.batch_idx} with {batch_size} {request.phase} requests")
    
    try:
        # Process batch through queue-saturated system
        results = await llm_manager.process_classification_batch(
            request.requests,
            request.phase,
            request.batch_idx
        )
        
        # Sort results by index to maintain order
        results.sort(key=lambda x: x['index'])
        
        total_time = time.time() - start_time
        logger.info(f"Completed batch {request.batch_idx} in {total_time:.2f}s - {batch_size} {request.phase} classifications")
        
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
    import argparse

    parser = argparse.ArgumentParser(description='LLM Server')
    parser.add_argument('--port', type=int, default=8002, help='Port to listen on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    args = parser.parse_args()

    # Configure uvicorn logging to suppress default logs and use our logger
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["loggers"]["uvicorn"]["level"] = "WARNING"
    log_config["loggers"]["uvicorn.error"]["level"] = "WARNING"
    log_config["loggers"]["uvicorn.access"]["level"] = "WARNING"

    # Log server startup through our logger
    logger.info(f"Starting LLM Server on {args.host}:{args.port}")

    # Run the server - pass app object directly since module path doesn't work
    # when running from different working directories
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_config=log_config
    )