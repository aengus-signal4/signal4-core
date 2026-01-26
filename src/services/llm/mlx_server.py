#!/usr/bin/env python3
"""
MLX Model Server for LLM Processing
====================================

FastAPI server that provides local LLM inference using MLX for Apple Silicon acceleration.
This server processes requests locally - the LLM balancer handles routing across nodes.

PREREQUISITES:
- MLX installed: `pip install mlx-lm mlx-vlm`
- Models downloaded locally or will be pulled on first use
- Apple Silicon Mac (M1/M2/M3/M4 series)

Key features:
1. Native MLX acceleration for Apple Silicon
2. Three-tier model system (tier_1/tier_2/tier_3) - automatically upgrades to better model if loaded
3. Priority-based queuing (1-99, lower numbers = higher priority)
4. Per-worker model restrictions based on config
5. Keeps models warm in memory (configurable timeout)
6. Audio input support via Qwen3-Omni (tier_2)

Three-tier model system:
- Tier 1 (best):     mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit  (text only)
- Tier 2 (balanced): local:mlx_qwen3_omni_4bit                        (text + audio, short clips <30s)
- Tier 3 (fast):     mlx-community/Qwen3-4B-Instruct-2507-4bit       (text only)

Usage examples:
  # Text-only request
  request = {"model": "tier_2", "messages": [{"role": "user", "content": "..."}]}

  # Audio + text request (tier_2 only)
  request = {
      "model": "tier_2",
      "messages": [{"role": "user", "content": "Analyze this audio..."}],
      "audio_path": "/path/to/audio.wav"  # Short clips <30s recommended
  }

Response includes "model_used" and "audio_processed" fields.
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
from enum import Enum

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from src.utils.logger import setup_worker_logger
from src.utils.config import load_config

logger = setup_worker_logger('mlx_model_server')
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

# MLX-VLM imports for Omni models (audio support)
try:
    from mlx_vlm import load as vlm_load
    from mlx_vlm.models.qwen3_omni_moe.omni_utils import prepare_omni_inputs
    MLX_VLM_AVAILABLE = True
except ImportError:
    logger.warning("mlx-vlm not available for Omni models - audio support disabled")
    MLX_VLM_AVAILABLE = False


# Model configurations with hierarchy (rank 1 = best quality)
# Models marked as "omni" support audio input via mlx-vlm
MODEL_CONFIGS = {
    "mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit": {
        "rank": 1,  # Tier 1 - Best quality
        "aliases": ["qwen3:80b", "80b", "large", "tier_1", "tier1", "best"],
        "description": "80B parameter model - best quality",
        "omni": False,
    },
    "local:mlx_qwen3_omni_4bit": {
        "rank": 2,  # Tier 2 - Balanced (30B Omni with audio support)
        "aliases": ["qwen3:omni", "omni", "30b", "medium", "tier_2", "tier2", "balanced"],
        "description": "30B parameter Omni model - text + audio input (short clips <30s)",
        "omni": True,
        "local_path": "scripts/qwen3_omni_captioner/mlx_qwen3_omni_4bit",
    },
    "mlx-community/Qwen3-4B-Instruct-2507-4bit": {
        "rank": 3,  # Tier 3 - Fast
        "aliases": ["qwen3:4b-instruct", "qwen3:4b", "4b", "tier_3", "tier3"],
        "description": "4B parameter model - fast",
        "omni": False,
    },
    "mlx-community/LFM2-8B-A1B-4bit": {
        "rank": 4,  # Tier 4 - Fastest (legacy)
        "aliases": ["lfm2:8b", "8b", "small", "fast", "fastest"],
        "description": "8B parameter model - fastest",
        "omni": False,
    }
}


# Request/Response models
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
    audio_path: Optional[str] = Field(default=None, description="Path to audio file (for Omni models)")


class LLMRequest(BaseModel):
    """Request for LLM processing."""
    messages: List[LLMMessage] = Field(..., description="List of messages")
    model: str = Field(
        default="tier_2",
        description="Model to use - supports tier_1 (80B), tier_2 (Omni 30B), tier_3 (4B) or full model names"
    )
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, description="Max tokens for response")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p sampling")
    seed: Optional[int] = Field(default=42, description="Random seed for reproducibility")
    priority: int = Field(default=1, ge=1, le=99, description="Request priority (1-99, lower = higher priority)")
    task_type: TaskType = Field(default=TaskType.TEXT, description="Task type")
    audio_path: Optional[str] = Field(default=None, description="Path to audio file for Omni model (short clips <30s recommended)")


class LLMResponse(BaseModel):
    """Generic LLM response."""
    response: str
    model_used: str
    processing_time: float
    request_id: str
    endpoint_used: str
    priority: int
    task_type: str


class ServerStatus(BaseModel):
    """Server status information."""
    status: str
    models_loaded: List[str]
    model_status: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    allowed_models: List[str]
    active_requests: int
    queued_requests: int
    total_requests_processed: int
    max_concurrent_requests: int
    max_queue_size: int
    model_keep_warm_seconds: int
    uptime_seconds: float
    requests_by_task_type: Dict[str, int] = Field(default_factory=dict)


class MLXManager:
    """Manages local MLX model loading and request processing."""

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

        # Priority queue: (priority, counter, request, future, request_id)
        self.request_queue = []
        self.queue_lock = asyncio.Lock()
        self.request_counter = 0

        # Background tasks
        self.unload_task = None
        self.queue_processors = []

        # Statistics
        self.requests_by_task_type = {task_type.value: 0 for task_type in TaskType}

    async def initialize(self):
        """Initialize MLX Manager."""
        logger.info("=== MLX Model Server Initialization ===")
        logger.info("Using MLX for Apple Silicon acceleration")

        if not MLX_AVAILABLE:
            raise RuntimeError("MLX not available - install with: pip install mlx-lm")

        # Load config to get model restrictions
        try:
            config = load_config()
            workers = config.get('processing', {}).get('workers', {})

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

        except Exception as e:
            logger.warning(f"Could not load config: {e}, allowing all models")
            self.allowed_models = list(MODEL_CONFIGS.keys())
            self.default_model = "mlx-community/Qwen3-4B-Instruct-2507-4bit"

        logger.info("Models will be loaded on first use and kept warm for {} seconds".format(self.model_keep_warm_seconds))
        logger.info(f"MLX supports up to {self.max_concurrent} concurrent requests with queue size {self.max_queue_size}")

        # Start background tasks
        self.unload_task = asyncio.create_task(self._model_unloader())
        logger.info("Model unloader task started")

        # Start queue processors
        for i in range(self.max_concurrent):
            processor = asyncio.create_task(self._queue_processor(i))
            self.queue_processors.append(processor)
        logger.info(f"Started {self.max_concurrent} queue processor tasks")

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
            # Find best allowed model as fallback (silently - completion log shows upgrade)
            allowed_by_rank = sorted(
                [m for m in self.allowed_models],
                key=lambda m: MODEL_CONFIGS[m]['rank']
            )
            if allowed_by_rank:
                resolved_model = allowed_by_rank[0]  # Best ranked allowed model
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

        return best_model

    async def _ensure_model_loaded(self, model_name: str) -> Tuple[Any, Any]:
        """Ensure a model is loaded and return (model, tokenizer/processor)."""
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

            # Check if this is an Omni model
            model_config = MODEL_CONFIGS.get(model_name, {})
            is_omni = model_config.get("omni", False)

            # Load model
            logger.info(f"Loading model {model_name} (omni={is_omni})...")
            try:
                if is_omni:
                    if not MLX_VLM_AVAILABLE:
                        raise RuntimeError("mlx-vlm not available - cannot load Omni model")
                    # Resolve local path for Omni model
                    local_path = model_config.get("local_path")
                    if local_path:
                        # Resolve relative to core directory
                        import pathlib
                        core_dir = pathlib.Path(__file__).resolve().parent.parent.parent.parent
                        model_path = str(core_dir / local_path)
                    else:
                        model_path = model_name
                    logger.info(f"Loading Omni model from: {model_path}")
                    model, processor = vlm_load(model_path, trust_remote_code=True)
                    self.loaded_models[model_name] = (model, processor)
                    logger.info(f"Successfully loaded Omni model {model_name}")
                else:
                    model, tokenizer = load(model_name)
                    self.loaded_models[model_name] = (model, tokenizer)
                    logger.info(f"Successfully loaded model {model_name}")
                return self.loaded_models[model_name]
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                raise

    async def _queue_processor(self, processor_id: int):
        """Process requests from the priority queue."""
        logger.info(f"Queue processor {processor_id} started")
        while True:
            try:
                request_item = await self._get_next_request()
                if not request_item:
                    await asyncio.sleep(0.1)
                    continue

                priority, counter, request, future, request_id = request_item

                async with self.request_lock:
                    self.active_requests += 1
                    self.queued_requests -= 1

                logger.debug(f"[{request_id}] Processing request (processor {processor_id}, priority {priority})")
                start_time = time.time()

                try:
                    response_data = await self._process_llm_request(request, request_id)
                    processing_time = time.time() - start_time

                    # Clean completion log
                    model_used = response_data.get('model_used', 'unknown')
                    model_requested = request.model

                    # Extract short model names
                    def get_tier(model_name):
                        if '80B' in model_name:
                            return 'tier_1'
                        elif '30B' in model_name or '4B' in model_name:
                            return 'tier_2'
                        elif '8B' in model_name:
                            return 'tier_3'
                        elif model_name.startswith('tier_'):
                            return model_name
                        return model_name.split('/')[-1][:20]

                    tier_used = get_tier(model_used)
                    tier_requested = get_tier(model_requested)

                    if tier_used != tier_requested:
                        logger.info(f"[{request_id}] Completed in {processing_time:5.2f}s ({tier_used} for {tier_requested})")
                    else:
                        logger.info(f"[{request_id}] Completed in {processing_time:5.2f}s ({tier_used})")

                    self.total_requests += 1
                    self.requests_by_task_type[request.task_type.value] += 1
                    future.set_result(response_data)

                except Exception as e:
                    processing_time = time.time() - start_time
                    logger.error(f"[{request_id}] Request failed after {processing_time:.2f}s: {e}")
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

    async def _process_llm_request(self, request: LLMRequest, request_id: str) -> Dict[str, Any]:
        """Process LLM request locally using MLX."""
        # Choose best model based on hierarchy
        best_model = self._choose_best_model(request.model)

        # Check if this is an Omni model
        model_config = MODEL_CONFIGS.get(best_model, {})
        is_omni = model_config.get("omni", False)

        # Ensure model is loaded
        model, tokenizer_or_processor = await self._ensure_model_loaded(best_model)

        # Update last used time
        self.models_last_used[best_model] = time.time()

        # Route to appropriate generation method
        loop = asyncio.get_event_loop()

        if is_omni and request.audio_path:
            # Use Omni model with audio input
            response_data = await loop.run_in_executor(
                self.executor,
                self._generate_omni_response,
                model,
                tokenizer_or_processor,
                request
            )
        elif is_omni:
            # Omni model but text-only request
            response_data = await loop.run_in_executor(
                self.executor,
                self._generate_omni_text_response,
                model,
                tokenizer_or_processor,
                request
            )
        else:
            # Standard text model
            # Build prompt from messages
            if tokenizer_or_processor.chat_template is not None:
                messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

                # Disable Qwen3 thinking mode for non-Instruct models (e.g., Qwen3-30B-A3B-4bit)
                # Instruct models don't have thinking mode, so skip them
                if "Instruct" not in best_model and "Qwen3" in best_model:
                    for i in range(len(messages) - 1, -1, -1):
                        if messages[i]["role"] == "user":
                            if "/no_think" not in messages[i]["content"]:
                                messages[i]["content"] = messages[i]["content"].rstrip() + " /no_think"
                            break

                prompt = tokenizer_or_processor.apply_chat_template(messages, add_generation_prompt=True)
            else:
                # Fallback to simple concatenation
                prompt = "\n\n".join([f"{msg.role}: {msg.content}" for msg in request.messages])

            response_data = await loop.run_in_executor(
                self.executor,
                self._generate_mlx_response,
                model,
                tokenizer_or_processor,
                prompt,
                request
            )

        return {
            "response": response_data['text'],
            "model_used": best_model,
            "endpoint_used": "localhost",
            "priority": request.priority,
            "task_type": request.task_type.value,
            "mlx_stats": response_data.get('stats', {}),
            "audio_processed": is_omni and request.audio_path is not None,
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

    def _generate_omni_response(self, model, processor, request: LLMRequest) -> Dict[str, Any]:
        """Generate response using Omni model with audio input (runs in executor).

        Audio Limitations:
        - Short clips recommended (<30 seconds)
        - Longer audio uses more memory (~80GB for 15s, ~145GB for 120s)
        - Audio is processed natively - no separate transcription needed
        """
        try:
            # Build conversation with audio
            # Get text content from the last user message
            text_content = ""
            for msg in reversed(request.messages):
                if msg.role == "user":
                    text_content = msg.content
                    break

            conversation = [{
                "role": "user",
                "content": [
                    {"type": "audio", "audio": request.audio_path},
                    {"type": "text", "text": text_content},
                ],
            }]

            # Prepare inputs using Omni utilities
            model_inputs, _ = prepare_omni_inputs(processor, conversation)

            # Generate (text only, no audio output)
            generate_kwargs = {
                "input_ids": model_inputs["input_ids"],
                "input_features": model_inputs.get("input_features"),
                "feature_attention_mask": model_inputs.get("feature_attention_mask"),
                "audio_feature_lengths": model_inputs.get("audio_feature_lengths"),
                "thinker_max_new_tokens": request.max_tokens or 2048,
                "return_audio": False,
            }
            # Filter None values
            generate_kwargs = {k: v for k, v in generate_kwargs.items() if v is not None}

            thinker_result, _ = model.generate(**generate_kwargs)
            full_text = processor.decode(thinker_result.sequences[0].tolist())

            return {
                "text": full_text.strip(),
                "stats": {"audio_input": True}
            }
        except Exception as e:
            logger.error(f"Omni generation error (audio): {e}")
            raise

    def _generate_omni_text_response(self, model, processor, request: LLMRequest) -> Dict[str, Any]:
        """Generate response using Omni model with text-only input (runs in executor)."""
        try:
            # Build text-only conversation
            text_content = ""
            for msg in reversed(request.messages):
                if msg.role == "user":
                    text_content = msg.content
                    break

            conversation = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": text_content},
                ],
            }]

            # Prepare inputs using Omni utilities
            model_inputs, _ = prepare_omni_inputs(processor, conversation)

            # Generate (text only)
            generate_kwargs = {
                "input_ids": model_inputs["input_ids"],
                "thinker_max_new_tokens": request.max_tokens or 2048,
                "return_audio": False,
            }

            thinker_result, _ = model.generate(**generate_kwargs)
            full_text = processor.decode(thinker_result.sequences[0].tolist())

            return {
                "text": full_text.strip(),
                "stats": {"audio_input": False}
            }
        except Exception as e:
            logger.error(f"Omni generation error (text): {e}")
            raise

    async def process_llm_request(self, request: LLMRequest) -> Dict[str, Any]:
        """Process a generic LLM request via priority queue."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        request_id = f"llm_{timestamp.replace(':', '').replace('.', '')}"

        logger.debug(f"[{request_id}] LLM Request - Model: {request.model}, Priority: {request.priority}")

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

        # Add to priority queue
        async with self.queue_lock:
            priority_item = (request.priority, self.request_counter, request, future, request_id)
            heapq.heappush(self.request_queue, priority_item)
            self.request_counter += 1

        async with self.request_lock:
            self.queued_requests += 1

        logger.debug(f"[{request_id}] Request queued with priority {request.priority} (active: {self.active_requests}, queued: {self.queued_requests})")

        return await future

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

        return {
            "status": "healthy",
            "models_loaded": list(self.loaded_models.keys()),
            "model_status": model_status,
            "allowed_models": self.allowed_models,
            "active_requests": self.active_requests,
            "queued_requests": self.queued_requests,
            "total_requests_processed": self.total_requests,
            "max_concurrent_requests": self.max_concurrent,
            "max_queue_size": self.max_queue_size,
            "model_keep_warm_seconds": self.model_keep_warm_seconds,
            "uptime_seconds": time.time() - self.start_time,
            "requests_by_task_type": dict(self.requests_by_task_type),
        }


# Global MLX manager instance
mlx_manager: Optional[MLXManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan manager."""
    global mlx_manager

    logger.info("=" * 60)
    logger.info("Starting MLX Model Server (Apple Silicon acceleration)")
    logger.info("=" * 60)

    max_concurrent = int(os.environ.get('LLM_MAX_CONCURRENT', 5))
    max_queue_size = int(os.environ.get('LLM_MAX_QUEUE_SIZE', 100))
    logger.info(f"Max concurrent requests: {max_concurrent}")
    logger.info(f"Max queue size: {max_queue_size}")

    mlx_manager = MLXManager(max_concurrent=max_concurrent, max_queue_size=max_queue_size)
    await mlx_manager.initialize()
    logger.info("MLX Model Server ready")

    yield

    logger.info("Shutting down MLX Model Server...")
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
        mlx_manager.executor.shutdown(wait=True)
    logger.info("MLX Model Server shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="MLX Model Server",
    description="Local LLM inference using MLX for Apple Silicon acceleration",
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
    if not mlx_manager:
        raise HTTPException(status_code=503, detail="Server not initialized")
    return mlx_manager.get_status()


@app.post("/llm-request", response_model=LLMResponse)
async def llm_request(request: LLMRequest):
    """Process LLM request with messages."""
    if not mlx_manager:
        raise HTTPException(status_code=503, detail="Server not initialized")

    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    request_id = f"llm_{timestamp.replace(':', '').replace('.', '')}"
    start_time = time.time()

    try:
        response_data = await mlx_manager.process_llm_request(request)
        total_time = time.time() - start_time

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='MLX Model Server')
    parser.add_argument('--port', type=int, default=8004, help='Port to listen on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    args = parser.parse_args()

    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["loggers"]["uvicorn"]["level"] = "WARNING"
    log_config["loggers"]["uvicorn.error"]["level"] = "WARNING"
    log_config["loggers"]["uvicorn.access"]["level"] = "WARNING"

    logger.info(f"Starting MLX Model Server on {args.host}:{args.port}")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_config=log_config
    )
