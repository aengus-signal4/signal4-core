#!/usr/bin/env python3
"""
Model Client Utility
===================

Provides a simple interface for processing steps to access models through the model server
with automatic fallback to other workers and local loading.
"""

import asyncio
import logging
import aiohttp
import socket
import os
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path
from src.utils.paths import get_project_root, get_config_path
import yaml

from src.utils.logger import setup_worker_logger

logger = setup_worker_logger('model_client')

# Cache for model server discovery
_discovery_cache: Optional[Dict[str, Any]] = None
_discovery_cache_time: Optional[datetime] = None
_discovery_cache_ttl = timedelta(seconds=60)  # Cache for 1 minute

# Get worker ID from environment or hostname
def get_current_worker_id() -> str:
    """Get the current worker ID from environment or hostname."""
    # First check environment
    worker_id = os.environ.get('WORKER_ID')
    if worker_id:
        return worker_id
    
    # Then check hostname
    hostname = socket.gethostname()
    if 'head' in hostname:
        return 'worker0'
    
    # Extract from hostname pattern
    import re
    match = re.search(r'worker(\d+)', hostname)
    if match:
        return f"worker{match.group(1)}"
    
    # Default to worker0
    return 'worker0'


async def get_model_server_discovery(force_refresh: bool = False) -> Dict[str, Any]:
    """Get model server discovery information from orchestrator."""
    global _discovery_cache, _discovery_cache_time
    
    # Check cache
    if not force_refresh and _discovery_cache and _discovery_cache_time:
        if datetime.now() - _discovery_cache_time < _discovery_cache_ttl:
            return _discovery_cache
    
    # Load config to get orchestrator URL
    config_path = get_config_path()
    orchestrator_url = "http://localhost:8001"  # Default
    
    try:
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                head_ip = config.get('network', {}).get('head_node_ip', '10.0.0.22')
                orchestrator_url = f"http://{head_ip}:8001"
    except Exception as e:
        logger.warning(f"Error loading config for orchestrator URL: {e}")
    
    # Query orchestrator for discovery
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{orchestrator_url}/model-servers/discovery", 
                                 timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    discovery = await response.json()
                    _discovery_cache = discovery
                    _discovery_cache_time = datetime.now()
                    return discovery
    except Exception as e:
        logger.warning(f"Failed to get model server discovery: {e}")
    
    # Return empty discovery on failure
    return {"status": "error", "discovery": {}}


async def find_available_model_server(model_type: str, model_name: str) -> Optional[str]:
    """Find an available model server for the given model type and name."""
    current_worker = get_current_worker_id()
    
    # Get discovery information
    discovery = await get_model_server_discovery()
    
    if discovery.get("status") != "success":
        return None
    
    servers = discovery.get("discovery", {}).get(model_type, [])
    
    # First, try to find local server
    for server in servers:
        if server.get("worker_id") == current_worker:
            # Check if this server has the specific model loaded or can load it
            models_loaded = server.get("models_loaded", {}).get(model_type, [])
            if not models_loaded or model_name in models_loaded:
                return server.get("url")
    
    # Then try other servers with lowest load
    available_servers = []
    for server in servers:
        if server.get("worker_id") != current_worker:
            active = server.get("active_requests", 0)
            max_concurrent = server.get("max_concurrent", 4)
            if active < max_concurrent:
                available_servers.append((server.get("url"), active))
    
    # Sort by load and return least loaded
    if available_servers:
        available_servers.sort(key=lambda x: x[1])
        return available_servers[0][0]
    
    return None


async def request_model(
    model_type: str,
    model_name: str,
    task: str,
    parameters: Dict[str, Any],
    fallback_to_local: bool = True,
    timeout: int = 300
) -> Tuple[bool, Any]:
    """
    Request a model operation through model server with fallback.
    
    Args:
        model_type: Type of model (llm, whisper, pyannote, etc.)
        model_name: Specific model name
        task: Task to perform
        parameters: Task-specific parameters
        fallback_to_local: Whether to fall back to local loading if servers unavailable
        timeout: Request timeout in seconds
    
    Returns:
        Tuple of (success: bool, result: Any)
    """
    # First try local model server
    local_url = "http://localhost:8002"
    
    try:
        logger.debug(f"Attempting to connect to local model server at {local_url} for {model_type}/{model_name}")
        async with aiohttp.ClientSession() as session:
            request_data = {
                "model_type": model_type,
                "model_name": model_name,
                "task": task,
                "parameters": parameters
            }
            
            async with session.post(
                f"{local_url}/model-request",
                json=request_data,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get("success"):
                        logger.debug(f"Successfully used local model server for {model_type}/{model_name}")
                        return True, result.get("result")
                    else:
                        logger.warning(f"Local model server returned unsuccessful response: {result}")
    except aiohttp.ClientConnectorError as e:
        logger.debug(f"Local model server connection failed at {local_url}: {e}")
    except Exception as e:
        logger.warning(f"Error with local model server: {e}")
    
    # Try to find another available server
    logger.debug(f"Looking for available model servers via discovery for {model_type}/{model_name}")
    remote_url = await find_available_model_server(model_type, model_name)
    
    if remote_url and remote_url != local_url:
        try:
            logger.info(f"Found remote model server at {remote_url}, attempting connection")
            async with aiohttp.ClientSession() as session:
                request_data = {
                    "model_type": model_type,
                    "model_name": model_name,
                    "task": task,
                    "parameters": parameters
                }
                
                async with session.post(
                    f"{remote_url}/model-request",
                    json=request_data,
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result.get("success"):
                            logger.info(f"Successfully used remote model server at {remote_url} for {model_type}/{model_name}")
                            return True, result.get("result")
                        else:
                            logger.warning(f"Remote model server returned unsuccessful response: {result}")
        except Exception as e:
            logger.warning(f"Error with remote model server {remote_url}: {e}")
    else:
        logger.debug(f"No remote model server found via discovery for {model_type}/{model_name}")
    
    # Fallback to local loading if enabled
    if fallback_to_local:
        logger.info(f"Falling back to local loading for {model_type}/{model_name}")
        return False, None  # Caller should handle local loading
    
    # No options available
    logger.error(f"No model server available for {model_type}/{model_name} and local fallback disabled")
    return False, None


# Convenience functions for specific model types
async def request_whisper_transcription(
    audio_file: str,
    model: str = "large-v3-turbo",
    language: Optional[str] = None,
    fallback_to_local: bool = True
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Request Whisper transcription through model server."""
    return await request_model(
        model_type="whisper",
        model_name=model,
        task="transcribe",
        parameters={
            "audio_file": audio_file,
            "language": language
        },
        fallback_to_local=fallback_to_local
    )


async def request_llm_completion(
    messages: List[Dict[str, str]],
    model: str = "qwen3:4b-instruct",
    temperature: float = 0.1,
    max_tokens: Optional[int] = None,
    fallback_to_local: bool = True
) -> Tuple[bool, Optional[str]]:
    """Request LLM completion through model server."""
    return await request_model(
        model_type="llm",
        model_name=model,
        task="chat",
        parameters={
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        },
        fallback_to_local=fallback_to_local
    )


async def request_embeddings(
    texts: List[str],
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
    normalize: bool = True,
    fallback_to_local: bool = True
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Request text embeddings through model server."""
    return await request_model(
        model_type="embedding",
        model_name=model,
        task="encode",
        parameters={
            "texts": texts,
            "normalize": normalize
        },
        fallback_to_local=fallback_to_local
    )


async def request_pyannote_diarization(
    audio_file: str,
    task: str = "diarization",
    fallback_to_local: bool = True
) -> Tuple[bool, Optional[Any]]:
    """Request PyAnnote diarization through model server."""
    return await request_model(
        model_type="pyannote",
        model_name=task,  # "diarization" or "embedding"
        task=task,
        parameters={
            "audio_file": audio_file
        },
        fallback_to_local=fallback_to_local
    )