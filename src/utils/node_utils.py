"""
Utilities for determining node type (head or worker).
"""

import socket
from typing import Dict, Any, Optional
import logging
from pathlib import Path

# Import the actual config loading function
from .config import load_config 

_node_type = None
_is_head_node = None

# Set up a logger for this module if needed
logger = logging.getLogger(__name__)

def get_local_ip() -> str:
    """Get local IP address of this machine."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't need to be reachable
        # Use a dummy IP, connect doesn't actually send packets for UDP
        s.connect(('10.255.255.255', 1))
        local_ip = s.getsockname()[0]
    except socket.error: # Be more specific with the exception
        local_ip = '127.0.0.1'
    finally:
        s.close()
    return local_ip

def _determine_node_type(config: Dict[str, Any]) -> None:
    """Internal function to determine and cache node type."""
    global _node_type, _is_head_node
    
    if _node_type is not None:
        return

    local_ip = get_local_ip()
    head_node_ip = config.get('network', {}).get('head_node_ip')
    
    if not head_node_ip:
        logging.warning("Head node IP not found in config. Defaulting node type check.")
        # Handle case where head_node_ip might be missing
        # Maybe default to 'unknown' or try other methods?
        # For now, let's assume it cannot be head if IP is missing.
        _node_type = 'unknown'
        _is_head_node = False
        # Check workers even if head IP is missing
        workers = config.get('processing', {}).get('workers', {})
        for worker_id, details in workers.items():
            if local_ip == details.get('eth') or local_ip == details.get('wifi'):
                _node_type = details.get('type', 'worker')
                break # Found a match, stop searching
        return # Exit after checking workers

    if local_ip == head_node_ip:
        _node_type = 'head'
        _is_head_node = True
    else:
        # Check if this IP matches any worker's IP (eth or wifi)
        workers = config.get('processing', {}).get('workers', {})
        is_worker = False
        for worker_id, details in workers.items():
            if local_ip == details.get('eth') or local_ip == details.get('wifi'):
                # Use 'worker' as type if explicitly set, otherwise default to 'worker'
                _node_type = details.get('type', 'worker') 
                is_worker = True
                break
        if not is_worker:
            # Neither head nor a listed worker
            _node_type = 'unknown' 
        # Set _is_head_node based on the final _node_type determination
        _is_head_node = (_node_type == 'head')


def is_head_node(config: Optional[Dict[str, Any]] = None) -> bool:
    """
    Check if the current node is the head node.
    Loads config if not provided.
    """
    # Ensure config is loaded if not provided
    current_config = config if config is not None else load_config()
    
    # Determine type if not already cached
    if _is_head_node is None:
        _determine_node_type(current_config)
        
    # Add a check in case _determine_node_type failed to set it
    if _is_head_node is None:
        logging.error("_is_head_node is still None after determination attempt.")
        return False # Default to False if determination failed
        
    return _is_head_node

def get_node_type(config: Optional[Dict[str, Any]] = None) -> str:
    """
    Get the type of the current node ('head', 'worker', or 'unknown').
    Loads config if not provided.
    """
    current_config = config if config is not None else load_config()
    
    if _node_type is None:
        _determine_node_type(current_config)
        
    # Add a check in case _determine_node_type failed to set it
    if _node_type is None:
        logging.error("_node_type is still None after determination attempt.")
        return 'unknown' # Default to 'unknown' if determination failed
        
    return _node_type

def get_appropriate_python_path(config: Optional[Dict[str, Any]] = None) -> str:
    """
    Get the appropriate python path (head or worker) based on the node type.
    Loads config if not provided.
    """
    current_config = config if config is not None else load_config()
            
    processing_config = current_config.get('processing', {})
    node_is_head = is_head_node(current_config) # Use the determined status
    
    if node_is_head:
        path = processing_config.get('head_python_path')
        if not path:
             raise ValueError("Configuration missing 'processing.head_python_path'")
        return path
    else:
        # Assume worker if not head (includes 'unknown' type for this logic)
        path = processing_config.get('worker_python_path')
        if not path:
             raise ValueError("Configuration missing 'processing.worker_python_path'")
        return path

def get_worker_name(config: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Get the name of the current worker based on IP address.
    Returns None if not a recognized worker.
    """
    current_config = config if config is not None else load_config()
    local_ip = get_local_ip()
    
    # Check if this is the head node
    head_node_ip = current_config.get('network', {}).get('head_node_ip')
    if local_ip == head_node_ip:
        return 'worker0'  # Head node is typically worker0
    
    # Check workers
    workers = current_config.get('processing', {}).get('workers', {})
    for worker_name, worker_info in workers.items():
        if local_ip == worker_info.get('ip') or local_ip == worker_info.get('eth') or local_ip == worker_info.get('wifi'):
            return worker_name
    
    return None

def get_conda_env_name(config: Optional[Dict[str, Any]] = None) -> str:
    """
    Extracts the conda environment name from the appropriate python path.
    Loads config if not provided.
    """
    # Load config within this function if not passed, to ensure it's available
    current_config = config if config is not None else load_config()
    python_path_str = get_appropriate_python_path(current_config)
    python_path = Path(python_path_str)
    
    try:
        # Expected structure: .../envs/{env_name}/bin/python
        parts = python_path.parts
        # Find the index of 'envs' directory in the path parts
        envs_index = parts.index('envs')
        # The environment name should be the next part
        if envs_index + 1 < len(parts):
             env_name = parts[envs_index + 1]
             return env_name
        else:
             raise ValueError(f"Path structure invalid after 'envs': {python_path_str}")
    except (ValueError, IndexError) as e:
         # Add original exception for context if needed
         logging.error(f"Error parsing conda env from path '{python_path_str}': {e}")
         # Use an f-string for the error message
         raise ValueError(f"Could not extract conda env name from path: {python_path_str}") from e 