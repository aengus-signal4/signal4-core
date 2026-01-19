"""
Configuration utilities for loading and managing config files.

This module provides centralized configuration loading with:
- YAML config file parsing
- Environment variable substitution (${VAR} syntax)
- Automatic .env file loading

Usage:
    from src.utils.config import load_config, get_credential

    config = load_config()  # Loads config with env substitution
    hf_token = get_credential('HF_TOKEN')  # Get credential from .env
"""
from pathlib import Path
import yaml
import os
import re
from typing import Dict, Optional, Any, List
import logging
from dotenv import load_dotenv

# Use standard logging to avoid circular import
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Track if .env has been loaded
_env_loaded = False


def _ensure_env_loaded():
    """Ensure .env file is loaded (once)."""
    global _env_loaded
    if not _env_loaded:
        from .paths import get_project_root
        env_path = get_project_root() / '.env'
        if env_path.exists():
            load_dotenv(env_path, override=True)
            logger.debug(f"Loaded environment from {env_path}")
        _env_loaded = True


def _substitute_env_vars(value: Any) -> Any:
    """Recursively substitute ${VAR} patterns with environment variables."""
    if isinstance(value, str):
        # Match ${VAR} pattern
        pattern = r'\$\{([^}]+)\}'
        matches = re.findall(pattern, value)
        for var_name in matches:
            env_value = os.getenv(var_name, '')
            value = value.replace(f'${{{var_name}}}', env_value)
        return value
    elif isinstance(value, dict):
        return {k: _substitute_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_substitute_env_vars(item) for item in value]
    return value


def load_config(config_path: Optional[Path] = None, substitute_env: bool = True) -> Dict:
    """Load configuration from yaml file with optional env variable substitution.

    Args:
        config_path: Optional path to config file. If not provided, will look in default location.
        substitute_env: If True, substitute ${VAR} patterns with environment variables.

    Returns:
        Dict containing configuration settings with env vars substituted.
    """
    # Ensure .env is loaded before reading config
    _ensure_env_loaded()

    if config_path is None:
        from .paths import get_config_path
        config_path = get_config_path()

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if substitute_env:
        config = _substitute_env_vars(config)

    return config


def get_credential(name: str, default: Optional[str] = None) -> Optional[str]:
    """Get a credential from environment variables.

    This is the preferred way to access credentials. It ensures .env is loaded.

    Args:
        name: Environment variable name (e.g., 'HF_TOKEN', 'S3_SECRET_KEY')
        default: Default value if not found

    Returns:
        Credential value or default
    """
    _ensure_env_loaded()
    return os.getenv(name, default)


def _deep_merge(default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries."""
    result = default.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


# LLM Configuration Functions
def get_llm_config() -> Dict[str, Any]:
    """Get LLM configuration from the main config."""
    config = load_config()

    # Get LLM config with defaults
    llm_config = config.get('processing', {}).get('llms', {})

    # Use centralized node detection (lazy import to avoid circular dependency)
    from src.utils.node_utils import is_head_node
    head_node = is_head_node(config)

    if head_node:
        default_server_url = 'http://localhost:8002'
    else:
        default_server_url = 'http://10.0.0.4:8002'
    
    # Set defaults if not specified
    defaults = {
        'server_url': default_server_url,
        'models': {
            'primary': {
                'name': 'qwen3:8b',
                'description': 'Primary model',
                'max_tokens_speaker_assignment': 10,
                'max_tokens_grammar': None
            },
            'fallback': {
                'name': 'qwen3:8b',
                'description': 'Fallback model',
                'max_tokens_speaker_assignment': 10,
                'max_tokens_grammar': None
            }
        },
        'server': {
            'max_concurrent_requests': 5,
            'request_timeout': 30,
            'comprehensive_timeout': 60
        },
        'temperature': {
            'speaker_assignment': {
                'initial': 0.1,
                'increment': 0.2,
                'max': 0.5
            },
            'grammar': {
                'initial': 0.1,
                'increment': 0.1,
                'max': 0.3
            }
        }
    }
    
    # Merge defaults with config
    return _deep_merge(defaults, llm_config)


def get_llm_server_url() -> str:
    """Get the LLM server URL from config."""
    llm_config = get_llm_config()

    # Use centralized node detection (lazy import to avoid circular dependency)
    from src.utils.node_utils import is_head_node
    head_node = is_head_node()

    if head_node:
        default_url = 'http://localhost:8002'
    else:
        default_url = 'http://10.0.0.4:8002'

    return llm_config.get('server_url', default_url)


def get_model_config(model_type: str = 'primary') -> Dict[str, Any]:
    """Get configuration for a specific model type."""
    llm_config = get_llm_config()
    models = llm_config.get('models', {})
    
    if model_type not in models:
        logger.warning(f"Model type '{model_type}' not found in config, using primary")
        model_type = 'primary'
    
    return models.get(model_type, {})


def get_temperature_config(task_type: str) -> Dict[str, float]:
    """Get temperature configuration for a specific task type."""
    llm_config = get_llm_config()
    temp_config = llm_config.get('temperature', {})
    
    if task_type not in temp_config:
        logger.warning(f"Temperature config for '{task_type}' not found, using defaults")
        return {'initial': 0.1, 'increment': 0.1, 'max': 0.3}
    
    return temp_config.get(task_type, {})


def get_server_config() -> Dict[str, Any]:
    """Get LLM server configuration."""
    llm_config = get_llm_config()
    return llm_config.get('server', {})


# Convenience functions
def get_max_concurrent_requests() -> int:
    """Get maximum concurrent requests setting."""
    server_config = get_server_config()
    return server_config.get('max_concurrent_requests', 5)


def get_request_timeout(comprehensive: bool = False) -> int:
    """Get request timeout in seconds."""
    server_config = get_server_config()
    if comprehensive:
        return server_config.get('comprehensive_timeout', 60)
    return server_config.get('request_timeout', 30)


def get_llm_backend_config() -> Dict[str, Any]:
    """Get LLM server backend configuration.

    Returns dict with:
        - backend: "mlx" or "ollama"
        - endpoints: list of endpoint IPs for the selected backend
        - endpoint_tiers: dict mapping endpoint -> list of supported tiers (mlx only)
        - port: port number for the backend (8004 for mlx, 11434 for ollama)
    """
    config = load_config()
    llm_server_config = config.get('processing', {}).get('llm_server', {})

    backend = llm_server_config.get('backend', 'mlx')

    if backend == 'mlx':
        mlx_endpoints = llm_server_config.get('mlx_endpoints', {})
        # Support both old list format and new dict format with tiers
        if isinstance(mlx_endpoints, list):
            # Old format: list of IPs - all tiers on all endpoints
            endpoints = mlx_endpoints
            endpoint_tiers = {ep: ["tier_1", "tier_2", "tier_3"] for ep in endpoints}
        else:
            # New format: dict with endpoint -> {tiers: [...]}
            endpoints = list(mlx_endpoints.keys())
            endpoint_tiers = {ep: info.get('tiers', ["tier_1", "tier_2", "tier_3"])
                           for ep, info in mlx_endpoints.items()}
        port = 8004  # model_server.py port
    else:
        endpoints = llm_server_config.get('ollama_endpoints', ['localhost'])
        endpoint_tiers = {}  # Not used for ollama
        port = 11434  # Ollama default port

    return {
        'backend': backend,
        'endpoints': endpoints,
        'endpoint_tiers': endpoint_tiers,
        'port': port
    }


def get_llm_task_routing() -> Dict[str, Optional[list]]:
    """Get task routing configuration for LLM server.

    Returns dict mapping task types to list of preferred endpoints (or None for any).
    Resolves worker names to IP addresses.
    """
    config = load_config()
    routing_config = config.get('processing', {}).get('llm_server', {}).get('task_routing', {})

    # Default routing if not specified
    default_routing = {
        'stitch': ['10.0.0.4', 'localhost'],
        'text': None,
        'analysis': None,
        'embedding': None
    }

    # Merge with defaults
    routing = {**default_routing, **routing_config}

    # Resolve worker names to IPs
    workers = config.get('processing', {}).get('workers', {})
    resolved_routing = {}

    for task_type, endpoints in routing.items():
        if endpoints is None:
            resolved_routing[task_type] = None
            continue

        resolved_endpoints = []
        for endpoint in endpoints:
            # If it's already an IP or localhost, keep it
            if endpoint == 'localhost' or endpoint.replace('.', '').isdigit():
                resolved_endpoints.append(endpoint)
            # Otherwise try to resolve as worker name
            elif endpoint in workers:
                worker_info = workers[endpoint]
                # Prefer eth over wifi
                ip = worker_info.get('eth') or worker_info.get('wifi')
                if ip:
                    resolved_endpoints.append(ip)
                else:
                    logger.warning(f"Could not resolve worker '{endpoint}' to IP")
            else:
                logger.warning(f"Unknown endpoint '{endpoint}' - keeping as-is")
                resolved_endpoints.append(endpoint)

        resolved_routing[task_type] = resolved_endpoints if resolved_endpoints else None

    return resolved_routing


def get_project_config(project: str) -> Optional[Dict[str, Any]]:
    """Get configuration for a specific project.

    Args:
        project: Project name (e.g., 'CPRMV')

    Returns:
        Project config dict with start_date, end_date, enabled, priority, etc.
        Returns None if project not found.
    """
    config = load_config()
    projects = config.get('active_projects', {})
    return projects.get(project)


def get_project_date_range(project: str) -> tuple:
    """Get the date range for a project.

    Args:
        project: Project name (e.g., 'CPRMV')

    Returns:
        Tuple of (start_date, end_date) as strings in 'YYYY-MM-DD' format.
        Returns (None, None) if project not found.
    """
    project_config = get_project_config(project)
    if not project_config:
        logger.warning(f"Project '{project}' not found in config")
        return (None, None)

    start_date = project_config.get('start_date')
    end_date = project_config.get('end_date')

    return (start_date, end_date)


def get_active_projects() -> List[str]:
    """Get list of all enabled project names.

    Returns:
        List of project names that have enabled=true in config.
    """
    config = load_config()
    projects = config.get('active_projects', {})
    return [name for name, cfg in projects.items() if cfg.get('enabled', False)]