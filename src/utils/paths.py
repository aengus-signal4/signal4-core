"""
Path utilities for consistent path resolution across the codebase.

Usage:
    from src.utils.paths import get_project_root, get_config_path

    root = get_project_root()
    config = get_config_path()
"""
from pathlib import Path
from typing import Optional
import os

# Cache the project root
_project_root: Optional[Path] = None


def get_project_root() -> Path:
    """Get the project root directory (where config/ lives).

    This is the canonical way to resolve paths - use this instead of
    repeated `Path(__file__).parent.parent...` patterns.

    Returns:
        Path to project root (e.g., /Users/signal4/signal4/core)
    """
    global _project_root
    if _project_root is None:
        # This file is at src/utils/paths.py, so go up 3 levels
        _project_root = Path(__file__).parent.parent.parent.resolve()
    return _project_root


# Cache the env path
_env_path: Optional[Path] = None


def get_env_path() -> Path:
    """Get the path to the .env file.

    The .env file lives in the parent signal4/ directory, not in core/.
    This is the canonical way to find credentials.

    Returns:
        Path to .env file (e.g., /Users/signal4/signal4/.env)
    """
    global _env_path
    if _env_path is None:
        # .env is in the parent directory (signal4/), not in core/
        _env_path = get_project_root().parent / '.env'
    return _env_path


def get_config_path(filename: str = "config.yaml") -> Path:
    """Get path to a config file.

    Args:
        filename: Config filename (default: config.yaml)

    Returns:
        Path to config file
    """
    return get_project_root() / "config" / filename


def get_log_dir() -> Path:
    """Get the centralized log directory.

    Returns:
        Path to log directory (creates if doesn't exist)
    """
    log_dir = Path("/Users/signal4/logs/content_processing")
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def get_test_content_path(content_id: str) -> Path:
    """Get path to test content directory.

    Args:
        content_id: Content ID for test data

    Returns:
        Path to test content directory
    """
    return get_project_root() / "tests" / "content" / content_id


def ensure_dir(path: Path) -> Path:
    """Ensure a directory exists, creating if necessary.

    Args:
        path: Directory path

    Returns:
        The same path (for chaining)
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_dashboard_config_path(dashboard_name: str, config_file: str = "config.yaml") -> Path:
    """Get path to a dashboard config file.

    Args:
        dashboard_name: Dashboard name (e.g., 'health_wellness')
        config_file: Config filename (default: config.yaml)

    Returns:
        Path to dashboard config file
    """
    return get_project_root() / "config" / "dashboards" / dashboard_name / config_file
