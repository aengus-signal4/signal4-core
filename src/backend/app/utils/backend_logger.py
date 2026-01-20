"""
Backend Hybrid Logging System
=============================

A two-tier logging system for the backend API:

1. **Workflow log**: Key milestones -> console + workflow.log
   - Request start/complete
   - Cache hit/miss status
   - Pipeline step completions with timing
   - Errors

2. **Component logs**: Debug detail -> per-component files (no console)
   - All debug/info messages for troubleshooting
   - Verbose internal state
   - Not shown on console to reduce noise

Usage:
    # For component-level debug logging (file only)
    from ..utils.backend_logger import get_logger
    logger = get_logger("embedding_service")
    logger.debug("Loading model...")

    # For workflow milestones (console + file)
    from ..utils.backend_logger import log_request_start, log_step_complete
    log_request_start(dashboard_id, workflow, query)
    log_step_complete(dashboard_id, step_num, total_steps, step_name, duration_ms, details)

Console output format (clean, scannable):
    2024-01-20 14:30:15 [health] [request] Starting discourse_summary (no query)
    2024-01-20 14:30:15 [health] [cache] Cache MISS - running full pipeline
    2024-01-20 14:30:16 [health] [1/5] expand_query: Generated 10 variations (450ms)
    2024-01-20 14:30:18 [health] [2/5] retrieve_segments: 342 segments (1892ms)
    2024-01-20 14:30:31 [health] [complete] Analysis complete (15892ms)
"""

import logging
import sys
import time
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional, Dict, Any
from functools import lru_cache

from src.utils.paths import get_project_root

# Log directory structure
LOG_DIR = get_project_root() / 'logs' / 'backend'
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Workflow log (milestones - console + file)
WORKFLOW_LOG = LOG_DIR / 'workflow.log'

# Cache for component loggers
_component_loggers: Dict[str, logging.Logger] = {}

# Workflow logger singleton
_workflow_logger: Optional[logging.Logger] = None

# Request timing tracking
_request_start_times: Dict[str, float] = {}


class WorkflowFormatter(logging.Formatter):
    """Formatter for workflow milestone messages (console + file)."""

    def format(self, record):
        """Format workflow events with clean, scannable output."""
        # Use the message directly - we construct it in the log functions
        return record.getMessage()


class ComponentFormatter(logging.Formatter):
    """Formatter for component debug logs (file only)."""

    def format(self, record):
        """Format component logs with timestamp, level, and message."""
        timestamp = self.formatTime(record, '%Y-%m-%d %H:%M:%S')
        return f"{timestamp} [{record.levelname}] {record.getMessage()}"


def _get_workflow_logger() -> logging.Logger:
    """Get or create the workflow logger (singleton)."""
    global _workflow_logger

    if _workflow_logger is not None:
        return _workflow_logger

    logger = logging.getLogger('backend.workflow')
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler - workflow milestones
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(WorkflowFormatter())
    # Ensure immediate flush
    console_handler.stream.reconfigure(line_buffering=True)
    logger.addHandler(console_handler)

    # File handler - workflow.log with rotation
    file_handler = RotatingFileHandler(
        WORKFLOW_LOG,
        maxBytes=50 * 1024 * 1024,  # 50MB
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(WorkflowFormatter())
    logger.addHandler(file_handler)

    _workflow_logger = logger
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a component logger for debug-level file logging.

    Component loggers write DEBUG+ to per-component files in logs/backend/
    but DO NOT write to console to reduce noise.

    Args:
        name: Component name (e.g., "embedding_service", "text_generator")

    Returns:
        Logger configured for file-only output

    Example:
        logger = get_logger("embedding_service")
        logger.debug("Loading model...")
        logger.info("Model loaded successfully")
    """
    if name in _component_loggers:
        return _component_loggers[name]

    # Normalize name (remove backend. prefix if present)
    clean_name = name.replace('backend.', '').replace('.', '_')
    logger_name = f'backend.{clean_name}'

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    # Clear any existing handlers
    logger.handlers.clear()

    # File handler only - no console to reduce noise
    log_file = LOG_DIR / f'{clean_name}.log'
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=3
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(ComponentFormatter())
    logger.addHandler(file_handler)

    _component_loggers[name] = logger
    return logger


# =============================================================================
# Workflow Event Logging Functions
# =============================================================================

def _format_timestamp() -> str:
    """Get current timestamp in standard format."""
    from datetime import datetime
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def log_workflow_event(dashboard_id: str, category: str, message: str):
    """
    Log a workflow milestone event to console and workflow.log.

    Args:
        dashboard_id: Dashboard identifier (e.g., "health", "cprmv")
        category: Event category (e.g., "request", "cache", "complete")
        message: Event message
    """
    logger = _get_workflow_logger()
    formatted = f"{_format_timestamp()} [{dashboard_id}] [{category}] {message}"
    logger.info(formatted)


def log_request_start(dashboard_id: str, workflow: str, query: Optional[str] = None):
    """
    Log the start of an analysis request.

    Args:
        dashboard_id: Dashboard identifier
        workflow: Workflow name (e.g., "simple_rag", "discourse_summary")
        query: User query (None for landing page workflows)
    """
    _request_start_times[dashboard_id] = time.time()

    if query:
        query_preview = f"'{query[:50]}...'" if len(query) > 50 else f"'{query}'"
        message = f"Starting {workflow} {query_preview}"
    else:
        message = f"Starting {workflow} (no query)"

    log_workflow_event(dashboard_id, "request", message)


def log_request_complete(dashboard_id: str, cached: bool = False):
    """
    Log the completion of an analysis request.

    Args:
        dashboard_id: Dashboard identifier
        cached: Whether the result was served from cache
    """
    start_time = _request_start_times.pop(dashboard_id, None)
    if start_time:
        duration_ms = int((time.time() - start_time) * 1000)
        cache_str = " (cached)" if cached else ""
        message = f"Analysis complete{cache_str} ({duration_ms}ms)"
    else:
        cache_str = " (cached)" if cached else ""
        message = f"Analysis complete{cache_str}"

    log_workflow_event(dashboard_id, "complete", message)


def log_cache_hit(dashboard_id: str, level: str, age_info: str = ""):
    """
    Log a cache hit event.

    Args:
        dashboard_id: Dashboard identifier
        level: Cache level (e.g., "full", "partial", "landing_page")
        age_info: Optional age information (e.g., "age=5.2m")
    """
    if age_info:
        message = f"Cache HIT ({level}, {age_info})"
    else:
        message = f"Cache HIT ({level})"

    log_workflow_event(dashboard_id, "cache", message)


def log_cache_miss(dashboard_id: str, reason: str = ""):
    """
    Log a cache miss event.

    Args:
        dashboard_id: Dashboard identifier
        reason: Optional reason for cache miss
    """
    if reason:
        message = f"Cache MISS - {reason}"
    else:
        message = "Cache MISS - running full pipeline"

    log_workflow_event(dashboard_id, "cache", message)


def log_step_complete(
    dashboard_id: str,
    step_num: int,
    total_steps: int,
    step_name: str,
    duration_ms: int,
    details: str = ""
):
    """
    Log a pipeline step completion.

    Args:
        dashboard_id: Dashboard identifier
        step_num: Current step number (1-indexed)
        total_steps: Total number of steps
        step_name: Name of the completed step
        duration_ms: Step duration in milliseconds
        details: Additional details (e.g., "342 segments")
    """
    if details:
        message = f"{step_name}: {details} ({duration_ms}ms)"
    else:
        message = f"{step_name}: complete ({duration_ms}ms)"

    log_workflow_event(dashboard_id, f"{step_num}/{total_steps}", message)


def log_step_start(dashboard_id: str, step_num: int, total_steps: int, step_name: str):
    """
    Log a pipeline step start (optional - for long-running steps).

    Args:
        dashboard_id: Dashboard identifier
        step_num: Current step number (1-indexed)
        total_steps: Total number of steps
        step_name: Name of the starting step
    """
    log_workflow_event(dashboard_id, f"{step_num}/{total_steps}", f"{step_name}: starting...")


def log_error(dashboard_id: str, error: str, step: Optional[str] = None):
    """
    Log an error event.

    Args:
        dashboard_id: Dashboard identifier
        error: Error message
        step: Optional step name where error occurred
    """
    if step:
        message = f"Error in {step}: {error}"
    else:
        message = f"Error: {error}"

    log_workflow_event(dashboard_id, "error", message)


# =============================================================================
# Backward Compatibility
# =============================================================================

def setup_backend_logger():
    """Legacy function - returns workflow logger for backward compatibility."""
    return _get_workflow_logger()


# Global backend logger for imports that expect it
backend_logger = _get_workflow_logger()
