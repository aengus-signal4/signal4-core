"""
System Monitoring Dashboard Package

Modular components for the system monitoring dashboard.
"""

from .config import (
    ORCHESTRATOR_API_URL,
    LLM_BALANCER_URL,
    MODEL_SERVERS,
    WORKER_PROCESSORS,
    LOG_SOURCES,
    QUICK_STATUS_GROUPS,
    PIPELINE_COLORS,
    TASK_COLORS,
    load_config,
)

__all__ = [
    'ORCHESTRATOR_API_URL',
    'LLM_BALANCER_URL',
    'MODEL_SERVERS',
    'WORKER_PROCESSORS',
    'LOG_SOURCES',
    'QUICK_STATUS_GROUPS',
    'PIPELINE_COLORS',
    'TASK_COLORS',
    'load_config',
]
