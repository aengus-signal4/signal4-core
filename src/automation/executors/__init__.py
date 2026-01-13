"""
Task Executors for Unified Scheduled Task Manager

Provides different execution strategies:
- CLIExecutor: Run shell commands
- SQLExecutor: Execute PostgreSQL functions
"""
from .base import BaseExecutor, ExecutionResult
from .cli import CLIExecutor
from .sql import SQLExecutor

__all__ = ['BaseExecutor', 'ExecutionResult', 'CLIExecutor', 'SQLExecutor']
