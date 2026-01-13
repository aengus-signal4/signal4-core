"""
Pipeline module for content processing task management.

This module provides:
- TaskResultHandler: Handles task completion/failure outcomes
- TaskCreator: Creates tasks based on content state
- StateEvaluator: Evaluates content state and determines next actions

Together these form the core of the PipelineManager functionality.
"""

from .task_creator import TaskCreator
from .result_handler import TaskResultHandler
from .state_evaluator import StateEvaluator

__all__ = ['TaskCreator', 'TaskResultHandler', 'StateEvaluator']
