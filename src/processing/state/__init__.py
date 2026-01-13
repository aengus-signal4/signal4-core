"""
State management module for content processing pipeline.

This module provides shared utilities for:
- S3 file existence checking (S3ContentChecker)
- Database flag reconciliation (FlagReconciler)
- Content state machine (ContentStateMachine)
- State-driven task handling (StateDrivenHandler)
- Linear processing state (ProcessingState, LinearStateMachine)
"""

from .s3_content_checker import S3ContentChecker, ContentFileIndex
from .flag_reconciler import FlagReconciler
from .content_state_machine import (
    ContentStateMachine,
    ContentState,
    FailureType,
    get_content_state
)
from .state_driven_handler import StateDrivenHandler
from .linear_state_model import (
    ProcessingState,
    LinearStateMachine,
)

__all__ = [
    # S3 checking
    'S3ContentChecker',
    'ContentFileIndex',
    # Flag reconciliation (legacy)
    'FlagReconciler',
    # Full state machine (complex)
    'ContentStateMachine',
    'ContentState',
    'FailureType',
    'get_content_state',
    'StateDrivenHandler',
    # Linear state model (simple, preferred)
    'ProcessingState',
    'LinearStateMachine',
]
