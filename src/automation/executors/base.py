"""
Base Executor Interface for Scheduled Tasks
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from datetime import datetime, timezone


@dataclass
class ExecutionResult:
    """Result of a task execution"""
    success: bool
    start_time: datetime
    end_time: datetime
    output: Optional[Dict[str, Any]] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def duration_seconds(self) -> float:
        return (self.end_time - self.start_time).total_seconds()


class BaseExecutor(ABC):
    """Abstract base class for task executors"""

    @abstractmethod
    async def execute(
        self,
        config: Dict[str, Any],
        context: Dict[str, Any],
        timeout_seconds: int = 3600
    ) -> ExecutionResult:
        """
        Execute the task.

        Args:
            config: Executor-specific configuration
            context: Execution context (e.g., orchestrator reference, working dir)
            timeout_seconds: Maximum execution time

        Returns:
            ExecutionResult with success status, timing, and output
        """
        pass

    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate the executor configuration.

        Args:
            config: Configuration to validate

        Returns:
            True if config is valid
        """
        pass
