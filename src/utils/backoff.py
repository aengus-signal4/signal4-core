"""
Exponential Backoff Mixin - Shared retry logic for workers and services

Provides consistent exponential backoff behavior:
- Base delay: 60 seconds
- Maximum delay: 24 hours
- Maximum attempts: 20
- Exponential growth: 2^attempts

Usage:
    class MyClass(ExponentialBackoffMixin):
        def __init__(self):
            super().__init__()
            self.init_backoff()  # Initialize backoff state
"""
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


class ExponentialBackoffMixin:
    """
    Mixin providing exponential backoff retry logic.

    Classes using this mixin should call init_backoff() in their __init__.
    """

    # Default configuration (can be overridden by subclasses)
    DEFAULT_BACKOFF_BASE = 60  # 1 minute
    DEFAULT_BACKOFF_MAX = 60 * 60 * 24  # 24 hours
    DEFAULT_MAX_ATTEMPTS = 20

    def init_backoff(
        self,
        base: int = None,
        max_delay: int = None,
        max_attempts: int = None
    ):
        """Initialize backoff state. Call this in subclass __init__."""
        self.restart_attempts = 0
        self.last_restart_attempt: Optional[datetime] = None
        self.restart_backoff_base = base or self.DEFAULT_BACKOFF_BASE
        self.restart_backoff_max = max_delay or self.DEFAULT_BACKOFF_MAX
        self.max_restart_attempts = max_attempts or self.DEFAULT_MAX_ATTEMPTS
        self.restart_lock = asyncio.Lock()

    def should_attempt_restart(self) -> bool:
        """Check if we should attempt a restart based on backoff logic."""
        if self.restart_attempts >= self.max_restart_attempts:
            return False

        if self.last_restart_attempt is None:
            return True

        backoff_time = self._calculate_backoff()
        time_since_last = (datetime.now(timezone.utc) - self.last_restart_attempt).total_seconds()
        return time_since_last >= backoff_time

    def get_restart_backoff_remaining(self) -> float:
        """Get remaining seconds until next restart attempt is allowed."""
        if self.last_restart_attempt is None:
            return 0.0
        if self.restart_attempts >= self.max_restart_attempts:
            return 0.0

        backoff_time = self._calculate_backoff()
        time_since_last = (datetime.now(timezone.utc) - self.last_restart_attempt).total_seconds()
        return max(0.0, backoff_time - time_since_last)

    def record_restart_attempt(self, success: bool):
        """Record a restart attempt and update backoff state."""
        self.last_restart_attempt = datetime.now(timezone.utc)

        if success:
            self.restart_attempts = 0
            self._log_restart_success()
        else:
            self.restart_attempts += 1
            self._log_restart_failure()

    def reset_backoff(self):
        """Reset backoff state (e.g., after manual intervention)."""
        self.restart_attempts = 0
        self.last_restart_attempt = None

    def _calculate_backoff(self) -> float:
        """Calculate current backoff delay in seconds."""
        return min(
            self.restart_backoff_base * (2 ** self.restart_attempts),
            self.restart_backoff_max
        )

    def _get_identifier(self) -> str:
        """Get identifier for logging. Override in subclasses."""
        if hasattr(self, 'worker_id'):
            return self.worker_id
        if hasattr(self, 'service_name'):
            return self.service_name
        return str(id(self))

    def _log_restart_success(self):
        """Log successful restart. Can be overridden."""
        logger.info(f"{self._get_identifier()} successfully restarted, reset retry count")

    def _log_restart_failure(self):
        """Log failed restart. Can be overridden."""
        next_backoff = self._calculate_backoff()
        logger.warning(
            f"{self._get_identifier()} restart attempt "
            f"{self.restart_attempts}/{self.max_restart_attempts} failed. "
            f"Next retry in {next_backoff}s"
        )

    async def attempt_restart_with_lock(self, restart_func) -> tuple[bool, bool]:
        """
        Attempt restart with lock to prevent concurrent attempts.

        Args:
            restart_func: Async function that performs the restart.
                         Should take the identifier as argument and return bool.

        Returns:
            (success: bool, attempted: bool) - whether restart succeeded and was attempted
        """
        if self.restart_lock.locked():
            logger.debug(f"{self._get_identifier()} restart already in progress, skipping")
            return False, False

        async with self.restart_lock:
            if not self.should_attempt_restart():
                backoff_remaining = self.get_restart_backoff_remaining()
                logger.debug(
                    f"{self._get_identifier()} restart not allowed: "
                    f"attempts={self.restart_attempts}/{self.max_restart_attempts}, "
                    f"backoff_remaining={backoff_remaining:.1f}s"
                )
                return False, False

            logger.info(
                f"Attempting to restart {self._get_identifier()} "
                f"(attempt {self.restart_attempts + 1}/{self.max_restart_attempts})"
            )

            restart_success = await restart_func(self._get_identifier())
            self.record_restart_attempt(restart_success)

            return restart_success, True
