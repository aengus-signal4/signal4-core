"""
Head Node Service Monitor - Health monitoring and auto-restart for head node services

Monitors:
- Backend API (port 7999)
- Embedding Server (port 8005)
- LLM Server (port 8002)
- Model Servers (port 8004+)

Uses exponential backoff with the same pattern as WorkerInfo for restart attempts.
"""
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, TYPE_CHECKING
from enum import Enum
import aiohttp

if TYPE_CHECKING:
    from src.workers.service_startup import ServiceStartupManager

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Status of a head node service"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    RESTARTING = "restarting"
    FAILED = "failed"
    PERMANENTLY_FAILED = "permanently_failed"
    DISABLED = "disabled"


class HeadNodeServiceInfo:
    """
    Tracks restart state for a single head node service.

    Uses the same exponential backoff pattern as WorkerInfo:
    - base=60s, max=24h, max_attempts=20
    - Requires 3 consecutive health check failures before restarting
    """

    def __init__(
        self,
        service_name: str,
        port: int,
        health_endpoint: str = "/health",
        enabled: bool = True,
        depends_on: Optional[str] = None
    ):
        self.service_name = service_name
        self.port = port
        self.health_endpoint = health_endpoint
        self.enabled = enabled
        self.depends_on = depends_on  # Service dependency (e.g., backend depends on embedding_server)

        # Status tracking
        self.status = ServiceStatus.HEALTHY if enabled else ServiceStatus.DISABLED
        self.consecutive_failures = 0
        self.failure_threshold = 3  # Require 3 consecutive failures before restart

        # Restart tracking with exponential backoff (same as WorkerInfo)
        self.restart_attempts = 0
        self.last_restart_attempt: Optional[datetime] = None
        self.max_restart_attempts = 20
        self.restart_backoff_base = 60  # 1 minute base
        self.restart_backoff_max = 60 * 60 * 24  # 24 hours max

        # Restart lock to prevent concurrent restart attempts
        self.restart_lock = asyncio.Lock()

        # Timestamps
        self.last_health_check: Optional[datetime] = None
        self.last_healthy: Optional[datetime] = None
        self.last_restart_success: Optional[datetime] = None

    def should_attempt_restart(self) -> bool:
        """Check if we should attempt to restart this service based on retry logic."""
        if self.restart_attempts >= self.max_restart_attempts:
            return False

        if self.last_restart_attempt is None:
            return True

        # Calculate exponential backoff
        backoff_time = min(
            self.restart_backoff_base * (2 ** self.restart_attempts),
            self.restart_backoff_max
        )

        time_since_last_attempt = (datetime.now(timezone.utc) - self.last_restart_attempt).total_seconds()
        return time_since_last_attempt >= backoff_time

    def get_restart_backoff_remaining(self) -> float:
        """Get remaining time until next restart attempt is allowed."""
        if self.last_restart_attempt is None or self.restart_attempts >= self.max_restart_attempts:
            return 0.0

        backoff_time = min(
            self.restart_backoff_base * (2 ** self.restart_attempts),
            self.restart_backoff_max
        )

        time_since_last_attempt = (datetime.now(timezone.utc) - self.last_restart_attempt).total_seconds()
        return max(0.0, backoff_time - time_since_last_attempt)

    def record_restart_attempt(self, success: bool):
        """Record a restart attempt and its outcome."""
        self.last_restart_attempt = datetime.now(timezone.utc)

        if success:
            # Reset retry tracking on successful restart
            self.restart_attempts = 0
            self.consecutive_failures = 0
            self.status = ServiceStatus.HEALTHY
            self.last_restart_success = datetime.now(timezone.utc)
            logger.info(f"Service {self.service_name} successfully restarted, reset retry count")
        else:
            self.restart_attempts += 1
            next_backoff = min(
                self.restart_backoff_base * (2 ** self.restart_attempts),
                self.restart_backoff_max
            )

            if self.restart_attempts >= self.max_restart_attempts:
                self.status = ServiceStatus.PERMANENTLY_FAILED
                logger.error(f"Service {self.service_name} permanently failed after {self.restart_attempts} restart attempts")
            else:
                self.status = ServiceStatus.FAILED
                logger.warning(
                    f"Service {self.service_name} restart attempt {self.restart_attempts}/{self.max_restart_attempts} failed. "
                    f"Next retry in {next_backoff}s"
                )

    def record_health_check(self, is_healthy: bool):
        """Record result of a health check."""
        self.last_health_check = datetime.now(timezone.utc)

        if is_healthy:
            self.consecutive_failures = 0
            if self.status not in (ServiceStatus.DISABLED, ServiceStatus.PERMANENTLY_FAILED):
                self.status = ServiceStatus.HEALTHY
            self.last_healthy = datetime.now(timezone.utc)
            # Reset restart attempts on sustained health
            if self.restart_attempts > 0:
                self.restart_attempts = 0
                logger.info(f"Service {self.service_name} is healthy, reset restart attempts")
        else:
            self.consecutive_failures += 1
            if self.status == ServiceStatus.HEALTHY:
                self.status = ServiceStatus.UNHEALTHY

    def needs_restart(self) -> bool:
        """Check if service has exceeded failure threshold and needs restart."""
        return (
            self.consecutive_failures >= self.failure_threshold and
            self.status not in (ServiceStatus.DISABLED, ServiceStatus.PERMANENTLY_FAILED, ServiceStatus.RESTARTING)
        )

    def get_status_dict(self) -> Dict[str, Any]:
        """Get service status as dictionary."""
        return {
            'service_name': self.service_name,
            'port': self.port,
            'enabled': self.enabled,
            'status': self.status.value,
            'consecutive_failures': self.consecutive_failures,
            'failure_threshold': self.failure_threshold,
            'depends_on': self.depends_on,
            'restart_info': {
                'restart_attempts': self.restart_attempts,
                'max_restart_attempts': self.max_restart_attempts,
                'last_restart_attempt': self.last_restart_attempt.isoformat() if self.last_restart_attempt else None,
                'last_restart_success': self.last_restart_success.isoformat() if self.last_restart_success else None,
                'can_restart': self.should_attempt_restart(),
                'backoff_remaining': self.get_restart_backoff_remaining()
            },
            'timestamps': {
                'last_health_check': self.last_health_check.isoformat() if self.last_health_check else None,
                'last_healthy': self.last_healthy.isoformat() if self.last_healthy else None
            }
        }


class HeadNodeServiceMonitor:
    """
    Monitors head node services and automatically restarts them on failure.

    Integrates with the HealthChecker background loop and uses ServiceStartupManager
    for actual restart operations.
    """

    def __init__(self, service_startup: 'ServiceStartupManager', config: Dict[str, Any]):
        self.service_startup = service_startup
        self.config = config
        self.services: Dict[str, HeadNodeServiceInfo] = {}

        # HTTP client for health checks
        self._session: Optional[aiohttp.ClientSession] = None
        self.health_check_timeout = 5.0  # seconds

        # Initialize services from config
        self._initialize_services()

        logger.info(f"HeadNodeServiceMonitor initialized with {len(self.services)} services")

    def _initialize_services(self):
        """Initialize service info from config."""
        services_config = self.config.get('services', {})
        processing_config = self.config.get('processing', {})

        # Backend API
        backend_config = services_config.get('backend', {})
        if backend_config.get('enabled', True):
            self.services['backend'] = HeadNodeServiceInfo(
                service_name='backend',
                port=backend_config.get('port', 7999),
                health_endpoint='/health',
                enabled=True,
                depends_on='embedding_server'  # Backend depends on embedding server
            )

        # Embedding Server
        embedding_config = services_config.get('embedding_server', {})
        if embedding_config.get('enabled', True):
            self.services['embedding_server'] = HeadNodeServiceInfo(
                service_name='embedding_server',
                port=embedding_config.get('port', 8005),
                health_endpoint='/health',
                enabled=True,
                depends_on=None
            )

        # LLM Server
        llm_config = processing_config.get('llm_server', {})
        if llm_config.get('enabled', False):
            self.services['llm_server'] = HeadNodeServiceInfo(
                service_name='llm_server',
                port=llm_config.get('port', 8002),
                health_endpoint='/health',
                enabled=True,
                depends_on=None
            )

        # Model Servers (local only - remote ones have their own monitoring)
        model_servers_config = services_config.get('model_servers', {})
        if model_servers_config.get('enabled', False):
            for instance_name, instance_config in model_servers_config.get('instances', {}).items():
                if not instance_config.get('enabled', True):
                    continue

                host = instance_config.get('host', 'localhost')
                # Only monitor local model servers
                if host in ('localhost', '10.0.0.4'):
                    service_name = f'model_server_{instance_name}'
                    self.services[service_name] = HeadNodeServiceInfo(
                        service_name=service_name,
                        port=instance_config.get('port', 8004),
                        health_endpoint='/health',
                        enabled=True,
                        depends_on=None
                    )

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.health_check_timeout)
            )
        return self._session

    async def close(self):
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def check_service_health(self, service_name: str) -> bool:
        """
        Check health of a specific service via HTTP health endpoint.

        Returns True if healthy, False otherwise.
        """
        service = self.services.get(service_name)
        if not service or not service.enabled:
            return True  # Consider disabled services as healthy

        try:
            session = await self._get_session()
            url = f"http://127.0.0.1:{service.port}{service.health_endpoint}"

            async with session.get(url) as response:
                is_healthy = response.status == 200
                service.record_health_check(is_healthy)
                return is_healthy

        except asyncio.TimeoutError:
            logger.warning(f"Health check timeout for service {service_name}")
            service.record_health_check(False)
            return False
        except aiohttp.ClientError as e:
            logger.warning(f"Health check failed for service {service_name}: {e}")
            service.record_health_check(False)
            return False
        except Exception as e:
            logger.error(f"Unexpected error checking health of {service_name}: {e}")
            service.record_health_check(False)
            return False

    async def check_all_services_health(self) -> Dict[str, bool]:
        """
        Check health of all monitored services.

        Returns dict of service_name -> is_healthy.
        """
        results = {}

        # Check services in dependency order
        # First: services with no dependencies
        for service_name, service in self.services.items():
            if service.depends_on is None:
                results[service_name] = await self.check_service_health(service_name)

        # Then: services with dependencies
        for service_name, service in self.services.items():
            if service.depends_on is not None:
                results[service_name] = await self.check_service_health(service_name)

        return results

    async def handle_unhealthy_service(self, service_name: str) -> bool:
        """
        Handle an unhealthy service - check if restart is needed and attempt it.

        Returns True if service was restarted successfully or no action needed.
        """
        service = self.services.get(service_name)
        if not service:
            return True

        # Check if service needs restart
        if not service.needs_restart():
            return True

        # Check if we can attempt restart
        if not service.should_attempt_restart():
            backoff_remaining = service.get_restart_backoff_remaining()
            logger.debug(
                f"Service {service_name} unhealthy but cannot restart yet "
                f"(backoff: {backoff_remaining:.1f}s, attempts: {service.restart_attempts}/{service.max_restart_attempts})"
            )
            return False

        # Attempt restart
        return await self._attempt_service_restart(service_name)

    async def _attempt_service_restart(self, service_name: str) -> bool:
        """
        Attempt to restart a service with lock to prevent concurrent attempts.

        Returns True if restart succeeded.
        """
        service = self.services.get(service_name)
        if not service:
            return False

        # Try to acquire lock
        if service.restart_lock.locked():
            logger.debug(f"Service {service_name} restart already in progress, skipping")
            return False

        async with service.restart_lock:
            # Double-check should_attempt_restart inside the lock
            if not service.should_attempt_restart():
                return False

            # Check dependency health first
            if service.depends_on:
                dep_service = self.services.get(service.depends_on)
                if dep_service and dep_service.status != ServiceStatus.HEALTHY:
                    logger.warning(
                        f"Cannot restart {service_name}: dependency {service.depends_on} is not healthy"
                    )
                    return False

            service.status = ServiceStatus.RESTARTING
            logger.info(
                f"Attempting to restart service {service_name} "
                f"(attempt {service.restart_attempts + 1}/{service.max_restart_attempts})"
            )

            try:
                # Use ServiceStartupManager to restart
                success = await self.service_startup.restart_service(service_name)

                if success:
                    # Wait a moment for service to fully start
                    await asyncio.sleep(3)

                    # Verify health after restart
                    is_healthy = await self.check_service_health(service_name)

                    if is_healthy:
                        service.record_restart_attempt(True)
                        logger.info(f"Successfully restarted service {service_name}")
                        return True
                    else:
                        logger.warning(f"Service {service_name} restarted but health check failed")
                        service.record_restart_attempt(False)
                        return False
                else:
                    logger.error(f"Failed to restart service {service_name}")
                    service.record_restart_attempt(False)
                    return False

            except Exception as e:
                logger.error(f"Error restarting service {service_name}: {e}")
                service.record_restart_attempt(False)
                return False

    async def manual_restart_service(self, service_name: str) -> bool:
        """
        Manually restart a service, ignoring backoff timers.

        Resets restart attempts before attempting restart.
        """
        service = self.services.get(service_name)
        if not service:
            logger.error(f"Unknown service: {service_name}")
            return False

        if service.status == ServiceStatus.DISABLED:
            logger.error(f"Cannot restart disabled service: {service_name}")
            return False

        # Reset retry tracking for manual restart
        logger.info(f"Manual restart requested for service {service_name}")
        service.restart_attempts = 0
        service.last_restart_attempt = None

        return await self._attempt_service_restart(service_name)

    def get_status(self) -> Dict[str, Any]:
        """Get status of all monitored services."""
        return {
            'services': {
                name: service.get_status_dict()
                for name, service in self.services.items()
            },
            'summary': {
                'total': len(self.services),
                'healthy': sum(1 for s in self.services.values() if s.status == ServiceStatus.HEALTHY),
                'unhealthy': sum(1 for s in self.services.values() if s.status == ServiceStatus.UNHEALTHY),
                'failed': sum(1 for s in self.services.values() if s.status in (ServiceStatus.FAILED, ServiceStatus.PERMANENTLY_FAILED)),
                'disabled': sum(1 for s in self.services.values() if s.status == ServiceStatus.DISABLED)
            }
        }

    def get_service_status(self, service_name: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific service."""
        service = self.services.get(service_name)
        if service:
            return service.get_status_dict()
        return None
