"""
S3 Monitor - Detects S3 errors and manages global pause functionality
"""
import asyncio
import aiohttp
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)

class S3Monitor:
    """Monitors S3 connectivity and manages global pauses on S3 errors"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.should_stop = False
        self._health_check_task: Optional[asyncio.Task] = None
        
        # S3 error detection
        self.s3_error_keywords = [
            "Could not connect to the endpoint URL",
            "Connect timeout on endpoint URL", 
            "S3 connection failed",
            "Failed to connect to S3",
            "endpoint URL",
            "Connection timeout",
            "ConnectTimeoutError"
        ]
        
        # Configuration
        self.pause_duration = 120  # 2 minutes default pause
        self.health_check_interval = 120  # 2 minutes between health checks during pause
        self.max_health_check_attempts = 10  # Max attempts before giving up
        
    def is_s3_error(self, error_message: str) -> bool:
        """Check if error message indicates S3 connectivity issues"""
        if not error_message:
            return False
            
        error_lower = error_message.lower()
        return any(keyword.lower() in error_lower for keyword in self.s3_error_keywords)
    
    def should_trigger_global_pause(self, error_message: str) -> bool:
        """Determine if error should trigger global pause"""
        return self.is_s3_error(error_message)
    
    async def check_s3_health(self) -> bool:
        """Check S3 connectivity by attempting to connect to endpoint"""
        try:
            # Get S3 config
            s3_config = self.config.get('storage', {}).get('s3', {})
            endpoint_url = s3_config.get('endpoint_url')
            
            if not endpoint_url:
                logger.error("No S3 endpoint URL configured")
                return False
            
            # Attempt to connect to MinIO health endpoint
            health_endpoint = f"{endpoint_url}/minio/health/ready"
            timeout = aiohttp.ClientTimeout(total=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                try:
                    async with session.head(health_endpoint) as response:
                        # Any response under 400 indicates S3 is healthy
                        is_healthy = response.status < 400
                        
                        if is_healthy:
                            logger.info("S3 health check passed")
                        else:
                            logger.warning(f"S3 health check failed with status {response.status}")
                        
                        return is_healthy
                        
                except asyncio.TimeoutError:
                    logger.warning("S3 health check timed out")
                    return False
                except aiohttp.ClientConnectorError as e:
                    logger.warning(f"S3 health check connection error: {str(e)}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error in S3 health check: {str(e)}")
            return False
    
    async def start_health_monitoring(self, orchestrator) -> asyncio.Task:
        """Start health monitoring during global pause"""
        if self._health_check_task and not self._health_check_task.done():
            logger.warning("S3 health monitoring already running")
            return self._health_check_task
            
        self.should_stop = False
        self._health_check_task = asyncio.create_task(
            self._health_check_loop(orchestrator)
        )
        
        logger.info("Started S3 health monitoring during global pause")
        return self._health_check_task
    
    async def stop_health_monitoring(self):
        """Stop health monitoring"""
        self.should_stop = True
        
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None
        
        logger.info("Stopped S3 health monitoring")
    
    async def _health_check_loop(self, orchestrator):
        """Health check loop that runs during global pause"""
        attempt = 0
        
        while not self.should_stop and attempt < self.max_health_check_attempts:
            try:
                # Check if global pause is still active
                if not orchestrator.global_pause_until:
                    logger.info("Global pause cleared externally, stopping S3 health monitoring")
                    break
                
                now = datetime.now(timezone.utc)
                if now >= orchestrator.global_pause_until:
                    logger.info("Global pause expired, stopping S3 health monitoring")
                    break
                
                # Check S3 health
                is_healthy = await self.check_s3_health()
                attempt += 1
                
                if is_healthy:
                    logger.info("S3 is healthy again, clearing global pause")
                    orchestrator.global_pause_until = None
                    break
                else:
                    remaining_time = (orchestrator.global_pause_until - now).total_seconds()
                    logger.info(f"S3 still unhealthy (attempt {attempt}/{self.max_health_check_attempts}). "
                              f"Global pause continues for {remaining_time:.1f}s more")
                
                # Wait before next check
                await asyncio.sleep(self.health_check_interval)
                
            except asyncio.CancelledError:
                logger.info("S3 health check loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in S3 health check loop: {str(e)}")
                await asyncio.sleep(self.health_check_interval)
        
        # If we reached max attempts without success, log warning
        if attempt >= self.max_health_check_attempts:
            logger.warning(f"S3 health check gave up after {attempt} attempts. "
                         "Global pause will continue until natural expiration.")
        
        logger.info("S3 health check monitoring stopped")
    
    def calculate_pause_duration(self, error_message: str) -> int:
        """Calculate how long to pause based on error type"""
        # Could implement different pause durations for different error types
        return self.pause_duration
    
    def extend_global_pause(self, orchestrator, error_message: str) -> datetime:
        """Extend or set global pause due to S3 error"""
        now = datetime.now(timezone.utc)
        pause_duration = self.calculate_pause_duration(error_message)
        new_pause_until = now + timedelta(seconds=pause_duration)
        
        # Only extend if new pause is longer than current
        if not orchestrator.global_pause_until or orchestrator.global_pause_until < new_pause_until:
            old_pause = orchestrator.global_pause_until
            orchestrator.global_pause_until = new_pause_until
            
            if old_pause:
                logger.warning(f"Extended global pause from {old_pause} to {new_pause_until} "
                             f"due to S3 error: {error_message[:100]}")
            else:
                logger.warning(f"Set global pause until {new_pause_until} "
                             f"due to S3 error: {error_message[:100]}")
            
            # Start health monitoring if not already running
            if not self._health_check_task or self._health_check_task.done():
                asyncio.create_task(self.start_health_monitoring(orchestrator))
        
        return orchestrator.global_pause_until
    
    def get_error_classification(self, error_message: str) -> Dict[str, Any]:
        """Classify an error message"""
        return {
            'is_s3_error': self.is_s3_error(error_message),
            'should_pause': self.should_trigger_global_pause(error_message),
            'pause_duration': self.calculate_pause_duration(error_message) if self.should_trigger_global_pause(error_message) else 0,
            'error_type': 's3_connectivity' if self.is_s3_error(error_message) else 'other'
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get S3 monitoring statistics"""
        return {
            'monitoring_active': self._health_check_task is not None and not self._health_check_task.done(),
            'pause_duration': self.pause_duration,
            'health_check_interval': self.health_check_interval,
            'max_health_check_attempts': self.max_health_check_attempts,
            's3_error_keywords': self.s3_error_keywords,
            's3_endpoint': self.config.get('storage', {}).get('s3', {}).get('endpoint_url', 'not_configured')
        }