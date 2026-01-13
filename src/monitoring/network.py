"""
Network Manager - Handles worker connectivity, health checks, and network operations
"""
import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ConnectionStatus(Enum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    TIMEOUT = "timeout"
    ERROR = "error"

@dataclass
class NetworkInterface:
    """Network interface information"""
    name: str
    ip: str
    is_primary: bool = False
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    consecutive_failures: int = 0

@dataclass
class WorkerNetworkStatus:
    """Network status for a worker"""
    worker_id: str
    current_interface: Optional[NetworkInterface] = None
    available_interfaces: List[NetworkInterface] = None
    last_successful_connection: Optional[datetime] = None
    health_check_url: str = ""
    api_url: str = ""

class NetworkManager:
    """Manages network connectivity and health checks for workers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logger
        
        # Network tracking
        self.worker_network_status: Dict[str, WorkerNetworkStatus] = {}
        
        # Configuration
        self.health_check_interval = 30  # seconds
        self.max_consecutive_failures = 3
        self.interface_cooldown = 60  # seconds before retrying failed interface
        self.connection_timeout = 10  # seconds
        
        # HTTP session for reuse
        self.session = None
    
    async def initialize(self):
        """Initialize network manager"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.connection_timeout)
        )
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
    
    def register_worker(self, worker_id: str, worker_config: Dict[str, Any]) -> bool:
        """Register a worker with its network configuration"""
        try:
            interfaces = []
            
            # Add ethernet interface if available
            if 'eth' in worker_config and worker_config['eth']:
                interfaces.append(NetworkInterface(
                    name='eth',
                    ip=worker_config['eth'],
                    is_primary=True
                ))
            
            # Add wifi interface if available
            if 'wifi' in worker_config and worker_config['wifi']:
                interfaces.append(NetworkInterface(
                    name='wifi',
                    ip=worker_config['wifi'],
                    is_primary=len(interfaces) == 0  # Primary if no eth
                ))
            
            if not interfaces:
                self.logger.error(f"No network interfaces found for worker {worker_id}")
                return False
            
            # Determine current interface (prefer primary)
            current_interface = None
            for interface in interfaces:
                if interface.is_primary:
                    current_interface = interface
                    break
            if not current_interface:
                current_interface = interfaces[0]
            
            # Create network status
            status = WorkerNetworkStatus(
                worker_id=worker_id,
                current_interface=current_interface,
                available_interfaces=interfaces,
                health_check_url=f"http://{current_interface.ip}:8000/tasks",
                api_url=f"http://{current_interface.ip}:8000"
            )
            
            self.worker_network_status[worker_id] = status
            self.logger.info(f"Registered worker {worker_id} with {len(interfaces)} interfaces")
            return True
            
        except Exception as e:
            self.logger.error(f"Error registering worker {worker_id}: {str(e)}")
            return False
    
    async def check_worker_health(self, worker_id: str) -> Tuple[bool, ConnectionStatus]:
        """Check if worker is healthy and reachable"""
        if worker_id not in self.worker_network_status:
            return False, ConnectionStatus.ERROR
        
        status = self.worker_network_status[worker_id]
        
        # Try current interface first
        if status.current_interface:
            result = await self._check_interface_health(worker_id, status.current_interface)
            if result == ConnectionStatus.CONNECTED:
                status.current_interface.last_success = datetime.now(timezone.utc)
                status.current_interface.consecutive_failures = 0
                status.last_successful_connection = datetime.now(timezone.utc)
                return True, ConnectionStatus.CONNECTED
            else:
                status.current_interface.last_failure = datetime.now(timezone.utc)
                status.current_interface.consecutive_failures += 1
        
        # If current interface failed, try alternatives
        if (status.current_interface and 
            status.current_interface.consecutive_failures >= self.max_consecutive_failures):
            
            for interface in status.available_interfaces:
                if interface == status.current_interface:
                    continue
                
                # Check if interface is in cooldown
                if (interface.last_failure and 
                    (datetime.now(timezone.utc) - interface.last_failure).seconds < self.interface_cooldown):
                    continue
                
                result = await self._check_interface_health(worker_id, interface)
                if result == ConnectionStatus.CONNECTED:
                    self.logger.info(f"Switched worker {worker_id} from {status.current_interface.name} to {interface.name}")
                    status.current_interface = interface
                    interface.last_success = datetime.now(timezone.utc)
                    interface.consecutive_failures = 0
                    status.last_successful_connection = datetime.now(timezone.utc)
                    
                    # Update URLs
                    status.health_check_url = f"http://{interface.ip}:8000/tasks"
                    status.api_url = f"http://{interface.ip}:8000"
                    
                    return True, ConnectionStatus.CONNECTED
                else:
                    interface.last_failure = datetime.now(timezone.utc)
                    interface.consecutive_failures += 1
        
        return False, ConnectionStatus.DISCONNECTED
    
    async def _check_interface_health(self, worker_id: str, interface: NetworkInterface) -> ConnectionStatus:
        """Check health of a specific network interface"""
        try:
            url = f"http://{interface.ip}:8000/tasks"
            
            if not self.session:
                await self.initialize()
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    return ConnectionStatus.CONNECTED
                else:
                    return ConnectionStatus.ERROR
                    
        except asyncio.TimeoutError:
            self.logger.debug(f"Health check timeout for {worker_id} on {interface.name}")
            return ConnectionStatus.TIMEOUT
        except aiohttp.ClientConnectorError:
            self.logger.debug(f"Connection failed for {worker_id} on {interface.name}")
            return ConnectionStatus.DISCONNECTED
        except Exception as e:
            self.logger.debug(f"Health check error for {worker_id} on {interface.name}: {str(e)}")
            return ConnectionStatus.ERROR
    
    def get_worker_api_url(self, worker_id: str) -> Optional[str]:
        """Get current API URL for worker"""
        if worker_id not in self.worker_network_status:
            return None
        
        status = self.worker_network_status[worker_id]
        return status.api_url
    
    def get_worker_ip(self, worker_id: str) -> Optional[str]:
        """Get current IP for worker"""
        if worker_id not in self.worker_network_status:
            return None
        
        status = self.worker_network_status[worker_id]
        if status.current_interface:
            return status.current_interface.ip
        return None
    
    async def send_request(self, worker_id: str, method: str, endpoint: str,
                          data: Optional[Dict] = None, timeout: float = 10.0) -> Tuple[bool, Any]:
        """Send HTTP request to worker with automatic retry on different interfaces"""
        if worker_id not in self.worker_network_status:
            return False, "Worker not registered"
        
        status = self.worker_network_status[worker_id]
        
        # Try current interface first
        if status.current_interface:
            success, response = await self._send_interface_request(
                status.current_interface, method, endpoint, data, timeout
            )
            if success:
                status.current_interface.last_success = datetime.now(timezone.utc)
                status.current_interface.consecutive_failures = 0
                status.last_successful_connection = datetime.now(timezone.utc)
                return True, response
            else:
                status.current_interface.last_failure = datetime.now(timezone.utc)
                status.current_interface.consecutive_failures += 1
        
        # Try alternative interfaces if current failed
        for interface in status.available_interfaces:
            if interface == status.current_interface:
                continue
            
            # Skip interfaces in cooldown
            if (interface.last_failure and 
                (datetime.now(timezone.utc) - interface.last_failure).seconds < self.interface_cooldown):
                continue
            
            success, response = await self._send_interface_request(
                interface, method, endpoint, data, timeout
            )
            if success:
                # Switch to this interface
                self.logger.info(f"Switched worker {worker_id} to {interface.name} for successful request")
                status.current_interface = interface
                interface.last_success = datetime.now(timezone.utc)
                interface.consecutive_failures = 0
                status.last_successful_connection = datetime.now(timezone.utc)
                
                # Update URLs
                status.health_check_url = f"http://{interface.ip}:8000/tasks"
                status.api_url = f"http://{interface.ip}:8000"
                
                return True, response
            else:
                interface.last_failure = datetime.now(timezone.utc)
                interface.consecutive_failures += 1
        
        return False, "All interfaces failed"
    
    async def _send_interface_request(self, interface: NetworkInterface, method: str,
                                    endpoint: str, data: Optional[Dict] = None,
                                    timeout: float = 10.0) -> Tuple[bool, Any]:
        """Send request to specific interface"""
        try:
            url = f"http://{interface.ip}:8000{endpoint}"

            if not self.session:
                await self.initialize()

            request_timeout = aiohttp.ClientTimeout(total=timeout)

            if method.upper() == 'GET':
                async with self.session.get(url, timeout=request_timeout) as response:
                    if response.status == 200:
                        result = await response.json()
                        return True, result
                    else:
                        error_text = await response.text()
                        return False, f"HTTP {response.status}: {error_text}"

            elif method.upper() == 'POST':
                async with self.session.post(url, json=data, timeout=request_timeout) as response:
                    if response.status == 200:
                        result = await response.json()
                        return True, result
                    else:
                        error_text = await response.text()
                        return False, f"HTTP {response.status}: {error_text}"

            else:
                return False, f"Unsupported method: {method}"

        except Exception as e:
            return False, str(e)
    
    def get_network_status(self, worker_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed network status for worker"""
        if worker_id not in self.worker_network_status:
            return None
        
        status = self.worker_network_status[worker_id]
        
        return {
            'worker_id': worker_id,
            'current_interface': {
                'name': status.current_interface.name,
                'ip': status.current_interface.ip,
                'last_success': status.current_interface.last_success,
                'consecutive_failures': status.current_interface.consecutive_failures
            } if status.current_interface else None,
            'available_interfaces': [
                {
                    'name': iface.name,
                    'ip': iface.ip,
                    'is_primary': iface.is_primary,
                    'last_success': iface.last_success,
                    'consecutive_failures': iface.consecutive_failures
                }
                for iface in status.available_interfaces
            ],
            'last_successful_connection': status.last_successful_connection,
            'api_url': status.api_url
        }
    
    async def perform_bulk_health_check(self) -> Dict[str, bool]:
        """Perform health check on all registered workers"""
        results = {}
        
        health_check_tasks = []
        worker_ids = list(self.worker_network_status.keys())
        
        for worker_id in worker_ids:
            task = asyncio.create_task(self.check_worker_health(worker_id))
            health_check_tasks.append((worker_id, task))
        
        for worker_id, task in health_check_tasks:
            try:
                healthy, status = await task
                results[worker_id] = healthy
            except Exception as e:
                self.logger.error(f"Health check failed for {worker_id}: {str(e)}")
                results[worker_id] = False
        
        return results