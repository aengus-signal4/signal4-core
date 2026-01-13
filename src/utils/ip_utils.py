"""
IP and Network Utilities
========================

Provides centralized IP address resolution, network interface detection,
and fallback logic for ethernet to WiFi failover.

Key features:
- Detect local ethernet and WiFi interfaces
- Resolve worker IPs from configuration
- Test network connectivity with timeouts
- Automatic fallback from eth to wifi
- Request routing based on source interface
"""

import socket
import subprocess
import re
import asyncio
import time
from typing import Dict, Optional, Tuple, List, Any
from pathlib import Path
import logging
import aiohttp
import requests

from src.utils.config import load_config

# Use standard logging to avoid circular import
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class NetworkInterface:
    """Represents a network interface with its IP and type."""
    
    def __init__(self, name: str, ip: str, interface_type: str = 'unknown'):
        self.name = name
        self.ip = ip
        self.type = interface_type  # 'ethernet', 'wifi', 'unknown'
        
    def __repr__(self):
        return f"NetworkInterface(name={self.name}, ip={self.ip}, type={self.type})"


class WorkerNetworkInfo:
    """Stores network information for a worker."""
    
    def __init__(self, worker_name: str, eth_ip: Optional[str] = None, 
                 wifi_ip: Optional[str] = None, enabled: bool = True):
        self.worker_name = worker_name
        self.eth_ip = eth_ip
        self.wifi_ip = wifi_ip
        self.enabled = enabled
        self._last_working_ip = None
        self._last_check_time = None
        
    @property
    def primary_ip(self) -> Optional[str]:
        """Get primary IP (prefer eth over wifi)."""
        return self.eth_ip or self.wifi_ip
        
    @property
    def fallback_ip(self) -> Optional[str]:
        """Get fallback IP."""
        if self.eth_ip and self.wifi_ip:
            return self.wifi_ip
        return None
        
    def get_ips(self) -> List[str]:
        """Get all available IPs in priority order."""
        ips = []
        if self.eth_ip:
            ips.append(self.eth_ip)
        if self.wifi_ip and self.wifi_ip != self.eth_ip:
            ips.append(self.wifi_ip)
        return ips
        
    def __repr__(self):
        return f"WorkerNetworkInfo(name={self.worker_name}, eth={self.eth_ip}, wifi={self.wifi_ip})"


class IPUtils:
    """Central utility class for IP and network operations."""
    
    def __init__(self):
        self._config = None
        self._workers_cache = {}
        self._local_interfaces = None
        self._last_config_load = 0
        self._config_ttl = 300  # Reload config every 5 minutes
        
    def _load_config_if_needed(self):
        """Load or reload configuration if needed."""
        current_time = time.time()
        if (self._config is None or 
            current_time - self._last_config_load > self._config_ttl):
            try:
                self._config = load_config()
                self._last_config_load = current_time
                self._workers_cache = {}  # Clear cache on reload
                logger.debug("Loaded/reloaded configuration")
            except Exception as e:
                logger.error(f"Failed to load configuration: {e}")
                if self._config is None:
                    raise
                    
    def get_local_interfaces(self, refresh: bool = False) -> List[NetworkInterface]:
        """Get all local network interfaces with their IPs."""
        if self._local_interfaces is not None and not refresh:
            return self._local_interfaces
            
        interfaces = []
        
        try:
            # Use ifconfig to get interface information
            result = subprocess.run(['ifconfig'], capture_output=True, text=True)
            if result.returncode == 0:
                # Parse ifconfig output
                current_interface = None
                for line in result.stdout.split('\n'):
                    # Check for interface name (starts at beginning of line)
                    if line and not line.startswith((' ', '\t')):
                        current_interface = line.split(':')[0].strip()
                    # Check for IPv4 address
                    elif current_interface and 'inet ' in line:
                        match = re.search(r'inet\s+(\d+\.\d+\.\d+\.\d+)', line)
                        if match:
                            ip = match.group(1)
                            # Determine interface type
                            if 'en0' in current_interface:
                                iface_type = 'wifi'  # en0 is typically WiFi on macOS
                            elif 'en' in current_interface:
                                iface_type = 'ethernet'
                            elif 'lo' in current_interface:
                                iface_type = 'loopback'
                            else:
                                iface_type = 'unknown'
                                
                            if ip != '127.0.0.1':  # Skip loopback
                                interfaces.append(NetworkInterface(current_interface, ip, iface_type))
                                
        except Exception as e:
            logger.error(f"Failed to get network interfaces: {e}")
            
        # Fallback to socket method
        if not interfaces:
            try:
                hostname = socket.gethostname()
                _, _, ips = socket.gethostbyname_ex(hostname)
                for ip in ips:
                    if ip != '127.0.0.1':
                        interfaces.append(NetworkInterface('unknown', ip, 'unknown'))
            except Exception as e:
                logger.error(f"Failed to get IPs via socket: {e}")
                
        self._local_interfaces = interfaces
        return interfaces
        
    def get_worker_ip(self) -> str:
        """Get the primary IP address of this worker (backward compatibility)."""
        interfaces = self.get_local_interfaces()
        
        # Prefer 10.0.0.x addresses
        for iface in interfaces:
            if iface.ip.startswith('10.0.0.'):
                return iface.ip
                
        # Return first non-loopback IP
        if interfaces:
            return interfaces[0].ip
            
        # Last resort
        return socket.gethostname()
        
    def get_worker_info(self, worker_name: str) -> Optional[WorkerNetworkInfo]:
        """Get network information for a specific worker."""
        self._load_config_if_needed()
        
        # Check cache
        if worker_name in self._workers_cache:
            return self._workers_cache[worker_name]
            
        # Load from config
        workers = self._config.get('processing', {}).get('workers', {})
        if worker_name not in workers:
            # Check if it's the head node
            if worker_name == 'head_node':
                head_ip = self._config.get('network', {}).get('head_node_ip')
                if head_ip:
                    info = WorkerNetworkInfo('head_node', eth_ip=head_ip)
                    self._workers_cache[worker_name] = info
                    return info
            return None
            
        worker_config = workers[worker_name]
        info = WorkerNetworkInfo(
            worker_name=worker_name,
            eth_ip=worker_config.get('eth'),
            wifi_ip=worker_config.get('wifi'),
            enabled=worker_config.get('enabled', True)
        )
        
        self._workers_cache[worker_name] = info
        return info
        
    def get_all_workers(self) -> Dict[str, WorkerNetworkInfo]:
        """Get network information for all workers."""
        self._load_config_if_needed()
        
        workers = {}
        worker_configs = self._config.get('processing', {}).get('workers', {})
        
        for worker_name, worker_config in worker_configs.items():
            if worker_config.get('enabled', True):
                info = self.get_worker_info(worker_name)
                if info:
                    workers[worker_name] = info
                    
        return workers
        
    def test_connectivity(self, ip: str, port: int, timeout: float = 2.0) -> bool:
        """Test if an IP:port is reachable (synchronous)."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((ip, port))
            sock.close()
            return result == 0
        except Exception as e:
            logger.debug(f"Connectivity test failed for {ip}:{port} - {e}")
            return False
            
    async def test_connectivity_async(self, ip: str, port: int, timeout: float = 2.0) -> bool:
        """Test if an IP:port is reachable (asynchronous)."""
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(ip, port),
                timeout=timeout
            )
            writer.close()
            await writer.wait_closed()
            return True
        except Exception as e:
            logger.debug(f"Async connectivity test failed for {ip}:{port} - {e}")
            return False
            
    def get_reachable_ip(self, worker_name: str, port: int, timeout: float = 2.0) -> Optional[str]:
        """Get the first reachable IP for a worker (synchronous)."""
        info = self.get_worker_info(worker_name)
        if not info:
            return None
            
        for ip in info.get_ips():
            if self.test_connectivity(ip, port, timeout):
                logger.debug(f"Worker {worker_name} reachable at {ip}:{port}")
                return ip
                
        logger.warning(f"No reachable IP found for worker {worker_name} on port {port}")
        return None
        
    async def get_reachable_ip_async(self, worker_name: str, port: int, 
                                     timeout: float = 2.0) -> Optional[str]:
        """Get the first reachable IP for a worker (asynchronous)."""
        info = self.get_worker_info(worker_name)
        if not info:
            return None
            
        for ip in info.get_ips():
            if await self.test_connectivity_async(ip, port, timeout):
                logger.debug(f"Worker {worker_name} reachable at {ip}:{port}")
                return ip
                
        logger.warning(f"No reachable IP found for worker {worker_name} on port {port}")
        return None
        
    def build_worker_url(self, worker_name: str, port: int, path: str = "", 
                        timeout: float = 2.0) -> Optional[str]:
        """Build a URL for a worker, testing connectivity first."""
        ip = self.get_reachable_ip(worker_name, port, timeout)
        if ip:
            return f"http://{ip}:{port}{path}"
        return None
        
    async def build_worker_url_async(self, worker_name: str, port: int, path: str = "", 
                                    timeout: float = 2.0) -> Optional[str]:
        """Build a URL for a worker, testing connectivity first (async)."""
        ip = await self.get_reachable_ip_async(worker_name, port, timeout)
        if ip:
            return f"http://{ip}:{port}{path}"
        return None
        
    def get_request_interface(self, request_ip: str) -> Optional[str]:
        """Determine which local interface received a request."""
        interfaces = self.get_local_interfaces()
        
        # Direct match
        for iface in interfaces:
            if iface.ip == request_ip:
                return iface.name
                
        # Check if request came through a known worker's IP
        for worker_info in self.get_all_workers().values():
            if request_ip in [worker_info.eth_ip, worker_info.wifi_ip]:
                # Find which of our interfaces is on the same subnet
                for iface in interfaces:
                    if self._same_subnet(iface.ip, request_ip):
                        return iface.name
                        
        return None
        
    def _same_subnet(self, ip1: str, ip2: str, mask: str = "255.255.255.0") -> bool:
        """Check if two IPs are on the same subnet."""
        try:
            # Simple implementation for /24 networks
            parts1 = ip1.split('.')
            parts2 = ip2.split('.')
            return parts1[:3] == parts2[:3]
        except:
            return False
            
    async def make_request_with_fallback(self, worker_name: str, port: int, 
                                        method: str, path: str,
                                        json_data: Optional[Dict] = None,
                                        headers: Optional[Dict] = None,
                                        timeout: float = 30.0) -> Optional[aiohttp.ClientResponse]:
        """Make an HTTP request to a worker with automatic fallback."""
        info = self.get_worker_info(worker_name)
        if not info:
            logger.error(f"Worker {worker_name} not found in configuration")
            return None
            
        last_error = None
        for ip in info.get_ips():
            url = f"http://{ip}:{port}{path}"
            try:
                logger.debug(f"Trying request to {url}")
                async with aiohttp.ClientSession() as session:
                    async with session.request(
                        method=method,
                        url=url,
                        json=json_data,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=timeout)
                    ) as response:
                        if response.status < 500:  # Don't retry on client errors
                            return response
                        last_error = f"HTTP {response.status}"
                        
            except Exception as e:
                last_error = str(e)
                logger.debug(f"Request to {url} failed: {last_error}")
                continue
                
        logger.error(f"All IPs failed for worker {worker_name}: {last_error}")
        return None


# Global instance
ip_utils = IPUtils()


# Convenience functions for backward compatibility and ease of use
def get_worker_ip() -> str:
    """Get the primary IP address of this worker."""
    return ip_utils.get_worker_ip()


def get_worker_info(worker_name: str) -> Optional[WorkerNetworkInfo]:
    """Get network information for a specific worker."""
    return ip_utils.get_worker_info(worker_name)


def test_connectivity(ip: str, port: int, timeout: float = 2.0) -> bool:
    """Test if an IP:port is reachable."""
    return ip_utils.test_connectivity(ip, port, timeout)


async def test_connectivity_async(ip: str, port: int, timeout: float = 2.0) -> bool:
    """Test if an IP:port is reachable (async)."""
    return await ip_utils.test_connectivity_async(ip, port, timeout)


def get_reachable_ip(worker_name: str, port: int, timeout: float = 2.0) -> Optional[str]:
    """Get the first reachable IP for a worker."""
    return ip_utils.get_reachable_ip(worker_name, port, timeout)


async def get_reachable_ip_async(worker_name: str, port: int, timeout: float = 2.0) -> Optional[str]:
    """Get the first reachable IP for a worker (async)."""
    return await ip_utils.get_reachable_ip_async(worker_name, port, timeout)


def build_worker_url(worker_name: str, port: int, path: str = "", timeout: float = 2.0) -> Optional[str]:
    """Build a URL for a worker, testing connectivity first."""
    return ip_utils.build_worker_url(worker_name, port, path, timeout)


async def build_worker_url_async(worker_name: str, port: int, path: str = "", 
                                timeout: float = 2.0) -> Optional[str]:
    """Build a URL for a worker, testing connectivity first (async)."""
    return await ip_utils.build_worker_url_async(worker_name, port, path, timeout)