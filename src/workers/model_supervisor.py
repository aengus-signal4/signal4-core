#!/usr/bin/env python3
"""
Model Server Supervisor
=======================

Supervises the model server process with automatic restarts, health checks,
and proper resource cleanup to avoid semaphore leaks and fork issues on macOS.
"""

import os
import sys
import time
import signal
import subprocess
import logging
import psutil
import asyncio
import aiohttp
from pathlib import Path
from src.utils.paths import get_project_root, get_config_path
from datetime import datetime
import multiprocessing

# Set start method to spawn to avoid fork issues on macOS
multiprocessing.set_start_method('spawn', force=True)

# Suppress multiprocessing resource tracker warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing.resource_tracker")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_server_supervisor')


class ModelServerSupervisor:
    def __init__(self, config_path: str, worker_id: str, port: int = 8002):
        self.config_path = Path(config_path)
        self.worker_id = worker_id
        self.port = port
        self.process = None
        self.running = True
        self.restart_count = 0
        self.max_restart_attempts = 10
        self.restart_delay = 5  # seconds
        self.health_check_interval = 30  # seconds
        self.last_health_check = None
        self.consecutive_failures = 0
        self.max_consecutive_failures = 3
        
        # Get project root
        self.project_root = get_project_root()
        self.model_server_path = self.project_root / "src" / "api" / "model_server.py"
        
        # Set up signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGHUP, self._reload_handler)
        
        logger.info(f"Model Server Supervisor initialized for {worker_id} on port {port}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
        self.stop_model_server()
        sys.exit(0)
    
    def _reload_handler(self, signum, frame):
        """Handle reload signal (SIGHUP) to restart the model server."""
        logger.info("Received SIGHUP, restarting model server...")
        self.restart_model_server()
    
    def start_model_server(self):
        """Start the model server process."""
        try:
            # Kill any existing model server processes
            self._kill_existing_processes()
            
            # Set up environment
            env = os.environ.copy()
            env['PYTHONPATH'] = f"{self.project_root}:{env.get('PYTHONPATH', '')}"
            env['MODEL_MAX_CONCURRENT'] = '4'
            env['WORKER_ID'] = self.worker_id
            env['MODEL_SERVER_PORT'] = str(self.port)
            
            # Suppress resource tracker warnings
            env['PYTHONWARNINGS'] = 'ignore::UserWarning:multiprocessing.resource_tracker'
            
            # Set multiprocessing start method
            env['MULTIPROCESSING_START_METHOD'] = 'spawn'
            
            # Start the model server process
            cmd = [
                sys.executable,
                str(self.model_server_path)
            ]
            
            logger.info(f"Starting model server with command: {' '.join(cmd)}")
            
            self.process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True,
                preexec_fn=os.setsid if sys.platform != 'win32' else None
            )
            
            # Start log monitoring in a separate thread
            asyncio.create_task(self._monitor_logs())
            
            logger.info(f"Model server started with PID {self.process.pid}")
            self.restart_count = 0
            self.consecutive_failures = 0
            
            # Wait a bit for the server to start
            time.sleep(5)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start model server: {e}")
            return False
    
    def stop_model_server(self):
        """Stop the model server process gracefully."""
        if self.process and self.process.poll() is None:
            try:
                logger.info(f"Stopping model server (PID {self.process.pid})...")
                
                # Try graceful shutdown first
                self.process.terminate()
                
                # Wait up to 10 seconds for graceful shutdown
                try:
                    self.process.wait(timeout=10)
                    logger.info("Model server stopped gracefully")
                except subprocess.TimeoutExpired:
                    # Force kill if graceful shutdown fails
                    logger.warning("Graceful shutdown timed out, force killing...")
                    if sys.platform != 'win32':
                        os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                    else:
                        self.process.kill()
                    self.process.wait()
                    logger.info("Model server force killed")
                    
            except Exception as e:
                logger.error(f"Error stopping model server: {e}")
    
    def _kill_existing_processes(self):
        """Kill any existing model server processes."""
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = proc.info.get('cmdline', [])
                    if cmdline and 'model_server.py' in ' '.join(cmdline):
                        logger.info(f"Killing existing model server process (PID {proc.pid})")
                        proc.kill()
                        proc.wait(timeout=5)
                except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                    pass
        except Exception as e:
            logger.error(f"Error killing existing processes: {e}")
    
    async def _monitor_logs(self):
        """Monitor model server output logs."""
        if not self.process:
            return
            
        try:
            while self.process.poll() is None:
                line = self.process.stdout.readline()
                if line:
                    # Log the output (strip trailing newline)
                    logger.info(f"[MODEL_SERVER] {line.rstrip()}")
                await asyncio.sleep(0.1)
        except Exception as e:
            logger.error(f"Error monitoring logs: {e}")
    
    async def check_health(self):
        """Check if the model server is healthy."""
        try:
            url = f"http://localhost:{self.port}/health"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.debug(f"Health check passed: {data}")
                        self.consecutive_failures = 0
                        return True
                    else:
                        logger.warning(f"Health check failed with status {response.status}")
                        self.consecutive_failures += 1
                        return False
                        
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            self.consecutive_failures += 1
            return False
    
    def restart_model_server(self):
        """Restart the model server."""
        logger.info("Restarting model server...")
        self.stop_model_server()
        time.sleep(self.restart_delay)
        
        if self.restart_count >= self.max_restart_attempts:
            logger.error(f"Max restart attempts ({self.max_restart_attempts}) reached. Giving up.")
            self.running = False
            return False
            
        self.restart_count += 1
        return self.start_model_server()
    
    async def run(self):
        """Main supervisor loop."""
        logger.info("Starting model server supervisor...")
        
        # Initial start
        if not self.start_model_server():
            logger.error("Failed to start model server initially")
            return
        
        last_health_check = time.time()
        
        while self.running:
            try:
                # Check if process is still running
                if self.process and self.process.poll() is not None:
                    exit_code = self.process.returncode
                    logger.warning(f"Model server exited with code {exit_code}")
                    
                    if self.running:  # Only restart if we're still supposed to be running
                        if not self.restart_model_server():
                            break
                
                # Periodic health check
                current_time = time.time()
                if current_time - last_health_check >= self.health_check_interval:
                    last_health_check = current_time
                    
                    if not await self.check_health():
                        if self.consecutive_failures >= self.max_consecutive_failures:
                            logger.error(f"Health check failed {self.consecutive_failures} times, restarting...")
                            if not self.restart_model_server():
                                break
                
                await asyncio.sleep(1)
                
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Error in supervisor loop: {e}")
                await asyncio.sleep(5)
        
        # Clean shutdown
        self.stop_model_server()
        logger.info("Model server supervisor stopped")


async def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Model Server Supervisor')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--worker-id', type=str, required=True,
                        help='Worker ID (e.g., worker0)')
    parser.add_argument('--port', type=int, default=8002,
                        help='Model server port')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    supervisor = ModelServerSupervisor(
        config_path=args.config,
        worker_id=args.worker_id,
        port=args.port
    )
    
    await supervisor.run()


if __name__ == '__main__':
    asyncio.run(main())