#!/usr/bin/env python3
"""
Backend API Server Startup Script
==================================

Run the backend API server directly (no reload mode for PyTorch compatibility).
"""

import os
import signal
import subprocess
import sys
from pathlib import Path
from src.utils.paths import get_project_root, get_config_path

# Add src/ to path for imports
sys.path.insert(0, str(get_project_root()))

# Set PyTorch environment before any imports
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import uvicorn
from src.backend.app.main import app, logger

PORT = 7999


def kill_process_on_port(port: int) -> None:
    """Kill any process currently using the specified port."""
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            text=True
        )
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                try:
                    os.kill(int(pid), signal.SIGKILL)
                    logger.info(f"Killed process {pid} on port {port}")
                except (ProcessLookupError, ValueError):
                    pass
    except Exception as e:
        logger.warning(f"Could not check/kill process on port {port}: {e}")


if __name__ == "__main__":
    kill_process_on_port(PORT)
    # Configure uvicorn logging to use our logger
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["loggers"]["uvicorn"]["level"] = "WARNING"
    log_config["loggers"]["uvicorn.error"]["level"] = "WARNING"
    log_config["loggers"]["uvicorn.access"]["level"] = "WARNING"

    logger.info("=" * 80)
    logger.info(f"Starting Backend API Server on http://0.0.0.0:{PORT}")
    logger.info("=" * 80)

    # Run without reload to avoid process forking issues with PyTorch models
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        log_config=log_config
    )
