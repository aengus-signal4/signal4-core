#!/usr/bin/env python3
"""
Backend API Server Startup Script
==================================

Run the backend API server directly (no reload mode for PyTorch compatibility).
"""

import os
import sys
from pathlib import Path
from src.utils.paths import get_project_root, get_config_path

# Add src/ to path for imports
sys.path.insert(0, str(get_project_root()))

# Set PyTorch environment before any imports
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import uvicorn
from app.main import app, logger

if __name__ == "__main__":
    # Configure uvicorn logging to use our logger
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["loggers"]["uvicorn"]["level"] = "WARNING"
    log_config["loggers"]["uvicorn.error"]["level"] = "WARNING"
    log_config["loggers"]["uvicorn.access"]["level"] = "WARNING"

    logger.info("=" * 80)
    logger.info("Starting Backend API Server on http://0.0.0.0:7999")
    logger.info("=" * 80)

    # Run without reload to avoid process forking issues with PyTorch models
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7999,
        log_config=log_config
    )
