"""
Backend Logger Configuration
=============================

Centralized logging for backend API using the worker logger system.
All logs go to logs/backend.log with proper formatting.
"""

import logging
import sys
from pathlib import Path
from src.utils.paths import get_project_root, get_config_path
from logging.handlers import RotatingFileHandler

# Log directory - use logs/backend/ for better organization
LOG_DIR = get_project_root() / 'logs' / "backend"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "backend.log"

def setup_backend_logger():
    """Setup centralized backend logger"""

    # Create logger
    logger = logging.getLogger("backend")
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    logger.handlers.clear()

    # File handler with rotation
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=50 * 1024 * 1024,  # 50MB
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - [BACKEND] - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger

# Global backend logger
backend_logger = setup_backend_logger()

def get_logger(name: str = "backend"):
    """Get a child logger for a specific module"""
    return backend_logger.getChild(name)
