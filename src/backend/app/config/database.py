"""
Database Configuration
======================

Centralized database configuration for the backend API.
All database connections should use these settings to ensure consistency
and proper credential management via environment variables.

Usage:
    from src.backend.app.config.database import get_db_config, get_database_url

    # For psycopg2
    config = get_db_config()
    conn = psycopg2.connect(**config)

    # For SQLAlchemy
    url = get_database_url()
    engine = create_engine(url)
"""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env file from core directory
_env_loaded = False


def _ensure_env_loaded():
    """Ensure .env file is loaded (once)."""
    global _env_loaded
    if not _env_loaded:
        env_path = Path(__file__).parents[4] / '.env'
        if env_path.exists():
            load_dotenv(env_path)
            logger.debug(f"Loaded environment from {env_path}")
        _env_loaded = True


def get_db_config() -> dict:
    """
    Get database configuration from environment variables.

    Returns:
        dict: Database configuration suitable for psycopg2.connect()

    Raises:
        ValueError: If POSTGRES_PASSWORD is not set
    """
    _ensure_env_loaded()

    password = os.getenv('POSTGRES_PASSWORD')
    if not password:
        raise ValueError(
            "POSTGRES_PASSWORD environment variable is required. "
            "Please set it in your .env file or environment."
        )

    return {
        'host': os.getenv('POSTGRES_HOST', '10.0.0.4'),
        'database': os.getenv('POSTGRES_DB', 'av_content'),
        'user': os.getenv('POSTGRES_USER', 'signal4'),
        'password': password,
        'port': int(os.getenv('POSTGRES_PORT', '5432')),
    }


def get_database_url() -> str:
    """
    Get SQLAlchemy-compatible database URL from environment variables.

    Returns:
        str: PostgreSQL connection URL

    Raises:
        ValueError: If POSTGRES_PASSWORD is not set
    """
    config = get_db_config()
    return (
        f"postgresql://{config['user']}:{config['password']}"
        f"@{config['host']}:{config['port']}/{config['database']}"
    )
