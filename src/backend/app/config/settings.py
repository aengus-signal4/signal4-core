"""
Application Settings
====================

Loads configuration from environment variables.
"""

import os
from pathlib import Path
from src.utils.paths import get_project_root, get_config_path, get_env_path
from dotenv import load_dotenv

# Load .env file from parent signal4/ directory
load_dotenv(get_env_path())

# Database - use centralized config
def _get_database_url():
    """Get database URL from environment variables."""
    password = os.getenv('POSTGRES_PASSWORD')
    if not password:
        # For backwards compatibility, allow DATABASE_URL override
        return os.getenv('DATABASE_URL', '')

    return (
        f"postgresql://{os.getenv('POSTGRES_USER', 'signal4')}:{password}"
        f"@{os.getenv('POSTGRES_HOST', '10.0.0.4')}:{os.getenv('POSTGRES_PORT', '5432')}"
        f"/{os.getenv('POSTGRES_DB', 'av_content')}"
    )

DATABASE_URL = _get_database_url()

# API Keys
XAI_API_KEY = os.getenv('XAI_API_KEY', '')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', '')

# Local LLM Backend (LLM Balancer)
# URL for the local LLM balancer that routes to MLX model servers
LLM_BACKEND_URL = os.getenv('LLM_BACKEND_URL', 'http://10.0.0.4:8002')

# Server
HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', '8002'))
WORKERS = int(os.getenv('WORKERS', '1'))

# Paths
BACKEND_ROOT = get_project_root()
CACHE_DIR = Path(os.getenv('CACHE_DIR', BACKEND_ROOT / '.cache'))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Logging
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# Validate required settings
def validate_settings():
    """Validate that required settings are present"""
    errors = []

    if not XAI_API_KEY:
        errors.append("XAI_API_KEY not set")

    # GOOGLE_API_KEY is optional - only used for some features
    # if not GOOGLE_API_KEY:
    #     errors.append("GOOGLE_API_KEY not set")

    if errors:
        raise ValueError(f"Missing required settings: {', '.join(errors)}")

# Validate on import
validate_settings()
