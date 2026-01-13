"""
Application Settings
====================

Loads configuration from environment variables.
"""

import os
from pathlib import Path
from src.utils.paths import get_project_root, get_config_path
from dotenv import load_dotenv

# Load .env file
env_path = get_project_root() / '.env'
load_dotenv(env_path)

# Database
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://signal4@localhost/av_content')

# API Keys
XAI_API_KEY = os.getenv('XAI_API_KEY', '')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', '')

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
