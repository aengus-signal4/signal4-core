"""
Environment variable loading utilities.

DEPRECATED: Use src.utils.config instead:
    from src.utils.config import load_config, get_credential

    config = load_config()  # Config with env substitution
    token = get_credential('HF_TOKEN')  # Direct credential access

This module is kept for backward compatibility.
"""
from pathlib import Path
import os
import warnings
from dotenv import load_dotenv


def load_env():
    """Load environment variables from .env file.

    DEPRECATED: Use src.utils.config.load_config() instead, which
    automatically loads .env and substitutes ${VAR} in config.yaml.
    """
    warnings.warn(
        "load_env() is deprecated. Use src.utils.config.load_config() instead.",
        DeprecationWarning,
        stacklevel=2
    )

    # Find the .env file (in parent signal4/ directory)
    from .paths import get_env_path
    env_path = get_env_path()

    # Load environment variables from .env file
    load_dotenv(env_path)

    # Verify required variables are set
    required_vars = [
        'POSTGRES_DB',
        'POSTGRES_PORT',
        'POSTGRES_PASSWORD',
    ]

    # Special handling for POSTGRES_HOST
    db_host = os.getenv('POSTGRES_HOST')
    if not db_host or not db_host.startswith('10.0.0.'):
        # Get current hostname and check if we're on the head node
        import socket
        hostname = socket.gethostname()
        try:
            local_ip = socket.gethostbyname(hostname)
            # If we're on the head node IP, use localhost
            if local_ip == '10.0.0.209':
                os.environ['POSTGRES_HOST'] = 'localhost'
            else:
                os.environ['POSTGRES_HOST'] = '10.0.0.4'
        except socket.gaierror:
            # If hostname resolution fails, default to remote DB
            os.environ['POSTGRES_HOST'] = '10.0.0.4'

    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    return {var: os.getenv(var) for var in required_vars + ['POSTGRES_HOST']}
