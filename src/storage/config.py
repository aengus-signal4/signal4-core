"""
Configuration settings for storage management.
"""
from pathlib import Path
import yaml

def get_storage_config():
    """Get storage configuration from config.yaml"""
    config_path = Path("config/config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    # Return just the S3 config section
    return config['storage']['s3']

def init_storage_config():
    """Initialize storage configuration"""
    return get_storage_config() 