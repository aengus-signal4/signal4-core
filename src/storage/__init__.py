"""
Storage management initialization.
"""
from .s3_utils import S3Storage, S3StorageConfig
from .content_storage import ContentStorageManager
from .config import get_storage_config, init_storage_config

def init_storage() -> ContentStorageManager:
    """Initialize storage system"""
    config = init_storage_config()
    
    # Initialize S3 storage
    s3_config = S3StorageConfig(
        endpoint_url=config['s3']['endpoint_url'],
        access_key=config['s3']['access_key'],
        secret_key=config['s3']['secret_key'],
        bucket_name=config['s3']['bucket_name'],
        use_ssl=config['s3']['use_ssl']
    )
    s3_storage = S3Storage(s3_config)
    
    # Initialize content storage manager
    return ContentStorageManager(s3_storage)

def create_s3_storage(config):
    """Create S3 storage instance from config"""
    s3_config = S3StorageConfig(
        endpoint_url=config['s3']['endpoint_url'],
        access_key=config['s3']['access_key'],
        secret_key=config['s3']['secret_key'],
        bucket_name=config['s3']['bucket_name'],
        use_ssl=config['s3']['use_ssl']
    )
    return S3Storage(s3_config)

__all__ = ['init_storage', 'S3Storage', 'S3StorageConfig', 
           'ContentStorageManager', 'get_storage_config'] 