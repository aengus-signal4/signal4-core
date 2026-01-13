from contextlib import contextmanager
from ..database.session import get_session
from ..utils.logger import logger
import traceback
import re
from typing import Optional

@contextmanager
def db_session():
    """Context manager for database sessions"""
    with get_session() as session:
        try:
            yield session
            session.commit()
        except Exception as e:
            logger.error(f"Database error: {str(e)}")
            logger.error(f"Error details: {traceback.format_exc()}")
            session.rollback()
            raise

def get_db_url() -> str:
    """Get database URL from environment variables"""
    from .env import get_env
    return get_env('DATABASE_URL')

def standardize_source_name(name: str, platform: Optional[str] = None) -> str:
    """
    Standardize source names across platforms for consistent file paths.
    
    Args:
        name: The original source name (e.g., "Rebel News", "Daily Wire")
        platform: Optional platform identifier (e.g., 'youtube', 'podcast')
        
    Returns:
        str: Standardized name safe for filesystem use (e.g., "rebel-news")
    """
    # Remove any special characters except alphanumeric and spaces
    safe_name = re.sub(r'[^\w\s-]', '', name)
    
    # Replace spaces and multiple hyphens with single hyphen
    safe_name = re.sub(r'[-\s]+', '-', safe_name.strip())
    
    # Convert to lowercase
    safe_name = safe_name.lower()
    
    # Remove leading/trailing hyphens
    safe_name = safe_name.strip('-')
    
    return safe_name 