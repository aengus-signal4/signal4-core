from functools import wraps
import time
from typing import Callable, Any, Optional
from ..utils.logger import logger

class LLMError(Exception):
    """Base class for LLM-related errors"""
    pass

class LLMTimeoutError(LLMError):
    """Raised when LLM call times out"""
    pass

class LLMInvalidResponseError(LLMError):
    """Raised when LLM response is invalid"""
    pass

def retry_llm_call(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 10.0,
    backoff_factor: float = 2.0,
    timeout: Optional[float] = 30.0
) -> Callable:
    """Decorator for retrying LLM calls with exponential backoff
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        backoff_factor: Factor to multiply delay by after each retry
        timeout: Maximum time to wait for response in seconds
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    start_time = time.time()
                    result = func(*args, **kwargs)
                    
                    # Check for timeout
                    if timeout and time.time() - start_time > timeout:
                        raise LLMTimeoutError(f"LLM call timed out after {timeout} seconds")
                    
                    # Only validate string responses, allow other return types (like lists)
                    if isinstance(result, str):
                        if not result:
                            raise LLMInvalidResponseError("Invalid or empty response from LLM")
                    
                    return result
                    
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"LLM call failed (attempt {attempt + 1}/{max_retries + 1}): {str(e)}"
                            f"\nRetrying in {delay} seconds..."
                        )
                        time.sleep(delay)
                        delay = min(delay * backoff_factor, max_delay)
                    else:
                        logger.error(f"LLM call failed after {max_retries + 1} attempts: {str(e)}")
                        raise
            
            raise last_exception
        
        return wrapper
    return decorator

def validate_llm_response(response: str, expected_format: str = "numeric") -> bool:
    """Validate LLM response format
    
    Args:
        response: Response string from LLM
        expected_format: Expected format ("numeric" or "text")
    
    Returns:
        bool: True if response is valid, False otherwise
    """
    if not response or not isinstance(response, str):
        logger.warning(f"Invalid response type or empty response: {type(response)}")
        return False
    
    response = response.strip()
    logger.debug(f"Validating response: '{response}' (format: {expected_format})")
    
    if expected_format == "numeric":
        # Check if response is "0" or comma-separated numbers
        if response == "0":
            return True
        
        try:
            # Split by comma and strip whitespace
            parts = [part.strip() for part in response.split(",")]
            # Try to convert each part to an integer
            numbers = [int(part) for part in parts if part]  # Skip empty parts
            # Check if we have at least one number and all are valid integers
            return len(numbers) > 0 and all(isinstance(n, int) for n in numbers)
        except ValueError as e:
            logger.warning(f"Failed to parse numeric response: {response}, error: {str(e)}")
            return False
    
    return True  # For text format, any non-empty string is valid

def sanitize_llm_response(response: str) -> str:
    """Clean and sanitize LLM response
    
    Args:
        response: Raw response from LLM
    
    Returns:
        str: Cleaned response
    """
    if not response:
        logger.warning("Empty response received in sanitize_llm_response")
        return ""
    
    logger.debug(f"Sanitizing raw response: '{response}'")
    
    # Remove any markdown formatting
    response = response.replace("```", "").replace("`", "")
    
    # Remove multiple newlines and spaces
    response = " ".join(response.split())
    
    # Remove any non-numeric characters except commas for numeric responses
    response = "".join(c for c in response if c.isdigit() or c == ",")
    
    cleaned = response.strip()
    logger.debug(f"Sanitized response: '{cleaned}'")
    
    return cleaned 