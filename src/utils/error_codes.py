"""
Error code definitions for the content processing pipeline.

This module defines standardized error codes and result structures
used across all processing steps for consistent error handling.
"""

from enum import Enum
from typing import Dict, Any, Optional


class ErrorCode(Enum):
    """Standardized error codes for the processing pipeline."""
    
    # Transient errors (should retry)
    NETWORK_ERROR = "network_error"
    S3_CONNECTION_ERROR = "s3_connection_error"
    TIMEOUT = "timeout"
    TEMPORARY_FAILURE = "temporary_failure"
    RATE_LIMITED = "rate_limited"  # HTTP 429 Too Many Requests
    
    # Permanent errors (should not retry)
    NOT_FOUND = "not_found"
    ACCESS_DENIED = "access_denied"
    INVALID_FORMAT = "invalid_format"
    CORRUPT_MEDIA = "corrupt_media"
    UNSUPPORTED_FORMAT = "unsupported_format"
    SSL_ERROR = "ssl_error"
    
    # Authentication errors (special handling)
    AUTH_REQUIRED = "auth_required"
    AUTH_FAILED = "auth_failed"
    YOUTUBE_AUTH = "youtube_auth"
    
    # Missing dependencies (trigger prerequisite creation)
    MISSING_AUDIO = "missing_audio"
    MISSING_TRANSCRIPT = "missing_transcript"
    MISSING_DIARIZATION = "missing_diarization"
    MISSING_CHUNKS = "missing_chunks"
    MISSING_SOURCE = "missing_source"
    
    # Content restrictions (block content)
    AGE_RESTRICTED = "age_restricted"
    MEMBERS_ONLY = "members_only"
    PRIVATE_CONTENT = "private_content"
    BAD_URL = "bad_url"
    VIDEO_UNAVAILABLE = "video_unavailable"
    LIVE_STREAM = "live_stream"
    CONTENT_GONE = "content_gone"  # HTTP 410 - permanently deleted
    FEED_DISABLED = "feed_disabled"  # HTTP 400 - RSS/podcast feed disabled
    
    # Processing results (special handling)
    EMPTY_RESULT = "empty_result"
    NO_SPEECH_DETECTED = "no_speech_detected"
    ALREADY_EXISTS = "already_exists"
    
    # System errors
    OUT_OF_MEMORY = "out_of_memory"
    DISK_FULL = "disk_full"
    PROCESS_FAILED = "process_failed"
    UNKNOWN_ERROR = "unknown_error"


def create_error_result(
    error_code: ErrorCode,
    error_message: str,
    error_details: Optional[Dict[str, Any]] = None,
    permanent: bool = False,
    skip_state_audit: bool = False
) -> Dict[str, Any]:
    """
    Create a standardized error result dictionary.
    
    Args:
        error_code: The ErrorCode enum value
        error_message: Human-readable error message
        error_details: Optional additional context
        permanent: Whether this is a permanent failure
        skip_state_audit: Whether to skip state reconciliation
        
    Returns:
        Standardized error result dictionary
    """
    return {
        'status': 'failed',
        'error_code': error_code.value,
        'error': error_message,  # Keep 'error' for backward compatibility
        'error_message': error_message,
        'error_details': error_details or {},
        'permanent': permanent,
        'skip_state_audit': skip_state_audit
    }


def create_success_result(
    data: Optional[Dict[str, Any]] = None,
    message: Optional[str] = None,
    skip_wait_time: bool = False
) -> Dict[str, Any]:
    """
    Create a standardized success result dictionary.
    
    Args:
        data: Optional task-specific return data
        message: Optional success message
        skip_wait_time: For download tasks, whether to skip behavior wait
        
    Returns:
        Standardized success result dictionary
    """
    result = {
        'status': 'completed',
        'data': data or {}
    }
    
    if message:
        result['message'] = message
        
    if skip_wait_time:
        result['skip_wait_time'] = skip_wait_time
        
    return result


def create_skipped_result(
    reason: str,
    skip_wait_time: bool = False,
    data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a standardized skipped result dictionary.
    
    Args:
        reason: Why the task was skipped
        skip_wait_time: For download tasks, whether to skip behavior wait
        data: Optional additional data
        
    Returns:
        Standardized skipped result dictionary
    """
    result = {
        'status': 'skipped',
        'reason': reason,
        'data': data or {}
    }
    
    if skip_wait_time:
        result['skip_wait_time'] = skip_wait_time
        result['skipped_existing'] = True  # For backward compatibility
        
    return result


# Error categories for policy handling
ERROR_CATEGORIES = {
    'transient': [
        ErrorCode.NETWORK_ERROR,
        ErrorCode.S3_CONNECTION_ERROR,
        ErrorCode.TIMEOUT,
        ErrorCode.TEMPORARY_FAILURE,
        ErrorCode.RATE_LIMITED
    ],
    'permanent': [
        ErrorCode.NOT_FOUND,
        ErrorCode.ACCESS_DENIED,
        ErrorCode.INVALID_FORMAT,
        ErrorCode.CORRUPT_MEDIA,
        ErrorCode.UNSUPPORTED_FORMAT,
        ErrorCode.SSL_ERROR
    ],
    'auth': [
        ErrorCode.AUTH_REQUIRED,
        ErrorCode.AUTH_FAILED,
        ErrorCode.YOUTUBE_AUTH
    ],
    'missing_deps': [
        ErrorCode.MISSING_AUDIO,
        ErrorCode.MISSING_TRANSCRIPT,
        ErrorCode.MISSING_DIARIZATION,
        ErrorCode.MISSING_CHUNKS,
        ErrorCode.MISSING_SOURCE
    ],
    'content_blocked': [
        ErrorCode.AGE_RESTRICTED,
        ErrorCode.MEMBERS_ONLY,
        ErrorCode.PRIVATE_CONTENT,
        ErrorCode.BAD_URL,
        ErrorCode.VIDEO_UNAVAILABLE,
        ErrorCode.LIVE_STREAM,
        ErrorCode.CONTENT_GONE,
        ErrorCode.FEED_DISABLED
    ],
    'special': [
        ErrorCode.EMPTY_RESULT,
        ErrorCode.NO_SPEECH_DETECTED,
        ErrorCode.ALREADY_EXISTS
    ],
    'system': [
        ErrorCode.OUT_OF_MEMORY,
        ErrorCode.DISK_FULL,
        ErrorCode.PROCESS_FAILED,
        ErrorCode.UNKNOWN_ERROR
    ]
}


def get_error_category(error_code: ErrorCode) -> Optional[str]:
    """Get the category for an error code."""
    for category, codes in ERROR_CATEGORIES.items():
        if error_code in codes:
            return category
    return None