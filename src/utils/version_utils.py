# src/utils/version_utils.py

import re
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

def parse_stitch_version(version_str: str) -> Tuple[str, Optional[str]]:
    """
    Parse a stitch version string into major and minor components.
    
    Args:
        version_str: Version string like "stitch_v13", "stitch_v13.1", "stitch_v13.2"
        
    Returns:
        Tuple of (major_version, minor_version)
        - major_version: "stitch_v13" 
        - minor_version: "1" or None if no minor version
        
    Examples:
        parse_stitch_version("stitch_v13") → ("stitch_v13", None)
        parse_stitch_version("stitch_v13.1") → ("stitch_v13", "1")
        parse_stitch_version("stitch_v13.2") → ("stitch_v13", "2")
    """
    if not version_str:
        return ("stitch_v1", None)
    
    # Pattern to match stitch_vX or stitch_vX.Y
    pattern = r'^(stitch_v\d+)(?:\.(\d+))?$'
    match = re.match(pattern, version_str)
    
    if not match:
        # For backward compatibility, treat unknown formats as major versions
        logger.warning(f"Unknown stitch version format: {version_str}, treating as major version")
        return (version_str, None)
    
    major_version = match.group(1)
    minor_version = match.group(2)
    
    return (major_version, minor_version)

def get_major_version(version_str: str) -> str:
    """
    Get the major version from a stitch version string.
    
    Args:
        version_str: Version string like "stitch_v13.1"
        
    Returns:
        Major version string like "stitch_v13"
        
    Examples:
        get_major_version("stitch_v13.1") → "stitch_v13"
        get_major_version("stitch_v13") → "stitch_v13"
    """
    major_version, _ = parse_stitch_version(version_str)
    return major_version

def is_stitch_version_compatible(current_version: str, content_version: str) -> bool:
    """
    Check if two stitch versions are compatible.
    
    Versions are compatible if they have the same major version.
    This allows sub-versions (e.g., stitch_v13.1) to be compatible with
    the base version (stitch_v13) and other sub-versions (stitch_v13.2).
    
    Args:
        current_version: Current stitch version from config
        content_version: Version stored in content record
        
    Returns:
        True if versions are compatible, False otherwise
        
    Examples:
        is_stitch_version_compatible("stitch_v13.1", "stitch_v13") → True
        is_stitch_version_compatible("stitch_v13", "stitch_v13.1") → True
        is_stitch_version_compatible("stitch_v13.1", "stitch_v13.2") → True
        is_stitch_version_compatible("stitch_v13", "stitch_v14") → False
    """
    if not current_version or not content_version:
        return False
    
    # Handle exact matches quickly
    if current_version == content_version:
        return True
    
    # Parse both versions
    current_major, _ = parse_stitch_version(current_version)
    content_major, _ = parse_stitch_version(content_version)
    
    # Compatible if major versions match
    return current_major == content_major

def should_recreate_stitch_task(current_version: str, content_version: str) -> bool:
    """
    Determine if a stitch task should be recreated based on version compatibility.
    
    This is the main function that should be used to replace direct version
    equality checks in the pipeline manager and task creation logic.
    
    Args:
        current_version: Current stitch version from config
        content_version: Version stored in content record
        
    Returns:
        True if task should be recreated, False if compatible
        
    Examples:
        should_recreate_stitch_task("stitch_v13.1", "stitch_v13") → False
        should_recreate_stitch_task("stitch_v13", "stitch_v14") → True
        should_recreate_stitch_task("stitch_v13.1", None) → True
    """
    # If content has no version, it needs to be stitched
    if not content_version:
        return True
    
    # If current version is not set, use compatibility check
    if not current_version:
        return False
    
    # Use compatibility check - recreate if NOT compatible
    return not is_stitch_version_compatible(current_version, content_version)

def format_version_comparison_log(current_version: str, content_version: str) -> str:
    """
    Format a log message for version comparison debugging.
    
    Args:
        current_version: Current stitch version from config
        content_version: Version stored in content record
        
    Returns:
        Formatted log message string
    """
    compatible = is_stitch_version_compatible(current_version, content_version)
    current_major = get_major_version(current_version)
    content_major = get_major_version(content_version) if content_version else "None"
    
    return (f"Version comparison: current='{current_version}' ({current_major}) vs "
            f"content='{content_version}' ({content_major}) → "
            f"{'COMPATIBLE' if compatible else 'INCOMPATIBLE'}")