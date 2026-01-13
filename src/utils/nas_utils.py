#!/usr/bin/env python3
import subprocess
from pathlib import Path
import logging
from typing import Optional, Tuple
import os
import time

logger = logging.getLogger(__name__)

def get_mount_point() -> Path:
    """Get the NAS mount point from environment or default"""
    return Path(os.getenv('NAS_MOUNT_POINT', '/Users/signal4/mnt/s4-storage'))

def check_mount_status(mount_point: Optional[Path] = None) -> Tuple[bool, str]:
    """
    Check if NAS is properly mounted and accessible
    
    Returns:
        Tuple[bool, str]: (is_mounted, message)
    """
    if mount_point is None:
        mount_point = get_mount_point()
        
    try:
        # Check if mount point exists
        if not mount_point.exists():
            return False, f"Mount point {mount_point} does not exist"
            
        if not mount_point.is_dir():
            return False, f"Mount point {mount_point} is not a directory"
            
        # Check if it's actually mounted using df instead of mount
        result = subprocess.run(['df', str(mount_point)], capture_output=True, text=True)
        if result.returncode != 0 or '10.0.0.251' not in result.stdout:
            return False, f"No valid mount found at {mount_point}"
            
        # Simple read test - don't write test files
        try:
            next(mount_point.iterdir())
            return True, "NAS mount is healthy"
        except StopIteration:
            # Empty directory is still a valid mount
            return True, "NAS mount is healthy (empty directory)"
        except Exception as e:
            return False, f"Cannot read from mount point: {e}"
            
    except Exception as e:
        return False, f"Error checking mount status: {e}"

def store_credentials(password: str) -> bool:
    """
    Store NAS credentials in the system keychain
    """
    try:
        # First delete any existing password (ignore errors)
        subprocess.run([
            'security', 'delete-internet-password',
            '-s', '10.0.0.251'
        ], stderr=subprocess.DEVNULL)
        
        # Store new password
        cmd = [
            'security', 'add-internet-password',
            '-a', 'signal4',  # account name
            '-s', '10.0.0.251',  # server
            '-r', 'smb ',  # protocol (exactly 4 chars with space)
            '-l', 'NAS Storage',  # label
            '-w', password,  # password
            '-P', '445',  # port
            '-t', 'cifs'  # authentication type (4 chars)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Failed to store credentials: {result.stderr}")
            return False
        return True
    except Exception as e:
        logger.error(f"Failed to store credentials: {e}")
        return False

def get_credentials() -> Optional[str]:
    """
    Retrieve NAS password from system keychain
    """
    try:
        cmd = [
            'security', 'find-internet-password',
            '-a', 'signal4',
            '-s', '10.0.0.251',
            '-r', 'smb ',  # protocol (exactly 4 chars with space)
            '-w'  # display only password
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except Exception as e:
        logger.error(f"Failed to get credentials: {e}")
        return None

def mount_nas(mount_point: Optional[Path] = None) -> Tuple[bool, str]:
    """
    Mount the NAS if not already mounted
    
    Returns:
        Tuple[bool, str]: (success, message)
    """
    if mount_point is None:
        mount_point = get_mount_point()
        
    # First check if already mounted
    is_mounted, msg = check_mount_status(mount_point)
    if is_mounted:
        return True, msg
        
    try:
        # Ensure mount point exists
        mount_point.mkdir(parents=True, exist_ok=True)
        
        # Get password from keychain
        password = get_credentials()
        if not password:
            return False, "Could not retrieve NAS password from keychain"
        
        # Mount using stored credentials with macOS-compatible options
        cmd = [
            'mount_smbfs',
            '-o', 'nobrowse,soft,rw,nosuid,nodev',  # macOS-compatible options
            f'//signal4:{password}@10.0.0.251/s4-storage',  # Include password in URL
            str(mount_point)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            return False, f"Mount command failed: {result.stderr}"
            
        # Verify mount with retries
        for attempt in range(3):
            time.sleep(1)  # Brief pause between attempts
            is_mounted, check_msg = check_mount_status(mount_point)
            if is_mounted:
                return True, "NAS mounted successfully"
                
        return False, "Mount command succeeded but verification failed"
        
    except Exception as e:
        return False, f"Error mounting NAS: {e}"

def ensure_nas_mounted(mount_point: Optional[Path] = None) -> Tuple[bool, str]:
    """
    Ensure NAS is mounted, mounting it if necessary
    
    Returns:
        Tuple[bool, str]: (success, message)
    """
    if mount_point is None:
        mount_point = get_mount_point()
        
    # Check current status
    is_mounted, msg = check_mount_status(mount_point)
    if is_mounted:
        return True, msg
        
    # Not mounted, try to mount it
    return mount_nas(mount_point)

def get_nas_path(relative_path: str, mount_point: Optional[Path] = None) -> Path:
    """
    Get absolute path on NAS for a relative path
    
    Args:
        relative_path: Path relative to NAS root
        mount_point: Optional custom mount point
        
    Returns:
        Path: Absolute path on NAS
    """
    if mount_point is None:
        mount_point = get_mount_point()
    return mount_point / relative_path 