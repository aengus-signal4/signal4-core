"""
Centralized environment setup for Signal4 processing scripts.
Import this FIRST in any script before other imports.
"""
import os
from typing import Optional

_env_setup_done = False


def setup_env(task_type: Optional[str] = None) -> None:
    """Configure environment variables for processing scripts."""
    global _env_setup_done
    if _env_setup_done:
        return

    # PATH: Add homebrew binaries (ffmpeg, ffprobe, yt-dlp)
    homebrew_bin = '/opt/homebrew/bin'
    if homebrew_bin not in os.environ.get('PATH', ''):
        os.environ['PATH'] = f"{homebrew_bin}:{os.environ.get('PATH', '')}"

    # DYLD: Add homebrew libraries (torchcodec, pyannote deps)
    homebrew_lib = '/opt/homebrew/lib'
    for dyld_var in ['DYLD_LIBRARY_PATH', 'DYLD_FALLBACK_LIBRARY_PATH']:
        if homebrew_lib not in os.environ.get(dyld_var, ''):
            current = os.environ.get(dyld_var, '')
            os.environ[dyld_var] = f"{homebrew_lib}:{current}" if current else homebrew_lib

    # macOS: Suppress malloc warnings
    os.environ['MallocStackLogging'] = '0'
    os.environ['MallocStackLoggingNoCompact'] = '0'

    # Task-specific settings
    if task_type in ('transcribe', 'stitch'):
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    _env_setup_done = True


def get_subprocess_env() -> dict:
    """Get environment dict for subprocess calls."""
    setup_env()
    return os.environ.copy()
