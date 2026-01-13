"""
Content ID Generation Utility
=============================

Generates deterministic, platform-prefixed content IDs for new content.
Provides utilities for working with both old-format and new-format IDs.

Format:
- YouTube: yt_{video_id} (e.g., yt_dQw4w9WgXcQ)
- Podcast: pod_{sha256(feed_url:guid)[:12]} (e.g., pod_a1b2c3d4e5f6)
- Rumble: rmb_{video_id} (e.g., rmb_v12345a)

Old format (pre-migration):
- YouTube: {video_id} directly
- Podcast: {sanitized_guid_or_title}
- Rumble: {video_id} directly
"""
import hashlib
import re
from typing import Optional

# Platform prefixes
PLATFORM_PREFIXES = {
    'youtube': 'yt_',
    'podcast': 'pod_',
    'rumble': 'rmb_',
}

# Regex pattern for detecting new format
NEW_FORMAT_PATTERN = re.compile(r'^(yt|pod|rmb)_[a-zA-Z0-9_-]+$')


def generate_content_id(platform: str, **kwargs) -> str:
    """
    Generate a deterministic content ID for new content.

    Args:
        platform: One of 'youtube', 'podcast', 'rumble'
        **kwargs: Platform-specific identifiers
            - youtube: video_id (str)
            - podcast: feed_url (str), episode_guid (str)
            - rumble: video_id (str)

    Returns:
        Platform-prefixed content ID

    Raises:
        ValueError: If platform is unknown or required kwargs are missing
    """
    platform_lower = platform.lower()
    prefix = PLATFORM_PREFIXES.get(platform_lower)
    if not prefix:
        raise ValueError(f"Unknown platform: {platform}")

    if platform_lower == 'youtube':
        video_id = kwargs.get('video_id')
        if not video_id:
            raise ValueError("video_id required for YouTube content")
        return f"{prefix}{video_id}"

    elif platform_lower == 'podcast':
        feed_url = kwargs.get('feed_url')
        episode_guid = kwargs.get('episode_guid')
        if not feed_url or not episode_guid:
            raise ValueError("feed_url and episode_guid required for podcast content")
        hash_input = f"{feed_url}:{episode_guid}"
        hash_value = hashlib.sha256(hash_input.encode()).hexdigest()[:12]
        return f"{prefix}{hash_value}"

    elif platform_lower == 'rumble':
        video_id = kwargs.get('video_id')
        if not video_id:
            raise ValueError("video_id required for Rumble content")
        return f"{prefix}{video_id}"

    raise ValueError(f"Unhandled platform: {platform}")


def is_new_format(content_id: str) -> bool:
    """Check if content_id uses the new prefixed format."""
    return bool(NEW_FORMAT_PATTERN.match(content_id))


def get_platform_from_id(content_id: str) -> Optional[str]:
    """
    Extract platform from a new-format content ID.
    Returns None for old-format IDs.
    """
    for platform, prefix in PLATFORM_PREFIXES.items():
        if content_id.startswith(prefix):
            return platform
    return None


def get_raw_id(content_id: str) -> str:
    """
    Extract the raw ID portion (without platform prefix).
    For old-format IDs, returns the ID unchanged.
    """
    for prefix in PLATFORM_PREFIXES.values():
        if content_id.startswith(prefix):
            return content_id[len(prefix):]
    return content_id
