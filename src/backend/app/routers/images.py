"""
Image Proxy Router
==================

Serves thumbnails and avatars from S3 storage with browser caching.

Endpoints:
- /api/images/channels/{channel_key}.jpg - Channel thumbnails
- /api/images/speakers/{speaker_id}.jpg - Speaker avatars

Images are served with aggressive cache headers (1 week) since they rarely change.
"""

from fastapi import APIRouter, HTTPException, Path, Response
from fastapi.responses import Response as FastAPIResponse
from typing import Optional
import hashlib

from ..utils.backend_logger import get_logger
logger = get_logger("images_router")

from src.database.session import get_session
from src.database.models import Channel, SpeakerIdentity
from src.storage.s3_utils import S3Storage, S3StorageConfig

router = APIRouter(prefix="/api/images", tags=["images"])

# Lazy-loaded S3 storage
_s3_storage: Optional[S3Storage] = None


def _get_s3_storage() -> S3Storage:
    """Get or create S3 storage instance."""
    global _s3_storage
    if _s3_storage is None:
        _s3_storage = S3Storage(S3StorageConfig())
    return _s3_storage


# Cache control: 1 week for thumbnails (they rarely change)
CACHE_MAX_AGE = 60 * 60 * 24 * 7  # 7 days in seconds
CACHE_HEADERS = {
    "Cache-Control": f"public, max-age={CACHE_MAX_AGE}, immutable",
    "X-Content-Type-Options": "nosniff",
}


def _generate_etag(data: bytes) -> str:
    """Generate ETag from image bytes."""
    return f'"{hashlib.md5(data).hexdigest()}"'


@router.get("/channels/{channel_key:path}.jpg")
async def get_channel_thumbnail(
    channel_key: str = Path(..., description="Channel key (e.g., podcast:some-name-abc123)")
):
    """
    Get channel thumbnail by channel_key.

    Returns the thumbnail image with caching headers.
    Falls back to a placeholder if no thumbnail exists.
    """
    try:
        # Look up channel by channel_key
        with get_session() as session:
            channel = session.query(Channel).filter(
                Channel.channel_key == channel_key
            ).first()

            if not channel:
                raise HTTPException(status_code=404, detail=f"Channel not found: {channel_key}")

            channel_id = channel.id

        # Try to fetch from S3
        s3 = _get_s3_storage()

        # Check for thumbnail in various formats
        for ext in ['jpg', 'png', 'webp']:
            s3_key = f"thumbnails/channels/{channel_id}.{ext}"
            if s3.file_exists(s3_key):
                # Download the image bytes directly
                try:
                    response = s3._client.get_object(
                        Bucket=s3.config.bucket_name,
                        Key=s3_key
                    )
                    image_data = response['Body'].read()

                    # Determine content type
                    content_type = {
                        'jpg': 'image/jpeg',
                        'png': 'image/png',
                        'webp': 'image/webp',
                    }.get(ext, 'image/jpeg')

                    return Response(
                        content=image_data,
                        media_type=content_type,
                        headers={
                            **CACHE_HEADERS,
                            "ETag": _generate_etag(image_data),
                        }
                    )
                except Exception as e:
                    logger.warning(f"Error downloading thumbnail for channel {channel_id}: {e}")
                    break

        # No thumbnail found - return 404
        # Frontend should handle this with a placeholder
        raise HTTPException(status_code=404, detail="Thumbnail not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching channel thumbnail for {channel_key}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/speakers/{speaker_id}.jpg")
async def get_speaker_avatar(
    speaker_id: int = Path(..., description="Speaker identity ID")
):
    """
    Get speaker avatar by speaker_identity ID.

    Returns the avatar image with caching headers.
    Falls back to a placeholder if no avatar exists.
    """
    try:
        # Verify speaker exists
        with get_session() as session:
            speaker = session.query(SpeakerIdentity).filter(
                SpeakerIdentity.id == speaker_id
            ).first()

            if not speaker:
                raise HTTPException(status_code=404, detail=f"Speaker not found: {speaker_id}")

        # Try to fetch from S3
        s3 = _get_s3_storage()

        # Check for avatar in various formats
        for ext in ['jpg', 'png', 'webp']:
            s3_key = f"thumbnails/speakers/{speaker_id}.{ext}"
            if s3.file_exists(s3_key):
                try:
                    response = s3._client.get_object(
                        Bucket=s3.config.bucket_name,
                        Key=s3_key
                    )
                    image_data = response['Body'].read()

                    content_type = {
                        'jpg': 'image/jpeg',
                        'png': 'image/png',
                        'webp': 'image/webp',
                    }.get(ext, 'image/jpeg')

                    return Response(
                        content=image_data,
                        media_type=content_type,
                        headers={
                            **CACHE_HEADERS,
                            "ETag": _generate_etag(image_data),
                        }
                    )
                except Exception as e:
                    logger.warning(f"Error downloading avatar for speaker {speaker_id}: {e}")
                    break

        # No avatar found
        raise HTTPException(status_code=404, detail="Avatar not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching speaker avatar for {speaker_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
