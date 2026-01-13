#!/usr/bin/env python3
"""
FastAPI Audio Server
Serves audio content from S3 storage with automatic decompression
"""
import os
import sys
import logging
from pathlib import Path
from src.utils.paths import get_project_root, get_config_path
from typing import Optional, Tuple, Dict
from io import BytesIO
import tempfile
from contextlib import asynccontextmanager
import asyncio
import time
from datetime import datetime, timedelta
from collections import defaultdict
import functools

from fastapi import FastAPI, HTTPException, Response, Query, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import uvicorn
from pydub import AudioSegment

# Add project root to path
project_root = get_project_root()
sys.path.insert(0, str(project_root))

from src.utils.config import load_config
from src.utils.logger import setup_worker_logger
from src.storage.s3_utils import create_s3_storage_from_config

# Setup logging
logger = setup_worker_logger("audio_server")

# Global variables for storage
s3_storage = None
config = None

# Safety features
processing_lock = asyncio.Lock()  # Ensure single processing
rate_limit_tracker: Dict[str, float] = defaultdict(float)  # Track last request time per IP
RATE_LIMIT_SECONDS = 1  # Minimum seconds between requests
REQUEST_TIMEOUT_SECONDS = 30  # Timeout for processing
request_counter = 0  # For logging


def initialize_storage():
    """Initialize S3 storage"""
    global s3_storage, config
    
    try:
        # Load configuration
        config = load_config()
        
        # Create S3 storage with proper config path
        s3_storage = create_s3_storage_from_config(config['storage']['s3'])
        
        logger.info("Storage initialization completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize storage: {e}")
        raise


def get_client_ip(request: Request) -> str:
    """Get client IP address from request"""
    # Check for X-Forwarded-For header (if behind proxy)
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    # Fall back to direct client IP
    return request.client.host if request.client else "unknown"


async def check_rate_limit(client_ip: str) -> None:
    """Check if client has exceeded rate limit"""
    # Skip rate limiting for localhost/internal requests
    # Flask proxy makes all requests from 127.0.0.1, so rate limiting here
    # is redundant - Flask's own caching layer handles the real throttling
    if client_ip in ["127.0.0.1", "::1", "localhost"]:
        logger.debug(f"Skipping rate limit for localhost request from {client_ip}")
        return
    
    current_time = time.time()
    last_request_time = rate_limit_tracker[client_ip]
    
    if last_request_time > 0:
        time_since_last = current_time - last_request_time
        if time_since_last < RATE_LIMIT_SECONDS:
            remaining = RATE_LIMIT_SECONDS - time_since_last
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Please wait {remaining:.1f} seconds before next request."
            )
    
    rate_limit_tracker[client_ip] = current_time


async def safe_process_request(func, *args, **kwargs):
    """Safely process a request with timeout and queue management"""
    global request_counter
    request_counter += 1
    request_id = request_counter
    
    logger.info(f"Request {request_id} queued for processing")
    
    async with processing_lock:
        logger.info(f"Request {request_id} started processing")
        try:
            # Apply timeout to the processing
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=REQUEST_TIMEOUT_SECONDS
            )
            logger.info(f"Request {request_id} completed successfully")
            return result
        except asyncio.TimeoutError:
            logger.error(f"Request {request_id} timed out after {REQUEST_TIMEOUT_SECONDS}s")
            raise HTTPException(status_code=504, detail="Request timeout")
        except Exception as e:
            logger.error(f"Request {request_id} failed: {e}")
            raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    # Startup
    initialize_storage()
    logger.info("Audio server started with safety features:")
    logger.info(f"- Rate limit: {RATE_LIMIT_SECONDS}s between requests per IP")
    logger.info(f"- Request timeout: {REQUEST_TIMEOUT_SECONDS}s")
    logger.info("- Single concurrent request processing")
    logger.info("- Request logging for audit trail")
    yield
    # Shutdown
    logger.info("Shutting down audio server")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all incoming requests before any other processing"""
    
    async def dispatch(self, request: Request, call_next):
        # Get client info
        client_ip = request.client.host if request.client else "unknown"
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()
        
        host_header = request.headers.get("host", "unknown")
        user_agent = request.headers.get("user-agent", "unknown")
        method = request.method
        url = str(request.url)
        
        # Log the request
        logger.info(f"Incoming request: {method} {url} from {client_ip} (Host: {host_header}, UA: {user_agent})")
        
        # Process the request
        try:
            response = await call_next(request)
            logger.info(f"Response: {response.status_code} for {method} {url} from {client_ip}")
            return response
        except Exception as e:
            logger.error(f"Error processing request {method} {url} from {client_ip}: {e}")
            raise


# Initialize FastAPI app with lifespan
app = FastAPI(title="Audio Server", version="1.0.0", lifespan=lifespan)

# Add request logging middleware first (processes before other middleware)
app.add_middleware(RequestLoggingMiddleware)

# Add middleware for additional security
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "0.0.0.0", "100.100.32.69", "100.82.128.31", "100.114.55.95", "100.114.55.95:8010", "100.83.28.83", "10.0.0.4", "10.0.0.4:8010", "10.0.0.215", "10.0.0.215:8010", "100.94.185.14", "100.94.185.14:8010"])


def get_audio_path(content_id: str, is_chunk: bool = False, chunk_index: Optional[int] = None) -> Tuple[str, Optional[str]]:
    """
    Get S3 paths for audio files
    Returns: (primary_path, compressed_path)
    """
    if is_chunk and chunk_index is not None:
        # Chunk audio path
        base_path = f"content/{content_id}/chunks/{chunk_index}/audio.wav"
        return base_path, None
    else:
        # Full audio paths - check for compressed versions
        wav_path = f"content/{content_id}/audio.wav"
        opus_path = f"content/{content_id}/audio.opus"
        mp3_path = f"content/{content_id}/audio.mp3"

        # Check which files exist
        if s3_storage.file_exists(opus_path):
            return opus_path, opus_path
        elif s3_storage.file_exists(mp3_path):
            return mp3_path, mp3_path
        else:
            return wav_path, None


def get_video_path(content_id: str) -> Optional[str]:
    """
    Get S3 path for video file
    Returns: video_path or None if no video exists
    """
    # Check for video file in various formats
    video_extensions = ['.mp4', '.webm', '.mkv', '.avi', '.mov']

    # First check for source.mp4 (common for YouTube/Rumble downloads)
    source_mp4 = f"content/{content_id}/source.mp4"
    if s3_storage.file_exists(source_mp4):
        return source_mp4

    # Then check for video.{ext} files
    for ext in video_extensions:
        video_path = f"content/{content_id}/video{ext}"
        if s3_storage.file_exists(video_path):
            return video_path

    return None


def get_s3_presigned_url(s3_path: str, expiration: int = 3600) -> str:
    """Generate a presigned URL for S3 object access"""
    try:
        url = s3_storage._client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': s3_storage.config.bucket_name,
                'Key': s3_path
            },
            ExpiresIn=expiration
        )
        return url
    except Exception as e:
        logger.error(f"Failed to generate presigned URL for {s3_path}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate presigned URL: {str(e)}")


async def extract_audio_segment_ffmpeg(s3_url: str, start_time: Optional[float], end_time: Optional[float]) -> bytes:
    """
    Extract audio segment using ffmpeg with HTTP range requests.
    Returns opus-encoded audio data.
    """
    import subprocess

    # Create temp output file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.opus') as temp_file:
        temp_output = temp_file.name

    try:
        # Build ffmpeg command
        cmd = ['ffmpeg']

        # Add seeking if start_time specified (before -i for fast seeking with range requests)
        if start_time is not None:
            cmd.extend(['-ss', str(start_time)])

        # Input URL
        cmd.extend(['-i', s3_url])

        # Duration if end_time specified
        if start_time is not None and end_time is not None:
            duration = end_time - start_time
            cmd.extend(['-t', str(duration)])
        elif end_time is not None:
            # No start time, just limit to end_time
            cmd.extend(['-t', str(end_time)])

        # Output options
        cmd.extend([
            '-c:a', 'libopus',  # Opus codec
            '-b:a', '64k',       # Audio bitrate
            '-vn',               # No video
            '-y',                # Overwrite output
            temp_output
        ])

        logger.info(f"Running ffmpeg with range request seeking")

        # Run ffmpeg
        result = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await result.communicate()

        if result.returncode != 0:
            logger.error(f"ffmpeg failed: {stderr.decode()}")
            raise RuntimeError(f"ffmpeg failed: {stderr.decode()}")

        # Read output file
        with open(temp_output, 'rb') as f:
            audio_data = f.read()

        return audio_data

    finally:
        # Clean up temp file
        if os.path.exists(temp_output):
            os.unlink(temp_output)


async def extract_video_segment_ffmpeg(s3_url: str, start_time: Optional[float], end_time: Optional[float], output_format: str = 'mp4') -> bytes:
    """
    Extract video segment using ffmpeg with HTTP range requests.
    Returns video data (MP4 or WebM format) with video + audio streams.

    Args:
        s3_url: Presigned S3 URL for video file
        start_time: Start time in seconds (None = start from beginning)
        end_time: End time in seconds (None = go to end)
        output_format: Output format ('mp4' or 'webm')

    Returns:
        bytes: Video file data
    """
    import subprocess

    # Determine output extension and codec settings
    if output_format == 'webm':
        ext = '.webm'
        video_codec = 'libvpx-vp9'
        audio_codec = 'libopus'
    else:
        ext = '.mp4'
        video_codec = 'copy'  # Use stream copy for speed
        audio_codec = 'copy'

    # Create temp output file
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
        temp_output = temp_file.name

    try:
        # Build ffmpeg command
        cmd = ['ffmpeg']

        # Add seeking if start_time specified (before -i for fast seeking with range requests)
        if start_time is not None:
            cmd.extend(['-ss', str(start_time)])

        # Input URL
        cmd.extend(['-i', s3_url])

        # Duration if end_time specified
        if start_time is not None and end_time is not None:
            duration = end_time - start_time
            cmd.extend(['-t', str(duration)])
        elif end_time is not None:
            # No start time, just limit to end_time
            cmd.extend(['-t', str(end_time)])

        # Output options
        cmd.extend([
            '-c:v', video_codec,  # Video codec (copy for speed)
            '-c:a', audio_codec,  # Audio codec (copy for speed)
        ])

        # Add format-specific options
        if output_format == 'mp4':
            # Enable streaming optimization (move metadata to beginning)
            cmd.extend(['-movflags', '+faststart'])

        # Force output format and overwrite
        cmd.extend(['-f', output_format, '-y', temp_output])

        logger.info(f"Running ffmpeg video extraction with range requests: format={output_format}")

        # Run ffmpeg
        result = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await result.communicate()

        if result.returncode != 0:
            logger.error(f"ffmpeg video extraction failed: {stderr.decode()}")
            raise RuntimeError(f"ffmpeg video extraction failed: {stderr.decode()}")

        # Read output file
        with open(temp_output, 'rb') as f:
            video_data = f.read()

        logger.info(f"Successfully extracted video segment: {len(video_data) / 1024 / 1024:.2f} MB")
        return video_data

    finally:
        # Clean up temp file
        if os.path.exists(temp_output):
            os.unlink(temp_output)


async def load_audio(s3_path: str) -> Tuple[bytes, str]:
    """Load audio from S3 and return data with format"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(s3_path).suffix) as temp_file:
            temp_path = temp_file.name

        # Download to temp file
        s3_storage.download_file(s3_path, temp_path)

        # Read the file data
        with open(temp_path, 'rb') as f:
            audio_data = f.read()

        # Clean up temp file
        os.unlink(temp_path)

        # Determine format
        if s3_path.endswith('.opus'):
            format_type = 'opus'
        elif s3_path.endswith('.mp3'):
            format_type = 'mp3'
        else:
            format_type = 'wav'

        return audio_data, format_type
    except Exception as e:
        logger.error(f"Failed to download audio {s3_path}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to download audio: {str(e)}")


def slice_audio_by_time(audio_data: bytes, audio_format: str, start_time: Optional[float], end_time: Optional[float]) -> bytes:
    """Slice audio data by time range with safety validations"""
    # Load audio from bytes based on format
    if audio_format == 'opus':
        audio = AudioSegment.from_ogg(BytesIO(audio_data))
    elif audio_format == 'mp3':
        audio = AudioSegment.from_mp3(BytesIO(audio_data))
    else:
        audio = AudioSegment.from_wav(BytesIO(audio_data))

    audio_duration_seconds = len(audio) / 1000.0

    # Convert times to milliseconds
    start_ms = int(start_time * 1000) if start_time is not None else 0
    end_ms = int(end_time * 1000) if end_time is not None else len(audio)

    # Validate time range with detailed error messages
    if start_ms < 0:
        start_ms = 0
    if end_ms > len(audio):
        end_ms = len(audio)
    if start_ms >= end_ms:
        raise ValueError(f"Invalid time range: start_time ({start_time}s) must be less than end_time ({end_time}s). Audio duration: {audio_duration_seconds:.1f}s")

    # Slice the audio
    sliced_audio = audio[start_ms:end_ms]

    # Export to bytes in opus format
    buffer = BytesIO()
    sliced_audio.export(buffer, format="opus", codec="libopus")
    return buffer.getvalue()


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Audio/Video Server",
        "version": "1.0.0",
        "endpoints": {
            "/audio/{content_id}": "Get audio for content (supports ?start_time=X&end_time=Y)",
            "/audio/{content_id}/chunk/{chunk_index}": "Get specific chunk audio",
            "/video/{content_id}": "Get video for content (supports ?start_time=X&end_time=Y&format=mp4)",
            "/health": "Health check",
            "/debug/{content_id}": "Debug content availability"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test S3 connection
        s3_storage.list_files("")
        return {"status": "healthy", "storage": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "storage": "disconnected", "error": str(e)}


@app.get("/debug/{content_id}")
async def debug_content(content_id: str):
    """Debug endpoint to check content availability"""
    try:
        # Get audio paths
        audio_path, compressed_path = get_audio_path(content_id)
        
        # Check file existence
        wav_exists = s3_storage.file_exists(audio_path)
        compressed_exists = s3_storage.file_exists(compressed_path) if compressed_path else False
        
        return {
            "content_id": content_id,
            "audio_path": audio_path,
            "compressed_path": compressed_path,
            "wav_exists": wav_exists,
            "compressed_exists": compressed_exists,
            "recommended_path": compressed_path if compressed_exists else audio_path
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/audio/{content_id}")
async def get_audio(
    request: Request,
    content_id: str,
    start_time: Optional[float] = Query(None, description="Start time in seconds"),
    end_time: Optional[float] = Query(None, description="End time in seconds")
):
    """
    Get audio for a content ID with safety features:
    - Rate limiting: 5 seconds between requests per IP
    - Single concurrent processing
    - Request timeout: 30 seconds
    - Automatic decompression of compressed formats
    """
    # Get client IP and check rate limit
    client_ip = get_client_ip(request)
    await check_rate_limit(client_ip)
    
    # Log request with detailed info
    host_header = request.headers.get("host", "unknown")
    user_agent = request.headers.get("user-agent", "unknown")
    logger.info(f"Audio request from {client_ip} (host: {host_header}, UA: {user_agent}) for content_id={content_id}, start={start_time}, end={end_time}")
    
    # Process request with safety wrapper
    async def process():
        try:
            # Get audio paths
            audio_path, compressed_path = get_audio_path(content_id)

            # Determine which file to use
            if compressed_path and s3_storage.file_exists(compressed_path):
                actual_path = compressed_path
            elif s3_storage.file_exists(audio_path):
                actual_path = audio_path
            else:
                logger.error(f"Audio not found for content_id: {content_id}. Checked paths: {audio_path}, {compressed_path}")
                raise HTTPException(status_code=404, detail=f"Audio not found for content_id: {content_id}")

            logger.info(f"Serving audio for {content_id} from {actual_path}")

            # If time range is specified, use ffmpeg with range requests (efficient!)
            if start_time is not None or end_time is not None:
                try:
                    logger.info(f"Using ffmpeg range request extraction for {content_id}: start={start_time}s, end={end_time}s")

                    # Generate presigned URL
                    s3_url = get_s3_presigned_url(actual_path)

                    # Extract segment using ffmpeg (only downloads needed bytes)
                    audio_data = await extract_audio_segment_ffmpeg(s3_url, start_time, end_time)

                    logger.info(f"Successfully extracted audio segment for {content_id}: {len(audio_data) / 1024 / 1024:.2f} MB")
                except ValueError as e:
                    logger.warning(f"Invalid time range for {content_id}: {e}")
                    raise HTTPException(status_code=400, detail=str(e))
                except Exception as e:
                    logger.error(f"Failed to extract audio segment for {content_id}: {e}")
                    raise HTTPException(status_code=500, detail=f"Failed to extract audio segment: {str(e)}")
            else:
                # No time range - load full audio and convert to opus if needed
                audio_data, audio_format = await load_audio(actual_path)

                if audio_format != 'opus':
                    # Convert to opus if not already opus
                    try:
                        audio = AudioSegment.from_wav(BytesIO(audio_data)) if audio_format == 'wav' else AudioSegment.from_mp3(BytesIO(audio_data))
                        buffer = BytesIO()
                        audio.export(buffer, format="opus", codec="libopus")
                        audio_data = buffer.getvalue()
                    except Exception as e:
                        logger.error(f"Failed to convert audio for {content_id}: {e}")
                        raise HTTPException(status_code=500, detail=f"Failed to convert audio: {str(e)}")

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing audio request for {content_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

        # Return audio as streaming response
        return StreamingResponse(
            BytesIO(audio_data),
            media_type="audio/ogg",
            headers={
                "Content-Disposition": f"inline; filename={content_id}.opus",
                "Content-Length": str(len(audio_data))
            }
        )
    
    # Execute with safety wrapper
    return await safe_process_request(process)


@app.get("/audio/{content_id}/chunk/{chunk_index}")
async def get_chunk_audio(
    request: Request,
    content_id: str,
    chunk_index: int
):
    """
    Get audio for a specific chunk with safety features
    Chunks are always stored as uncompressed WAV files
    """
    # Get client IP and check rate limit
    client_ip = get_client_ip(request)
    await check_rate_limit(client_ip)
    
    # Log request
    logger.info(f"Chunk audio request from {client_ip} for content_id={content_id}, chunk={chunk_index}")
    
    # Process request with safety wrapper
    async def process():
        # Get chunk audio path
        chunk_path, _ = get_audio_path(content_id, is_chunk=True, chunk_index=chunk_index)

        # Check if chunk exists
        if not s3_storage.file_exists(chunk_path):
            raise HTTPException(status_code=404, detail=f"Chunk audio not found: {content_id}/chunk/{chunk_index}")

        logger.info(f"Serving chunk audio for {content_id}/chunk/{chunk_index}")

        # Load chunk audio
        audio_data, audio_format = await load_audio(chunk_path)

        # Convert to opus if not already
        if audio_format != 'opus':
            try:
                audio = AudioSegment.from_wav(BytesIO(audio_data)) if audio_format == 'wav' else AudioSegment.from_mp3(BytesIO(audio_data))
                buffer = BytesIO()
                audio.export(buffer, format="opus", codec="libopus")
                audio_data = buffer.getvalue()
            except Exception as e:
                logger.error(f"Failed to convert chunk audio for {content_id}/chunk/{chunk_index}: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to convert audio: {str(e)}")

        # Return audio as streaming response
        return StreamingResponse(
            BytesIO(audio_data),
            media_type="audio/ogg",
            headers={
                "Content-Disposition": f"inline; filename={content_id}_chunk_{chunk_index}.opus",
                "Content-Length": str(len(audio_data))
            }
        )
    
    # Execute with safety wrapper
    return await safe_process_request(process)


@app.get("/video/{content_id}")
async def get_video(
    request: Request,
    content_id: str,
    start_time: Optional[float] = Query(None, description="Start time in seconds"),
    end_time: Optional[float] = Query(None, description="End time in seconds"),
    format: str = Query("mp4", description="Output format (mp4 or webm)")
):
    """
    Get video for a content ID with safety features and efficient segment extraction.

    - Uses HTTP range requests for efficient seeking (only downloads needed data)
    - Returns video with both video and audio streams
    - Supports time-based clipping with start_time/end_time
    - Rate limiting and request queuing for safety

    Query Parameters:
        start_time: Start time in seconds (optional)
        end_time: End time in seconds (optional)
        format: Output format - 'mp4' (default) or 'webm'
    """
    # Get client IP and check rate limit
    client_ip = get_client_ip(request)
    await check_rate_limit(client_ip)

    # Log request with detailed info
    host_header = request.headers.get("host", "unknown")
    user_agent = request.headers.get("user-agent", "unknown")
    logger.info(f"Video request from {client_ip} (host: {host_header}, UA: {user_agent}) for content_id={content_id}, start={start_time}, end={end_time}, format={format}")

    # Validate format
    if format not in ['mp4', 'webm']:
        raise HTTPException(status_code=400, detail="Format must be 'mp4' or 'webm'")

    # Process request with safety wrapper
    async def process():
        try:
            # Get video path
            video_path = get_video_path(content_id)

            if not video_path:
                logger.error(f"Video not found for content_id: {content_id}")
                raise HTTPException(status_code=404, detail=f"Video not found for content_id: {content_id}")

            logger.info(f"Serving video for {content_id} from {video_path}")

            # If time range is specified, use ffmpeg with range requests (efficient!)
            if start_time is not None or end_time is not None:
                try:
                    logger.info(f"Using ffmpeg range request extraction for {content_id}: start={start_time}s, end={end_time}s, format={format}")

                    # Generate presigned URL
                    s3_url = get_s3_presigned_url(video_path)

                    # Extract segment using ffmpeg (only downloads needed bytes)
                    video_data = await extract_video_segment_ffmpeg(s3_url, start_time, end_time, output_format=format)

                    logger.info(f"Successfully extracted video segment for {content_id}: {len(video_data) / 1024 / 1024:.2f} MB")

                except ValueError as e:
                    logger.warning(f"Invalid time range for {content_id}: {e}")
                    raise HTTPException(status_code=400, detail=str(e))
                except Exception as e:
                    logger.error(f"Failed to extract video segment for {content_id}: {e}")
                    raise HTTPException(status_code=500, detail=f"Failed to extract video segment: {str(e)}")
            else:
                # No time range - return presigned URL for direct access (avoid downloading full file)
                # For full video, it's more efficient to redirect to presigned URL
                logger.info(f"Generating presigned URL for full video {content_id}")
                s3_url = get_s3_presigned_url(video_path, expiration=3600)

                # Return redirect to presigned URL
                from fastapi.responses import RedirectResponse
                return RedirectResponse(url=s3_url, status_code=302)

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing video request for {content_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

        # Return video as streaming response (for time-ranged requests)
        media_type = "video/webm" if format == "webm" else "video/mp4"
        return StreamingResponse(
            BytesIO(video_data),
            media_type=media_type,
            headers={
                "Content-Disposition": f"inline; filename={content_id}.{format}",
                "Content-Length": str(len(video_data)),
                "Accept-Ranges": "bytes"
            }
        )

    # Execute with safety wrapper
    return await safe_process_request(process)


if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8010,
        log_level="info"
    )