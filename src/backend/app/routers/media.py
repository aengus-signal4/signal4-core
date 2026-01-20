"""
Media Router
============

Unified media endpoint serving audio/video content with optional transcription.

Features:
- Audio and video content from S3 storage
- Automatic decompression and segment extraction
- Optional AssemblyAI transcription (transcribe=true)
- Streaming SSE mode for progressive loading (stream=true)
- Efficient HTTP range requests

Endpoints:
- GET /api/media/content/{content_id} - Get media with optional transcription
"""

import os
import sys
import tempfile
import asyncio
import time
import json
import re
from pathlib import Path
from src.utils.paths import get_project_root, get_config_path
from typing import Optional, Tuple, AsyncGenerator
from io import BytesIO
from collections import defaultdict

from fastapi import APIRouter, HTTPException, Response, Query, Request


# Content ID validation pattern - alphanumeric, underscores, hyphens, periods
CONTENT_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_.-]+$')


def validate_content_id(content_id: str) -> None:
    """
    Validate content_id to prevent path traversal and injection attacks.

    Raises HTTPException 400 if content_id is invalid.
    """
    if not content_id:
        raise HTTPException(status_code=400, detail="content_id is required")
    if not CONTENT_ID_PATTERN.match(content_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid content_id format. Only alphanumeric characters, underscores, hyphens, and periods are allowed."
        )
    # Explicit path traversal check
    if '..' in content_id or '/' in content_id or '\\' in content_id:
        raise HTTPException(status_code=400, detail="Invalid content_id: path traversal not allowed")
from fastapi.responses import StreamingResponse, RedirectResponse, JSONResponse
from pydub import AudioSegment
from sqlalchemy import func

# Add project root to path
project_root = get_project_root()
sys.path.insert(0, str(project_root))

from src.utils.config import load_config
from src.storage.s3_utils import create_s3_storage_from_config
from src.database.models import EmbeddingSegment, AlternativeTranscription, Content, Speaker, SpeakerIdentity
from src.utils.speaker_segments import format_speaker_attributed_text

from ..utils.backend_logger import get_logger
logger = get_logger("media_router")

router = APIRouter(prefix="/api/media", tags=["media"])

# Global variables for storage and services
s3_storage = None
config = None
_assemblyai_service = None
_db_engine = None
_SessionLocal = None

# Safety features
processing_lock = asyncio.Lock()  # Ensure single processing
REQUEST_TIMEOUT_SECONDS = 30  # Timeout for processing
request_counter = 0

# Public API URL for media URLs returned to frontend
PUBLIC_API_URL = os.environ.get('PUBLIC_API_URL', '')


def initialize_storage():
    """Initialize S3 storage and database connection"""
    global s3_storage, config, _db_engine, _SessionLocal

    try:
        config = load_config()
        s3_storage = create_s3_storage_from_config(config['storage']['s3'])

        # Initialize database connection for transcription
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from src.backend.app.config.database import get_database_url
        _db_engine = create_engine(get_database_url())
        _SessionLocal = sessionmaker(bind=_db_engine)

        logger.info("Media storage and database initialization completed successfully")
    except Exception as e:
        logger.error(f"Failed to initialize media storage: {e}")
        raise


def get_assemblyai_service():
    """Get or create AssemblyAI service"""
    global _assemblyai_service
    if _assemblyai_service is None:
        from ..services.assemblyai_service import AssemblyAIService
        _assemblyai_service = AssemblyAIService()
    return _assemblyai_service


def get_db_session():
    """Get database session"""
    if _SessionLocal is None:
        initialize_storage()
    return _SessionLocal()


async def safe_process_request(func, *args, **kwargs):
    """Safely process a request with timeout and queue management"""
    global request_counter
    request_counter += 1
    request_id = request_counter

    logger.debug(f"Request {request_id} queued for processing")

    async with processing_lock:
        logger.debug(f"Request {request_id} started processing")
        try:
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=REQUEST_TIMEOUT_SECONDS
            )
            logger.debug(f"Request {request_id} completed successfully")
            return result
        except asyncio.TimeoutError:
            logger.error(f"Request {request_id} timed out after {REQUEST_TIMEOUT_SECONDS}s")
            raise HTTPException(status_code=504, detail="Request timeout")
        except Exception as e:
            logger.error(f"Request {request_id} failed: {e}")
            raise


def get_audio_path(content_id: str) -> Tuple[str, Optional[str]]:
    """Get S3 paths for audio files. Returns: (primary_path, compressed_path)"""
    wav_path = f"content/{content_id}/audio.wav"
    opus_path = f"content/{content_id}/audio.opus"
    mp3_path = f"content/{content_id}/audio.mp3"

    if s3_storage.file_exists(opus_path):
        return opus_path, opus_path
    elif s3_storage.file_exists(mp3_path):
        return mp3_path, mp3_path
    else:
        return wav_path, None


def get_video_path(content_id: str) -> Optional[str]:
    """Get S3 path for video file. Returns: video_path or None if no video exists"""
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
    """Extract audio segment using ffmpeg with HTTP range requests. Returns webm-encoded audio data."""
    import subprocess

    with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_file:
        temp_output = temp_file.name

    try:
        cmd = ['ffmpeg']

        if start_time is not None:
            cmd.extend(['-ss', str(start_time)])

        cmd.extend(['-i', s3_url])

        if start_time is not None and end_time is not None:
            duration = end_time - start_time
            cmd.extend(['-t', str(duration)])
        elif end_time is not None:
            cmd.extend(['-t', str(end_time)])

        cmd.extend([
            '-c:a', 'libopus',
            '-b:a', '64k',
            '-vn',
            '-f', 'webm',
            '-y',
            temp_output
        ])

        logger.debug(f"Running ffmpeg audio extraction with range requests")

        result = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await result.communicate()

        if result.returncode != 0:
            logger.error(f"ffmpeg failed: {stderr.decode()}")
            raise RuntimeError(f"ffmpeg failed: {stderr.decode()}")

        with open(temp_output, 'rb') as f:
            audio_data = f.read()

        return audio_data

    finally:
        if os.path.exists(temp_output):
            os.unlink(temp_output)


async def _stream_media_with_transcription(
    content_id: str,
    segment_id: Optional[int],
    start_time: Optional[float],
    end_time: Optional[float],
    media_type: str,
    format: str,
    transcribe: bool,
    transcription_model: str,
    language: Optional[str],
    speaker_labels: bool,
    enable_translation: bool,
    force_retranscribe: bool
) -> AsyncGenerator[str, None]:
    """
    SSE stream generator for media + transcription with parallel processing

    Optimized Architecture:
    1. Query EmbeddingSegment FIRST (~10ms) to get correct segment boundaries
    2. IN PARALLEL:
       - Extract media using boundaries (~1-2s)
       - Query DB for existing transcripts (embedding_segment or alternative_transcription)
    3. Send media_ready IMMEDIATELY when extraction completes
    4. Send transcript from DB if found, otherwise generate with AssemblyAI if needed

    Key optimizations:
    - Audio extraction + DB transcript lookup happen in parallel (saves ~1-2s)
    - Media sent immediately without waiting for transcript processing
    - Smart transcript fallback: DB → AssemblyAI (if transcribe=True and ≤120s)
    """
    extracted_audio_path = None

    try:
        yield f"data: {json.dumps({'event': 'started'})}\n\n"
        await asyncio.sleep(0)

        # STEP 1: Get segment boundaries (required first - everything else depends on this)
        yield f"data: {json.dumps({'event': 'segment_lookup'})}\n\n"
        await asyncio.sleep(0)

        embedding_segment = _get_embedding_segment_transcript(
            content_id=content_id,
            start_time=start_time,
            end_time=end_time,
            segment_id=segment_id
        )

        # Determine actual boundaries
        if embedding_segment:
            actual_start_time = embedding_segment['start_time']
            actual_end_time = embedding_segment['end_time']
            segment_duration = actual_end_time - actual_start_time
            logger.info(f"Using EmbeddingSegment boundaries: {actual_start_time:.2f}s - {actual_end_time:.2f}s")
        else:
            actual_start_time = start_time
            actual_end_time = end_time
            segment_duration = (end_time - start_time) if (start_time is not None and end_time is not None) else None
            logger.info(f"No EmbeddingSegment found, using requested times: {start_time:.2f}s - {end_time:.2f}s")

        # Check segment length for transcription limits
        MAX_TRANSCRIPTION_LENGTH = 120.0
        skip_new_transcription = False
        if transcribe and segment_duration and segment_duration > MAX_TRANSCRIPTION_LENGTH:
            logger.warning(f"Segment too long for transcription: {segment_duration:.1f}s > {MAX_TRANSCRIPTION_LENGTH}s")
            skip_new_transcription = True

        # STEP 2: Launch parallel operations - media extraction + DB transcript lookup
        yield f"data: {json.dumps({'event': 'parallel_processing'})}\n\n"
        await asyncio.sleep(0)

        # Determine media extraction parameters
        video_path = get_video_path(content_id)
        has_video = video_path is not None

        if media_type == 'audio':
            extract_video_flag = False
        elif media_type == 'video':
            if not has_video:
                raise HTTPException(status_code=404, detail=f"Video not found for {content_id}")
            extract_video_flag = True
        else:  # auto
            extract_video_flag = has_video

        # Define media extraction coroutine
        async def extract_media():
            if extract_video_flag:
                s3_url = get_s3_presigned_url(video_path)
                media_data = await extract_video_segment_ffmpeg(s3_url, actual_start_time, actual_end_time, output_format=format)
                return media_data, 'video'
            else:
                audio_path, compressed_path = get_audio_path(content_id)
                if compressed_path and s3_storage.file_exists(compressed_path):
                    actual_path = compressed_path
                elif s3_storage.file_exists(audio_path):
                    actual_path = audio_path
                else:
                    raise HTTPException(status_code=404, detail=f"No media found for {content_id}")

                s3_url = get_s3_presigned_url(actual_path)
                media_data = await extract_audio_segment_ffmpeg(s3_url, actual_start_time, actual_end_time)
                return media_data, 'audio'

        # Define DB transcript lookup coroutine
        async def check_db_transcript():
            # Check AlternativeTranscription table first (cached AssemblyAI results)
            db = get_db_session()
            try:
                content = db.query(Content).filter(Content.content_id == content_id).first()
                if content:
                    existing = db.query(AlternativeTranscription).filter(
                        AlternativeTranscription.content_id == content.id,
                        AlternativeTranscription.provider == "assemblyai",
                        AlternativeTranscription.model == transcription_model
                    ).first()

                    if existing:
                        logger.info("Found cached AssemblyAI transcript in DB")
                        speaker_transcript = _format_speaker_labels(existing.speaker_labels)
                        return {
                            'source': 'assemblyai',
                            'cached': True,
                            'transcript': speaker_transcript or existing.transcription_text,
                            'language': existing.language,
                            'translation_en': existing.translation_en,
                            'translation_fr': existing.translation_fr,
                            'confidence': existing.confidence
                        }
            finally:
                db.close()

            # Fallback to embedding_segment if no AssemblyAI cache
            if embedding_segment:
                logger.info("Using EmbeddingSegment transcript from parallel lookup")
                return {
                    'source': 'embedding_segment',
                    'cached': True,
                    'transcript': embedding_segment['transcript'],
                    'language': embedding_segment.get('language'),
                    'translation_en': None,
                    'translation_fr': None,
                    'confidence': None
                }

            return None

        # Execute in parallel
        media_result, db_transcript = await asyncio.gather(
            extract_media(),
            check_db_transcript(),
            return_exceptions=True
        )

        # Handle media extraction result
        if isinstance(media_result, Exception):
            raise media_result

        media_data, actual_media_type = media_result

        # STEP 3: Send media URL IMMEDIATELY (don't wait for transcript)
        media_size_mb = len(media_data) / 1024 / 1024
        logger.info(f"[MEDIA] Extracted {media_size_mb:.2f}MB, sending media_ready event")

        # Build media URL (use public URL if configured for external access)
        base_url = PUBLIC_API_URL if PUBLIC_API_URL else ""
        media_url = f"{base_url}/api/media/content/{content_id}"
        params = []
        if actual_start_time is not None:
            params.append(f"start_time={actual_start_time}")
        if actual_end_time is not None:
            params.append(f"end_time={actual_end_time}")
        params.append(f"media_type={media_type}")
        params.append(f"format={format}")
        media_url += "?" + "&".join(params) if params else ""

        # Fetch episode metadata for modal display
        episode_metadata = _get_episode_metadata(content_id)

        media_ready_event = {
            'event': 'media_ready',
            'url': media_url,
            'media_type': actual_media_type,
            'requested_type': media_type,
            'start_time': actual_start_time,
            'end_time': actual_end_time
        }

        # Include episode metadata if available
        if episode_metadata:
            media_ready_event.update(episode_metadata)

        yield f"data: {json.dumps(media_ready_event)}\n\n"
        await asyncio.sleep(0)

        # STEP 4: Handle transcript
        transcript_returned = False

        # Check if we got a DB transcript from parallel lookup
        if db_transcript and not isinstance(db_transcript, Exception):
            logger.info(f"Returning DB transcript (source: {db_transcript['source']})")
            yield f"data: {json.dumps({'event': 'transcript_cached', **db_transcript})}\n\n"
            transcript_returned = True

        # If no DB transcript and transcribe=True, generate with AssemblyAI
        if not transcript_returned and transcribe and not skip_new_transcription:
            logger.info(f"No cached transcript, generating with AssemblyAI (duration: {segment_duration:.1f}s)")

            # Convert to WAV for transcription
            import tempfile

            if extract_video_flag:
                temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=f'.{format}')
                temp_video.write(media_data)
                temp_video.close()

                temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                temp_audio.close()

                cmd = [
                    'ffmpeg', '-i', temp_video.name,
                    '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le',
                    '-y', temp_audio.name
                ]

                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                await proc.communicate()

                os.unlink(temp_video.name)
                extracted_audio_path = temp_audio.name
            else:
                temp_opus = tempfile.NamedTemporaryFile(delete=False, suffix='.webm')
                temp_opus.write(media_data)
                temp_opus.close()

                temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                temp_audio.close()

                cmd = [
                    'ffmpeg', '-i', temp_opus.name,
                    '-ar', '16000', '-ac', '1', '-c:a', 'pcm_s16le',
                    '-y', temp_audio.name
                ]

                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                await proc.communicate()

                os.unlink(temp_opus.name)
                extracted_audio_path = temp_audio.name

            # Generate transcript
            transcript_data = await _get_or_generate_transcript(
                content_id=content_id,
                start_time=actual_start_time,
                end_time=actual_end_time,
                model=transcription_model,
                language=language,
                speaker_labels=speaker_labels,
                enable_translation=enable_translation,
                force=force_retranscribe,
                extracted_audio_path=extracted_audio_path,
                cache_only=False
            )

            if transcript_data:
                yield f"data: {json.dumps({'event': 'transcript_ready', **transcript_data['data'], 'source': 'assemblyai'})}\n\n"
                transcript_returned = True

        # Handle cases where no transcript was found or generated
        if not transcript_returned:
            if skip_new_transcription:
                logger.warning(f"No transcript available for long segment ({segment_duration:.1f}s)")
                yield f"data: {json.dumps({'event': 'transcript_skipped', 'reason': 'no_transcript_available', 'duration': segment_duration})}\n\n"
            else:
                logger.warning(f"No transcript found for {content_id}")
                yield f"data: {json.dumps({'event': 'transcript_skipped', 'reason': 'not_found'})}\n\n"

        yield f"data: {json.dumps({'event': 'complete'})}\n\n"

    except Exception as e:
        logger.error(f"Streaming error: {e}", exc_info=True)
        yield f"data: {json.dumps({'event': 'error', 'message': str(e)})}\n\n"
    finally:
        # Cleanup temp files
        if extracted_audio_path and os.path.exists(extracted_audio_path):
            os.unlink(extracted_audio_path)


async def _get_or_generate_transcript(
    content_id: str,
    start_time: Optional[float],
    end_time: Optional[float],
    model: str,
    language: Optional[str],
    speaker_labels: bool,
    enable_translation: bool,
    force: bool,
    extracted_audio_path: Optional[str] = None,  # Accept pre-extracted audio
    cache_only: bool = False  # NEW: Only return cached, never generate
) -> Optional[dict]:
    """
    Get cached transcript or generate new one

    Args:
        extracted_audio_path: Optional pre-extracted audio file path (WAV format).
                             If provided, skips download step and uses this file directly.
        cache_only: If True, only return cached transcript, never generate new.
                   Returns None if no cache exists.

    Returns:
        Dict with transcript data, or None if cache_only=True and no cache exists
    """
    db = get_db_session()
    provider = "assemblyai"

    try:
        # Look up content in database
        content = db.query(Content).filter(Content.content_id == content_id).first()
        if not content:
            raise HTTPException(status_code=404, detail=f"Content {content_id} not found")

        db_content_id = content.id

        # Auto-detect language if not provided
        if not language and content.main_language:
            language = content.main_language[:2].lower()

        # Check cache unless forced
        if not force:
            # Try to find matching transcription
            existing = db.query(AlternativeTranscription).filter(
                AlternativeTranscription.content_id == db_content_id,
                AlternativeTranscription.provider == provider,
                AlternativeTranscription.model == model
            ).first()

            if existing:
                # Format speaker transcript
                speaker_transcript = _format_speaker_labels(existing.speaker_labels)
                return {
                    'cached': True,
                    'data': {
                        'transcript': speaker_transcript or existing.transcription_text,
                        'language': existing.language,
                        'translation_en': existing.translation_en,
                        'translation_fr': existing.translation_fr,
                        'confidence': existing.confidence
                    }
                }

        # If cache_only mode and no cache found, return None
        if cache_only:
            logger.info(f"Cache-only mode: No cached transcript found for {content_id}")
            return None

        # Cache miss - generate new transcription
        assemblyai_service = get_assemblyai_service()

        # Use pre-extracted audio (required)
        if not extracted_audio_path:
            # This should never happen in the new architecture
            logger.error(f"No pre-extracted audio provided for {content_id} - this is a bug in the streaming code")
            raise HTTPException(
                status_code=500,
                detail="Internal error: Audio must be pre-extracted before transcription"
            )

        audio_file = extracted_audio_path
        should_cleanup = False  # Caller will cleanup

        try:
            # Transcribe
            result = assemblyai_service.transcribe_from_file(
                audio_path=audio_file,
                language=language,
                speaker_labels=speaker_labels,
                enable_translation=enable_translation,
                config={"speech_model": model} if model and model != "best" else None
            )

            # Save to database
            alt_trans = AlternativeTranscription(
                segment_id=None,
                content_id=db_content_id,
                provider=provider,
                model=model,
                language=result.get("language") or language,
                transcription_text=result["transcription_text"],
                translation_en=result.get("translation_en"),
                translation_fr=result.get("translation_fr"),
                confidence=result.get("confidence"),
                word_timings=result.get("word_timings"),
                speaker_labels=result.get("speaker_labels"),
                audio_duration=result.get("audio_duration"),
                processing_time=result.get("processing_time"),
                meta_data=result.get("metadata", {}),
                api_cost=result.get("api_cost"),
                created_at=datetime.utcnow()
            )

            db.add(alt_trans)
            db.commit()

            speaker_transcript = _format_speaker_labels(result.get("speaker_labels"))

            return {
                'cached': False,
                'data': {
                    'transcript': speaker_transcript or result["transcription_text"],
                    'language': result.get("language"),
                    'translation_en': result.get("translation_en"),
                    'translation_fr': result.get("translation_fr"),
                    'confidence': result.get("confidence")
                }
            }

        finally:
            # Only cleanup if we downloaded it ourselves
            if should_cleanup:
                assemblyai_service.cleanup_temp_file(audio_file)

    finally:
        db.close()


def _format_transcript_with_speakers(segment: EmbeddingSegment, db) -> str:
    """
    Format transcript with speaker attribution using speaker_positions data.

    Output format: "Speaker A (Name)" or "Speaker A (Unknown)" for unidentified speakers.

    Args:
        segment: EmbeddingSegment with text and speaker_positions
        db: Database session for querying speaker names

    Returns:
        Formatted transcript with letter-based speaker labels (Speaker A, B, C...)
        Falls back to plain text if no speaker_positions available
    """
    # If no speaker_positions, return plain text
    if not segment.speaker_positions:
        return segment.text

    try:
        # Get speaker IDs from positions (sorted for consistent letter assignment)
        speaker_ids_raw = list(segment.speaker_positions.keys())
        speaker_ids = sorted([int(sid) for sid in speaker_ids_raw])

        # Query speaker names from database (with identity mapping)
        speaker_name_map = {}
        if speaker_ids:
            speakers_query = db.query(
                Speaker.id,
                Speaker.display_name,
                SpeakerIdentity.primary_name
            ).outerjoin(
                SpeakerIdentity,
                Speaker.speaker_identity_id == SpeakerIdentity.id
            ).filter(
                Speaker.id.in_(speaker_ids)
            ).all()

            # Build speaker name map with priority: identity name > display name > None
            for speaker_id, display_name, identity_name in speakers_query:
                # Check if identity_name is UNIDENTIFIED_CLUSTER pattern
                if identity_name and identity_name.startswith('UNIDENTIFIED_CLUSTER'):
                    speaker_name_map[str(speaker_id)] = None  # Treat as unknown
                elif identity_name:
                    speaker_name_map[str(speaker_id)] = identity_name
                elif display_name:
                    speaker_name_map[str(speaker_id)] = display_name
                else:
                    speaker_name_map[str(speaker_id)] = None

        # Create letter-based labels (A, B, C...) with name fallback
        # Map speaker_id -> "Speaker A (Name)" or "Speaker A (Unknown)"
        letter_labels = {}
        for idx, speaker_id in enumerate(speaker_ids):
            letter = chr(65 + idx)  # 65 = 'A' in ASCII
            speaker_id_str = str(speaker_id)

            # Get name or use "Unknown"
            name = speaker_name_map.get(speaker_id_str)
            if name:
                label = f"Speaker {letter} ({name})"
            else:
                label = f"Speaker {letter} (Unknown)"

            letter_labels[speaker_id_str] = label

        # Replace speaker IDs in positions dict with letter-based labels
        enriched_positions = {}
        for speaker_id, positions in segment.speaker_positions.items():
            label = letter_labels.get(speaker_id, f"Speaker ? (Unknown)")
            enriched_positions[label] = positions

        # Use utility function to format with enriched labels
        formatted = format_speaker_attributed_text(segment.text, enriched_positions)
        logger.debug(f"Formatted transcript with {len(speaker_ids)} speakers using letter labels")
        return formatted

    except Exception as e:
        logger.warning(f"Failed to format transcript with speakers: {e}, falling back to plain text")
        return segment.text


def _get_embedding_segment_transcript(
    content_id: str,
    start_time: Optional[float],
    end_time: Optional[float],
    segment_id: Optional[int] = None
) -> Optional[dict]:
    """
    Get transcript from EmbeddingSegment table (existing transcripts from Whisper/etc)

    Returns speaker-attributed transcript when speaker_positions data is available.

    Lookup strategy:
    1. If segment_id provided, try direct lookup (fast path)
    2. If not found or not provided, find closest segment by start_time (no distance limit)

    Returns:
        Dict with transcript data (speaker-attributed if available), or None if not found
    """
    db = get_db_session()

    try:
        # Look up content in database
        content = db.query(Content).filter(Content.content_id == content_id).first()
        if not content:
            logger.warning(f"Content not found for {content_id}")
            return None

        db_content_id = content.id

        # Strategy 1: Use segment_id if provided (lightning fast)
        if segment_id:
            segment = db.query(EmbeddingSegment).filter(
                EmbeddingSegment.id == segment_id,
                EmbeddingSegment.content_id == db_content_id
            ).first()

            if segment:
                logger.info(f"✓ Found EmbeddingSegment by segment_id: {segment_id} (fast path)")
                transcript = _format_transcript_with_speakers(segment, db)
                return {
                    'transcript': transcript,
                    'language': content.main_language,
                    'segment_id': segment.id,
                    'start_time': segment.start_time,
                    'end_time': segment.end_time
                }
            else:
                logger.warning(f"segment_id {segment_id} not found, falling back to timing lookup")

        # Strategy 2: Find closest segment by start_time (no tolerance limit)
        if start_time is not None:
            # Find the segment with start_time closest to the requested time
            # No distance limit - just get the closest one
            segment = db.query(EmbeddingSegment).filter(
                EmbeddingSegment.content_id == db_content_id
            ).order_by(
                func.abs(EmbeddingSegment.start_time - start_time)
            ).first()

            if segment:
                time_diff = abs(segment.start_time - start_time)
                logger.info(f"✓ Found closest EmbeddingSegment by timing: {segment.start_time:.2f}s (requested: {start_time:.2f}s, diff: {time_diff:.2f}s)")
                transcript = _format_transcript_with_speakers(segment, db)
                return {
                    'transcript': transcript,
                    'language': content.main_language,
                    'segment_id': segment.id,
                    'start_time': segment.start_time,
                    'end_time': segment.end_time
                }

        logger.info(f"No EmbeddingSegment found for content_id={content_id}, start_time={start_time}")
        return None

    except Exception as e:
        logger.error(f"Error querying EmbeddingSegment: {e}", exc_info=True)
        return None
    finally:
        db.close()


def _format_speaker_labels(speaker_labels: Optional[list]) -> Optional[str]:
    """Format transcript with speaker labels"""
    if not speaker_labels:
        return None

    formatted_lines = []
    for utterance in speaker_labels:
        speaker = utterance.get("speaker", "Unknown")
        text = utterance.get("text", "")
        formatted_lines.append(f"Speaker {speaker}: {text}")

    return "\n".join(formatted_lines)


def _get_episode_metadata(content_id: str) -> Optional[dict]:
    """
    Get episode metadata from Content table.

    Returns:
        Dict with episode_title, episode_description, channel_name, publish_date, platform, source_url
        or None if content not found
    """
    db = get_db_session()
    try:
        content = db.query(Content).filter(Content.content_id == content_id).first()
        if not content:
            logger.warning(f"Content not found for metadata lookup: {content_id}")
            return None

        # Extract source URL from metadata for podcasts
        source_url = None
        if content.meta_data and isinstance(content.meta_data, dict):
            source_url = content.meta_data.get('episode_url')

        return {
            'episode_title': content.title,
            'episode_description': content.description,
            'channel_name': content.channel_name,
            'publish_date': content.publish_date.isoformat() if content.publish_date else None,
            'platform': content.platform,
            'source_url': source_url
        }
    except Exception as e:
        logger.error(f"Error fetching episode metadata: {e}")
        return None
    finally:
        db.close()


async def extract_video_segment_ffmpeg(s3_url: str, start_time: Optional[float], end_time: Optional[float], output_format: str = 'mp4') -> bytes:
    """Extract video segment using ffmpeg with HTTP range requests. Returns video data (MP4 or WebM format)."""
    import subprocess

    if output_format == 'webm':
        ext = '.webm'
        video_codec = 'libvpx-vp9'
        audio_codec = 'libopus'
    else:
        ext = '.mp4'
        video_codec = 'copy'
        audio_codec = 'copy'

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
        temp_output = temp_file.name

    try:
        cmd = ['ffmpeg']

        if start_time is not None:
            cmd.extend(['-ss', str(start_time)])

        cmd.extend(['-i', s3_url])

        if start_time is not None and end_time is not None:
            duration = end_time - start_time
            cmd.extend(['-t', str(duration)])
        elif end_time is not None:
            cmd.extend(['-t', str(end_time)])

        cmd.extend([
            '-c:v', video_codec,
            '-c:a', audio_codec,
        ])

        if output_format == 'mp4':
            cmd.extend(['-movflags', '+faststart'])

        cmd.extend(['-f', output_format, '-y', temp_output])

        logger.debug(f"Running ffmpeg video extraction with range requests: format={output_format}")

        result = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await result.communicate()

        if result.returncode != 0:
            logger.error(f"ffmpeg video extraction failed: {stderr.decode()}")
            raise RuntimeError(f"ffmpeg video extraction failed: {stderr.decode()}")

        with open(temp_output, 'rb') as f:
            video_data = f.read()

        logger.info(f"Successfully extracted video segment: {len(video_data) / 1024 / 1024:.2f} MB")
        return video_data

    finally:
        if os.path.exists(temp_output):
            os.unlink(temp_output)


@router.get("/segment/{segment_id}")
async def get_media_by_segment(
    request: Request,
    segment_id: int,
    media_type: str = Query("auto", description="Media type: 'video', 'audio', or 'auto'"),
    format: str = Query("mp4", description="Output format (mp4 or webm for video)"),
    transcribe: bool = Query(False, description="Enable AssemblyAI transcription (currently disabled)"),
    transcription_provider: str = Query("assemblyai", description="Transcription provider"),
    transcription_model: str = Query("best", description="Transcription model"),
    language: Optional[str] = Query(None, description="Language code (auto-detect if None)"),
    speaker_labels: bool = Query(True, description="Enable speaker diarization"),
    enable_translation: bool = Query(True, description="Enable translation"),
    force_retranscribe: bool = Query(False, description="Force re-transcription"),
    stream: bool = Query(False, description="Enable SSE streaming mode")
):
    """
    Get media by segment_id (looks up content_id and timing from EmbeddingSegment table).

    **Usage:**
    ```
    GET /api/media/segment/12345678?stream=true&transcribe=true
    ```

    Backend looks up segment, extracts content_id + timing, and serves media.
    """
    logger.info(f"[MEDIA] Segment lookup request: segment_id={segment_id}, stream={stream}, transcribe={transcribe}")

    # Look up segment in database
    db = get_db_session()
    try:
        segment = db.query(EmbeddingSegment).filter(EmbeddingSegment.id == segment_id).first()
        if not segment:
            raise HTTPException(status_code=404, detail=f"Segment not found: {segment_id}")

        content = db.query(Content).filter(Content.id == segment.content_id).first()
        if not content:
            raise HTTPException(status_code=404, detail=f"Content not found for segment: {segment_id}")

        content_id = content.content_id  # YouTube ID
        start_time = segment.start_time
        end_time = segment.end_time

        logger.info(f"[MEDIA] ✓ Resolved segment {segment_id} -> content_id={content_id}, time={start_time:.2f}-{end_time:.2f}s")
    finally:
        db.close()

    # Delegate to streaming function
    if stream:
        return StreamingResponse(
            _stream_media_with_transcription(
                content_id=content_id,
                segment_id=segment_id,
                start_time=start_time,
                end_time=end_time,
                media_type=media_type,
                format=format,
                transcribe=transcribe,
                transcription_model=transcription_model,
                language=language,
                speaker_labels=speaker_labels,
                enable_translation=enable_translation,
                force_retranscribe=force_retranscribe
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    else:
        # Non-streaming mode - return JSON (not commonly used, but supported)
        raise HTTPException(status_code=400, detail="Non-streaming mode not supported for segment endpoint. Use stream=true")


@router.get("/content/{content_id}")
async def get_media(
    request: Request,
    content_id: str,
    segment_id: Optional[int] = Query(None, description="EmbeddingSegment ID for fast lookup (optional, falls back to timing)"),
    start_time: Optional[float] = Query(None, description="Start time in seconds"),
    end_time: Optional[float] = Query(None, description="End time in seconds"),
    media_type: str = Query("auto", description="Media type: 'video', 'audio', or 'auto' (default: auto - returns video if available, else audio)"),
    format: str = Query("mp4", description="Output format (mp4 or webm for video)"),

    # NEW: Transcription parameters (AssemblyAI disabled - see DISABLED_ASSEMBLYAI)
    transcribe: bool = Query(False, description="Enable AssemblyAI transcription (currently disabled)"),
    transcription_provider: str = Query("assemblyai", description="Transcription provider (only 'assemblyai' supported)"),
    transcription_model: str = Query("best", description="Transcription model"),
    language: Optional[str] = Query(None, description="Language code (e.g., 'en', 'fr') - auto-detect if None"),
    speaker_labels: bool = Query(True, description="Enable speaker diarization"),
    enable_translation: bool = Query(True, description="Enable translation to en and fr"),
    force_retranscribe: bool = Query(False, description="Force re-transcription (bypass cache)"),

    # NEW: Streaming mode
    stream: bool = Query(False, description="Enable SSE streaming mode for progressive loading")
):
    """
    Unified media endpoint with optional transcription.

    **Basic Usage (media only):**
    ```
    GET /api/media/content/{content_id}?start_time=10&end_time=30
    ```

    **With Transcription:**
    ```
    GET /api/media/content/{content_id}?start_time=10&end_time=30&transcribe=true
    ```

    **Streaming Mode (SSE):**
    ```
    GET /api/media/content/{content_id}?start_time=10&end_time=30&transcribe=true&stream=true
    ```

    Query Parameters:
        media_type:
            - 'auto' (default): Returns video if available, otherwise audio
            - 'video': Returns only video (404 if not available)
            - 'audio': Returns only audio
        format: Output format for video ('mp4' or 'webm')
        transcribe: Enable AssemblyAI transcription (default: false)
        stream: Enable SSE streaming for progressive loading (default: false)

    Returns:
        - If stream=false: JSON with media_url and optional transcription
        - If stream=true: SSE stream with media_ready and transcript_ready events
    """
    request_start = time.time()

    # Validate content_id to prevent path traversal
    validate_content_id(content_id)

    logger.info(f"[MEDIA] Request: content_id={content_id}, media_type={media_type}, start={start_time}, end={end_time}, format={format}, transcribe={transcribe}, stream={stream}")

    if media_type not in ['auto', 'video', 'audio']:
        raise HTTPException(status_code=400, detail="media_type must be 'auto', 'video', or 'audio'")

    if format not in ['mp4', 'webm']:
        raise HTTPException(status_code=400, detail="Format must be 'mp4' or 'webm'")

    if transcribe and transcription_provider != "assemblyai":
        raise HTTPException(status_code=400, detail="Only 'assemblyai' transcription provider is supported")

    # DISABLED_ASSEMBLYAI: AssemblyAI transcription is disabled - use existing transcripts only
    # To re-enable, remove this block and the service will generate new transcriptions
    if transcribe:
        logger.info(f"[MEDIA] AssemblyAI transcription disabled - will use cached/existing transcripts only")
        transcribe = False

    # Streaming mode
    if stream:
        return StreamingResponse(
            _stream_media_with_transcription(
                content_id=content_id,
                segment_id=segment_id,
                start_time=start_time,
                end_time=end_time,
                media_type=media_type,
                format=format,
                transcribe=transcribe,
                transcription_model=transcription_model,
                language=language,
                speaker_labels=speaker_labels,
                enable_translation=enable_translation,
                force_retranscribe=force_retranscribe
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    async def process():
        try:
            # Determine what media to return
            video_path = get_video_path(content_id)
            has_video = video_path is not None

            # Decide based on media_type parameter
            if media_type == 'audio':
                # User explicitly wants audio only
                return_video = False
            elif media_type == 'video':
                # User explicitly wants video only
                if not has_video:
                    raise HTTPException(status_code=404, detail=f"Video not found for content_id: {content_id}")
                return_video = True
            else:  # media_type == 'auto'
                # Auto: prefer video if available, fallback to audio
                return_video = has_video

            if return_video:
                # Return video (includes audio track)
                logger.info(f"[MEDIA] Serving video for {content_id} from {video_path}")

                if start_time is not None or end_time is not None:
                    try:
                        extract_start = time.time()
                        s3_url = get_s3_presigned_url(video_path)
                        video_data = await extract_video_segment_ffmpeg(s3_url, start_time, end_time, output_format=format)
                        extract_duration_ms = (time.time() - extract_start) * 1000
                        logger.info(f"[MEDIA] ✓ Extracted video segment for {content_id}: {len(video_data) / 1024 / 1024:.2f}MB in {extract_duration_ms:.0f}ms")
                    except ValueError as e:
                        logger.warning(f"Invalid time range for {content_id}: {e}")
                        raise HTTPException(status_code=400, detail=str(e))
                    except Exception as e:
                        logger.error(f"Failed to extract video segment for {content_id}: {e}")
                        raise HTTPException(status_code=500, detail=f"Failed to extract video segment: {str(e)}")
                else:
                    # Redirect to presigned URL for full video
                    s3_url = get_s3_presigned_url(video_path, expiration=3600)
                    return RedirectResponse(url=s3_url, status_code=302)

                total_duration_ms = (time.time() - request_start) * 1000
                logger.info(f"[MEDIA] ✓ Returning video response for {content_id}: {len(video_data)/1024/1024:.2f}MB (total: {total_duration_ms:.0f}ms)")

                content_type = "video/webm" if format == "webm" else "video/mp4"
                return StreamingResponse(
                    BytesIO(video_data),
                    media_type=content_type,
                    headers={
                        "Content-Disposition": f"inline; filename={content_id}.{format}",
                        "Content-Length": str(len(video_data)),
                        "Accept-Ranges": "bytes",
                        "X-Media-Type": "video",
                        "X-Cache": "NONE",
                        "X-Processing-Time-Ms": str(int(total_duration_ms))
                    }
                )
            else:
                # No video - return audio
                logger.info(f"[MEDIA] No video found for {content_id}, serving audio")
                audio_path, compressed_path = get_audio_path(content_id)

                if compressed_path and s3_storage.file_exists(compressed_path):
                    actual_path = compressed_path
                elif s3_storage.file_exists(audio_path):
                    actual_path = audio_path
                else:
                    logger.error(f"[MEDIA] ✗ No media found for content_id: {content_id}")
                    raise HTTPException(status_code=404, detail=f"No media found for content_id: {content_id}")

                if start_time is not None or end_time is not None:
                    try:
                        extract_start = time.time()
                        s3_url = get_s3_presigned_url(actual_path)
                        audio_data = await extract_audio_segment_ffmpeg(s3_url, start_time, end_time)
                        extract_duration_ms = (time.time() - extract_start) * 1000
                        logger.info(f"[MEDIA] ✓ Extracted audio segment for {content_id}: {len(audio_data) / 1024 / 1024:.2f}MB in {extract_duration_ms:.0f}ms")
                    except ValueError as e:
                        logger.warning(f"Invalid time range for {content_id}: {e}")
                        raise HTTPException(status_code=400, detail=str(e))
                    except Exception as e:
                        logger.error(f"Failed to extract audio segment for {content_id}: {e}")
                        raise HTTPException(status_code=500, detail=f"Failed to extract audio segment: {str(e)}")
                else:
                    # Redirect to presigned URL for full audio
                    s3_url = get_s3_presigned_url(actual_path, expiration=3600)
                    return RedirectResponse(url=s3_url, status_code=302)

                total_duration_ms = (time.time() - request_start) * 1000
                logger.info(f"[MEDIA] ✓ Returning audio response for {content_id}: {len(audio_data)/1024/1024:.2f}MB (total: {total_duration_ms:.0f}ms)")

                return StreamingResponse(
                    BytesIO(audio_data),
                    media_type="audio/webm",
                    headers={
                        "Content-Disposition": f"inline; filename={content_id}.webm",
                        "Content-Length": str(len(audio_data)),
                        "X-Media-Type": "audio",
                        "X-Cache": "NONE",
                        "X-Processing-Time-Ms": str(int(total_duration_ms))
                    }
                )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing media request for {content_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing media: {str(e)}")

    # Batch mode (non-streaming)
    result = await safe_process_request(process)

    # If transcription requested in batch mode, add it to the response
    if transcribe and not stream:
        try:
            transcript_data = await _get_or_generate_transcript(
                content_id=content_id,
                start_time=start_time,
                end_time=end_time,
                model=transcription_model,
                language=language,
                speaker_labels=speaker_labels,
                enable_translation=enable_translation,
                force=force_retranscribe
            )

            # Return JSON response with both media URL and transcription
            base_url = PUBLIC_API_URL if PUBLIC_API_URL else ""
            media_url = f"{base_url}/api/media/content/{content_id}"
            params = []
            if start_time is not None:
                params.append(f"start_time={start_time}")
            if end_time is not None:
                params.append(f"end_time={end_time}")
            params.append(f"media_type={media_type}")
            params.append(f"format={format}")
            media_url += "?" + "&".join(params) if params else ""

            processing_time_ms = (time.time() - request_start) * 1000

            return JSONResponse({
                'media_url': media_url,
                'media_type': media_type,
                'transcription': transcript_data['data'],
                'cached': transcript_data['cached'],
                'processing_time_ms': processing_time_ms
            })

        except Exception as e:
            logger.error(f"Transcription error in batch mode: {e}", exc_info=True)
            # Return media without transcription on error
            return result

    return result


# Initialize storage on module load
initialize_storage()
