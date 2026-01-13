#!/usr/bin/env python3
"""
Stage 1: Data Loading
====================

First stage of the stitch pipeline that loads all necessary data from S3 or local storage.

Key Responsibilities:
- Load diarization.json with speaker segments
- Load and combine transcript chunks (transcript_words.json)
- Load and decompress audio files (WAV, Opus, MP3)
- Filter out SPEAKER_99 (silence/noise segments)
- Apply time range filtering if specified
- Handle compressed files (.gz, .opus, .mp3)

Input Files:
- diarization.json or diarization.json.gz
- chunks/N/transcript_words.json (multiple chunks)
- audio.wav, audio.opus, or audio.mp3

Output:
- Loaded diarization data with SPEAKER_99 filtered
- Combined transcript segments with adjusted timestamps
- Path to audio file (optional, for later stages)

Methods:
- load_stage(): Main entry point called by stitch pipeline
- _load_json_with_compression_support(): Loads JSON files with .gz support
- _download_and_decompress_audio(): Handles audio file loading/conversion
- _discover_chunks_from_s3/local(): Finds all transcript chunks
- _load_chunks_from_local/s3(): Loads and combines chunks with timestamp adjustment
"""

import sys
import json
import gzip
import argparse
import logging
import time
import tempfile
import subprocess
from pathlib import Path

from src.utils.paths import get_project_root
from typing import Optional, Tuple, Dict, Any, List

# Add the project root to Python path
sys.path.append(str(get_project_root()))

from src.utils.logger import setup_worker_logger
from src.storage.s3_utils import S3Storage, S3StorageConfig, create_s3_storage_from_config
from src.database.models import Content, ContentChunk
from src.database.session import get_session

logger = setup_worker_logger('stitch')
logger.setLevel(logging.INFO)


def _check_transcription_completeness(content_id: str) -> Dict[str, Any]:
    """
    Check if all chunks for the given content are fully transcribed.

    Args:
        content_id: Content ID to check

    Returns:
        Dictionary with:
        - is_complete: True if all chunks are transcribed
        - total_chunks: Total number of chunks
        - completed_chunks: Number of completed chunks
        - pending_chunks: List of chunk indices that are not completed
        - status_summary: Dict of status -> count
        - error: Error message if check failed
    """
    try:
        with get_session() as session:
            # Get content record by content_id (string)
            content = session.query(Content).filter_by(content_id=content_id).first()
            if not content:
                return {
                    'is_complete': False,
                    'error': f'Content {content_id} not found in database'
                }

            # Check content-level transcription flag first
            if not content.is_transcribed:
                logger.debug(f"[{content_id}] Content is_transcribed=False, checking individual chunks")
            
            # Get all chunks for this content using the integer primary key
            chunks = session.query(ContentChunk).filter_by(
                content_id=content.id
            ).order_by(ContentChunk.chunk_index).all()
            
            if not chunks:
                return {
                    'is_complete': False,
                    'total_chunks': 0,
                    'completed_chunks': 0,
                    'pending_chunks': [],
                    'status_summary': {},
                    'error': f'No chunks found for content {content_id}'
                }
            
            # Analyze transcription status
            total_chunks = len(chunks)
            completed_chunks = []
            pending_chunks = []
            status_summary = {}
            
            for chunk in chunks:
                status = chunk.transcription_status or 'unknown'
                status_summary[status] = status_summary.get(status, 0) + 1
                
                if chunk.transcription_status == 'completed':
                    completed_chunks.append(chunk.chunk_index)
                else:
                    pending_chunks.append({
                        'chunk_index': chunk.chunk_index,
                        'status': status,
                        'attempts': chunk.transcription_attempts
                    })
            
            is_complete = len(completed_chunks) == total_chunks

            return {
                'is_complete': is_complete,
                'total_chunks': total_chunks,
                'completed_chunks': len(completed_chunks),
                'completed_chunk_indices': completed_chunks,
                'pending_chunks': pending_chunks,
                'status_summary': status_summary,
                'content_is_transcribed_flag': content.is_transcribed
            }
            
    except Exception as e:
        logger.error(f"[{content_id}] Error checking transcription completeness: {str(e)}")
        return {
            'is_complete': False,
            'error': str(e)
        }


class NumpyJsonEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.str_):
            return str(obj)
        elif hasattr(obj, 'item'):  # For numpy scalars
            return obj.item()
        return super().default(obj)


def _load_json_with_compression_support(s3_storage: S3Storage,
                                      content_id: str,
                                      json_filename: str,
                                      temp_dir: Path,
                                      test_mode: bool = False,
                                      cache_in_test_mode: bool = True,
                                      prefer_local: bool = True) -> Optional[Dict]:
    """
    Load a JSON file, checking local files first, then S3.
    
    Args:
        s3_storage: S3 storage instance
        content_id: Content ID
        json_filename: Name of the JSON file (e.g., 'diarization.json')
        temp_dir: Temporary directory for downloads
        test_mode: If True, use test mode caching
        cache_in_test_mode: If True and test_mode is True, cache files locally
        prefer_local: If True, check for local files first before S3
        
    Returns:
        Loaded JSON data or None if file not found
    """
    try:
        # Step 1: Check for local files first (if prefer_local is True)
        if prefer_local:
            # Define potential local file locations
            local_dirs = []
            
            # Always check test inputs directory
            test_inputs_dir = get_project_root() / "tests" / "content" / content_id / "inputs"
            local_dirs.append(test_inputs_dir)
            
            # Also check current working directory structure
            cwd_content_dir = Path.cwd() / "content" / content_id
            local_dirs.append(cwd_content_dir)
            
            # Check data directory structure  
            data_content_dir = Path.cwd() / "data" / "content" / content_id
            local_dirs.append(data_content_dir)
            
            for local_dir in local_dirs:
                if not local_dir.exists():
                    continue
                    
                local_path = local_dir / json_filename
                local_gz_path = local_dir / f"{json_filename}.gz"
                
                # Check for uncompressed file first
                if local_path.exists():
                    logger.debug(f"[{content_id}] Using local {json_filename} from {local_path}")
                    with open(local_path, 'r') as f:
                        data = json.load(f)
                    
                    # Apply SPEAKER_99 filtering for diarization data
                    if json_filename == "diarization.json" and isinstance(data, dict) and 'segments' in data:
                        original_count = len(data['segments'])
                        data['segments'] = [
                            seg for seg in data['segments'] 
                            if seg.get('speaker') != 'SPEAKER_99'
                        ]
                        filtered_count = len(data['segments'])
                        if filtered_count < original_count:
                            logger.debug(f"[{content_id}] Filtered out {original_count - filtered_count} SPEAKER_99 segments from local diarization")
                    
                    return data
                
                # Check for compressed file
                elif local_gz_path.exists():
                    logger.debug(f"[{content_id}] Using local compressed {json_filename}.gz from {local_gz_path}")
                    with gzip.open(local_gz_path, 'rt') as f:
                        data = json.load(f)
                    
                    # Apply SPEAKER_99 filtering for diarization data
                    if json_filename == "diarization.json" and isinstance(data, dict) and 'segments' in data:
                        original_count = len(data['segments'])
                        data['segments'] = [
                            seg for seg in data['segments'] 
                            if seg.get('speaker') != 'SPEAKER_99'
                        ]
                        filtered_count = len(data['segments'])
                        if filtered_count < original_count:
                            logger.debug(f"[{content_id}] Filtered out {original_count - filtered_count} SPEAKER_99 segments from local diarization")
                    
                    return data
            
            logger.debug(f"[{content_id}] No local {json_filename} found, checking S3")
        
        # Step 2: Check S3 (always happens if no local file or prefer_local is False)
        if test_mode and cache_in_test_mode:
            # Create test inputs directory for caching if it doesn't exist
            test_inputs_dir = get_project_root() / "tests" / "content" / content_id / "inputs"
            test_inputs_dir.mkdir(parents=True, exist_ok=True)
            cached_path = test_inputs_dir / json_filename
            cached_gz_path = test_inputs_dir / f"{json_filename}.gz"
            
        # Use S3Storage's flexible JSON reading methods
        s3_key = f"content/{content_id}/{json_filename}"

        if test_mode and cache_in_test_mode:
            # Download and cache using flexible method
            logger.debug(f"[{content_id}] Downloading {json_filename} from S3 with flexible method (caching)")
            if s3_storage.download_json_flexible(s3_key, str(cached_path)):
                logger.debug(f"[{content_id}] Downloaded and cached {json_filename}")
                with open(cached_path, 'r') as f:
                    data = json.load(f)

                # Special handling for diarization data - filter out SPEAKER_99
                if json_filename == "diarization.json" and isinstance(data, dict) and 'segments' in data:
                    original_count = len(data['segments'])
                    data['segments'] = [
                        seg for seg in data['segments']
                        if seg.get('speaker') != 'SPEAKER_99'
                    ]
                    filtered_count = len(data['segments'])
                    if filtered_count < original_count:
                        logger.debug(f"[{content_id}] Filtered out {original_count - filtered_count} SPEAKER_99 segments from diarization")

                return data
            else:
                logger.error(f"[{content_id}] Failed to download {json_filename} from S3")
                return None
        else:
            # Read directly from S3 into memory using flexible method
            logger.debug(f"[{content_id}] Reading {json_filename} from S3 with flexible method")
            data = s3_storage.read_json_flexible(s3_key)

            if data is None:
                logger.error(f"[{content_id}] Failed to read {json_filename} from S3")
                return None

            # Special handling for diarization data - filter out SPEAKER_99
            if json_filename == "diarization.json" and isinstance(data, dict) and 'segments' in data:
                original_count = len(data['segments'])
                data['segments'] = [
                    seg for seg in data['segments']
                    if seg.get('speaker') != 'SPEAKER_99'
                ]
                filtered_count = len(data['segments'])
                if filtered_count < original_count:
                    logger.debug(f"[{content_id}] Filtered out {original_count - filtered_count} SPEAKER_99 segments from diarization")

            return data
            
    except Exception as e:
        logger.error(f"[{content_id}] Error loading {json_filename}: {str(e)}")
        logger.debug(f"[{content_id}] JSON loading error details:", exc_info=True)
        return None


def _download_and_decompress_audio(s3_storage: S3Storage,
                                 content_id: str,
                                 output_wav_path: str,
                                 prefer_local: bool = True) -> Optional[str]:
    """
    Download audio, checking local files first, then S3.
    
    Args:
        s3_storage: S3 storage instance
        content_id: Content ID
        output_wav_path: Path where the WAV file should be saved
        prefer_local: If True, check for local files first before S3
        
    Returns:
        Path to the WAV file if successful, None if failed
    """
    try:
        # Step 1: Check for local audio files first (if prefer_local is True)
        if prefer_local:
            # Define potential local file locations
            local_dirs = []
            
            # Always check test inputs directory
            test_inputs_dir = get_project_root() / "tests" / "content" / content_id / "inputs"
            local_dirs.append(test_inputs_dir)
            
            # Also check current working directory structure
            cwd_content_dir = Path.cwd() / "content" / content_id
            local_dirs.append(cwd_content_dir)
            
            # Check data directory structure  
            data_content_dir = Path.cwd() / "data" / "content" / content_id
            local_dirs.append(data_content_dir)
            
            # Audio file formats to check for (in priority order)
            audio_formats = ["audio.wav", "audio.opus", "audio.mp3"]
            
            for local_dir in local_dirs:
                if not local_dir.exists():
                    continue
                    
                for audio_filename in audio_formats:
                    local_audio_path = local_dir / audio_filename
                    
                    if local_audio_path.exists():
                        logger.debug(f"[{content_id}] Found local audio file: {local_audio_path}")
                        
                        if audio_filename == "audio.wav":
                            # For WAV files, we can use them directly without copying
                            logger.debug(f"[{content_id}] Using local WAV file directly: {local_audio_path}")
                            return str(local_audio_path)
                        else:
                            # Convert compressed formats to WAV
                            format_type = audio_filename.split('.')[-1]  # 'opus' or 'mp3'
                            try:
                                # Ensure output directory exists
                                Path(output_wav_path).parent.mkdir(parents=True, exist_ok=True)
                                if _decompress_audio_to_wav(str(local_audio_path), output_wav_path, format_type):
                                    logger.debug(f"[{content_id}] Converted local {audio_filename} to WAV at {output_wav_path}")
                                    return output_wav_path
                                else:
                                    logger.warning(f"[{content_id}] Failed to convert local {audio_filename}")
                            except Exception as e:
                                logger.error(f"[{content_id}] Error converting {audio_filename}: {e}")
            
            logger.debug(f"[{content_id}] No local audio files found, checking S3")
        
        # Step 2: Check S3 for audio files
        # Check for various audio file formats in S3 (compressed first)
        audio_formats = [
            ("audio.opus", "opus"),
            ("audio.mp3", "mp3"),
            ("audio.wav", "wav")
        ]
        
        for filename, format_type in audio_formats:
            s3_audio_key = f"content/{content_id}/{filename}"
            
            if s3_storage.file_exists(s3_audio_key):
                logger.debug(f"[{content_id}] Found {filename} in S3")
                
                if format_type == "wav":
                    # Direct download for WAV
                    if s3_storage.download_file(s3_audio_key, output_wav_path):
                        return output_wav_path
                else:
                    # Download compressed file and convert to WAV
                    temp_compressed_path = str(Path(output_wav_path).parent / filename)
                    
                    if s3_storage.download_file(s3_audio_key, temp_compressed_path):
                        logger.debug(f"[{content_id}] Downloaded compressed {filename}")
                        
                        # Convert to WAV using appropriate decoder
                        if _decompress_audio_to_wav(temp_compressed_path, output_wav_path, format_type):
                            logger.debug(f"[{content_id}] Decompressed {filename} to WAV")
                            # Clean up compressed file
                            try:
                                Path(temp_compressed_path).unlink()
                            except OSError:
                                pass
                            return output_wav_path
                        else:
                            logger.error(f"[{content_id}] Failed to decompress {filename}")
                            # Clean up compressed file on failure
                            try:
                                Path(temp_compressed_path).unlink()
                            except OSError:
                                pass
        
        logger.warning(f"[{content_id}] No audio file found in S3 (checked: {[f[0] for f in audio_formats]})")
        return None
        
    except Exception as e:
        logger.error(f"[{content_id}] Error downloading and decompressing audio: {str(e)}")
        return None


def _decompress_audio_to_wav(compressed_path: str, wav_path: str, format_type: str) -> bool:
    """
    Decompress audio file to WAV format using appropriate tools.
    
    Args:
        compressed_path: Path to compressed audio file
        wav_path: Path where WAV file should be saved
        format_type: Type of compressed audio ('opus' or 'mp3')
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if format_type == "opus":
            # Try opusdec first, then FFmpeg
            if _try_opusdec_decoding(compressed_path, wav_path):
                return True
            elif _try_ffmpeg_opus_decoding(compressed_path, wav_path):
                return True
        elif format_type == "mp3":
            # Use FFmpeg for MP3 decoding
            if _try_ffmpeg_mp3_decoding(compressed_path, wav_path):
                return True
        
        logger.error(f"Failed to decompress {format_type} audio file")
        return False
        
    except Exception as e:
        logger.error(f"Error decompressing {format_type} audio: {e}")
        return False


def _try_opusdec_decoding(opus_path: str, wav_path: str) -> bool:
    """Try decoding using dedicated opusdec tool."""
    try:
        # Try full path first, then fall back to PATH
        opusdec_paths = ['/opt/homebrew/bin/opusdec', 'opusdec']
        
        for opusdec_cmd in opusdec_paths:
            try:
                cmd = [
                    opusdec_cmd,
                    '--rate', '16000',  # Decode to 16kHz for ML processing
                    opus_path,
                    wav_path
                ]
                
                logger.debug(f"Running opusdec: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0 and Path(wav_path).exists():
                    logger.debug(f"Successfully decoded Opus to WAV using opusdec")
                    return True
                else:
                    logger.debug(f"opusdec failed (cmd: {opusdec_cmd}): {result.stderr}")
                    continue  # Try next path
                    
            except FileNotFoundError:
                logger.debug(f"opusdec not found at: {opusdec_cmd}")
                continue
                
        return False
        
    except subprocess.TimeoutExpired:
        logger.error("opusdec timed out after 5 minutes")
        return False
    except Exception as e:
        logger.debug(f"opusdec error: {e}")
        return False


def _try_ffmpeg_opus_decoding(opus_path: str, wav_path: str) -> bool:
    """Try decoding using FFmpeg."""
    try:
        # Try full path first, then fall back to PATH
        ffmpeg_paths = ['/opt/homebrew/bin/ffmpeg', 'ffmpeg']
        
        for ffmpeg_cmd in ffmpeg_paths:
            try:
                cmd = [
                    ffmpeg_cmd, '-i', opus_path,
                    '-c:a', 'pcm_s16le',
                    '-ar', '16000',  # 16kHz for ML processing
                    '-y', wav_path
                ]
                
                logger.debug(f"Running FFmpeg decode: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0 and Path(wav_path).exists():
                    logger.debug(f"Successfully decoded Opus to WAV using FFmpeg")
                    return True
                else:
                    logger.debug(f"FFmpeg decode failed (cmd: {ffmpeg_cmd}): {result.stderr}")
                    continue  # Try next path
                    
            except FileNotFoundError:
                logger.debug(f"ffmpeg not found at: {ffmpeg_cmd}")
                continue
                
        return False
        
    except subprocess.TimeoutExpired:
        logger.error("FFmpeg timed out after 5 minutes")
        return False
    except Exception as e:
        logger.debug(f"FFmpeg decode error: {e}")
        return False


def _try_ffmpeg_mp3_decoding(mp3_path: str, wav_path: str) -> bool:
    """Try decoding MP3 using FFmpeg."""
    try:
        # Try full path first, then fall back to PATH
        ffmpeg_paths = ['/opt/homebrew/bin/ffmpeg', 'ffmpeg']
        
        for ffmpeg_cmd in ffmpeg_paths:
            try:
                cmd = [
                    ffmpeg_cmd, '-i', mp3_path,
                    '-c:a', 'pcm_s16le',
                    '-ar', '16000',  # 16kHz for ML processing
                    '-y', wav_path
                ]
                
                logger.debug(f"Running FFmpeg MP3 decode: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0 and Path(wav_path).exists():
                    logger.debug(f"Successfully decoded MP3 to WAV using FFmpeg")
                    return True
                else:
                    logger.debug(f"FFmpeg MP3 decode failed (cmd: {ffmpeg_cmd}): {result.stderr}")
                    continue  # Try next path
                    
            except FileNotFoundError:
                logger.debug(f"ffmpeg not found at: {ffmpeg_cmd}")
                continue
                
        return False
        
    except Exception as e:
        logger.debug(f"FFmpeg MP3 decode error: {e}")
        return False


def _discover_chunks_from_s3(s3_storage: S3Storage, content_id: str) -> List[Dict]:
    """Discover chunk structure by listing S3 files."""
    try:
        # List all chunk transcript files (both compressed and uncompressed)
        prefix = f"content/{content_id}/chunks/"
        s3_objects_uncompressed = s3_storage.list_s3_objects(prefix=prefix, suffix="transcript_words.json")
        s3_objects_compressed = s3_storage.list_s3_objects(prefix=prefix, suffix="transcript_words.json.gz")
        s3_objects = s3_objects_uncompressed + s3_objects_compressed
        
        chunks = []
        for obj_key in s3_objects:
            # Extract chunk index from path like "content/ID/chunks/0/transcript_words.json" or "content/ID/chunks/0/transcript_words.json.gz"
            parts = obj_key.replace(prefix, "").split("/")
            if len(parts) >= 2:
                try:
                    chunk_index = int(parts[0])
                    # Estimate timing (assume 300s chunks with 2s overlap)
                    start_time = chunk_index * 298.0  # 300 - 2 overlap
                    chunks.append({
                        "index": chunk_index,
                        "start_time": start_time,
                        "end_time": start_time + 300.0,
                        "duration": 300.0,
                        "s3_key": obj_key
                    })
                except ValueError:
                    logger.warning(f"Could not parse chunk index from {obj_key}")
        
        # Sort by index
        chunks.sort(key=lambda x: x["index"])
        logger.debug(f"[{content_id}] Discovered {len(chunks)} chunks from S3 listing")
        
        return chunks
            
    except Exception as e:
        logger.error(f"Error discovering chunks from S3: {str(e)}")
        return []


def _discover_chunks_from_local(content_id: str) -> List[Dict]:
    """Discover chunk structure from local files."""
    try:
        # Define potential local chunk directories
        local_dirs = []
        
        # Check test inputs directory
        test_chunks_dir = get_project_root() / "tests" / "content" / content_id / "chunks"
        local_dirs.append(test_chunks_dir)
        
        # Check current working directory structure
        cwd_chunks_dir = Path.cwd() / "content" / content_id / "chunks"
        local_dirs.append(cwd_chunks_dir)
        
        # Check data directory structure  
        data_chunks_dir = Path.cwd() / "data" / "content" / content_id / "chunks"
        local_dirs.append(data_chunks_dir)
        
        for chunks_dir in local_dirs:
            if not chunks_dir.exists():
                continue
                
            chunks = []
            # Look for numbered chunk directories
            for chunk_path in chunks_dir.iterdir():
                if chunk_path.is_dir() and chunk_path.name.isdigit():
                    chunk_index = int(chunk_path.name)
                    transcript_file = chunk_path / "transcript_words.json"
                    transcript_file_gz = chunk_path / "transcript_words.json.gz"
                    
                    # Check for either uncompressed or compressed version
                    if transcript_file.exists() or transcript_file_gz.exists():
                        # Use compressed version if available, otherwise uncompressed
                        actual_file = transcript_file_gz if transcript_file_gz.exists() else transcript_file
                        
                        # Estimate timing (assume 300s chunks with 2s overlap)
                        start_time = chunk_index * 298.0  # 300 - 2 overlap
                        chunks.append({
                            "index": chunk_index,
                            "start_time": start_time,
                            "end_time": start_time + 300.0,
                            "duration": 300.0,
                            "local_path": str(actual_file)
                        })
            
            if chunks:
                # Sort by index
                chunks.sort(key=lambda x: x["index"])
                logger.debug(f"[{content_id}] Discovered {len(chunks)} local chunks from {chunks_dir}")
                return chunks
        
        return []
            
    except Exception as e:
        logger.error(f"Error discovering local chunks: {str(e)}")
        return []


def _load_chunks_from_local(chunks: List[Dict], content_id: str) -> List[Dict]:
    """Load and combine local transcript chunks."""
    try:
        all_segments = []
        
        for chunk in chunks:
            chunk_index = chunk["index"]
            chunk_start_offset = chunk["start_time"]  # Absolute start time for this chunk
            local_path = chunk["local_path"]
            
            try:
                # Handle both compressed and uncompressed transcript files
                if local_path.endswith('.gz'):
                    import gzip
                    with gzip.open(local_path, 'rt', encoding='utf-8') as f:
                        transcript = json.load(f)
                else:
                    with open(local_path, 'r') as f:
                        transcript = json.load(f)
                
                # Process segments from this chunk and adjust timestamps
                segments = transcript.get('segments', [])
                for segment in segments:
                    # Create adjusted segment with correct absolute timestamps
                    adjusted_segment = {
                        'text': segment.get('text', '').strip(),
                        'start': segment.get('start', 0.0) + chunk_start_offset,
                        'end': segment.get('end', 0.0) + chunk_start_offset,
                        'words': []
                    }
                    
                    # Adjust word-level timestamps if available
                    raw_words = segment.get('words', [])
                    if raw_words and isinstance(raw_words, list):
                        adjusted_words = []
                        for word_item in raw_words:
                            if isinstance(word_item, dict) and 'word' in word_item:
                                adjusted_word = {
                                    'word': word_item['word'],
                                    'start': word_item.get('start', segment.get('start', 0.0)) + chunk_start_offset,
                                    'end': word_item.get('end', segment.get('end', 0.0)) + chunk_start_offset
                                }
                                # Preserve additional word properties if they exist
                                for key in ['confidence', 'probability', 'prob']:
                                    if key in word_item:
                                        adjusted_word[key] = word_item[key]
                                adjusted_words.append(adjusted_word)
                        adjusted_segment['words'] = adjusted_words
                    
                    all_segments.append(adjusted_segment)
                
                logger.debug(f"[{content_id}:{chunk_index}] Loaded {len(segments)} segments from local chunk with offset {chunk_start_offset}s")
                
            except json.JSONDecodeError as e:
                logger.error(f"[{content_id}:{chunk_index}] Error decoding local JSON: {e}")
            except Exception as e:
                logger.error(f"[{content_id}:{chunk_index}] Error processing local transcript: {e}")
        
        if not all_segments:
            logger.warning(f"[{content_id}] No segments found after loading local chunks")
            return []
        
        # Sort by start time (stage3 will handle timing cleanup)
        all_segments.sort(key=lambda x: x.get('start', 0.0))
        logger.debug(f"[{content_id}] Loaded {len(all_segments)} segments from {len(chunks)} local chunks with proper timestamp offsets")
        
        return all_segments
        
    except Exception as e:
        logger.error(f"Error loading local transcript chunks: {str(e)}")
        return []


def _load_chunks_from_local_or_s3(s3_storage: S3Storage,
                                 content_id: str,
                                 temp_dir: Path,
                                 test_mode: bool = False,
                                 prefer_local: bool = True) -> List[Dict]:
    """Load and combine transcript chunks from local files or S3."""
    try:
        chunks = []
        
        # Step 1: Check for local chunks first (if prefer_local is True)
        if prefer_local:
            chunks = _discover_chunks_from_local(content_id)
            if chunks:
                logger.debug(f"[{content_id}] Using local chunks")
                return _load_chunks_from_local(chunks, content_id)
            else:
                logger.debug(f"[{content_id}] No local chunks found, checking S3")
        
        # Step 2: Fall back to S3 chunks
        chunks = _discover_chunks_from_s3(s3_storage, content_id)
        if not chunks:
            logger.error(f"[{content_id}] No chunks discovered in S3")
            return []
        
        # Load from S3 chunks (existing logic)
        return _load_chunks_from_s3_impl(s3_storage, content_id, temp_dir, chunks)
    
    except Exception as e:
        logger.error(f"Error loading transcript chunks: {str(e)}")
        logger.debug("Chunk loading error details:", exc_info=True)
        return []


def _load_chunks_from_s3_impl(s3_storage: S3Storage,
                             content_id: str,
                             temp_dir: Path,
                             chunks: List[Dict]) -> List[Dict]:
    """Implementation for loading chunks from S3."""
    try:
        # Load each chunk
        all_segments = []
        for chunk in chunks:
            chunk_index = chunk["index"]
            chunk_start_offset = chunk["start_time"]  # Absolute start time for this chunk
            s3_key = chunk["s3_key"]
            temp_path = temp_dir / f"transcript_{chunk_index}.json"

            if s3_storage.download_file(s3_key, str(temp_path)):
                try:
                    # Handle both compressed and uncompressed transcript files
                    if s3_key.endswith('.gz'):
                        import gzip
                        with gzip.open(temp_path, 'rt', encoding='utf-8') as f:
                            transcript = json.load(f)
                    else:
                        with open(temp_path, 'r') as f:
                            transcript = json.load(f)
                    
                    # Process segments from this chunk and adjust timestamps
                    segments = transcript.get('segments', [])
                    for segment in segments:
                        # Create adjusted segment with correct absolute timestamps
                        adjusted_segment = {
                            'text': segment.get('text', '').strip(),
                            'start': segment.get('start', 0.0) + chunk_start_offset,
                            'end': segment.get('end', 0.0) + chunk_start_offset,
                            'words': []
                        }
                        
                        # Adjust word-level timestamps if available
                        raw_words = segment.get('words', [])
                        if raw_words and isinstance(raw_words, list):
                            adjusted_words = []
                            for word_item in raw_words:
                                if isinstance(word_item, dict) and 'word' in word_item:
                                    adjusted_word = {
                                        'word': word_item['word'],
                                        'start': word_item.get('start', segment.get('start', 0.0)) + chunk_start_offset,
                                        'end': word_item.get('end', segment.get('end', 0.0)) + chunk_start_offset
                                    }
                                    # Preserve additional word properties if they exist
                                    for key in ['confidence', 'probability', 'prob']:
                                        if key in word_item:
                                            adjusted_word[key] = word_item[key]
                                    adjusted_words.append(adjusted_word)
                            adjusted_segment['words'] = adjusted_words
                        
                        all_segments.append(adjusted_segment)
                    
                    logger.debug(f"[{content_id}:{chunk_index}] Loaded {len(segments)} segments with chunk offset {chunk_start_offset}s")
                    
                except json.JSONDecodeError as e:
                    logger.error(f"[{content_id}:{chunk_index}] Error decoding JSON: {e}")
                except Exception as e:
                    logger.error(f"[{content_id}:{chunk_index}] Error processing transcript: {e}")
                finally:
                    # Clean up temp file
                    if temp_path.exists():
                        try:
                            temp_path.unlink()
                        except OSError:
                            pass
            else:
                logger.warning(f"[{content_id}:{chunk_index}] Failed to download transcript from {s3_key}")

        if not all_segments:
            logger.warning(f"[{content_id}] No segments found after loading chunks")
            return []

        # Sort by start time (stage3 will handle timing cleanup)
        all_segments.sort(key=lambda x: x.get('start', 0.0))
        logger.debug(f"[{content_id}] Loaded {len(all_segments)} segments from {len(chunks)} chunks with proper timestamp offsets")

        return all_segments

    except Exception as e:
        logger.error(f"Error loading transcript chunks: {str(e)}")
        logger.debug("Chunk loading error details:", exc_info=True)
        return []


async def load_stage(content_id: str,
                    test_mode: bool = False,
                    time_range: Optional[Tuple[float, float]] = None,
                    s3_storage: Optional[S3Storage] = None,
                    prefer_local: bool = True) -> Dict[str, Any]:
    """
    Main entry point for Stage 1: Load raw data from storage.
    
    This is the primary method called by the stitch pipeline. It orchestrates loading
    of all required data files, handling both local and S3 storage, compression formats,
    and various audio codecs.
    
    Args:
        content_id: Content ID to process (e.g., "Bdb001")
        test_mode: If True, saves outputs locally for debugging and caches downloaded files
        time_range: Optional (start, end) tuple in seconds for processing a specific time range
        s3_storage: Optional S3Storage instance. If None, creates one from environment variables
        prefer_local: If True, checks local directories before S3 (speeds up testing)
        
    Returns:
        Dictionary containing:
        - status: 'success' or 'error'
        - data: Dict with diarization_data, transcript_data, and audio_path
        - stats: Processing statistics (segments loaded, time spans, duration)
        - error: Error message if status is 'error'
        
    Example:
        result = await load_stage("Bdb001", test_mode=True, time_range=(60.0, 120.0))
        if result['status'] == 'success':
            diarization = result['data']['diarization_data']
            transcript = result['data']['transcript_data']
    """
    start_time = time.time()
    
    logger.info(f"[{content_id}] Starting Stage 1: Data Loading")
    logger.info(f"[{content_id}] Mode: {'test' if test_mode else 'production'}")
    if time_range:
        logger.info(f"[{content_id}] Time range: {time_range[0]:.1f}s - {time_range[1]:.1f}s")
    
    result = {
        'status': 'pending',
        'content_id': content_id,
        'test_mode': test_mode,
        'time_range': time_range,
        'stage': 'data_loading',
        'data': {
            'diarization_data': None,
            'transcript_data': None,
            'audio_path': None
        },
        'stats': {},
        'error': None
    }
    
    try:
        # Step 1: Check transcription completeness before proceeding
        logger.info(f"[{content_id}] Checking transcription completeness...")
        transcription_status = _check_transcription_completeness(content_id)
        
        if 'error' in transcription_status:
            raise ValueError(f"Failed to check transcription status: {transcription_status['error']}")
        
        if not transcription_status['is_complete']:
            # Log detailed status information
            pending = transcription_status['pending_chunks']
            status_summary = transcription_status['status_summary']
            logger.error(f"[{content_id}] Transcription not complete:")
            logger.error(f"[{content_id}]   Total chunks: {transcription_status['total_chunks']}")
            logger.error(f"[{content_id}]   Completed: {transcription_status['completed_chunks']}")
            logger.error(f"[{content_id}]   Status summary: {status_summary}")
            if pending:
                logger.error(f"[{content_id}]   Incomplete chunks: {[p['chunk_index'] for p in pending]}")
            
            raise ValueError(
                f"Cannot proceed with stitch: transcription incomplete. "
                f"Completed: {transcription_status['completed_chunks']}/{transcription_status['total_chunks']} chunks. "
                f"Pending chunks: {[p['chunk_index'] for p in pending]} "
                f"Status summary: {status_summary}"
            )
        
        logger.info(f"[{content_id}] ✓ All {transcription_status['total_chunks']} chunks are transcribed")

        # Use provided S3 storage or create from env vars
        if s3_storage is None:
            s3_config = S3StorageConfig()
            s3_storage = S3Storage(s3_config)
        
        # Create temporary directory
        with tempfile.TemporaryDirectory(prefix="stitch_1_load_") as temp_dir:
            temp_path = Path(temp_dir)
            logger.debug(f"[{content_id}] Created temp directory: {temp_path}")
            
            # Load diarization data
            logger.debug(f"[{content_id}] Loading diarization data")
            diarization_data = _load_json_with_compression_support(
                s3_storage=s3_storage,
                content_id=content_id,
                json_filename="diarization.json",
                temp_dir=temp_path,
                test_mode=test_mode,
                cache_in_test_mode=True,
                prefer_local=prefer_local
            )
            
            # Load transcript chunks
            logger.debug(f"[{content_id}] Loading transcript chunks")
            transcript_segments = _load_chunks_from_local_or_s3(
                s3_storage=s3_storage,
                content_id=content_id,
                temp_dir=temp_path,
                test_mode=test_mode,
                prefer_local=prefer_local
            )
            
            # Load audio file (optional)
            logger.debug(f"[{content_id}] Loading audio file")
            audio_path = None
            if test_mode:
                # In test mode, save to test directory
                test_audio_dir = get_project_root() / "tests" / "content" / content_id / "inputs"
                test_audio_dir.mkdir(parents=True, exist_ok=True)
                audio_output_path = test_audio_dir / "audio.wav"
            else:
                # In production mode, save to a persistent temp directory that won't be auto-deleted
                # The parent stitch.py will handle cleanup
                persistent_temp_dir = Path("/tmp/stitch_audio") / content_id
                persistent_temp_dir.mkdir(parents=True, exist_ok=True)
                audio_output_path = persistent_temp_dir / "audio.wav"
            
            audio_path = _download_and_decompress_audio(
                s3_storage=s3_storage,
                content_id=content_id,
                output_wav_path=str(audio_output_path),
                prefer_local=prefer_local
            )
            
            logger.info(f"[{content_id}] Audio loading result: {audio_path}")
            if audio_path:
                logger.debug(f"[{content_id}] Audio file exists: {Path(audio_path).exists()}")
                logger.debug(f"[{content_id}] Audio file size: {Path(audio_path).stat().st_size if Path(audio_path).exists() else 'N/A'} bytes")
            
            # Validate loaded data
            if not diarization_data:
                raise ValueError("No diarization data found")

            # Check for empty diarization segments (e.g., music-only content with no speech)
            diar_segments = diarization_data.get('segments', []) if isinstance(diarization_data, dict) else diarization_data
            if not diar_segments or len(diar_segments) == 0:
                raise ValueError("No speaker segments in diarization - content may be music-only or have no detectable speech")

            if not transcript_segments:
                raise ValueError("No transcript data found")
            if not audio_path:
                raise ValueError("No audio file found. Audio is required for speaker embedding calculation.")
            
            # Apply time range filtering if specified
            if time_range:
                start_range, end_range = time_range
                
                # Filter diarization segments
                if isinstance(diarization_data, dict) and 'segments' in diarization_data:
                    original_segments = diarization_data['segments']
                    filtered_segments = [s for s in original_segments 
                                      if s.get('start', 0) < end_range and s.get('end', 0) > start_range]
                    diarization_data['segments'] = filtered_segments
                    logger.info(f"[{content_id}] Filtered diarization segments: {len(original_segments)} -> {len(filtered_segments)}")
                
                # Filter transcript segments
                original_count = len(transcript_segments)
                transcript_segments = [s for s in transcript_segments 
                                    if s.get('start', 0) < end_range and s.get('end', 0) > start_range]
                logger.info(f"[{content_id}] Filtered transcript segments: {original_count} -> {len(transcript_segments)}")
            
            # Extract original diarization speakers for database record creation
            original_diarization_speakers = set()
            if isinstance(diarization_data, dict) and 'segments' in diarization_data:
                for segment in diarization_data['segments']:
                    speaker = segment.get('speaker')
                    if speaker and speaker != 'SPEAKER_99':  # Already filtered out SPEAKER_99
                        original_diarization_speakers.add(speaker)
            
            # Store data and statistics
            result['data']['diarization_data'] = diarization_data
            result['data']['transcript_data'] = {'segments': transcript_segments}
            result['data']['audio_path'] = str(audio_path) if audio_path else None
            result['data']['original_diarization_speakers'] = sorted(original_diarization_speakers)
            
            stage_duration = time.time() - start_time
            result['stats'] = {
                'duration': stage_duration,
                'diarization_segments': len(diarization_data.get('segments', [])) if isinstance(diarization_data, dict) else len(diarization_data),
                'transcript_segments': len(transcript_segments),
                'diarization_time_span': _calculate_time_span(diarization_data.get('segments', []) if isinstance(diarization_data, dict) else diarization_data),
                'transcript_time_span': _calculate_time_span(transcript_segments),
                'audio_loaded': audio_path is not None,
                'transcription_verification': {
                    'total_chunks': transcription_status['total_chunks'],
                    'completed_chunks': transcription_status['completed_chunks'],
                    'status_summary': transcription_status['status_summary'],
                    'content_is_transcribed_flag': transcription_status['content_is_transcribed_flag']
                }
            }
            
            result['status'] = 'success'
            
            logger.info(f"[{content_id}] Stage 1 completed successfully in {stage_duration:.2f}s")
            logger.info(f"[{content_id}] Loaded {result['stats']['diarization_segments']} diarization segments")
            logger.info(f"[{content_id}] Loaded {result['stats']['transcript_segments']} transcript segments")
            if audio_path:
                logger.info(f"[{content_id}] Loaded audio file: {audio_path}")
            else:
                logger.warning(f"[{content_id}] No audio file loaded (optional)")
            
            # Save outputs in test mode
            if test_mode and result['status'] == 'success':
                test_output_dir = get_project_root() / "tests" / "content" / content_id / "stage_outputs"
                test_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Save stage result
                stage_result_path = test_output_dir / "1_load_result.json"
                with open(stage_result_path, 'w') as f:
                    json.dump(result, f, indent=2, cls=NumpyJsonEncoder)
                
                logger.debug(f"[{content_id}] Stage 1 test outputs saved to: {test_output_dir}")
            
            return result
            
    except Exception as e:
        logger.error(f"[{content_id}] Stage 1 failed: {str(e)}")
        logger.error(f"[{content_id}] Error details:", exc_info=True)
        
        result.update({
            'status': 'error',
            'error': str(e),
            'duration': time.time() - start_time
        })
        return result


def _calculate_time_span(data: List[Dict]) -> Dict[str, float]:
    """Calculate time span statistics for loaded data."""
    if not data:
        return {'start': 0.0, 'end': 0.0, 'duration': 0.0}
    
    # Handle both diarization and transcript data formats
    times = []
    for item in data:
        if 'start' in item and 'end' in item:
            times.extend([item['start'], item['end']])
        elif 'words' in item:  # Transcript segment with words
            for word in item['words']:
                if 'start' in word and 'end' in word:
                    times.extend([word['start'], word['end']])
    
    if not times:
        return {'start': 0.0, 'end': 0.0, 'duration': 0.0}
    
    start_time = min(times)
    end_time = max(times)
    
    return {
        'start': start_time,
        'end': end_time,
        'duration': end_time - start_time
    }


async def main():
    """Main entry point for Stage 1: Data Loading."""
    parser = argparse.ArgumentParser(description='Stage 1: Data Loading - Load diarization, transcript, and audio data')
    parser.add_argument('--content', required=True, help='Content ID to process')
    parser.add_argument('--test', action='store_true', help='Test mode: save outputs locally for debugging')
    parser.add_argument('--start', type=float, help='Start time in seconds for focused processing')
    parser.add_argument('--end', type=float, help='End time in seconds for focused processing')
    
    args = parser.parse_args()
    
    # Parse time range if provided
    time_range = None
    if args.start is not None and args.end is not None:
        time_range = (args.start, args.end)
        logger.info(f"Processing time range: {args.start:.1f}s - {args.end:.1f}s")
    elif args.start is not None or args.end is not None:
        logger.warning("Both --start and --end must be provided for time range filtering")
    
    try:
        # Execute stage
        result = await load_stage(
            content_id=args.content,
            test_mode=args.test,
            time_range=time_range
        )
        
        # Print result as JSON
        print(json.dumps(result, indent=2, cls=NumpyJsonEncoder))
        
        # Exit with appropriate status code
        exit_code = 0 if result['status'] == 'success' else 1
        sys.exit(exit_code)
        
    except Exception as e:
        logger.error(f"Unhandled error in Stage 1: {str(e)}")
        
        error_result = {
            'status': 'error',
            'content_id': args.content,
            'stage': 'data_loading',
            'error': str(e)
        }
        print(json.dumps(error_result, cls=NumpyJsonEncoder))
        sys.exit(1)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())