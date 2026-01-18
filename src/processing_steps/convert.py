#!/usr/bin/env python3
# Centralized environment setup (must be before other imports)
from src.utils.env_setup import setup_env
setup_env()

import sys
from pathlib import Path

from src.utils.paths import get_project_root
from src.utils.config import load_config
import asyncio
import logging
import json
import yaml
import torch
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
import tempfile
import shutil
import argparse
import uuid
import subprocess
import traceback
import time

# Add the project root to Python path
sys.path.append(str(get_project_root()))

# Load environment variables (only S3 vars needed for convert)
from dotenv import load_dotenv
load_dotenv(get_project_root() / '.env')

# Import required modules
from src.utils.logger import setup_worker_logger
from src.storage.s3_utils import S3StorageConfig, S3Storage
from src.storage.content_storage import ContentStorageManager
from src.utils.media_utils import get_audio_duration # Import the shared function

# Initialize logger but configure it later based on debug flag
logger = None

# --- Helper function for duration --- (Removed - now imported)

# --- Helper function for chunk plan --- (Implement if not imported)
def create_chunk_plan(total_duration: float, chunk_size: int, chunk_overlap: int) -> List[Dict]:
    """Calculate chunk start/end times."""
    chunks = []
    current_time = 0.0
    chunk_index = 0
    min_chunk_duration = 5.0  # Minimum duration for a standalone chunk
    
    while current_time < total_duration:
        start_time = current_time
        end_time = min(current_time + chunk_size, total_duration)
        duration = end_time - start_time
        
        # If this would be a tiny final chunk, extend the previous chunk instead
        if duration < min_chunk_duration:
            if chunks:
                # Extend the previous chunk to the end
                chunks[-1]['end'] = total_duration
                chunks[-1]['duration'] = chunks[-1]['end'] - chunks[-1]['start']
            else:
                # If this is the only chunk and it's small, keep it
                chunks.append({
                    'index': chunk_index,
                    'start': start_time,
                    'end': end_time,
                    'duration': duration
                })
            break
            
        chunks.append({
            'index': chunk_index,
            'start': start_time,
            'end': end_time,
            'duration': duration
        })
        
        # Calculate next chunk start, ensuring we don't exceed total_duration
        next_start = end_time - chunk_overlap
        if next_start >= total_duration:
            break
            
        current_time = next_start
        chunk_index += 1
        
    return chunks

class AudioExtractor:
    def __init__(self, debug_mode=False):
        """Initialize the audio extractor"""
        self.debug_mode = debug_mode
        self.corrupt_media_detected = False
        
        # Load config
        self.config = load_config()
            
        # Initialize S3 storage
        s3_config = S3StorageConfig(
            endpoint_url=self.config['storage']['s3']['endpoint_url'],
            access_key=self.config['storage']['s3']['access_key'],
            secret_key=self.config['storage']['s3']['secret_key'],
            bucket_name=self.config['storage']['s3']['bucket_name'],
            use_ssl=self.config['storage']['s3']['use_ssl']
        )
        self.s3_storage = S3Storage(s3_config)
        self.storage_manager = ContentStorageManager(self.s3_storage)
        
        # Set up temp directory
        # Use unique temp dir per run to simplify cleanup
        self.temp_dir = Path(tempfile.mkdtemp(prefix="audio_extractor_"))
        logger.info(f"Created temporary directory: {self.temp_dir}")

    def _extract_audio(self, content_id: str, input_path: Path, output_path: Path) -> bool:
        """Extract audio to 16kHz mono WAV using ffmpeg, intelligently checking format first."""
        try:
            logger.debug(f"[{content_id}] Checking audio properties for {input_path}")
            # Get the S3 key for the source file - we know it's the input_path's name
            s3_source_key = f"content/{content_id}/source{input_path.suffix}"
            
            # Check audio properties using ffprobe first to see if conversion needed
            audio_probe_cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'a:0',  # First audio stream
                '-show_entries', 'stream=codec_name,sample_rate,channels',
                '-of', 'json',
                str(input_path)
            ]

            if self.debug_mode:
                logger.info(f"[DEBUG] Running audio probe command: {' '.join(audio_probe_cmd)}")
                
            audio_probe_result = subprocess.run(audio_probe_cmd, capture_output=True, text=True, check=False) # check=False to handle errors

            # Check for invalid/corrupt file
            is_corrupt = False
            file_size = input_path.stat().st_size if input_path.exists() else 0
            
            if audio_probe_result.returncode != 0:
                if self.debug_mode:
                    logger.info(f"[DEBUG] ffprobe failed with return code: {audio_probe_result.returncode}")
                    logger.info(f"[DEBUG] ffprobe stderr: {audio_probe_result.stderr}")
                    logger.info(f"[DEBUG] ffprobe stdout: {audio_probe_result.stdout}")
                    
                    # Check if file exists
                    if not input_path.exists():
                        logger.info(f"[DEBUG] Input file does not exist: {input_path}")
                    else:
                        logger.info(f"[DEBUG] Input file exists with size: {file_size} bytes")
                        # Try more verbose ffprobe to get more info
                        verbose_probe = subprocess.run(
                            ['ffprobe', '-v', 'warning', str(input_path)], 
                            capture_output=True, text=True
                        )
                        logger.info(f"[DEBUG] Verbose ffprobe output: {verbose_probe.stderr}")
                
                # Check for specific corruption error patterns
                error_text = audio_probe_result.stderr.lower()
                if ("failed to find" in error_text and "frames" in error_text) or \
                   ("invalid data found" in error_text) or \
                   ("error opening input" in error_text) or \
                   ("partial file" in error_text):  # Truncated/incomplete download
                    # This is likely a corrupted or invalid media file
                    is_corrupt = True
                    # Check file size to confirm it's suspiciously small
                    if file_size < 50000:  # Less than 50KB is suspicious for any audio/video file
                        logger.error(f"[{content_id}] File appears to be corrupt or invalid media: size={file_size} bytes, probe error: {audio_probe_result.stderr}")
                        # Special error for task orchestrator to reset download flag
                        self.corrupt_media_detected = True

                        # Delete the corrupt source file from S3 - we know exactly which file it is
                        try:
                            logger.warning(f"[{content_id}] Deleting corrupt source file from S3: {s3_source_key}")
                            self.s3_storage.delete_file(s3_source_key)
                        except Exception as delete_err:
                            logger.error(f"[{content_id}] Failed to delete corrupt source file from S3: {delete_err}")

                        return False
                
                logger.warning(f"[{content_id}] ffprobe failed for {input_path}: {audio_probe_result.stderr}. Assuming conversion needed.")
                needs_conversion = True
                is_mp4 = False # Can't assume it's a pre-processed MP4 if ffprobe fails
            else:
                audio_info = json.loads(audio_probe_result.stdout)
                if not audio_info.get('streams'):
                    logger.error(f"[{content_id}] No audio streams found in file: {input_path}")
                    if self.debug_mode:
                        logger.info(f"[DEBUG] ffprobe found no streams. Full output: {audio_probe_result.stdout}")
                    
                    # If no streams found and file is small, mark as corrupt
                    if file_size < 50000:
                        logger.error(f"[{content_id}] File appears to be invalid media: no audio streams, size={file_size} bytes")
                        self.corrupt_media_detected = True
                        
                        # Delete the corrupt source file from S3 - we know exactly which file it is
                        try:
                            logger.warning(f"[{content_id}] Deleting corrupt source file from S3: {s3_source_key}")
                            self.s3_storage.delete_file(s3_source_key)
                        except Exception as delete_err:
                            logger.error(f"[{content_id}] Failed to delete corrupt source file from S3: {delete_err}")
                        
                        return False

                stream = audio_info['streams'][0]
                sample_rate = int(stream.get('sample_rate', 0))
                channels = int(stream.get('channels', 0))
                codec = stream.get('codec_name', '')
                is_mp4 = input_path.suffix.lower() == '.mp4'

                # Check if audio meets our requirements (16kHz mono PCM)
                is_correct_format = (
                    sample_rate == 16000 and
                    channels == 1 and
                    codec == 'pcm_s16le'
                )
                # Special handling for MP4 which might contain correct AAC/MP3 stream
                is_preprocessed_container = (
                    is_mp4 and
                    sample_rate == 16000 and
                    channels == 1 and
                    codec in ['aac', 'mp3'] # Common codecs for pre-processed MP4
                )

                needs_conversion = not (is_correct_format or is_preprocessed_container)
                logger.debug(f"[{content_id}] Input: {input_path.name}, Sample Rate: {sample_rate}, Channels: {channels}, Codec: {codec}, Needs Conversion: {needs_conversion}")

            if needs_conversion:
                 # Standard audio extraction and conversion
                logger.info(f"[{content_id}] Converting audio to 16kHz mono WAV: {input_path.name}")
                cmd = [
                    'ffmpeg', '-y',
                    '-i', str(input_path),
                    '-vn',  # No video
                    '-acodec', 'pcm_s16le',  # 16-bit PCM
                    '-ar', '16000',  # 16kHz sample rate
                    '-ac', '1',  # Mono
                    '-af', 'loudnorm', # Normalize audio levels
                    '-f', 'wav', # Force WAV format
                    str(output_path)
                ]
            elif is_preprocessed_container and is_mp4:
                 # For pre-processed MP4s containing 16k/mono AAC/MP3, extract directly to WAV without re-encoding audio
                logger.info(f"[{content_id}] Extracting compatible audio stream directly from MP4 to WAV: {input_path.name}")
                cmd = [
                    'ffmpeg', '-y',
                    '-i', str(input_path),
                    '-vn', # No video
                    '-c:a', 'pcm_s16le', # Convert container to WAV, keep audio codec properties
                    '-ar', '16000',
                    '-ac', '1',
                    str(output_path)
                ]
            else: # Already correct WAV format
                logger.info(f"[{content_id}] Input audio already in correct format (WAV 16kHz mono), processing with ffmpeg -c copy: {input_path.name}")
                # If the input IS the output path, skip copy
                if input_path.resolve() == output_path.resolve():
                    logger.debug(f"[{content_id}] Input and output paths are the same, skipping copy.")
                    return True
                # Use ffmpeg -c copy to ensure standard WAV container, even if codec is correct
                cmd = [
                    'ffmpeg', '-y',
                    '-i', str(input_path),
                    '-c', 'copy', # Copy audio stream without re-encoding
                    '-map', '0:a:0?', # Select only the first audio stream if it exists
                    '-acodec', 'pcm_s16le', # Ensure the output codec is explicitly set (safer)
                    str(output_path)
                ]

            logger.debug(f"[{content_id}] Running command: {' '.join(cmd)}")
            
            # Use more verbose ffmpeg output in debug mode
            if self.debug_mode:
                # Change from error to warning level for more info
                cmd[2] = 'warning' if cmd[1] == '-v' else cmd[2]
                logger.info(f"[DEBUG] Running ffmpeg command: {' '.join(cmd)}")
                
            # Run ffmpeg without text=True to get raw bytes output
            result = subprocess.run(cmd, capture_output=True, check=False)

            # Check if output file was created successfully first
            output_created = output_path.exists() and output_path.stat().st_size > 0
            
            if result.returncode != 0:
                # Decode stderr with error handling for non-UTF-8 characters
                error_output = result.stderr.decode('utf-8', errors='replace')
                
                # If output was created despite errors, treat as warning rather than failure
                if output_created:
                    logger.warning(f"[{content_id}] FFmpeg completed with warnings but output file was created: {error_output}")
                else:
                    logger.error(f"[{content_id}] FFmpeg error processing {input_path.name}: {error_output}")
                    
                    # Check for invalid media errors
                    error_lower = error_output.lower()
                    if ("error opening input" in error_lower or
                        "invalid data found" in error_lower or
                        "could not find codec parameters" in error_lower or
                        "partial file" in error_lower):  # Truncated/incomplete download
                        logger.error(f"[{content_id}] File appears to be invalid or corrupt media (partial/truncated file)")
                        self.corrupt_media_detected = True
                    
                    if self.debug_mode:
                        # Print detailed file info for debugging
                        logger.info(f"[DEBUG] Input file details for failed conversion:")
                        try:
                            file_info = subprocess.run(
                                ['file', '-b', str(input_path)], 
                                capture_output=True, text=True
                            )
                            logger.info(f"[DEBUG] File type: {file_info.stdout.strip()}")
                            
                            # Check if output directory exists and is writable
                            output_dir = output_path.parent
                            logger.info(f"[DEBUG] Output directory exists: {output_dir.exists()}")
                            logger.info(f"[DEBUG] Output directory writable: {os.access(output_dir, os.W_OK)}")
                            
                            # Try listing the first few bytes of the file for inspection
                            if input_path.exists() and input_path.stat().st_size > 0:
                                head_cmd = subprocess.run(
                                    ['hexdump', '-n', '64', '-C', str(input_path)],
                                    capture_output=True, text=True
                                )
                                logger.info(f"[DEBUG] File header (hex): {head_cmd.stdout}")
                        except Exception as e:
                            logger.info(f"[DEBUG] Error getting file details: {str(e)}")
                    
                    # Only return False if no output was created
                    return False

            # Validate the output file
            # Check for minimum valid WAV size (header is ~78 bytes, need actual audio data)
            MIN_VALID_WAV_SIZE = 1000  # WAV header + at least some audio data
            output_size = output_path.stat().st_size if output_path.exists() else 0

            if not output_path.exists() or output_size < MIN_VALID_WAV_SIZE:
                logger.error(f"[{content_id}] Output audio file is missing or too small after processing: {output_path} ({output_size} bytes)")

                # Check stderr for empty output indicator (truncated source with valid headers)
                error_output = result.stderr.decode('utf-8', errors='replace').lower() if result.stderr else ""
                if "output file is empty" in error_output or "nothing was encoded" in error_output:
                    logger.error(f"[{content_id}] Source file has valid headers but no actual media data (truncated download)")
                    self.corrupt_media_detected = True
                elif output_size > 0 and output_size < MIN_VALID_WAV_SIZE:
                    # Output exists but is just a header - source is likely truncated
                    logger.error(f"[{content_id}] Output is just a WAV header ({output_size} bytes) - source is likely truncated")
                    self.corrupt_media_detected = True

                if self.debug_mode:
                    logger.info(f"[DEBUG] Output file exists: {output_path.exists()}")
                    if output_path.exists():
                        logger.info(f"[DEBUG] Output file size: {output_size} bytes")
                return False

            # Verify the output file format in debug mode
            if self.debug_mode and output_path.exists():
                verify_cmd = [
                    'ffprobe',
                    '-v', 'warning',
                    '-select_streams', 'a:0',
                    '-show_entries', 'stream=codec_name,sample_rate,channels,duration',
                    '-of', 'json',
                    str(output_path)
                ]
                verify_result = subprocess.run(verify_cmd, capture_output=True, text=True, check=False)
                if verify_result.returncode == 0:
                    logger.info(f"[DEBUG] Verification of output file: {verify_result.stdout}")
                else:
                    logger.info(f"[DEBUG] Failed to verify output file: {verify_result.stderr}")

            logger.info(f"[{content_id}] Successfully processed audio to: {output_path.name}")
            return True

        except Exception as e:
            logger.error(f"[{content_id}] Error extracting audio from {input_path}: {str(e)}")
            logger.error(f"[{content_id}] Traceback: {traceback.format_exc()}\n")
            if self.debug_mode:
                # More detailed error information for debugging
                logger.info(f"[DEBUG] Exception type: {type(e).__name__}")
                logger.info(f"[DEBUG] Exception args: {e.args}")
                logger.info(f"[DEBUG] Full traceback: {traceback.format_exc()}")
            return False

    def _verify_audio_format(self, content_id: str, file_path: Path) -> bool:
        """Verify using ffprobe if the audio file is 16kHz mono pcm_s16le."""
        try:
            logger.debug(f"[{content_id}] Verifying audio format for {file_path}")
            audio_probe_cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'a:0',
                '-show_entries', 'stream=codec_name,sample_rate,channels',
                '-of', 'json',
                str(file_path)
            ]
            
            if self.debug_mode:
                logger.info(f"[DEBUG] Running verification command: {' '.join(audio_probe_cmd)}")
                
            result = subprocess.run(audio_probe_cmd, capture_output=True, text=True, check=False)

            if result.returncode != 0:
                logger.warning(f"[{content_id}] ffprobe failed during format verification: {result.stderr}")
                if self.debug_mode:
                    logger.info(f"[DEBUG] ffprobe verification return code: {result.returncode}")
                    logger.info(f"[DEBUG] ffprobe verification stderr: {result.stderr}")
                return False

            audio_info = json.loads(result.stdout)
            if not audio_info.get('streams'):
                logger.warning(f"[{content_id}] No audio streams found during format verification.")
                if self.debug_mode:
                    logger.info(f"[DEBUG] ffprobe found no streams in verification. Output: {result.stdout}")
                return False

            stream = audio_info['streams'][0]
            sample_rate = int(stream.get('sample_rate', 0))
            channels = int(stream.get('channels', 0))
            codec = stream.get('codec_name', '')

            is_valid = (sample_rate == 16000 and channels == 1 and codec == 'pcm_s16le')
            if is_valid:
                 logger.debug(f"[{content_id}] Audio format verified successfully.")
            else:
                 logger.warning(f"[{content_id}] Audio format verification failed. Found: Rate={sample_rate}, Channels={channels}, Codec={codec}")
            return is_valid

        except Exception as e:
            logger.error(f"[{content_id}] Error during audio format verification: {e}")
            if self.debug_mode:
                logger.info(f"[DEBUG] Verification exception: {traceback.format_exc()}")
            return False

    async def process_content(self, content_id: str) -> Dict:
        """Extract audio, split into chunks, and upload to S3"""
        start_time = time.time()
        content_temp_dir = self.temp_dir / content_id
        output_data = {
            'status': 'pending',
            'content_id': content_id,
            'audio_path': None,
            'chunk_plan': [],
            'error': None,
            'duration': None,
            'corrupt_media_detected': False  # Add flag to track corrupt media
        }

        local_source_path = None # Initialize to None
        local_audio_path = None # Initialize to None

        try:
            content_temp_dir.mkdir(exist_ok=True)
            logger.info(f"[{content_id}] Starting audio extraction process")

            # Standardized audio file path (local and S3)
            local_audio_path = content_temp_dir / 'audio.wav'
            s3_audio_key = f"content/{content_id}/audio.wav"

            # --- NEW: Check if conversion already completed ---
            logger.debug(f"[{content_id}] Checking if standardized audio already exists at {s3_audio_key}")
            if self.s3_storage.file_exists(s3_audio_key):
                logger.info(f"[{content_id}] Found existing standardized audio file {s3_audio_key}. Checking duration and chunks.")
                # Need to get duration. Download temporarily to check.
                temp_audio_check_path = content_temp_dir / f"check_{uuid.uuid4()}.wav"
                try:
                    if self.s3_storage.download_file(s3_audio_key, str(temp_audio_check_path)):
                        duration = get_audio_duration(str(temp_audio_check_path))
                        if duration and duration > 0:
                             logger.info(f"[{content_id}] Existing audio duration: {duration:.2f}s. Generating chunk plan.")
                             chunk_size = self.config['processing']['chunk_size']
                             chunk_overlap = self.config['processing']['chunk_overlap']
                             chunk_plan = create_chunk_plan(duration, chunk_size, chunk_overlap)
                             logger.info(f"[{content_id}] Checking existence of {len(chunk_plan)} chunks in S3.")

                             # Check if all corresponding chunk files exist in S3
                             all_chunks_exist = True
                             for chunk_info in chunk_plan:
                                 chunk_paths = self.storage_manager.get_chunk_paths(content_id, chunk_info['index'])
                                 chunk_s3_key = chunk_paths['audio']
                                 if not self.s3_storage.file_exists(chunk_s3_key):
                                     logger.warning(f"[{content_id}] Missing chunk file in S3: {chunk_s3_key}. Proceeding with conversion.")
                                     all_chunks_exist = False
                                     break # No need to check further chunks

                             if all_chunks_exist:
                                 logger.info(f"[{content_id}] All {len(chunk_plan)} chunks found in S3. Skipping conversion.")
                                 # Prepare skipped output data
                                 output_data['status'] = 'skipped' # Use 'skipped' status
                                 output_data['reason'] = 'already_converted'
                                 output_data['duration'] = duration
                                 output_data['standard_audio_key'] = s3_audio_key
                                 output_data['audio_path'] = s3_audio_key
                                 output_data['chunk_plan'] = [{
                                     'index': c['index'],
                                     'start_time': c['start'],
                                     'end_time': c['end'],
                                     'duration': c['duration'],
                                     'extraction_status': 'completed' # Mark all as completed
                                 } for c in chunk_plan]
                                 output_data['total_chunks'] = len(chunk_plan)
                                 output_data['chunks_created'] = len(chunk_plan)
                                 output_data['chunks_failed'] = 0
                                 return output_data # Return the skipped result immediately
                        else:
                             logger.warning(f"[{content_id}] Could not get valid duration from existing audio {s3_audio_key}. Proceeding with conversion.")
                    else:
                         logger.warning(f"[{content_id}] Failed to download existing audio {s3_audio_key} for verification. Proceeding with conversion.")
                except Exception as check_err:
                     logger.warning(f"[{content_id}] Error checking existing audio {s3_audio_key}: {check_err}. Proceeding with conversion.")
                finally:
                     if temp_audio_check_path.exists():
                         try: temp_audio_check_path.unlink()
                         except OSError as e: logger.warning(f"[{content_id}] Failed to delete temp audio check file {temp_audio_check_path}: {e}")
            else:
                 logger.debug(f"[{content_id}] Standardized audio {s3_audio_key} not found. Proceeding with conversion.")
            # --- END NEW ---

            # --- NEW: Check for source file dependency ---\n
            # Find the source file key by checking common extensions
            s3_source_key = None
            source_exists = False
            extensions = ['.mp3', '.mp4', '.wav', '.m4a', '.webm', '.mkv', '.avi', '.mov']
            base_key = f"content/{content_id}/source"
            
            for ext in extensions:
                potential_key = f"{base_key}{ext}"
                if self.s3_storage.file_exists(potential_key):
                    s3_source_key = potential_key
                    source_exists = True
                    break
            
            # Fallback: check without extension
            if not source_exists and self.s3_storage.file_exists(base_key):
                s3_source_key = base_key
                source_exists = True

            if not source_exists:
                error_msg = f"Missing input file: Source file (content/{content_id}/source.*) not found in S3."
                logger.error(f"[{content_id}] {error_msg}")
                output_data['status'] = 'error'
                output_data['error'] = error_msg
                output_data['error_type'] = 'missing_dependency'
                return output_data
            logger.info(f"[{content_id}] Found source file: {s3_source_key}")
            # --- END NEW ---

            # Download source file
            source_filename = Path(s3_source_key).name
            local_source_path = content_temp_dir / source_filename

            logger.info(f"[{content_id}] Downloading source file from {s3_source_key} to {local_source_path}")
            if not self.s3_storage.download_file(s3_source_key, str(local_source_path)):
                logger.error(f"[{content_id}] Failed to download source file {s3_source_key}")
                output_data['status'] = 'error'
                output_data['error'] = f'Failed to download source file {s3_source_key}'
                return output_data

            # Extract audio
            logger.info(f"[{content_id}] Extracting audio from {local_source_path}")
            if not self._extract_audio(content_id, local_source_path, local_audio_path):
                logger.error(f"[{content_id}] Failed to extract audio from {local_source_path}")
                output_data['status'] = 'error'
                output_data['error'] = 'Failed to extract audio from source file'
                # Set corrupt_media_detected flag if it was detected during extraction
                if self.corrupt_media_detected:
                    output_data['corrupt_media_detected'] = True
                    output_data['error_code'] = 'corrupt_media'
                    output_data['permanent'] = True  # Permanently block reprocessing
                    logger.warning(f"[{content_id}] Corrupt/truncated media detected - permanently blocking")
                return output_data

            # Verify audio format
            logger.info(f"[{content_id}] Verifying audio format for {local_audio_path}")
            if not self._verify_audio_format(content_id, local_audio_path):
                logger.error(f"[{content_id}] Audio format verification failed for {local_audio_path}")
                output_data['status'] = 'error'
                output_data['error'] = 'Audio format verification failed'
                # Set corrupt_media_detected flag if it was detected during verification
                if self.corrupt_media_detected:
                    output_data['corrupt_media_detected'] = True
                    output_data['error_code'] = 'corrupt_media'
                    output_data['permanent'] = True  # Permanently block reprocessing
                    logger.warning(f"[{content_id}] Corrupt/truncated media detected - permanently blocking")
                return output_data

            # Upload audio to S3
            logger.info(f"[{content_id}] Uploading audio to S3: {s3_audio_key}")
            if not self.s3_storage.upload_file(str(local_audio_path), s3_audio_key):
                logger.error(f"[{content_id}] Failed to upload audio to S3: {s3_audio_key}")
                output_data['status'] = 'error'
                output_data['error'] = 'Failed to upload audio to S3'
                return output_data

            # Get audio duration
            duration = get_audio_duration(str(local_audio_path))
            if duration is None or duration <= 0.0:
                logger.error(f"[{content_id}] Failed to get valid duration from audio file")
                output_data['status'] = 'error'
                output_data['error'] = 'Failed to get valid duration from audio file'
                return output_data
            output_data['duration'] = duration

            # Create chunk plan
            logger.info(f"[{content_id}] Creating chunk plan")
            chunk_size = self.config['processing']['chunk_size']
            chunk_overlap = self.config['processing']['chunk_overlap']
            chunk_plan = create_chunk_plan(duration, chunk_size, chunk_overlap)
            total_chunks_expected = len(chunk_plan)
            output_data['chunk_plan'] = [{
                'index': chunk['index'],
                'start_time': chunk['start'],
                'end_time': chunk['end'],
                'duration': chunk['duration'],
                'extraction_status': 'completed'  # All chunks are completed since they exist in S3
            } for chunk in chunk_plan]

            # Upload chunks to S3
            logger.info(f"[{content_id}] Uploading chunks to S3")
            chunks_failed_count = 0
            chunks_created_count = 0
            for i, chunk in enumerate(chunk_plan):
                chunk_index = chunk['index']
                chunk_temp_path = content_temp_dir / f"chunk_{chunk_index}.wav"
                
                cmd_chunk = [
                    'ffmpeg', '-y',
                    '-ss', str(chunk['start']),
                    '-i', str(local_audio_path),
                    '-t', str(chunk['duration']),
                    '-vn',
                    '-c:a', 'copy',
                    str(chunk_temp_path)
                ]
                result_chunk = subprocess.run(cmd_chunk, capture_output=True, text=False)  # Don't try to decode output as text

                # Check if chunk was created successfully (regardless of return code)
                if chunk_temp_path.exists() and chunk_temp_path.stat().st_size > 0:
                    if result_chunk.returncode != 0:
                        error_msg = result_chunk.stderr.decode('utf-8', errors='replace') if result_chunk.stderr else "Unknown error"
                        logger.warning(f"[{content_id}] Chunk {chunk_index} created with warnings: {error_msg}")
                    
                    if not self.storage_manager.upload_chunk(content_id, chunk_index, str(chunk_temp_path), chunk['start'], chunk['end']):
                        logger.error(f"[{content_id}] Failed to upload chunk {chunk_index} to S3")
                        chunks_failed_count += 1
                    else:
                        chunks_created_count += 1
                else:
                    error_msg = result_chunk.stderr.decode('utf-8', errors='replace') if result_chunk.stderr else "Unknown error"
                    logger.error(f"[{content_id}] Failed to extract chunk {chunk_index}: {error_msg}")
                    chunks_failed_count += 1
                
                if chunk_temp_path.exists():
                    try: chunk_temp_path.unlink()
                    except OSError as e: logger.warning(f"[{content_id}] Failed to delete temp chunk {chunk_temp_path}: {e}")

            logger.info(f"[{content_id}] Chunk processing completed. Created: {chunks_created_count}, Failed: {chunks_failed_count}")
            total_chunks_processed = chunks_created_count # Actual chunks in S3

            # Update output data
            output_data['status'] = 'completed' if chunks_failed_count == 0 else 'partial_success'
            output_data['total_chunks'] = total_chunks_expected # Expected chunks based on plan
            output_data['chunks_created'] = chunks_created_count
            output_data['chunks_failed'] = chunks_failed_count
            output_data['standard_audio_key'] = s3_audio_key
            output_data['audio_path'] = s3_audio_key

            return output_data
                
        except Exception as e:
            logger.error(f"[{content_id}] Error processing content: {str(e)}")
            logger.error(traceback.format_exc())
            output_data['status'] = 'error'
            output_data['error'] = str(e)
            # Set corrupt_media_detected flag if it was detected during processing
            if self.corrupt_media_detected:
                output_data['corrupt_media_detected'] = True
                output_data['error_code'] = 'corrupt_media'
                output_data['permanent'] = True  # Permanently block reprocessing
                logger.warning(f"[{content_id}] Corrupt/truncated media detected - permanently blocking")
            return output_data
        finally:
            # Clean up local temporary files
            if local_source_path and local_source_path.exists():
                try: local_source_path.unlink()
                except OSError as e: logger.warning(f"[{content_id}] Failed to delete temp source {local_source_path}: {e}")
            if local_audio_path and local_audio_path.exists():
                try: local_audio_path.unlink()
                except OSError as e: logger.warning(f"[{content_id}] Failed to delete temp full audio {local_audio_path}: {e}")
            # Clean up the unique temp directory itself
            if self.temp_dir and self.temp_dir.exists():
                 try:
                     shutil.rmtree(self.temp_dir)
                     logger.debug(f"[{content_id}] Cleaned up main temporary directory: {self.temp_dir}")
                 except Exception as e:
                     logger.warning(f"[{content_id}] Failed to clean up temp directory {self.temp_dir}: {e}")
            logger.info(f"[{content_id}] Total processing time: {time.time() - start_time:.2f}s")

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Standardize audio, create chunks, and upload to S3.')
    parser.add_argument('--content', required=True, help='Content ID to process')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with more verbose output')
    
    args = parser.parse_args()
    
    # Setup logger without modifying the logger.py API
    global logger
    logger = setup_worker_logger('extract_audio')
    
    # If debug mode is enabled, add a console handler for DEBUG level messages
    if args.debug:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        logger.info("[DEBUG] Debug mode enabled - will show detailed diagnostic information")
    
    try:
        # Initialize processor and run
        processor = AudioExtractor(debug_mode=args.debug)
        result = await processor.process_content(args.content)
        
        # Print result as JSON for task processor to consume
        print(json.dumps(result))
        
        # Exit with appropriate status code
        exit_code = 0
        status = result.get('status')
        if status == 'skipped':
            exit_code = 0
        elif status in ['completed', 'success']:
             exit_code = 0
        elif status == 'error' and result.get('error') == 'invalid_media_file':
             exit_code = 2 # Special exit code for invalid media
        else:
             exit_code = 1 # General error

        sys.exit(exit_code)
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        print(json.dumps({'status': 'error', 'error': str(e)}))
        sys.exit(1)

if __name__ == '__main__':
    asyncio.run(main()) 