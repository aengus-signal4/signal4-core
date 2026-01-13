#!/usr/bin/env python3
import os
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
import mlx_whisper
import time
import threading
import queue
import psutil
import traceback
import multiprocessing
from contextlib import redirect_stderr, contextmanager
import io
import aiohttp

# Add the project root to Python path
sys.path.append(str(get_project_root()))

# Import required modules
from src.utils.logger import setup_worker_logger
from src.storage.s3_utils import S3StorageConfig, S3Storage
from src.storage.content_storage import ContentStorageManager
from src.database.session import get_session
from src.database.models import Content, ContentChunk
# Model server imports removed - using direct model loading only

# Set up logging - use a single logger for all functions
logger = setup_worker_logger('transcribe')

@contextmanager
def suppress_stderr():
    """Context manager to temporarily suppress stderr output"""
    with open(os.devnull, 'w') as devnull:
        stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = stderr

def _run_mlx_transcribe(audio_path, language=None, max_chunk_duration=300.0):
    """Run MLX Whisper transcription with automatic chunking

    Args:
        audio_path: Path to audio file
        language: Optional language code (e.g., 'english', 'french')
        max_chunk_duration: Maximum chunk duration in seconds (default 300s = 5 minutes)
    """
    try:
        with suppress_stderr():
            import mlx_whisper
            import time
            import os
            import logging
            import json
            import torchaudio
            import torch

            # Set up logger
            logger = logging.getLogger('mlx_transcribe')

            # Check audio duration
            waveform, sample_rate = torchaudio.load(audio_path)
            audio_duration = waveform.shape[1] / sample_rate

            # If audio is short enough, transcribe directly
            if audio_duration <= max_chunk_duration:
                logger.info(f"Audio duration {audio_duration:.1f}s <= {max_chunk_duration}s, transcribing directly")

                # Prepare transcription kwargs
                transcribe_kwargs = {
                    "path_or_hf_repo": "mlx-community/whisper-large-v3-turbo",
                    "word_timestamps": True
                }

                if language:
                    transcribe_kwargs["language"] = language
                    logger.info(f"Using specified language: {language}")

                # Run transcription
                start_time = time.time()
                result = mlx_whisper.transcribe(audio_path, **transcribe_kwargs)
                elapsed_time = time.time() - start_time
                logger.info(f"Transcription completed in {elapsed_time:.2f} seconds")

                return result

            # Audio is too long, need to chunk it
            logger.info(f"Audio duration {audio_duration:.1f}s > {max_chunk_duration}s, chunking required")

            # Create temp directory for chunks
            temp_dir = Path(tempfile.mkdtemp(prefix="whisper_chunks_"))
            try:
                # Convert to mono if needed
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)

                # Resample to 16kHz if needed
                if sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                    waveform = resampler(waveform)
                    sample_rate = 16000

                # Split into chunks
                chunk_samples = int(max_chunk_duration * sample_rate)
                num_chunks = int(np.ceil(waveform.shape[1] / chunk_samples))

                logger.info(f"Splitting into {num_chunks} chunks of max {max_chunk_duration}s")

                all_segments = []
                current_offset = 0.0

                for chunk_idx in range(num_chunks):
                    start_sample = chunk_idx * chunk_samples
                    end_sample = min((chunk_idx + 1) * chunk_samples, waveform.shape[1])

                    chunk_waveform = waveform[:, start_sample:end_sample]
                    chunk_duration = chunk_waveform.shape[1] / sample_rate

                    # Save chunk
                    chunk_path = temp_dir / f"chunk_{chunk_idx}.wav"
                    torchaudio.save(str(chunk_path), chunk_waveform, sample_rate)

                    logger.info(f"Transcribing chunk {chunk_idx + 1}/{num_chunks} ({chunk_duration:.1f}s)")

                    # Transcribe chunk
                    transcribe_kwargs = {
                        "path_or_hf_repo": "mlx-community/whisper-large-v3-turbo",
                        "word_timestamps": True
                    }

                    if language:
                        transcribe_kwargs["language"] = language

                    chunk_result = mlx_whisper.transcribe(str(chunk_path), **transcribe_kwargs)

                    # Adjust timestamps
                    for segment in chunk_result.get('segments', []):
                        segment['start'] = segment.get('start', 0) + current_offset
                        segment['end'] = segment.get('end', 0) + current_offset

                        if 'words' in segment:
                            for word in segment['words']:
                                word['start'] = word.get('start', 0) + current_offset
                                word['end'] = word.get('end', 0) + current_offset

                        all_segments.append(segment)

                    current_offset += chunk_duration

                # Combine results
                full_text = ' '.join(seg.get('text', '') for seg in all_segments)

                result = {
                    'text': full_text,
                    'segments': all_segments,
                    'language': all_segments[0].get('language', 'en') if all_segments else 'en'
                }

                logger.info(f"Chunked transcription complete: {len(all_segments)} segments")
                return result

            finally:
                # Clean up temp directory
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)

    except Exception as e:
        return {'error': str(e)}

def run_transcribe_wrapper(audio_path, queue, language=None):
    """Wrapper function to run transcription and put result in queue
    
    Args:
        audio_path: Path to audio file
        queue: Queue to put result in
        language: Optional language code for transcription
    """
    try:
        with suppress_stderr():
            result = _run_mlx_transcribe(audio_path, language=language)
            queue.put(result)
    except Exception as e:
        queue.put({'error': str(e)})

class TranscriptionProcessor:
    # Language code mapping from ISO 639-1 to Whisper language names
    LANGUAGE_MAPPING = {
        'en': 'english',
        'fr': 'french',
        'it': 'italian',
        'uk': 'ukrainian',
        'de': 'german',
        'es': 'spanish',
        'ca': 'catalan',
        'ru': 'russian'
    }
    
    def __init__(self):
        """Initialize the transcription processor"""
        with suppress_stderr():
            # Set implementation and device
            self.implementation = "mlx_whisper"
            self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
            # Use the global logger for consistency
            self.logger = logger
            
            # Load config
            self.config = load_config()
            
            # Model server deprecated - using direct model loading only
            self.model_server_enabled = False
                
            # Initialize S3 storage
            s3_config = S3StorageConfig(
                endpoint_url=self.config['storage']['s3']['endpoint_url'],
                fallback_endpoint_url=self.config['storage']['s3'].get('fallback_endpoint_url'),
                access_key=self.config['storage']['s3']['access_key'],
                secret_key=self.config['storage']['s3']['secret_key'],
                bucket_name=self.config['storage']['s3']['bucket_name'],
                use_ssl=self.config['storage']['s3']['use_ssl']
            )
            self.s3_storage = S3Storage(s3_config)
            self.storage_manager = ContentStorageManager(self.s3_storage)
            
            # Set up temp directory
            self.temp_dir = Path("/tmp/whisper_chunks")
            self.temp_dir.mkdir(parents=True, exist_ok=True)

    def _sanitize_segments(self, segments: List[Dict]) -> List[Dict]:
        """Sanitize segments data to ensure it's compatible with PostgreSQL JSON"""
        sanitized = []
        last_end_time = 0.0
        
        # Constants for duration validation
        WORDS_PER_SECOND = 2.5  # Average speaking rate
        MIN_SEGMENT_DURATION = 0.3  # Minimum duration in seconds
        MAX_SEGMENT_DURATION = 20.0  # Maximum duration in seconds
        
        for segment in segments:
            # Skip segments with empty text
            text = segment.get('text', '').strip()
            if not text:
                continue
                
            # Create clean segment
            clean_segment = {
                'text': text,
                'start_time': None,
                'end_time': None
            }
            
            # Clean up timestamps
            start_time = segment.get('start_time', segment.get('start'))
            end_time = segment.get('end_time', segment.get('end'))
            
            # Handle NaN or invalid values
            if isinstance(start_time, float) and (start_time != start_time):  # NaN check
                start_time = last_end_time
            if isinstance(end_time, float) and (end_time != end_time):  # NaN check
                end_time = start_time + 1.0 if start_time is not None else last_end_time + 1.0
                
            # Ensure valid numeric values
            try:
                start_time = float(start_time) if start_time is not None else last_end_time
                end_time = float(end_time) if end_time is not None else start_time + 1.0
            except (ValueError, TypeError):
                start_time = last_end_time
                end_time = start_time + 1.0
            
            # Fix inverted timestamps
            if end_time < start_time:
                start_time, end_time = end_time, start_time
            
            # Calculate expected duration based on text length
            word_count = len(text.split())
            expected_duration = max(MIN_SEGMENT_DURATION, word_count / WORDS_PER_SECOND)
            expected_duration = min(expected_duration, MAX_SEGMENT_DURATION)
            
            # Adjust duration if it's significantly different from expected
            current_duration = end_time - start_time
            if current_duration < MIN_SEGMENT_DURATION or current_duration > MAX_SEGMENT_DURATION:
                end_time = start_time + expected_duration
            elif current_duration > expected_duration * 2:
                end_time = start_time + expected_duration * 1.5
                
            clean_segment['start_time'] = start_time
            clean_segment['end_time'] = end_time
            
            # Copy any other fields except word timestamps
            for key, value in segment.items():
                if key not in ('text', 'start', 'start_time', 'end', 'end_time', 'words'):
                    clean_segment[key] = value
                    
            sanitized.append(clean_segment)
            last_end_time = end_time
            
        return sanitized


    async def _run_inference(self, chunk_data: Dict) -> Dict:
        """Run inference on a chunk of audio"""
        try:
            if 'audio_path' not in chunk_data:
                return {'status': 'error', 'error': "Missing required 'audio_path' in chunk data"}
                
            audio_path = chunk_data['audio_path']
            content_id = chunk_data.get('content_id', 'unknown')
            chunk_index = chunk_data.get('chunk_index', -1)
            language = chunk_data.get('language')  # Get language if provided
            
            # Direct MLX Whisper loading only (model server deprecated)
            self.logger.info(f"[{content_id}:{chunk_index}] Using direct MLX Whisper loading")
            
            if self.implementation != "mlx_whisper":
                return {'status': 'error', 'error': f"Unsupported implementation: {self.implementation}"}
                
            self.logger.debug(f"Running MLX Whisper transcription on {audio_path}")
            
            # Create a queue for getting the result from the process
            result_queue = multiprocessing.Queue()
            
            # Create and start the process with stderr suppressed
            with suppress_stderr():
                process = multiprocessing.Process(
                    target=run_transcribe_wrapper,
                    args=(audio_path, result_queue, language)
                )
                process.start()
            
            # Wait for the result
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    result_queue.get,
                    60  # timeout in seconds
                )
            except queue.Empty:
                process.terminate()
                return {'status': 'error', 'error': "Transcription timeout"}
            finally:
                process.join(timeout=1)
                if process.is_alive():
                    process.terminate()
                    process.join()
                
            if isinstance(result, dict) and 'error' in result:
                return {'status': 'error', 'error': result['error']}
            
            if not result:
                return {'status': 'error', 'error': "Transcription returned None"}
                
            return {
                'status': 'success',
                'text': result.get('text', ''),
                'segments': result.get('segments', []),
                'language': result.get('language', 'en'),
                'method': 'whisper_mlx_turbo'
            }
                    
        except Exception as e:
            self.logger.error(f"MLX Whisper transcription error: {str(e)}")
            return {'status': 'error', 'error': str(e)}

    async def process_chunk(self, content_id: str, chunk_index: int, rewrite: bool = False) -> Dict:
        """Process a single chunk of audio"""
        output_data = {
            'status': 'pending',
            'content_id': content_id,
            'chunk_index': chunk_index,
            'error': None,
            'text': None,
            'segments': None,
            'language': None
        }
        try:
            # Get content language from database
            whisper_language = None
            with get_session() as session:
                content = session.query(Content).filter_by(content_id=content_id).first()
                if content and content.main_language:
                    # Get first two characters and convert to lowercase
                    lang_code = content.main_language[:2].lower()
                    # Map to Whisper language name if supported
                    whisper_language = self.LANGUAGE_MAPPING.get(lang_code)
                    if whisper_language:
                        self.logger.info(f"[{content_id}:{chunk_index}] Using language '{whisper_language}' from content.main_language='{content.main_language}'")
            
            # Create temp directory for this chunk
            chunk_temp_dir = self.temp_dir / f"{content_id}_{chunk_index}"
            chunk_temp_dir.mkdir(exist_ok=True)
            
            # Check for audio chunk dependency
            chunk_paths = self.storage_manager.get_chunk_paths(content_id, chunk_index)
            s3_chunk_audio_key = chunk_paths.get('audio')
            if not s3_chunk_audio_key or not self.s3_storage.file_exists(s3_chunk_audio_key):
                error_msg = f"Missing input file: Audio chunk ({s3_chunk_audio_key or 'path not generated'}) not found in S3."
                logger.error(f"[{content_id}:{chunk_index}] {error_msg}")
                output_data['status'] = 'error'
                output_data['error'] = error_msg
                output_data['error_type'] = 'missing_dependency'
                return output_data
            logger.info(f"[{content_id}:{chunk_index}] Found required audio chunk: {s3_chunk_audio_key}")
            
            # Check if transcript already exists in S3
            target_key = chunk_paths.get('transcript')
            if not target_key:
                # Handle cases where the path might not be generated (e.g., missing config)
                error_msg = f"Could not determine transcript S3 key for chunk {chunk_index}."
                logger.error(f"[{content_id}:{chunk_index}] {error_msg}")
                output_data['status'] = 'error'
                output_data['error'] = error_msg
                output_data['error_type'] = 'processing_error' # Internal error, not dependency
                return output_data

            if not rewrite and self.s3_storage.file_exists(target_key):
                logger.info(f"[{content_id}:{chunk_index}] Transcript already exists at {target_key}. Verifying validity...")
                # Optional: Add a validity check here if needed (e.g., download and check format/content)
                # For now, assuming existence implies validity based on logs
                logger.info(f"[{content_id}:{chunk_index}] Existing transcript {target_key} is considered valid. Skipping transcription.")
                output_data['status'] = 'skipped'
                output_data['reason'] = 'transcript_exists'
                output_data['transcript_path'] = target_key
                # Clean up temp dir even on skip
                if chunk_temp_dir.exists():
                     try: shutil.rmtree(chunk_temp_dir)
                     except Exception as e_clean: logger.warning(f"[{content_id}:{chunk_index}] Failed to clean up temp dir on skip: {e_clean}")
                return output_data

            # Download audio chunk
            local_audio_path = chunk_temp_dir / f"audio.wav"
            if not self.storage_manager.download_chunk(content_id, chunk_index, str(local_audio_path)):
                error_msg = f'Failed to download audio chunk: {s3_chunk_audio_key}'
                logger.error(f"[{content_id}:{chunk_index}] {error_msg}")
                output_data['status'] = 'error'
                output_data['error'] = error_msg
                return output_data
            
            # Run transcription
            start_time = time.time()
            inference_data = {
                'audio_path': str(local_audio_path), 
                'content_id': content_id, 
                'chunk_index': chunk_index
            }
            # Add language if available
            if whisper_language:
                inference_data['language'] = whisper_language
            
            result = await self._run_inference(inference_data)
            duration = time.time() - start_time
            
            if result['status'] == 'success':
                # Update database to record which transcription method was used
                transcription_method = result.get('method', 'whisper_mlx_turbo')
                with get_session() as db_session:
                    db_content = db_session.query(Content).filter_by(content_id=content_id).first()
                    if db_content:
                        chunk = db_session.query(ContentChunk).filter_by(
                            content_id=db_content.id,
                            chunk_index=chunk_index
                        ).first()
                        if chunk:
                            chunk.transcribed_with = transcription_method
                            db_session.commit()

                # Create the complete transcript structure with words from segments
                transcript = {
                    'text': result.get('text', ''),
                    'segments': [],
                    'language': result.get('language', 'en')
                }
                
                # Process each segment
                if 'segments' in result:
                    for segment in result['segments']:
                        # Keep only the essential fields we need
                        clean_segment = {
                            'text': segment.get('text', ''),
                            'start': segment.get('start', 0.0),
                            'end': segment.get('end', 0.0),
                            'words': []
                        }
                        
                        # Add words with their timestamps
                        if 'words' in segment:
                            for word_data in segment['words']:
                                clean_segment['words'].append({
                                    'word': word_data.get('word', ''),
                                    'start': word_data.get('start', 0.0),
                                    'end': word_data.get('end', 0.0)
                                })
                        
                        transcript['segments'].append(clean_segment)
                
                # Write transcript to temp file
                temp_transcript_path = chunk_temp_dir / "transcript_words.json"
                with open(temp_transcript_path, 'w') as f:
                    json.dump(transcript, f)
                
                # Upload to S3
                if self.s3_storage.upload_file(str(temp_transcript_path), target_key):
                    logger.info(f"Transcription for {content_id} chunk {chunk_index} completed in {duration:.1f}s")
                    # Return minimal success status
                    return {
                        'status': 'success',
                        'transcript_path': target_key, # Include path for reference
                        'language': result.get('language', 'en') # Keep language code
                    }
                else:
                    # Return error if upload fails
                    error_msg = f'Failed to upload transcript to S3: {target_key}'
                    logger.error(f"[{content_id}:{chunk_index}] {error_msg}")
                    return {
                        'status': 'error',
                        'error': error_msg
                    }
            else:
                # Propagate error status from _run_inference
                logger.error(f"[{content_id}:{chunk_index}] Transcription inference failed: {result.get('error')}")
                return result # result already contains {'status': 'error', 'error': ...}
                
        except Exception as e:
            logger.error(f"Error processing chunk: {str(e)}")
            return {'status': 'error', 'error': str(e)}
        finally:
            # Clean up temp directory
            if 'chunk_temp_dir' in locals() and chunk_temp_dir.exists():
                try:
                    shutil.rmtree(chunk_temp_dir)
                except Exception as e:
                    logger.warning(f"Failed to clean up temp directory: {e}")

async def main():
    """Main entry point"""
    with suppress_stderr():
        parser = argparse.ArgumentParser(description='Run transcription on a chunk')
        parser.add_argument('--content', required=True, help='Content ID')
        parser.add_argument('--chunk', type=int, required=True, help='Chunk index')
        parser.add_argument('--rewrite', action='store_true', help='Rewrite existing transcript')
        
        args = parser.parse_args()
        
        try:
            # Initialize processor
            processor = TranscriptionProcessor()
            
            # Process the chunk
            result = await processor.process_chunk(args.content, args.chunk, args.rewrite)
            
            # Print result as JSON
            print(json.dumps(result))
            
            # Exit with 0 if status is 'success' OR 'skipped', otherwise 1
            success_status = result.get('status') in ['success', 'skipped', 'completed'] # Treat skipped as success
            sys.exit(0 if success_status else 1)
            
        except Exception as e:
            logger.error(f"Error processing chunk {args.chunk} for content {args.content}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            print(json.dumps({
                'status': 'error', 
                'error': f'Main execution error: {str(e)}',
                'content_id': args.content,
                'chunk_index': args.chunk
            }))
            sys.exit(1)

if __name__ == '__main__':
    asyncio.run(main()) 