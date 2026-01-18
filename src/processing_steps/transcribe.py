#!/usr/bin/env python3
"""
Transcription Orchestrator

Top-level orchestrator for transcription tasks. Handles:
- S3 downloads/uploads
- Database updates
- VAD execution (if needed by method)
- Temp file management
- Method routing

Transcription logic is delegated to self-contained methods in transcribe_methods/
"""

# Centralized environment setup (must be before other imports)
from src.utils.env_setup import setup_env
setup_env('transcribe')

import os
import sys
from pathlib import Path

from src.utils.paths import get_project_root
from src.utils.config import load_config
import asyncio
import json
import yaml
import subprocess
import traceback
import tempfile
import shutil
import argparse
import time
import importlib
from datetime import datetime
from typing import Dict, Optional

# Add project root to Python path
sys.path.append(str(get_project_root()))

from src.utils.logger import setup_worker_logger
from src.storage.s3_utils import S3StorageConfig, S3Storage
from src.storage.content_storage import ContentStorageManager
from src.database.session import get_session
from src.database.models import Content, ContentChunk

logger = setup_worker_logger('transcribe_orchestrator')


def run_fluidaudio_vad(audio_path: str, output_path: str) -> bool:
    """
    Run FluidAudio VAD to detect speech segments.

    Args:
        audio_path: Path to input audio file
        output_path: Path to save VAD results JSON

    Returns:
        True if successful, False otherwise
    """
    try:
        # Path to FluidVAD Swift executable
        fluid_vad = Path(__file__).parent / "swift" / ".build" / "debug" / "FluidVAD"

        if not fluid_vad.exists():
            logger.error(f"FluidVAD executable not found at {fluid_vad}")
            logger.error("Build FluidVAD with: cd src/processing_steps/swift && swift build")
            return False

        # Run VAD analysis
        cmd = [str(fluid_vad), str(audio_path), str(output_path)]
        logger.info(f"Running FluidAudio VAD: {' '.join(cmd)}")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            logger.error(f"FluidAudio VAD failed: {result.stderr}")
            return False

        if result.stdout:
            logger.debug(f"FluidAudio VAD output: {result.stdout}")

        return Path(output_path).exists()

    except subprocess.TimeoutExpired:
        logger.error("FluidAudio VAD timed out")
        return False
    except Exception as e:
        logger.error(f"Error running FluidAudio VAD: {e}")
        logger.error(traceback.format_exc())
        return False


class TranscriptionProcessor:
    """
    Orchestrator for transcription tasks.

    Responsibilities:
    - Load configuration
    - Manage S3 storage
    - Download/upload audio and transcripts
    - Run VAD if needed
    - Route to appropriate transcription method
    - Update database
    """

    # Supported language codes (ISO 639-1)
    SUPPORTED_LANGUAGES = {
        'bg', 'hr', 'cs', 'da', 'nl', 'en', 'et', 'fi', 'fr', 'de',
        'el', 'hu', 'it', 'lv', 'lt', 'mt', 'pl', 'pt', 'ro', 'sk',
        'sl', 'es', 'sv', 'ru', 'uk', 'ca'
    }

    def __init__(self):
        """Initialize the transcription processor"""
        self.logger = logger

        # Load config
        self.config = load_config()

        # Get transcription method configuration
        self.default_method = self.config.get('processing', {}).get('transcription', {}).get('method', 'vad_hybrid_2pass')
        self.english_method = self.config.get('processing', {}).get('transcription', {}).get('english_method', 'parakeet_single_pass')
        logger.info(f"Default transcription method: {self.default_method}")
        logger.info(f"English-specific method: {self.english_method}")

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

        # Don't load method yet - will load dynamically based on language
        self.method = None
        self.method_name = None

    def _load_method(self, method_name: str):
        """
        Dynamically load transcription method.

        Args:
            method_name: Name of the method (e.g., 'vad_hybrid_2pass')

        Returns:
            Loaded method module
        """
        try:
            module = importlib.import_module(f'src.processing_steps.transcribe_methods.{method_name}')
            logger.info(f"Loaded transcription method: {method_name}")
            logger.info(f"  Requires VAD: {module.REQUIRES_VAD}")
            logger.info(f"  Requires Diarization: {module.REQUIRES_DIARIZATION}")
            return module
        except ImportError as e:
            logger.error(f"Failed to load transcription method '{method_name}': {e}")
            logger.error("Available methods: whisper_baseline, vad_hybrid_2pass, vad_concat_2pass")
            raise

    async def process_chunk(self, content_id: str, chunk_index: int, rewrite: bool = False, test_mode: bool = False, skip_method_selection: bool = False) -> Dict:
        """
        Process a single chunk of audio with the configured transcription method.

        Args:
            content_id: Content ID
            chunk_index: Chunk index
            rewrite: Force rewrite of existing transcript
            test_mode: Test mode - use local test directory instead of S3

        Returns:
            Dict with processing result
        """
        output_data = {
            'status': 'pending',
            'content_id': content_id,
            'chunk_index': chunk_index,
            'error': None
        }

        chunk_temp_dir = None

        try:
            # Get content language from database and select method
            language_code = None
            with get_session() as session:
                content = session.query(Content).filter_by(content_id=content_id).first()
                if content and content.main_language:
                    lang_code = content.main_language[:2].lower()
                    if lang_code in self.SUPPORTED_LANGUAGES:
                        language_code = lang_code
                        self.logger.info(f"[{content_id}:{chunk_index}] Using language '{language_code}' from content.main_language='{content.main_language}'")

            # Select transcription method based on language (unless skipped for comparison mode)
            if not skip_method_selection:
                if language_code == 'en':
                    selected_method = self.english_method
                    logger.info(f"[{content_id}:{chunk_index}] English content detected, using method: {selected_method}")
                else:
                    selected_method = self.default_method
                    logger.info(f"[{content_id}:{chunk_index}] Non-English content ({language_code}), using method: {selected_method}")

                # Load method if not already loaded or if it changed
                if self.method is None or self.method_name != selected_method:
                    self.method = self._load_method(selected_method)
                    self.method_name = selected_method
            else:
                logger.info(f"[{content_id}:{chunk_index}] Skipping method selection, using pre-set method: {self.method_name}")

            # Create temp directory for this chunk
            chunk_temp_dir = self.temp_dir / f"{content_id}_{chunk_index}"
            chunk_temp_dir.mkdir(exist_ok=True)

            # Check for audio chunk dependency in S3
            chunk_paths = self.storage_manager.get_chunk_paths(content_id, chunk_index)
            s3_chunk_audio_key = chunk_paths.get('audio')
            chunk_audio_exists = s3_chunk_audio_key and self.s3_storage.file_exists(s3_chunk_audio_key)

            if chunk_audio_exists:
                logger.info(f"[{content_id}:{chunk_index}] Found audio chunk in S3: {s3_chunk_audio_key}")
            else:
                logger.info(f"[{content_id}:{chunk_index}] Audio chunk not in S3, will slice from full audio")

            # Determine output path
            if test_mode:
                # Save to test_zone/transcribe_testing so transcripts don't overwrite each other
                test_base_dir = get_project_root() / 'test_zone' / "transcribe_testing" / content_id / self.method_name
                test_base_dir.mkdir(parents=True, exist_ok=True)
                target_key = str(test_base_dir / f"chunk_{chunk_index}_transcript.json")
                logger.info(f"[{content_id}:{chunk_index}] Test mode: Will write to {target_key}")
            else:
                target_key = chunk_paths.get('transcript')
                if not target_key:
                    error_msg = f"Could not determine transcript S3 key for chunk {chunk_index}."
                    logger.error(f"[{content_id}:{chunk_index}] {error_msg}")
                    output_data['status'] = 'error'
                    output_data['error'] = error_msg
                    return output_data

                if not rewrite and self.s3_storage.file_exists(target_key):
                    logger.info(f"[{content_id}:{chunk_index}] Transcript already exists at {target_key}. Skipping transcription.")
                    output_data['status'] = 'skipped'
                    output_data['reason'] = 'transcript_exists'
                    output_data['transcript_path'] = target_key
                    if chunk_temp_dir.exists():
                        try:
                            shutil.rmtree(chunk_temp_dir)
                        except Exception as e_clean:
                            logger.warning(f"[{content_id}:{chunk_index}] Failed to clean up temp dir on skip: {e_clean}")
                    return output_data

            # Download audio chunk from S3 or audio server
            local_audio_path = chunk_temp_dir / "audio.wav"

            if chunk_audio_exists:
                # Download from S3
                if not self.storage_manager.download_chunk(content_id, chunk_index, str(local_audio_path)):
                    error_msg = f'Failed to download audio chunk from S3: {s3_chunk_audio_key}'
                    logger.error(f"[{content_id}:{chunk_index}] {error_msg}")
                    output_data['status'] = 'error'
                    output_data['error'] = error_msg
                    return output_data
            else:
                # Fallback: slice audio directly from S3 using ffmpeg range requests
                with get_session() as session:
                    content = session.query(Content).filter_by(content_id=content_id).first()
                    if not content:
                        error_msg = f"Content not found in database: {content_id}"
                        logger.error(f"[{content_id}:{chunk_index}] {error_msg}")
                        output_data['status'] = 'error'
                        output_data['error'] = error_msg
                        return output_data

                    chunk = session.query(ContentChunk).filter_by(
                        content_id=content.id,
                        chunk_index=chunk_index
                    ).first()

                    if not chunk:
                        error_msg = f"Chunk {chunk_index} not found in database"
                        logger.error(f"[{content_id}:{chunk_index}] {error_msg}")
                        output_data['status'] = 'error'
                        output_data['error'] = error_msg
                        return output_data

                    # Download audio slice directly from S3 using ffmpeg range requests
                    start_time = chunk.start_time
                    end_time = chunk.end_time

                    logger.info(f"[{content_id}:{chunk_index}] Calling download_audio_slice for {start_time:.1f}s - {end_time:.1f}s")
                    download_start = time.time()

                    try:
                        success = await self.s3_storage.download_audio_slice(content_id, start_time, end_time, str(local_audio_path))
                        download_time = time.time() - download_start
                        logger.info(f"[{content_id}:{chunk_index}] download_audio_slice returned {success} in {download_time:.2f}s")

                        if not success:
                            error_msg = f'Failed to download audio slice from S3 for time range {start_time:.1f}s - {end_time:.1f}s'
                            logger.error(f"[{content_id}:{chunk_index}] {error_msg}")
                            output_data['status'] = 'error'
                            output_data['error'] = error_msg
                            return output_data
                    except Exception as e:
                        download_time = time.time() - download_start
                        logger.error(f"[{content_id}:{chunk_index}] download_audio_slice raised exception after {download_time:.2f}s: {e}")
                        logger.error(traceback.format_exc())
                        error_msg = f'Exception during audio slice download: {str(e)}'
                        output_data['status'] = 'error'
                        output_data['error'] = error_msg
                        return output_data

            # Run VAD if method needs it
            vad_json_path = None
            if self.method.REQUIRES_VAD:
                logger.info(f"[{content_id}:{chunk_index}] Method requires VAD, running FluidAudio VAD...")
                vad_output_path = chunk_temp_dir / "vad_result.json"

                if not run_fluidaudio_vad(str(local_audio_path), str(vad_output_path)):
                    logger.warning(f"[{content_id}:{chunk_index}] VAD failed, cannot proceed with method {self.method_name}")
                    output_data['status'] = 'error'
                    output_data['error'] = 'VAD failed'
                    return output_data

                vad_json_path = str(vad_output_path)

                # Log VAD results
                with open(vad_json_path) as f:
                    vad_data = json.load(f)
                vad_segments = vad_data.get('segments', [])
                logger.info(f"[{content_id}:{chunk_index}] VAD detected {len(vad_segments)} speech segments")

                if not vad_segments:
                    logger.warning(f"[{content_id}:{chunk_index}] No speech detected by VAD")
                    # Create empty transcript
                    transcript = {
                        'text': '',
                        'segments': [],
                        'language': language_code or 'en',
                        'method': self.method.METHOD_NAME,
                        'vad_used': True
                    }
                    temp_transcript_path = chunk_temp_dir / "transcript_words.json"
                    with open(temp_transcript_path, 'w') as f:
                        json.dump(transcript, f)

                    if test_mode:
                        final_path = Path(target_key)
                        shutil.copy(temp_transcript_path, final_path)
                    else:
                        self.s3_storage.upload_file(str(temp_transcript_path), target_key)

                    return {
                        'status': 'success',
                        'transcript_path': target_key,
                        'language': language_code or 'en',
                        'method': self.method.METHOD_NAME,
                        'vad_used': True
                    }

            # Download diarization if method needs it
            diarization_json_path = None
            if self.method.REQUIRES_DIARIZATION:
                logger.info(f"[{content_id}:{chunk_index}] Method requires diarization")

                # Diarization is stored at content level, not per-chunk
                # Download full diarization and filter for chunk time range
                diar_key = f"content/{content_id}/diarization.json"
                diar_local_path = chunk_temp_dir / "diarization.json"

                # Use download_json_flexible to handle both .json and .json.gz
                if self.s3_storage.download_json_flexible(diar_key, str(diar_local_path)):
                    # Filter diarization segments for this chunk's time range
                    with get_session() as db_session:
                        db_content = db_session.query(Content).filter_by(content_id=content_id).first()
                        if db_content:
                            chunk = db_session.query(ContentChunk).filter_by(
                                content_id=db_content.id,
                                chunk_index=chunk_index
                            ).first()

                            if chunk:
                                chunk_start = chunk.start_time
                                chunk_end = chunk.end_time

                                # Load and filter diarization
                                with open(diar_local_path) as f:
                                    full_diar = json.load(f)

                                # Filter segments that overlap with chunk time range
                                chunk_segments = []
                                for seg in full_diar.get('segments', []):
                                    seg_start = seg['start']
                                    seg_end = seg['end']

                                    # Check if segment overlaps with chunk
                                    if seg_end > chunk_start and seg_start < chunk_end:
                                        # Adjust segment times to be relative to chunk start
                                        chunk_segments.append({
                                            'start': max(0, seg_start - chunk_start),
                                            'end': min(chunk_end - chunk_start, seg_end - chunk_start),
                                            'speaker': seg['speaker']
                                        })

                                # Write filtered diarization
                                filtered_diar = {
                                    'segments': chunk_segments
                                }
                                with open(diar_local_path, 'w') as f:
                                    json.dump(filtered_diar, f)

                                diarization_json_path = str(diar_local_path)
                                logger.info(f"[{content_id}:{chunk_index}] Downloaded and filtered diarization: {len(chunk_segments)} segments for chunk time range {chunk_start:.1f}s-{chunk_end:.1f}s")
                else:
                    logger.warning(f"[{content_id}:{chunk_index}] Failed to download diarization from {diar_key} (tried .json.gz and .json)")

            # Call transcription method
            logger.info(f"[{content_id}:{chunk_index}] Running transcription with method: {self.method_name}")
            start_time = time.time()

            result = self.method.main(
                audio_path=str(local_audio_path),
                language=language_code or 'en',
                vad_json_path=vad_json_path,
                diarization_json_path=diarization_json_path
            )

            duration = time.time() - start_time

            if 'error' in result:
                logger.error(f"[{content_id}:{chunk_index}] Transcription failed: {result['error']}")
                output_data['status'] = 'error'
                output_data['error'] = result['error']
                return output_data

            # Update database to record transcription method and language
            transcription_method = result.get('method', self.method.METHOD_NAME)
            # Include language suffix for audit: e.g., "whisper_mlx_turbo_fr"
            transcribed_with = f"{transcription_method}_{language_code}" if language_code else transcription_method
            with get_session() as db_session:
                db_content = db_session.query(Content).filter_by(content_id=content_id).first()
                if db_content:
                    chunk = db_session.query(ContentChunk).filter_by(
                        content_id=db_content.id,
                        chunk_index=chunk_index
                    ).first()
                    if chunk:
                        chunk.transcribed_with = transcribed_with
                        chunk.transcription_status = 'completed'
                        chunk.transcription_completed_at = datetime.utcnow()
                        db_session.commit()

                        # Check if all chunks are now transcribed
                        all_chunks = db_session.query(ContentChunk).filter_by(content_id=db_content.id).all()
                        if all_chunks and all(c.transcription_status == 'completed' for c in all_chunks):
                            db_content.is_transcribed = True
                            db_content.last_updated = datetime.utcnow()
                            db_session.commit()
                            logger.info(f"[{content_id}:{chunk_index}] All chunks transcribed, marked content as transcribed")

            # Create transcript structure
            transcript = {
                'text': ' '.join(seg.get('text', '') for seg in result.get('segments', [])),
                'segments': result.get('segments', []),
                'language': result.get('language', language_code or 'en'),
                'method': transcription_method,
                'stats': result.get('stats', {})
            }

            # Write transcript to temp file
            temp_transcript_path = chunk_temp_dir / "transcript_words.json"
            with open(temp_transcript_path, 'w') as f:
                json.dump(transcript, f)

            # Save output
            if test_mode:
                final_path = Path(target_key)
                final_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(temp_transcript_path, final_path)

                # Also save audio in test_zone/transcribe_testing/content_id/
                test_zone_root = get_project_root() / 'test_zone' / "transcribe_testing" / content_id
                test_zone_root.mkdir(parents=True, exist_ok=True)
                audio_output_path = test_zone_root / f"chunk_{chunk_index}_audio.wav"
                if not audio_output_path.exists():
                    shutil.copy(local_audio_path, audio_output_path)

                logger.info(f"[{content_id}:{chunk_index}] Test mode: Wrote transcript to {final_path}")
                logger.info(f"[{content_id}:{chunk_index}] Transcription completed in {duration:.1f}s")
                return {
                    'status': 'success',
                    'transcript_path': str(final_path),
                    'audio_path': str(audio_output_path),
                    'language': result.get('language', 'en'),
                    'method': transcription_method,
                    'stats': result.get('stats', {})
                }
            else:
                if self.s3_storage.upload_file(str(temp_transcript_path), target_key):
                    logger.info(f"[{content_id}:{chunk_index}] Transcription completed in {duration:.1f}s")
                    total_words = result.get('stats', {}).get('total_words', 0)
                    logger.info(f"[{content_id}:{chunk_index}] Total words: {total_words}")
                    return {
                        'status': 'success',
                        'transcript_path': target_key,
                        'language': result.get('language', 'en'),
                        'method': transcription_method,
                        'stats': result.get('stats', {})
                    }
                else:
                    error_msg = f'Failed to upload transcript to S3: {target_key}'
                    logger.error(f"[{content_id}:{chunk_index}] {error_msg}")
                    return {
                        'status': 'error',
                        'error': error_msg
                    }

        except Exception as e:
            logger.error(f"Error processing chunk: {str(e)}")
            logger.error(traceback.format_exc())
            return {'status': 'error', 'error': str(e)}

        finally:
            # Clean up temp directory
            if chunk_temp_dir and chunk_temp_dir.exists():
                try:
                    shutil.rmtree(chunk_temp_dir)
                except Exception as e:
                    logger.warning(f"Failed to clean up temp directory: {e}")


async def compare_methods(content_id: str, chunk_index: int, test_mode: bool = False):
    """
    Compare all transcription methods on the same chunk.

    Runs each available method and generates a comparison report including:
    - Word counts
    - Processing time
    - Transcript alignment
    - Coverage metrics
    - Diarization alignment (if available)

    Args:
        content_id: Content ID
        chunk_index: Chunk index
        test_mode: Whether to use test mode

    Returns:
        Dict with comparison results
    """
    import glob
    from datetime import datetime

    logger.info(f"=== Starting method comparison for {content_id}:{chunk_index} ===")

    # Find all available methods
    methods_dir = Path(__file__).parent / "transcribe_methods"
    method_files = glob.glob(str(methods_dir / "*.py"))
    available_methods = []

    for method_file in method_files:
        method_name = Path(method_file).stem
        if method_name.startswith('_') or method_name == '__init__':
            continue
        available_methods.append(method_name)

    logger.info(f"Found {len(available_methods)} methods: {', '.join(available_methods)}")

    # Results storage
    results = {
        'content_id': content_id,
        'chunk_index': chunk_index,
        'timestamp': datetime.utcnow().isoformat(),
        'methods': {}
    }

    # Determine cache directory
    if test_mode:
        cache_dir = get_project_root() / 'test_zone' / "transcribe_testing" / content_id
        cache_dir.mkdir(parents=True, exist_ok=True)
    else:
        cache_dir = None

    # Test each method
    for method_name in available_methods:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing method: {method_name}")
        logger.info(f"{'='*60}")

        # Note: Caching disabled in comparison mode to force fresh transcription
        try:
            # Create processor with this method
            processor = TranscriptionProcessor()
            processor.method_name = method_name
            processor.method = processor._load_method(method_name)

            # Process chunk with skip_method_selection to force using the comparison method
            start_time = time.time()
            result = await processor.process_chunk(content_id, chunk_index, rewrite=True, test_mode=test_mode, skip_method_selection=True)
            processing_time = time.time() - start_time

            # Load transcript if successful
            transcript_data = None
            if result.get('status') == 'success':
                transcript_path = result['transcript_path']
                with open(transcript_path) as f:
                    transcript_data = json.load(f)

            # Store results
            method_result = {
                'status': result.get('status'),
                'processing_time': round(processing_time, 2),
                'word_count': len(transcript_data.get('text', '').split()) if transcript_data else 0,
                'segment_count': len(transcript_data.get('segments', [])) if transcript_data else 0,
                'language': result.get('language'),
                'stats': result.get('stats', {}),
                'error': result.get('error'),
                'transcript_path': result.get('transcript_path')
            }

            results['methods'][method_name] = method_result

            # Cache the metadata
            if cache_dir and result.get('status') == 'success':
                metadata_cache = cache_dir / f"{method_name}_chunk_{chunk_index}_metadata.json"
                with open(metadata_cache, 'w') as f:
                    json.dump(method_result, f, indent=2)
                logger.info(f"Cached metadata for {method_name}")

            logger.info(f"✓ {method_name}: {method_result['word_count']} words in {processing_time:.1f}s")

        except Exception as e:
            logger.error(f"✗ {method_name} failed: {e}")
            results['methods'][method_name] = {
                'status': 'error',
                'error': str(e)
            }

    # Generate comparison analysis
    logger.info(f"\n{'='*60}")
    logger.info("COMPARISON SUMMARY")
    logger.info(f"{'='*60}")

    # Sort by processing time
    successful_methods = {k: v for k, v in results['methods'].items() if v.get('status') == 'success'}

    if successful_methods:
        logger.info("\n--- Processing Time ---")
        for method, data in sorted(successful_methods.items(), key=lambda x: x[1]['processing_time']):
            logger.info(f"  {method:25s}: {data['processing_time']:6.1f}s")

        logger.info("\n--- Word Count ---")
        for method, data in sorted(successful_methods.items(), key=lambda x: x[1]['word_count'], reverse=True):
            logger.info(f"  {method:25s}: {data['word_count']:5d} words")

        logger.info("\n--- Segment Count ---")
        for method, data in sorted(successful_methods.items(), key=lambda x: x[1]['segment_count'], reverse=True):
            logger.info(f"  {method:25s}: {data['segment_count']:5d} segments")

    # Save comparison results
    if test_mode:
        output_dir = get_project_root() / 'test_zone' / "transcribe_testing" / content_id
        output_dir.mkdir(parents=True, exist_ok=True)
        comparison_file = output_dir / f"chunk_{chunk_index}_comparison.json"
        with open(comparison_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nComparison results saved to: {comparison_file}")
        results['comparison_file'] = str(comparison_file)

    return results


async def main():
    """Main entry point for command-line usage"""
    parser = argparse.ArgumentParser(description='Run transcription on a chunk')
    parser.add_argument('--content', required=True, help='Content ID')
    parser.add_argument('--chunk', type=int, required=True, help='Chunk index')
    parser.add_argument('--rewrite', action='store_true', help='Rewrite existing transcript')
    parser.add_argument('--test', action='store_true', help='Test mode (write to local tests/ directory)')
    parser.add_argument('--compare', action='store_true', help='Compare all transcription methods')

    args = parser.parse_args()

    try:
        # Compare mode - run all methods
        if args.compare:
            logger.info("Running comparison mode - testing all methods")
            result = await compare_methods(args.content, args.chunk, args.test)
            print(json.dumps(result, indent=2))
            sys.exit(0)

        # Normal mode - single method
        # Initialize processor
        processor = TranscriptionProcessor()

        # Log start
        logger.info(f"Starting transcription for content {args.content}, chunk {args.chunk}")
        if args.test:
            logger.info("Running in TEST MODE - using local files in tests/ directory")

        # Process the chunk
        result = await processor.process_chunk(args.content, args.chunk, args.rewrite, args.test)

        # Print result as JSON
        print(json.dumps(result))

        # Exit with 0 if status is 'success' OR 'skipped', otherwise 1
        success_status = result.get('status') in ['success', 'skipped', 'completed']
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
