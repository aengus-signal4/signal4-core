#!/usr/bin/env python3
"""
PyAnnote Speaker Diarization Processor
======================================

This module performs GPU-intensive speaker diarization using PyAnnote on audio files.
It is focused solely on the diarization task and produces:
- diarization.json: Speaker segments with timestamps

This is intentionally kept lightweight and focused on the GPU-intensive PyAnnote
processing. All text processing, speaker embeddings, and speaker assignment
is handled by the stitch.py wrapper script.

Usage:
    python diarize.py --content <content_id> [--test] [--rewrite]
"""

# Centralized environment setup (must be before other imports)
from src.utils.env_setup import setup_env
setup_env()

import os
import sys
from pathlib import Path

from src.utils.paths import get_project_root
from src.utils.config import load_config
import asyncio
import json
import yaml
import torch
import subprocess
import argparse
import traceback
import warnings
import time
import tempfile
import shutil
from typing import Dict, List
from datetime import datetime, timezone
import aiohttp

# Suppress PyTorch Lightning migration warnings
warnings.filterwarnings("ignore", message="You have multiple `ModelCheckpoint` callback states")
warnings.filterwarnings("ignore", message="Lightning automatically upgraded your loaded checkpoint")
warnings.filterwarnings("ignore", message="Model was trained with pyannote.audio")
warnings.filterwarnings("ignore", message="Model was trained with torch")
warnings.filterwarnings("ignore", message="Model was trained with.*pytorch_lightning")
warnings.filterwarnings("ignore", message=".*does not support seeking.")

# Add the project root to Python path
sys.path.append(str(get_project_root()))

# Import required modules
from src.utils.logger import setup_worker_logger
from src.storage.s3_utils import S3StorageConfig, S3Storage
from src.storage.content_storage import ContentStorageManager
from src.database.session import get_session
from src.database.models import TaskQueue, Content
from src.database.state_manager import StateManager
# Model server imports removed - using direct model loading only

# Import PyAnnote after other imports to avoid conflicts
import pyannote.audio
from pyannote.audio import Pipeline, Audio
from pyannote.core import Segment

logger = setup_worker_logger('diarize')

class DiarizationProcessor:
    """Focused PyAnnote diarization processor for GPU-intensive speaker segmentation."""
    
    def __init__(self):
        """Initialize the diarization processor."""
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
        self.temp_dir = Path(tempfile.mkdtemp(prefix="diarization_"))
        logger.info(f"Created temporary directory: {self.temp_dir}")
        
        # Set up device
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Model server deprecated - using direct model loading only
        self.model_server_enabled = False
        
        # Initialize models to None, we'll load them later
        self.diarization_pipeline = None
        self.audio = None

    async def _initialize_models(self):
        """Initialize the PyAnnote diarization pipeline."""
        try:
            # Initialize Diarization Pipeline from local cache
            if self.diarization_pipeline is None:
                logger.info("Initializing PyAnnote diarization pipeline...")
                try:
                    pipeline_model_name = self.config['processing']['diarization'].get('model', "pyannote/speaker-diarization-3.1")
                    
                    # Check for local model cache first
                    project_root = get_project_root()
                    local_cache_dir = project_root / "models" / "local_cache"
                    model_cache_name = f"models--{pipeline_model_name.replace('/', '--')}"
                    local_model_path = local_cache_dir / model_cache_name
                    
                    logger.info(f"Target diarization model: {pipeline_model_name}")
                    logger.info(f"Checking local cache: {local_model_path}")
                    
                    # Set up local cache as HF_HOME 
                    import os
                    original_hf_home = os.environ.get('HF_HOME')
                    
                    try:
                        # Point HF_HOME to our local cache
                        os.environ['HF_HOME'] = str(local_cache_dir)
                        os.environ['TRANSFORMERS_CACHE'] = str(local_cache_dir)
                        
                        logger.info(f"Set HF_HOME to local cache: {local_cache_dir}")
                        logger.info("🚀 Loading diarization pipeline from LOCAL CACHE (no network required)")
                        
                        self.diarization_pipeline = Pipeline.from_pretrained(
                            pipeline_model_name,
                            use_auth_token=self.config['processing'].get('hf_token')
                        )
                        
                        logger.info("✅ Successfully loaded diarization pipeline (from local cache if available)")
                        
                    except Exception as e:
                        logger.error(f"Failed to load diarization pipeline: {e}")
                        
                        # Restore original HF_HOME if it was set
                        if original_hf_home:
                            os.environ['HF_HOME'] = original_hf_home
                        else:
                            os.environ.pop('HF_HOME', None)
                        os.environ.pop('TRANSFORMERS_CACHE', None)
                        
                        logger.warning("⚠️  Falling back to default Hugging Face download")
                        
                        # Fallback to original approach
                        self.diarization_pipeline = Pipeline.from_pretrained(
                            pipeline_model_name,
                            use_auth_token=self.config['processing'].get('hf_token')
                        )
                    
                    self.diarization_pipeline.to(self.device)
                    logger.info(f"✅ Diarization pipeline ({pipeline_model_name}) initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize diarization pipeline: {str(e)}")
                    return False

            # Initialize audio processor
            if self.audio is None:
                logger.info("Initializing audio processor...")
                try:
                    self.audio = Audio()
                    logger.info("Audio processor initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize audio processor: {str(e)}")
                    return False

            logger.info("Successfully initialized all PyAnnote components")
            return True

        except Exception as e:
            logger.error(f"Failed during model initialization: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False


    async def process_content(self, content_id: str, rewrite: bool = False, test_mode: bool = False) -> Dict:
        """Perform speaker diarization on the content's audio."""
        start_time = time.time()
        content_temp_dir = self.temp_dir / content_id
        output_data = {
            'status': 'pending',
            'content_id': content_id,
            'diarization_path': None,
            'error': None,
            'error_type': None
        }

        try:
            logger.info(f"[{content_id}] Starting PyAnnote diarization")

            # Check if diarization results already exist (moved up for early exit)
            s3_diarization_key = f"content/{content_id}/diarization.json"
            
            if not rewrite and not test_mode:
                if self.s3_storage.file_exists(s3_diarization_key):
                    logger.info(f"[{content_id}] Diarization already exists, marking as skipped")
                    output_data.update({
                        'status': 'skipped',
                        'diarization_path': s3_diarization_key,
                        'skipped_existing': True
                    })
                    return output_data

            content_temp_dir.mkdir(exist_ok=True)

            # Ensure models are initialized
            if not await self._initialize_models():
                error_msg = "Failed to initialize PyAnnote models"
                logger.error(f"[{content_id}] {error_msg}")
                output_data.update({
                    'status': 'error',
                    'error': error_msg,
                    'error_type': 'initialization_error'
                })
                return output_data

            # Download and prepare audio file (handles multiple formats and decompression)
            local_audio_path = content_temp_dir / "audio.wav"
            logger.info(f"[{content_id}] Downloading and preparing audio file")
            if not self.s3_storage.download_audio_flexible(content_id, str(local_audio_path)):
                error_msg = f"Failed to download and prepare audio file for {content_id}"
                logger.error(f"[{content_id}] {error_msg}")
                output_data['status'] = 'error'
                output_data['error'] = error_msg
                return output_data

            # Try model server first if enabled, otherwise use direct PyAnnote
            diarization_segments = []
            diarization_method = "unknown"
            audio_duration = None
            speakers_detected = 0
            
            # Direct PyAnnote loading only (model server deprecated)
            logger.info(f"[{content_id}] Using direct PyAnnote diarization")
            
            # Ensure models are initialized for direct loading
            if not await self._initialize_models():
                error_msg = "Failed to initialize PyAnnote models for direct loading"
                logger.error(f"[{content_id}] {error_msg}")
                output_data.update({
                    'status': 'error',
                    'error': error_msg,
                    'error_type': 'initialization_error'
                })
                return output_data
            
            try:
                # Get audio duration BEFORE running the pipeline
                audio_duration = self.audio.get_duration(local_audio_path)
                logger.info(f"[{content_id}] Audio duration: {audio_duration:.2f} seconds")

                # Run PyAnnote diarization
                diarization_result = self.diarization_pipeline(str(local_audio_path))
                if not diarization_result:
                    logger.warning(f"Diarization pipeline returned no result for {content_id}")
                    return {'status': 'error', 'error': 'Diarization pipeline returned empty result'}
                if not isinstance(diarization_result, pyannote.core.Annotation):
                    logger.error(f"Diarization pipeline did not return an Annotation object. Got: {type(diarization_result)}")
                    return {'status': 'error', 'error': 'Unexpected result type from diarization pipeline'}

                speakers_detected = len(diarization_result.labels())
                logger.info(f"[{content_id}] PyAnnote found {speakers_detected} speakers and {len(list(diarization_result.itersegments()))} segments")
                diarization_method = "direct_pyannote"
                
                # Collect all PyAnnote segments
                for segment, _, speaker_label in diarization_result.itertracks(yield_label=True):
                    try:
                        adjusted_segment = segment
                        # Clamp segment end time to audio duration if necessary
                        if segment.end > audio_duration:
                            new_end = audio_duration
                            if new_end > segment.start:
                                adjusted_segment = Segment(segment.start, new_end)
                                logger.warning(f"Adjusted segment end time from {segment.end:.3f}s to {audio_duration:.3f}s for speaker {speaker_label}")
                            else:
                                logger.warning(f"Segment for {speaker_label} [{segment.start:.3f}s - {segment.end:.3f}s] starts after audio duration. Skipping.")
                                continue

                        # Skip segments that are too short (less than 0.1 seconds)
                        if adjusted_segment.duration < 0.1:
                            logger.warning(f"Skipping very short segment for {speaker_label}: {adjusted_segment.duration:.3f}s")
                            continue

                        # Add to segments data
                        segment_info = {
                            "start": adjusted_segment.start,
                            "end": adjusted_segment.end,
                            "duration": adjusted_segment.duration,
                            "speaker": speaker_label
                        }
                        diarization_segments.append(segment_info)
                    except Exception as e:
                        logger.error(f"Error processing segment {adjusted_segment}: {str(e)}")
                        continue
            
            except Exception as e:
                logger.error(f"Error in PyAnnote diarization pipeline: {str(e)}\n{traceback.format_exc()}")
                return {'status': 'error', 'error': f'Diarization pipeline failed: {str(e)}'}

            logger.info(f"[{content_id}] Collected {len(diarization_segments)} diarization segments using {diarization_method}")

            # Prepare output
            processed_at = datetime.now(timezone.utc).isoformat()
            
            # Diarization output with method tracking
            diarization_output = {
                "content_id": content_id,
                "processed_at": processed_at,
                "method": diarization_method,
                "model": self.config['processing']['diarization'].get('model', "pyannote/speaker-diarization-3.1"),
                "audio_duration": audio_duration or 0,
                "speakers_detected": speakers_detected,
                "segments": diarization_segments,
                "metadata": {
                    "device": str(self.device),
                    "processing_time_seconds": time.time() - start_time
                }
            }
            temp_diarization_path = content_temp_dir / "diarization.json"
            with open(temp_diarization_path, 'w') as f:
                json.dump(diarization_output, f, indent=2)

            # Upload or save locally
            if test_mode:
                # Test mode - save locally
                test_output_dir = get_project_root() / "tests" / "content" / content_id
                test_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Save input files for debugging
                inputs_dir = test_output_dir / "inputs"
                inputs_dir.mkdir(parents=True, exist_ok=True)
                
                # Save the input audio file
                if local_audio_path.exists():
                    audio_copy_path = inputs_dir / "audio.wav"
                    shutil.copy2(local_audio_path, audio_copy_path)
                    logger.info(f"[{content_id}] Test mode: Saved input audio to {audio_copy_path}")
                
                # Save outputs
                outputs_dir = test_output_dir / "outputs"
                outputs_dir.mkdir(parents=True, exist_ok=True)
                
                # Save diarization results
                diarization_path = outputs_dir / "diarization.json"
                with open(diarization_path, 'w') as f:
                    json.dump(diarization_output, f, indent=2)
                logger.info(f"[{content_id}] Test mode: Saved diarization results to {diarization_path}")
                
                end_time = time.time()
                processing_time = end_time - start_time
                logger.info(f"[{content_id}] Successfully processed diarization in test mode in {processing_time:.2f} seconds")
                
                output_data.update({
                    'status': 'success',
                    'segments_count': len(diarization_segments),
                    'speakers_count': len(diarization_result.labels()),
                    'diarization_path': str(diarization_path),
                    'processing_time_seconds': processing_time,
                    'test_mode': True,
                    'test_output_dir': str(test_output_dir)
                })
            else:
                # Normal mode - upload to S3
                logger.info(f"[{content_id}] Uploading diarization results to S3")
                try:
                    self.s3_storage.upload_file(str(temp_diarization_path), s3_diarization_key)
                    logger.info(f"Uploaded diarization segments to {s3_diarization_key}")
                except Exception as e:
                    logger.error(f"Error uploading diarization to S3: {str(e)}\n{traceback.format_exc()}")
                    return {'status': 'error', 'error': f'Failed to upload diarization to S3: {str(e)}'}

                end_time = time.time()
                processing_time = end_time - start_time
                logger.info(f"[{content_id}] Successfully processed diarization in {processing_time:.2f} seconds")

                output_data.update({
                    'status': 'success',
                    'segments_count': len(diarization_segments),
                    'speakers_count': len(diarization_result.labels()),
                    'diarization_path': s3_diarization_key,
                    'processing_time_seconds': processing_time,
                    'metadata': {
                        'audio_duration': audio_duration,
                        'device': str(self.device)
                    }
                })
            
            return output_data

        except Exception as e:
            logger.error(f"Unhandled error processing content {content_id}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            output_data.update({
                'status': 'error',
                'error': f'Unhandled processing error: {str(e)}'
            })
            return output_data
        finally:
            # Clean up temp directory for this content (skip in test mode to allow inspection)
            if not test_mode and content_temp_dir.exists():
                try:
                    shutil.rmtree(content_temp_dir)
                    logger.debug(f"Cleaned up temporary directory: {content_temp_dir}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temp directory {content_temp_dir}: {e}")
            elif test_mode and content_temp_dir.exists():
                logger.info(f"Test mode: Temporary directory preserved at {content_temp_dir}")

    def cleanup(self):
        """Clean up resources."""
        try:
            if self.temp_dir and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up main temporary directory: {self.temp_dir}")
            # Release models
            self.diarization_pipeline = None
            self.audio = None
            logger.info("PyAnnote processor resources cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run PyAnnote speaker diarization')
    parser.add_argument('--content', required=False, help='Content ID to process')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--database', action='store_true', help='Update task and content status in database')
    parser.add_argument('--next', action='store_true', help='Get and process next pending task from queue')
    parser.add_argument('--task-id', help='Task ID if processing from task queue')
    parser.add_argument('--test', action='store_true', help='Test mode: write results locally and preserve temp directory')
    parser.add_argument('--rewrite', action='store_true', help='Rewrite existing diarization results')
    
    args = parser.parse_args()
    
    # Initialize processor variable
    processor = None
    
    try:
        # Initialize processor
        processor = DiarizationProcessor()
        state_manager = StateManager() if args.database else None
        start_time = time.time()
        result = None
        task_id = args.task_id
        content_id = args.content

        # Get next task if requested
        if args.next:
            logger.info("Looking for next pending diarization task...")
            with get_session() as session:
                task = session.query(TaskQueue).filter_by(
                    status='pending',
                    task_type='diarize'
                ).order_by(
                    TaskQueue.priority.desc(),
                    TaskQueue.created_at.asc()
                ).first()
                
                if task:
                    task_id = task.id
                    content_id = task.content_id
                    # Mark as processing
                    task.status = 'processing'
                    task.started_at = datetime.now()
                    session.commit()
                    logger.info(f"Processing task {task_id} for content {content_id}")
                else:
                    logger.info("No pending diarization tasks found")
                    return

        if not content_id:
            logger.error("No content ID provided and no task found")
            sys.exit(1)

        # Process the content
        result = await processor.process_content(content_id, rewrite=args.rewrite, test_mode=args.test)
        duration = time.time() - start_time
        
        # Update database if requested
        if args.database and state_manager:
            success = result.get('status') == 'success'
            error = result.get('error')
            
            # Update task status if we have a task_id
            if task_id:
                with get_session() as session:
                    task = session.query(TaskQueue).filter_by(id=task_id).first()
                    if task:
                        task.status = 'completed' if success else 'failed'
                        task.result = result
                        if error:
                            task.error = error
                        task.completed_at = datetime.now()
                        task.duration = duration
                        session.commit()
                        logger.info(f"Updated task {task_id} status to {task.status}")
            
            # Update content status
            await state_manager.update_diarization_complete_status(
                content_id=content_id,
                success=success,
                result_data=result if success else None,
                error=error,
                diarization_method='pyannote3.1'
            )
            logger.info(f"Updated content {content_id} diarization status")
        
        # Print result as JSON for task processor to consume
        print(json.dumps(result))
        
        # Exit with appropriate status code
        sys.exit(0 if result.get('status') in ['completed', 'success', 'skipped'] else 1)
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        print(json.dumps({'status': 'error', 'error': str(e)}))
        sys.exit(1)
    finally:
        if processor:
            processor.cleanup()

if __name__ == '__main__':
    asyncio.run(main())