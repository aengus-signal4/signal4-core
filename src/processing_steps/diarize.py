#!/usr/bin/env python3
"""
FluidAudio Speaker Diarization Processor
========================================

This module performs speaker diarization using FluidAudio (CoreML-based) on audio files.
It uses a Swift wrapper to interface with FluidAudio and produces:
- diarization.json: Speaker segments with timestamps

This processor duplicates the functionality of diarize.py but uses FluidInference/FluidAudio
instead of PyAnnote, offering potentially faster processing on Apple Silicon devices.

Usage:
    python diarize_fluid.py --content <content_id> [--test] [--rewrite]
"""

import sys
from pathlib import Path

from src.utils.paths import get_project_root
from src.utils.config import load_config
import asyncio
import json
import yaml
import subprocess
import argparse
import traceback
import warnings
import time
import tempfile
import shutil
from typing import Dict, List
from datetime import datetime, timezone

# Add the project root to Python path
sys.path.append(str(get_project_root()))

# Import required modules
from src.utils.logger import setup_worker_logger
from src.storage.s3_utils import S3StorageConfig, S3Storage
from src.storage.content_storage import ContentStorageManager
from src.database.session import get_session
from src.database.models import TaskQueue, Content
from src.database.state_manager import StateManager

logger = setup_worker_logger('diarize_fluid')

class FluidDiarizationProcessor:
    """FluidAudio-based diarization processor for Apple Silicon devices."""
    
    def __init__(self):
        """Initialize the FluidAudio diarization processor."""
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
        self.temp_dir = Path(tempfile.mkdtemp(prefix="fluid_diarization_"))
        logger.info(f"Created temporary directory: {self.temp_dir}")
        
        # Swift script path
        self.swift_script_path = Path(__file__).parent / "swift" / "diarize_fluid.swift"
        
        # Get diarization threshold from config or use default
        self.threshold = self.config.get('processing', {}).get('diarization', {}).get('threshold', 0.7)
        
        # Verify Swift script exists
        if not self.swift_script_path.exists():
            logger.error(f"Swift diarization script not found at {self.swift_script_path}")
            raise FileNotFoundError(f"Swift script missing: {self.swift_script_path}")
        
        logger.info(f"Using FluidAudio with threshold: {self.threshold}")
        logger.info(f"Swift script path: {self.swift_script_path}")

    async def _check_swift_dependencies(self) -> bool:
        """Check if Swift and required dependencies are available."""
        try:
            # Check if swift is available
            result = subprocess.run(['swift', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                logger.error("Swift compiler not found. Please install Xcode or Swift toolchain.")
                return False
            
            logger.info(f"Swift version: {result.stdout.strip().split()[0]}")
            
            # Check if we're on macOS (required for FluidAudio)
            import platform
            if platform.system() != 'Darwin':
                logger.error("FluidAudio requires macOS. Current platform: " + platform.system())
                return False
                
            logger.info("✅ Swift and macOS dependencies satisfied")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("Swift version check timed out")
            return False
        except Exception as e:
            logger.error(f"Error checking Swift dependencies: {e}")
            return False

    async def _run_swift_diarization(self, audio_path: Path, output_path: Path) -> Dict:
        """Run the Swift FluidAudio diarization script."""
        try:
            # Use the compiled Swift executable instead of running script directly
            swift_executable = self.swift_script_path.parent / ".build" / "debug" / "FluidDiarization"
            cmd = [
                str(swift_executable),
                str(audio_path),
                str(output_path),
                str(self.threshold)
            ]
            
            logger.info(f"Running Swift diarization: {' '.join(cmd)}")
            
            # Run with timeout to prevent hanging
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.swift_script_path.parent)
            )
            
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=300)  # 5 minute timeout
            
            stdout_text = stdout.decode('utf-8') if stdout else ""
            stderr_text = stderr.decode('utf-8') if stderr else ""
            
            if stdout_text:
                logger.info(f"Swift stdout: {stdout_text}")
            if stderr_text:
                logger.warning(f"Swift stderr: {stderr_text}")
            
            if process.returncode != 0:
                error_msg = f"Swift script failed with return code {process.returncode}"
                if stderr_text:
                    error_msg += f": {stderr_text}"
                logger.error(error_msg)
                return {'status': 'error', 'error': error_msg, 'error_type': 'swift_execution_error'}
            
            # Check if output file was created
            if not output_path.exists():
                error_msg = "Swift script completed but output file not found"
                logger.error(error_msg)
                return {'status': 'error', 'error': error_msg, 'error_type': 'missing_output'}
            
            # Load and validate the JSON output
            try:
                with open(output_path, 'r') as f:
                    result_data = json.load(f)
                
                logger.info(f"Successfully loaded diarization result with {len(result_data.get('segments', []))} segments")
                return {'status': 'success', 'data': result_data}
                
            except json.JSONDecodeError as e:
                error_msg = f"Failed to parse Swift output JSON: {e}"
                logger.error(error_msg)
                return {'status': 'error', 'error': error_msg, 'error_type': 'json_parse_error'}
            
        except asyncio.TimeoutError:
            error_msg = "Swift diarization timed out after 5 minutes"
            logger.error(error_msg)
            return {'status': 'error', 'error': error_msg, 'error_type': 'timeout_error'}
        except Exception as e:
            error_msg = f"Error running Swift diarization: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            return {'status': 'error', 'error': error_msg, 'error_type': 'execution_error'}

    async def process_content(self, content_id: str, rewrite: bool = False, test_mode: bool = False) -> Dict:
        """Perform speaker diarization on the content's audio using FluidAudio."""
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
            logger.info(f"[{content_id}] Starting FluidAudio diarization")

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

            # Check Swift dependencies
            if not await self._check_swift_dependencies():
                error_msg = "Swift dependencies not satisfied"
                logger.error(f"[{content_id}] {error_msg}")
                output_data.update({
                    'status': 'error',
                    'error': error_msg,
                    'error_type': 'dependency_error'
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

            # Run FluidAudio diarization via Swift
            temp_output_path = content_temp_dir / "diarization_raw.json"
            logger.info(f"[{content_id}] Running FluidAudio diarization")
            
            swift_result = await self._run_swift_diarization(local_audio_path, temp_output_path)
            
            if swift_result['status'] != 'success':
                logger.error(f"[{content_id}] FluidAudio diarization failed: {swift_result.get('error')}")
                output_data.update({
                    'status': 'error',
                    'error': swift_result.get('error'),
                    'error_type': swift_result.get('error_type', 'unknown_error')
                })
                return output_data

            diarization_data = swift_result['data']
            segments = diarization_data.get('segments', [])
            speakers_detected = diarization_data.get('speakersDetected', 0)
            speaker_embeddings = diarization_data.get('speakerEmbeddings', {})

            logger.info(f"[{content_id}] FluidAudio found {speakers_detected} speakers and {len(segments)} segments")
            if speaker_embeddings:
                logger.info(f"[{content_id}] Received embeddings for {len(speaker_embeddings)} speakers")

            # Update metadata with processing time
            processing_time = time.time() - start_time
            diarization_data['metadata']['processingTimeSeconds'] = processing_time

            # Prepare final output
            temp_diarization_path = content_temp_dir / "diarization.json"
            with open(temp_diarization_path, 'w') as f:
                json.dump(diarization_data, f, indent=2)

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
                    json.dump(diarization_data, f, indent=2)
                logger.info(f"[{content_id}] Test mode: Saved diarization results to {diarization_path}")
                
                logger.info(f"[{content_id}] Successfully processed diarization in test mode in {processing_time:.2f} seconds")
                
                output_data.update({
                    'status': 'success',
                    'segments_count': len(segments),
                    'speakers_count': speakers_detected,
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

                # Save speaker embeddings to database
                if speaker_embeddings:
                    try:
                        self._save_speaker_embeddings_to_db(content_id, speaker_embeddings)
                        logger.info(f"[{content_id}] Saved {len(speaker_embeddings)} speaker embeddings to database")
                    except Exception as e:
                        logger.error(f"[{content_id}] Failed to save speaker embeddings to database: {e}")
                        # Don't fail the whole task if database save fails
                else:
                    logger.warning(f"[{content_id}] No speaker embeddings to save to database")

                logger.info(f"[{content_id}] Successfully processed diarization in {processing_time:.2f} seconds")

                output_data.update({
                    'status': 'success',
                    'segments_count': len(segments),
                    'speakers_count': speakers_detected,
                    'diarization_path': s3_diarization_key,
                    'processing_time_seconds': processing_time,
                    'metadata': {
                        'audio_duration': diarization_data.get('audioDuration'),
                        'device': 'mps',  # Apple Silicon
                        'method': 'fluid_audio'
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


    def _save_speaker_embeddings_to_db(self, content_id: str, speaker_embeddings: Dict[str, List[float]]):
        """Save raw FluidAudio speaker embeddings to database.

        Args:
            content_id: Content ID string
            speaker_embeddings: Dict mapping speaker IDs (e.g., "1", "2") to 256-dim embedding arrays
        """
        import numpy as np
        from src.database.models import Speaker, SpeakerProcessingStatus

        for speaker_numeric_id, embedding_list in speaker_embeddings.items():
            # Convert FluidAudio speaker ID "1" -> "SPEAKER_1" (matches diarization.json format)
            # FluidAudio uses 1-based indexing without zero-padding (1, 2, 3...)
            # This must match the speaker IDs in diarization.json segments that stitch reads
            local_speaker_id = f"SPEAKER_{speaker_numeric_id}"

            # Convert to numpy array
            embedding_array = np.array(embedding_list, dtype=np.float32)

            # Normalize embedding
            norm = np.linalg.norm(embedding_array)
            if norm > 0:
                embedding_array = embedding_array / norm
            else:
                logger.warning(f"[{content_id}] Speaker {local_speaker_id} has zero-norm embedding, skipping")
                continue

            logger.debug(f"[{content_id}] Processing embedding for {local_speaker_id} (dim: {len(embedding_array)})")

            with get_session() as session:
                # Find or create speaker
                speaker = Speaker.find_by_content_speaker(session, content_id, local_speaker_id)

                if speaker:
                    # Update existing speaker with raw diarization embedding
                    speaker.embedding_diarization = embedding_array
                    speaker.embedding_diarization_quality = 1.0  # FluidAudio doesn't provide quality scores
                    speaker.diarization_method = 'fluid_audio'
                    speaker.updated_at = datetime.now(timezone.utc)
                    logger.info(f"[{content_id}] Updated existing speaker {speaker.speaker_hash} with diarization embedding")
                else:
                    # Create new speaker with raw diarization embedding only
                    speaker = Speaker.create_with_sequential_name(
                        session, content_id, local_speaker_id,
                        embedding=None  # No enriched embedding yet - that comes from stitch
                    )
                    speaker.embedding_diarization = embedding_array
                    speaker.embedding_diarization_quality = 1.0
                    speaker.diarization_method = 'fluid_audio'
                    speaker.rebase_status = SpeakerProcessingStatus.PENDING
                    speaker.notes = f"Created by FluidAudio diarization for {content_id}"

                    session.add(speaker)
                    logger.info(f"[{content_id}] Created new speaker {speaker.speaker_hash} with diarization embedding")

                session.commit()

    def cleanup(self):
        """Clean up resources."""
        try:
            if self.temp_dir and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up main temporary directory: {self.temp_dir}")
            logger.info("FluidAudio processor resources cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run FluidAudio speaker diarization')
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
        processor = FluidDiarizationProcessor()
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
                diarization_method='fluid_audio'
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