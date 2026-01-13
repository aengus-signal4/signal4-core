import logging
from datetime import datetime, timezone
from typing import Dict, Optional
from pathlib import Path
import asyncio
import time

from ..utils.logger import setup_worker_logger, setup_task_logger
from ..storage.s3_utils import S3StorageConfig, S3Storage
from ..storage.content_storage import ContentStorageManager
from ..database.models import Content

class WorkerRestartRequest(Exception):
    """Raised when a worker needs to restart after completing its tasks"""
    pass

class WorkerMode:
    HEAD = 'head'
    WORKER = 'worker'
    
    @staticmethod
    def determine_mode(args, config):
        if args.head:
            return WorkerMode.HEAD
        if args.worker:
            return WorkerMode.WORKER
            
        # Auto-detect based on IP
        from ..utils.logger import get_worker_ip
        current_ip = get_worker_ip()
        if current_ip == config['network']['head_node_ip']:
            return WorkerMode.HEAD
        return WorkerMode.WORKER

class WorkerState:
    def __init__(self, worker_type: str, config: Dict, quiet: bool = False, process_count: int = 0):
        self.worker_type = worker_type
        self.config = config
        self.quiet = quiet
        self.tasks_completed = 0
        self.current_content_id = None
        self.chunks_processed_for_content = 0
        self.should_exit = False
        self.worker_logger = setup_worker_logger(worker_type)
        self.task_logger = setup_task_logger(worker_type)
        self.process_count = process_count
        
        # Get worker limits from config
        worker_limits = config['processing']['distributed']['worker_limits'].get(worker_type, {})
        self.tasks_per_run = worker_limits.get('tasks_per_run', 99999)
        self.cooldown = worker_limits.get('cooldown', 60)
        
        # Get transcription timeout from config
        self.max_transcription_duration = config['processing']['transcription'].get('max_transcription_duration', 120)
        
        # Get cooldown periods from config
        self.post_kill_cooldown = config['processing']['cooldowns']['post_kill']
        self.post_restart_cooldown = config['processing']['cooldowns']['post_restart']
        
        # Log worker startup with limits
        self.worker_logger.info(
            f"Starting {worker_type} worker (limit: {self.tasks_per_run} tasks, "
            f"cooldown: {self.cooldown}s, max transcription duration: {self.max_transcription_duration}s, "
            f"process count: {self.process_count})"
        )
        
        # Initialize storage components if needed
        self.s3_storage = None
        self.storage_manager = None
        if worker_type in ['transcribe', 'extract_audio', 'stitch', 'download_youtube', 'download_podcast']:
            s3_config = S3StorageConfig(
                endpoint_url=config['storage']['s3']['endpoint_url'],
                access_key=config['storage']['s3']['access_key'],
                secret_key=config['storage']['s3']['secret_key'],
                bucket_name=config['storage']['s3']['bucket_name'],
                use_ssl=config['storage']['s3']['use_ssl']
            )
            self.s3_storage = S3Storage(s3_config)
            self.storage_manager = ContentStorageManager(self.s3_storage)

    def request_exit(self):
        """Request graceful shutdown"""
        self.should_exit = True
        self.worker_logger.info(f"Exit requested for {self.worker_type} worker")
        
    def should_restart(self) -> bool:
        """Check if worker should restart based on tasks completed"""
        if self.tasks_completed >= self.tasks_per_run:
            self.worker_logger.info(
                f"Worker {self.worker_type} completed {self.tasks_completed}/{self.tasks_per_run} tasks, "
                f"will restart after {self.cooldown}s cooldown"
            )
            return True
        return False

    def check_limit(self, content_id: str) -> bool:
        """Check if we can process a new content item without exceeding the limit"""
        if content_id != self.current_content_id:
            # This would be a new content item
            if self.tasks_completed >= self.tasks_per_run:
                self.worker_logger.info(
                    f"Task limit reached ({self.tasks_completed}/{self.tasks_per_run}). "
                    f"Cannot process new content {content_id}"
                )
                raise WorkerRestartRequest()
        return True
        
    def log_task_completion(self, task_id: str, content_id: str, duration: float, content: Optional[Content] = None):
        """Log task completion and update counters"""
        try:
            # Check if we can process this content
            self.check_limit(content_id)
            
            # Update counters
            if content_id != self.current_content_id:
                self.current_content_id = content_id
                self.chunks_processed_for_content = 0
                self.tasks_completed += 1
            
            self.chunks_processed_for_content += 1
            self.last_task_time = time.time()
            
            # Increment process count
            self.process_count += 1
            
            # Log task completion status
            self.worker_logger.info(
                f"Task completion status: {self.tasks_completed}/{self.tasks_per_run} tasks completed (Process total: {self.process_count})"
            )
            
            # Log task completion with session info
            from ..utils.logger import log_task_completion as log_completion
            log_completion(
                self.worker_type,
                task_id,
                content_id,
                duration,
                content.publish_date if content else None,
                self.tasks_completed,
                self.tasks_per_run,
                self.process_count
            )
            
        except WorkerRestartRequest:
            raise

    async def cleanup(self):
        """Clean up any resources before worker restart"""
        self.worker_logger.info(f"Cleaning up worker state after {self.tasks_completed} tasks")
        self.current_content_id = None
        self.chunks_processed_for_content = 0
        self.tasks_completed = 0
        
        try:
            if self.worker_type == 'transcribe':
                # Special cleanup for transcription tasks
                # This ensures model is properly unloaded and GPU memory is freed
                from ..processing.transcription import TranscriptionPipeline
                transcriber = TranscriptionPipeline(
                    implementation=self.config['processing']['transcription']['implementation'],
                    device=self.config['processing']['transcription']['device'],
                    logger=self.worker_logger,
                    quiet=self.quiet
                )
                await transcriber._cleanup_model(force_deep=True)
                
            # Clean up temp directories if they exist
            if self.worker_type in ['transcribe', 'extract_audio', 'stitch']:
                temp_dirs = {
                    'transcribe': Path("/tmp/whisper_chunks"),
                    'extract_audio': Path(self.config['storage']['nas']['mount_point']) / "temp",
                    'stitch': Path("/tmp/transcripts")
                }
                temp_dir = temp_dirs.get(self.worker_type)
                if temp_dir and temp_dir.exists():
                    try:
                        import shutil
                        shutil.rmtree(temp_dir)
                        self.worker_logger.info(f"Cleaned up temp directory: {temp_dir}")
                    except Exception as e:
                        self.worker_logger.warning(f"Error cleaning up temp directory: {e}")
                        
        except Exception as e:
            self.worker_logger.error(f"Error during worker cleanup: {e}")
            
        # Log cleanup completion
        self.worker_logger.info(f"Cleanup completed for {self.worker_type} worker")

    async def comprehensive_cleanup(self, stage: str = "pre-restart"):
        """Perform comprehensive cleanup including MPS, Python GC, and system memory"""
        try:
            self.worker_logger.info(f"Starting comprehensive cleanup - {stage}")
            
            # 1. Clean up MPS memory
            if self.worker_type == 'transcribe':
                try:
                    import torch
                    if hasattr(torch.mps, 'empty_cache'):
                        torch.mps.empty_cache()
                        self.worker_logger.info("Cleared MPS cache")
                except Exception as e:
                    self.worker_logger.warning(f"Error clearing MPS cache: {e}")
            
            # 2. Force Python garbage collection
            import gc
            gc.collect()
            self.worker_logger.info("Ran garbage collection")
            
            # 3. Clean up temp directories
            temp_dirs = {
                'transcribe': Path("/tmp/whisper_chunks"),
                'extract_audio': Path(self.config['storage']['nas']['mount_point']) / "temp",
                'stitch': Path("/tmp/transcripts")
            }
            temp_dir = temp_dirs.get(self.worker_type)
            if temp_dir and temp_dir.exists():
                try:
                    import shutil
                    shutil.rmtree(temp_dir)
                    self.worker_logger.info(f"Cleaned up temp directory: {temp_dir}")
                except Exception as e:
                    self.worker_logger.warning(f"Error cleaning up temp directory: {e}")
            
            self.worker_logger.info(f"Completed comprehensive cleanup - {stage}")
            
        except Exception as e:
            self.worker_logger.error(f"Error during comprehensive cleanup: {e}") 