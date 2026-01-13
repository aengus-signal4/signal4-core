"""
State Evaluator - Evaluates content state and determines necessary tasks.

This module handles the core "what needs to happen next" logic:
1. Check what files exist in S3 (via S3ContentChecker)
2. Reconcile database flags (via FlagReconciler)
3. Determine what tasks need to be created
4. Create those tasks (via TaskCreator)

This is the main entry point for state-driven task creation.
"""

import re
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from sqlalchemy.orm import Session
from sqlalchemy import text

from src.database.models import Content, ContentChunk, TaskQueue, EmbeddingSegment, Sentence
from src.storage.s3_utils import S3Storage, S3StorageConfig
from src.utils.logger import setup_worker_logger
from src.utils.version_utils import should_recreate_stitch_task

from src.processing.state.s3_content_checker import S3ContentChecker, ContentFileIndex
from src.processing.state.flag_reconciler import FlagReconciler
from .task_creator import TaskCreator

logger = setup_worker_logger('state_evaluator')


class StateEvaluator:
    """
    Evaluates content state and creates necessary tasks.

    This class orchestrates:
    1. Building file index from S3
    2. Reconciling database flags with S3 reality
    3. Determining what tasks are needed
    4. Creating those tasks

    It replaces the monolithic evaluate_content_state() method from PipelineManager.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize StateEvaluator.

        Args:
            config: Application configuration dict
        """
        self.config = config

        # Initialize S3 storage
        s3_config = S3StorageConfig(
            endpoint_url=config['storage']['s3']['endpoint_url'],
            access_key=config['storage']['s3']['access_key'],
            secret_key=config['storage']['s3']['secret_key'],
            bucket_name=config['storage']['s3']['bucket_name'],
            use_ssl=config['storage']['s3']['use_ssl']
        )
        self.s3_storage = S3Storage(s3_config)

        # Initialize sub-components
        self.s3_checker = S3ContentChecker(self.s3_storage)
        self.flag_reconciler = FlagReconciler(config, self.s3_storage)
        self.task_creator = TaskCreator(config)

    def get_current_stitch_version(self) -> str:
        """Get the current stitch version from config."""
        try:
            return self.config.get('processing', {}).get('stitch', {}).get('current_version', 'stitch_v1')
        except Exception:
            return 'stitch_v1'

    async def evaluate_content_state(
        self,
        session: Session,
        content: Content,
        db_task: Optional[TaskQueue] = None
    ) -> Dict[str, Any]:
        """
        Reconcile database state with S3 reality and create missing tasks.

        This is the main entry point for state evaluation.

        Args:
            session: Active database session
            content: Content object to evaluate
            db_task: Optional TaskQueue object that triggered evaluation

        Returns:
            Dict containing:
                - content_id: Content ID
                - flags_updated: List of flags that were updated
                - tasks_created: List of tasks that were created
                - tasks_blocked: List of tasks that were blocked (with reasons)
                - errors: List of errors encountered
        """
        logger.debug(f"Evaluating state for content {content.content_id}")

        # Refresh content to get latest state
        content_id = content.content_id
        session.expire_all()
        content = session.query(Content).filter_by(content_id=content_id).first()

        if not content:
            logger.error(f"Content {content_id} not found after refresh")
            return {
                'content_id': content_id,
                'flags_updated': [],
                'tasks_created': [],
                'tasks_blocked': [],
                'errors': ['Content not found after refresh']
            }

        results = {
            'content_id': content.content_id,
            'flags_updated': [],
            'tasks_created': [],
            'tasks_blocked': [],
            'errors': []
        }

        try:
            # Step 1: Build file index from S3
            file_index = self.s3_checker.get_content_file_index(content.content_id)

            if not file_index.all_files and not content.blocked_download:
                logger.debug(f"No files found in S3 for {content.content_id}")

            # Step 2: Reconcile database flags
            reconcile_result = self.flag_reconciler.reconcile_content_flags(
                session, content, file_index, commit=True
            )
            results['flags_updated'] = reconcile_result.flags_updated
            results['errors'].extend(reconcile_result.errors)

            # Refresh content after flag updates
            session.refresh(content)

            # Step 3: Determine and create necessary tasks
            project = content.projects[0] if content.projects else 'unknown'

            # Check if within project date range
            if not self.task_creator.is_content_within_project_date_range(content, project):
                logger.debug(f"Content {content.content_id} outside project date range")
                return results

            # Create tasks based on current state
            task_results = await self._create_necessary_tasks(
                session, content, file_index, project
            )
            results['tasks_created'] = task_results['created']
            results['tasks_blocked'] = task_results['blocked']
            results['errors'].extend(task_results['errors'])

            # Commit task creations
            if results['tasks_created']:
                session.commit()

            return results

        except Exception as e:
            import traceback
            error_msg = f"Error evaluating content state: {e}"
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
            results['errors'].append(error_msg)
            return results

    async def _create_necessary_tasks(
        self,
        session: Session,
        content: Content,
        file_index: ContentFileIndex,
        project: str
    ) -> Dict[str, List[str]]:
        """
        Determine and create necessary tasks based on content state.

        Returns dict with 'created', 'blocked', 'errors' lists.
        """
        result = {'created': [], 'blocked': [], 'errors': []}

        # Check download task needed
        if not file_index.has_source_files and not file_index.has_audio and not content.blocked_download:
            await self._maybe_create_download_task(session, content, file_index, project, result)

        # Check convert task needed
        if file_index.has_source_files and not file_index.has_audio:
            await self._maybe_create_convert_task(session, content, project, result)

        # Also create convert if audio exists but conversion incomplete
        should_be_converted = self.s3_checker.check_converted_state(file_index)
        if file_index.has_audio and not should_be_converted:
            logger.info(f"Audio exists but conversion incomplete for {content.content_id}")
            await self._maybe_create_convert_task(session, content, project, result)

        # If converted, check for diarize, transcribe, stitch, cleanup
        if should_be_converted:
            await self._create_post_convert_tasks(session, content, file_index, project, result)

        return result

    async def _maybe_create_download_task(
        self,
        session: Session,
        content: Content,
        file_index: ContentFileIndex,
        project: str,
        result: Dict[str, List[str]]
    ):
        """Create download task if appropriate."""
        # Check for too many completed downloads with no files (silent failure)
        completed_downloads = session.query(TaskQueue).filter(
            TaskQueue.content_id == content.content_id,
            TaskQueue.task_type == f"download_{content.platform}",
            TaskQueue.status == 'completed'
        ).count()

        if completed_downloads >= 3:
            logger.warning(f"Content {content.content_id} has {completed_downloads} completed downloads but no files. Blocking.")
            content.blocked_download = True
            content.last_updated = datetime.now(timezone.utc)
            session.add(content)
            session.commit()
            result['blocked'].append(f"download_{content.platform}: Silent failure after {completed_downloads} attempts")
            return

        # Check for conda run failure
        failed_task = session.query(TaskQueue).filter(
            TaskQueue.content_id == content.content_id,
            TaskQueue.task_type == f"download_{content.platform}",
            TaskQueue.status == 'failed'
        ).first()

        if failed_task and failed_task.error and "conda run" in str(failed_task.error):
            logger.debug(f"Skipping download task for {content.content_id} due to conda run failure")
            result['blocked'].append(f"download_{content.platform}: conda run failure")
            return

        # Create the task
        task_id, block_reason = await self.task_creator.create_task_if_not_exists(
            session,
            content_id=content.content_id,
            task_type=f"download_{content.platform}",
            input_data={'project': project},
            content=content
        )
        if task_id:
            result['created'].append(f"download_{content.platform}")
        elif block_reason:
            result['blocked'].append(f"download_{content.platform}: {block_reason}")

    async def _maybe_create_convert_task(
        self,
        session: Session,
        content: Content,
        project: str,
        result: Dict[str, List[str]]
    ):
        """Create convert task if appropriate."""
        task_id, block_reason = await self.task_creator.create_task_if_not_exists(
            session,
            content_id=content.content_id,
            task_type='convert',
            input_data={'project': project},
            content=content
        )
        if task_id:
            result['created'].append('convert')
        elif block_reason:
            result['blocked'].append(f"convert: {block_reason}")

    async def _create_post_convert_tasks(
        self,
        session: Session,
        content: Content,
        file_index: ContentFileIndex,
        project: str,
        result: Dict[str, List[str]]
    ):
        """Create tasks that depend on conversion being complete."""

        # Diarize task
        if not file_index.has_diarization:
            if content.meta_data and content.meta_data.get('diarization_ignored'):
                logger.info(f"Skipping diarize for {content.content_id} - diarization_ignored flag set")
            else:
                task_id, block_reason = await self.task_creator.create_task_if_not_exists(
                    session,
                    content_id=content.content_id,
                    task_type='diarize',
                    input_data={'project': project},
                    content=content
                )
                if task_id:
                    result['created'].append('diarize')
                elif block_reason:
                    result['blocked'].append(f"diarize: {block_reason}")

        # Transcribe tasks (require diarization or diarization_ignored)
        diarization_ready = file_index.has_diarization or (
            content.meta_data and content.meta_data.get('diarization_ignored')
        )

        if diarization_ready:
            await self._create_transcribe_tasks(session, content, file_index, project, result)

        # Check for missing chunk extractions
        await self._check_missing_extractions(session, content, file_index, project, result)

        # Stitch task
        await self._maybe_create_stitch_task(session, content, file_index, project, result)

        # Cleanup task
        await self._maybe_create_cleanup_task(session, content, file_index, project, result)

    async def _create_transcribe_tasks(
        self,
        session: Session,
        content: Content,
        file_index: ContentFileIndex,
        project: str,
        result: Dict[str, List[str]]
    ):
        """Create transcribe tasks for chunks without transcripts."""
        chunks_with_audio = file_index.get_chunk_indices_with_audio()
        chunks_with_transcripts = file_index.get_chunk_indices_with_transcripts()

        for chunk_index in chunks_with_audio:
            if chunk_index in chunks_with_transcripts:
                continue

            # Check for failed task with audio download error
            failed_task = session.query(TaskQueue).filter(
                TaskQueue.content_id == content.content_id,
                TaskQueue.task_type == 'transcribe',
                TaskQueue.status == 'failed',
                text("input_data->>'chunk_index' = :chunk_index").params(chunk_index=str(chunk_index))
            ).first()

            if failed_task and failed_task.error and "Failed to download audio chunk" in str(failed_task.error):
                logger.debug(f"Skipping transcribe for chunk {chunk_index} - audio download failure")
                continue

            task_id, block_reason = await self.task_creator.create_task_if_not_exists(
                session,
                content_id=content.content_id,
                task_type='transcribe',
                input_data={'project': project, 'chunk_index': chunk_index},
                content=content
            )
            if task_id:
                result['created'].append(f'transcribe_{chunk_index}')
            elif block_reason:
                result['blocked'].append(f"transcribe_{chunk_index}: {block_reason}")

    async def _check_missing_extractions(
        self,
        session: Session,
        content: Content,
        file_index: ContentFileIndex,
        project: str,
        result: Dict[str, List[str]]
    ):
        """Check for and handle missing chunk extractions."""
        if content.is_transcribed:
            return

        chunks = session.query(ContentChunk).filter_by(content_id=content.id).all()
        if not chunks:
            return

        missing = [c for c in chunks if c.extraction_status != 'completed']
        if missing:
            chunk_indices = [c.chunk_index for c in missing]
            logger.info(f"Content {content.content_id} has missing extractions: {chunk_indices}")

            task_id, block_reason = await self.task_creator.create_task_if_not_exists(
                session,
                content_id=content.content_id,
                task_type='convert',
                input_data={'project': project},
                content=content
            )
            if task_id:
                result['created'].append('convert')
                logger.info(f"Created convert task to re-extract missing chunks for {content.content_id}")
            elif block_reason:
                result['blocked'].append(f"convert: {block_reason}")

    async def _maybe_create_stitch_task(
        self,
        session: Session,
        content: Content,
        file_index: ContentFileIndex,
        project: str,
        result: Dict[str, List[str]]
    ):
        """Create stitch task if appropriate."""
        # Check prerequisites
        chunks = session.query(ContentChunk).filter_by(content_id=content.id).all()
        all_transcribed = bool(chunks) and all(c.transcription_status == 'completed' for c in chunks)

        if not all_transcribed or not file_index.has_diarization:
            return

        current_version = self.get_current_stitch_version()
        needs_stitch = (
            not content.is_stitched or
            should_recreate_stitch_task(current_version, content.stitch_version)
        )

        logger.debug(f"Stitch check for {content.content_id}: all_transcribed={all_transcribed}, "
                    f"diarization={file_index.has_diarization}, is_stitched={content.is_stitched}, "
                    f"version={content.stitch_version}, needs_stitch={needs_stitch}")

        if needs_stitch:
            task_id, block_reason = await self.task_creator.create_task_if_not_exists(
                session,
                content_id=content.content_id,
                task_type='stitch',
                input_data={'project': project},
                content=content
            )
            if task_id:
                result['created'].append('stitch')
            elif block_reason:
                result['blocked'].append(f"stitch: {block_reason}")

    async def _maybe_create_cleanup_task(
        self,
        session: Session,
        content: Content,
        file_index: ContentFileIndex,
        project: str,
        result: Dict[str, List[str]]
    ):
        """Create cleanup task if appropriate."""
        needs_cleanup = content.is_stitched and not content.is_compressed

        logger.debug(f"Cleanup check for {content.content_id}: is_stitched={content.is_stitched}, "
                    f"is_compressed={content.is_compressed}, needs_cleanup={needs_cleanup}")

        if needs_cleanup:
            task_id, block_reason = await self.task_creator.create_task_if_not_exists(
                session,
                content_id=content.content_id,
                task_type='cleanup',
                input_data={'project': project},
                content=content
            )
            if task_id:
                result['created'].append('cleanup')
            elif block_reason:
                result['blocked'].append(f"cleanup: {block_reason}")

    async def check_and_create_stitch_task(
        self,
        session: Session,
        content: Content
    ) -> tuple[bool, Optional[int]]:
        """
        Legacy method for checking stitch readiness.

        Returns (task_created, task_id).
        """
        logger.debug(f"Checking stitch readiness for {content.content_id}")

        current_version = self.get_current_stitch_version()

        # Already stitched with current version?
        if content.is_stitched and not should_recreate_stitch_task(current_version, content.stitch_version):
            return False, None

        # Check all chunks transcribed
        total = session.query(ContentChunk).filter_by(content_id=content.id).count()
        if total == 0:
            return False, None

        completed = session.query(ContentChunk).filter(
            ContentChunk.content_id == content.id,
            ContentChunk.transcription_status == 'completed'
        ).count()

        if total != completed:
            return False, None

        if not content.is_diarized:
            return False, None

        logger.info(f"Content {content.content_id} ready for stitching")
        project = content.projects[0] if content.projects else 'unknown'

        task_id, _ = await self.task_creator.create_task_if_not_exists(
            session,
            content_id=content.content_id,
            task_type='stitch',
            input_data={'project': project},
            content=content
        )

        return bool(task_id), task_id
