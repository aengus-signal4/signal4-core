"""
Content state reconciliation - ensures database matches S3 reality.

This is the main orchestration class that:
1. Checks what files actually exist in S3
2. Updates database flags to match reality
3. Creates any tasks needed based on actual state
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from datetime import datetime, timezone
import logging
import re

from sqlalchemy.orm import Session

from src.database.models import Content, ContentChunk, Sentence
from src.processing.content_state.file_checker import ContentFileChecker, ContentFiles
from src.processing.content_state.flag_updater import FlagUpdater
from src.storage.s3_utils import S3Storage

if TYPE_CHECKING:
    from src.processing.task_creation.factory import TaskFactory

logger = logging.getLogger(__name__)


@dataclass
class ReconciliationResult:
    """Result of reconciling content state."""
    content_id: str
    flags_updated: List[str] = field(default_factory=list)
    tasks_created: List[str] = field(default_factory=list)
    tasks_blocked: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    outcome_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'content_id': self.content_id,
            'flags_updated': self.flags_updated,
            'tasks_created': self.tasks_created,
            'tasks_blocked': self.tasks_blocked,
            'errors': self.errors,
            'outcome_message': self.outcome_message,
        }


class StateReconciler:
    """
    Reconciles database state with S3 storage reality.

    This class:
    1. Checks what files actually exist in S3
    2. Updates database flags to match reality
    3. Creates any tasks needed based on actual state
    """

    def __init__(
        self,
        config: Dict[str, Any],
        file_checker: ContentFileChecker,
        flag_updater: FlagUpdater,
        task_factory: 'TaskFactory',
        s3_storage: S3Storage
    ):
        self.config = config
        self.file_checker = file_checker
        self.flag_updater = flag_updater
        self.task_factory = task_factory
        self.s3_storage = s3_storage

    async def reconcile(
        self,
        session: Session,
        content: Content,
        skip_task_creation: bool = False
    ) -> ReconciliationResult:
        """
        Reconcile a single content item's state.

        Args:
            session: Active database session
            content: Content object to reconcile
            skip_task_creation: If True, only update flags, don't create tasks

        Returns:
            ReconciliationResult with actions taken
        """
        # Refresh content to ensure we have latest state
        content_id = content.content_id
        session.expire_all()
        content = session.query(Content).filter_by(content_id=content_id).first()

        if not content:
            return ReconciliationResult(
                content_id=content_id,
                errors=['Content not found after refresh']
            )

        result = ReconciliationResult(content_id=content_id)

        try:
            # Step 1: Get file listing from S3
            files = self.file_checker.check_from_prefix(self.s3_storage, content_id)

            if not files.all_files and not self._has_known_state(content):
                # S3 error or truly empty - don't modify state
                logger.warning(f"No files found and no known state for {content_id} - skipping reconciliation")
                return result

            logger.debug(f"Found {len(files.all_files)} files for {content_id}")

            # Step 2: Update database flags
            flag_result = self.flag_updater.update_all_flags(session, content, files)
            result.flags_updated = flag_result.flags_updated
            result.errors.extend(flag_result.errors)

            # Refresh content after flag updates
            session.refresh(content)

            # Step 3: Create missing tasks if not skipped
            if not skip_task_creation:
                await self._create_missing_tasks(session, content, files, result)

            # Build outcome message
            result.outcome_message = self._build_outcome_message(result)

            return result

        except Exception as e:
            logger.error(f"Error reconciling {content_id}: {e}", exc_info=True)
            result.errors.append(str(e))
            return result

    def _has_known_state(self, content: Content) -> bool:
        """Check if content has any known processing state."""
        return (
            content.is_downloaded or
            content.is_converted or
            content.is_transcribed or
            content.is_diarized or
            content.is_stitched or
            content.is_embedded or
            content.is_compressed
        )

    async def _create_missing_tasks(
        self,
        session: Session,
        content: Content,
        files: ContentFiles,
        result: ReconciliationResult
    ) -> None:
        """Create any tasks needed based on current state."""
        project = content.projects[0] if content.projects else 'unknown'

        # Check project date range
        if not self._is_content_within_project_date_range(content, project):
            logger.debug(
                f"Skipping task creation for {content.content_id} - "
                f"outside project {project} date range"
            )
            return

        # Download task
        if not files.source_exists and not files.audio_exists and not content.blocked_download:
            await self._create_download_task(session, content, files, project, result)

        # Convert task
        if files.source_exists and not files.audio_exists:
            task_id, reason = await self.task_factory.create_task(
                session, content, 'convert', project
            )
            if task_id:
                result.tasks_created.append('convert')
            elif reason:
                result.tasks_blocked.append(f"convert: {reason}")

        # Also create convert if audio exists but conversion incomplete
        if files.audio_exists and not content.is_converted:
            logger.info(f"Audio exists but conversion incomplete for {content.content_id}")
            task_id, reason = await self.task_factory.create_task(
                session, content, 'convert', project
            )
            if task_id:
                result.tasks_created.append('convert')
            elif reason:
                result.tasks_blocked.append(f"convert: {reason}")

        # Diarize task
        if content.is_converted and not files.diarization_exists:
            # Check diarization_ignored flag
            if content.meta_data and content.meta_data.get('diarization_ignored'):
                logger.info(f"Skipping diarize for {content.content_id} - diarization_ignored flag")
            else:
                task_id, reason = await self.task_factory.create_task(
                    session, content, 'diarize', project
                )
                if task_id:
                    result.tasks_created.append('diarize')
                elif reason:
                    result.tasks_blocked.append(f"diarize: {reason}")

        # Transcribe tasks for chunks
        if content.is_converted and (files.diarization_exists or
                                      (content.meta_data and content.meta_data.get('diarization_ignored'))):
            await self._create_transcribe_tasks(session, content, files, project, result)

        # Check for missing chunk extractions
        await self._check_missing_extractions(session, content, files, project, result)

        # Stitch task
        await self._create_stitch_task_if_needed(session, content, files, project, result)

        # Cleanup task
        if content.is_stitched and not content.is_compressed:
            task_id, reason = await self.task_factory.create_task(
                session, content, 'cleanup', project
            )
            if task_id:
                result.tasks_created.append('cleanup')
            elif reason:
                result.tasks_blocked.append(f"cleanup: {reason}")

        # Commit any tasks created
        if result.tasks_created:
            session.commit()
            logger.debug(f"Committed {len(result.tasks_created)} tasks for {content.content_id}")

    async def _create_download_task(
        self,
        session: Session,
        content: Content,
        files: ContentFiles,
        project: str,
        result: ReconciliationResult
    ) -> None:
        """Create download task with silent failure detection."""
        from src.database.models import TaskQueue

        # Check for silent download failures
        completed_downloads = session.query(TaskQueue).filter(
            TaskQueue.content_id == content.content_id,
            TaskQueue.task_type == f"download_{content.platform}",
            TaskQueue.status == 'completed'
        ).count()

        if completed_downloads >= 3:
            logger.warning(
                f"Content {content.content_id} has {completed_downloads} completed downloads "
                "but no files - blocking further attempts"
            )
            content.blocked_download = True
            content.last_updated = datetime.now(timezone.utc)
            session.add(content)
            session.commit()
            result.flags_updated.append('blocked_download')
            result.tasks_blocked.append(
                f"download_{content.platform}: Silent failure after {completed_downloads} attempts"
            )
            return

        # Check for conda run failures
        failed_task = session.query(TaskQueue).filter(
            TaskQueue.content_id == content.content_id,
            TaskQueue.task_type == f"download_{content.platform}",
            TaskQueue.status == 'failed'
        ).first()

        if failed_task and failed_task.error:
            if "conda run" in str(failed_task.error) and "failed" in str(failed_task.error):
                logger.debug(f"Skipping download for {content.content_id} - conda run failure")
                return

        # Create download task
        task_id, reason = await self.task_factory.create_task(
            session, content, f"download_{content.platform}", project
        )
        if task_id:
            result.tasks_created.append(f"download_{content.platform}")
        elif reason:
            result.tasks_blocked.append(f"download_{content.platform}: {reason}")

    async def _create_transcribe_tasks(
        self,
        session: Session,
        content: Content,
        files: ContentFiles,
        project: str,
        result: ReconciliationResult
    ) -> None:
        """Create transcribe tasks for chunks needing transcription."""
        from src.database.models import TaskQueue
        from sqlalchemy import text

        chunks_needing_transcription = files.get_chunks_needing_transcription()

        for chunk_index in chunks_needing_transcription:
            # Check for failed audio download
            failed_task = session.query(TaskQueue).filter(
                TaskQueue.content_id == content.content_id,
                TaskQueue.task_type == 'transcribe',
                TaskQueue.status == 'failed',
                text("input_data->>'chunk_index' = :chunk_index").params(chunk_index=str(chunk_index))
            ).first()

            if failed_task and failed_task.error:
                if "Failed to download audio chunk" in str(failed_task.error):
                    logger.debug(f"Skipping transcribe for chunk {chunk_index} - audio download failure")
                    continue

            task_id, reason = await self.task_factory.create_task(
                session, content, 'transcribe', project,
                input_data={'chunk_index': chunk_index}
            )
            if task_id:
                result.tasks_created.append(f'transcribe_{chunk_index}')
            elif reason:
                result.tasks_blocked.append(f"transcribe_{chunk_index}: {reason}")

    async def _check_missing_extractions(
        self,
        session: Session,
        content: Content,
        files: ContentFiles,
        project: str,
        result: ReconciliationResult
    ) -> None:
        """Check for and handle missing chunk extractions."""
        if content.is_transcribed:
            return

        chunks = session.query(ContentChunk).filter_by(content_id=content.id).all()
        if not chunks:
            return

        missing_extraction = [c for c in chunks if c.extraction_status != 'completed']
        if missing_extraction:
            chunk_indices = [c.chunk_index for c in missing_extraction]
            logger.info(
                f"Content {content.content_id} has missing chunk extractions: {chunk_indices}"
            )

            task_id, reason = await self.task_factory.create_task(
                session, content, 'convert', project
            )
            if task_id:
                result.tasks_created.append('convert')
                logger.info(f"Created convert task {task_id} for missing chunks")
            elif reason:
                result.tasks_blocked.append(f"convert: {reason}")

    async def _create_stitch_task_if_needed(
        self,
        session: Session,
        content: Content,
        files: ContentFiles,
        project: str,
        result: ReconciliationResult
    ) -> None:
        """Create stitch task if content is ready."""
        from src.utils.version_utils import should_recreate_stitch_task

        current_version = self.flag_updater.get_current_stitch_version()

        # Check all chunks transcribed
        chunks = session.query(ContentChunk).filter_by(content_id=content.id).all()
        all_chunks_transcribed = chunks and all(
            c.transcription_status == 'completed' for c in chunks
        )

        needs_stitch = (
            all_chunks_transcribed and
            files.diarization_exists and
            (not content.is_stitched or
             should_recreate_stitch_task(current_version, content.stitch_version))
        )

        logger.debug(
            f"Stitch decision for {content.content_id}: "
            f"all_transcribed={all_chunks_transcribed}, diarization={files.diarization_exists}, "
            f"is_stitched={content.is_stitched}, version={content.stitch_version}, "
            f"needs_stitch={needs_stitch}"
        )

        if needs_stitch:
            task_id, reason = await self.task_factory.create_task(
                session, content, 'stitch', project
            )
            if task_id:
                result.tasks_created.append('stitch')
            elif reason:
                result.tasks_blocked.append(f"stitch: {reason}")

        # Also create stitch if file exists but no DB records
        elif files.stitched_exists and not content.is_stitched:
            sentence_count = session.query(Sentence).filter(
                Sentence.content_id == content.id,
                Sentence.stitch_version == current_version
            ).count()

            if sentence_count == 0 and all_chunks_transcribed and files.diarization_exists:
                logger.warning(
                    f"Found transcript_diarized.json but no Sentence records for {content.content_id}"
                )
                task_id, reason = await self.task_factory.create_task(
                    session, content, 'stitch', project
                )
                if task_id:
                    result.tasks_created.append('stitch')
                elif reason:
                    result.tasks_blocked.append(f"stitch: {reason}")

    def _is_content_within_project_date_range(self, content: Content, project: str) -> bool:
        """Check if content's publish_date falls within project's date range."""
        if not content.publish_date:
            return True

        try:
            project_config = self.config.get('active_projects', {}).get(project, {})
            if not project_config:
                return True

            if not project_config.get('enabled', True):
                return False

            start_date_str = project_config.get('start_date')
            end_date_str = project_config.get('end_date')

            if start_date_str:
                start_date = datetime.strptime(start_date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
                if content.publish_date < start_date:
                    return False

            if end_date_str:
                from datetime import timedelta
                end_date = datetime.strptime(end_date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
                end_date_exclusive = end_date + timedelta(days=1)
                if content.publish_date >= end_date_exclusive:
                    return False

            return True

        except Exception as e:
            logger.warning(f"Error checking date range for {content.content_id}: {e}")
            return True

    def _build_outcome_message(self, result: ReconciliationResult) -> str:
        """Build human-readable outcome message."""
        parts = []
        if result.flags_updated:
            parts.append(f"Updated {', '.join(result.flags_updated)}")
        if result.tasks_created:
            parts.append(f"Created {', '.join(result.tasks_created)}")
        if result.tasks_blocked:
            parts.append(f"Blocked {', '.join(result.tasks_blocked)}")
        return '. '.join(parts) if parts else "No changes needed"
