"""
Database flag updater for content state.

Updates content database flags based on file existence in S3,
ensuring database state matches storage reality.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone
import logging

from sqlalchemy.orm import Session

from src.database.models import Content, ContentChunk, EmbeddingSegment, Sentence
from src.processing.content_state.file_checker import ContentFiles
from src.utils.version_utils import should_recreate_stitch_task

logger = logging.getLogger(__name__)


@dataclass
class FlagUpdateResult:
    """Result of flag update operations."""
    flags_updated: List[str]
    errors: List[str]

    def __bool__(self):
        return len(self.flags_updated) > 0


class FlagUpdater:
    """Updates content database flags based on file existence."""

    def __init__(self, config: Dict[str, Any], s3_storage=None):
        self.config = config
        self.s3_storage = s3_storage
        self._safe_models_english = config.get('processing', {}).get(
            'transcription', {}
        ).get('safe_models_english', [])
        self._safe_models_other = config.get('processing', {}).get(
            'transcription', {}
        ).get('safe_models_other', [])

    def get_current_stitch_version(self) -> str:
        """Get current stitch version from config."""
        try:
            return self.config.get('processing', {}).get(
                'stitch', {}
            ).get('current_version', 'stitch_v1')
        except Exception as e:
            logger.warning(f"Failed to get stitch version from config: {e}, using default 'stitch_v1'")
            return 'stitch_v1'

    def update_all_flags(
        self,
        session: Session,
        content: Content,
        files: ContentFiles
    ) -> FlagUpdateResult:
        """
        Update all content flags based on file existence.

        Args:
            session: Active database session
            content: Content object to update
            files: ContentFiles with file existence info

        Returns:
            FlagUpdateResult with list of updated flags
        """
        result = FlagUpdateResult(flags_updated=[], errors=[])

        try:
            # Update each flag
            if flag := self.update_download_flag(session, content, files):
                result.flags_updated.append(flag)

            if flag := self.update_convert_flag(session, content, files):
                result.flags_updated.append(flag)

            # Update chunk status (this modifies chunks, not content flags directly)
            chunk_updates = self.update_chunk_status(session, content, files)
            if chunk_updates:
                result.flags_updated.extend(chunk_updates)

            if flag := self.update_transcription_flag(session, content, files):
                result.flags_updated.append(flag)

            if flag := self.update_diarization_flag(session, content, files):
                result.flags_updated.append(flag)

            if flag := self.update_stitch_flag(session, content, files):
                result.flags_updated.append(flag)

            if flag := self.update_embedded_flag(session, content, files):
                result.flags_updated.append(flag)

            if flag := self.update_compressed_flag(session, content, files):
                result.flags_updated.append(flag)

            # Commit if any updates were made
            if result.flags_updated:
                content.last_updated = datetime.now(timezone.utc)
                session.add(content)
                session.commit()
                session.refresh(content)
                logger.debug(f"Updated content state for {content.content_id}: {', '.join(result.flags_updated)}")

        except Exception as e:
            error_msg = f"Error updating flags for {content.content_id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            result.errors.append(error_msg)

        return result

    def update_download_flag(
        self,
        session: Session,
        content: Content,
        files: ContentFiles
    ) -> Optional[str]:
        """Update is_downloaded flag. Returns flag name if updated."""
        should_be_downloaded = files.is_downloadable

        if should_be_downloaded != content.is_downloaded:
            logger.debug(
                f"State change for {content.content_id}: "
                f"is_downloaded {content.is_downloaded} -> {should_be_downloaded}"
            )
            logger.debug(
                f"  Reason: source={files.source_exists}, audio={files.audio_exists}, "
                f"manifest={files.storage_manifest_exists}"
            )

            content.is_downloaded = should_be_downloaded
            content.last_updated = datetime.now(timezone.utc)
            session.add(content)
            session.commit()
            session.refresh(content)
            return 'is_downloaded'
        return None

    def update_convert_flag(
        self,
        session: Session,
        content: Content,
        files: ContentFiles
    ) -> Optional[str]:
        """Update is_converted flag. Returns flag name if updated."""
        should_be_converted = False

        if files.audio_exists:
            # Check if chunk audio files exist
            if files.has_chunk_audio:
                should_be_converted = True
            else:
                # No chunk audio - check if transcript FILES exist in S3 (post-cleanup case)
                chunks = session.query(ContentChunk).filter_by(content_id=content.id).all()
                if chunks:
                    # Check if any transcript files exist in S3
                    if files.has_chunk_transcripts:
                        should_be_converted = True
                        logger.debug(
                            f"No chunk audio but transcript files exist for {content.content_id} - "
                            "assuming post-cleanup"
                        )
                    else:
                        logger.warning(
                            f"Audio exists but no chunk audio or transcript files for "
                            f"{content.content_id} - conversion incomplete"
                        )
                else:
                    logger.warning(
                        f"Audio exists but no chunks in database for {content.content_id} - "
                        "conversion incomplete"
                    )

        if should_be_converted != content.is_converted:
            logger.debug(
                f"State change for {content.content_id}: "
                f"is_converted {content.is_converted} -> {should_be_converted}"
            )
            content.is_converted = should_be_converted
            content.last_updated = datetime.now(timezone.utc)
            session.add(content)
            session.commit()
            session.refresh(content)
            return 'is_converted'
        return None

    def update_chunk_status(
        self,
        session: Session,
        content: Content,
        files: ContentFiles
    ) -> List[str]:
        """
        Update chunk extraction and transcription status based on files.

        Returns list of chunk-related updates made.
        """
        updates = []
        chunks = session.query(ContentChunk).filter_by(content_id=content.id).all()

        for chunk in chunks:
            chunk_index = chunk.chunk_index
            prefix = files.prefix

            # Check file existence
            has_audio = chunk_index in files.chunk_indices_with_audio
            has_transcript = chunk_index in files.chunk_indices_with_transcript

            # Update extraction status
            new_extraction_status = 'completed' if has_audio else 'pending'
            if chunk.extraction_status != new_extraction_status:
                # Allow status changes if content is not compressed or we're upgrading status
                allow_change = (
                    not files.storage_manifest_exists or
                    new_extraction_status == 'completed' or
                    (files.storage_manifest_exists and not has_transcript)
                )

                if allow_change:
                    chunk.extraction_status = new_extraction_status
                    updates.append(f'chunk_{chunk_index}_extraction')
                    logger.debug(f"Updated chunk {chunk_index} extraction status to {new_extraction_status}")

            # Update transcription status with safe model check
            is_safe_model = self._check_safe_model(content, chunk, files)
            new_transcription_status = 'completed' if (has_transcript and is_safe_model) else 'pending'

            if chunk.transcription_status != new_transcription_status:
                if chunk.transcription_status == 'completed' and new_transcription_status == 'pending':
                    if not has_transcript:
                        logger.warning(
                            f"Transcript file missing in S3 for {content.content_id} chunk {chunk_index} "
                            "but was marked as completed. Resetting to pending."
                        )
                    elif not is_safe_model:
                        logger.debug(
                            f"Transcript exists but model '{chunk.transcribed_with}' not in safe models "
                            f"for {content.content_id} chunk {chunk_index}. Resetting to pending."
                        )

                chunk.transcription_status = new_transcription_status
                updates.append(f'chunk_{chunk_index}_transcription')
                logger.debug(
                    f"Updated chunk {chunk_index} transcription status to {new_transcription_status} "
                    f"(transcript={'exists' if has_transcript else 'missing'}, safe_model={is_safe_model})"
                )

        if updates:
            session.flush()

        return updates

    def _check_safe_model(
        self,
        content: Content,
        chunk: ContentChunk,
        files: ContentFiles
    ) -> bool:
        """Check if chunk was transcribed with a safe model."""
        has_transcript = chunk.chunk_index in files.chunk_indices_with_transcript

        # Handle legacy transcripts with no model information
        if not chunk.transcribed_with and has_transcript:
            # Legacy transcript - accept it
            if not hasattr(chunk, '_legacy_checked'):
                chunk.transcribed_with = 'legacy_whisper'
                chunk._legacy_checked = True
                logger.debug(
                    f"Found legacy transcript for {content.content_id} chunk {chunk.chunk_index} - "
                    "setting transcribed_with='legacy_whisper'"
                )
            return True

        if not chunk.transcribed_with:
            return False

        # Legacy whisper is always safe
        if chunk.transcribed_with == 'legacy_whisper':
            return True

        # Determine language
        is_english = content.main_language and content.main_language.lower().startswith('en')

        # Helper to check if model matches safe list
        def is_model_safe(model_name: str, safe_list: list) -> bool:
            if model_name in safe_list:
                return True
            # Strip language suffix (e.g., whisper_mlx_turbo_fr -> whisper_mlx_turbo)
            if '_' in model_name:
                base_model = '_'.join(model_name.rsplit('_', 1)[:-1])
                suffix = model_name.rsplit('_', 1)[-1]
                if len(suffix) in (2, 3) and suffix.isalpha() and suffix.islower():
                    return base_model in safe_list
            return False

        # Check appropriate safe model list based on language
        if is_english:
            return is_model_safe(chunk.transcribed_with, self._safe_models_english)
        else:
            return is_model_safe(chunk.transcribed_with, self._safe_models_other)

    def update_transcription_flag(
        self,
        session: Session,
        content: Content,
        files: ContentFiles
    ) -> Optional[str]:
        """Update is_transcribed flag based on chunk status."""
        chunks = session.query(ContentChunk).filter_by(content_id=content.id).all()

        all_chunks_transcribed = False
        if chunks:
            all_chunks_transcribed = all(
                chunk.transcription_status == 'completed' for chunk in chunks
            )
            logger.debug(
                f"Checking transcription for {content.content_id}: "
                f"{sum(1 for c in chunks if c.transcription_status == 'completed')} "
                f"completed out of {len(chunks)} chunks"
            )

        if all_chunks_transcribed != content.is_transcribed:
            content.is_transcribed = all_chunks_transcribed
            content.last_updated = datetime.now(timezone.utc)
            session.add(content)
            session.commit()
            session.refresh(content)
            logger.debug(
                f"Updated is_transcribed to {all_chunks_transcribed} for {content.content_id} "
                f"(all {len(chunks)} chunks have transcripts)"
            )
            return 'is_transcribed'
        return None

    def update_diarization_flag(
        self,
        session: Session,
        content: Content,
        files: ContentFiles
    ) -> Optional[str]:
        """Update is_diarized flag."""
        if files.diarization_exists != content.is_diarized:
            content.is_diarized = files.diarization_exists
            return 'is_diarized'
        return None

    def update_stitch_flag(
        self,
        session: Session,
        content: Content,
        files: ContentFiles
    ) -> Optional[str]:
        """Update is_stitched flag with version check."""
        current_version = self.get_current_stitch_version()

        if not files.stitched_exists and not content.is_stitched:
            return None

        # Check for Sentence records with current version
        sentence_count = session.query(Sentence).filter(
            Sentence.content_id == content.id,
            Sentence.stitch_version == current_version
        ).count()
        sentences_exist = sentence_count > 0

        # Check what versions exist
        all_versions = session.query(Sentence.stitch_version).filter(
            Sentence.content_id == content.id
        ).distinct().all()
        version_list = [v[0] for v in all_versions] if all_versions else []
        any_sentences_exist = len(version_list) > 0

        # Determine if content should be considered stitched
        if files.stitched_exists and any_sentences_exist and not sentences_exist:
            # File exists with old version sentences - needs re-stitching
            logger.debug(
                f"Found sentences for {content.content_id} with version(s) {version_list}, "
                f"but current version is '{current_version}'. Content needs re-stitching."
            )
            should_be_stitched = False
        else:
            should_be_stitched = files.stitched_exists and sentences_exist

        logger.debug(
            f"Stitch state for {content.content_id}: file={files.stitched_exists}, "
            f"db_records={sentences_exist}, current_version='{current_version}', "
            f"db_versions={version_list}, should_be_stitched={should_be_stitched}"
        )

        flags_updated = []

        # Update stitch_version if needed
        if should_be_stitched and should_recreate_stitch_task(current_version, content.stitch_version):
            content.stitch_version = current_version
            flags_updated.append('stitch_version')

        # Update is_stitched if needed
        if should_be_stitched != content.is_stitched:
            content.is_stitched = should_be_stitched
            content.stitch_version = current_version if should_be_stitched else 'none'
            flags_updated.append('is_stitched')

        if flags_updated:
            content.last_updated = datetime.now(timezone.utc)
            session.add(content)
            session.commit()
            session.refresh(content)

        return 'is_stitched' if 'is_stitched' in flags_updated else (
            'stitch_version' if 'stitch_version' in flags_updated else None
        )

    def update_embedded_flag(
        self,
        session: Session,
        content: Content,
        files: ContentFiles
    ) -> Optional[str]:
        """Update is_embedded flag."""
        from src.processing_steps.stitch_steps.stage14_segment import get_current_segment_version

        if files.semantic_segments_exist:
            # Verify EmbeddingSegment records exist
            embedding_segments_exist = session.query(EmbeddingSegment).filter(
                EmbeddingSegment.content_id == content.id
            ).count() > 0

            should_be_embedded = files.semantic_segments_exist and embedding_segments_exist
        else:
            should_be_embedded = False

        if should_be_embedded != content.is_embedded:
            content.is_embedded = should_be_embedded

            # Update segment_version in meta_data if embedded
            if should_be_embedded:
                current_segment_version = get_current_segment_version()
                meta_data = dict(content.meta_data) if content.meta_data else {}
                meta_data['segment_version'] = current_segment_version
                content.meta_data = meta_data

            content.last_updated = datetime.now(timezone.utc)
            session.add(content)
            session.commit()
            session.refresh(content)
            return 'is_embedded'
        return None

    def update_compressed_flag(
        self,
        session: Session,
        content: Content,
        files: ContentFiles
    ) -> Optional[str]:
        """Update is_compressed flag."""
        if files.storage_manifest_exists != content.is_compressed:
            content.is_compressed = files.storage_manifest_exists
            return 'is_compressed'
        return None
