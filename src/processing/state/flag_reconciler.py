"""
Flag Reconciler - Updates database flags based on actual S3 state.

This module handles the reconciliation of content database flags (is_downloaded,
is_converted, is_transcribed, etc.) based on what files actually exist in S3.

The key principle: S3 is the source of truth for file existence, and database
flags should reflect that reality.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from src.database.models import Content, ContentChunk, EmbeddingSegment, Sentence
from src.storage.s3_utils import S3Storage
from src.utils.logger import setup_worker_logger
from src.utils.version_utils import should_recreate_stitch_task
from src.processing_steps.stitch_steps.stage14_segment import get_current_segment_version

from .s3_content_checker import ContentFileIndex

logger = setup_worker_logger('flag_reconciler')


@dataclass
class ReconciliationResult:
    """Result of a flag reconciliation operation."""
    content_id: str
    flags_updated: List[str] = field(default_factory=list)
    chunks_updated: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def had_updates(self) -> bool:
        return bool(self.flags_updated) or self.chunks_updated > 0


class FlagReconciler:
    """
    Reconciles database flags with S3 file reality.

    This class is responsible for:
    1. Checking what files exist in S3 (via ContentFileIndex)
    2. Updating Content flags to match reality
    3. Updating ContentChunk status to match reality
    4. Handling version-aware reconciliation (stitch_version, segment_version)
    """

    def __init__(self, config: Dict[str, Any], s3_storage: S3Storage):
        """
        Initialize the FlagReconciler.

        Args:
            config: Application configuration dict
            s3_storage: S3Storage instance for reading transcript files
        """
        self.config = config
        self.s3_storage = s3_storage

        # Load safe transcription models from config
        self.safe_transcription_models_english = config.get('processing', {}).get(
            'transcription', {}
        ).get('safe_models_english', [])
        self.safe_transcription_models_other = config.get('processing', {}).get(
            'transcription', {}
        ).get('safe_models_other', [])

    def get_current_stitch_version(self) -> str:
        """Get the current stitch version from config."""
        try:
            return self.config.get('processing', {}).get('stitch', {}).get('current_version', 'stitch_v1')
        except Exception as e:
            logger.warning(f"Failed to get stitch version from config: {e}")
            return 'stitch_v1'

    def reconcile_content_flags(
        self,
        session: Session,
        content: Content,
        file_index: ContentFileIndex,
        commit: bool = True
    ) -> ReconciliationResult:
        """
        Reconcile a content item's flags with S3 reality.

        Args:
            session: Database session
            content: Content object to reconcile
            file_index: Pre-built file index for this content
            commit: Whether to commit changes (default True)

        Returns:
            ReconciliationResult with details of what changed
        """
        result = ReconciliationResult(content_id=content.content_id)
        updates_needed = False

        try:
            # --- is_downloaded ---
            should_be_downloaded = (
                file_index.has_source_files or
                file_index.has_audio or
                file_index.has_storage_manifest
            )
            if should_be_downloaded != content.is_downloaded:
                logger.info(f"State change for {content.content_id}: is_downloaded {content.is_downloaded} -> {should_be_downloaded}")
                content.is_downloaded = should_be_downloaded
                result.flags_updated.append('is_downloaded')
                updates_needed = True

            # --- is_converted ---
            should_be_converted = self._check_converted_state(content, file_index, session)
            if should_be_converted != content.is_converted:
                logger.info(f"State change for {content.content_id}: is_converted {content.is_converted} -> {should_be_converted}")
                content.is_converted = should_be_converted
                result.flags_updated.append('is_converted')
                updates_needed = True

            # --- Update chunk status ---
            if should_be_converted:
                chunks_updated = self._reconcile_chunk_status(session, content, file_index)
                result.chunks_updated = chunks_updated

            # --- is_diarized ---
            if file_index.has_diarization != content.is_diarized:
                logger.info(f"State change for {content.content_id}: is_diarized {content.is_diarized} -> {file_index.has_diarization}")
                content.is_diarized = file_index.has_diarization
                result.flags_updated.append('is_diarized')
                updates_needed = True

            # --- is_transcribed ---
            chunks = session.query(ContentChunk).filter_by(content_id=content.id).all()
            all_transcribed = bool(chunks) and all(c.transcription_status == 'completed' for c in chunks)
            if all_transcribed != content.is_transcribed:
                logger.info(f"State change for {content.content_id}: is_transcribed {content.is_transcribed} -> {all_transcribed}")
                content.is_transcribed = all_transcribed
                result.flags_updated.append('is_transcribed')
                updates_needed = True

            # --- is_stitched ---
            stitch_result = self._reconcile_stitch_state(session, content, file_index)
            if stitch_result['updated']:
                result.flags_updated.extend(stitch_result['flags'])
                updates_needed = True

            # --- is_embedded ---
            embed_result = self._reconcile_embed_state(session, content, file_index)
            if embed_result['updated']:
                result.flags_updated.extend(embed_result['flags'])
                updates_needed = True

            # --- is_compressed ---
            if file_index.has_storage_manifest != content.is_compressed:
                content.is_compressed = file_index.has_storage_manifest
                result.flags_updated.append('is_compressed')
                updates_needed = True

            # Commit if needed
            if updates_needed and commit:
                content.last_updated = datetime.now(timezone.utc)
                session.add(content)
                session.commit()
                session.refresh(content)
                logger.info(f"Updated flags for {content.content_id}: {', '.join(result.flags_updated)}")

        except Exception as e:
            error_msg = f"Error reconciling flags for {content.content_id}: {e}"
            logger.error(error_msg, exc_info=True)
            result.errors.append(error_msg)

        return result

    def _check_converted_state(
        self,
        content: Content,
        file_index: ContentFileIndex,
        session: Session
    ) -> bool:
        """Check if content should be marked as converted."""
        if not file_index.has_audio:
            return False

        # Check if chunk audio exists
        chunks_with_audio = file_index.get_chunk_indices_with_audio()
        if chunks_with_audio:
            return True

        # Check if transcripts exist (post-cleanup case)
        chunks = session.query(ContentChunk).filter_by(content_id=content.id).all()
        if chunks:
            chunks_with_transcripts = file_index.get_chunk_indices_with_transcripts()
            if chunks_with_transcripts:
                logger.debug(f"No chunk audio but transcript files exist for {content.content_id} - assuming post-cleanup")
                return True

        return False

    def _reconcile_chunk_status(
        self,
        session: Session,
        content: Content,
        file_index: ContentFileIndex
    ) -> int:
        """
        Reconcile chunk extraction and transcription status.

        Returns number of chunks updated.
        """
        chunks = session.query(ContentChunk).filter_by(content_id=content.id).all()
        chunks_updated = 0

        for chunk in chunks:
            chunk_updated = False

            # Check extraction status
            has_audio = file_index.has_chunk_audio(chunk.chunk_index)
            has_transcript = file_index.has_chunk_transcript(chunk.chunk_index)

            # Update extraction status (careful with compressed content)
            new_extraction_status = 'completed' if has_audio else 'pending'
            if chunk.extraction_status != new_extraction_status:
                # Don't downgrade if content is compressed and transcripts exist
                if file_index.has_storage_manifest and has_transcript:
                    logger.debug(f"Skipping extraction status downgrade for {content.content_id} chunk {chunk.chunk_index} - content compressed")
                else:
                    chunk.extraction_status = new_extraction_status
                    chunk_updated = True

            # Update transcription status
            is_safe_model = self._is_safe_transcription_model(chunk, content)
            new_transcription_status = 'completed' if (has_transcript and is_safe_model) else 'pending'

            if chunk.transcription_status != new_transcription_status:
                if chunk.transcription_status == 'completed' and new_transcription_status == 'pending':
                    if not has_transcript:
                        logger.warning(f"Transcript missing for {content.content_id} chunk {chunk.chunk_index} - resetting to pending")
                    elif not is_safe_model:
                        logger.info(f"Model '{chunk.transcribed_with}' not safe for {content.content_id} chunk {chunk.chunk_index}")

                chunk.transcription_status = new_transcription_status
                chunk_updated = True

            if chunk_updated:
                chunks_updated += 1

        return chunks_updated

    def _is_safe_transcription_model(self, chunk: ContentChunk, content: Content) -> bool:
        """Check if the chunk was transcribed with a safe model."""
        if not chunk.transcribed_with:
            return False

        if chunk.transcribed_with == 'legacy_whisper':
            return True

        is_english = content.main_language and content.main_language.lower().startswith('en')
        safe_models = self.safe_transcription_models_english if is_english else self.safe_transcription_models_other

        # Direct match
        if chunk.transcribed_with in safe_models:
            return True

        # Check with language suffix stripped (e.g., whisper_mlx_turbo_fr -> whisper_mlx_turbo)
        if '_' in chunk.transcribed_with:
            base_model = '_'.join(chunk.transcribed_with.rsplit('_', 1)[:-1])
            suffix = chunk.transcribed_with.rsplit('_', 1)[-1]
            if len(suffix) in (2, 3) and suffix.isalpha() and suffix.islower():
                if base_model in safe_models:
                    return True

        return False

    def _reconcile_stitch_state(
        self,
        session: Session,
        content: Content,
        file_index: ContentFileIndex
    ) -> Dict[str, Any]:
        """Reconcile is_stitched flag and stitch_version."""
        result = {'updated': False, 'flags': []}
        current_version = self.get_current_stitch_version()

        if not file_index.has_stitched_transcript and not content.is_stitched:
            return result

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

        # Determine if should be stitched
        any_sentences = len(version_list) > 0
        if file_index.has_stitched_transcript and any_sentences and not sentences_exist:
            # File exists but wrong version - needs re-stitch
            should_be_stitched = False
        else:
            should_be_stitched = file_index.has_stitched_transcript and sentences_exist

        # Update stitch_version if needed
        if should_be_stitched and should_recreate_stitch_task(current_version, content.stitch_version):
            content.stitch_version = current_version
            result['flags'].append('stitch_version')
            result['updated'] = True

        # Update is_stitched if needed
        if should_be_stitched != content.is_stitched:
            content.is_stitched = should_be_stitched
            content.stitch_version = current_version if should_be_stitched else 'none'
            result['flags'].append('is_stitched')
            result['updated'] = True

        return result

    def _reconcile_embed_state(
        self,
        session: Session,
        content: Content,
        file_index: ContentFileIndex
    ) -> Dict[str, Any]:
        """Reconcile is_embedded flag and segment_version."""
        result = {'updated': False, 'flags': []}
        current_segment_version = get_current_segment_version()

        if not file_index.has_semantic_segments:
            if content.is_embedded:
                content.is_embedded = False
                result['flags'].append('is_embedded')
                result['updated'] = True
            return result

        # Verify EmbeddingSegment records exist
        segment_count = session.query(EmbeddingSegment).filter(
            EmbeddingSegment.content_id == content.id
        ).count()
        segments_exist = segment_count > 0

        should_be_embedded = file_index.has_semantic_segments and segments_exist

        if should_be_embedded != content.is_embedded:
            content.is_embedded = should_be_embedded
            if should_be_embedded:
                meta_data = dict(content.meta_data) if content.meta_data else {}
                meta_data['segment_version'] = current_segment_version
                content.meta_data = meta_data
            result['flags'].append('is_embedded')
            result['updated'] = True

        return result

    def bulk_reconcile(
        self,
        session: Session,
        content_list: List[Content],
        file_indices: Dict[str, ContentFileIndex]
    ) -> Dict[str, Any]:
        """
        Bulk reconcile multiple content items efficiently.

        Args:
            session: Database session
            content_list: List of Content objects
            file_indices: Dict mapping content_id to ContentFileIndex

        Returns:
            Summary dict with statistics
        """
        stats = {
            'total': len(content_list),
            'updated': 0,
            'flag_counts': {},
            'errors': []
        }

        for content in content_list:
            file_index = file_indices.get(content.content_id)
            if not file_index:
                stats['errors'].append(f"No file index for {content.content_id}")
                continue

            result = self.reconcile_content_flags(
                session, content, file_index, commit=False
            )

            if result.had_updates:
                stats['updated'] += 1
                for flag in result.flags_updated:
                    stats['flag_counts'][flag] = stats['flag_counts'].get(flag, 0) + 1

            stats['errors'].extend(result.errors)

        # Commit all changes at once
        if stats['updated'] > 0:
            session.commit()

        return stats
