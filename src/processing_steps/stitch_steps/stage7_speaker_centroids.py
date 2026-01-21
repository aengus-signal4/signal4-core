#!/usr/bin/env python3
"""
Stage 7: Speaker Centroid Database Integration
==============================================

Seventh stage of the stitch pipeline that integrates speaker centroids with the database.

Key Responsibilities:
- Create or update Speaker records for each content-speaker pair
- Store speaker centroid embeddings from Stage 6 in the database
- Mark speakers with rebase_status='PENDING' for daily clustering
- Update WordTable with global speaker IDs and universal names
- Each content-speaker pair has exactly one record (enforced by unique constraint)

Input:
- WordTable from Stage 6 with embedding-based speaker assignments
- Speaker centroid data from Stage 6 containing embedding vectors
- Content ID for database operations

Output:
- WordTable with global speaker IDs (SPEAKER_a1b2c3d4 format)
- Speaker matches mapping local to global IDs
- Quality metrics and similarity scores
- Database records for new speakers

Key Components:
- SpeakerMatch: Data class for speaker mapping information
- SpeakerCentroidProcessor: Main processing class for database integration

Methods:
- speaker_centroids_stage(): Main entry point called by stitch pipeline
- SpeakerCentroidProcessor.process_speaker_centroids(): Core processing logic

Performance:
- Lightweight stage with mostly database operations
- O(n) complexity for centroid processing
- Fast vector operations for similarity calculations
"""

import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass

from src.utils.logger import setup_worker_logger
from .stage3_tables import WordTable
from src.database.models import Speaker
from src.database.session import get_session
from sqlalchemy import text
from datetime import datetime, timezone

logger = setup_worker_logger('stitch')
logger.setLevel(logging.INFO)

@dataclass
class SpeakerMatch:
    """Represents a match between a local speaker and database speaker."""
    local_speaker_id: str  # e.g., SPEAKER_00
    global_speaker_id: str  # e.g., SPEAKER_a1b2c3d4
    universal_name: str     # e.g., Speaker 12345
    display_name: Optional[str]
    similarity_score: float
    is_new: bool
    speaker_db_id: int

class SpeakerCentroidProcessor:
    """Processes speaker centroids and integrates with database."""
    
    def __init__(self,
                 min_similarity_threshold: float = 0.85,
                 min_quality_score: float = 0.5,
                 test_mode: bool = False):
        """
        Initialize the processor.
        
        Args:
            min_similarity_threshold: Minimum similarity for database speaker match
            min_quality_score: Minimum quality score to process a centroid
            test_mode: If True, only compare to database without writing
        """
        self.min_similarity_threshold = min_similarity_threshold
        self.min_quality_score = min_quality_score
        self.test_mode = test_mode
        
        logger.info(f"SpeakerCentroidProcessor initialized with: "
                   f"similarity={min_similarity_threshold}, "
                   f"quality={min_quality_score}, "
                   f"test_mode={test_mode}")
    
    def process_speaker_centroids(self,
                                speaker_centroid_data: Dict[str, Dict],
                                word_table: WordTable,
                                content_id: str,
                                original_diarization_speakers: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Process speaker centroids and match with database.
        
        Args:
            speaker_centroid_data: Centroid data from Stage 6
            word_table: WordTable with speaker assignments
            content_id: Content ID being processed
            
        Returns:
            Dictionary with processing results including speaker matches
        """
        start_time = time.time()
        
        try:
            # Even if no centroids, we still need to process speakers in word table
            if not speaker_centroid_data:
                logger.warning(f"[{content_id}] No speaker centroids provided, but will still process speakers from word table")
                
                # Get all speakers from word table that need database records
                all_speakers = set()
                for idx, word in word_table.df.iterrows():
                    local_speaker = word.get('speaker_current') if pd.notna(word.get('speaker_current')) else None
                    if local_speaker and local_speaker != 'UNKNOWN' and local_speaker.startswith('SPEAKER_'):
                        all_speakers.add(local_speaker)
                
                if not all_speakers and not original_diarization_speakers:
                    logger.warning(f"[{content_id}] No speakers found in word table or original diarization")
                    return {
                        'status': 'no_speakers',
                        'duration': time.time() - start_time
                    }
                
                logger.info(f"[{content_id}] Found {len(all_speakers)} speakers in word table without centroids")
                
                # Create empty matches list and proceed to update_word_assignments
                # which will handle creating database records for these speakers
                speaker_matches = []
                
                with get_session() as session:
                    # Update word assignments will create database records for all speakers
                    words_updated, all_speaker_matches = self._update_word_assignments(
                        word_table, speaker_matches, content_id, session, original_diarization_speakers
                    )
                    
                    # Commit all changes
                    session.commit()
                    
                    logger.info(f"[{content_id}] Successfully created database records for {len(all_speaker_matches)} speakers without centroids")
                
                return {
                    'status': 'success',
                    'duration': time.time() - start_time,
                    'centroids_processed': 0,
                    'valid_centroids': 0,
                    'high_quality_centroids': 0,
                    'low_quality_centroids': 0,
                    'speaker_matches': len(all_speaker_matches),
                    'words_updated': words_updated,
                    'speaker_mapping': {
                        match.local_speaker_id: {
                            'global_id': match.global_speaker_id,
                            'universal_name': match.universal_name,
                            'similarity': match.similarity_score,
                            'is_new': match.is_new
                        }
                        for match in all_speaker_matches
                    }
                }
            
            logger.info(f"[{content_id}] Processing {len(speaker_centroid_data)} speaker centroids")
            
            # Filter centroids - only process SPEAKER_XX IDs
            # Note: We don't filter by quality here because ALL speakers in the word table need database mappings
            valid_centroids = {}
            low_quality_centroids = {}
            for speaker_id, centroid_info in speaker_centroid_data.items():
                # Only process actual speaker IDs (SPEAKER_00, SPEAKER_01, etc.)
                if not speaker_id.startswith('SPEAKER_'):
                    logger.debug(f"[{content_id}] Skipping non-speaker ID: {speaker_id}")
                    continue
                # Debug: Check centroid_info type
                if not isinstance(centroid_info, dict):
                    logger.error(f"[{content_id}] Centroid info for {speaker_id} is not a dict: {type(centroid_info)}")
                    continue
                
                # Safely access quality_score
                quality_score = centroid_info.get('quality_score', 0) if isinstance(centroid_info, dict) else 0
                if quality_score >= self.min_quality_score:
                    valid_centroids[speaker_id] = centroid_info
                    logger.debug(f"[{content_id}] {speaker_id}: high quality ({quality_score:.3f})")
                else:
                    # Still process low quality centroids, but mark them as such
                    low_quality_centroids[speaker_id] = centroid_info
                    logger.info(f"[{content_id}] {speaker_id}: low quality ({quality_score:.3f} < {self.min_quality_score}) - will still process for database mapping")
            
            # Combine high and low quality centroids - all speakers need database mappings
            all_speaker_centroids = {**valid_centroids, **low_quality_centroids}
            
            if not all_speaker_centroids:
                logger.warning(f"[{content_id}] No speaker centroids to process")
                return {
                    'status': 'no_centroids',
                    'duration': time.time() - start_time
                }
            
            logger.info(f"[{content_id}] Processing {len(all_speaker_centroids)} speaker centroids ({len(valid_centroids)} high-quality, {len(low_quality_centroids)} low-quality)")
            
            # Log filtered out categories
            filtered_out = [speaker_id for speaker_id in speaker_centroid_data.keys() 
                           if not speaker_id.startswith('SPEAKER_')]
            if filtered_out:
                logger.info(f"[{content_id}] Filtered out non-speaker categories: {sorted(set(filtered_out))}")
            
            # NEW APPROACH: Store centroids and create temporary speakers for daily rebase
            with get_session() as session:
                speaker_matches = self._store_centroids_and_create_speakers(
                    all_speaker_centroids, session, content_id
                )
                
                # Update word table with temporary speaker IDs
                # This method will also create database records for speakers without centroids
                # and return the total number of matches including those without centroids
                words_updated, all_speaker_matches = self._update_word_assignments(
                    word_table, speaker_matches, content_id, session, original_diarization_speakers
                )
                
                # Use all_speaker_matches which includes both centroid and non-centroid speakers
                speaker_matches = all_speaker_matches
                
                # Commit all changes
                session.commit()
                
                if self.test_mode:
                    logger.info(f"[{content_id}] TEST MODE: Stored {len(speaker_matches)} speakers (including {len(all_speaker_centroids)} with centroids)")
                else:
                    logger.info(f"[{content_id}] Successfully stored {len(speaker_matches)} speakers (including {len(all_speaker_centroids)} with centroids)")
            
            return {
                'status': 'success',
                'duration': time.time() - start_time,
                'centroids_processed': len(speaker_centroid_data),
                'valid_centroids': len(all_speaker_centroids),
                'high_quality_centroids': len(valid_centroids),
                'low_quality_centroids': len(low_quality_centroids),
                'speaker_matches': len(speaker_matches),
                'words_updated': words_updated,
                'speaker_mapping': {
                    match.local_speaker_id: {
                        'global_id': match.global_speaker_id,
                        'universal_name': match.universal_name,
                        'similarity': match.similarity_score,
                        'is_new': match.is_new
                    }
                    for match in speaker_matches
                }
            }
            
        except Exception as e:
            logger.error(f"[{content_id}] Speaker centroid processing failed: {e}")
            logger.error(f"[{content_id}] Error details:", exc_info=True)
            # Rollback session if there's an error to avoid "transaction has been rolled back" issues
            if 'session' in locals():
                try:
                    session.rollback()
                except:
                    pass  # Session may already be closed
            return {
                'status': 'error',
                'error': str(e),
                'duration': time.time() - start_time
            }
    
    def _store_centroids_and_create_speakers(self,
                                           valid_centroids: Dict[str, Dict],
                                           session,
                                           content_id: str) -> List[SpeakerMatch]:
        """Store centroids directly in unified speaker table.
        
        For each content-speaker pair, either:
        - Updates the existing speaker record with new embedding (if reprocessing)
        - Creates a new speaker record (if first time processing)
        
        The unique constraint on (content_id, local_speaker_id) ensures
        exactly one record per content-speaker combination.
        """
        matches = []
        
        # Import necessary modules
        from src.database.models import SpeakerProcessingStatus
        import uuid
        
        # Generate a unique batch ID for this rebase operation
        rebase_batch_id = f"batch_{content_id}_{int(time.time())}"
        
        for local_speaker_id, centroid_info in valid_centroids.items():
            # Safely get centroid array
            if not isinstance(centroid_info, dict):
                logger.error(f"[{content_id}] Centroid info is not a dict for {local_speaker_id}")
                continue
                
            centroid = centroid_info.get('centroid')
            if centroid is None:
                logger.error(f"[{content_id}] No centroid found for {local_speaker_id}")
                continue
            
            # Ensure centroid is a numpy array
            if not isinstance(centroid, np.ndarray):
                centroid = np.array(centroid)
            
            # Check if speaker already exists for this content-speaker pair
            existing_speaker = Speaker.find_by_content_speaker(session, content_id, local_speaker_id)
            
            if existing_speaker:
                # Update existing speaker with enriched centroid from stitch
                # Note: We preserve embedding_diarization (raw FluidAudio embedding) if it exists
                has_diarization_embedding = existing_speaker.embedding_diarization is not None

                existing_speaker.embedding = centroid  # Update enriched embedding
                existing_speaker.embedding_quality_score = float(centroid_info.get('quality_score', 0))
                existing_speaker.duration = float(centroid_info.get('total_duration', 0))
                existing_speaker.segment_count = centroid_info.get('embeddings_count', 0)
                existing_speaker.algorithm_version = 'stage7_stitch'  # Algorithm for enriched embedding
                existing_speaker.rebase_status = SpeakerProcessingStatus.PENDING
                existing_speaker.rebase_batch_id = rebase_batch_id
                existing_speaker.updated_at = datetime.now(timezone.utc)

                # Note: We don't touch embedding_diarization or speaker_identity here
                # - embedding_diarization is preserved from diarize.py
                # - speaker_identity_id is populated by clustering/assignment phases

                embedding_status = "with diarization embedding" if has_diarization_embedding else "without diarization embedding"
                logger.info(f"[{content_id}] Updated existing speaker {existing_speaker.speaker_hash} for {local_speaker_id} with enriched centroid ({embedding_status})")
                
                match = SpeakerMatch(
                    local_speaker_id=local_speaker_id,
                    global_speaker_id=existing_speaker.speaker_hash,
                    universal_name=existing_speaker.speaker_hash,  # Using hash as universal name for now
                    display_name=existing_speaker.display_name,
                    similarity_score=1.0,
                    is_new=False,
                    speaker_db_id=existing_speaker.id
                )
            else:
                # Create new speaker record with enriched centroid
                # Note: embedding_diarization may have been set earlier by diarize.py, or will remain NULL
                speaker = Speaker.create_with_sequential_name(
                    session, content_id, local_speaker_id, centroid,
                    duration=float(centroid_info.get('total_duration', 0)),
                    segment_count=centroid_info.get('embeddings_count', 0),
                    embedding_quality_score=float(centroid_info.get('quality_score', 0)),
                    algorithm_version='stage7_stitch',
                    rebase_status=SpeakerProcessingStatus.PENDING,
                    rebase_batch_id=rebase_batch_id
                )

                # Set metadata for newly created speaker
                speaker.meta_data = {
                    'source_content_id': content_id,
                    'created_by_stitch': True,
                    'rebase_batch_id': rebase_batch_id
                }

                # Note: speaker_identity_id is left as NULL - will be assigned by speaker management
                # Note: embedding_diarization will remain NULL here - diarize.py sets it before stitch runs

                session.add(speaker)
                session.flush()  # Get ID

                # Create a match record for word table updates
                match = SpeakerMatch(
                    local_speaker_id=local_speaker_id,
                    global_speaker_id=speaker.speaker_hash,
                    universal_name=speaker.speaker_hash,  # Using hash as universal name for now
                    display_name=speaker.display_name,
                    similarity_score=1.0,
                    is_new=True,
                    speaker_db_id=speaker.id
                )

                logger.info(f"[{content_id}] Created new speaker {speaker.speaker_hash} for {local_speaker_id} with enriched centroid")
            
            matches.append(match)
        
        # Fix sentences linkage for all speakers
        self._fix_sentences_linkage(session, content_id, matches)

        return matches

    def _fix_sentences_linkage(self, session, content_id: str, matches: List[SpeakerMatch]):
        """Fix sentences table to point to correct speaker IDs based on speaker hash matching.

        This handles the case where speakers were deleted and recreated, leaving
        sentences with orphaned speaker_id references. We match via the speakers table
        using speaker_hash which is stable across recreations.
        """
        try:
            # Get content integer ID for sentences table
            content_result = session.execute(
                text("SELECT id FROM content WHERE content_id = :content_id"),
                {'content_id': content_id}
            ).fetchone()

            if not content_result:
                logger.warning(f"[{content_id}] Content not found in database, skipping sentences fix")
                return

            content_int_id = content_result.id

            # Update sentences for each speaker - sentences doesn't have speaker_hash,
            # so we need to find sentences by old speaker_id and update to new speaker_id
            fixed_count = 0
            for match in matches:
                # Find old speaker IDs that have same content_id and local_speaker_id
                # but different id than the current match
                result = session.execute(
                    text("""
                        UPDATE sentences
                        SET speaker_id = :new_speaker_id
                        WHERE content_id = :content_int_id
                          AND speaker_id IN (
                              SELECT id FROM speakers
                              WHERE content_id = :content_id_str
                                AND local_speaker_id = :local_speaker_id
                                AND id != :new_speaker_id
                          )
                    """),
                    {
                        'new_speaker_id': match.speaker_db_id,
                        'content_int_id': content_int_id,
                        'content_id_str': content_id,
                        'local_speaker_id': match.local_speaker_id
                    }
                )

                if result.rowcount > 0:
                    fixed_count += result.rowcount
                    logger.info(f"[{content_id}] Fixed {result.rowcount} sentences for {match.global_speaker_id}")

            if fixed_count > 0:
                logger.info(f"[{content_id}] Total fixed sentences: {fixed_count}")

        except Exception as e:
            logger.warning(f"[{content_id}] Failed to fix sentences linkage: {e}")
            # Don't fail the whole process if fixing sentences fails

        return matches
    
    # DEPRECATED: No longer used with new temporary speaker approach
    def _find_similar_speakers_DEPRECATED(self, session, embedding: np.ndarray, content_id: str, quality_score: float = 1.0) -> List[Tuple]:
        """Find similar speakers using pgvector similarity search."""
        # Determine embedding dimension
        embedding_dim = len(embedding)
        
        # Convert embedding to list for pgvector
        embedding_list = embedding.tolist()
        
        # For low-quality centroids, use a more lenient similarity threshold
        threshold = self.min_similarity_threshold
        if quality_score < self.min_quality_score:
            threshold = max(0.5, self.min_similarity_threshold - 0.05)  # Reduce threshold by 0.05 for low-quality
            logger.debug(f"[{content_id}] Using reduced similarity threshold {threshold:.3f} for low-quality centroid (quality={quality_score:.3f})")
        
        # Use pgvector similarity search with dynamic vector size
        query = text(f"""
            WITH similar_speakers AS (
                SELECT 
                    s.id,
                    s.speaker_hash,
                    s.display_name,
                    s.duration,
                    s.segment_count,
                    1 - (s.embedding <=> CAST(:embedding AS vector({embedding_dim}))) as similarity
                FROM speakers s
                WHERE 1 - (s.embedding <=> CAST(:embedding AS vector({embedding_dim}))) > :threshold
                ORDER BY similarity DESC
                LIMIT 5
            )
            SELECT * FROM similar_speakers
        """)
        
        result = session.execute(query, {
            'embedding': embedding_list,
            'threshold': threshold
        })
        
        # Convert results to tuples
        similar_speakers = []
        for row in result:
            speaker = session.query(Speaker).filter_by(id=row.id).first()
            if speaker:
                similarity = float(row.similarity)
                similar_speakers.append((speaker, similarity))
                logger.debug(f"[{content_id}] Found similar speaker {speaker.speaker_hash} "
                           f"with similarity {similarity:.3f}")
        
        return similar_speakers
    
    # No longer needed - embeddings are stored directly in the unified Speaker table
    
    # DEPRECATED: No longer used with new temporary speaker approach
    def _update_speaker_centroid_DEPRECATED(self, session, speaker: Speaker, 
                               new_centroid: np.ndarray, centroid_info: Dict):
        """Update the speaker's main embedding with weighted average of centroids.
        
        NOTE: This method is deprecated and references the old SpeakerEmbedding table.
        With the unified Speaker model, embeddings are stored directly in the Speaker table.
        """
        # This method is no longer used - keeping for reference only
        # In the unified model, we would query Speaker directly:
        # embeddings_query = session.query(Speaker).filter(
        #     Speaker.id == speaker.id,
        #     Speaker.embedding_quality_score >= 0.5
        # ).order_by(
        #     Speaker.embedding_quality_score.desc()
        # ).limit(20)  # Use top 20 embeddings
        
        # The rest of this deprecated method is commented out
        # embeddings = embeddings_query.all()
        # 
        # if embeddings:
        #     # Calculate weighted average of all centroids
        #     all_embeddings = []
        #     all_weights = []
        #     
        #     for emb_record in embeddings:
        #         emb_vector = np.array(emb_record.embedding)
        #         weight = emb_record.duration * emb_record.embedding_quality_score
        #         all_embeddings.append(emb_vector)
        #         all_weights.append(weight)
        #     
        #     # Add the new centroid
        #     all_embeddings.append(new_centroid)
        #     all_weights.append(centroid_info.get('total_duration', 0) * centroid_info.get('quality_score', 0))
        #     
        #     # Calculate weighted average
        #     all_embeddings = np.array(all_embeddings)
        #     all_weights = np.array(all_weights)
        #     all_weights = all_weights / all_weights.sum()
        #     
        #     updated_centroid = np.average(all_embeddings, axis=0, weights=all_weights)
        #     updated_centroid = updated_centroid / np.linalg.norm(updated_centroid)
        #     
        #     # Update speaker's embedding
        #     speaker.embedding = updated_centroid
        #     
        #     logger.debug(f"Updated {speaker.universal_name} centroid using "
        #                f"{len(embeddings)} historical + 1 new embedding")
        pass
    
    # DEPRECATED: Using new temporary speaker approach for both test and production modes
    def _match_with_database_readonly_DEPRECATED(self,
                                    valid_centroids: Dict[str, Dict],
                                    session,
                                    content_id: str) -> List[SpeakerMatch]:
        """Match speaker centroids against database speakers (read-only for test mode)."""
        matches = []
        
        for local_speaker_id, centroid_info in valid_centroids.items():
            # Safely get centroid array
            if not isinstance(centroid_info, dict):
                logger.error(f"[{content_id}] Centroid info is not a dict for {local_speaker_id}")
                continue
                
            centroid = centroid_info.get('centroid')
            if centroid is None:
                logger.error(f"[{content_id}] No centroid found for {local_speaker_id}")
                continue
            
            # Ensure centroid is a numpy array
            if not isinstance(centroid, np.ndarray):
                centroid = np.array(centroid)
            
            # Find similar speakers in database using pgvector
            quality_score = centroid_info.get('quality_score', 0)
            similar_speakers = self._find_similar_speakers(
                session, centroid, content_id, quality_score
            )
            
            if similar_speakers:
                # Use the most similar speaker
                best_match, similarity = similar_speakers[0]
                
                logger.info(f"[{content_id}] TEST MODE: Would match {local_speaker_id} -> {best_match.speaker_hash} "
                          f"(similarity={similarity:.3f})")
                
                match = SpeakerMatch(
                    local_speaker_id=local_speaker_id,
                    global_speaker_id=best_match.speaker_hash,
                    universal_name=best_match.speaker_hash,  # Using hash as universal name for now
                    display_name=best_match.display_name,
                    similarity_score=similarity,
                    is_new=False,
                    speaker_db_id=best_match.id
                )
                
                matches.append(match)
            else:
                # Would create new speaker but don't in test mode
                logger.info(f"[{content_id}] TEST MODE: No match found for {local_speaker_id}, would create new speaker")
                
                # Generate a test speaker ID
                import uuid
                test_global_id = f"TEST_{uuid.uuid4().hex[:8]}"
                test_universal_name = f"Speaker TEST{local_speaker_id[-2:]}"
                
                match = SpeakerMatch(
                    local_speaker_id=local_speaker_id,
                    global_speaker_id=test_global_id,
                    universal_name=test_universal_name,
                    display_name=None,
                    similarity_score=1.0,
                    is_new=True,
                    speaker_db_id=-1  # Fake ID for test mode
                )
                
                matches.append(match)
                
                logger.info(f"[{content_id}] TEST MODE: Would create new speaker {test_universal_name} "
                          f"({test_global_id}) for {local_speaker_id}")
        
        return matches
    
    def _update_word_assignments(self,
                               word_table: WordTable,
                               speaker_matches: List[SpeakerMatch],
                               content_id: str,
                               session=None,
                               original_diarization_speakers: Optional[List[str]] = None) -> Tuple[int, List[SpeakerMatch]]:
        """Update word table with global speaker IDs.
        
        Returns:
            Tuple of (words_updated, all_speaker_matches) where all_speaker_matches
            includes both the original matches and any new speakers created for
            speakers without centroids.
        """
        words_updated = 0
        # Start with the existing matches
        all_matches = list(speaker_matches)
        
        # Create mapping from local to global IDs
        speaker_map = {
            match.local_speaker_id: {
                'global_id': match.global_speaker_id,
                'universal_name': match.universal_name,
                'speaker_db_id': match.speaker_db_id
            }
            for match in speaker_matches
        }
        
        # First, get all unique speakers in word table to check coverage
        all_speakers = set()
        for idx, word in word_table.df.iterrows():
            local_speaker = word.get('speaker_current') if pd.notna(word.get('speaker_current')) else None
            if local_speaker and local_speaker != 'UNKNOWN':
                all_speakers.add(local_speaker)
        
        # Log coverage
        mapped_speakers = set(speaker_map.keys())
        unmapped_speakers = all_speakers - mapped_speakers
        if unmapped_speakers:
            logger.warning(f"[{content_id}] Speakers in word table but not in centroids: {unmapped_speakers}")
            logger.info(f"[{content_id}] Mapped speakers: {mapped_speakers}")
            logger.info(f"[{content_id}] All speakers in word table: {all_speakers}")
        
        # Update words
        for idx, word in word_table.df.iterrows():
            local_speaker = word.get('speaker_current') if pd.notna(word.get('speaker_current')) else None
            if not local_speaker or local_speaker == 'UNKNOWN':
                continue
            
            if local_speaker in speaker_map:
                word_table.df.at[idx, 'speaker_global'] = speaker_map[local_speaker]['global_id']
                word_table.df.at[idx, 'speaker_universal'] = speaker_map[local_speaker]['universal_name']
                
                # Update metadata
                metadata = word['metadata'] if 'metadata' in word and pd.notna(word['metadata']) else {}
                if not isinstance(metadata, dict):
                    metadata = {}
                
                metadata['speaker_db_mapping'] = {
                    'global_id': speaker_map[local_speaker]['global_id'],
                    'universal_name': speaker_map[local_speaker]['universal_name'],
                    'speaker_db_id': speaker_map[local_speaker]['speaker_db_id'],
                    'mapped_from': local_speaker
                }
            else:
                # For unmapped speakers, we'll handle them in the speaker dictionary below
                metadata = word['metadata'] if 'metadata' in word and pd.notna(word['metadata']) else {}
                if not isinstance(metadata, dict):
                    metadata = {}
            
            # Add assignment history entry for the mapping (metadata-only update)
            current_history = word_table.df.at[idx, 'assignment_history']
            if not isinstance(current_history, list):
                current_history = []
            
            # Determine what kind of mapping we applied
            if local_speaker in speaker_map:
                reason = f"Mapped local speaker {local_speaker} to global ID {speaker_map[local_speaker]['global_id']}"
            else:
                reason = f"Speaker {local_speaker} will be mapped in comprehensive dictionary"
            
            mapping_entry = {
                'stage': 'stage6b_speaker_centroids',
                'timestamp': time.time(),
                'speaker': local_speaker,  # Speaker didn't change, just mapped
                'method': 'global_speaker_mapping',
                'confidence': 1.0,
                'reason': reason
            }
            current_history.append(mapping_entry)
            word_table.df.at[idx, 'assignment_history'] = current_history
            
            word_table.df.at[idx, 'metadata'] = metadata
            words_updated += 1
        
        # Create a comprehensive speaker dictionary that maps ALL speakers to database IDs
        # This includes both speakers with centroids AND speakers that appear in word table
        speaker_db_dictionary = {}
        
        # First, add all speakers from centroids/matches
        for local_speaker_id, mapping in speaker_map.items():
            speaker_db_dictionary[local_speaker_id] = {
                'speaker_db_id': mapping['speaker_db_id'],
                'global_id': mapping['global_id'],
                'universal_name': mapping['universal_name'],
                'source': 'centroid_match'
            }
        
        # Handle unmapped speakers - create database entries for actual speakers without centroids
        if unmapped_speakers:
            logger.info(f"[{content_id}] Found {len(unmapped_speakers)} speakers without centroids - handling them now")
            
            for speaker_id in sorted(unmapped_speakers):
                # Skip category labels - these should not get database mappings
                if speaker_id in ['MULTI_SPEAKER', 'NEEDS_EMBEDDING', 'NEEDS_LLM', 
                                  'GOOD_GRAMMAR_MULTI', 'BAD_GRAMMAR_MULTI', 
                                  'GOOD_GRAMMAR_SINGLE', 'BAD_GRAMMAR_SINGLE', 'UNKNOWN']:
                    logger.info(f"[{content_id}]   {speaker_id} - skipping (category label, not a real speaker)")
                    continue
                
                # For actual SPEAKER_XX IDs without centroids, create database entries
                if speaker_id.startswith('SPEAKER_'):
                    logger.warning(f"[{content_id}]   {speaker_id} - creating database entry for speaker without centroid")
                    
                    if not self.test_mode:
                        # In production mode, create a new speaker entry with empty embedding
                        try:
                            # Check if speaker already exists
                            existing_speaker = Speaker.find_by_content_speaker(session, content_id, speaker_id)
                            if existing_speaker:
                                # Update existing speaker with current processing info
                                logger.info(f"[{content_id}]   {speaker_id} already exists with hash {existing_speaker.speaker_hash} (db_id: {existing_speaker.id}) - updating")
                                existing_speaker.embedding = None  # No embedding for speakers without centroids
                                existing_speaker.embedding_quality_score = 1.0
                                existing_speaker.algorithm_version = 'stage6b_tight_clusters'
                                existing_speaker.notes = f"Updated for {speaker_id} without centroid in {content_id} - insufficient data for embedding"
                                existing_speaker.updated_at = datetime.now(timezone.utc)
                                new_speaker = existing_speaker
                            else:
                                new_speaker = Speaker.create_with_sequential_name(
                                    session, content_id, speaker_id, None  # No embedding for speakers without centroids
                                )
                                new_speaker.notes = f"Created for {speaker_id} without centroid in {content_id} - insufficient data for embedding"
                                
                                session.add(new_speaker)
                                session.flush()
                            
                            # Add to speaker dictionary
                            speaker_db_dictionary[speaker_id] = {
                                'speaker_db_id': new_speaker.id,
                                'global_id': new_speaker.speaker_hash,
                                'universal_name': new_speaker.speaker_hash,  # Using hash as universal name for now
                                'source': 'no_centroid_fallback'
                            }
                            
                            # Also create a SpeakerMatch for this speaker and add to all_matches
                            new_match = SpeakerMatch(
                                local_speaker_id=speaker_id,
                                global_speaker_id=new_speaker.speaker_hash,
                                universal_name=new_speaker.speaker_hash,
                                display_name=new_speaker.display_name,
                                similarity_score=1.0,
                                is_new=True,
                                speaker_db_id=new_speaker.id
                            )
                            all_matches.append(new_match)
                            
                            logger.info(f"[{content_id}]   Created {new_speaker.speaker_hash} (db_id: {new_speaker.id}) for {speaker_id}")
                            
                        except Exception as e:
                            logger.error(f"[{content_id}] Failed to create speaker for {speaker_id}: {e}")
                    else:
                        # In test mode, create a fake entry
                        import uuid
                        test_global_id = f"TEST_NOCENT_{uuid.uuid4().hex[:8]}"
                        test_universal_name = f"Speaker TESTNOCENT{speaker_id[-2:]}"
                        test_db_id = -100 - len(speaker_db_dictionary)  # Unique negative ID
                        
                        speaker_db_dictionary[speaker_id] = {
                            'speaker_db_id': test_db_id,
                            'global_id': test_global_id,
                            'universal_name': test_universal_name,
                            'source': 'no_centroid_test'
                        }
                        
                        logger.info(f"[{content_id}]   TEST MODE: Would create {test_universal_name} for {speaker_id}")
                else:
                    # Other non-category labels
                    logger.info(f"[{content_id}]   {speaker_id} - skipping (unknown label type)")
        
        # Handle original diarization speakers that may not appear in word table yet
        if original_diarization_speakers:
            logger.info(f"[{content_id}] Checking {len(original_diarization_speakers)} original diarization speakers")
            
            for speaker_id in original_diarization_speakers:
                # Skip if already handled
                if speaker_id in speaker_db_dictionary:
                    continue
                
                # Only process actual SPEAKER_XX IDs
                if speaker_id.startswith('SPEAKER_'):
                    logger.warning(f"[{content_id}]   {speaker_id} - creating database entry for original diarization speaker not in word table")
                    
                    if not self.test_mode:
                        # In production mode, create a new speaker entry with empty embedding
                        try:
                            # Check if speaker already exists
                            existing_speaker = Speaker.find_by_content_speaker(session, content_id, speaker_id)
                            if existing_speaker:
                                # Update existing speaker with current processing info
                                logger.info(f"[{content_id}]   {speaker_id} already exists with hash {existing_speaker.speaker_hash} (db_id: {existing_speaker.id}) - updating")
                                existing_speaker.embedding = None  # No embedding for speakers without centroids
                                existing_speaker.embedding_quality_score = 1.0
                                existing_speaker.algorithm_version = 'stage6b_tight_clusters'
                                existing_speaker.notes = f"Updated for {speaker_id} from original diarization in {content_id} - no words assigned during stitch"
                                existing_speaker.updated_at = datetime.now(timezone.utc)
                                new_speaker = existing_speaker
                            else:
                                new_speaker = Speaker.create_with_sequential_name(
                                    session, content_id, speaker_id, None  # No embedding for speakers without centroids
                                )
                                new_speaker.notes = f"Created for {speaker_id} from original diarization in {content_id} - no words assigned during stitch"
                                
                                session.add(new_speaker)
                                session.flush()
                            
                            # Add to speaker dictionary
                            speaker_db_dictionary[speaker_id] = {
                                'speaker_db_id': new_speaker.id,
                                'global_id': new_speaker.speaker_hash,
                                'universal_name': new_speaker.speaker_hash,  # Using hash as universal name for now
                                'source': 'original_diarization_no_words'
                            }
                            
                            # Also create a SpeakerMatch for this speaker and add to all_matches
                            new_match = SpeakerMatch(
                                local_speaker_id=speaker_id,
                                global_speaker_id=new_speaker.speaker_hash,
                                universal_name=new_speaker.speaker_hash,
                                display_name=new_speaker.display_name,
                                similarity_score=1.0,
                                is_new=True,
                                speaker_db_id=new_speaker.id
                            )
                            all_matches.append(new_match)
                            
                            logger.info(f"[{content_id}]   Created {new_speaker.speaker_hash} (db_id: {new_speaker.id}) for original diarization speaker {speaker_id}")
                            
                        except Exception as e:
                            logger.error(f"[{content_id}] Failed to create speaker for original diarization speaker {speaker_id}: {e}")
                    else:
                        # In test mode, create a fake entry
                        import uuid
                        test_global_id = f"TEST_ORIGDIAR_{uuid.uuid4().hex[:8]}"
                        test_universal_name = f"Speaker TESTORIGDIAR{speaker_id[-2:]}"
                        test_db_id = -200 - len(speaker_db_dictionary)  # Unique negative ID
                        
                        speaker_db_dictionary[speaker_id] = {
                            'speaker_db_id': test_db_id,
                            'global_id': test_global_id,
                            'universal_name': test_universal_name,
                            'source': 'original_diarization_test'
                        }
                        
                        logger.info(f"[{content_id}]   TEST MODE: Would create {test_universal_name} for original diarization speaker {speaker_id}")
                else:
                    logger.debug(f"[{content_id}]   {speaker_id} - skipping (not a SPEAKER_XX ID)")
        
        # Store the complete speaker dictionary on the word_table object
        word_table.speaker_db_dictionary = speaker_db_dictionary
        
        logger.info(f"[{content_id}] Updated {words_updated} words with global speaker IDs")
        logger.info(f"[{content_id}] Created speaker database dictionary with {len(speaker_db_dictionary)} speakers:")
        for speaker_id, mapping in speaker_db_dictionary.items():
            logger.info(f"[{content_id}]   {speaker_id} -> {mapping['universal_name']} (db_id: {mapping['speaker_db_id']}, source: {mapping['source']})")
        
        return words_updated, all_matches

def speaker_centroids_stage(content_id: str,
                          word_table: WordTable,
                          speaker_centroid_data: Dict[str, Dict],
                          test_mode: bool = False,
                          overwrite: bool = False,
                          original_diarization_speakers: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Main entry point for Stage 7: Speaker Centroid Database Integration.
    
    This is the primary method called by the stitch pipeline. It takes speaker
    centroids from Stage 6 and integrates them with the global speaker database,
    creating new speakers as needed and updating the word table with global IDs.
    
    Args:
        content_id: Content ID being processed (e.g., "Bdb001")
        word_table: WordTable from Stage 6 with local speaker assignments
        speaker_centroid_data: Dictionary mapping local speaker IDs to centroid data
                              including embedding vectors and quality metrics
        test_mode: If True, only compare to database without writing new records
        overwrite: If True, delete existing speakers for this content before processing
        original_diarization_speakers: List of all speakers from original diarization.json
                                     to ensure database records are created for all speakers
        
    Returns:
        Dictionary containing:
            - status: 'success' or 'error'
            - data: Dict with updated word_table and speaker_mapping
            - stats: Processing statistics including speaker counts
            - error: Error message if status is 'error'
            
    Example:
        result = speaker_centroids_stage("Bdb001", word_table, centroids, test_mode=True)
        if result['status'] == 'success':
            mappings = result['data']['speaker_mapping']
    """
    start_time = time.time()
    
    logger.info(f"[{content_id}] Starting Stage 6b: Speaker Centroid Database Integration")
    logger.info(f"[{content_id}] Received {len(speaker_centroid_data) if speaker_centroid_data else 0} centroids")
    
    result = {
        'status': 'pending',
        'content_id': content_id,
        'stage': 'speaker_centroids',
        'data': {
            'word_table': None,
            'speaker_mapping': {}
        },
        'stats': {},
        'error': None
    }
    
    try:
        # Don't return early if no centroids - we still need to process speakers
        # The processor will handle creating database records for speakers without centroids
        if not speaker_centroid_data:
            logger.warning(f"[{content_id}] No speaker centroids provided to Stage 6b, but will still process speakers from word table")
        
        # NOTE: Overwrite mode no longer deletes existing speakers
        # Instead, we use the update path (find_by_content_speaker) which:
        # - Updates enriched embeddings (from stitch)
        # - Preserves diarization embeddings (from diarize.py)
        # - Overwrites other fields (duration, segment_count, quality_score, etc.)
        # This is simpler and automatically preserves diarization embeddings
        if overwrite and not test_mode:
            logger.info(f"[{content_id}] Overwrite mode: will update existing speakers (preserving diarization embeddings)")
        
        # Initialize processor with test mode
        processor = SpeakerCentroidProcessor(test_mode=test_mode)

        # Process speaker centroids
        processing_result = processor.process_speaker_centroids(
            speaker_centroid_data, word_table, content_id, original_diarization_speakers
        )
        
        if processing_result['status'] != 'success':
            raise ValueError(f"Centroid processing failed: {processing_result.get('error', 'Unknown error')}")
        
        # Store results
        result['data']['word_table'] = word_table
        result['data']['speaker_mapping'] = processing_result.get('speaker_mapping', {})
        result['stats'] = {
            'duration': time.time() - start_time,
            'processing_result': processing_result
        }
        result['status'] = 'success'
        
        logger.info(f"[{content_id}] Stage 6b completed successfully in {time.time() - start_time:.2f}s")
        logger.info(f"[{content_id}] Processing results: {processing_result}")
        
        # Log speaker mapping summary
        if processing_result.get('speaker_mapping'):
            logger.info(f"[{content_id}] Speaker mappings:")
            for local_id, mapping in processing_result['speaker_mapping'].items():
                logger.info(f"[{content_id}]   {local_id} -> {mapping['universal_name']} "
                          f"(similarity: {mapping['similarity']:.3f}, new: {mapping['is_new']})")
        
        return result
        
    except Exception as e:
        logger.error(f"[{content_id}] Stage 6b failed: {str(e)}")
        logger.error(f"[{content_id}] Error details:", exc_info=True)
        
        # Rollback any session if there's an error
        if 'session' in locals():
            try:
                session.rollback()
            except:
                pass  # Session may already be closed
        
        result.update({
            'status': 'error',
            'error': str(e),
            'stats': {'duration': time.time() - start_time}
        })
        # Ensure word table is preserved even on error
        result['data']['word_table'] = word_table
        return result