#!/usr/bin/env python3
"""
Stage 9: Bad Grammar + Multi-Speaker Assignment
===============================================

Ninth stage of the stitch pipeline that handles segments with bad grammar and multiple speakers.

Key Responsibilities:
- Process all words categorized as BAD_GRAMMAR_MULTI by Stage 3
- Map segments to diarization boundaries for precise speaker assignment
- Re-transcribe segments using advanced models (Parakeet v2 for English, Whisper MLX Turbo for other languages) with proper prompting
- Assign words to speakers based on diarization overlap analysis
- Apply grammar enhancement (punctuation and capitalization)
- Handle the most challenging segments requiring both speaker and grammar resolution

Input:
- WordTable from Stage 8 with categorized words and assignments
- Audio file for re-transcription
- Diarization data with speaker segments
- Segments marked as BAD_GRAMMAR_MULTI needing complex processing

Output:
- WordTable with speaker assignments for BAD_GRAMMAR_MULTI words
- Enhanced text with improved grammar and punctuation
- Assignment statistics including re-transcription metrics

Key Components:
- GrammarAnalyzer: Analyzes text quality using Stage 3 criteria
- DiarizationMapper: Maps segments to precise diarization boundaries
- Re-transcription engine: Advanced transcription with language-specific models
- Language detection: Automatically detects text language to choose optimal model
- Grammar enhancement: Punctuation and capitalization improvements

Methods:
- bad_grammar_multi_speaker_stage(): Main entry point called by stitch pipeline
- map_segment_to_diarizations(): Maps segments to speaker boundaries
- retranscribe_segment_group_by_diarization(): Re-transcribes with advanced models

Performance:
- Most complex stage requiring re-transcription and multiple processing steps
- Uses async processing for efficiency
- Language-specific model selection (Parakeet v2 for English, Whisper MLX Turbo for all other languages)
- Moderate computational cost (10-15% of pipeline time)
"""

import asyncio
import logging
import time
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from src.utils.paths import get_project_root, get_config_path
import tempfile
import shutil

from src.utils.logger import setup_worker_logger
from src.processing_steps.stitch_steps.stage3_tables import WordTable
from src.processing_steps.stitch_steps.util_stitch import (
    load_stage_config,
    update_processing_status,
    format_stage_stats,
    summarize_speaker_assignments)
from src.storage.s3_utils import S3Storage
from src.storage.content_storage import ContentStorageManager
# Model server imports removed - using direct model loading only

logger = setup_worker_logger('stitch')


class GrammarAnalyzer:
    """Analyzes text segments for grammar quality."""
    
    def has_good_grammar(self, text: str) -> bool:
        """
        Check if text has good grammar with strict criteria.
        Uses EXACT same logic as Stage 3 for consistency.
        
        Good grammar requires EITHER:
        - A capitalized word anywhere in the text, OR
        - Any punctuation (!.,?) anywhere in the text
        
        Anything else is considered bad grammar.
        """
        if not text:
            return False
        
        text = text.strip()
        if not text:
            return False
            
        # Check for at least one capital letter anywhere in the text
        has_capital = any(c.isupper() for c in text)
        
        # Check for any punctuation (!.,?) anywhere in the text
        has_punctuation = any(p in text for p in ['!', '.', ',', '?'])
        
        # Good grammar requires capitals OR punctuation
        # Everything else is bad grammar
        return has_capital or has_punctuation


class DiarizationSegmentMapper:
    """Maps bad grammar segments to diarization segments for re-transcription."""
    
    def __init__(self, diarization_segments: List[Dict]):
        """Initialize with diarization data."""
        self.diarization_segments = sorted(diarization_segments, key=lambda x: x.get('start', 0))
        self.diarization_df = pd.DataFrame(diarization_segments)
        if len(self.diarization_df) > 0:
            self.diarization_df = self.diarization_df.sort_values('start').reset_index(drop=True)
    
    def map_segment_to_diarizations(self, segment_start: float, segment_end: float) -> List[Dict]:
        """
        Map a segment to overlapping diarization segments.
        
        Returns:
            List of diarization segments that overlap with the given segment
        """
        overlapping_segments = []
        
        for _, dia_seg in self.diarization_df.iterrows():
            dia_start = dia_seg['start']
            dia_end = dia_seg['end']
            
            # Check for overlap
            if segment_start < dia_end and segment_end > dia_start:
                overlap_start = max(segment_start, dia_start)
                overlap_end = min(segment_end, dia_end)
                overlap_duration = overlap_end - overlap_start
                
                overlapping_segments.append({
                    'diarization_segment': dia_seg.to_dict(),
                    'overlap_start': overlap_start,
                    'overlap_end': overlap_end,
                    'overlap_duration': overlap_duration,
                    'overlap_percentage': overlap_duration / (segment_end - segment_start) if segment_end > segment_start else 0
                })
        
        # Sort by overlap start time
        overlapping_segments.sort(key=lambda x: x['overlap_start'])
        
        return overlapping_segments


class DiarizationRetranscriber:
    """Handles re-transcription of segments at diarization boundaries."""
    
    def __init__(self, s3_storage: S3Storage, language: str = 'en', audio_path: Optional[str] = None, test_mode: bool = False):
        """
        Initialize the retranscriber.

        Args:
            s3_storage: S3 storage instance
            language: Language code (e.g., 'en', 'es', 'fr')
            audio_path: Path to audio file passed from main pipeline
            test_mode: Whether to create test directory structure
        """
        self.s3_storage = s3_storage
        self.storage_manager = ContentStorageManager(s3_storage)
        self.language = language
        self.audio_path = audio_path  # Store the audio path from pipeline
        self.test_mode = test_mode

        # Set up test directory structure only in test mode
        if test_mode:
            self.test_base_dir = get_project_root() / "tests" / "content"
            self.test_base_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.test_base_dir = None

        # Pre-load models to avoid loading them repeatedly for each segment
        self._parakeet_model = None
        self._whisper_model = None

        # Determine which model to load based on language
        # Use Parakeet v2 only for English, Whisper MLX Turbo for all other languages

        if language == 'en':
            logger.info(f"Pre-loading Parakeet v2 model for English")
            try:
                from parakeet_mlx import from_pretrained
                self._parakeet_model = from_pretrained("mlx-community/parakeet-tdt-0.6b-v2")
                logger.info("Parakeet v2 model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to pre-load Parakeet model: {e}")
        else:
            logger.info(f"Pre-loading MLX Whisper Turbo model for language: {language}")
            try:
                import mlx_whisper
                # Note: mlx_whisper doesn't have explicit model loading, but we import it here
                # The first transcription call will load the model and cache it
                self._whisper_model = mlx_whisper
                logger.info("MLX Whisper module loaded")
            except Exception as e:
                logger.warning(f"Failed to import MLX Whisper: {e}")
    
    async def retranscribe_segment_group_by_diarization(self, 
                                                content_id: str,
                                                group_index: int,
                                                segment_indices: List[int],
                                                group_text: str,
                                                diarization_mappings: List[Dict],
                                                group_start: float,
                                                group_end: float,
                                                test_mode: bool = False,
                                                word_table: Optional[WordTable] = None) -> Dict:
        """
        Re-transcribe a group of consecutive segments. First tries full group, then falls back to per-diarization.
        
        Args:
            content_id: Content ID
            group_index: Group index
            segment_indices: List of original segment indices in the group
            group_text: Combined text of all segments in the group
            diarization_mappings: List of overlapping diarization segments
            group_start: Start time of the full group
            group_end: End time of the full group
            
        Returns:
            Dictionary with re-transcription results
        """
        try:
            # Set up content-specific directory (test or temp based on mode)
            if self.test_mode and self.test_base_dir:
                content_test_dir = self.test_base_dir / content_id
                content_inputs_dir = content_test_dir / "inputs"
                content_inputs_dir.mkdir(parents=True, exist_ok=True)
            else:
                # Use a temp directory for non-test mode
                import tempfile
                content_inputs_dir = Path(tempfile.mkdtemp(prefix=f"stage9_{content_id}_"))
            
            # Use audio file path from pipeline (same approach as Stage 6)
            local_audio_path = None
            
            # First priority: Use audio path provided by the pipeline
            if self.audio_path and Path(self.audio_path).exists():
                logger.debug(f"Using audio file from pipeline: {self.audio_path}")
                local_audio_path = Path(self.audio_path)
            else:
                # Fallback: Check test directory cache (only if in test mode)
                if self.test_mode and content_inputs_dir:
                    cached_audio_path = content_inputs_dir / "audio.wav"
                    if cached_audio_path.exists():
                        logger.debug(f"Using cached audio from test directory: {cached_audio_path}")
                        local_audio_path = cached_audio_path
                
                if not local_audio_path:
                    # Final fallback: Check persistent temp directory (where Stage 1 saves in production)
                    persistent_temp_dir = Path("/tmp/stitch_audio") / content_id
                    persistent_audio_path = persistent_temp_dir / "audio.wav"
                    
                    if persistent_audio_path.exists():
                        logger.debug(f"Using audio from persistent temp directory: {persistent_audio_path}")
                        local_audio_path = persistent_audio_path
                    else:
                        # No audio file found anywhere
                        error_msg = (f"Audio file not found for content {content_id}. "
                                   f"Checked: pipeline_path={self.audio_path}, "
                                   f"test_cache={cached_audio_path}, "
                                   f"persistent_temp={persistent_audio_path}")
                        return {
                            'status': 'error',
                            'error': error_msg
                        }
            
            # First attempt: Try to re-transcribe the entire group
            logger.info(f"[{content_id}] Attempting full group re-transcription for group {group_index} (segments {segment_indices})")
            
            group_id_full = f"group{group_index}_full_{group_start:.1f}_{group_end:.1f}"
            temp_dir_full = content_inputs_dir / f"retranscribe_{group_id_full}"
            temp_dir_full.mkdir(exist_ok=True)
            
            try:
                full_group_audio_path = temp_dir_full / "group_audio.wav"
                
                # Calculate smart boundaries - extend to halfway between words
                # This helps capture word beginnings/endings if timestamps are slightly off
                # Get the actual word timings from the segment group
                first_word_start = group_start
                last_word_end = group_end
                
                # Find the gap to previous word (if any)
                prev_word_end = None
                next_word_start = None
                
                # Get all words from word table to find neighboring words
                all_words_df = word_table.df.sort_values('start')
                
                # Find the first word in our group
                first_segment_idx = segment_indices[0]
                first_word_in_group = word_table.df[word_table.df['segment_index'] == first_segment_idx].sort_values('start').iloc[0] if len(word_table.df[word_table.df['segment_index'] == first_segment_idx]) > 0 else None
                
                if first_word_in_group is not None:
                    first_word_idx = first_word_in_group.name
                    # Find previous word
                    if first_word_idx > 0:
                        prev_word = all_words_df.iloc[first_word_idx - 1]
                        prev_word_end = prev_word['end']
                
                # Find the last word in our group
                last_segment_idx = segment_indices[-1]
                last_word_in_group = word_table.df[word_table.df['segment_index'] == last_segment_idx].sort_values('start').iloc[-1] if len(word_table.df[word_table.df['segment_index'] == last_segment_idx]) > 0 else None
                
                if last_word_in_group is not None:
                    last_word_idx = last_word_in_group.name
                    # Find next word
                    if last_word_idx < len(all_words_df) - 1:
                        next_word = all_words_df.iloc[last_word_idx + 1]
                        next_word_start = next_word['start']
                
                # Calculate extended boundaries
                if prev_word_end is not None:
                    # Extend to halfway between previous word and our first word
                    extended_start = prev_word_end + (group_start - prev_word_end) / 2
                else:
                    # No previous word, just use a small buffer
                    extended_start = max(0, group_start - 0.1)
                
                if next_word_start is not None:
                    # Extend to halfway between our last word and next word
                    extended_end = group_end + (next_word_start - group_end) / 2
                else:
                    # No next word, just use a small buffer
                    extended_end = group_end + 0.1
                
                logger.debug(f"[{content_id}] Group {group_index} boundaries: original [{group_start:.2f}, {group_end:.2f}], extended [{extended_start:.2f}, {extended_end:.2f}]")
                
                if self.extract_audio_segment(
                    str(local_audio_path),
                    str(full_group_audio_path),
                    extended_start,
                    extended_end
                ):
                    # Create prompt for full group
                    context_prompt = (
                        f"This audio contains multiple speakers in conversation. "
                        f"Please transcribe with proper punctuation and capitalization. "
                        f"Use complete sentences and proper grammar."
                    )

                    # Use configured language from database (no detection)
                    content_language = self.language

                    # Run transcription
                    # Use Parakeet v2 only for English, Whisper MLX Turbo for all other languages
                    use_parakeet = content_language == 'en'
                    transcription_method = 'parakeet_mlx_v2' if use_parakeet else 'whisper_mlx_turbo'
                    language_name = 'English' if use_parakeet else content_language
                    logger.info(f"[{content_id}] Re-transcribing with {transcription_method} (configured language: {language_name} [{content_language}])")
                    transcription_start = time.time()

                    result = await self._transcribe_with_prompt(
                        audio_path=str(full_group_audio_path),
                        content_id=content_id,
                        chunk_index=f"stage9_full_{group_id_full}",
                        prompt=context_prompt,
                        use_parakeet=use_parakeet,
                        audio_offset=extended_start,  # Pass the offset for timestamp adjustment
                        detected_language=content_language
                    )
                    
                    transcription_duration = time.time() - transcription_start
                    
                    if result['status'] == 'success':
                        full_text = result.get('text', '')
                        
                        # Check if the transcription has good grammar
                        grammar_analyzer = GrammarAnalyzer()
                        has_good_grammar = grammar_analyzer.has_good_grammar(full_text)
                        
                        if test_mode and not has_good_grammar:
                            # Debug why grammar check failed
                            has_capital = any(c.isupper() for c in full_text)
                            has_punctuation = any(p in full_text for p in ['!', '.', ',', '?'])
                            logger.debug(f"[{content_id}] Grammar check failed - Has capital: {has_capital}, Has punctuation: {has_punctuation}")
                        
                        if has_good_grammar:
                            logger.info(f"[{content_id}] Full group transcription has good grammar - using Stage 8 approach")
                            
                            # Return the full transcription result for Stage 8-style processing
                            return {
                                'status': 'success',
                                'group_index': group_index,
                                'segment_indices': segment_indices,
                                'original_text': group_text,
                                'retranscription_mode': 'full_group',
                                'full_transcription': {
                                    'text': full_text,
                                    'segments': result.get('segments', []),
                                    'transcription_time': transcription_duration,
                                    'audio_offset': extended_start  # Store the offset for timestamp adjustment
                                },
                                'diarization_mappings': diarization_mappings,
                                'group_start': group_start,  # Store original boundaries
                                'group_end': group_end
                            }
                        else:
                            logger.info(f"[{content_id}] Full group transcription lacks good grammar - falling back to NEEDS_LLM")
                            if test_mode:
                                logger.info(f"[{content_id}] Original text: \"{group_text[:200]}...\"" if len(group_text) > 200 else f"[{content_id}] Original text: \"{group_text}\"")
                                logger.info(f"[{content_id}] Re-transcribed text: \"{full_text[:200]}...\"" if len(full_text) > 200 else f"[{content_id}] Re-transcribed text: \"{full_text}\"")
                
            finally:
                # Clean up temp directory
                if temp_dir_full.exists() and 'retranscribe_' in str(temp_dir_full):
                    try:
                        shutil.rmtree(temp_dir_full)
                    except Exception as e:
                        logger.warning(f"Failed to clean up temp directory: {e}")
            
            # Fallback: Try diarization-based assignment before NEEDS_LLM
            logger.info(f"[{content_id}] Full group transcription failed or has poor grammar - attempting diarization-based assignment for group {group_index}")
            
            # Gather original words from the segments (we need them for actual timings)
            original_words = []
            if word_table is not None:
                for seg_idx in segment_indices:
                    # Get words from word table for this segment
                    seg_words_df = word_table.df[word_table.df['segment_index'] == seg_idx]
                    for _, word in seg_words_df.iterrows():
                        original_words.append({
                            'text': word['text'],
                            'start': word['start'],
                            'end': word['end']
                        })
            
            # Try to assign words based on diarization boundaries with 0.3s protection
            diarization_assigned_words = self._assign_words_by_diarization_with_protection(
                content_id, group_text, diarization_mappings, group_start, group_end, 
                protection_buffer=0.3, original_words=original_words
            )
            
            if diarization_assigned_words and diarization_assigned_words['assigned_word_count'] > 0:
                logger.info(f"[{content_id}] Diarization-based assignment resolved {diarization_assigned_words['assigned_word_count']}/{diarization_assigned_words['total_word_count']} words")
                
                return {
                    'status': 'success',
                    'group_index': group_index,
                    'segment_indices': segment_indices,
                    'original_text': group_text,
                    'retranscription_mode': 'diarization_protected',
                    'diarization_assignment': diarization_assigned_words,
                    'reason': 'Partial resolution using diarization boundaries with protection buffer'
                }
            else:
                logger.info(f"[{content_id}] No words could be assigned by diarization - assigning all to NEEDS_LLM")
                
                return {
                    'status': 'success',
                    'group_index': group_index,
                    'segment_indices': segment_indices,
                    'original_text': group_text,
                    'retranscription_mode': 'needs_llm',
                    'needs_llm_assignment': True,
                    'reason': 'Full group transcription failed and diarization assignment not possible'
                }
            
        except Exception as e:
            logger.error(f"Error in segment re-transcription: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
        finally:
            # Clean up temporary directory if not in test mode
            if not self.test_mode and content_inputs_dir and content_inputs_dir.exists():
                try:
                    shutil.rmtree(content_inputs_dir)
                    logger.debug(f"[{content_id}] Cleaned up temporary directory: {content_inputs_dir}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary directory {content_inputs_dir}: {e}")
    
    async def _transcribe_with_prompt(self, audio_path: str, content_id: str, 
                                     chunk_index: str, prompt: str, use_parakeet: bool = True,
                                     audio_offset: float = 0.0, detected_language: str = None) -> Dict:
        """
        Transcribe audio using Parakeet MLX v2 (English only) or Whisper MLX Turbo (all other languages).
        Note: Parakeet doesn't support prompts, but provides excellent word-level timestamps for English.
        Note: Whisper MLX Turbo provides excellent multilingual support with word-level timestamps.

        Args:
            audio_path: Path to audio file
            content_id: Content ID
            chunk_index: Chunk identifier
            prompt: Transcription prompt (not used by either model currently)
            use_parakeet: Whether to use Parakeet (for supported languages) or MLX Whisper
            audio_offset: Offset to add to timestamps (when audio was extracted from a larger file)
            detected_language: Detected language code (e.g., 'en', 'fr', 'es')
        """
        try:
            # Use local models directly (model server deprecated)
            # Try Parakeet v2 for English only
            if use_parakeet:
                logger.debug(f"Using Parakeet v2 for {detected_language} transcription")

                try:
                    logger.debug(f"Transcribing with Parakeet MLX v2 (prompt ignored: {prompt[:50]}...)")

                    # Use pre-loaded Parakeet model
                    if self._parakeet_model is None:
                        logger.warning("Parakeet model not pre-loaded, loading now (this will be slow)")
                        from parakeet_mlx import from_pretrained
                        self._parakeet_model = from_pretrained("mlx-community/parakeet-tdt-0.6b-v2")

                    # Run transcription (Parakeet doesn't support prompts but gives good word-level timestamps)
                    result = self._parakeet_model.transcribe(audio_path)
                    
                    # Convert AlignedResult to standard format
                    text = ""
                    words = []
                    
                    # Debug log to see what Parakeet returned
                    logger.debug(f"Parakeet result type: {type(result)}")
                    logger.debug(f"Parakeet result attributes: {dir(result)}")
                    
                    if hasattr(result, 'text'):
                        text = result.text
                        logger.debug(f"Parakeet text: {text[:100] if text else 'None'}...")
                    
                    # Extract word-level timestamps from AlignedResult
                    if hasattr(result, 'alignment') and result.alignment:
                        logger.debug(f"Parakeet alignment length: {len(result.alignment)}")
                        for word_info in result.alignment:
                            if hasattr(word_info, 'word') and hasattr(word_info, 'start') and hasattr(word_info, 'end'):
                                words.append({
                                    'word': word_info.word.strip(),
                                    'start': float(word_info.start),
                                    'end': float(word_info.end),
                                    'probability': getattr(word_info, 'confidence', 1.0)
                                })
                    else:
                        logger.debug("No alignment data in Parakeet result")
                    
                    # Create segments structure compatible with Whisper format
                    segments = []
                    if words:
                        segments = [{
                            'text': text,
                            'start': words[0]['start'] if words else 0.0,
                            'end': words[-1]['end'] if words else 0.0,
                            'words': words
                        }]
                    else:
                        # No word-level timestamps from Parakeet
                        # We'll let process_full_segment_like_stage8 handle word creation
                        # Note: start/end will be set properly from diarization boundaries later
                        segments = [{
                            'text': text,
                            'start': 0.0,
                            'end': 0.0,
                            'words': []
                        }]
                    
                    # Format result
                    if text and segments:  # If we have text and segments, it's a success
                        return {
                            'status': 'success',
                            'text': text,
                            'segments': segments,
                            'language': detected_language,
                            'method': 'parakeet_mlx_v2'
                        }
                except Exception as parakeet_error:
                    logger.warning(f"Parakeet failed: {str(parakeet_error)}")
                    # Continue to Whisper fallback below
            
            # If we reach here, Parakeet either wasn't used or failed
            # Use MLX Whisper as fallback
            # Use detected language if provided, otherwise fall back to self.language
            whisper_language = detected_language if detected_language else self.language
            logger.debug(f"Using MLX Whisper transcription with language: {whisper_language}")

            # Use pre-loaded Whisper module
            if self._whisper_model is None:
                logger.warning("MLX Whisper not pre-loaded, loading now")
                import mlx_whisper
                self._whisper_model = mlx_whisper

            # Transcribe with MLX Whisper
            result = self._whisper_model.transcribe(
                audio_path,
                path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
                language=whisper_language,
                word_timestamps=True
            )
            
            # Convert Whisper result to standard format
            text = result.get('text', '').strip()
            segments = []
            
            for segment in result.get('segments', []):
                seg_words = []
                for word_info in segment.get('words', []):
                    seg_words.append({
                        'word': word_info.get('word', '').strip(),
                        'start': word_info.get('start', 0.0),
                        'end': word_info.get('end', 0.0),
                        'probability': word_info.get('probability', 1.0)
                    })
                
                segments.append({
                    'text': segment.get('text', '').strip(),
                    'start': segment.get('start', 0.0),
                    'end': segment.get('end', 0.0),
                    'words': seg_words
                })
            
            # Format result
            return {
                'status': 'success',
                'text': text,
                'segments': segments,
                'language': whisper_language,
                'method': 'whisper_mlx_turbo'
            }
                
        except Exception as e:
            logger.error(f"Error in transcription: {str(e)}")
            # No fallback - if advanced models fail, the stage should fail
            return {
                'status': 'error',
                'error': f'Advanced transcription models failed: {str(e)}'
            }
    
    def _assign_words_by_diarization_with_protection(self, content_id: str, group_text: str, 
                                                     diarization_mappings: List[Dict],
                                                     group_start: float, group_end: float,
                                                     protection_buffer: float = 0.3,
                                                     original_words: Optional[List[Dict]] = None) -> Dict:
        """
        Assign words to speakers based on diarization boundaries with protection buffer.
        
        Words that are more than protection_buffer seconds away from any diarization boundary
        and fully contained within a single speaker's segment get assigned to that speaker.
        Others are marked as NEEDS_LLM.
        
        Args:
            content_id: Content ID
            group_text: Text of the group
            diarization_mappings: List of overlapping diarization segments
            group_start: Start time of the group
            group_end: End time of the group
            protection_buffer: Buffer in seconds from diarization boundaries (default 0.3s)
            
        Returns:
            Dictionary with word assignments
        """
        try:
            # Use original words if provided, otherwise split text and estimate positions
            if original_words:
                # Use actual word timings from the word table
                words_with_timing = original_words
            else:
                # Split text into words and estimate their positions
                words = group_text.split()
                if not words:
                    return {'total_word_count': 0, 'assigned_word_count': 0, 'words': []}
                
                # Calculate approximate word timings based on uniform distribution
                duration = group_end - group_start
                word_duration = duration / len(words) if words else 0
                
                words_with_timing = []
                for i, word in enumerate(words):
                    word_start = group_start + (i * word_duration)
                    word_end = word_start + word_duration
                    words_with_timing.append({
                        'text': word,
                        'start': word_start,
                        'end': word_end
                    })
            
            if not words_with_timing:
                return {'total_word_count': 0, 'assigned_word_count': 0, 'words': []}
            
            word_assignments = []
            assigned_count = 0
            
            for word_info in words_with_timing:
                # Get word timing
                word_text = word_info['text']
                word_start = word_info['start']
                word_end = word_info['end']
                word_center = (word_start + word_end) / 2
                
                # Check which diarization segments this word overlaps with
                overlapping_speakers = []
                safe_from_boundaries = True
                
                for mapping in diarization_mappings:
                    diar_seg = mapping['diarization_segment']
                    diar_start = diar_seg['start']
                    diar_end = diar_seg['end']
                    speaker = diar_seg['speaker']
                    
                    # Check if word overlaps with this diarization segment
                    if word_start < diar_end and word_end > diar_start:
                        overlapping_speakers.append(speaker)
                        
                        # Check if word is too close to boundaries
                        distance_to_start = abs(word_center - diar_start)
                        distance_to_end = abs(word_center - diar_end)
                        
                        if distance_to_start < protection_buffer or distance_to_end < protection_buffer:
                            safe_from_boundaries = False
                
                # Determine assignment
                if safe_from_boundaries and len(set(overlapping_speakers)) == 1:
                    # Word is safely within a single speaker's territory
                    assigned_speaker = overlapping_speakers[0]
                    assigned_count += 1
                else:
                    # Word is near boundaries or overlaps multiple speakers
                    assigned_speaker = 'NEEDS_LLM'
                
                word_assignments.append({
                    'text': word_text,
                    'start': word_start,
                    'end': word_end,
                    'speaker': assigned_speaker,
                    'confidence': 0.9 if assigned_speaker != 'NEEDS_LLM' else 0.0,
                    'reason': 'diarization_protected' if assigned_speaker != 'NEEDS_LLM' else 'near_boundary_or_multi_speaker'
                })
            
            logger.info(f"[{content_id}] Diarization protection assignment: {assigned_count}/{len(words_with_timing)} words assigned to speakers")
            
            return {
                'total_word_count': len(words_with_timing),
                'assigned_word_count': assigned_count,
                'words': word_assignments
            }
            
        except Exception as e:
            logger.error(f"[{content_id}] Error in diarization-based word assignment: {str(e)}")
            return {'total_word_count': 0, 'assigned_word_count': 0, 'words': []}


    def extract_audio_segment(self, input_path: str, output_path: str, start_time: float, end_time: float) -> bool:
        """Extract a segment of audio using ffmpeg."""
        try:
            import subprocess
            
            duration = end_time - start_time
            cmd = [
                'ffmpeg', '-i', input_path,
                '-ss', str(start_time),
                '-t', str(duration),
                '-c', 'copy',
                '-y',  # Overwrite output file
                output_path
            ]
            
            logger.debug(f"Extracting audio segment: {start_time}s to {end_time}s")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"FFmpeg error: {result.stderr}")
                return False
                
            logger.debug(f"Successfully extracted segment to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error extracting audio segment: {str(e)}")
            return False
    
    def _detect_text_language(self, text: str, content_id: str) -> str:
        """
        Detect the language of a text segment.
        
        Args:
            text: Text to analyze
            content_id: Content ID for logging
            
        Returns:
            Language code (e.g., 'en', 'es', 'fr') or self.language as fallback
        """
        try:
            # Check if text is substantial enough for detection
            if not text or len(text.strip()) < 20:
                logger.debug(f"[{content_id}] Text too short for language detection, using configured language: {self.language}")
                return self.language
            
            # Try to use langdetect library
            try:
                from langdetect import detect_langs, LangDetectException
                
                # Detect language with confidence scores
                langs = detect_langs(text)
                
                if langs and len(langs) > 0:
                    # Get the most confident language
                    detected_lang = langs[0].lang
                    confidence = langs[0].prob
                    
                    # Only use detected language if confidence is high enough
                    if confidence > 0.8:
                        logger.info(f"[{content_id}] Detected language: {detected_lang} (confidence: {confidence:.2f})")
                        return detected_lang
                    else:
                        logger.debug(f"[{content_id}] Low confidence language detection: {detected_lang} ({confidence:.2f}), using configured language: {self.language}")
                        return self.language
                        
            except ImportError:
                logger.warning(f"[{content_id}] langdetect library not available, using configured language: {self.language}")
                return self.language
            except LangDetectException as e:
                logger.warning(f"[{content_id}] Language detection failed: {str(e)}, using configured language: {self.language}")
                return self.language
                
        except Exception as e:
            logger.error(f"[{content_id}] Error in language detection: {str(e)}, using configured language: {self.language}")
            return self.language


def process_full_segment_like_stage8(segment_index: int,
                                   full_transcription: Dict,
                                   diarization_mappings: List[Dict],
                                   word_table: WordTable,
                                   content_id: str,
                                   group_start: float = None,
                                   group_end: float = None) -> Dict:
    """
    Process a full segment re-transcription using Stage 8's approach.
    
    This assigns sentences to speakers based on diarization overlap,
    ensuring sentence integrity while respecting diarization boundaries.
    
    Args:
        segment_index: Index of the segment being processed
        full_transcription: Transcription result from re-transcription
        diarization_mappings: Diarization segments that overlap with this segment
        word_table: The word table being updated
        content_id: Content ID for logging
        group_start: Original start time of the group in the full audio (for offset calculation)
        group_end: Original end time of the group in the full audio
    
    Returns:
        Dictionary with reconstruction results and word assignments
    """
    # Extract words from the full transcription
    all_words = []
    word_index = 0
    
    # Determine the audio offset if group_start is provided
    # Re-transcription timestamps start at 0, but we need them relative to the full audio
    if group_start is None and diarization_mappings:
        group_start = diarization_mappings[0]['diarization_segment']['start']
    if group_end is None and diarization_mappings:
        group_end = diarization_mappings[-1]['diarization_segment']['end']
    
    # Check if the transcription result includes an audio_offset
    # This offset aligns timestamps from the extracted audio (starting at 0) to the full audio timeline
    audio_offset = full_transcription.get('audio_offset', group_start if group_start is not None else 0)
    
    # Get segments from transcription result
    segments = full_transcription.get('segments', [])
    if not segments and full_transcription.get('text'):
        # If no segments but we have text, create one segment
        segments = [{
            'text': full_transcription['text'],
            'start': group_start if group_start is not None else 0,
            'end': group_end if group_end is not None else 0,
            'words': []
        }]
    elif segments:
        # Check if segments have invalid timestamps (0.0, 0.0) and fix them
        for segment in segments:
            if segment.get('start', 0) == 0 and segment.get('end', 0) == 0 and group_start is not None:
                # Use group boundaries
                segment['start'] = group_start
                segment['end'] = group_end
                logger.debug(f"[{content_id}] Fixed segment timestamps: {segment['start']:.2f} - {segment['end']:.2f}")
    
    # Extract all words
    logger.debug(f"[{content_id}] Processing {len(segments)} segments with audio offset: {audio_offset:.2f}s")
    for i, segment in enumerate(segments):
        segment_words = segment.get('words', [])
        logger.debug(f"[{content_id}] Segment {i}: {segment.get('start', 0):.2f}-{segment.get('end', 0):.2f}, words: {len(segment_words)}, text length: {len(segment.get('text', ''))}")
        
        if segment_words:
            # We have word-level timestamps from re-transcription
            # These timestamps are relative to the extracted audio segment (starting at 0)
            # We need to offset them to match the original audio timeline
            for word in segment_words:
                all_words.append({
                    'word_index': word_index,
                    'text': word.get('word', ''),
                    'start': word.get('start', 0) + audio_offset,  # Apply offset to get absolute time
                    'end': word.get('end', 0) + audio_offset,      # Apply offset to get absolute time
                    'confidence': word.get('probability', 1.0)
                })
                word_index += 1
        else:
            # No word-level timestamps, create words from segment text
            segment_text = segment.get('text', '').strip()
            if segment_text:
                # Split text into words and assign them evenly across the segment
                words_in_text = segment_text.split()
                # Note: segment start/end should already be in absolute time
                segment_start = segment.get('start', 0)
                segment_end = segment.get('end', 0)
                segment_duration = segment_end - segment_start
                
                if words_in_text and segment_duration > 0:
                    word_duration = segment_duration / len(words_in_text)
                    
                    for i, word_text in enumerate(words_in_text):
                        word_start = segment_start + (i * word_duration)
                        word_end = word_start + word_duration
                        
                        all_words.append({
                            'word_index': word_index,
                            'text': word_text,
                            'start': word_start,  # Already in absolute time
                            'end': word_end,      # Already in absolute time
                            'confidence': 0.9  # Slightly lower confidence since no word-level timestamps
                        })
                        word_index += 1
    
    # Sort words by start time
    all_words.sort(key=lambda x: x['start'])
    
    # Find sentence boundaries
    sentence_boundaries = []
    for i, word in enumerate(all_words):
        text = word['text']
        if any(text.endswith(p) for p in ['.', '!', '?']):
            sentence_boundaries.append(i)
    
    # If no sentence boundaries, treat the whole segment as one sentence
    if not sentence_boundaries:
        sentence_boundaries = [len(all_words) - 1]
    
    # Add start boundary
    sentence_boundaries = [-1] + sentence_boundaries
    
    # Extract diarization segments from mappings
    diarization_segments = [m['diarization_segment'] for m in diarization_mappings]
    
    # Assign speakers to sentences based on overlap
    for sent_idx in range(len(sentence_boundaries) - 1):
        start_idx = sentence_boundaries[sent_idx] + 1
        end_idx = sentence_boundaries[sent_idx + 1] + 1
        
        if start_idx >= len(all_words):
            continue
            
        # Get sentence time boundaries
        sentence_words = all_words[start_idx:end_idx]
        if not sentence_words:
            continue
            
        sent_start = min(w['start'] for w in sentence_words)
        sent_end = max(w['end'] for w in sentence_words)
        sent_duration = sent_end - sent_start
        
        # Calculate overlap with each diarization segment
        best_speaker = None
        best_overlap = 0
        
        for dia_seg in diarization_segments:
            dia_start = dia_seg['start']
            dia_end = dia_seg['end']
            dia_speaker = dia_seg['speaker']
            
            # Calculate overlap
            overlap_start = max(sent_start, dia_start)
            overlap_end = min(sent_end, dia_end)
            
            if overlap_start < overlap_end:
                overlap_duration = overlap_end - overlap_start
                overlap_percentage = overlap_duration / sent_duration if sent_duration > 0 else 0
                
                if overlap_percentage > best_overlap:
                    best_overlap = overlap_percentage
                    best_speaker = dia_speaker
        
        # Assign speaker to all words in sentence
        if best_speaker:
            for word_idx in range(start_idx, end_idx):
                all_words[word_idx]['speaker'] = best_speaker
                all_words[word_idx]['speaker_confidence'] = best_overlap
        else:
            # No overlap found - mark for embedding processing
            for word_idx in range(start_idx, end_idx):
                all_words[word_idx]['speaker'] = 'NEEDS_EMBEDDING'
                all_words[word_idx]['speaker_confidence'] = 0.0
    
    # Ensure each diarization segment has at least one sentence fragment
    for dia_seg in diarization_segments:
        dia_speaker = dia_seg['speaker']
        dia_start = dia_seg['start']
        dia_end = dia_seg['end']
        
        # Check if this diarization segment has any words assigned
        has_assignment = any(
            w.get('speaker') == dia_speaker and 
            w['start'] >= dia_start and 
            w['end'] <= dia_end
            for w in all_words
        )
        
        if not has_assignment:
            # Find words that overlap with this diarization segment
            overlapping_words = []
            for i, word in enumerate(all_words):
                if word['start'] < dia_end and word['end'] > dia_start:
                    overlap_start = max(word['start'], dia_start)
                    overlap_end = min(word['end'], dia_end)
                    overlap_duration = overlap_end - overlap_start
                    word_duration = word['end'] - word['start']
                    overlap_percentage = overlap_duration / word_duration if word_duration > 0 else 1.0
                    
                    if overlap_percentage > 0.5:  # Word mostly in this diarization segment
                        overlapping_words.append((i, overlap_percentage))
            
            # Assign at least one word to this speaker
            if overlapping_words:
                # Sort by overlap percentage
                overlapping_words.sort(key=lambda x: -x[1])
                # Assign the word with highest overlap
                best_word_idx, overlap_pct = overlapping_words[0]
                all_words[best_word_idx]['speaker'] = dia_speaker
                all_words[best_word_idx]['speaker_confidence'] = overlap_pct
                logger.debug(f"[{content_id}] Assigned word {all_words[best_word_idx]['text']} to {dia_speaker} to ensure coverage")
    
    # Ensure all words have a speaker assigned (fallback for any words that might have been missed)
    for word in all_words:
        if 'speaker' not in word:
            word['speaker'] = 'NEEDS_EMBEDDING'
            word['speaker_confidence'] = 0.0
            logger.debug(f"[{content_id}] Assigned fallback speaker to word: {word['text']}")
    
    # Build reconstructed text
    reconstructed_text = ' '.join([w['text'] for w in all_words])
    
    # Get unique speakers
    speakers = list(set(w.get('speaker', 'UNKNOWN') for w in all_words))
    
    return {
        'segment_index': segment_index,
        'reconstructed_text': reconstructed_text,
        'words': all_words,
        'word_count': len(all_words),
        'speakers': speakers
    }



def group_consecutive_segments(segments_df: pd.DataFrame) -> List[Dict]:
    """
    Group consecutive BAD_GRAMMAR_MULTI segments together for batch processing.
    
    Args:
        segments_df: DataFrame with segment information
        
    Returns:
        List of segment groups, each containing consecutive segments
    """
    if len(segments_df) == 0:
        return []
    
    # Sort segments by index to ensure correct ordering
    segments_df = segments_df.sort_values('segment_index').reset_index(drop=True)
    
    groups = []
    current_group = None
    
    for _, segment in segments_df.iterrows():
        segment_idx = segment['segment_index']
        
        if current_group is None:
            # Start first group
            current_group = {
                'segment_indices': [segment_idx],
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'],
                'segments': [segment]
            }
        else:
            # Check if this segment is consecutive (next index) 
            last_idx = current_group['segment_indices'][-1]
            
            if segment_idx == last_idx + 1:
                # Consecutive segment - add to current group
                current_group['segment_indices'].append(segment_idx)
                current_group['end'] = segment['end']
                current_group['text'] += ' ' + segment['text']
                current_group['segments'].append(segment)
            else:
                # Non-consecutive segment - finalize current group and start new one
                groups.append(current_group)
                current_group = {
                    'segment_indices': [segment_idx],
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': segment['text'],
                    'segments': [segment]
                }
    
    # Add the last group
    if current_group is not None:
        groups.append(current_group)
    
    return groups




async def bad_grammar_multi_speaker_stage(content_id: str,
                                        word_table: WordTable,
                                        diarization_data: Dict,
                                        s3_storage: S3Storage,
                                        test_mode: bool = False,
                                        max_concurrent: int = 3,
                                        language: str = 'en',
                                        audio_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Execute Stage 9: Bad Grammar + Multi-Speaker Assignment via Re-transcription.
    
    This stage handles segments with bad grammar and multiple speakers by:
    1. Mapping segments to diarization boundaries
    2. Re-transcribing each diarization segment  
    3. Reconstructing segments with proper speaker assignment
    
    Args:
        content_id: Content ID to process
        word_table: WordTable from stage 6
        diarization_data: Diarization data with speaker segments
        s3_storage: S3 storage instance
        test_mode: Whether running in test mode
        max_concurrent: Maximum concurrent re-transcriptions
        language: Language code for transcription (e.g., 'en', 'es', 'fr')
        audio_path: Path to audio file from pipeline (Stage 1)
        
    Returns:
        Dictionary with assignment results
    """
    start_time = time.time()
    stage_name = 'bad_grammar_multi_speaker'
    
    logger.info(f"[{content_id}] Stage 9: Bad Grammar + Multi-Speaker Assignment via Re-transcription (language: {language})")
    
    result = {
        'status': 'success',
        'content_id': content_id,
        'stage': stage_name,
        'data': {
            'word_table': word_table
        },
        'stats': {
            'segments_analyzed': 0,
            'bad_grammar_multi_segments': 0,
            'consecutive_groups_formed': 0,
            'groups_retranscribed': 0,
            'words_reconstructed': 0,
            'retranscription_time': 0
        },
        'error': None
    }
    
    try:
        # Get diarization segments
        diarization_segments = diarization_data.get('segments', [])
        if not diarization_segments:
            logger.warning(f"[{content_id}] No diarization segments found - cannot re-transcribe")
            result['stats']['status'] = 'no_diarization_data'
            return result
        
        logger.info(f"[{content_id}] Found {len(diarization_segments)} diarization segments")
        
        # Get segments from word table with Stage 3 analysis results
        segments_df = word_table.df.groupby('segment_index').agg({
            'text': lambda x: ' '.join(x),
            'start': 'min',
            'end': 'max',
            'speaker_current': 'first',
            'segment_has_good_grammar': 'first',
            'segment_has_multiple_speakers': 'first'
        }).reset_index()
        
        # Filter for segments with BAD_GRAMMAR_MULTI category from Stage 3
        # We trust Stage 3's categorization completely
        bad_grammar_multi_segments = segments_df[
            segments_df['speaker_current'] == 'BAD_GRAMMAR_MULTI'
        ].copy()
        
        logger.info(f"[{content_id}] Found {len(bad_grammar_multi_segments)} bad grammar + multi-speaker segments to process")
        
        if len(bad_grammar_multi_segments) == 0:
            logger.info(f"[{content_id}] No bad grammar + multi-speaker segments to process")
            result['stats']['status'] = 'no_target_segments'
            return result
        
        # Initialize mapper and retranscriber (pass audio_path from pipeline)
        mapper = DiarizationSegmentMapper(diarization_segments)
        retranscriber = DiarizationRetranscriber(s3_storage, language=language, audio_path=audio_path, test_mode=test_mode)
        
        # Group consecutive BAD_GRAMMAR_MULTI segments together
        segment_groups = group_consecutive_segments(bad_grammar_multi_segments)
        logger.info(f"[{content_id}] Grouped {len(bad_grammar_multi_segments)} segments into {len(segment_groups)} consecutive groups")
        
        # Prepare re-transcription tasks (one task per group)
        retranscription_tasks = []
        
        for group_idx, segment_group in enumerate(segment_groups):
            group_start = segment_group['start']
            group_end = segment_group['end']
            group_text = segment_group['text']
            segment_indices = segment_group['segment_indices']
            
            # Map the entire group to diarization segments
            diarization_mappings = mapper.map_segment_to_diarizations(group_start, group_end)
            
            if diarization_mappings:
                retranscription_tasks.append({
                    'group_index': group_idx,
                    'segment_indices': segment_indices,
                    'group_text': group_text,
                    'group_start': group_start,
                    'group_end': group_end,
                    'diarization_mappings': diarization_mappings
                })
                
                logger.info(f"[{content_id}] Group {group_idx} (segments {segment_indices}) mapped to {len(diarization_mappings)} diarization segments")
            else:
                logger.warning(f"[{content_id}] Group {group_idx} (segments {segment_indices}) has no diarization mapping")
        
        # Process re-transcriptions with concurrency control
        logger.info(f"[{content_id}] Starting {len(retranscription_tasks)} group re-transcription tasks (max concurrent: {max_concurrent})")
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_task(task):
            async with semaphore:
                return await retranscriber.retranscribe_segment_group_by_diarization(
                    content_id=content_id,
                    group_index=task['group_index'],
                    segment_indices=task['segment_indices'],
                    group_text=task['group_text'],
                    diarization_mappings=task['diarization_mappings'],
                    group_start=task['group_start'],
                    group_end=task['group_end'],
                    test_mode=test_mode,
                    word_table=word_table
                )
        
        # Run all re-transcription tasks
        retranscription_start = time.time()
        retranscription_results = await asyncio.gather(
            *[process_task(task) for task in retranscription_tasks],
            return_exceptions=True
        )
        total_retranscription_time = time.time() - retranscription_start
        
        # Process results and update word table
        segments_retranscribed = 0
        total_words_reconstructed = 0
        failed_retranscriptions = 0
        
        for i, retrans_result in enumerate(retranscription_results):
            if isinstance(retrans_result, Exception):
                logger.error(f"[{content_id}] Re-transcription task failed with exception: {retrans_result}")
                failed_retranscriptions += 1
                continue
            
            if retrans_result['status'] != 'success':
                error_msg = retrans_result.get('error', 'Unknown error')
                logger.warning(f"[{content_id}] Re-transcription failed: {error_msg}")
                
                # Check if this is a missing audio error that should fail the entire stage
                if 'Audio file not found' in error_msg or 'Failed to download audio file' in error_msg:
                    from src.utils.error_codes import ErrorCode, create_error_result
                    logger.error(f"[{content_id}] Stage 9 failing due to missing audio file")
                    return create_error_result(
                        error_code=ErrorCode.MISSING_AUDIO,
                        error_message=error_msg,
                        error_details={
                            'content_id': content_id,
                            'stage': 'stage9_bad_grammar_multi',
                            'retranscription_task': i
                        },
                        permanent=False
                    )
                
                failed_retranscriptions += 1
                continue
            
            segments_retranscribed += 1
            
            # Check which mode was used
            group_index = retrans_result['group_index']
            segment_indices = retrans_result['segment_indices']
            original_text = retrans_result['original_text']
            retranscription_mode = retrans_result.get('retranscription_mode', 'per_diarization')
            
            if retranscription_mode == 'full_group':
                # Handle full group re-transcription using Stage 8 approach
                logger.debug(f"[{content_id}] Processing {segment_indices}) with Stage 8 approach (full group with good grammar)")
                
                full_transcription = retrans_result['full_transcription']
                diarization_mappings = retrans_result['diarization_mappings']
                
                # Process like Stage 8 would (using first segment index for compatibility)
                reconstruction = process_full_segment_like_stage8(
                    segment_indices[0], full_transcription, diarization_mappings, word_table, content_id,
                    group_start=retrans_result.get('group_start'),
                    group_end=retrans_result.get('group_end')
                )
                
                # Log results
                logger.info(f"[{content_id}] Full group re-transcription for group {group_index}:")
                logger.info(f"[{content_id}]   Original: \"{original_text[:100]}...\"")
                logger.info(f"[{content_id}]   Re-transcribed: \"{full_transcription['text'][:100]}...\"")
                logger.info(f"[{content_id}]   Words: {len(reconstruction['words'])}")
                logger.info(f"[{content_id}]   Speakers: {', '.join(reconstruction['speakers'])}")
                
            elif retranscription_mode == 'diarization_protected':
                # Handle diarization-based assignment with protection buffer
                logger.debug(f"[{content_id}] Processing {segment_indices}) with diarization protection assignment")
                
                diarization_assignment = retrans_result['diarization_assignment']
                assigned_words = diarization_assignment['words']
                
                # Create reconstruction from diarization assignment
                all_words = []
                speakers_used = set()
                
                for i, word_info in enumerate(assigned_words):
                    all_words.append({
                        'word_index': i,
                        'text': word_info['text'],
                        'start': word_info['start'],
                        'end': word_info['end'],
                        'speaker': word_info['speaker'],
                        'confidence': word_info['confidence']
                    })
                    speakers_used.add(word_info['speaker'])
                
                reconstruction = {
                    'segment_index': segment_indices[0],
                    'reconstructed_text': ' '.join([w['text'] for w in all_words]),
                    'words': all_words,
                    'word_count': len(all_words),
                    'speakers': sorted(list(speakers_used))
                }
                
                assigned_count = diarization_assignment['assigned_word_count']
                total_count = diarization_assignment['total_word_count']
                logger.info(f"[{content_id}] Diarization protection resolved {assigned_count}/{total_count} words ({assigned_count/total_count*100:.1f}%)")
                
            elif retranscription_mode == 'needs_llm':
                # Handle NEEDS_LLM assignment when full transcription failed or had poor grammar
                logger.info(f"[{content_id}] Assigning group {group_index} (segments {segment_indices}) to NEEDS_LLM")
                
                # Create a simple reconstruction that preserves the original text but assigns NEEDS_LLM
                all_words = []
                word_index = 0
                
                # Get words from the original segments
                for segment_idx in segment_indices:
                    segment_words = word_table.df[word_table.df['segment_index'] == segment_idx]
                    for _, word in segment_words.iterrows():
                        all_words.append({
                            'word_index': word_index,
                            'text': word['text'],
                            'start': word['start'],
                            'end': word['end'],
                            'speaker': 'NEEDS_LLM',
                            'confidence': 0.0
                        })
                        word_index += 1
                
                reconstruction = {
                    'segment_index': segment_indices[0],
                    'reconstructed_text': original_text,
                    'words': all_words,
                    'word_count': len(all_words),
                    'speakers': ['NEEDS_LLM']
                }
                
                logger.info(f"[{content_id}] Marked {len(all_words)} words for LLM resolution")
                
            else:
                # Handle per-diarization segment approach (original method) - should not reach here anymore
                logger.error(f"[{content_id}] Unexpected retranscription mode: {retranscription_mode}")
                continue
            
            # Update word table with new words and speaker assignments
            # First, remove ALL old words from ALL segments in this group (complete replacement)
            for segment_index in segment_indices:
                word_table.df = word_table.df[word_table.df['segment_index'] != segment_index]
            
            # Prepare words for the helper method
            retranscribed_words = []
            for word in reconstruction['words']:
                retranscribed_words.append({
                    'text': word.get('text', ''),
                    'start': word.get('start', 0),
                    'end': word.get('end', 0),
                    'speaker': word.get('speaker', 'NEEDS_EMBEDDING'),
                    'confidence': word.get('confidence', 0.8),
                    'metadata': {
                        'original_text': original_text,
                        'retranscription_mode': retranscription_mode,
                        'pending_embedding_confirmation': retranscription_mode == 'full_group',  # Only for successful full group transcriptions
                        'needs_llm_resolution': retranscription_mode == 'needs_llm',
                        'diarization_protected': retranscription_mode == 'diarization_protected'
                    },
                    'reason': f"Re-transcribed multi-speaker group {group_index} (segments {segment_indices}) ({retranscription_mode})"
                })
            
            if retranscribed_words:
                # Use the helper method to add the new words
                # Note: remove_overlapping=False because we already removed all words from the group
                words_added = word_table.update_segment_with_retranscription(
                    segment_index=segment_indices[0],  # Use first segment index for compatibility
                    retranscribed_words=retranscribed_words,
                    original_category='BAD_GRAMMAR_MULTI',
                    method_suffix='retranscribed',
                    stage='stage9',
                    remove_overlapping=False  # We already removed all words above
                )
                
                total_words_reconstructed += words_added
        
        # Count groups by resolution type
        groups_with_good_grammar = 0
        groups_needs_llm = 0
        groups_diarization_protected = 0
        for i, retrans_result in enumerate(retranscription_results):
            if isinstance(retrans_result, dict) and retrans_result.get('status') == 'success':
                mode = retrans_result.get('retranscription_mode')
                if mode == 'full_group':
                    groups_with_good_grammar += 1
                elif mode == 'needs_llm':
                    groups_needs_llm += 1
                elif mode == 'diarization_protected':
                    groups_diarization_protected += 1
        
        # Update processing status
        words_updated = update_processing_status(word_table, stage_name)
        
        result['stats'].update({
            'duration': time.time() - start_time,
            'status': 'retranscription_completed',
            'segments_analyzed': len(bad_grammar_multi_segments),
            'bad_grammar_multi_segments': len(bad_grammar_multi_segments),
            'consecutive_groups_formed': len(segment_groups),
            'groups_retranscribed': segments_retranscribed,
            'groups_with_good_grammar': groups_with_good_grammar,
            'groups_diarization_protected': groups_diarization_protected,
            'groups_needs_llm': groups_needs_llm,
            'failed_retranscriptions': failed_retranscriptions,
            'words_reconstructed': total_words_reconstructed,
            'retranscription_time': total_retranscription_time,
            'words_updated': words_updated
        })
        
        # Summary logging
        logger.info(f"[{content_id}] STAGE 9 SUMMARY:")
        logger.info(f"[{content_id}]   Individual segments analyzed: {len(bad_grammar_multi_segments)}")
        logger.info(f"[{content_id}]   Consecutive groups formed: {len(segment_groups)}")
        logger.info(f"[{content_id}]   Groups successfully re-transcribed: {groups_with_good_grammar}")
        logger.info(f"[{content_id}]   Groups with diarization protection: {groups_diarization_protected}")
        logger.info(f"[{content_id}]   Groups assigned to NEEDS_LLM: {groups_needs_llm}")
        logger.info(f"[{content_id}]   Failed re-transcriptions: {failed_retranscriptions}")
        logger.info(f"[{content_id}]   Words reconstructed: {total_words_reconstructed}")
        logger.info(f"[{content_id}]   Total re-transcription time: {total_retranscription_time:.2f}s")
        logger.info(f"[{content_id}]   Default language: {language}")
        logger.info(f"[{content_id}]   Note: Language-based model selection - Parakeet MLX v2 for English, Whisper MLX Turbo for all other languages")
        logger.info(f"[{content_id}]   Note: Diarization protection uses 0.3s boundary buffer")
        logger.info(f"[{content_id}]   Note: NEEDS_LLM segments will be resolved in Stage 10")
        
        if test_mode:
            # Save detailed results in test mode
            test_output_dir = get_project_root() / "tests" / "content" / content_id / "outputs"
            test_output_dir.mkdir(parents=True, exist_ok=True)
            
            analysis_results = {
                'content_id': content_id,
                'stage': stage_name,
                'stats': result['stats'],
                'retranscription_tasks': len(retranscription_tasks),
                'segment_groups': len(segment_groups),
                'individual_segments': len(bad_grammar_multi_segments)
            }
            
            with open(test_output_dir / "bad_grammar_multi_retranscription.json", 'w') as f:
                json.dump(analysis_results, f, indent=2, default=str)
            logger.info(f"[{content_id}] Detailed re-transcription results saved to test outputs")
        
        format_stage_stats(content_id, stage_name, result['stats'], start_time)
        
        
        # Add summary of speaker assignments in test mode
        if test_mode:
            summarize_speaker_assignments(word_table, 'stage9_bad_grammar_multi', content_id, test_mode)
        
        return result
        
    except Exception as e:
        error_msg = str(e) if e else "Unknown error in bad grammar + multi-speaker re-transcription"
        logger.error(f"[{content_id}] Stage 9 failed: {error_msg}")
        logger.error(f"[{content_id}] Error details:", exc_info=True)
        
        result.update({
            'status': 'error',
            'error': error_msg,
            'stats': {
                'duration': time.time() - start_time,
                'status': 'retranscription_failed',
                'segments_processed': 0
            }
        })
        
        return result


if __name__ == "__main__":
    """Test the bad grammar + multi-speaker re-transcription stage independently."""
    import argparse
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = get_project_root()
    sys.path.append(str(project_root))
    
    parser = argparse.ArgumentParser(description='Test Bad Grammar + Multi-Speaker Re-transcription Stage')
    parser.add_argument('--content', required=True, help='Content ID to process')
    args = parser.parse_args()
    
    content_id = args.content
    
    logger.info(f"Testing Stage 9: Bad Grammar + Multi-Speaker Re-transcription for content {content_id}")
    
    try:
        # This would need to be implemented with proper test setup
        logger.info(f"Stage 9 test setup not yet implemented")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        logger.error(f"Error details:", exc_info=True)