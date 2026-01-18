#!/usr/bin/env python3
"""
Stitch Pipeline - Modular Word-Level Speaker Attribution
========================================================

This is a simplified implementation of the stitch pipeline with focused stages:
1. Loading diarization.json and transcript_words.json
2. Data cleaning and artifact removal
3. Creating word and segment tables for tracking
4. Good grammar + single speaker assignment (slam-dunk)
5. Bad grammar + single speaker assignment (with punctuation/NER)
6. Speaker embeddings extraction
7. Speaker centroids and database mapping
8. Good grammar + multi-speaker analysis
9. Bad grammar + multi-speaker analysis
10. Resolution stage for remaining conflicts
11. Final cleanup
12. Output generation (SpeakerTranscription to database)
13. Semantic segmentation for retrieval (EmbeddingSegment to database)

Each stage is implemented as a separate module for better maintainability
and testability.

Usage:
    python stitch.py --content <content_id> [--test] [--stages N]

Examples:
    python stitch.py --content Bdb001 --test --stages 4  # Stop after slam-dunk assignments
    python stitch.py --content Bdb001 --test --stages 12 # Stop after speaker turns (skip segmentation)
    python stitch.py --content Bdb001 --test --stages 13 # Run full pipeline with segmentation
    python stitch.py --content Bdb001 --test             # Run full pipeline
"""

# Centralized environment setup (must be before other imports)
from src.utils.env_setup import setup_env
setup_env('stitch')

import os

# PyTorch 2.6+ changed default weights_only=True which breaks loading older models
# that contain pickle-serialized objects like pytorch_lightning callbacks and omegaconf.
# Monkey-patch torch.load to use weights_only=False for all model loading.
# This must happen BEFORE any pyannote imports.
import torch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

import sys
import asyncio
import json
import argparse
import logging
import pandas as pd
import time
import yaml
from pathlib import Path

# Check required NLP dependencies early - fail fast with clear error
_missing_packages = []
try:
    import spacy
except ImportError:
    _missing_packages.append("spacy")
try:
    from langdetect import detect
except ImportError:
    _missing_packages.append("langdetect")
try:
    from deepmultilingualpunctuation import PunctuationModel
except ImportError:
    _missing_packages.append("deepmultilingualpunctuation")

if _missing_packages:
    raise ImportError(f"Missing required NLP packages for stitch: {', '.join(_missing_packages)}. Run: uv add {' '.join(_missing_packages)}")

from src.utils.paths import get_project_root
from src.utils.config import load_config
from typing import Optional, Tuple, Dict
import numpy as np
import tempfile
import shutil
import hashlib
from datetime import timezone

# Add the project root to Python path
sys.path.append(str(get_project_root()))

# Load environment variables (S3 credentials, etc.)
from dotenv import load_dotenv
load_dotenv(get_project_root() / '.env')

from src.utils.logger import setup_worker_logger
from src.storage.s3_utils import S3StorageConfig, S3Storage, create_s3_storage_from_config
from src.storage.content_storage import ContentStorageManager
from src.utils.error_codes import ErrorCode, create_error_result, create_success_result, create_skipped_result
from src.database.models import Content
from stitch_steps.stage1_load import load_stage
from stitch_steps.stage2_clean import clean_stage
from stitch_steps.stage3_tables import tables_stage
from stitch_steps.stage4_good_grammar_single import slamdunk_assignment_stage
from stitch_steps.stage5_bad_grammar_single import bad_grammar_single_assignment_stage
from stitch_steps.stage6_speaker_embeddings import speaker_stage as original_speaker_embeddings_stage
from stitch_steps.stage7_speaker_centroids import speaker_centroids_stage as original_speaker_centroids_stage
from stitch_steps.stage8_good_grammar_multi import good_grammar_multi_speaker_stage
from stitch_steps.stage9_bad_grammar_multi import bad_grammar_multi_speaker_stage
from stitch_steps.stage10_resolutions import stage10_resolutions
from stitch_steps.stage11_cleanup import stage11_cleanup
from stitch_steps.stage12_output import stage12_output
from stitch_steps.stage13_emotion import stage13_emotion, update_sentence_emotions
from stitch_steps.stage14_segment import stage14_segment
from stitch_steps.util_stitch import smart_join_words
from stitch_steps.util_cache import get_stage_cache, StageCache

logger = setup_worker_logger('stitch')
logger.setLevel(logging.INFO)


def _generate_stage_output_with_unknowns(word_table, include_unknowns: bool = True) -> tuple:
    """
    Generate transcripts that include UNKNOWN speakers for partial stage evaluation.
    
    Args:
        word_table: WordTable instance
        include_unknowns: Whether to include UNKNOWN speakers
        
    Returns:
        Tuple of (readable_transcript, detailed_transcript)
    """
    if word_table is None or word_table.df is None:
        return "ERROR: No word table available", "ERROR: No word table available"
    
    # Debug: Check word table state
    logger.debug(f"Word table debugging: {len(word_table.df)} total words")
    logger.debug(f"Word table columns: {list(word_table.df.columns)}")
    logger.debug(f"Word table dtypes: {word_table.df.dtypes.to_dict()}")
    
    # Check for any problematic data
    if 'text' in word_table.df.columns:
        text_types = word_table.df['text'].apply(type).value_counts()
        logger.debug(f"Text column types: {text_types.to_dict()}")
        
        # Check for any non-string text values
        non_string_mask = ~word_table.df['text'].apply(lambda x: isinstance(x, str))
        if non_string_mask.any():
            non_string_samples = word_table.df[non_string_mask]['text'].head(5).tolist()
            logger.warning(f"Found {non_string_mask.sum()} non-string text values. Samples: {non_string_samples}")
    else:
        logger.error("No 'text' column found in word table!")
    
    # Filter words based on whether to include unknowns
    try:
        if include_unknowns:
            # Include all words that have any speaker assignment (including UNKNOWN)
            valid_words = word_table.df[
                (word_table.df['speaker_current'].notna()) & 
                (word_table.df['speaker_current'] != '')
            ].copy()
        else:
            # Only include words with proper speaker assignments (original behavior)
            valid_words = word_table.df[
                (word_table.df['speaker_current'].notna()) & 
                (word_table.df['speaker_current'] != 'UNKNOWN') &
                (word_table.df['speaker_current'] != '')
            ].copy()
        
        logger.debug(f"Valid words type: {type(valid_words)}")
        logger.debug(f"Valid words shape: {valid_words.shape if hasattr(valid_words, 'shape') else 'no shape attr'}")
        logger.debug(f"Filtered to {len(valid_words)} valid words from {len(word_table.df)} total")
        
    except Exception as e:
        logger.error(f"Error filtering words: {e}")
        logger.error(f"Word table shape: {word_table.df.shape}")
        logger.error(f"Speaker current column sample: {word_table.df['speaker_current'].head(10).tolist()}")
        raise
    
    if len(valid_words) == 0:
        return "ERROR: No words with speaker assignments found", "ERROR: No words with speaker assignments found"
    
    sorted_words = valid_words.sort_values('start')

    # Detect transcription model type from word spacing
    sample_words = sorted_words['text'].head(50).tolist()
    sample_words = [w for w in sample_words if isinstance(w, str) and len(w) > 1]

    if sample_words:
        words_with_leading_space = sum(1 for w in sample_words if w[0] == ' ')
        model_type = 'parakeet' if words_with_leading_space / len(sample_words) > 0.7 else 'whisper'
    else:
        model_type = 'whisper'  # Default

    logger.info(f"Detected transcription model type: {model_type}")

    # Generate readable transcript
    transcript_lines = []
    current_speaker = None
    current_text = []
    last_end_time = 0
    
    logger.debug(f"Starting to iterate over {len(sorted_words)} sorted words")
    
    try:
        for word_idx, word in sorted_words.iterrows():
            # Check if this is a new speaker or significant time gap (>2s)
            is_new_segment = (
                word['speaker_current'] != current_speaker or
                word['start'] - last_end_time > 2.0
            )
            
            if is_new_segment and current_text:
                # Output previous segment
                timestamp = _format_timestamp(last_end_time)
                speaker_label = f"[{current_speaker}]"
                # Join words based on model type
                if model_type == 'parakeet':
                    # Parakeet: words already have leading spaces, just join them
                    text_output = "".join(current_text)
                else:
                    # Whisper: words don't have spaces, use smart_join_words
                    text_output = smart_join_words(current_text)
                transcript_lines.append(f"{timestamp} {speaker_label}: {text_output}")
                current_text = []
            
            # Update tracking
            current_speaker = word['speaker_current']
            # Debug: Check word text type
            word_text = word['text']
            if not isinstance(word_text, str):
                logger.warning(f"Non-string word text found: {word_text} (type: {type(word_text)})")
                word_text = str(word_text)
            current_text.append(word_text)
            last_end_time = word['end']
            
    except Exception as e:
        logger.error(f"Error during word iteration at index {word_idx}: {e}")
        logger.error(f"Current word: {word}")
        logger.error(f"Word keys: {list(word.keys()) if hasattr(word, 'keys') else 'not dict-like'}")
        raise
    
    # Add final segment
    if current_text:
        timestamp = _format_timestamp(last_end_time)
        speaker_label = f"[{current_speaker}]"
        # Join words based on model type
        if model_type == 'parakeet':
            # Parakeet: words already have leading spaces, just join them
            text_output = "".join(current_text)
        else:
            # Whisper: words don't have spaces, use smart_join_words
            text_output = smart_join_words(current_text)
        transcript_lines.append(f"{timestamp} {speaker_label}: {text_output}")
    
    readable_transcript = "\n".join(transcript_lines)
    
    # Generate detailed transcript
    detailed_lines = []
    current_speaker = None
    current_speaker_words = []
    speaker_start_time = None
    
    for _, word in sorted_words.iterrows():
        # Check if this is a new speaker segment
        if word['speaker_current'] != current_speaker:
            # Output previous speaker segment if exists
            if current_speaker is not None and current_speaker_words:
                # Create speaker header line
                timestamp = _format_timestamp(speaker_start_time)
                speaker_label = f"[{current_speaker}]"
                # Debug: Check for problematic text values
                try:
                    text_parts = []
                    for w in current_speaker_words:
                        text_part = w['text']
                        if not isinstance(text_part, str):
                            logger.warning(f"Non-string text found: {text_part} (type: {type(text_part)})")
                            text_part = str(text_part)
                        text_parts.append(text_part)
                    # Join words based on model type
                    if model_type == 'parakeet':
                        # Parakeet: words already have leading spaces, just join them
                        text = "".join(text_parts)
                    else:
                        # Whisper: words don't have spaces, use smart_join_words
                        text = smart_join_words(text_parts)
                except Exception as e:
                    logger.error(f"Error joining text parts: {e}")
                    logger.error(f"Current speaker words: {current_speaker_words}")
                    raise
                detailed_lines.append(f"{timestamp} {speaker_label} {text}")
                
                # Add indented word details
                for w in current_speaker_words:
                    word_text = w['text']
                    padded_word = f"{word_text:<20}"
                    word_timestamp = _format_timestamp(w['start'])
                    
                    # Get assignment method
                    method = w.get('assignment_method', 'none')
                    method = _format_method_8char(method)
                    
                    # Get confidence 
                    confidence = w.get('assignment_confidence', 0.0)
                    if pd.isna(confidence):
                        confidence_str = "nan"
                    else:
                        confidence_str = f"{confidence:.1f}"
                    
                    detailed_lines.append(f"    - {padded_word} [{w['start']:.2f}s] {word_timestamp} [{method}] [{confidence_str}]")
            
            # Reset for new speaker
            current_speaker = word['speaker_current']
            current_speaker_words = []
            speaker_start_time = word['start']
        
        # Add word to current speaker segment (convert Series to dict for consistent access)
        current_speaker_words.append(word.to_dict())
    
    # Output final speaker segment
    if current_speaker is not None and current_speaker_words:
        timestamp = _format_timestamp(speaker_start_time)
        speaker_label = f"[{current_speaker}]"
        text = smart_join_words([w['text'] for w in current_speaker_words])
        detailed_lines.append(f"{timestamp} {speaker_label} {text}")
        
        # Add indented word details
        for w in current_speaker_words:
            word_text = w['text']
            padded_word = f"{word_text:<20}"
            word_timestamp = _format_timestamp(w['start'])
            
            method = w.get('resolution_method', 'none')
            method = _format_method_8char(method)
            
            confidence = w.get('assignment_confidence', 0.0)
            if pd.isna(confidence):
                confidence_str = "nan"
            else:
                confidence_str = f"{confidence:.1f}"
            
            detailed_lines.append(f"    - {padded_word} [{w['start']:.2f}s] {word_timestamp} [{method}] [{confidence_str}]")
    
    detailed_transcript = "\n".join(detailed_lines)
    
    return readable_transcript, detailed_transcript


def _format_timestamp(seconds: float) -> str:
    """Format timestamp in both [30s] and [MM:SS] formats."""
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    return f"[{int(seconds)}s] [{minutes:02d}:{remaining_seconds:02d}]"


def _format_method_8char(method) -> str:
    """Format method name to exactly 8 characters for alignment."""
    # Handle non-string values (e.g., NaN, None, floats)
    if method is None or (isinstance(method, float) and pd.isna(method)):
        method = 'none'
    elif not isinstance(method, str):
        method = str(method)
    
    method_map = {
        # Stage 5: Speaker Assignment
        'diarization_confirmed': '5_diar_ok',
        'cross_speaker_reassignment': '5_cross_r',
        'diarization_trusted_low_similarity': '5_trust_d',
        'similarity_assignment': '5_sim_asn',
        'no_match_unknown': '5_no_mtch',
        'low_similarity_unknown': '5_low_sim',
        
        # Stage 7: Word-level Diarization
        'diarization_overlap': '7_diar_ov',
        'failed_unknown': '7_fail_uk',
        'failed_multi_speaker': '7_fail_ms',
        
        # Stage 8: Word Embeddings
        'embedding_similarity': '8_wrd_emb',
        'sentence_propagation': '8_sent_pr',
        'interruption_absorbed': '8_intr_ab',
        
        # Stage 9: LLM Coherence
        'llm_before': '9_llm_bef',
        'llm_after': '9_llm_aft',
        'llm_split': '9_llm_spl',
        'same_speaker_context': '9_same_sp',
        
        # Default/initial
        'none': 'none    ',
    }
    
    if method in method_map:
        return method_map[method]
    else:
        # Truncate or pad to exactly 8 characters
        if len(method) > 8:
            return method[:8]
        else:
            return method.ljust(8)


def _generate_stage_output(content_id: str, 
                          word_table, 
                          stage_number: int,
                          test_mode: bool = False) -> Dict[str, str]:
    """
    Generate output transcripts for any stage that has a word table.
    
    Args:
        content_id: Content ID
        word_table: WordTable instance
        stage_number: Stage number for identification
        test_mode: Whether to save files in test mode
        
    Returns:
        Dictionary with readable and detailed transcripts
    """
    if word_table is None:
        return {
            'readable_transcript': 'ERROR: No word table available',
            'detailed_transcript': 'ERROR: No word table available'
        }
    
    try:
        # For intermediate stages, include UNKNOWN speakers to show progress
        include_unknowns = stage_number < 11
        
        # Generate transcripts with or without unknowns
        readable_transcript, detailed_transcript = _generate_stage_output_with_unknowns(
            word_table, include_unknowns=include_unknowns
        )
        
        # Add stage information
        if include_unknowns:
            prefix = f"STAGE {stage_number} PARTIAL RESULTS (includes UNKNOWN speakers):\n" + "="*60 + "\n\n"
            readable_transcript = prefix + readable_transcript
            detailed_transcript = prefix + detailed_transcript
        
        # Save to files if in test mode
        if test_mode:
            output_dir = get_project_root() / "tests" / "content" / content_id / "outputs"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save readable transcript
            readable_path = output_dir / f"{content_id}_transcript_stage{stage_number}.txt"
            with open(readable_path, 'w') as f:
                f.write(readable_transcript)
            
            # Save detailed transcript
            detailed_path = output_dir / f"{content_id}_transcript_detailed_stage{stage_number}.txt"
            with open(detailed_path, 'w') as f:
                f.write(detailed_transcript)
            
            logger.info(f"[{content_id}] Stage {stage_number} transcripts saved to {output_dir}")
        
        return {
            'readable_transcript': readable_transcript,
            'detailed_transcript': detailed_transcript
        }
        
    except Exception as e:
        logger.error(f"[{content_id}] Failed to generate stage {stage_number} output: {e}")
        logger.error(f"[{content_id}] Full traceback:", exc_info=True)
        import traceback
        tb_str = traceback.format_exc()
        logger.error(f"[{content_id}] Detailed traceback: {tb_str}")
        return {
            'readable_transcript': f'ERROR: Failed to generate transcript: {e}',
            'detailed_transcript': f'ERROR: Failed to generate transcript: {e}'
        }


class NumpyJsonEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types and provides better error handling."""
    def default(self, obj):
        try:
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32,
                              np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, (np.str_, str)):
                return str(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.void):
                return None
            elif hasattr(obj, 'dtype') and obj.dtype.kind in ['i', 'u', 'f']:  # Handle other numpy scalar types
                return obj.item()
            elif hasattr(obj, 'tolist'):  # Handle any other array-like objects
                return obj.tolist()
            elif hasattr(obj, '__dict__'):  # Handle custom objects
                return {key: value for key, value in obj.__dict__.items() 
                       if not key.startswith('_')}  # Skip private attributes
            return super().default(obj)
        except Exception as e:
            logger.warning(f"JSON encoding error for object type {type(obj)}: {str(e)}")
            return str(obj)  # Fallback to string representation


class StitchPipeline:
    """Main stitch pipeline orchestrator."""
    
    def __init__(self):
        """Initialize the stitch pipeline with required components."""
        self.word_table = None
        self.segment_table = None
        self.speaker_assignment_engine = None
        
        # Load config using centralized loader (handles .env and ${VAR} substitution)
        logger.debug("Loading configuration with environment variable substitution...")
        try:
            self.config = load_config()
            logger.debug("Configuration loaded successfully.")
        except FileNotFoundError as e:
            logger.error(f"Config file not found: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load or parse config: {e}")
            raise
            
        # Store S3 config dict for fresh connections
        self.s3_config_dict = self.config['storage']['s3']
        # Create initial S3 storage for compatibility (rarely used)
        self.s3_storage = create_s3_storage_from_config(self.s3_config_dict)
        # Note: storage_manager is not used in the stitch pipeline, keeping for backward compatibility
        self.storage_manager = ContentStorageManager(self.s3_storage)
        logger.debug("S3 Storage and ContentStorageManager initialized.")
        
        # Set up temp directory
        self.temp_dir = Path("/tmp/stitch_temp")
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Temporary directory set to: {self.temp_dir}")
        
    def _get_fresh_s3_storage(self) -> S3Storage:
        """Get a fresh S3 storage connection to avoid stale connections."""
        return create_s3_storage_from_config(self.s3_config_dict)
        
    def _file_exists_flexible(self, s3_key: str) -> bool:
        """Check if a file exists in S3, handling both compressed and uncompressed versions.
        
        Uses a fresh S3 connection to avoid stale connection issues.
        
        Args:
            s3_key: S3 key for the file (without .gz extension)
            
        Returns:
            True if either compressed or uncompressed version exists
        """
        # Use fresh S3 connection for reliability
        fresh_s3 = self._get_fresh_s3_storage()
        return fresh_s3.file_exists_flexible(s3_key)
        
    async def process_content(self, 
                            content_id: str,
                            test_mode: bool = False,
                            time_range: Optional[Tuple[float, float]] = None,
                            stop_at_stage: Optional[int] = None,
                            start_from_stage: Optional[int] = None,
                            overwrite: bool = False,
                            use_cache: bool = True,
                            clear_cache: bool = False) -> dict:
        """
        Process content through the modular stitch pipeline.
        
        Args:
            content_id: Content ID to process
            test_mode: Whether to run in test mode (local file output)
            time_range: Optional (start, end) time range for focused processing
            stop_at_stage: Optional stage number to stop at (1-11)
            start_from_stage: Optional stage number to start from (requires test_mode for caching)
            overwrite: Whether to overwrite existing outputs
            use_cache: Whether to use stage caching (only applies in test_mode)
            clear_cache: Whether to clear all cached stages before running
            
        Returns:
            Dictionary with processing results
        """
        start_time = time.time()
        
        logger.info(f"[{content_id}] Starting modular stitch pipeline")
        logger.info(f"[{content_id}] Mode: {'test' if test_mode else 'production'}")
        logger.info(f"[{content_id}] Overwrite: {overwrite}")
        logger.info(f"[{content_id}] Cache: {'enabled' if use_cache else 'disabled'}")
        if time_range:
            logger.info(f"[{content_id}] Time range: {time_range[0]:.1f}s - {time_range[1]:.1f}s")
        if stop_at_stage:
            logger.info(f"[{content_id}] Will stop after stage {stop_at_stage}")
        if start_from_stage:
            logger.info(f"[{content_id}] Starting from stage {start_from_stage} using cached results")
        
        # Initialize cache - only use caching in test mode for consistency
        cache = get_stage_cache() if (use_cache and test_mode) else None
        
        # Validate incompatible argument combinations
        if start_from_stage and start_from_stage > 1 and not cache:
            raise ValueError(f"--start-from-stage requires caching to be enabled. Use --test mode to enable caching, or remove --start-from-stage to run the full pipeline.")
        
        # Clear cache if requested
        if clear_cache and cache:
            # Validate incompatible combination: can't clear cache and start from later stage
            if start_from_stage and start_from_stage > 1:
                raise ValueError(f"Cannot use --clear-cache with --start-from-stage {start_from_stage}. Either remove --clear-cache to use existing cached results, or remove --start-from-stage to run the full pipeline.")
            
            logger.info(f"[{content_id}] Clearing all cached stages")
            cache.clear_stage_cache(content_id)
        
        # If starting from a later stage, load cached results
        stage_results = {}
        if start_from_stage and start_from_stage > 1 and cache:
            logger.info(f"[{content_id}] Loading cached results for stages 1-{start_from_stage-1}")
            cached_stages = cache.list_cached_stages(content_id)
            logger.info(f"[{content_id}] Available cached stages: {cached_stages}")
            
            # We'll load these as we go through the pipeline
        
        # Set up output directory for test mode
        output_dir = None
        if test_mode:
            output_dir = get_project_root() / "tests" / "content" / content_id / "outputs"
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[{content_id}] Test output directory: {output_dir}")
        
        result = {
            'status': 'pending',
            'content_id': content_id,
            'test_mode': test_mode,
            'time_range': time_range,
            'processing_stages': {},
            'error': None
        }
        
        try:
            # Check if outputs already exist (unless overwrite is True)
            if not overwrite and not test_mode:
                # Check S3 for existing outputs
                transcript_key = f"content/{content_id}/transcript_diarized.json"
                centroids_key = f"content/{content_id}/speaker_centroids.json"
                
                logger.info(f"[{content_id}] Checking for existing stitch outputs...")
                transcript_exists = self._file_exists_flexible(transcript_key)
                centroids_exist = self._file_exists_flexible(centroids_key)
                
                # Check database for is_stitched flag
                from src.database.models import Content
                from src.database.session import get_session
                
                db_is_stitched = False
                main_language = 'en'  # Default language
                with get_session() as session:
                    content = session.query(Content).filter_by(content_id=content_id).first()
                    if content:
                        if content.is_stitched:
                            db_is_stitched = True
                        if content.main_language:
                            main_language = content.main_language
                            logger.debug(f"[{content_id}] Content language: {main_language}")
                
                logger.info(f"[{content_id}] Output check results:")
                logger.info(f"[{content_id}]   - transcript_diarized.json: {transcript_exists}")
                logger.info(f"[{content_id}]   - speaker_centroids.json: {centroids_exist}")
                logger.info(f"[{content_id}]   - Database is_stitched: {db_is_stitched}")
                
                # If all outputs exist, skip processing
                if transcript_exists and centroids_exist and db_is_stitched:
                    logger.info(f"[{content_id}] Stitch outputs already exist, marking as skipped")
                    logger.info(f"[{content_id}] Skipping stitch processing (use --rewrite to force)")
                    
                    result.update({
                        'status': 'skipped',
                        'reason': 'outputs_exist',
                        'existing_outputs': {
                            'transcript_diarized': transcript_key,
                            'speaker_centroids': centroids_key,
                            'is_stitched': db_is_stitched
                        },
                        'skipped_existing': True,
                        'processing_time': time.time() - start_time
                    })
                    return result
            
            # Stage 1: Load raw data
            logger.info("\n" + "="*80)
            logger.info(f"[{content_id}] STAGE 1: LOADING RAW DATA")
            logger.info("="*80)
            
            # Check cache or run stage
            stage1_result = None
            if start_from_stage and start_from_stage > 1 and cache:
                stage1_result = cache.load_stage_result(content_id, 'stage1_load')
                if stage1_result:
                    logger.debug(f"[{content_id}] Using cached stage 1 results")
            
            if stage1_result is None:
                stage1_start = time.time()
                stage1_result = await load_stage(
                    content_id=content_id,
                    test_mode=test_mode,
                    time_range=time_range,
                    s3_storage=self._get_fresh_s3_storage()
                )
                stage1_duration = time.time() - stage1_start
                
                # Cache successful results
                if cache and stage1_result['status'] == 'success':
                    cache.save_stage_result(content_id, 'stage1_load', stage1_result)
            else:
                stage1_duration = stage1_result['stats'].get('duration', 0)
            
            if stage1_result['status'] != 'success':
                raise ValueError(f"Stage 1 failed: {stage1_result.get('error', 'Unknown error')}")
            
            result['processing_stages']['data_loading'] = stage1_result['stats']
            result['processing_stages']['data_loading']['duration'] = stage1_duration
            
            diarization_data = stage1_result['data']['diarization_data']
            transcript_data = stage1_result['data']['transcript_data']
            audio_path = stage1_result['data'].get('audio_path')
            
            logger.info(f"[{content_id}] Stage 1 completed in {stage1_duration:.2f}s: {stage1_result['stats']['diarization_segments']} diarization, {stage1_result['stats']['transcript_segments']} transcript segments")
            
            # Load content language information if not already loaded
            if 'main_language' not in locals():
                main_language = 'en'  # Default language
                from src.database.models import Content
                from src.database.session import get_session
                
                with get_session() as session:
                    content = session.query(Content).filter_by(content_id=content_id).first()
                    if content and content.main_language:
                        main_language = content.main_language
                        logger.debug(f"[{content_id}] Content language: {main_language}")
            
            # Check if we should stop after stage 1
            if stop_at_stage == 1:
                logger.info(f"[{content_id}] Stopping after stage 1 as requested")
                
                total_time = time.time() - start_time
                result.update({
                    'status': 'success',
                    'stopped_at_stage': 1,
                    'processing_time': total_time,
                    'readable_transcript': 'Stage 1 does not have speaker assignments yet.',
                    'detailed_transcript': 'Stage 1 only loads raw data - no speaker assignments available.',
                    'summary': {
                        'diarization_segments': stage1_result['stats']['diarization_segments'],
                        'transcript_segments': stage1_result['stats']['transcript_segments'],
                        'stopped_at': 'stage_1_data_loading'
                    }
                })
                
                logger.info(f"[{content_id}] Pipeline stopped at stage 1 in {total_time:.2f}s")
                return result
            
            # Stage 2: Data cleaning and artifact removal
            logger.info("\n" + "="*80)
            logger.info(f"[{content_id}] STAGE 2: DATA CLEANING AND ARTIFACT REMOVAL")
            logger.info("="*80)
            
            # Check cache or run stage
            stage2_result = None
            if cache and (start_from_stage is None or start_from_stage > 2):
                stage2_result = cache.load_stage_result(content_id, 'stage2_clean')
                if stage2_result:
                    logger.debug(f"[{content_id}] Using cached stage 2 results")
            
            if stage2_result is None and (start_from_stage is None or start_from_stage <= 2):
                stage2_start = time.time()
                stage2_result = clean_stage(content_id, diarization_data, transcript_data, test_mode)
                stage2_duration = time.time() - stage2_start
                
                # Cache successful results
                if cache and stage2_result['status'] == 'success':
                    cache.save_stage_result(content_id, 'stage2_clean', stage2_result)
            else:
                stage2_duration = stage2_result['stats'].get('duration', 0) if stage2_result else 0
            
            # Validate that we have stage 2 results (the upfront validation should prevent this)
            if stage2_result is None:
                raise ValueError(f"Stage 2 results not available. This should not happen with current validation logic.")
            
            if stage2_result['status'] != 'success':
                raise ValueError(f"Stage 2 failed: {stage2_result.get('error', 'Unknown error')}")
            
            result['processing_stages']['data_cleaning'] = stage2_result['stats']
            result['processing_stages']['data_cleaning']['duration'] = stage2_duration
            
            cleaned_diarization_data = stage2_result['data']['cleaned_diarization_data']
            cleaned_transcript_data = stage2_result['data']['cleaned_transcript_data']
            
            cleaning_stats = stage2_result['stats']['transcript_cleaning_stats']
            logger.info(f"[{content_id}] Stage 2 completed in {stage2_duration:.2f}s: {cleaning_stats['total_final_words']} words, {cleaning_stats.get('total_artifacts_removed', 0)} artifacts removed")
            
            # Check if we should stop after stage 2
            if stop_at_stage == 2:
                logger.info(f"[{content_id}] Stopping after stage 2 as requested")
                
                total_time = time.time() - start_time
                result.update({
                    'status': 'success',
                    'stopped_at_stage': 2,
                    'processing_time': total_time,
                    'readable_transcript': 'Stage 2 does not have speaker assignments yet.',
                    'detailed_transcript': 'Stage 2 only cleans raw data - no speaker assignments available.',
                    'summary': {
                        'cleaned_words': cleaning_stats['total_final_words'],
                        'artifacts_removed': cleaning_stats.get('total_artifacts_removed', 0),
                        'cleaned_transcript_segments': len(cleaned_transcript_data.get('segments', [])),
                        'cleaned_diarization_segments': len(cleaned_diarization_data.get('segments', [])),
                        'stopped_at': 'stage_2_data_cleaning'
                    }
                })
                
                logger.info(f"[{content_id}] Pipeline stopped at stage 2 in {total_time:.2f}s")
                return result
            
            
            # Stage 3: Create word and segment tables
            logger.info("\n" + "="*80)
            logger.info(f"[{content_id}] STAGE 3: CREATING WORD AND SEGMENT TABLES")
            logger.info("="*80)
            
            # Check cache or run stage
            stage3_result = None
            if cache and (start_from_stage is None or start_from_stage > 3):
                stage3_result = cache.load_stage_result(content_id, 'stage3_tables')
                if stage3_result:
                    logger.debug(f"[{content_id}] Using cached stage 3 results")
            
            if stage3_result is None and (start_from_stage is None or start_from_stage <= 3):
                stage3_start = time.time()
                stage3_result = tables_stage(content_id, cleaned_diarization_data, cleaned_transcript_data)
                stage3_duration = time.time() - stage3_start
                
                # Cache successful results
                if cache and stage3_result['status'] == 'success':
                    cache.save_stage_result(content_id, 'stage3_tables', stage3_result)
            else:
                stage3_duration = stage3_result['stats'].get('duration', 0) if stage3_result else 0
            
            # Validate that we have stage 3 results (the upfront validation should prevent this)
            if stage3_result is None:
                raise ValueError(f"Stage 3 results not available. This should not happen with current validation logic.")
            
            if stage3_result['status'] != 'success':
                raise ValueError(f"Stage 3 failed: {stage3_result.get('error', 'Unknown error')}")
            
            result['processing_stages']['table_creation'] = stage3_result['stats']
            result['processing_stages']['table_creation']['duration'] = stage3_duration
            
            # Trust the cache to have properly reconstructed objects
            self.word_table = stage3_result['data']['word_table']
            self.segment_table = stage3_result['data']['segment_table']

            # Check for empty word table - indicates no transcribable content
            if self.word_table is None or len(self.word_table.df) == 0:
                raise ValueError("Empty word table - no transcribable content found")

            logger.info(f"[{content_id}] Stage 3 completed in {stage3_duration:.2f}s: {stage3_result['stats']['words_created']} words, {stage3_result['stats']['segments_created']} segments")
            
            # Check if we should stop after stage 3
            if stop_at_stage == 3:
                logger.info(f"[{content_id}] Stopping after stage 3 as requested")
                
                # Generate output transcripts
                stage_output = _generate_stage_output(content_id, self.word_table, 3, test_mode)
                
                total_time = time.time() - start_time
                result.update({
                    'status': 'success',
                    'stopped_at_stage': 3,
                    'processing_time': total_time,
                    'readable_transcript': stage_output['readable_transcript'],
                    'detailed_transcript': stage_output['detailed_transcript'],
                    'summary': {
                        'words_processed': stage3_result['stats']['words_created'],
                        'segments_processed': stage3_result['stats']['segments_created'],
                        'artifacts_removed': cleaning_stats.get('total_artifacts_removed', 0),
                        'stopped_at': 'stage_3_tables'
                    }
                })
                
                logger.info(f"[{content_id}] Pipeline stopped at stage 3 in {total_time:.2f}s")
                return result
            
            # Stage 4: Good Grammar + Single Speaker Assignment (Slam-Dunk)
            logger.info("\n" + "="*80)
            logger.info(f"[{content_id}] STAGE 4: GOOD GRAMMAR + SINGLE SPEAKER ASSIGNMENT (SLAM-DUNK)")
            logger.info("="*80)
            
            # Check cache or run stage
            stage4_slamdunk_result = None
            if cache and (start_from_stage is None or start_from_stage > 4):
                stage4_slamdunk_result = cache.load_stage_result(content_id, 'stage4_slamdunk')
                if stage4_slamdunk_result:
                    logger.debug(f"[{content_id}] Using cached stage 4 results")
            
            if stage4_slamdunk_result is None and (start_from_stage is None or start_from_stage <= 4):
                stage4_slamdunk_start = time.time()
                stage4_slamdunk_result = slamdunk_assignment_stage(
                    content_id, 
                    self.word_table, 
                    cleaned_diarization_data,
                    test_mode=test_mode
                )
                stage4_slamdunk_duration = time.time() - stage4_slamdunk_start
                
                # Cache successful results
                if cache and stage4_slamdunk_result['status'] == 'success':
                    cache.save_stage_result(content_id, 'stage4_slamdunk', stage4_slamdunk_result)
            else:
                stage4_slamdunk_duration = stage4_slamdunk_result['stats'].get('duration', 0) if stage4_slamdunk_result else 0
            
            # Validate that we have stage 4 results
            if stage4_slamdunk_result is None:
                raise ValueError(f"Stage 4 results not available. This should not happen with current validation logic.")
            
            if stage4_slamdunk_result['status'] != 'success':
                raise ValueError(f"Stage 4 failed: {stage4_slamdunk_result.get('error', 'Unknown error')}")
            
            result['processing_stages']['slamdunk_assignment'] = stage4_slamdunk_result['stats']
            result['processing_stages']['slamdunk_assignment']['duration'] = stage4_slamdunk_duration
            
            # Update word table from stage 4 results
            self.word_table = stage4_slamdunk_result['data']['word_table']
            
            slamdunk_stats = stage4_slamdunk_result['stats']
            logger.info(f"[{content_id}] Stage 4 completed in {stage4_slamdunk_duration:.2f}s: {slamdunk_stats['slam_dunk_assignments']} slam-dunk assignments ({slamdunk_stats.get('assignment_rate', 0):.1f}%)")
            
            # Wav2Vec2 utility is disabled and not called
            # (Script remains available but is not executed in the pipeline)
            
            # Check if we should stop after stage 4
            if stop_at_stage == 4:
                logger.info(f"[{content_id}] Stopping after stage 4 as requested")
                
                # Generate output transcripts
                stage_output = _generate_stage_output(content_id, self.word_table, 4, test_mode)
                
                total_time = time.time() - start_time
                result.update({
                    'status': 'success',
                    'stopped_at_stage': 4,
                    'processing_time': total_time,
                    'readable_transcript': stage_output['readable_transcript'],
                    'detailed_transcript': stage_output['detailed_transcript'],
                    'summary': {
                        'words_processed': stage3_result['stats']['words_created'],
                        'segments_processed': stage3_result['stats']['segments_created'],
                        'artifacts_removed': cleaning_stats.get('total_artifacts_removed', 0),
                        'slamdunk_assignments': slamdunk_stats,
                        'stopped_at': 'stage_4_slamdunk_assignment'
                    }
                })
                
                logger.info(f"[{content_id}] Pipeline stopped at stage 4 in {total_time:.2f}s")
                return result
            
            # Stage 5: Bad Grammar + Single Speaker Assignment
            logger.info("\n" + "="*80)
            logger.info(f"[{content_id}] STAGE 5: BAD GRAMMAR + SINGLE SPEAKER ASSIGNMENT")
            logger.info("="*80)
            
            # Check cache or run stage
            stage5_bad_grammar_result = None
            if cache and (start_from_stage is None or start_from_stage > 5):
                stage5_bad_grammar_result = cache.load_stage_result(content_id, 'stage5_bad_grammar_single')
                if stage5_bad_grammar_result:
                    logger.debug(f"[{content_id}] Using cached stage 5 results")
            
            if stage5_bad_grammar_result is None and (start_from_stage is None or start_from_stage <= 5):
                stage5_bad_grammar_start = time.time()
                stage5_bad_grammar_result = bad_grammar_single_assignment_stage(
                    content_id, 
                    self.word_table, 
                    cleaned_diarization_data,
                    test_mode=test_mode
                )
                stage5_bad_grammar_duration = time.time() - stage5_bad_grammar_start
                
                # Cache successful results
                if cache and stage5_bad_grammar_result['status'] == 'success':
                    cache.save_stage_result(content_id, 'stage5_bad_grammar_single', stage5_bad_grammar_result)
            else:
                stage5_bad_grammar_duration = stage5_bad_grammar_result['stats'].get('duration', 0) if stage5_bad_grammar_result else 0
            
            # Validate that we have stage 5 results
            if stage5_bad_grammar_result is None:
                raise ValueError(f"Stage 5 results not available. This should not happen with current validation logic.")
            
            if stage5_bad_grammar_result['status'] != 'success':
                raise ValueError(f"Stage 5 failed: {stage5_bad_grammar_result.get('error', 'Unknown error')}")
            
            result['processing_stages']['bad_grammar_single_assignment'] = stage5_bad_grammar_result['stats']
            result['processing_stages']['bad_grammar_single_assignment']['duration'] = stage5_bad_grammar_duration
            
            # Update word table from stage 5 results
            self.word_table = stage5_bad_grammar_result['data']['word_table']
            
            bad_grammar_stats = stage5_bad_grammar_result['stats']
            logger.info(f"[{content_id}] Stage 5 completed in {stage5_bad_grammar_duration:.2f}s: {bad_grammar_stats['bad_grammar_single_assignments']} assignments ({bad_grammar_stats.get('assignment_rate', 0):.1f}%), enhanced {bad_grammar_stats.get('text_enhanced', 0)} segments")
            
            # Check if we should stop after stage 5
            if stop_at_stage == 5:
                logger.info(f"[{content_id}] Stopping after stage 5 as requested")
                
                # Generate output transcripts
                stage_output = _generate_stage_output(content_id, self.word_table, 5, test_mode)
                
                total_time = time.time() - start_time
                result.update({
                    'status': 'success',
                    'stopped_at_stage': 5,
                    'processing_time': total_time,
                    'readable_transcript': stage_output['readable_transcript'],
                    'detailed_transcript': stage_output['detailed_transcript'],
                    'summary': {
                        'words_processed': stage3_result['stats']['words_created'],
                        'segments_processed': stage3_result['stats']['segments_created'],
                        'artifacts_removed': cleaning_stats.get('total_artifacts_removed', 0),
                        'slamdunk_assignments': slamdunk_stats,
                        'bad_grammar_single_assignments': bad_grammar_stats,
                        'stopped_at': 'stage_5_bad_grammar_single_assignment'
                    }
                })
                
                logger.info(f"[{content_id}] Pipeline stopped at stage 5 in {total_time:.2f}s")
                return result
            
            # Stage 6: Speaker Embeddings (Placeholder)
            logger.info("\n" + "="*80)
            logger.info(f"[{content_id}] STAGE 6: SPEAKER EMBEDDINGS")
            logger.info("="*80)
            
            # Check cache or run stage
            stage6_good_grammar_multi_result = None
            if cache and (start_from_stage is None or start_from_stage > 6):
                stage6_good_grammar_multi_result = cache.load_stage_result(content_id, 'stage6_good_grammar_multi')
                if stage6_good_grammar_multi_result:
                    logger.debug(f"[{content_id}] Using cached stage 6 results")
            
            if stage6_good_grammar_multi_result is None and (start_from_stage is None or start_from_stage <= 6):
                stage6_good_grammar_multi_start = time.time()
                # Stage 6 handles its own speaker analysis internally
                # Pass empty speaker_context_stats to let stage 6 use its own logic
                speaker_context_stats = {}
                
                # Call the actual speaker embeddings stage
                stage6_good_grammar_multi_result = await original_speaker_embeddings_stage(
                    content_id=content_id,
                    word_table=self.word_table,
                    speaker_context_stats=speaker_context_stats,
                    config=self.config,
                    audio_path=audio_path,
                    test_mode=test_mode
                )
                stage6_good_grammar_multi_duration = time.time() - stage6_good_grammar_multi_start
                
                # Cache successful results
                if cache and stage6_good_grammar_multi_result['status'] == 'success':
                    cache.save_stage_result(content_id, 'stage6_good_grammar_multi', stage6_good_grammar_multi_result)
            else:
                stage6_good_grammar_multi_duration = stage6_good_grammar_multi_result['stats'].get('duration', 0) if stage6_good_grammar_multi_result else 0
            
            # Validate that we have stage 6 results
            if stage6_good_grammar_multi_result is None:
                raise ValueError(f"Stage 6 results not available. This should not happen with current validation logic.")
            
            if stage6_good_grammar_multi_result['status'] != 'success':
                raise ValueError(f"Stage 6 failed: {stage6_good_grammar_multi_result.get('error', 'Unknown error')}")
            
            result['processing_stages']['good_grammar_multi_analysis'] = stage6_good_grammar_multi_result['stats']
            result['processing_stages']['good_grammar_multi_analysis']['duration'] = stage6_good_grammar_multi_duration
            
            # Update word table from stage 6 results
            self.word_table = stage6_good_grammar_multi_result['data']['word_table']
            
            good_grammar_multi_stats = stage6_good_grammar_multi_result['stats']
            logger.info(f"[{content_id}] Stage 6 completed in {stage6_good_grammar_multi_duration:.2f}s: speaker embeddings processed")
            
            # Check if we should stop after stage 6
            if stop_at_stage == 6:
                logger.info(f"[{content_id}] Stopping after stage 6 as requested")
                
                # Generate output transcripts
                stage_output = _generate_stage_output(content_id, self.word_table, 6, test_mode)
                
                total_time = time.time() - start_time
                result.update({
                    'status': 'success',
                    'stopped_at_stage': 6,
                    'processing_time': total_time,
                    'readable_transcript': stage_output['readable_transcript'],
                    'detailed_transcript': stage_output['detailed_transcript'],
                    'summary': {
                        'words_processed': stage3_result['stats']['words_created'],
                        'segments_processed': stage3_result['stats']['segments_created'],
                        'artifacts_removed': cleaning_stats.get('total_artifacts_removed', 0),
                        'slamdunk_assignments': slamdunk_stats,
                        'bad_grammar_single_assignments': bad_grammar_stats,
                        'good_grammar_multi_analysis': good_grammar_multi_stats,
                        'stopped_at': 'stage_6_good_grammar_multi_analysis'
                    }
                })
                
                logger.info(f"[{content_id}] Pipeline stopped at stage 6 in {total_time:.2f}s")
                return result
            
            # Stage 7: Speaker Centroids (Placeholder)
            logger.info("\n" + "="*80)
            logger.info(f"[{content_id}] STAGE 7: SPEAKER CENTROIDS")
            logger.info("="*80)
            
            # Check cache or run stage
            stage7_bad_grammar_multi_result = None
            if cache and (start_from_stage is None or start_from_stage > 7):
                stage7_bad_grammar_multi_result = cache.load_stage_result(content_id, 'stage7_bad_grammar_multi')
                if stage7_bad_grammar_multi_result:
                    logger.debug(f"[{content_id}] Using cached stage 7 results")
            
            if stage7_bad_grammar_multi_result is None and (start_from_stage is None or start_from_stage <= 7):
                stage7_bad_grammar_multi_start = time.time()
                # Stage 7 expects speaker centroid data from stage 6
                speaker_centroid_data = {}
                if stage6_good_grammar_multi_result and 'data' in stage6_good_grammar_multi_result:
                    # Extract centroid data from stage 6 results if available
                    if 'assignment_result' in stage6_good_grammar_multi_result['data']:
                        assignment_result = stage6_good_grammar_multi_result['data']['assignment_result']
                        # Use speaker_centroid_data which contains the full metadata, not speaker_centroids which is just vectors
                        if assignment_result and 'speaker_centroid_data' in assignment_result:
                            speaker_centroid_data = assignment_result['speaker_centroid_data']
                        elif assignment_result and 'speaker_centroids' in assignment_result:
                            # Fallback to speaker_centroids if speaker_centroid_data not available (shouldn't happen)
                            logger.warning(f"[{content_id}] Using speaker_centroids instead of speaker_centroid_data - this may cause issues")
                            speaker_centroid_data = assignment_result['speaker_centroids']
                
                # Call the actual speaker centroids stage
                original_diarization_speakers = stage1_result['data'].get('original_diarization_speakers', [])
                stage7_bad_grammar_multi_result = original_speaker_centroids_stage(
                    content_id=content_id,
                    word_table=self.word_table,
                    speaker_centroid_data=speaker_centroid_data,
                    test_mode=test_mode,
                    overwrite=overwrite,
                    original_diarization_speakers=original_diarization_speakers
                )
                stage7_bad_grammar_multi_duration = time.time() - stage7_bad_grammar_multi_start
                
                # Cache successful results
                if cache and stage7_bad_grammar_multi_result['status'] == 'success':
                    cache.save_stage_result(content_id, 'stage7_bad_grammar_multi', stage7_bad_grammar_multi_result)
            else:
                stage7_bad_grammar_multi_duration = stage7_bad_grammar_multi_result['stats'].get('duration', 0) if stage7_bad_grammar_multi_result else 0
            
            # Validate that we have stage 7 results
            if stage7_bad_grammar_multi_result is None:
                raise ValueError(f"Stage 7 results not available. This should not happen with current validation logic.")
            
            if stage7_bad_grammar_multi_result['status'] != 'success':
                raise ValueError(f"Stage 7 failed: {stage7_bad_grammar_multi_result.get('error', 'Unknown error')}")
            
            result['processing_stages']['bad_grammar_multi_analysis'] = stage7_bad_grammar_multi_result['stats']
            result['processing_stages']['bad_grammar_multi_analysis']['duration'] = stage7_bad_grammar_multi_duration
            
            # Update word table from stage 7 results
            self.word_table = stage7_bad_grammar_multi_result['data']['word_table']
            
            bad_grammar_multi_stats = stage7_bad_grammar_multi_result['stats']
            logger.info(f"[{content_id}] Stage 7 completed in {stage7_bad_grammar_multi_duration:.2f}s: speaker centroids processed")
            
            # Check if we should stop after stage 7
            if stop_at_stage == 7:
                logger.info(f"[{content_id}] Stopping after stage 7 as requested")
                
                # Save speaker_centroids.json to S3 before stopping
                try:
                    centroids_key = f"content/{content_id}/speaker_centroids.json"
                    
                    # Get speaker centroids from stage 6 results
                    speaker_centroids_to_save = {}
                    if stage6_good_grammar_multi_result and 'data' in stage6_good_grammar_multi_result:
                        if 'assignment_result' in stage6_good_grammar_multi_result['data']:
                            assignment_result = stage6_good_grammar_multi_result['data']['assignment_result']
                            # Get the speaker_centroid_data which contains full metadata
                            if assignment_result and 'speaker_centroid_data' in assignment_result:
                                speaker_centroids_to_save = assignment_result['speaker_centroid_data']
                            elif assignment_result and 'speaker_centroids' in assignment_result:
                                # Fallback to speaker_centroids if speaker_centroid_data not available
                                speaker_centroids_to_save = assignment_result['speaker_centroids']
                    
                    if speaker_centroids_to_save:
                        # Convert numpy arrays to lists for JSON serialization
                        serializable_centroids = {}
                        for speaker_id, centroid_info in speaker_centroids_to_save.items():
                            if isinstance(centroid_info, dict):
                                serializable_centroid = {}
                                for key, value in centroid_info.items():
                                    if isinstance(value, np.ndarray):
                                        serializable_centroid[key] = value.tolist()
                                    elif isinstance(value, (np.float32, np.float64, np.int32, np.int64)):
                                        # Convert numpy scalars to Python types
                                        serializable_centroid[key] = float(value) if 'float' in str(type(value)) else int(value)
                                    else:
                                        serializable_centroid[key] = value
                                serializable_centroids[speaker_id] = serializable_centroid
                            else:
                                # If it's not a dict, try to convert directly
                                if hasattr(centroid_info, '__dict__'):
                                    # It's an object, convert to dict
                                    obj_dict = {}
                                    for attr in dir(centroid_info):
                                        if not attr.startswith('_'):
                                            val = getattr(centroid_info, attr)
                                            if isinstance(val, np.ndarray):
                                                obj_dict[attr] = val.tolist()
                                            elif isinstance(val, (np.float32, np.float64, np.int32, np.int64)):
                                                # Convert numpy scalars to Python types
                                                obj_dict[attr] = float(val) if 'float' in str(type(val)) else int(val)
                                            elif not callable(val):
                                                obj_dict[attr] = val
                                    serializable_centroids[speaker_id] = obj_dict
                                else:
                                    serializable_centroids[speaker_id] = centroid_info
                        
                        centroids_data = {
                            'content_id': content_id,
                            'created_at': pd.Timestamp.now().isoformat(),
                            'speaker_centroids': serializable_centroids
                        }
                        
                        # Create temp file and save
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                            json.dump(centroids_data, f, indent=2, cls=NumpyJsonEncoder)
                            centroids_path = f.name
                        
                        # Upload to S3
                        fresh_s3 = self._get_fresh_s3_storage()
                        if fresh_s3.upload_file(centroids_path, centroids_key):
                            logger.info(f"[{content_id}] Uploaded speaker_centroids.json to S3: {centroids_key}")
                        else:
                            logger.warning(f"[{content_id}] Failed to upload speaker_centroids.json to S3: {centroids_key}")
                        
                        # Clean up temp file
                        os.unlink(centroids_path)
                    else:
                        logger.debug(f"[{content_id}] No speaker centroids to save at stage 7")
                        
                except Exception as e:
                    logger.warning(f"[{content_id}] Failed to save speaker_centroids.json: {e}")
                    # Don't fail the pipeline for this
                
                # Generate output transcripts
                stage_output = _generate_stage_output(content_id, self.word_table, 7, test_mode)
                
                total_time = time.time() - start_time
                result.update({
                    'status': 'success',
                    'stopped_at_stage': 7,
                    'processing_time': total_time,
                    'readable_transcript': stage_output['readable_transcript'],
                    'detailed_transcript': stage_output['detailed_transcript'],
                    'summary': {
                        'words_processed': stage3_result['stats']['words_created'],
                        'segments_processed': stage3_result['stats']['segments_created'],
                        'artifacts_removed': cleaning_stats.get('total_artifacts_removed', 0),
                        'slamdunk_assignments': slamdunk_stats,
                        'bad_grammar_single_assignments': bad_grammar_stats,
                        'good_grammar_multi_analysis': good_grammar_multi_stats,
                        'bad_grammar_multi_analysis': bad_grammar_multi_stats,
                        'stopped_at': 'stage_7_bad_grammar_multi_analysis'
                    }
                })
                
                logger.info(f"[{content_id}] Pipeline stopped at stage 7 in {total_time:.2f}s")
                return result
            
            # Stage 8: Good Grammar + Multi-Speaker Analysis
            logger.info("\n" + "="*80)
            logger.info(f"[{content_id}] STAGE 8: GOOD GRAMMAR + MULTI-SPEAKER ANALYSIS")
            logger.info("="*80)
            
            # Check cache or run stage
            stage8_good_grammar_multi_result = None
            if cache and (start_from_stage is None or start_from_stage > 8):
                stage8_good_grammar_multi_result = cache.load_stage_result(content_id, 'stage8_good_grammar_multi')
                if stage8_good_grammar_multi_result:
                    logger.debug(f"[{content_id}] Using cached stage 8 results")
            
            if stage8_good_grammar_multi_result is None and (start_from_stage is None or start_from_stage <= 8):
                stage8_good_grammar_multi_start = time.time()
                stage8_good_grammar_multi_result = good_grammar_multi_speaker_stage(
                    content_id, 
                    self.word_table, 
                    cleaned_diarization_data,
                    s3_storage=self._get_fresh_s3_storage(),
                    test_mode=test_mode
                )
                stage8_good_grammar_multi_duration = time.time() - stage8_good_grammar_multi_start
                
                # Cache successful results
                if cache and stage8_good_grammar_multi_result['status'] == 'success':
                    cache.save_stage_result(content_id, 'stage8_good_grammar_multi', stage8_good_grammar_multi_result)
            else:
                stage8_good_grammar_multi_duration = stage8_good_grammar_multi_result['stats'].get('duration', 0) if stage8_good_grammar_multi_result else 0
            
            # Validate that we have stage 8 results
            if stage8_good_grammar_multi_result is None:
                raise ValueError(f"Stage 8 results not available. This should not happen with current validation logic.")
            
            if stage8_good_grammar_multi_result['status'] != 'success':
                raise ValueError(f"Stage 8 failed: {stage8_good_grammar_multi_result.get('error', 'Unknown error')}")
            
            result['processing_stages']['good_grammar_multi_analysis'] = stage8_good_grammar_multi_result['stats']
            result['processing_stages']['good_grammar_multi_analysis']['duration'] = stage8_good_grammar_multi_duration
            
            # Update word table from stage 8 results
            # Preserve speaker_db_dictionary through stage 8
            new_word_table = stage8_good_grammar_multi_result['data']['word_table']
            if hasattr(self.word_table, 'speaker_db_dictionary'):
                new_word_table.speaker_db_dictionary = self.word_table.speaker_db_dictionary
                logger.debug(f"[{content_id}] Preserved speaker_db_dictionary through stage 8")
            self.word_table = new_word_table
            
            good_grammar_multi_stats = stage8_good_grammar_multi_result['stats']
            logger.info(f"[{content_id}] Stage 8 completed in {stage8_good_grammar_multi_duration:.2f}s: analyzed {good_grammar_multi_stats.get('good_grammar_multi_segments', 0)} good grammar + multi-speaker segments")
            
            # Check if we should stop after stage 8
            if stop_at_stage == 8:
                logger.info(f"[{content_id}] Stopping after stage 8 as requested")
                
                # Generate output transcripts
                stage_output = _generate_stage_output(content_id, self.word_table, 8, test_mode)
                
                total_time = time.time() - start_time
                result.update({
                    'status': 'success',
                    'stopped_at_stage': 8,
                    'processing_time': total_time,
                    'readable_transcript': stage_output['readable_transcript'],
                    'detailed_transcript': stage_output['detailed_transcript'],
                    'summary': {
                        'words_processed': stage3_result['stats']['words_created'],
                        'segments_processed': stage3_result['stats']['segments_created'],
                        'artifacts_removed': cleaning_stats.get('total_artifacts_removed', 0),
                        'slamdunk_assignments': slamdunk_stats,
                        'bad_grammar_single_assignments': bad_grammar_stats,
                        'good_grammar_multi_analysis': good_grammar_multi_stats,
                        'bad_grammar_multi_analysis': bad_grammar_multi_stats,
                        'stopped_at': 'stage_8_good_grammar_multi_analysis'
                    }
                })
                
                logger.info(f"[{content_id}] Pipeline stopped at stage 8 in {total_time:.2f}s")
                return result
            
            # Stage 9: Bad Grammar + Multi-Speaker Analysis
            logger.info("\n" + "="*80)
            logger.info(f"[{content_id}] STAGE 9: BAD GRAMMAR + MULTI-SPEAKER ANALYSIS")
            logger.info("="*80)
            
            # Check cache or run stage
            stage9_bad_grammar_multi_result = None
            if cache and (start_from_stage is None or start_from_stage > 9):
                stage9_bad_grammar_multi_result = cache.load_stage_result(content_id, 'stage9_bad_grammar_multi')
                if stage9_bad_grammar_multi_result:
                    logger.debug(f"[{content_id}] Using cached stage 9 results")
            
            if stage9_bad_grammar_multi_result is None and (start_from_stage is None or start_from_stage <= 9):
                stage9_bad_grammar_multi_start = time.time()
                stage9_bad_grammar_multi_result = await bad_grammar_multi_speaker_stage(
                    content_id, 
                    self.word_table, 
                    cleaned_diarization_data,
                    s3_storage=self._get_fresh_s3_storage(),
                    test_mode=test_mode,
                    language=main_language,
                    audio_path=audio_path
                )
                stage9_bad_grammar_multi_duration = time.time() - stage9_bad_grammar_multi_start
                
                # Cache successful results
                if cache and stage9_bad_grammar_multi_result['status'] == 'success':
                    cache.save_stage_result(content_id, 'stage9_bad_grammar_multi', stage9_bad_grammar_multi_result)
            else:
                stage9_bad_grammar_multi_duration = stage9_bad_grammar_multi_result['stats'].get('duration', 0) if stage9_bad_grammar_multi_result else 0
            
            # Validate that we have stage 9 results
            if stage9_bad_grammar_multi_result is None:
                raise ValueError(f"Stage 9 results not available. This should not happen with current validation logic.")
            
            if stage9_bad_grammar_multi_result['status'] != 'success':
                raise ValueError(f"Stage 9 failed: {stage9_bad_grammar_multi_result.get('error', 'Unknown error')}")
            
            result['processing_stages']['bad_grammar_multi_analysis'] = stage9_bad_grammar_multi_result['stats']
            result['processing_stages']['bad_grammar_multi_analysis']['duration'] = stage9_bad_grammar_multi_duration
            
            # Update word table from stage 9 results
            # Preserve speaker_db_dictionary through stage 9
            new_word_table = stage9_bad_grammar_multi_result['data']['word_table']
            if hasattr(self.word_table, 'speaker_db_dictionary'):
                new_word_table.speaker_db_dictionary = self.word_table.speaker_db_dictionary
                logger.debug(f"[{content_id}] Preserved speaker_db_dictionary through stage 9")
            self.word_table = new_word_table
            
            bad_grammar_multi_stats = stage9_bad_grammar_multi_result['stats']
            logger.info(f"[{content_id}] Stage 9 completed in {stage9_bad_grammar_multi_duration:.2f}s: analyzed {bad_grammar_multi_stats.get('bad_grammar_multi_segments', 0)} bad grammar + multi-speaker segments")
            
            # Check if we should stop after stage 9
            if stop_at_stage == 9:
                logger.info(f"[{content_id}] Stopping after stage 9 as requested")
                
                # Generate output transcripts
                stage_output = _generate_stage_output(content_id, self.word_table, 9, test_mode)
                
                total_time = time.time() - start_time
                result.update({
                    'status': 'success',
                    'stopped_at_stage': 9,
                    'processing_time': total_time,
                    'readable_transcript': stage_output['readable_transcript'],
                    'detailed_transcript': stage_output['detailed_transcript'],
                    'summary': {
                        'words_processed': stage3_result['stats']['words_created'],
                        'segments_processed': stage3_result['stats']['segments_created'],
                        'artifacts_removed': cleaning_stats.get('total_artifacts_removed', 0),
                        'slamdunk_assignments': slamdunk_stats,
                        'bad_grammar_single_assignments': bad_grammar_stats,
                        'good_grammar_multi_analysis': good_grammar_multi_stats,
                        'bad_grammar_multi_analysis': bad_grammar_multi_stats,
                        'stopped_at': 'stage_9_bad_grammar_multi_analysis'
                    }
                })
                
                logger.info(f"[{content_id}] Pipeline stopped at stage 9 in {total_time:.2f}s")
                return result
            
            # Stage 10: LLM Resolution for Remaining UNKNOWN Words
            logger.info("\n" + "="*80)
            logger.info(f"[{content_id}] STAGE 10: LLM RESOLUTION FOR REMAINING UNKNOWN WORDS")
            logger.info("="*80)
            
            # Check cache or run stage
            stage10_llm_result = None
            if cache and (start_from_stage is None or start_from_stage > 10):
                stage10_llm_result = cache.load_stage_result(content_id, 'stage10_llm_resolution')
                if stage10_llm_result:
                    logger.debug(f"[{content_id}] Using cached stage 10 results")
            
            if stage10_llm_result is None and (start_from_stage is None or start_from_stage <= 10):
                stage10_llm_start = time.time()
                
                # Extract speaker centroids from stage6 results
                speaker_centroids = None
                if stage6_good_grammar_multi_result and 'data' in stage6_good_grammar_multi_result:
                    # Try direct speaker_centroids first (new location)
                    if 'speaker_centroids' in stage6_good_grammar_multi_result['data']:
                        speaker_centroids = stage6_good_grammar_multi_result['data']['speaker_centroids']
                        logger.debug(f"[{content_id}] Found {len(speaker_centroids)} speaker centroids from stage6")
                    # Fall back to assignment_result location (old location)
                    elif 'assignment_result' in stage6_good_grammar_multi_result['data']:
                        assignment_result = stage6_good_grammar_multi_result['data']['assignment_result']
                        if assignment_result and 'speaker_centroids' in assignment_result:
                            speaker_centroids = assignment_result['speaker_centroids']
                            logger.debug(f"[{content_id}] Found {len(speaker_centroids)} speaker centroids from stage6 assignment_result")
                
                stage10_llm_result = await stage10_resolutions(
                    content_id,
                    self.word_table,
                    speaker_centroids,
                    cleaned_diarization_data,
                    s3_storage=self._get_fresh_s3_storage(),
                    test_mode=test_mode,
                    audio_path=audio_path
                )
                stage10_llm_duration = time.time() - stage10_llm_start
                
                # Cache successful results
                if cache and stage10_llm_result['status'] == 'success':
                    cache.save_stage_result(content_id, 'stage10_llm_resolution', stage10_llm_result)
            else:
                stage10_llm_duration = stage10_llm_result['stats'].get('duration', 0) if stage10_llm_result else 0
            
            # Validate that we have stage 10 results
            if stage10_llm_result is None:
                raise ValueError(f"Stage 10 results not available. This should not happen with current validation logic.")
            
            if stage10_llm_result['status'] != 'success':
                raise ValueError(f"Stage 10 failed: {stage10_llm_result.get('error', 'Unknown error')}")
            
            result['processing_stages']['llm_resolution'] = stage10_llm_result['stats']
            result['processing_stages']['llm_resolution']['duration'] = stage10_llm_duration
            
            # Update word table from stage 10 results
            # Preserve speaker_db_dictionary through stage 10
            new_word_table = stage10_llm_result['data']['word_table']
            if hasattr(self.word_table, 'speaker_db_dictionary'):
                new_word_table.speaker_db_dictionary = self.word_table.speaker_db_dictionary
                logger.debug(f"[{content_id}] Preserved speaker_db_dictionary through stage 10")
            self.word_table = new_word_table
            
            llm_resolution_stats = stage10_llm_result['stats']
            logger.info(f"[{content_id}] Stage 10 completed in {stage10_llm_duration:.2f}s: resolved {llm_resolution_stats.get('words_resolved', 0)} UNKNOWN words")
            
            # Check if we should stop after stage 10
            if stop_at_stage == 10:
                logger.info(f"[{content_id}] Stopping after stage 10 as requested")
                
                # Generate output transcripts
                stage_output = _generate_stage_output(content_id, self.word_table, 10, test_mode)
                
                total_time = time.time() - start_time
                result.update({
                    'status': 'success',
                    'stopped_at_stage': 10,
                    'processing_time': total_time,
                    'readable_transcript': stage_output['readable_transcript'],
                    'detailed_transcript': stage_output['detailed_transcript'],
                    'summary': {
                        'words_processed': stage3_result['stats']['words_created'],
                        'segments_processed': stage3_result['stats']['segments_created'],
                        'artifacts_removed': cleaning_stats.get('total_artifacts_removed', 0),
                        'slamdunk_assignments': slamdunk_stats,
                        'bad_grammar_single_assignments': bad_grammar_stats,
                        'good_grammar_multi_analysis': good_grammar_multi_stats,
                        'bad_grammar_multi_analysis': bad_grammar_multi_stats,
                        'llm_resolution': llm_resolution_stats,
                        'stopped_at': 'stage_10_llm_resolution'
                    }
                })
                
                logger.info(f"[{content_id}] Pipeline stopped at stage 10 in {total_time:.2f}s")
                return result
            
            # Stage 11: Final Cleanup
            logger.info("\n" + "="*80)
            logger.info(f"[{content_id}] STAGE 11: FINAL CLEANUP")
            logger.info("="*80)
            stage11_start = time.time()
            
            # Execute stage 11: Final cleanup and grammar enhancement
            stage11_result = stage11_cleanup(
                content_id,
                self.word_table,
                test_mode=test_mode
            )
            stage11_duration = time.time() - stage11_start
            
            if stage11_result['status'] != 'success':
                error_msg = stage11_result.get('error', 'Unknown error')
                logger.error(f"[{content_id}] Stage 11 failed: {error_msg}")
                raise ValueError(f"Stage 11 failed: {error_msg}")
            
            # Get the word_table from stage 11 but preserve speaker_db_dictionary if it exists
            new_word_table = stage11_result['data']['word_table']
            # Preserve speaker_db_dictionary from the original word_table if it exists
            if hasattr(self.word_table, 'speaker_db_dictionary'):
                new_word_table.speaker_db_dictionary = self.word_table.speaker_db_dictionary
                logger.debug(f"[{content_id}] Preserved speaker_db_dictionary with {len(self.word_table.speaker_db_dictionary)} entries through stage 11")
            self.word_table = new_word_table
            result['processing_stages']['final_cleanup'] = stage11_result['stats']
            
            logger.info(f"[{content_id}] Stage 11 completed in {stage11_duration:.2f}s")
            
            # Check if we should stop after stage 11
            if stop_at_stage == 11:
                logger.info(f"[{content_id}] Stopping after stage 11 as requested")
                
                # Generate output transcripts
                stage_output = _generate_stage_output(content_id, self.word_table, 11, test_mode)
                
                total_time = time.time() - start_time
                result.update({
                    'status': 'success',
                    'stopped_at_stage': 11,
                    'processing_time': total_time,
                    'readable_transcript': stage_output['readable_transcript'],
                    'detailed_transcript': stage_output['detailed_transcript'],
                    'summary': {
                        'words_processed': stage3_result['stats']['words_created'],
                        'segments_processed': stage3_result['stats']['segments_created'],
                        'stopped_at': 'stage_11_final_cleanup'
                    }
                })
                
                logger.info(f"[{content_id}] Pipeline stopped at stage 11 in {total_time:.2f}s")
                return result
            
            # Stage 12: Final Output Generation
            logger.info("\n" + "="*80)
            logger.info(f"[{content_id}] STAGE 12: FINAL OUTPUT GENERATION")
            logger.info("="*80)
            stage12_start = time.time()
            
            # Set output directory for test mode
            stage12_output_dir = None
            if test_mode:
                stage12_output_dir = get_project_root() / "tests" / "content" / content_id / "outputs"
            
            # Stage 12: Final Output Generation
            stage12_result = stage12_output(
                content_id,
                self.word_table,
                output_dir=stage12_output_dir,
                test_mode=test_mode
            )
            stage12_duration = time.time() - stage12_start
            
            if stage12_result['status'] != 'success':
                raise ValueError(f"Stage 12 failed: {stage12_result.get('error', 'Unknown error')}")
            
            result['processing_stages']['final_output'] = stage12_result['stats']
            result['processing_stages']['final_output']['duration'] = stage12_duration
            
            # Add transcripts to test output if in test mode
            if test_mode:
                result['processing_stages']['final_output'].update({
                    'readable_transcript': stage12_result['data']['readable_transcript'],
                    'detailed_transcript': stage12_result['data']['detailed_transcript']
                })
            
            logger.info(f"[{content_id}] Stage 12 completed in {stage12_duration:.2f}s: Generated {len(stage12_result['data']['output_files'])} output files")

            # Check if we should stop after stage 12
            if stop_at_stage == 12:
                logger.info(f"[{content_id}] Stopping after stage 12 as requested")

                total_time = time.time() - start_time
                result.update({
                    'status': 'success',
                    'stopped_at_stage': 12,
                    'processing_time': total_time,
                    'readable_transcript': stage12_result['data']['readable_transcript'],
                    'detailed_transcript': stage12_result['data']['detailed_transcript'],
                    'summary': {
                        'words_processed': stage3_result['stats']['words_created'],
                        'segments_processed': stage3_result['stats']['segments_created'],
                        'stopped_at': 'stage_12_output'
                    }
                })

                logger.info(f"[{content_id}] Pipeline stopped at stage 12 in {total_time:.2f}s")
                return result

            # Get sentences from stage12 - these are the atomic units for emotion and segmentation
            sentences = stage12_result['data'].get('sentences', [])
            if not sentences:
                raise ValueError("Stage 12 did not generate sentences - required for pipeline")

            logger.info(f"[{content_id}] Stage 12 generated {len(sentences)} sentences")

            # Stage 13: Emotion Detection - SKIPPED FOR NOW
            # Emotion detection can be run as a separate batch process later
            # Sentences proceed to Stage 14 without emotion data (emotion fields stay NULL)
            logger.info("\n" + "="*80)
            logger.info(f"[{content_id}] STAGE 13: SKIPPED (emotion detection disabled)")
            logger.info("="*80)

            result['processing_stages']['emotion_detection'] = {
                'skipped': True,
                'reason': 'Emotion detection temporarily disabled - can run as batch later',
                'sentences_total': len(sentences),
                'sentences_with_emotion': 0,
                'duration': 0
            }

            # Check if we should stop after stage 13
            if stop_at_stage == 13:
                logger.info(f"[{content_id}] Stopping after stage 13 as requested")

                total_time = time.time() - start_time
                result.update({
                    'status': 'success',
                    'stopped_at_stage': 13,
                    'processing_time': total_time,
                    'readable_transcript': stage12_result['data']['readable_transcript'],
                    'detailed_transcript': stage12_result['data']['detailed_transcript'],
                    'summary': {
                        'sentences_total': len(sentences),
                        'sentences_with_emotion': 0,
                        'stopped_at': 'stage_13_emotion_detection_skipped'
                    }
                })

                logger.info(f"[{content_id}] Pipeline stopped at stage 13 in {total_time:.2f}s")
                return result

            # Stage 14: Semantic Segmentation for Retrieval
            logger.info("\n" + "="*80)
            logger.info(f"[{content_id}] STAGE 14: SEMANTIC SEGMENTATION FOR RETRIEVAL")
            logger.info("="*80)
            stage14_start = time.time()

            logger.info(f"[{content_id}] Using {len(sentences)} sentences for segmentation")

            stage14_result = stage14_segment(
                content_id,
                self.word_table,
                sentences=sentences,
                test_mode=test_mode
            )
            stage14_duration = time.time() - stage14_start

            if stage14_result['status'] != 'success':
                raise ValueError(f"Stage 14 failed: {stage14_result.get('error', 'Unknown error')}")

            result['processing_stages']['semantic_segmentation'] = stage14_result['stats']
            result['processing_stages']['semantic_segmentation']['duration'] = stage14_duration

            # Store segments for database insertion
            embedding_segments = stage14_result['data']['segments']

            logger.info(f"[{content_id}] Stage 14 completed in {stage14_duration:.2f}s: Created {len(embedding_segments)} retrieval-optimized segments")

            # Save word_table to S3 as compressed pickle for debugging and reconstruction
            if not test_mode and self.s3_storage:
                logger.info(f"[{content_id}] Saving word_table to S3 as compressed pickle")
                try:
                    import pickle
                    import gzip
                    import io

                    # Pickle the word_table DataFrame
                    pickle_buffer = io.BytesIO()
                    pickle.dump(self.word_table.df, pickle_buffer, protocol=pickle.HIGHEST_PROTOCOL)
                    pickle_data = pickle_buffer.getvalue()

                    # Compress with gzip
                    compressed_buffer = io.BytesIO()
                    with gzip.GzipFile(fileobj=compressed_buffer, mode='wb', compresslevel=9) as gz:
                        gz.write(pickle_data)
                    compressed_data = compressed_buffer.getvalue()

                    # Upload to S3
                    s3_key = f"content/{content_id}/word_table.pkl.gz"
                    self.s3_storage._client.put_object(
                        Bucket=self.s3_storage.config.bucket_name,
                        Key=s3_key,
                        Body=compressed_data,
                        ContentType='application/x-pickle'
                    )

                    uncompressed_mb = len(pickle_data) / 1024 / 1024
                    compressed_mb = len(compressed_data) / 1024 / 1024
                    compression_ratio = len(pickle_data) / len(compressed_data)

                    logger.info(f"[{content_id}] Saved word_table: {compressed_mb:.2f} MB compressed "
                               f"({uncompressed_mb:.2f} MB uncompressed, {compression_ratio:.1f}x compression)")

                except Exception as e:
                    logger.error(f"[{content_id}] Failed to save word_table to S3: {e}")
                    # Non-fatal - continue processing

            # Save EmbeddingSegment records to database (unless in test mode)
            if not test_mode and embedding_segments:
                logger.info(f"[{content_id}] Saving {len(embedding_segments)} embedding segments to database")

                try:
                    from src.database.models import EmbeddingSegment, Content, ThemeClassification
                    from src.database.session import get_session
                    from src.processing_steps.stitch_steps.stage12_output import get_current_stitch_version
                    from src.processing_steps.stitch_steps.stage14_segment import get_current_segment_version

                    with get_session() as session:
                        # Get content record
                        content_record = session.query(Content).filter_by(content_id=content_id).first()
                        if not content_record:
                            raise ValueError(f"Content {content_id} not found in database")

                        # Delete existing embedding segments in a separate transaction to avoid conflicts
                        existing_count = session.query(EmbeddingSegment).filter_by(content_id=content_record.id).count()
                        if existing_count > 0:
                            # First delete theme_classifications that reference these segments (FK constraint)
                            segment_ids = [s.id for s in session.query(EmbeddingSegment.id).filter_by(content_id=content_record.id).all()]
                            if segment_ids:
                                theme_count = session.query(ThemeClassification).filter(ThemeClassification.segment_id.in_(segment_ids)).count()
                                if theme_count > 0:
                                    logger.info(f"[{content_id}] Deleting {theme_count} theme classifications (FK cascade)")
                                    session.query(ThemeClassification).filter(ThemeClassification.segment_id.in_(segment_ids)).delete(synchronize_session=False)

                            logger.info(f"[{content_id}] Deleting {existing_count} existing embedding segments")
                            session.query(EmbeddingSegment).filter_by(content_id=content_record.id).delete()
                            session.commit()  # Commit delete to release unique constraint
                            logger.debug(f"[{content_id}] Committed delete operation")

                        # Verify deletion completed before proceeding
                        remaining = session.query(EmbeddingSegment).filter_by(content_id=content_record.id).count()
                        if remaining > 0:
                            raise RuntimeError(f"Failed to delete all existing segments - {remaining} remain after delete operation")

                        # Prepare segment dictionaries for bulk insert
                        segment_dicts = []
                        current_timestamp = pd.Timestamp.now(timezone.utc)
                        segments_with_speaker_positions = 0

                        for idx, segment in enumerate(embedding_segments):
                            # Generate segment hash using timing to guarantee uniqueness
                            # (stage13 may produce segments with overlapping text starts)
                            segment_data = f"{content_id}:{idx}:{segment['start_time']:.3f}:{segment['end_time']:.3f}"
                            segment_hash = hashlib.sha256(segment_data.encode()).hexdigest()[:8]

                            # Get speaker hashes from speaker IDs
                            speaker_hashes = None
                            if segment.get('speaker_ids'):
                                from src.database.models import Speaker
                                speaker_hashes = [
                                    session.query(Speaker.speaker_hash).filter_by(id=sid).scalar()
                                    for sid in segment['speaker_ids']
                                ]
                                speaker_hashes = [h for h in speaker_hashes if h]  # Filter None values

                            metadata = {
                                'speaker_ids': segment['speaker_ids'],
                                'speaker_names': segment.get('speaker_names', []),
                                'segment_method': 'beam_search',
                                'similarity_model': 'XLM-R',
                                'timing_method': segment.get('timing_method', 'word_table_precise'),
                                'pipeline_version': get_current_stitch_version(),
                                'embeddings_pending': True,
                                'word_count': segment.get('word_count', 0),
                                'sentence_count': segment.get('sentence_count', 0)
                            }

                            # Track speaker_positions for logging
                            speaker_positions = segment.get('speaker_positions')
                            if speaker_positions:
                                segments_with_speaker_positions += 1

                            segment_dicts.append({
                                'content_id': content_record.id,
                                'segment_index': idx,
                                'text': segment['text'],
                                'start_time': segment['start_time'],
                                'end_time': segment['end_time'],
                                'token_count': segment['token_count'],
                                'segment_type': segment['segment_type'],
                                'source_transcription_ids': segment.get('source_transcription_ids', []),
                                'source_sentence_ids': segment.get('source_sentence_ids', []),
                                'emotion_summary': segment.get('emotion_summary'),
                                'source_start_char': None,
                                'source_end_char': None,
                                'embedding': None,  # Will be populated by hydrate_embeddings.py
                                'embedding_alt': None,
                                'embedding_alt_model': None,
                                'meta_data': metadata,
                                'created_at': current_timestamp,
                                'stitch_version': get_current_stitch_version(),
                                'embedding_version': get_current_segment_version(),
                                'segment_hash': segment_hash,
                                'content_id_string': content_id,
                                'source_speaker_hashes': speaker_hashes,
                                'speaker_positions': speaker_positions  # Add speaker position mapping
                            })

                        # Bulk insert with ON CONFLICT DO UPDATE to handle duplicates
                        if segment_dicts:
                            from sqlalchemy.dialects.postgresql import insert as pg_insert

                            # Use PostgreSQL INSERT ... ON CONFLICT DO UPDATE
                            # This handles race conditions where segments weren't deleted properly
                            stmt = pg_insert(EmbeddingSegment).values(segment_dicts)
                            stmt = stmt.on_conflict_do_update(
                                constraint='uq_embedding_segments_content_hash',
                                set_={
                                    'text': stmt.excluded.text,
                                    'start_time': stmt.excluded.start_time,
                                    'end_time': stmt.excluded.end_time,
                                    'token_count': stmt.excluded.token_count,
                                    'segment_type': stmt.excluded.segment_type,
                                    'source_transcription_ids': stmt.excluded.source_transcription_ids,
                                    'source_sentence_ids': stmt.excluded.source_sentence_ids,
                                    'emotion_summary': stmt.excluded.emotion_summary,
                                    'meta_data': stmt.excluded.meta_data,
                                    'created_at': stmt.excluded.created_at,
                                    'stitch_version': stmt.excluded.stitch_version,
                                    'embedding_version': stmt.excluded.embedding_version,
                                    'source_speaker_hashes': stmt.excluded.source_speaker_hashes,
                                    'speaker_positions': stmt.excluded.speaker_positions
                                }
                            )
                            session.execute(stmt)
                            session.commit()
                            logger.info(f"[{content_id}] Successfully saved {len(segment_dicts)} embedding segments to database (with upsert)")
                            logger.info(f"[{content_id}] speaker_positions populated for {segments_with_speaker_positions}/{len(segment_dicts)} segments")

                            # Update content status and segment version
                            # Note: is_embedded will be set to True once embeddings are hydrated by hydrate_embeddings.py
                            content_record.is_embedded = False  # Embeddings not yet generated

                            # Set segment_version in meta_data to prevent segment task creation
                            # This tells pipeline_manager that segmentation is complete (even though embeddings pending)
                            segment_version = get_current_segment_version()
                            meta_data = dict(content_record.meta_data) if content_record.meta_data else {}
                            meta_data['segment_version'] = segment_version
                            content_record.meta_data = meta_data

                            logger.info(f"[{content_id}] Set segment_version={segment_version} in meta_data (embeddings pending)")
                            session.commit()

                except Exception as e:
                    logger.error(f"[{content_id}] Failed to save embedding segments: {e}", exc_info=True)
                    # This is a critical error - we should not continue
                    raise Exception(f"Failed to save embedding segments: {e}") from e

            elif test_mode and embedding_segments:
                logger.info(f"[{content_id}] TEST MODE: Would save {len(embedding_segments)} embedding segments to database")

            # Save outputs to S3 (unless in test mode)
            if not test_mode:
                logger.info("\n" + "="*80)
                logger.info(f"[{content_id}] SAVING OUTPUTS TO S3")
                logger.info("="*80)
                s3_upload_start = time.time()
                
                try:
                    # Create temp directory for JSON files
                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_dir_path = Path(temp_dir)
                        
                        # Save transcript_diarized.json
                        transcript_key = f"content/{content_id}/transcript_diarized.json"
                        # Get current stitch version from config
                        from src.processing_steps.stitch_steps.stage12_output import get_current_stitch_version
                        transcript_data = {
                            'content_id': content_id,
                            'stitch_version': get_current_stitch_version(),
                            'created_at': pd.Timestamp.now().isoformat(),
                            'speaker_turns': stage12_result['data']['speaker_turns'],
                            'stats': stage12_result['stats']
                        }
                        
                        # Save to temp file
                        transcript_path = temp_dir_path / "transcript_diarized.json"
                        with open(transcript_path, 'w') as f:
                            json.dump(transcript_data, f, indent=2, cls=NumpyJsonEncoder)
                        
                        # Upload to S3 using fresh connection
                        fresh_s3 = self._get_fresh_s3_storage()
                        if fresh_s3.upload_file(str(transcript_path), transcript_key):
                            logger.debug(f"[{content_id}] Uploaded transcript_diarized.json to S3: {transcript_key}")
                        else:
                            raise RuntimeError(f"Failed to upload transcript_diarized.json to S3: {transcript_key}")
                        
                        # Save speaker_centroids.json
                        centroids_key = f"content/{content_id}/speaker_centroids.json"
                        
                        # Get speaker centroids from stage 6 results
                        speaker_centroids_to_save = {}
                        if stage6_good_grammar_multi_result and 'data' in stage6_good_grammar_multi_result:
                            if 'assignment_result' in stage6_good_grammar_multi_result['data']:
                                assignment_result = stage6_good_grammar_multi_result['data']['assignment_result']
                                # Get the speaker_centroid_data which contains full metadata
                                if assignment_result and 'speaker_centroid_data' in assignment_result:
                                    speaker_centroids_to_save = assignment_result['speaker_centroid_data']
                                elif assignment_result and 'speaker_centroids' in assignment_result:
                                    # Fallback to speaker_centroids if speaker_centroid_data not available
                                    speaker_centroids_to_save = assignment_result['speaker_centroids']
                        
                        # Convert numpy arrays to lists for JSON serialization
                        serializable_centroids = {}
                        for speaker_id, centroid_info in speaker_centroids_to_save.items():
                            if isinstance(centroid_info, dict):
                                serializable_centroid = {}
                                for key, value in centroid_info.items():
                                    if isinstance(value, np.ndarray):
                                        serializable_centroid[key] = value.tolist()
                                    elif isinstance(value, (np.float32, np.float64, np.int32, np.int64)):
                                        # Convert numpy scalars to Python types
                                        serializable_centroid[key] = float(value) if 'float' in str(type(value)) else int(value)
                                    else:
                                        serializable_centroid[key] = value
                                serializable_centroids[speaker_id] = serializable_centroid
                            else:
                                # If it's not a dict, try to convert directly
                                if hasattr(centroid_info, '__dict__'):
                                    # It's an object, convert to dict
                                    obj_dict = {}
                                    for attr in dir(centroid_info):
                                        if not attr.startswith('_'):
                                            val = getattr(centroid_info, attr)
                                            if isinstance(val, np.ndarray):
                                                obj_dict[attr] = val.tolist()
                                            elif isinstance(val, (np.float32, np.float64, np.int32, np.int64)):
                                                # Convert numpy scalars to Python types
                                                obj_dict[attr] = float(val) if 'float' in str(type(val)) else int(val)
                                            elif not callable(val):
                                                obj_dict[attr] = val
                                    serializable_centroids[speaker_id] = obj_dict
                                else:
                                    serializable_centroids[speaker_id] = centroid_info
                        
                        centroids_data = {
                            'content_id': content_id,
                            'created_at': pd.Timestamp.now().isoformat(),
                            'speaker_centroids': serializable_centroids
                        }
                        
                        # Save speaker centroids from stage 6
                        
                        # Save to temp file
                        centroids_path = temp_dir_path / "speaker_centroids.json"
                        with open(centroids_path, 'w') as f:
                            json.dump(centroids_data, f, indent=2, cls=NumpyJsonEncoder)
                        
                        # Upload to S3 using fresh connection
                        fresh_s3 = self._get_fresh_s3_storage()
                        if fresh_s3.upload_file(str(centroids_path), centroids_key):
                            logger.debug(f"[{content_id}] Uploaded speaker_centroids.json to S3: {centroids_key}")
                        else:
                            raise RuntimeError(f"Failed to upload speaker_centroids.json to S3: {centroids_key}")
                        
                        s3_upload_duration = time.time() - s3_upload_start
                        logger.info(f"[{content_id}] S3 upload completed in {s3_upload_duration:.2f}s")
                        result['processing_stages']['s3_upload'] = {'duration': s3_upload_duration}
                    
                except Exception as e:
                    s3_upload_duration = time.time() - s3_upload_start
                    logger.error(f"[{content_id}] Failed to upload outputs to S3: {e}")
                    logger.error(f"[{content_id}] Error details:", exc_info=True)
                    # S3 upload failures are critical - raise the exception
                    raise RuntimeError(f"Failed to upload outputs to S3: {e}") from e
            
            # Save outputs in test mode
            if test_mode:
                test_output_start = time.time()
                logger.info("\n" + "="*80)
                logger.info(f"[{content_id}] SAVING TEST OUTPUTS")
                logger.info("="*80)
                
                test_output_dir = get_project_root() / "tests" / "content" / content_id / "outputs"
                test_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Save word table
                word_table_path = test_output_dir / "word_table.json"
                self.word_table.save_to_file(word_table_path)
                
                # Save segment table
                segment_table_path = test_output_dir / "segment_table.json"
                self.segment_table.save_to_file(segment_table_path)
                
                # Save speaker assignment results if available
                assignment_results_path = None
                assignment_summary = {}  # Default empty summary for now
                if assignment_summary and assignment_summary.get('status') != 'skipped':
                    assignment_results_path = test_output_dir / "speaker_assignment_results.json"
                    with open(assignment_results_path, 'w') as f:
                        json.dump(assignment_summary, f, indent=2, cls=NumpyJsonEncoder)
                
                # Save processing metadata
                metadata = {
                    'content_id': content_id,
                    'processing_time': time.time() - start_time,
                    'time_range': time_range,
                    'stage_results': result['processing_stages'],
                    'speaker_assignment_summary': assignment_summary
                }
                
                metadata_path = test_output_dir / "processing_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2, cls=NumpyJsonEncoder)
                
                test_output_duration = time.time() - test_output_start
                result['processing_stages']['test_output_saving'] = {
                    'duration': test_output_duration,
                    'output_files': {
                        'word_table': str(word_table_path),
                        'segment_table': str(segment_table_path),
                        'metadata': str(metadata_path),
                        'speaker_assignment_results': str(assignment_results_path) if assignment_results_path else None
                    }
                }
                
                logger.debug(f"[{content_id}] Test outputs saved in {test_output_duration:.2f}s to: {test_output_dir}")
            
            # Final result
            total_time = time.time() - start_time
            result.update({
                'status': 'success',
                'processing_time': total_time,
                'summary': {
                    'words_processed': stage3_result['stats']['words_created'],
                    'segments_processed': stage3_result['stats']['segments_created'],
                    'artifacts_removed': cleaning_stats.get('total_artifacts_removed', 0),
                    'slamdunk_assignments': slamdunk_stats.get('slam_dunk_assignments', 0),
                    'bad_grammar_single_assignments': bad_grammar_stats.get('bad_grammar_single_assignments', 0),
                    'good_grammar_multi_analyzed': good_grammar_multi_stats.get('good_grammar_multi_segments', 0),
                    'bad_grammar_multi_analyzed': bad_grammar_multi_stats.get('bad_grammar_multi_segments', 0)
                }
            })
            
            logger.info(f"[{content_id}] Pipeline completed successfully in {total_time:.2f}s")

            # NOTE: EmbeddingSegment cleanup is now handled by Stage 13 before insertion
            # Old cleanup block removed (was deleting segments that Stage 13 just created)

            # Clean up temporary audio files in production mode
            if not test_mode:
                try:
                    persistent_temp_dir = Path("/tmp/stitch_audio") / content_id
                    if persistent_temp_dir.exists():
                        shutil.rmtree(persistent_temp_dir)
                        logger.debug(f"[{content_id}] Cleaned up temporary audio directory: {persistent_temp_dir}")
                except Exception as cleanup_e:
                    logger.warning(f"[{content_id}] Failed to clean up temp audio directory: {cleanup_e}")
            
            return result
            
        except Exception as e:
            logger.error(f"[{content_id}] Pipeline failed: {str(e)}")
            logger.error(f"[{content_id}] Error details:", exc_info=True)
            
            # Clean up temporary audio files even on error (in production mode)
            if not test_mode:
                try:
                    persistent_temp_dir = Path("/tmp/stitch_audio") / content_id
                    if persistent_temp_dir.exists():
                        shutil.rmtree(persistent_temp_dir)
                        logger.debug(f"[{content_id}] Cleaned up temporary audio directory after error: {persistent_temp_dir}")
                except Exception as cleanup_e:
                    logger.warning(f"[{content_id}] Failed to clean up temp audio directory: {cleanup_e}")
            
            result.update({
                'status': 'error',
                'error': str(e),
                'processing_time': time.time() - start_time
            })
            return result


async def main():
    """Main entry point for the stitch pipeline."""
    parser = argparse.ArgumentParser(description='Clean Stitch Pipeline - Word-level speaker attribution')
    parser.add_argument('--content', required=True, help='Content ID to process')
    parser.add_argument('--test', action='store_true', help='Test mode: save outputs locally for debugging')
    parser.add_argument('--start', type=float, help='Start time in seconds for focused processing')
    parser.add_argument('--end', type=float, help='End time in seconds for focused processing')
    parser.add_argument('--stages', type=int, help='Stop after this stage number (1-13) and output results')
    parser.add_argument('--rewrite', action='store_true', help='Overwrite existing outputs in S3 and database')
    parser.add_argument('--start-from-stage', type=int, help='Start from this stage number using cached results from previous stages')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching for this run')
    parser.add_argument('--clear-cache', action='store_true', help='Clear all cached stages for this content before running')
    
    args = parser.parse_args()
    
    # Parse time range if provided
    time_range = None
    if args.start is not None and args.end is not None:
        time_range = (args.start, args.end)
        logger.info(f"Processing time range: {args.start:.1f}s - {args.end:.1f}s")
    elif args.start is not None or args.end is not None:
        logger.warning("Both --start and --end must be provided for time range filtering")
    
    # Validate stages argument
    if args.stages is not None and (args.stages < 1 or args.stages > 13):
        logger.error("--stages must be between 1 and 13")
        print("Error: --stages must be between 1 and 13")
        sys.exit(1)
    
    try:
        # Initialize and run pipeline
        pipeline = StitchPipeline()
        
        result = await pipeline.process_content(
            content_id=args.content,
            test_mode=args.test,
            time_range=time_range,
            stop_at_stage=args.stages,
            start_from_stage=args.start_from_stage,
            overwrite=args.rewrite,
            use_cache=not args.no_cache,
            clear_cache=args.clear_cache
        )
        
        # Output result as JSON for task processor
        if result['status'] == 'success':
            # For success, use the standardized result format
            success_result = create_success_result(
                data={
                    "content_id": args.content,
                    "processing_time": result.get('processing_time', 0),
                    "summary": result.get('summary', {}),
                    "stopped_at_stage": result.get('stopped_at_stage')
                }
            )
            print(json.dumps(success_result, cls=NumpyJsonEncoder))
            
            # If test mode and stopped at stage, also print transcripts for debugging
            if args.test and 'stopped_at_stage' in result:
                logger.info("\n" + "="*80)
                logger.info("READABLE TRANSCRIPT:")
                logger.info("="*80)
                logger.info(result['readable_transcript'])
                logger.info("\n" + "="*80)
                logger.info("DETAILED TRANSCRIPT:")
                logger.info("="*80)
                logger.info(result['detailed_transcript'])
            sys.exit(0)
        elif result['status'] == 'skipped':
            # Output skip using standardized format
            skip_result = create_skipped_result(
                reason=result.get('reason', 'Unknown reason'),
                data={
                    "content_id": args.content,
                    "existing_outputs": result.get('existing_outputs', {})
                }
            )
            print(json.dumps(skip_result, cls=NumpyJsonEncoder))
            sys.exit(0)  # Exit with success for skipped
        else:
            # Output error using standardized format
            error_msg = result.get('error', 'Unknown error')
            
            # Map common error messages to error codes
            if "No audio file found" in error_msg or "Audio file not found" in error_msg or "Failed to download audio file" in error_msg:
                error_code = ErrorCode.MISSING_AUDIO
            elif "No diarization data found" in error_msg:
                error_code = ErrorCode.MISSING_DIARIZATION
            elif "music-only" in error_msg or "no detectable speech" in error_msg or "No speaker segments in diarization" in error_msg or "Empty diarization segments" in error_msg or "Empty word table" in error_msg:
                error_code = ErrorCode.NO_SPEECH_DETECTED
            elif "No transcript data found" in error_msg:
                error_code = ErrorCode.MISSING_TRANSCRIPT
            elif "deepmultilingual" in error_msg.lower() or "model loading failed" in error_msg.lower() or "parakeet" in error_msg.lower() or "mlx_whisper" in error_msg.lower():
                error_code = ErrorCode.PROCESS_FAILED
            elif "Failed to upload" in error_msg and "S3" in error_msg:
                error_code = ErrorCode.S3_CONNECTION_ERROR
            else:
                error_code = ErrorCode.UNKNOWN_ERROR

            # NO_SPEECH_DETECTED is a permanent error - content will never have speech
            is_permanent = error_code == ErrorCode.NO_SPEECH_DETECTED

            error_result = create_error_result(
                error_code=error_code,
                error_message=error_msg,
                error_details={
                    "content_id": args.content,
                    "stage": result.get('stage', 'unknown')
                },
                permanent=is_permanent
            )
            print(json.dumps(error_result, cls=NumpyJsonEncoder))
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Unhandled error in main: {str(e)}")
        # Output error using standardized format
        error_msg = str(e)
        
        # Map common error messages to error codes
        if "No audio file found" in error_msg or "Audio file not found" in error_msg or "Failed to download audio file" in error_msg:
            error_code = ErrorCode.MISSING_AUDIO
        elif "No diarization data found" in error_msg:
            error_code = ErrorCode.MISSING_DIARIZATION
        elif "music-only" in error_msg or "no detectable speech" in error_msg or "No speaker segments in diarization" in error_msg or "Empty diarization segments" in error_msg or "Empty word table" in error_msg:
            error_code = ErrorCode.NO_SPEECH_DETECTED
        elif "No transcript data found" in error_msg:
            error_code = ErrorCode.MISSING_TRANSCRIPT
        elif "deepmultilingual" in error_msg.lower() or "model loading failed" in error_msg.lower() or "parakeet" in error_msg.lower() or "mlx_whisper" in error_msg.lower():
            error_code = ErrorCode.PROCESS_FAILED
        elif "Failed to upload" in error_msg and "S3" in error_msg:
            error_code = ErrorCode.S3_CONNECTION_ERROR
        else:
            error_code = ErrorCode.UNKNOWN_ERROR

        # NO_SPEECH_DETECTED is a permanent error - content will never have speech
        is_permanent = error_code == ErrorCode.NO_SPEECH_DETECTED

        error_result = create_error_result(
            error_code=error_code,
            error_message=error_msg,
            error_details={
                "content_id": args.content,
                "stage": "main"
            },
            permanent=is_permanent
        )
        print(json.dumps(error_result))
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())