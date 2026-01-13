#!/usr/bin/env python3
"""
Wav2Vec2 Utility for Word-Level Alignment
==========================================

This utility provides Wav2Vec2 forced alignment for all words that haven't been
assigned to actual speakers (SPEAKER_XX) in the stitch pipeline. It's called as 
a helper function (not a numbered stage) to improve word-level timing accuracy.

Key features:
- Processes words with category assignments (BAD_GRAMMAR_SINGLE, GOOD_GRAMMAR_MULTI, etc.)
- Processes UNKNOWN and NEEDS_EMBEDDING words
- Preserves existing SPEAKER_XX assignments (doesn't re-align already assigned words)
- Updates word-level timings using Wav2Vec2 CTC alignment
- Adjusts segment boundaries based on updated word timings
"""

import logging
import time
import torch
import numpy as np
import pandas as pd
# aiohttp import removed - using model client instead
import asyncio
import tempfile
import yaml
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from src.utils.paths import get_project_root, get_config_path

from src.utils.logger import setup_worker_logger
from src.utils.config import load_config
from src.processing_steps.stitch_steps.stage3_tables import WordTable
from src.processing_steps.stitch_steps.archive.stage4_wav2vec2 import OptimizedForcedAlignmentProcessor
# Model server imports removed - using direct model loading only

logger = setup_worker_logger('stitch')


def _load_config() -> Dict[str, Any]:
    """Load configuration from config.yaml."""
    config_path = get_config_path()
    try:
        with open(config_path) as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.warning(f"Failed to load config: {e}, using defaults")
        return {}


async def _try_model_server_wav2vec2(audio_path: Path, word_table_data: List[Dict], 
                                    sample_rate: int, config: Dict[str, Any]) -> Optional[List]:
    """Model server deprecated - returns None to use local processing."""
    logger.debug("Model server deprecated, using local processing")
    return None  # Always use local processing


def run_wav2vec2_on_unassigned_words(content_id: str,
                                   word_table: WordTable,
                                   audio_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Run Wav2Vec2 forced alignment on all words not assigned to actual speakers.
    
    This utility function:
    1. Identifies all words that are not assigned to SPEAKER_XX (includes categories like
       BAD_GRAMMAR_SINGLE, GOOD_GRAMMAR_MULTI, BAD_GRAMMAR_MULTI, NEEDS_EMBEDDING, UNKNOWN)
    2. Runs Wav2Vec2 forced alignment on those words (via model server with local fallback)
    3. Updates word-level start/end times
    4. Adjusts segment boundaries based on new word timings
    
    Args:
        content_id: Content ID to process
        word_table: WordTable with current assignments
        audio_path: Path to audio file for alignment
        
    Returns:
        Dictionary with alignment results and statistics
    """
    start_time = time.time()
    
    logger.info(f"[{content_id}] Running Wav2Vec2 alignment utility on words not assigned to speakers")
    
    # Load configuration
    config = _load_config()
    
    result = {
        'status': 'success',
        'content_id': content_id,
        'utility': 'wav2vec2_unassigned',
        'stats': {
            'total_words': 0,
            'unassigned_words': 0,
            'words_processed': 0,
            'words_improved': 0,
            'segments_updated': 0,
            'average_confidence': 0.0,
            'timing_improvements': 0.0
        },
        'error': None
    }
    
    try:
        if not audio_path or not audio_path.exists():
            logger.warning(f"[{content_id}] No audio file available for forced alignment")
            result['stats']['status'] = 'no_audio'
            return result
        
        # Check if speaker_current column exists
        if 'speaker_current' not in word_table.df.columns:
            logger.warning(f"[{content_id}] No speaker_current column found - initializing all as UNKNOWN")
            word_table.df['speaker_current'] = 'UNKNOWN'
        
        # Get statistics on current assignments
        total_words = len(word_table.df)
        assignment_counts = word_table.df['speaker_current'].value_counts()
        
        # Count words that need alignment (not assigned to SPEAKER_XX)
        speaker_assigned_mask = word_table.df['speaker_current'].str.match(r'^SPEAKER_\d{2}$', na=False)
        speaker_assigned_count = speaker_assigned_mask.sum()
        needs_alignment_count = total_words - speaker_assigned_count
        
        logger.info(f"[{content_id}] Word assignment status:")
        for speaker, count in assignment_counts.items():
            logger.info(f"  {speaker}: {count} words")
        
        result['stats']['total_words'] = total_words
        result['stats']['unassigned_words'] = needs_alignment_count
        
        # Filter for words that need alignment (not assigned to actual speakers)
        # This includes categories and UNKNOWN words
        needs_alignment_mask = ~word_table.df['speaker_current'].str.match(r'^SPEAKER_\d{2}$', na=False)
        unassigned_word_indices = word_table.df[needs_alignment_mask].index.tolist()
        
        if len(unassigned_word_indices) == 0:
            logger.info(f"[{content_id}] All words are assigned to speakers - skipping alignment")
            result['stats']['status'] = 'all_words_assigned'
            return result
        
        logger.info(f"[{content_id}] Processing {needs_alignment_count} words that need alignment (not assigned to SPEAKER_XX)")
        
        # Prepare word data for alignment
        word_table_data = []
        for idx in unassigned_word_indices:
            row = word_table.df.loc[idx]
            word_table_data.append({
                'text': row['text'],
                'start': row['start'],
                'end': row['end'],
                'index': idx,
                'segment_index': row['segment_index']
            })
        
        # Load audio
        try:
            import torchaudio
            audio_waveform, sample_rate = torchaudio.load(audio_path)
            if audio_waveform.dim() > 1:
                audio_waveform = audio_waveform[0]  # Take first channel
            audio_waveform = audio_waveform.squeeze()
            logger.info(f"[{content_id}] Loaded audio: {len(audio_waveform)} samples at {sample_rate}Hz")
        except Exception as e:
            logger.error(f"[{content_id}] Failed to load audio: {e}")
            result['stats']['status'] = 'audio_load_failed'
            return result
        
        # Try model server first, then fallback to local processing
        logger.info(f"[{content_id}] Starting Wav2Vec2 forced alignment on {len(word_table_data)} words")
        alignment_start = time.time()
        alignment_results = None
        alignment_method = "unknown"
        
        # Attempt model server processing first
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, create a new event loop
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run, 
                        _try_model_server_wav2vec2(audio_path, word_table_data, sample_rate, config)
                    )
                    model_server_results = future.result()
            else:
                model_server_results = loop.run_until_complete(
                    _try_model_server_wav2vec2(audio_path, word_table_data, sample_rate, config)
                )
            
            if model_server_results:
                alignment_results = model_server_results
                alignment_method = "model_server"
                logger.info(f"[{content_id}] Using model server Wav2Vec2 alignment results")
        except Exception as e:
            logger.warning(f"[{content_id}] Model server attempt failed: {e}")
        
        # Fallback to local processing if model server failed
        if alignment_results is None:
            logger.info(f"[{content_id}] Falling back to local Wav2Vec2 processing")
            processor = OptimizedForcedAlignmentProcessor(device='auto', batch_size=64)
            alignment_results = processor.forced_align(audio_waveform.numpy(), word_table_data, sample_rate)
            alignment_method = "local_processor"
            # Clear cache to free memory
            processor.clear_cache()
        
        alignment_duration = time.time() - alignment_start
        
        logger.info(f"[{content_id}] Alignment completed using {alignment_method} in {alignment_duration:.3f}s ({len(word_table_data)/alignment_duration:.0f} words/second)")
        
        # Apply alignment results to word table
        words_improved = 0
        total_confidence = 0.0
        total_timing_improvement = 0.0
        updated_segments = set()
        
        for alignment_result in alignment_results:
            # Find corresponding word in table
            word_mask = (word_table.df['text'] == alignment_result.word) & \
                       (abs(word_table.df['start'] - alignment_result.original_start) < 0.1)
            
            if word_mask.any():
                idx = word_mask.idxmax()
                
                # Calculate timing improvement
                original_duration = alignment_result.original_end - alignment_result.original_start
                aligned_duration = alignment_result.aligned_end - alignment_result.aligned_start
                timing_improvement = abs(aligned_duration - original_duration)
                
                # Apply quality thresholds
                confidence_threshold = 0.3
                max_timing_change = 0.5
                
                should_apply_alignment = (
                    alignment_result.confidence > confidence_threshold and
                    timing_improvement < max_timing_change and
                    aligned_duration > 0.01
                )
                
                if should_apply_alignment:
                    # Update word table with refined timings
                    word_table.df.at[idx, 'start'] = alignment_result.aligned_start
                    word_table.df.at[idx, 'end'] = alignment_result.aligned_end
                    words_improved += 1
                    total_confidence += alignment_result.confidence
                    total_timing_improvement += timing_improvement
                    
                    # Track which segments were updated
                    segment_index = word_table.df.at[idx, 'segment_index']
                    updated_segments.add(segment_index)
                
                # Add alignment metadata
                if 'metadata' not in word_table.df.columns:
                    word_table.df['metadata'] = [{}] * len(word_table.df)
                
                metadata = word_table.df.at[idx, 'metadata']
                if not isinstance(metadata, dict):
                    metadata = {}
                
                metadata['wav2vec2_alignment'] = {
                    'original_start': alignment_result.original_start,
                    'original_end': alignment_result.original_end,
                    'confidence': alignment_result.confidence,
                    'timing_improvement': timing_improvement,
                    'applied': should_apply_alignment,
                    'utility_stage': True
                }
                
                word_table.df.at[idx, 'metadata'] = metadata
        
        # Update segment boundaries based on new word timings
        segments_updated = _update_segment_boundaries(word_table, updated_segments, content_id)
        
        # Calculate final statistics
        avg_confidence = total_confidence / max(1, words_improved)
        improvement_rate = (words_improved / len(word_table_data)) * 100 if word_table_data else 0
        
        result['stats'].update({
            'duration': time.time() - start_time,
            'status': 'wav2vec2_alignment_completed',
            'alignment_method': alignment_method,
            'words_processed': len(word_table_data),
            'words_improved': words_improved,
            'segments_updated': segments_updated,
            'average_confidence': avg_confidence,
            'timing_improvements': total_timing_improvement,
            'improvement_rate': improvement_rate,
            'alignment_duration': alignment_duration
        })
        
        logger.info(f"[{content_id}] Wav2Vec2 alignment utility completed:")
        logger.info(f"[{content_id}] - Words processed: {len(word_table_data)}")
        logger.info(f"[{content_id}] - Words improved: {words_improved} ({improvement_rate:.1f}%)")
        logger.info(f"[{content_id}] - Segments updated: {segments_updated}")
        logger.info(f"[{content_id}] - Average confidence: {avg_confidence:.3f}")
        logger.info(f"[{content_id}] - Total timing improvement: {total_timing_improvement:.3f}s")
        
        return result
        
    except Exception as e:
        error_msg = str(e) if e else "Unknown error in Wav2Vec2 alignment utility"
        logger.error(f"[{content_id}] Wav2Vec2 alignment utility failed: {error_msg}")
        logger.error(f"[{content_id}] Error details:", exc_info=True)
        
        result.update({
            'status': 'error',
            'error': error_msg,
            'stats': {
                'duration': time.time() - start_time,
                'status': 'alignment_failed'
            }
        })
        
        return result


def _update_segment_boundaries(word_table: WordTable, updated_segments: set, content_id: str) -> int:
    """
    Update segment start/end boundaries based on updated word timings.
    
    Args:
        word_table: WordTable with updated word timings
        updated_segments: Set of segment indices that had word timing updates
        content_id: Content ID for logging
        
    Returns:
        Number of segments that were updated
    """
    if not updated_segments:
        return 0
    
    logger.info(f"[{content_id}] Updating boundaries for {len(updated_segments)} segments")
    
    segments_updated = 0
    
    for segment_index in updated_segments:
        # Get all words in this segment
        segment_words = word_table.df[word_table.df['segment_index'] == segment_index]
        
        if len(segment_words) > 0:
            # Calculate new segment boundaries
            new_start = segment_words['start'].min()
            new_end = segment_words['end'].max()
            
            # Update segment-level columns if they exist
            if 'segment_start' in word_table.df.columns:
                word_table.df.loc[word_table.df['segment_index'] == segment_index, 'segment_start'] = new_start
            
            if 'segment_end' in word_table.df.columns:
                word_table.df.loc[word_table.df['segment_index'] == segment_index, 'segment_end'] = new_end
            
            segments_updated += 1
    
    logger.info(f"[{content_id}] Updated boundaries for {segments_updated} segments")
    
    return segments_updated