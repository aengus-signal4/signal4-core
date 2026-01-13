#!/usr/bin/env python3
"""
Stage 5: Bad Grammar + Single Speaker Assignment (SAFE segments only)
====================================================================

Fifth stage of the stitch pipeline that handles poorly punctuated single-speaker segments.

Key Responsibilities:
- Process all words categorized as BAD_GRAMMAR_SINGLE by Stage 3
- Find dominant speaker (>50% overlap) for each segment using vectorized operations
- Assign all words in segment to the dominant speaker
- Mark segments without dominant speaker as NEEDS_EMBEDDING
- Only processes SAFE segments (not within 1.0s of non-majority speaker)

Input:
- WordTable from Stage 4 with categorized words and some assignments
- Diarization data with speaker segments

Output:
- WordTable with speaker assignments for BAD_GRAMMAR_SINGLE words
- Words either assigned to specific speakers or marked as NEEDS_EMBEDDING
- Assignment statistics including segments processed and confidence scores

Key Concepts:
- SAFE segments: Not within 1.0s of a different speaker (determined by Stage 3)
- Dominant speaker: Speaker with >50% overlap with the segment
- Vectorized processing: Uses numpy broadcasting for efficient overlap calculation

Methods:
- bad_grammar_single_assignment_stage(): Main entry point called by stitch pipeline

Performance:
- Fully vectorized using numpy arrays for overlap calculation
- Processes all segments in a single batch operation
- Efficient pandas updates for word assignments
"""

import logging
import time
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

from src.utils.logger import setup_worker_logger
from src.processing_steps.stitch_steps.stage3_tables import WordTable
from src.processing_steps.stitch_steps.util_stitch import (
    load_stage_config,
    update_processing_status,
    format_stage_stats,
    summarize_speaker_assignments)

logger = setup_worker_logger('stitch')


def bad_grammar_single_assignment_stage(content_id: str,
                                       word_table: WordTable,
                                       diarization_data: Dict,
                                       test_mode: bool = False) -> Dict[str, Any]:
    """
    Main entry point for Stage 5: Bad Grammar + Single Speaker Assignment.
    
    This is the primary method called by the stitch pipeline. It processes all
    BAD_GRAMMAR_SINGLE words identified by Stage 3, assigning them to speakers
    based on dominant speaker analysis (>50% overlap). Only processes SAFE segments
    that are not within 1.0s of a different speaker.
    
    Strategy:
    1. Find segments with BAD_GRAMMAR_SINGLE category (already determined safe by Stage 3)
    2. Calculate dominant speaker for each segment using vectorized overlap analysis
    3. Assign all words in segment to dominant speaker
    4. Mark segments without dominant speaker as NEEDS_EMBEDDING
    
    Args:
        content_id: Content ID being processed (e.g., "Bdb001")
        word_table: WordTable from Stage 4 with categorized words and some assignments
        diarization_data: Dictionary with 'segments' list containing speaker segments
        test_mode: If True, saves detailed outputs for debugging (not currently used)
        
    Returns:
        Dictionary containing:
            - status: 'success' or 'error'
            - data: Dict with updated word_table
            - stats: Assignment statistics and performance metrics
            - error: Error message if status is 'error'
            
    Example:
        result = bad_grammar_single_assignment_stage("Bdb001", word_table, diarization_data)
        if result['status'] == 'success':
            print(f"Assigned {result['stats']['words_assigned']} words")
    """
    stage_start_time = time.time()
    
    logger.info(f"[{content_id}] Starting Stage 5: Bad Grammar + Single Speaker Assignment (SAFE only)")
    
    try:
        # Convert diarization data to DataFrame for vectorized operations
        diarization_segments = diarization_data.get('segments', [])
        if not diarization_segments:
            logger.warning(f"[{content_id}] No diarization segments found")
            return {
                'status': 'success',
                'data': {'word_table': word_table},
                'stats': {
                    'segments_processed': 0,
                    'segments_analyzed': 0,
                    'bad_grammar_single_assignments': 0,
                    'words_assigned': 0,
                    'assignment_rate': 0.0,
                    'duration': time.time() - stage_start_time
                }
            }
        
        diarization_df = pd.DataFrame(diarization_segments)
        diarization_df = diarization_df.sort_values('start').reset_index(drop=True)
        
        # Get segments from word table with Stage 3 analysis results
        segments_df = word_table.df.groupby('segment_index').agg({
            'text': lambda x: ' '.join(x),
            'start': 'min',
            'end': 'max',
            'speaker_current': 'first',
            'segment_has_good_grammar': 'first',
            'segment_has_multiple_speakers': 'first',
            'segment_within_1s_of_non_majority_speaker': 'first'
        }).reset_index()
        
        # Add word count
        word_counts = word_table.df.groupby('segment_index').size().reset_index(name='word_count')
        segments_df = segments_df.merge(word_counts, on='segment_index', how='left')
        
        logger.debug(f"[{content_id}] Processing")
        
        # Get segments with BAD_GRAMMAR_SINGLE category (these are already safe)
        bad_grammar_single_segments = segments_df[
            segments_df['speaker_current'] == 'BAD_GRAMMAR_SINGLE'
        ].copy()
        
        logger.info(f"[{content_id}] Found {len(bad_grammar_single_segments)} BAD_GRAMMAR_SINGLE segments (all safe)")
        
        if len(bad_grammar_single_segments) == 0:
            logger.info(f"[{content_id}] No bad grammar + single speaker segments to process")
            return {
                'status': 'success',
                'data': {'word_table': word_table},
                'stats': {
                    'segments_processed': 0,
                    'segments_analyzed': 0,
                    'bad_grammar_single_assignments': 0,
                    'words_assigned': 0,
                    'assignment_rate': 0.0,
                    'duration': time.time() - stage_start_time
                }
            }
        
        # Verify all segments are safe (they should be based on Stage 3 logic)
        unsafe_count = bad_grammar_single_segments['segment_within_1s_of_non_majority_speaker'].sum()
        if unsafe_count > 0:
            logger.warning(f"[{content_id}] Found {unsafe_count} unsafe segments with BAD_GRAMMAR_SINGLE category - this shouldn't happen!")
        
        # Vectorized processing: Find dominant speakers for all segments at once
        logger.info(f"[{content_id}] Starting vectorized speaker assignment...")
        vectorization_start = time.time()
        
        # Convert segment data to numpy arrays for vectorized operations
        n_segments = len(bad_grammar_single_segments)
        n_diarization = len(diarization_df)
        
        if n_segments == 0 or n_diarization == 0:
            logger.info(f"[{content_id}] No segments or diarization data to process")
            segment_assignments = []
        else:
            # Prepare arrays for vectorized computation
            seg_starts = bad_grammar_single_segments['start'].values[:, np.newaxis]  # (n_segments, 1)
            seg_ends = bad_grammar_single_segments['end'].values[:, np.newaxis]      # (n_segments, 1)
            dia_starts = diarization_df['start'].values[np.newaxis, :]              # (1, n_diarization)
            dia_ends = diarization_df['end'].values[np.newaxis, :]                  # (1, n_diarization)
            dia_speakers = diarization_df['speaker'].values                          # (n_diarization,)
            
            # Calculate overlaps between all segments and all diarization segments
            overlap_starts = np.maximum(seg_starts, dia_starts)  # (n_segments, n_diarization)
            overlap_ends = np.minimum(seg_ends, dia_ends)        # (n_segments, n_diarization)
            overlap_durations = np.maximum(0, overlap_ends - overlap_starts)  # (n_segments, n_diarization)
            
            # Calculate segment durations
            seg_durations = (seg_ends - seg_starts).flatten()  # (n_segments,)
            
            # Get unique speakers
            unique_speakers = np.unique(dia_speakers)
            n_speakers = len(unique_speakers)
            speaker_to_idx = {speaker: idx for idx, speaker in enumerate(unique_speakers)}
            
            # Create speaker overlap matrix (n_segments, n_speakers)
            speaker_overlaps = np.zeros((n_segments, n_speakers))
            
            # Aggregate overlaps by speaker
            for dia_idx in range(n_diarization):
                speaker = dia_speakers[dia_idx]
                speaker_idx = speaker_to_idx[speaker]
                speaker_overlaps[:, speaker_idx] += overlap_durations[:, dia_idx]
            
            # Calculate overlap percentages
            with np.errstate(divide='ignore', invalid='ignore'):
                overlap_percentages = speaker_overlaps / seg_durations[:, np.newaxis]
                overlap_percentages = np.nan_to_num(overlap_percentages, 0.0)
            
            # Find dominant speakers (>50% overlap)
            max_overlap_pcts = np.max(overlap_percentages, axis=1)
            dominant_speaker_indices = np.argmax(overlap_percentages, axis=1)
            has_dominant = max_overlap_pcts > 0.5
            
            # Create assignment list
            segment_assignments = []
            
            for seg_idx in range(n_segments):
                segment_row = bad_grammar_single_segments.iloc[seg_idx]
                
                if not has_dominant[seg_idx]:
                    logger.warning(f"[{content_id}] Segment {segment_row['segment_index']} has no dominant speaker - marking as NEEDS_EMBEDDING")
                    # Mark segments with no dominant speaker as NEEDS_EMBEDDING
                    segment_assignments.append({
                        'segment_index': segment_row['segment_index'],
                        'speaker': 'NEEDS_EMBEDDING',
                        'confidence': 0.0,
                        'start': segment_row['start'],
                        'end': segment_row['end'],
                        'word_count': segment_row['word_count']
                    })
                    continue
                
                speaker_idx = dominant_speaker_indices[seg_idx]
                dominant_speaker = unique_speakers[speaker_idx]
                confidence = max_overlap_pcts[seg_idx]
                
                # Check if dominant speaker is SPEAKER_99 (uncertain)
                if dominant_speaker == 'SPEAKER_99':
                    logger.info(f"[{content_id}] Segment {segment_row['segment_index']} has SPEAKER_99 as dominant - marking as NEEDS_EMBEDDING")
                    segment_assignments.append({
                        'segment_index': segment_row['segment_index'],
                        'speaker': 'NEEDS_EMBEDDING',
                        'confidence': confidence,
                        'start': segment_row['start'],
                        'end': segment_row['end'],
                        'word_count': segment_row['word_count']
                    })
                else:
                    segment_assignments.append({
                        'segment_index': segment_row['segment_index'],
                        'speaker': dominant_speaker,
                        'confidence': confidence,
                        'start': segment_row['start'],
                        'end': segment_row['end'],
                        'word_count': segment_row['word_count']
                    })
        
        vectorization_duration = time.time() - vectorization_start
        logger.info(f"[{content_id}] Found {len(segment_assignments)} segments with dominant speakers")
        logger.debug(f"[{content_id}] Vectorized in {vectorization_duration:.3f}s")
        
        # Process all assignments
        assignments_made = 0
        if segment_assignments:
            # Build a mapping of segment_index to speaker
            segment_to_speaker = {sa['segment_index']: (sa['speaker'], sa['confidence']) 
                                  for sa in segment_assignments}
            
            # Get all words for segments that currently have BAD_GRAMMAR_SINGLE category
            segment_indices = list(segment_to_speaker.keys())
            words_mask = (
                word_table.df['segment_index'].isin(segment_indices) &
                (word_table.df['speaker_current'] == 'BAD_GRAMMAR_SINGLE')
            )
            
            if words_mask.any():
                # Fully vectorized assignment using pandas operations
                words_to_update_indices = word_table.df[words_mask].index
                
                # Map segment indices to speakers and confidences
                new_speakers = word_table.df.loc[words_to_update_indices, 'segment_index'].map(
                    lambda seg_idx: segment_to_speaker[seg_idx][0]
                )
                new_confidences = word_table.df.loc[words_to_update_indices, 'segment_index'].map(
                    lambda seg_idx: segment_to_speaker[seg_idx][1]
                )
                
                # Vectorized assignment - update all at once
                word_table.df.loc[words_to_update_indices, 'speaker_current'] = new_speakers
                word_table.df.loc[words_to_update_indices, 'confidence'] = new_confidences
                word_table.df.loc[words_to_update_indices, 'method'] = 'bad_grammar_single_speaker_safe'
                word_table.df.loc[words_to_update_indices, 'stage'] = 'stage5'
                word_table.df.loc[words_to_update_indices, 'reason'] = new_confidences.map(
                    lambda conf: f"Bad grammar single speaker (safe) with {conf:.1%} overlap"
                )
                
                assignments_made = len(words_to_update_indices)
        
        # Calculate final stats
        total_words_in_segments = bad_grammar_single_segments['word_count'].sum()
        assignment_rate = (assignments_made / total_words_in_segments * 100) if total_words_in_segments > 0 else 0
        
        stage_duration = time.time() - stage_start_time
        
        result = {
            'status': 'success',
            'data': {'word_table': word_table},
            'stats': {
                'segments_analyzed': len(bad_grammar_single_segments),
                'segments_with_dominant_speaker': len(segment_assignments),
                'safe_segments_processed': len(segment_assignments),
                'words_assigned': assignments_made,
                'bad_grammar_single_assignments': assignments_made,
                'assignment_rate': assignment_rate,
                'duration': stage_duration
            }
        }
        
        logger.info(f"[{content_id}] Stage 5 completed: {assignments_made}/{total_words_in_segments} words assigned ({assignment_rate:.1f}%) in {stage_duration:.2f}s")
        logger.info(f"[{content_id}] - All {len(segment_assignments)} segments were safe and processed without re-transcription")
        
        return result
        
    except Exception as e:
        stage_duration = time.time() - stage_start_time
        logger.error(f"[{content_id}] Stage 5 failed: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'stats': {'duration': stage_duration}
        }


