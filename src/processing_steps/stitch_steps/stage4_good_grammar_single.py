#!/usr/bin/env python3
"""
Stage 4: Good Grammar + Single Speaker Assignment (Slam-Dunk)
=============================================================

Fourth stage of the stitch pipeline that performs high-confidence speaker assignments.

Key Responsibilities:
- Process all words categorized as GOOD_GRAMMAR_SINGLE by Stage 3
- Find the diarization speaker with highest overlap for each segment
- Assign speakers without requiring perfect overlap (trusts Stage 3's single-speaker determination)
- Mark segments without diarization overlap as NEEDS_EMBEDDING
- Provide foundation of confident assignments before complex stages

Input:
- WordTable from Stage 3 with categorized words
- Diarization data with speaker segments

Output:
- WordTable with speaker assignments for GOOD_GRAMMAR_SINGLE words
- Words either assigned to specific speakers or marked as NEEDS_EMBEDDING
- Assignment statistics and confidence scores

Classes:
- VectorizedSlamDunkAssigner: Performs vectorized speaker assignments using pandas

Methods:
- slamdunk_assignment_stage(): Main entry point called by stitch pipeline
- VectorizedSlamDunkAssigner.analyze_segments_vectorized(): Batch process all segments
- VectorizedSlamDunkAssigner._find_perfect_overlaps_vectorized(): Find best speaker overlaps
- VectorizedSlamDunkAssigner._generate_reasons_vectorized(): Generate assignment reasons
"""

import logging
import time
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from src.utils.paths import get_project_root, get_config_path

from src.utils.logger import setup_worker_logger
from src.processing_steps.stitch_steps.stage3_tables import WordTable
from src.processing_steps.stitch_steps.util_stitch import (
    load_stage_config,
    update_processing_status,
    format_stage_stats,
    summarize_speaker_assignments
)

logger = setup_worker_logger('stitch')




class VectorizedSlamDunkAssigner:
    """
    Performs vectorized speaker assignments for GOOD_GRAMMAR_SINGLE segments.
    
    This class uses pandas operations for efficient batch processing of segments,
    finding the best overlapping speaker for each segment using numpy vectorization.
    Trusts Stage 3's single-speaker determination and assigns to best match.
    """
    
    def __init__(self, diarization_segments: List[Dict]):
        """
        Initialize the vectorized slam-dunk assigner.
        
        Args:
            diarization_segments: List of diarization segments with speaker, start, end
        """
        # Convert diarization segments to DataFrame for vectorized operations
        self.diarization_df = pd.DataFrame(diarization_segments)
        if len(self.diarization_df) > 0:
            self.diarization_df = self.diarization_df.sort_values('start').reset_index(drop=True)
        
        logger.debug(f"Initialized vectorized assigner with {len(self.diarization_df)} diarization segments")
    
    def analyze_segments_vectorized(self, segments_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze all segments for slam-dunk assignment using vectorized operations.
        
        This method processes all GOOD_GRAMMAR_SINGLE segments in batch, finding the
        best overlapping speaker for each segment. It trusts Stage 3's grammar and
        single-speaker analysis, assigning to the highest overlap speaker.
        
        Args:
            segments_df: DataFrame with columns:
                - segment_index: Unique segment identifier
                - text: Segment text
                - start/end: Timing boundaries
                - segment_has_good_grammar: Grammar quality from Stage 3
                - segment_has_multiple_speakers: Multi-speaker flag from Stage 3
            
        Returns:
            DataFrame with additional columns:
                - perfect_speaker: Best overlapping speaker
                - overlap_percentage: Overlap percentage (0.0-1.0)
                - is_slam_dunk: Whether segment can be assigned
                - assigned_speaker: Final speaker assignment or NEEDS_EMBEDDING
                - reason: Human-readable assignment reason
        """
        # Create a copy to avoid modifying original
        result_df = segments_df.copy()
        
        # Use Stage 3's grammar analysis instead of re-analyzing
        result_df['has_good_grammar'] = result_df['segment_has_good_grammar']
        
        # Vectorized overlap analysis
        overlap_results = self._find_perfect_overlaps_vectorized(segments_df[['start', 'end']])
        result_df['perfect_speaker'] = overlap_results['perfect_speaker']
        result_df['overlap_percentage'] = overlap_results['overlap_percentage']
        
        # For GOOD_GRAMMAR_SINGLE segments from Stage 3, we trust the categorization
        # and assign to the best overlapping speaker (if any exists)
        result_df['is_slam_dunk'] = (
            result_df['has_good_grammar'] & 
            (~result_df['segment_has_multiple_speakers']) &  # Single speaker segments only
            result_df['perfect_speaker'].notna() & 
            (result_df['perfect_speaker'] != 'NONE')
        )
        
        # Vectorized speaker assignment
        # For GOOD_GRAMMAR_SINGLE segments that can't be assigned due to insufficient
        # diarization, mark as NEEDS_EMBEDDING for later processing
        result_df['assigned_speaker'] = np.where(
            result_df['is_slam_dunk'],
            result_df['perfect_speaker'], 
            np.where(
                (result_df['has_good_grammar']) & 
                (~result_df['segment_has_multiple_speakers']) &
                ((result_df['perfect_speaker'] == 'NONE') | result_df['perfect_speaker'].isna()),
                'NEEDS_EMBEDDING',
                'UNKNOWN'
            )
        )
        
        # Vectorized reason generation
        result_df['reason'] = self._generate_reasons_vectorized(result_df)
        
        return result_df
    
    def _find_perfect_overlaps_vectorized(self, segment_times: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Find best speaker overlaps for all segments using vectorized operations.
        
        For GOOD_GRAMMAR_SINGLE segments, we accept the speaker with highest overlap
        rather than requiring 100% overlap, trusting Stage 3's single-speaker determination.
        Uses numpy broadcasting for efficient overlap calculation across all segments.
        
        Args:
            segment_times: DataFrame with start, end columns for each segment
            
        Returns:
            Dictionary containing:
                - perfect_speaker: Series with best speaker for each segment
                - overlap_percentage: Series with overlap percentage (0.0-1.0)
        """
        n_segments = len(segment_times)
        n_diarization = len(self.diarization_df)
        
        if n_segments == 0 or n_diarization == 0:
            return {
                'perfect_speaker': pd.Series(['NONE'] * n_segments),
                'overlap_percentage': pd.Series([0.0] * n_segments)
            }
        
        # Create matrices for vectorized overlap calculation
        seg_starts = segment_times['start'].values[:, np.newaxis]  # (n_segments, 1)
        seg_ends = segment_times['end'].values[:, np.newaxis]     # (n_segments, 1)
        dia_starts = self.diarization_df['start'].values[np.newaxis, :]  # (1, n_diarization)
        dia_ends = self.diarization_df['end'].values[np.newaxis, :]      # (1, n_diarization)
        
        # Calculate segment durations
        seg_durations = seg_ends - seg_starts  # (n_segments, 1)
        
        # Vectorized overlap calculation
        overlap_starts = np.maximum(seg_starts, dia_starts)  # (n_segments, n_diarization)
        overlap_ends = np.minimum(seg_ends, dia_ends)        # (n_segments, n_diarization)
        overlap_durations = np.maximum(0, overlap_ends - overlap_starts)  # (n_segments, n_diarization)
        
        # Calculate overlap percentages
        with np.errstate(divide='ignore', invalid='ignore'):
            overlap_percentages = overlap_durations / seg_durations  # (n_segments, n_diarization)
            overlap_percentages = np.nan_to_num(overlap_percentages, 0.0)
        
        # For GOOD_GRAMMAR_SINGLE segments, we trust Stage 3's categorization
        # and assign to the speaker with the highest overlap, regardless of percentage
        
        # For each segment, find the best speaker assignment
        perfect_speakers = []
        max_overlaps = []
        
        for i in range(n_segments):
            segment_overlaps = overlap_percentages[i, :]
            
            if len(segment_overlaps) > 0 and np.max(segment_overlaps) > 0:
                # Find the speaker with maximum overlap
                best_idx = np.argmax(segment_overlaps)
                best_overlap = segment_overlaps[best_idx]
                speaker = self.diarization_df.iloc[best_idx]['speaker']
                
                # SPEAKER_99 indicates uncertain/unassigned speaker - needs embedding
                if speaker == 'SPEAKER_99':
                    perfect_speakers.append('NONE')  # Will trigger NEEDS_EMBEDDING
                    max_overlaps.append(best_overlap)
                else:
                    # For GOOD_GRAMMAR_SINGLE, we trust Stage 3 and assign to best speaker
                    perfect_speakers.append(speaker)
                    max_overlaps.append(best_overlap)
            else:
                # No overlaps at all - this shouldn't happen if diarization is complete
                perfect_speakers.append('NONE')
                max_overlaps.append(0.0)
        
        return {
            'perfect_speaker': pd.Series(perfect_speakers),
            'overlap_percentage': pd.Series(max_overlaps)
        }
    
    def _generate_reasons_vectorized(self, result_df: pd.DataFrame) -> pd.Series:
        """
        Generate human-readable assignment reasons using vectorized operations.
        
        Creates detailed explanations for each assignment decision, including
        overlap percentages and why segments were or weren't assigned.
        
        Args:
            result_df: DataFrame with assignment analysis results
            
        Returns:
            Series of reason strings for each segment
        """
        def get_reason(row):
            if row['is_slam_dunk']:
                overlap_pct = row['overlap_percentage'] * 100
                return f"SLAM_DUNK: Good grammar + single speaker + {overlap_pct:.0f}% overlap with {row['perfect_speaker']}"
            elif not row['has_good_grammar']:
                return f"NO_ASSIGNMENT: Poor grammar (Stage 3 analysis)"
            elif row['segment_has_multiple_speakers']:
                return f"NO_ASSIGNMENT: Multiple speakers (Stage 3 analysis)"
            elif row['perfect_speaker'] in ['NONE', None]:
                if row['assigned_speaker'] == 'NEEDS_EMBEDDING':
                    return f"NEEDS_EMBEDDING: Good grammar + single speaker but insufficient diarization overlap"
                else:
                    return f"NO_ASSIGNMENT: Good grammar + single speaker but no diarization overlap found"
            else:
                return "NO_ASSIGNMENT: Unknown reason"
        
        return result_df.apply(get_reason, axis=1)
    
    def get_stats(self, result_df: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive statistics from vectorized assignment results.
        
        Computes metrics including assignment rates, speaker distributions,
        and segments needing further processing.
        
        Args:
            result_df: DataFrame with assignment results
            
        Returns:
            Dictionary with statistics:
                - segments_analyzed: Total segments processed
                - good_grammar_segments: Segments with good grammar
                - slam_dunk_assignments: Successfully assigned segments
                - segments_needing_embedding: Segments marked for embedding
                - speaker_assignments: Distribution of speaker assignments
        """
        total_segments = len(result_df)
        good_grammar_count = result_df['has_good_grammar'].sum()
        perfect_overlap_count = (result_df['perfect_speaker'] != 'NONE').sum()
        slam_dunk_count = result_df['is_slam_dunk'].sum()
        unknown_count = (result_df['assigned_speaker'] == 'UNKNOWN').sum()
        needs_embedding_count = (result_df['assigned_speaker'] == 'NEEDS_EMBEDDING').sum()
        
        # Count speaker assignments
        speaker_assignments = {}
        assigned_speakers = result_df[(result_df['assigned_speaker'] != 'UNKNOWN') & 
                                     (result_df['assigned_speaker'] != 'NEEDS_EMBEDDING')]['assigned_speaker']
        if len(assigned_speakers) > 0:
            speaker_assignments = assigned_speakers.value_counts().to_dict()
        
        return {
            'segments_analyzed': total_segments,
            'good_grammar_segments': int(good_grammar_count),
            'perfect_overlap_segments': int(perfect_overlap_count),
            'slam_dunk_assignments': int(slam_dunk_count),
            'segments_needing_embedding': int(needs_embedding_count),
            'segments_remaining_unknown': int(unknown_count),
            'speaker_assignments': speaker_assignments
        }


def slamdunk_assignment_stage(content_id: str,
                            word_table: WordTable,
                            diarization_data: Dict,
                            test_mode: bool = False) -> Dict[str, Any]:
    """
    Main entry point for Stage 4: Good Grammar + Single Speaker Assignment.
    
    This is the primary method called by the stitch pipeline. It processes all
    GOOD_GRAMMAR_SINGLE words identified by Stage 3, assigning them to speakers
    based on diarization overlap. Trusts Stage 3's single-speaker determination
    and assigns to the best overlapping speaker without requiring perfect overlap.
    
    Args:
        content_id: Content ID being processed (e.g., "Bdb001")
        word_table: WordTable from Stage 3 with categorized words
        diarization_data: Dictionary with 'segments' list containing speaker segments
        test_mode: If True, saves detailed outputs for debugging
        
    Returns:
        Dictionary containing:
            - status: 'success' or 'error'
            - data: Dict with updated word_table
            - stats: Assignment statistics and performance metrics
            - error: Error message if status is 'error'
            
    Example:
        result = slamdunk_assignment_stage("Bdb001", word_table, diarization_data, test_mode=True)
        if result['status'] == 'success':
            updated_table = result['data']['word_table']
            print(f"Assigned {result['stats']['slam_dunk_assignments']} segments")
    """
    start_time = time.time()
    stage_name = 'slamdunk_assignment'
    
    logger.info(f"[{content_id}] Stage 4: Good Grammar + Single Speaker Assignment (Slam-Dunk)")
    
    result = {
        'status': 'success',
        'content_id': content_id,
        'stage': stage_name,
        'data': {
            'word_table': word_table
        },
        'stats': {
            'segments_processed': 0,
            'slam_dunk_assignments': 0,
            'speaker_assignments': {},
            'assignment_rate': 0.0
        },
        'error': None
    }
    
    try:
        # Get diarization segments
        diarization_segments = diarization_data.get('segments', [])
        if not diarization_segments:
            logger.warning(f"[{content_id}] No diarization segments found - cannot perform speaker assignment")
            result['stats']['status'] = 'no_diarization_data'
            return result
        
        logger.info(f"[{content_id}] Found {len(diarization_segments)} diarization segments")
        
        # Get unique segments from word table, including Stage 3 analysis flags
        segments_df = word_table.df.groupby('segment_index').agg({
            'text': lambda x: ' '.join(x),
            'start': 'min',
            'end': 'max',
            'segment_has_good_grammar': 'first',
            'segment_has_multiple_speakers': 'first'
        }).reset_index()
        
        logger.debug(f"[{content_id}] Processing {len(segments_df)} transcript segments")
        
        # Initialize vectorized slam-dunk assigner
        assigner = VectorizedSlamDunkAssigner(diarization_segments)
        
        # Brief planning log
        logger.debug(f"[{content_id}] Analyzing {len(segments_df)} segments with {len(diarization_segments)} diarization segments")
        
        # Vectorized analysis of all segments
        assignment_start = time.time()
        assignment_results_df = assigner.analyze_segments_vectorized(segments_df)
        assignment_duration = time.time() - assignment_start
        
        logger.debug(f"[{content_id}] Vectorized in {assignment_duration:.3f}s ({len(segments_df)/assignment_duration:.0f} segments/second)")
        
        # Apply assignments to word table using vectorized operations
        if 'speaker_current' not in word_table.df.columns:
            word_table.df['speaker_current'] = 'UNKNOWN'
        
        # Log segments that couldn't be assigned
        unassigned_good_grammar_single = assignment_results_df[
            (assignment_results_df['has_good_grammar']) & 
            (~assignment_results_df['segment_has_multiple_speakers']) &
            (assignment_results_df['assigned_speaker'] == 'UNKNOWN')
        ]
        
        if len(unassigned_good_grammar_single) > 0:
            logger.warning(f"[{content_id}] {len(unassigned_good_grammar_single)} GOOD_GRAMMAR_SINGLE segments could not be assigned:")
            for _, seg in unassigned_good_grammar_single.iterrows():
                logger.warning(f"[{content_id}]   Segment {seg['segment_index']}: '{seg['text'][:50]}...' - reason: {seg['reason']}")
        
        # Create a mapping from segment_index to assigned_speaker for all assignments (including NEEDS_EMBEDDING)
        assignments_to_apply = assignment_results_df[
            (assignment_results_df['assigned_speaker'] != 'UNKNOWN') & 
            (assignment_results_df['has_good_grammar']) & 
            (~assignment_results_df['segment_has_multiple_speakers'])
        ]
        segment_speaker_map = dict(zip(assignments_to_apply['segment_index'], assignments_to_apply['assigned_speaker']))
        
        # Only update words that are currently GOOD_GRAMMAR_SINGLE and have successful slam-dunk assignments
        # This preserves the category assignments from Stage 3 for other words
        good_grammar_single_mask = word_table.df['speaker_current'] == 'GOOD_GRAMMAR_SINGLE'
        successful_segment_mask = word_table.df['segment_index'].isin(segment_speaker_map.keys())
        update_mask = good_grammar_single_mask & successful_segment_mask
        
        # Apply speaker assignments only to successful slam-dunk words
        word_table.df.loc[update_mask, 'speaker_current'] = word_table.df.loc[update_mask, 'segment_index'].map(segment_speaker_map)
        
        # Initialize assignment method and confidence columns if they don't exist
        if 'assignment_method' not in word_table.df.columns:
            word_table.df['assignment_method'] = 'unassigned'
        if 'assignment_confidence' not in word_table.df.columns:
            word_table.df['assignment_confidence'] = 0.0
        
        # Update assignment method and confidence based on assignment type
        # For NEEDS_EMBEDDING, set method appropriately
        needs_embedding_mask = update_mask & (word_table.df['speaker_current'] == 'NEEDS_EMBEDDING')
        slam_dunk_mask = update_mask & (word_table.df['speaker_current'] != 'NEEDS_EMBEDDING')
        
        word_table.df.loc[slam_dunk_mask, 'assignment_method'] = 'slam_dunk'
        word_table.df.loc[needs_embedding_mask, 'assignment_method'] = 'needs_embedding'
        
        # Set confidence for slam dunk assignments
        if slam_dunk_mask.any():
            word_table.df.loc[slam_dunk_mask, 'assignment_confidence'] = word_table.df.loc[slam_dunk_mask, 'segment_index'].map(
                dict(zip(assignments_to_apply[assignments_to_apply['assigned_speaker'] != 'NEEDS_EMBEDDING']['segment_index'], 
                        assignments_to_apply[assignments_to_apply['assigned_speaker'] != 'NEEDS_EMBEDDING']['overlap_percentage']))
            )
        
        # Update processing status
        words_updated = update_processing_status(word_table, stage_name)
        
        # Calculate final statistics from vectorized results
        stats = assigner.get_stats(assignment_results_df)
        assignment_rate = (stats['slam_dunk_assignments'] / stats['segments_analyzed']) * 100 if stats['segments_analyzed'] > 0 else 0
        
        result['stats'].update({
            'duration': time.time() - start_time,
            'status': 'slam_dunk_completed',
            'segments_processed': stats['segments_analyzed'],
            'good_grammar_segments': stats['good_grammar_segments'],
            'perfect_overlap_segments': stats['perfect_overlap_segments'],
            'slam_dunk_assignments': stats['slam_dunk_assignments'],
            'segments_remaining_unknown': stats['segments_remaining_unknown'],
            'speaker_assignments': stats['speaker_assignments'],
            'assignment_rate': assignment_rate,
            'words_updated': words_updated
        })
        
        # Summary logging
        logger.info(f"[{content_id}] SLAM-DUNK SUMMARY: {stats['slam_dunk_assignments']}/{stats['segments_analyzed']} assignments ({assignment_rate:.1f}%), speakers: {stats['speaker_assignments']}")
        
        if test_mode:
            # Save detailed results in test mode
            test_output_dir = get_project_root() / "tests" / "content" / content_id / "outputs"
            test_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Convert DataFrame to serializable format
            segment_assignments_list = assignment_results_df.to_dict('records')
            
            assignment_results = {
                'content_id': content_id,
                'stage': stage_name,
                'stats': result['stats'],
                'segment_assignments': segment_assignments_list,
                'speaker_assignments': stats['speaker_assignments'],
                'vectorized': True,
                'processing_rate_segments_per_second': len(segments_df)/assignment_duration if assignment_duration > 0 else 0
            }
            
            import json
            with open(test_output_dir / "slamdunk_assignments.json", 'w') as f:
                json.dump(assignment_results, f, indent=2, default=str)
            logger.info(f"[{content_id}] Detailed vectorized assignment results saved to test outputs")
        
        format_stage_stats(content_id, stage_name, result['stats'], start_time)
        
        # Add summary of speaker assignments in test mode
        if test_mode:
            summarize_speaker_assignments(word_table, 'stage4_good_grammar_single', content_id, test_mode)
        
        return result
        
    except Exception as e:
        error_msg = str(e) if e else "Unknown error in slam-dunk assignment"
        logger.error(f"[{content_id}] Stage 5 failed: {error_msg}")
        logger.error(f"[{content_id}] Error details:", exc_info=True)
        
        result.update({
            'status': 'error',
            'error': error_msg,
            'stats': {
                'duration': time.time() - start_time,
                'status': 'assignment_failed',
                'segments_processed': 0
            }
        })
        
        return result


