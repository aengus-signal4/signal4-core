#!/usr/bin/env python3
"""
Stage 8: Good Grammar + Multi-Speaker Assignment
================================================

Eighth stage of the stitch pipeline that handles segments with good grammar but multiple speakers.

Key Responsibilities:
- Process all words categorized as GOOD_GRAMMAR_MULTI by Stage 3
- Assign individual words to diarization segments based on overlap
- Use sentence boundaries (periods) to handle speaker transitions
- Mark words as NEEDS_EMBEDDING when diarization overlap is unclear
- Provide detailed assignment statistics and confidence scores

Input:
- WordTable from Stage 7 with categorized words and assignments
- Diarization data with speaker segments
- Segments marked as GOOD_GRAMMAR_MULTI needing speaker resolution

Output:
- WordTable with speaker assignments for GOOD_GRAMMAR_MULTI words
- Words assigned to specific speakers or marked as NEEDS_EMBEDDING
- Assignment statistics including transitions and confidence scores

Key Components:
- WordLevelSpeakerAssigner: Main class for word-level speaker assignment
- DiarizationOverlapAnalyzer: Analyzes overlap between words and diarization
- Sentence boundary detection for natural speaker transitions

Methods:
- good_grammar_multi_speaker_stage(): Main entry point called by stitch pipeline
- WordLevelSpeakerAssigner.assign_words_to_speakers(): Core assignment logic
- DiarizationOverlapAnalyzer.get_word_diarization_overlaps(): Overlap calculation

Performance:
- Moderate complexity with overlap calculations
- Uses vectorized operations where possible
- O(n*m) complexity for n words and m diarization segments
"""

import logging
import time
import json
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
    summarize_speaker_assignments)
from src.processing_steps.stitch_steps.util_wav2vec2 import run_wav2vec2_on_unassigned_words
from src.storage.s3_utils import S3Storage

logger = setup_worker_logger('stitch')


class DiarizationOverlapAnalyzer:
    """Analyzes overlap between transcript segments and diarization data."""
    
    def __init__(self, diarization_segments: List[Dict]):
        """Initialize with diarization data (should be pre-merged from Stage 3)."""
        self.diarization_segments = sorted(diarization_segments, key=lambda x: x.get('start', 0))
    
    def get_word_diarization_overlaps(self, word_start: float, word_end: float) -> List[Dict]:
        """
        Get all diarization segments that overlap with a word's time range.
        
        Returns:
            List of overlapping diarization segments with overlap percentages
        """
        overlapping_speakers = []
        word_duration = word_end - word_start
        
        if word_duration <= 0:
            return []
        
        for dia_seg in self.diarization_segments:
            dia_start = dia_seg.get('start', 0)
            dia_end = dia_seg.get('end', 0)
            dia_speaker = dia_seg.get('speaker', 'UNKNOWN')
            
            # Check for overlap
            overlap_start = max(word_start, dia_start)
            overlap_end = min(word_end, dia_end)
            
            if overlap_start < overlap_end:
                overlap_duration = overlap_end - overlap_start
                overlap_percentage = overlap_duration / word_duration
                overlapping_speakers.append({
                    'speaker': dia_speaker,
                    'overlap_percentage': overlap_percentage,
                    'overlap_duration': overlap_duration,
                    'diarization_start': dia_start,
                    'diarization_end': dia_end
                })
        
        # Sort by overlap percentage (descending)
        overlapping_speakers.sort(key=lambda x: x['overlap_percentage'], reverse=True)
        
        return overlapping_speakers
    
    def find_sentence_boundaries(self, words: List[Dict]) -> List[int]:
        """
        Find indices of words that end sentences (have period, !, or ?).
        
        Returns:
            List of word indices that are sentence endings
        """
        sentence_endings = []
        for i, word in enumerate(words):
            text = word.get('text', '')
            if any(text.endswith(p) for p in ['.', '!', '?']):
                sentence_endings.append(i)
        return sentence_endings


class WordLevelSpeakerAssigner:
    """Assigns speakers to individual words in multi-speaker segments."""
    
    def __init__(self, diarization_segments: List[Dict]):
        """Initialize with diarization data."""
        self.overlap_analyzer = DiarizationOverlapAnalyzer(diarization_segments)
        self.diarization_segments = diarization_segments
        
    # DEPRECATED: This method is no longer used as it conflicts with sentence-integrity approach
    # def ensure_diarization_coverage(self, assignments: List[Dict], segment_words: List[Dict]) -> List[Dict]:
    #     """
    #     [DEPRECATED] This method would reassign individual words to ensure each diarization 
    #     segment has coverage, but this breaks up sentences which we want to keep together.
    #     Keeping the code for reference but it should not be used.
    #     """
    #     pass
        
    def assign_words_to_speakers(self, segment_words: List[Dict], use_sentence_boundaries: bool = True) -> List[Dict]:
        """
        Assign speakers to individual words based on diarization overlap.
        Prioritizes keeping sentences together with their dominant speaker.
        Uses vectorized operations for efficient processing.
        
        Args:
            segment_words: List of word dictionaries with timing info
            use_sentence_boundaries: Whether to consider sentence boundaries for speaker changes
            
        Returns:
            List of assignment decisions for each word
        """
        if not segment_words:
            return []
            
        # Convert to numpy arrays for vectorization
        word_starts = np.array([w.get('start', 0) for w in segment_words])
        word_ends = np.array([w.get('end', 0) for w in segment_words])
        
        # Find sentence boundaries
        sentence_boundaries = self.overlap_analyzer.find_sentence_boundaries(segment_words) if use_sentence_boundaries else []
        sentence_boundaries = [-1] + sentence_boundaries + [len(segment_words) - 1]
        
        # Prepare diarization data for vectorization
        n_diarization = len(self.diarization_segments)
        if n_diarization == 0:
            # No diarization data - assign all to NEEDS_EMBEDDING or NEEDS_LLM
            assignments = []
            for i, word in enumerate(segment_words):
                duration = word_ends[i] - word_starts[i]
                speaker = 'NEEDS_EMBEDDING' if duration > 1.0 else 'NEEDS_LLM'
                assignments.append({
                    'word_index': i,
                    'word_text': word.get('text', ''),
                    'speaker': speaker,
                    'confidence': 0.0,
                    'reason': f"no_diarization_data"
                })
            return assignments
        
        dia_starts = np.array([d.get('start', 0) for d in self.diarization_segments])
        dia_ends = np.array([d.get('end', 0) for d in self.diarization_segments])
        dia_speakers = [d.get('speaker', 'UNKNOWN') for d in self.diarization_segments]
        
        # Get unique speakers and create mapping
        unique_speakers = list(set(dia_speakers))
        speaker_to_idx = {speaker: idx for idx, speaker in enumerate(unique_speakers)}
        n_speakers = len(unique_speakers)
        
        # Create speaker index array for diarization segments
        dia_speaker_indices = np.array([speaker_to_idx[speaker] for speaker in dia_speakers])
        
        # Process sentences in batch
        n_sentences = len(sentence_boundaries) - 1
        sentence_speakers = []
        sentence_confidences = []
        sentence_reasons = []
        
        # Prepare sentence boundaries
        sent_starts = np.zeros(n_sentences)
        sent_ends = np.zeros(n_sentences)
        
        for i in range(n_sentences):
            start_idx = sentence_boundaries[i] + 1
            end_idx = sentence_boundaries[i + 1] + 1
            if start_idx < len(word_starts) and start_idx < end_idx:
                sent_starts[i] = word_starts[start_idx:end_idx].min()
                sent_ends[i] = word_ends[start_idx:end_idx].max()
        
        # Vectorized overlap calculation for all sentences
        # Shape: (n_sentences, n_diarization)
        overlap_starts = np.maximum(sent_starts[:, np.newaxis], dia_starts[np.newaxis, :])
        overlap_ends = np.minimum(sent_ends[:, np.newaxis], dia_ends[np.newaxis, :])
        overlap_durations = np.maximum(0, overlap_ends - overlap_starts)
        
        # Aggregate overlaps by speaker
        # Shape: (n_sentences, n_speakers)
        speaker_overlaps = np.zeros((n_sentences, n_speakers))
        for dia_idx in range(n_diarization):
            speaker_idx = dia_speaker_indices[dia_idx]
            speaker_overlaps[:, speaker_idx] += overlap_durations[:, dia_idx]
        
        # Calculate sentence durations
        sent_durations = sent_ends - sent_starts
        
        # Process each sentence's results
        for sent_idx in range(n_sentences):
            sent_duration = sent_durations[sent_idx]
            sent_overlaps = speaker_overlaps[sent_idx]
            
            # Find if any speaker has overlap
            if np.sum(sent_overlaps) == 0:
                # No overlap
                if sent_duration > 1.0:
                    sentence_speakers.append('NEEDS_EMBEDDING')
                    sentence_confidences.append(0.0)
                    sentence_reasons.append(f"no_diarization_overlap_duration_{sent_duration:.2f}s")
                else:
                    sentence_speakers.append('NEEDS_LLM')
                    sentence_confidences.append(0.0)
                    sentence_reasons.append(f"no_diarization_overlap_short_duration_{sent_duration:.2f}s")
            else:
                # Find dominant speaker
                best_speaker_idx = np.argmax(sent_overlaps)
                best_overlap = sent_overlaps[best_speaker_idx]
                best_speaker = unique_speakers[best_speaker_idx]
                
                # Calculate confidence
                confidence = best_overlap / sent_duration if sent_duration > 0 else 1.0
                
                # Adjust if multiple speakers
                n_speakers_with_overlap = np.sum(sent_overlaps > 0)
                if n_speakers_with_overlap > 1:
                    confidence *= 0.9
                
                sentence_speakers.append(best_speaker)
                sentence_confidences.append(confidence)
                sentence_reasons.append(f"sentence_dominant_speaker_{confidence:.1%}_overlap")
        
        # Create assignments for all words
        assignments = []
        for sent_idx in range(n_sentences):
            start_idx = sentence_boundaries[sent_idx] + 1
            end_idx = sentence_boundaries[sent_idx + 1] + 1
            
            for word_idx in range(start_idx, min(end_idx, len(segment_words))):
                assignments.append({
                    'word_index': word_idx,
                    'word_text': segment_words[word_idx].get('text', ''),
                    'speaker': sentence_speakers[sent_idx],
                    'confidence': sentence_confidences[sent_idx],
                    'reason': sentence_reasons[sent_idx]
                })
        
        return assignments


def log_batch_processing_summary(content_id: str, segment_results: List[Dict], all_assignments: List[Dict]):
    """
    Log summary of batch processing results.
    """
    if not segment_results:
        return
        
    # Overall statistics
    total_segments = len(segment_results)
    total_words = sum(sr['words_assigned'] for sr in segment_results)
    total_transitions = sum(sr['speaker_transitions'] for sr in segment_results)
    avg_confidence = np.mean([sr['avg_confidence'] for sr in segment_results])
    
    logger.info(f"[{content_id}] STAGE 8 BATCH PROCESSING SUMMARY")
    logger.info(f"[{content_id}] " + "=" * 80)
    logger.info(f"[{content_id}] Segments processed: {total_segments}")
    logger.info(f"[{content_id}] Total words assigned: {total_words}")
    logger.info(f"[{content_id}] Total speaker transitions: {total_transitions}")
    logger.info(f"[{content_id}] Average confidence: {avg_confidence:.1%}")
    
    # Speaker distribution
    speaker_counts = {}
    for assignment in all_assignments:
        speaker = assignment['speaker']
        speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
    
    logger.info(f"[{content_id}] Speaker distribution:")
    for speaker, count in sorted(speaker_counts.items(), key=lambda x: -x[1]):
        percentage = (count / total_words) * 100 if total_words > 0 else 0
        logger.info(f"[{content_id}]   {speaker}: {count} words ({percentage:.1f}%)")
    
    # Special assignments
    needs_embedding = speaker_counts.get('NEEDS_EMBEDDING', 0)
    needs_llm = speaker_counts.get('NEEDS_LLM', 0)
    if needs_embedding > 0 or needs_llm > 0:
        logger.info(f"[{content_id}] Special assignments:")
        if needs_embedding > 0:
            logger.info(f"[{content_id}]   NEEDS_EMBEDDING: {needs_embedding} words")
        if needs_llm > 0:
            logger.info(f"[{content_id}]   NEEDS_LLM: {needs_llm} words")
    
    logger.info(f"[{content_id}] " + "=" * 80)


def good_grammar_multi_speaker_stage(content_id: str,
                                   word_table: WordTable,
                                   diarization_data: Dict,
                                   s3_storage: Optional[S3Storage] = None,
                                   test_mode: bool = False) -> Dict[str, Any]:
    """
    Execute Stage 8: Good Grammar + Multi-Speaker Assignment.
    
    This is the primary method called by the stitch pipeline. It processes all
    GOOD_GRAMMAR_MULTI words identified by Stage 3, assigning them to speakers
    based on diarization overlap analysis. Uses sentence boundaries to handle
    natural speaker transitions and marks unclear words as NEEDS_EMBEDDING
    
    Args:
        content_id: Content ID to process
        word_table: WordTable from stage 5
        diarization_data: Diarization data with speaker segments
        s3_storage: S3 storage instance (optional, for wav2vec2)
        test_mode: Whether running in test mode
        
    Returns:
        Dictionary with assignment results
    """
    start_time = time.time()
    stage_name = 'good_grammar_multi_speaker'
    
    logger.info(f"[{content_id}] Stage 8: Good Grammar + Multi-Speaker Assignment")
    
    result = {
        'status': 'success',
        'content_id': content_id,
        'stage': stage_name,
        'data': {
            'word_table': word_table
        },
        'stats': {
            'segments_processed': 0,
            'good_grammar_multi_segments': 0,
            'words_assigned': 0,
            'assignment_confidence': 0.0,
            'speaker_transitions': 0
        },
        'error': None
    }
    
    try:
        # Get diarization segments
        diarization_segments = diarization_data.get('segments', [])
        if not diarization_segments:
            logger.warning(f"[{content_id}] No diarization segments found - cannot perform multi-speaker assignment")
            result['stats']['status'] = 'no_diarization_data'
            return result
        
        logger.info(f"[{content_id}] Found {len(diarization_segments)} diarization segments")
        
        # Get segments from word table with Stage 3 analysis results
        # For speaker_current, check if all words in segment have same value
        segments_df = word_table.df.groupby('segment_index').agg({
            'text': lambda x: ' '.join(x),
            'start': 'min',
            'end': 'max',
            'speaker_current': lambda x: x.iloc[0] if x.nunique() == 1 else 'MIXED',
            'segment_has_good_grammar': 'first',
            'segment_has_multiple_speakers': 'first'
        }).reset_index()
        
        # Check for segments with mixed speaker assignments
        mixed_segments = segments_df[segments_df['speaker_current'] == 'MIXED']
        if len(mixed_segments) > 0:
            logger.warning(f"[{content_id}] Found {len(mixed_segments)} segments with mixed speaker assignments - this shouldn't happen!")
        
        # Debug: Log all GOOD_GRAMMAR_MULTI segments
        all_good_grammar_multi = segments_df[segments_df['speaker_current'] == 'GOOD_GRAMMAR_MULTI']
        logger.info(f"[{content_id}] Total segments with GOOD_GRAMMAR_MULTI: {len(all_good_grammar_multi)}")
        
        # Get unique segment indices where ANY word still has GOOD_GRAMMAR_MULTI
        good_grammar_multi_segment_indices = word_table.df[
            word_table.df['speaker_current'] == 'GOOD_GRAMMAR_MULTI'
        ]['segment_index'].unique()
        
        # Filter segments_df to only include these segments
        good_grammar_multi_segments = segments_df[
            segments_df['segment_index'].isin(good_grammar_multi_segment_indices)
        ].copy()
        
        logger.info(f"[{content_id}] Found {len(good_grammar_multi_segments)} segments with GOOD_GRAMMAR_MULTI words to process")
        
        if len(good_grammar_multi_segments) == 0:
            logger.info(f"[{content_id}] No good grammar + multi-speaker segments to process")
            result['stats']['status'] = 'no_target_segments'
            return result
        
        # Note: Wav2vec2 forced alignment is already run between stages 4 and 5 in the main pipeline
        # We don't need to run it again here for multi-speaker segments
        logger.debug(f"[{content_id}] Using earlier wav2vec2 alignment")
        
        # Initialize word-level speaker assigner
        assigner = WordLevelSpeakerAssigner(diarization_segments)
        overlap_analyzer = DiarizationOverlapAnalyzer(diarization_segments)
        
        # Get all GOOD_GRAMMAR_MULTI words at once
        all_multi_words_mask = (
            word_table.df['speaker_current'] == 'GOOD_GRAMMAR_MULTI'
        )
        all_multi_words_df = word_table.df[all_multi_words_mask].copy()
        
        if len(all_multi_words_df) == 0:
            logger.info(f"[{content_id}] No GOOD_GRAMMAR_MULTI words to process")
            result['stats']['status'] = 'no_words_to_process'
            return result
            
        logger.debug(f"[{content_id}] Processing")
        
        # Prepare diarization data for vectorized operations
        n_diarization = len(diarization_segments)
        dia_starts = np.array([d.get('start', 0) for d in diarization_segments])
        dia_ends = np.array([d.get('end', 0) for d in diarization_segments])
        dia_speakers = [d.get('speaker', 'UNKNOWN') for d in diarization_segments]
        
        # Get unique speakers
        unique_speakers = list(set(dia_speakers))
        speaker_to_idx = {speaker: idx for idx, speaker in enumerate(unique_speakers)}
        n_speakers = len(unique_speakers)
        
        # Process all segments in batch
        batch_start_time = time.time()
        
        # Collect all assignments
        all_assignments = []
        total_words_assigned = 0
        total_confidence = 0
        total_transitions = 0
        segment_results = []
        
        # Group words by segment for batch processing
        for segment_idx in good_grammar_multi_segment_indices:
            segment_words_df = all_multi_words_df[
                all_multi_words_df['segment_index'] == segment_idx
            ].sort_values('start')
            
            if len(segment_words_df) == 0:
                continue
            
            # Convert to list format for assignment function
            segment_words = []
            word_indices = []
            for idx, word_row in segment_words_df.iterrows():
                segment_words.append({
                    'text': word_row['text'],
                    'start': word_row['start'],
                    'end': word_row['end']
                })
                word_indices.append(idx)
            
            # Get assignments for this segment
            assignments = assigner.assign_words_to_speakers(segment_words, use_sentence_boundaries=True)
            
            # Map back to actual word indices
            for assignment in assignments:
                assignment['actual_word_index'] = word_indices[assignment['word_index']]
                assignment['segment_index'] = segment_idx
                all_assignments.append(assignment)
                total_confidence += assignment['confidence']
            
            # Count transitions for this segment
            transitions = 0
            for i in range(1, len(assignments)):
                if assignments[i]['speaker'] != assignments[i-1]['speaker']:
                    transitions += 1
            total_transitions += transitions
            
            # Store segment result
            segment_results.append({
                'segment_index': segment_idx,
                'words_assigned': len(assignments),
                'speaker_transitions': transitions,
                'avg_confidence': np.mean([a['confidence'] for a in assignments]) if assignments else 0
            })
        
        batch_duration = time.time() - batch_start_time
        logger.info(f"[{content_id}] Batch processing completed in {batch_duration:.3f}s ({len(all_assignments)/batch_duration:.0f} words/second)")
        
        # Apply all assignments to word table in batch
        if all_assignments:
            # Prepare update arrays
            word_indices = [a['actual_word_index'] for a in all_assignments]
            speakers = [a['speaker'] for a in all_assignments]
            confidences = [a['confidence'] for a in all_assignments]
            reasons = [a['reason'] for a in all_assignments]
            
            # Batch update
            word_table.df.loc[word_indices, 'speaker_current'] = speakers
            word_table.df.loc[word_indices, 'assignment_method'] = 'good_grammar_multi'
            word_table.df.loc[word_indices, 'assignment_confidence'] = confidences
            
            # Update metadata in batch
            if 'metadata' not in word_table.df.columns:
                word_table.df['metadata'] = [{}] * len(word_table.df)
            
            for i, idx in enumerate(word_indices):
                metadata = word_table.df.at[idx, 'metadata']
                if not isinstance(metadata, dict):
                    metadata = {}
                
                metadata['stage8_assignment'] = {
                    'reason': reasons[i],
                    'confidence': confidences[i],
                    'pending_embedding_confirmation': True
                }
                
                word_table.df.at[idx, 'metadata'] = metadata
            
            total_words_assigned = len(all_assignments)
        
        logger.info(f"[{content_id}] Applied {total_words_assigned} word assignments in batch")
        
        # Log batch processing summary
        log_batch_processing_summary(content_id, segment_results, all_assignments)
        
        # Update processing status
        words_updated = update_processing_status(word_table, stage_name)
        
        # Calculate final statistics
        avg_confidence = total_confidence / total_words_assigned if total_words_assigned > 0 else 0
        
        result['stats'].update({
            'duration': time.time() - start_time,
            'status': 'assignment_completed',
            'segments_processed': len(good_grammar_multi_segments),
            'good_grammar_multi_segments': len(good_grammar_multi_segments),
            'words_assigned': total_words_assigned,
            'assignment_confidence': avg_confidence,
            'speaker_transitions': total_transitions,
            'words_updated': words_updated
        })
        
        # Summary logging
        logger.info(f"[{content_id}] STAGE 8 SUMMARY:")
        logger.info(f"[{content_id}]   Segments processed: {len(good_grammar_multi_segments)}")
        logger.info(f"[{content_id}]   Words assigned: {total_words_assigned}")
        logger.info(f"[{content_id}]   Average confidence: {avg_confidence:.1%}")
        logger.info(f"[{content_id}]   Total speaker transitions: {total_transitions}")
        logger.info(f"[{content_id}]   Note: Speaker embeddings confirmation pending (stages 6 & 7)")
        
        if test_mode:
            # Save detailed results in test mode
            test_output_dir = get_project_root() / "tests" / "content" / content_id / "outputs"
            test_output_dir.mkdir(parents=True, exist_ok=True)
            
            analysis_results = {
                'content_id': content_id,
                'stage': stage_name,
                'stats': result['stats'],
                'segment_results': segment_results
            }
            
            with open(test_output_dir / "good_grammar_multi_assignments.json", 'w') as f:
                json.dump(analysis_results, f, indent=2, default=str)
            logger.info(f"[{content_id}] Detailed assignment results saved to test outputs")
        
        format_stage_stats(content_id, stage_name, result['stats'], start_time)
        
        
        # Add summary of speaker assignments in test mode
        if test_mode:
            summarize_speaker_assignments(word_table, 'stage8_good_grammar_multi', content_id, test_mode)
        
        return result
        
    except Exception as e:
        error_msg = str(e) if e else "Unknown error in good grammar + multi-speaker assignment"
        logger.error(f"[{content_id}] Stage 8 failed: {error_msg}")
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


if __name__ == "__main__":
    """Test the good grammar + multi-speaker assignment stage independently."""
    import argparse
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = get_project_root()
    sys.path.append(str(project_root))
    
    parser = argparse.ArgumentParser(description='Test Good Grammar + Multi-Speaker Assignment Stage')
    parser.add_argument('--content', required=True, help='Content ID to process')
    args = parser.parse_args()
    
    content_id = args.content
    
    logger.info(f"Testing Stage 8: Good Grammar + Multi-Speaker Assignment for content {content_id}")
    
    try:
        # This would need to be implemented with proper test setup
        logger.info(f"Stage 8 test setup not yet implemented")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        logger.error(f"Error details:", exc_info=True)