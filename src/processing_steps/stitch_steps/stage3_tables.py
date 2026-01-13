#!/usr/bin/env python3
"""
Stage 3: Table Creation - Central Word Tracking
===============================================

Third stage of the stitch pipeline that creates the core data structures.

Key Responsibilities:
- Create WordTable as central tracking for all words through the pipeline
- Create SegmentTable for diarization segment reference
- Analyze grammar quality for each segment/sentence
- Determine single vs multi-speaker segments
- Categorize words into 4 groups for processing:
  * GOOD_GRAMMAR_SINGLE: Good grammar + single speaker
  * GOOD_GRAMMAR_MULTI: Good grammar + multiple speakers
  * BAD_GRAMMAR_SINGLE: Bad grammar + single speaker  
  * BAD_GRAMMAR_MULTI: Bad grammar + multiple speakers

Input:
- Cleaned diarization segments from Stage 2
- Cleaned transcript segments from Stage 2

Output:
- WordTable with all words categorized for speaker assignment
- SegmentTable with diarization segments
- Grammar and speaker analysis for each segment

Classes:
- WordTable: Main tracking structure for word-level data
- SegmentTable: Reference structure for diarization segments

Methods:
- tables_stage(): Main entry point called by stitch pipeline
- WordTable.initialize_from_transcript_data(): Create word entries
- SegmentTable.initialize_from_diarization_data(): Create segment entries
"""

import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import Counter
import pandas as pd
import numpy as np

from src.utils.logger import setup_worker_logger

logger = setup_worker_logger('stitch')


class WordTable:
    """
    Tracks individual words through the stitch pipeline.
    
    Each word has:
    - word_id: Unique identifier
    - text: The actual word text
    - start/end: Timing information
    - confidence: Whisper confidence score
    - speaker_current: Current speaker assignment (starts as UNKNOWN)
    - resolution_method: How the speaker was assigned
    - processing_status: Current processing status
    - assignment_history: Detailed tracking of all speaker changes through stages
    """
    
    def __init__(self, content_id: str):
        """Initialize a new WordTable."""
        self.content_id = content_id
        self.diarization_segments = []  # Store diarization data for overlap analysis
        self._segment_speaker_cache = {}  # Cache for segment-level speaker analysis
        self.df = pd.DataFrame(columns=[
            'word_id',               # Unique identifier for this word
            'text',                  # The actual word text
            'start',                 # Start time in seconds
            'end',                   # End time in seconds
            'confidence',            # Whisper confidence score
            'segment_index',         # Which segment this word came from
            'word_index',            # Position within the segment
            'sentence_id',           # Which sentence this word belongs to
            'speaker_current',       # Current speaker assignment
            'assignment_confidence', # Confidence in speaker assignment
            'resolution_method',     # How speaker was assigned
            'processing_status',     # Current processing status
            'metadata',              # Additional metadata dict
            'assignment_history',    # List of all speaker assignments through pipeline stages
            'has_good_grammar',      # Whether the word's sentence has good grammar (1+ punctuation, 1+ capital)
            'has_multiple_speakers', # Whether the word's sentence potentially has multiple speakers
            'segment_has_good_grammar',      # Whether the word's segment has good grammar
            'segment_has_multiple_speakers', # Whether the word's segment potentially has multiple speakers
            'segment_within_1s_of_non_majority_speaker'  # Whether segment is within 1.0s of non-majority speaker
        ])
        
        logger.debug(f"[{content_id}] WordTable initialized")
    
    def set_diarization_data(self, diarization_segments: List[Dict]) -> None:
        """
        Set diarization data for overlap analysis, merging sequential segments from the same speaker.
        
        Args:
            diarization_segments: List of diarization segment dictionaries
        """
        logger.debug(f"[{self.content_id}] set_diarization_data called with {len(diarization_segments) if diarization_segments else 0} segments")
        logger.debug(f"[{self.content_id}] diarization_segments type: {type(diarization_segments)}")
        if diarization_segments and len(diarization_segments) > 0:
            logger.debug(f"[{self.content_id}] First segment sample: {diarization_segments[0]}")
        
        if not diarization_segments:
            logger.error(f"[{self.content_id}] Empty diarization_segments provided to set_diarization_data")
            raise ValueError(f"Empty diarization segments provided - cannot process without speaker data")
        
        # Sort segments by start time
        sorted_segments = sorted(diarization_segments, key=lambda x: x.get('start', 0))
        
        # Merge sequential segments from the same speaker (regardless of timing gaps)
        merged_segments = []
        current_segment = None
        
        for segment in sorted_segments:
            speaker = segment.get('speaker', 'UNKNOWN')
            start = segment.get('start', 0)
            end = segment.get('end', 0)
            
            if current_segment is None:
                # First segment
                current_segment = {
                    'speaker': speaker,
                    'start': start,
                    'end': end
                }
            elif current_segment['speaker'] == speaker:
                # Same speaker, merge with current segment (extend end time)
                current_segment['end'] = end
            else:
                # Different speaker, save current and start new
                merged_segments.append(current_segment)
                current_segment = {
                    'speaker': speaker,
                    'start': start,
                    'end': end
                }
        
        # Don't forget the last segment
        if current_segment is not None:
            merged_segments.append(current_segment)
        
        self.diarization_segments = merged_segments
        
        # Clear cache when diarization data changes
        self._segment_speaker_cache = {}
        
        logger.debug(f"[{self.content_id}] Set diarization data: {len(diarization_segments)} original segments -> {len(merged_segments)} merged segments")
    
    def _precompute_segment_speaker_overlaps(self) -> None:
        """
        Precompute diarization overlap analysis for all segments vectorized.
        This is much more efficient than computing overlap for each segment individually.
        """
        if not self.diarization_segments or len(self.df) == 0:
            return
        
        # Get unique segments from word table
        segment_groups = self.df.groupby('segment_index').agg({
            'start': 'min',
            'end': 'max'
        }).reset_index()
        
        if len(segment_groups) == 0:
            return
        
        logger.debug(f"[{self.content_id}] Precomputing speaker overlaps for {len(segment_groups)} segments")
        
        # Convert to numpy arrays for vectorized computation
        seg_starts = segment_groups['start'].values
        seg_ends = segment_groups['end'].values
        seg_indices = segment_groups['segment_index'].values
        
        diar_starts = np.array([seg['start'] for seg in self.diarization_segments])
        diar_ends = np.array([seg['end'] for seg in self.diarization_segments])
        diar_speakers = np.array([seg['speaker'] for seg in self.diarization_segments])
        
        # Vectorized overlap calculation
        # Broadcasting: seg_starts[:, None] creates (N_seg, 1), diar_starts creates (1, M_diar)
        overlap_starts = np.maximum(seg_starts[:, None], diar_starts[None, :])  # (N_seg, M_diar)
        overlap_ends = np.minimum(seg_ends[:, None], diar_ends[None, :])        # (N_seg, M_diar)
        overlap_durations = np.maximum(0, overlap_ends - overlap_starts)        # (N_seg, M_diar)
        
        # Calculate segment durations
        segment_durations = seg_ends - seg_starts
        
        # For each segment, find the speaker with maximum overlap
        for i, seg_idx in enumerate(seg_indices):
            seg_duration = segment_durations[i]
            if seg_duration <= 0:
                self._segment_speaker_cache[seg_idx] = False  # Assume single speaker for zero duration
                continue
            
            # Get overlaps for this segment
            seg_overlaps = overlap_durations[i, :]  # (M_diar,)
            
            # Calculate overlap percentages
            overlap_percentages = seg_overlaps / seg_duration
            
            # Find speaker with maximum overlap
            max_overlap_idx = np.argmax(overlap_percentages)
            max_overlap_pct = overlap_percentages[max_overlap_idx]
            
            # Check if single speaker dominates (90%+ overlap)
            # If any speaker has ≥90% overlap, it's considered single speaker
            # Otherwise it's multi-speaker (speaker transition or multiple speakers)
            has_multiple_speakers = max_overlap_pct < 0.9
            
            self._segment_speaker_cache[seg_idx] = has_multiple_speakers
        
        logger.debug(f"[{self.content_id}] Precomputed speaker overlaps: {sum(self._segment_speaker_cache.values())} multi-speaker segments")
    
    def _merge_split_contractions(self, words: List[Dict]) -> List[Dict]:
        """Merge words that have no space between them (based on Whisper's spacing)."""
        if not words or len(words) < 2:
            return words
        
        # Convert to DataFrame for vectorized operations
        df = pd.DataFrame(words)
        
        # Handle column access safely
        if 'word' in df.columns:
            df['word'] = df['word'].astype(str)  # Keep original spacing
        else:
            df['word'] = ''
            
        if 'start' in df.columns:
            df['start'] = df['start'].astype(float)
        else:
            df['start'] = 0.0
            
        if 'end' in df.columns:
            df['end'] = df['end'].astype(float)
        else:
            df['end'] = 0.0
            
        if 'confidence' in df.columns:
            df['confidence'] = df['confidence'].astype(float)
        else:
            df['confidence'] = 1.0
        
        # Create next word columns for comparison
        df['next_word'] = df['word'].shift(-1).fillna('')
        df['next_start'] = df['start'].shift(-1).fillna(0)
        df['next_end'] = df['end'].shift(-1).fillna(0)
        df['next_confidence'] = df['confidence'].shift(-1).fillna(1.0)
        
        # FIXED: Whisper words DON'T include spaces - they're just raw tokens
        # Only merge actual contractions (words starting with apostrophe or similar)
        # Examples: "I" + "'m" → "I'm", "don" + "'t" → "don't"
        df['next_is_contraction'] = df['next_word'].str.match(r'^[\'\u2019]')  # Apostrophe or right single quote

        # Final merge condition - only merge actual contractions
        df['should_merge'] = (
            (df['next_word'] != '') &  # Next word exists
            df['next_is_contraction']  # Next word is a contraction part (starts with apostrophe)
        )
        
        # Build result list efficiently
        merged_words = []
        skip_next = False
        
        for i, row in df.iterrows():
            if skip_next:
                skip_next = False
                continue
                
            if row['should_merge']:
                # Create merged word - combine without extra spaces
                merged_text = row['word'] + row['next_word']  # Direct concatenation
                merged_word = {
                    'word': merged_text,
                    'start': row['start'],
                    'end': row['next_end'],
                    'confidence': min(row['confidence'], row['next_confidence']),
                    '_merged_from_split': True,
                    '_original_parts': [row['word'], row['next_word']]
                }
                merged_words.append(merged_word)
                skip_next = True  # Skip the next word since we merged it
            else:
                # Keep original word
                original_word = {
                    'word': row['word'],
                    'start': row['start'], 
                    'end': row['end'],
                    'confidence': row['confidence']
                }
                # Preserve any other original fields
                for key, value in words[i].items():
                    if key not in original_word:
                        original_word[key] = value
                merged_words.append(original_word)
        
        # Log merging statistics with details
        original_count = len(words)
        merged_count = len(merged_words)
        if merged_count < original_count:
            merges_made = original_count - merged_count
            
            # Debug: show what was merged if many merges
            if merges_made >= 3:
                merge_details = []
                for word in merged_words:
                    if word.get('_merged_from_split'):
                        original_parts = word.get('_original_parts', [])
                        merge_details.append(f"{'|'.join(original_parts)}→{word['word']}")
                logger.debug(f"[{self.content_id}] Merged {merges_made} contractions ({original_count}→{merged_count}): {', '.join(merge_details[:3])}{'...' if len(merge_details) > 3 else ''}")
            else:
                logger.debug(f"[{self.content_id}] Merged {merges_made} split contractions: {original_count} -> {merged_count} words")
        
        return merged_words
    
    def _clean_duplicate_segments(self, transcript_segments: List[Dict]) -> List[Dict]:
        """Clean duplicate and overlapping segments using confidence-based selection and overlap rejection."""
        if not transcript_segments:
            return []
        
        logger.debug(f"[{self.content_id}] Cleaning duplicate segments from {len(transcript_segments)} segments")
        
        # Step 1: Remove empty segments
        non_empty_segments = []
        empty_removed = 0
        
        for segment in transcript_segments:
            text = segment.get('text', '').strip()
            if text:
                non_empty_segments.append(segment)
            else:
                empty_removed += 1
        
        # Step 2: Remove segments that overlap significantly with earlier segments
        cleaned_segments = self._remove_overlapping_segments(non_empty_segments, overlap_threshold=0.3)
        overlap_removed = len(non_empty_segments) - len(cleaned_segments)
        
        # Step 3: Group remaining close segments and pick the best from each group
        final_segments = self._select_best_from_groups(cleaned_segments, time_window=0.5)
        group_filtered = len(cleaned_segments) - len(final_segments)
        
        total_removed = empty_removed + overlap_removed + group_filtered
        logger.debug(f"[{self.content_id}] Segment cleaning complete: {len(transcript_segments)} -> {len(final_segments)} segments")
        logger.debug(f"[{self.content_id}] Removed: {empty_removed} empty, {overlap_removed} overlapping, {group_filtered} low-quality duplicates")
        
        return final_segments
    
    def _remove_overlapping_segments(self, segments: List[Dict], overlap_threshold: float = 0.3) -> List[Dict]:
        """Remove segments that overlap significantly with earlier segments."""
        sorted_segments = sorted(segments, key=lambda x: float(x.get('start', 0)))
        result = []
        
        for segment in sorted_segments:
            seg_start = float(segment.get('start', 0))
            seg_end = float(segment.get('end', 0))
            
            # Check overlap with existing segments
            has_significant_overlap = False
            
            for existing_seg in result:
                existing_start = float(existing_seg.get('start', 0))
                existing_end = float(existing_seg.get('end', 0))
                
                # Calculate overlap
                overlap_start = max(seg_start, existing_start)
                overlap_end = min(seg_end, existing_end)
                overlap_duration = max(0, overlap_end - overlap_start)
                
                seg_duration = seg_end - seg_start
                if seg_duration > 0:
                    overlap_ratio = overlap_duration / seg_duration
                    if overlap_ratio > overlap_threshold:
                        has_significant_overlap = True
                        logger.debug(f"[{self.content_id}] Removing overlapping segment: '{segment.get('text', '')[:50]}...' (overlap ratio: {overlap_ratio:.2f})")
                        break
            
            if not has_significant_overlap:
                result.append(segment)
        
        return result
    
    def _select_best_from_groups(self, segments: List[Dict], time_window: float = 0.5) -> List[Dict]:
        """Group segments by time windows and select the best from each group."""
        if not segments:
            return []
        
        # Sort by start time
        sorted_segments = sorted(segments, key=lambda x: float(x.get('start', 0)))
        
        result = []
        current_window_start = None
        current_window_segments = []
        
        for segment in sorted_segments:
            seg_start = float(segment.get('start', 0))
            
            # Start new window if needed
            if current_window_start is None or seg_start >= current_window_start + time_window:
                # Process previous window
                if current_window_segments:
                    best_segment = self._select_best_segment(current_window_segments)
                    result.append(best_segment)
                
                # Start new window
                current_window_start = seg_start
                current_window_segments = [segment]
            else:
                # Add to current window
                current_window_segments.append(segment)
        
        # Process final window
        if current_window_segments:
            best_segment = self._select_best_segment(current_window_segments)
            result.append(best_segment)
        
        return result
    
    def _select_best_segment(self, segments: List[Dict]) -> Dict:
        """Select the best segment from a group based on quality indicators."""
        if len(segments) == 1:
            return segments[0]
        
        # Score each segment based on multiple factors
        for seg in segments:
            score = 0
            
            # Longer text segments are generally better
            text = seg.get('text', '')
            text_length = len(text.strip())
            score += text_length * 0.3
            
            # Segments with word-level timing are better
            words = seg.get('words', [])
            if words:
                # Check if words have valid timing (not all same timestamp)
                timestamps = [float(w.get('start', 0)) for w in words if w.get('start') is not None]
                unique_timestamps = len(set(round(t, 1) for t in timestamps))
                
                if unique_timestamps > 1:
                    score += 50  # Good word-level timing
                elif len(words) > 0:
                    score += 20  # Has words but poor timing
                
                # Reward segments where words span the full segment duration
                if timestamps:
                    word_span = max(timestamps) - min(timestamps)
                    seg_duration = float(seg.get('end', 0)) - float(seg.get('start', 0))
                    if seg_duration > 0 and word_span > seg_duration * 0.8:
                        score += 30  # Words span most of segment
            
            # Penalize segments with repetitive or garbled text
            words_in_text = text.lower().split()
            if len(words_in_text) > 3:
                unique_words = len(set(words_in_text))
                repetition_ratio = unique_words / len(words_in_text)
                if repetition_ratio < 0.7:  # More than 30% repeated words
                    score -= 20
            
            # Slight preference for earlier segments (original transcription)
            start_time = float(seg.get('start', 0))
            score -= start_time * 0.001  # Very small penalty for later start
            
            seg['_selection_score'] = score
        
        # Return the highest scoring segment
        best_segment = max(segments, key=lambda s: s.get('_selection_score', 0))
        
        # Log the selection if multiple candidates
        if len(segments) > 1:
            logger.debug(f"[{self.content_id}] Selected best from {len(segments)} segments: '{best_segment.get('text', '')[:50]}...' (score: {best_segment.get('_selection_score', 0):.1f})")
        
        # Clean up score
        del best_segment['_selection_score']
        
        return best_segment
    
    def initialize_from_transcript_data(self, transcript_segments: List[Dict]) -> int:
        """Initialize from transcript segment data."""
        logger.debug(f"[{self.content_id}] Initializing WordTable from {len(transcript_segments)} segments")
        
        # First, clean duplicate segments before processing
        cleaned_segments = self._clean_duplicate_segments(transcript_segments)
        logger.debug(f"[{self.content_id}] After duplicate cleaning: {len(cleaned_segments)} segments (removed {len(transcript_segments) - len(cleaned_segments)} duplicates)")
        
        words_data = []
        total_words = 0
        
        for segment_idx, segment in enumerate(cleaned_segments):
            segment_start = segment.get('start', 0)
            segment_end = segment.get('end', 0)
            segment_text = segment.get('text', '')
            words = segment.get('words', [])
            
            if words:
                # Use word-level timing if available - merge split contractions first
                merged_words = self._merge_split_contractions(words)
                
                for word_idx, word_data in enumerate(merged_words):
                    word_id = str(uuid.uuid4())
                    
                    word_entry = {
                        'word_id': word_id,
                        'text': word_data.get('word', '').strip(),
                        'start': float(word_data.get('start', segment_start)),
                        'end': float(word_data.get('end', segment_end)),
                        'confidence': float(word_data.get('confidence', 1.0)),
                        'segment_index': segment_idx,
                        'word_index': word_idx,
                        'sentence_id': None,  # Will be populated after all words are created
                        'speaker_current': 'PENDING_CATEGORY_ASSIGNMENT',  # Will be updated after sentence/segment analysis
                        'assignment_confidence': 0.0,  # Initialize confidence
                        'resolution_method': 'none',
                        'processing_status': 'initialized',
                        'metadata': {
                            'original_segment_text': segment_text,
                            'segment_start': segment_start,
                            'segment_end': segment_end,
                            'merged_from_split': word_data.get('_merged_from_split', False)
                        },
                        'assignment_history': [{
                            'stage': 'stage3_initialization',
                            'timestamp': time.time(),
                            'speaker': 'UNKNOWN',
                            'method': 'initialization',
                            'confidence': 0.0,
                            'reason': 'Word table initialization'
                        }],
                        'has_good_grammar': False,  # Will be set during sentence ID assignment
                        'has_multiple_speakers': False,  # Will be set during sentence ID assignment
                        'segment_has_good_grammar': False,  # Will be set during segment analysis
                        'segment_has_multiple_speakers': False,  # Will be set during segment analysis
                        'segment_within_1s_of_non_majority_speaker': False  # Will be set during segment analysis
                    }
                    
                    words_data.append(word_entry)
                    total_words += 1
            else:
                # No word-level data, estimate from segment text
                if segment_text.strip():
                    words_list = segment_text.strip().split()
                    segment_duration = segment_end - segment_start
                    
                    for word_idx, word_text in enumerate(words_list):
                        word_id = str(uuid.uuid4())
                        
                        # Estimate word timing
                        word_start = segment_start + (word_idx / len(words_list)) * segment_duration
                        word_end = segment_start + ((word_idx + 1) / len(words_list)) * segment_duration
                        
                        word_entry = {
                            'word_id': word_id,
                            'text': word_text,
                            'start': word_start,
                            'end': word_end,
                            'confidence': 1.0,  # Default confidence for estimated words
                            'segment_index': segment_idx,
                            'word_index': word_idx,
                            'sentence_id': None,  # Will be populated after all words are created
                            'speaker_current': 'PENDING_CATEGORY_ASSIGNMENT',  # Will be updated after sentence/segment analysis
                            'assignment_confidence': 0.0,  # Initialize confidence
                            'resolution_method': 'none',
                            'processing_status': 'estimated_timing',
                            'metadata': {
                                'original_segment_text': segment_text,
                                'segment_start': segment_start,
                                'segment_end': segment_end,
                                'timing_estimated': True
                            },
                            'assignment_history': [{
                                'stage': 'stage3_initialization',
                                'timestamp': time.time(),
                                'speaker': 'PENDING_CATEGORY_ASSIGNMENT',
                                'method': 'initialization',
                                'confidence': 0.0,
                                'reason': 'Word table initialization (estimated timing)'
                            }],
                            'has_good_grammar': False,  # Will be set during sentence ID assignment
                            'has_multiple_speakers': False,  # Will be set during sentence ID assignment
                            'segment_has_good_grammar': False,  # Will be set during segment analysis
                            'segment_has_multiple_speakers': False,  # Will be set during segment analysis
                            'segment_within_1s_of_non_majority_speaker': False  # Will be set during segment analysis
                        }
                        
                        words_data.append(word_entry)
                        total_words += 1
        
        # Create DataFrame
        if words_data:
            self.df = pd.DataFrame(words_data)
            # Sort by start time
            self.df = self.df.sort_values('start').reset_index(drop=True)
            
            # Precompute segment speaker overlaps for efficiency
            self._precompute_segment_speaker_overlaps()
            
            # Assign sentence IDs based on punctuation and timing
            self._assign_sentence_ids()
        
        logger.debug(f"[{self.content_id}] WordTable initialized with {total_words} words")
        return total_words

    def _has_good_grammar(self, text: str) -> bool:
        """
        Check if text has good grammar with strict criteria.
        
        Good grammar requires EITHER:
        - A capitalized word anywhere in the text, OR
        - Any punctuation (!.,?) anywhere in the text
        
        Anything else is considered bad grammar.
        
        Args:
            text: Text to check
            
        Returns:
            True if has good grammar, False otherwise (bad grammar)
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
    
    def _has_multiple_speakers(self, words_df: pd.DataFrame) -> bool:
        """
        Check if a segment/word group has multiple speakers based on diarization overlap.
        Uses precomputed cache for efficiency.
        
        A segment is considered multi-speaker if no single speaker has ≥90% overlap.
        This threshold helps identify segments where speaker transitions occur or
        where multiple speakers are present.
        
        Args:
            words_df: DataFrame of words to check
            
        Returns:
            True if potentially has multiple speakers, False if single speaker
        """
        if len(words_df) == 0:
            return False
        
        # If no diarization data available, assume single speaker
        if not self.diarization_segments:
            return False
        
        # Get the segment index from the first word (all words in group should be same segment for segment-level analysis)
        segment_index = words_df['segment_index'].iloc[0]
        
        # Check cache first
        if segment_index in self._segment_speaker_cache:
            return self._segment_speaker_cache[segment_index]
        
        # Fallback: compute on demand (should be rare if precompute was called)
        segment_start = words_df['start'].min()
        segment_end = words_df['end'].max()
        segment_duration = segment_end - segment_start
        
        if segment_duration <= 0:
            self._segment_speaker_cache[segment_index] = False
            return False
        
        # Calculate overlap with each diarization speaker
        speaker_overlaps = {}
        
        for dia_seg in self.diarization_segments:
            dia_start = dia_seg.get('start', 0)
            dia_end = dia_seg.get('end', 0)
            dia_speaker = dia_seg.get('speaker', 'UNKNOWN')
            
            # Check for overlap
            overlap_start = max(segment_start, dia_start)
            overlap_end = min(segment_end, dia_end)
            
            if overlap_start < overlap_end:
                overlap_duration = overlap_end - overlap_start
                overlap_percentage = overlap_duration / segment_duration
                
                if dia_speaker not in speaker_overlaps:
                    speaker_overlaps[dia_speaker] = 0
                speaker_overlaps[dia_speaker] += overlap_percentage
        
        # Check if we have multiple unique speakers
        unique_speakers = len(speaker_overlaps)
        
        if unique_speakers <= 1:
            # Only one speaker (or no speakers) - definitely single speaker
            has_multiple = False
        elif unique_speakers > 1:
            # Multiple speakers - check if any single speaker dominates with 90%+ overlap
            total_overlap = sum(speaker_overlaps.values())
            max_overlap = max(speaker_overlaps.values())
            
            # Debug logging for segments with high overlap
            if total_overlap > 1.1:  # More than 110% total overlap suggests overlap calculation issues
                logger.debug(f"[{self.content_id}] Segment {segment_index}: High total overlap {total_overlap:.1%} - speaker overlaps: {speaker_overlaps}")
            
            if max_overlap >= 0.9:
                # Single speaker dominates (90%+ coverage) even though multiple speakers present
                has_multiple = False
            else:
                # Multiple speakers with no dominant speaker
                has_multiple = True
                # Log details for debugging (only first time for this segment)
                if segment_index not in self._segment_speaker_cache:
                    dominant_speaker = max(speaker_overlaps.items(), key=lambda x: x[1])
                    logger.debug(f"[{self.content_id}] Segment {segment_index} ({segment_start:.1f}-{segment_end:.1f}s) marked as multi-speaker: "
                               f"dominant={dominant_speaker[0]} with {dominant_speaker[1]:.1%} overlap, "
                               f"all speakers={dict(sorted(speaker_overlaps.items(), key=lambda x: -x[1]))}")
        else:
            # No speakers at all - treat as single speaker
            has_multiple = False
        
        # Cache the result
        self._segment_speaker_cache[segment_index] = has_multiple
        return has_multiple

    def _check_1s_of_non_majority_speaker(self, words_df: pd.DataFrame) -> bool:
        """
        Check if a segment is within 1.0s of a DIFFERENT speaker based on diarization.
        
        This determines if a whisper segment has a different speaker within 1.0s of its edges,
        indicating potential speaker boundary uncertainty that may require re-transcription.
        
        A segment is considered UNSAFE only if:
        1. There's a different speaker within 1.0s before the segment start, OR
        2. There's a different speaker within 1.0s after the segment end
        
        Args:
            words_df: DataFrame of words to check
            
        Returns:
            True if segment is within 1.0s of a different speaker, False otherwise
        """
        if len(words_df) == 0:
            return False
        
        # If no diarization data available, assume no adjacency issues
        if not self.diarization_segments:
            return False
        
        # Get segment boundaries
        segment_start = words_df['start'].min()
        segment_end = words_df['end'].max()
        segment_duration = segment_end - segment_start
        
        if segment_duration <= 0:
            return False
        
        # Calculate overlap with each diarization speaker
        speaker_overlaps = {}
        
        for dia_seg in self.diarization_segments:
            dia_start = dia_seg.get('start', 0)
            dia_end = dia_seg.get('end', 0)
            dia_speaker = dia_seg.get('speaker', 'UNKNOWN')
            
            # Check for overlap
            overlap_start = max(segment_start, dia_start)
            overlap_end = min(segment_end, dia_end)
            
            if overlap_start < overlap_end:
                overlap_duration = overlap_end - overlap_start
                
                if dia_speaker not in speaker_overlaps:
                    speaker_overlaps[dia_speaker] = 0
                speaker_overlaps[dia_speaker] += overlap_duration
        
        if not speaker_overlaps:
            return False
        
        # Find majority speaker (speaker with most overlap)
        majority_speaker = max(speaker_overlaps.items(), key=lambda x: x[1])[0]
        
        # Find the speakers immediately before and after this segment
        speaker_before = None
        speaker_after = None
        
        for dia_seg in self.diarization_segments:
            dia_start = dia_seg.get('start', 0)
            dia_end = dia_seg.get('end', 0)
            dia_speaker = dia_seg.get('speaker', 'UNKNOWN')
            
            # Check for speaker before segment (within 1.0s)
            if dia_end <= segment_start and (segment_start - dia_end) <= 1.0:
                # This diarization ends before our segment starts (within 1.0s)
                if speaker_before is None or dia_end > speaker_before[1]:
                    speaker_before = (dia_speaker, dia_end)
            
            # Check for speaker after segment (within 1.0s)
            if dia_start >= segment_end and (dia_start - segment_end) <= 1.0:
                # This diarization starts after our segment ends (within 1.0s)
                if speaker_after is None or dia_start < speaker_after[1]:
                    speaker_after = (dia_speaker, dia_start)
        
        # Check if we have different speakers adjacent to the segment
        is_unsafe = False
        
        if speaker_before and speaker_before[0] != majority_speaker:
            logger.debug(f"[{self.content_id}] Segment has different speaker before: {speaker_before[0]} != {majority_speaker}")
            is_unsafe = True
        
        if speaker_after and speaker_after[0] != majority_speaker:
            logger.debug(f"[{self.content_id}] Segment has different speaker after: {speaker_after[0]} != {majority_speaker}")
            is_unsafe = True
        
        return is_unsafe

    def _assign_sentence_ids(self) -> None:
        """
        Assign sentence IDs to words based on enhanced punctuation from Stage 3 and timing patterns.
        This leverages the LLM-enhanced punctuation to create better sentence boundaries.
        Also assigns grammar quality and multi-speaker flags.
        """
        if len(self.df) == 0:
            return
        
        # Add new columns for grammar and multi-speaker detection
        if 'has_good_grammar' not in self.df.columns:
            self.df['has_good_grammar'] = False
        if 'has_multiple_speakers' not in self.df.columns:
            self.df['has_multiple_speakers'] = False
        
        current_sentence_id = 0
        # Enhanced punctuation detection (covers more cases thanks to Stage 3)
        sentence_end_punctuation = {'.', '!', '?', '...', '."', '!"', '?"', ".'", "!'", "?'"}
        timing_gap_threshold = 1.5  # Slightly longer threshold since we have better punctuation
        
        # Track sentence context for better boundary detection
        words_in_current_sentence = []
        sentence_start_idx = 0
        
        for idx in range(len(self.df)):
            # Assign current sentence ID to this word
            self.df.at[idx, 'sentence_id'] = current_sentence_id
            
            word_text = self.df.at[idx, 'text'].strip()
            words_in_current_sentence.append(word_text)
            
            # Check for sentence ending conditions (improved with Stage 3 punctuation)
            is_sentence_end = False
            
            # 1. Enhanced punctuation-based ending (primary method)
            if word_text.endswith(tuple(sentence_end_punctuation)):
                is_sentence_end = True
                logger.debug(f"[{self.content_id}] Sentence end detected by punctuation: '{word_text}' at sentence {current_sentence_id}")
            
            # 2. Look ahead for capitalization patterns (Stage 3 should have improved this)
            if not is_sentence_end and idx < len(self.df) - 1:
                next_word = self.df.at[idx + 1, 'text'].strip()
                # If next word starts with capital and current has some punctuation, likely sentence boundary
                if (next_word and next_word[0].isupper() and 
                    any(p in word_text for p in ['.', '!', '?', ',', ';', ':'])):
                    is_sentence_end = True
                    logger.debug(f"[{self.content_id}] Sentence end detected by capitalization pattern: '{word_text}' -> '{next_word}' at sentence {current_sentence_id}")
            
            # 3. Large timing gap to next word (speaker change or pause)
            if not is_sentence_end and idx < len(self.df) - 1:
                next_word_start = self.df.at[idx + 1, 'start']
                current_word_end = self.df.at[idx, 'end']
                time_gap = next_word_start - current_word_end
                
                if time_gap > timing_gap_threshold:
                    is_sentence_end = True
                    logger.debug(f"[{self.content_id}] Sentence end detected by timing gap: {time_gap:.2f}s at sentence {current_sentence_id}")
            
            # 4. End of Whisper segment (but be more conservative now that we have better punctuation)
            if not is_sentence_end and idx < len(self.df) - 1:
                next_segment_index = self.df.at[idx + 1, 'segment_index']
                current_segment_index = self.df.at[idx, 'segment_index']
                
                if next_segment_index != current_segment_index:
                    # Only end sentence at segment boundary if we haven't had a recent sentence end
                    if len(words_in_current_sentence) > 3:  # Don't break very short sentences
                        is_sentence_end = True
                        logger.debug(f"[{self.content_id}] Sentence end detected by segment boundary at sentence {current_sentence_id}")
            
            # 5. Very long sentences (safety valve)
            if not is_sentence_end and len(words_in_current_sentence) > 25:
                # Look for natural break points in long sentences
                if any(p in word_text for p in [',', ';', ':', '--']):
                    is_sentence_end = True
                    logger.debug(f"[{self.content_id}] Long sentence split at natural break: '{word_text}' at sentence {current_sentence_id}")
            
            # 6. Last word overall
            if idx == len(self.df) - 1:
                is_sentence_end = True
            
            # If sentence ended, start new sentence
            if is_sentence_end:
                # Process the completed sentence
                sentence_text = ' '.join(words_in_current_sentence)
                
                # Check grammar quality for the sentence
                has_good_grammar = self._has_good_grammar(sentence_text)
                
                # Check for multiple speakers in the sentence
                sentence_words_df = self.df.iloc[sentence_start_idx:idx+1]
                has_multiple_speakers = self._has_multiple_speakers(sentence_words_df)
                
                # Apply flags to all words in the sentence
                for sentence_idx in range(sentence_start_idx, idx+1):
                    self.df.at[sentence_idx, 'has_good_grammar'] = has_good_grammar
                    self.df.at[sentence_idx, 'has_multiple_speakers'] = has_multiple_speakers
                
                logger.debug(f"[{self.content_id}] Completed sentence {current_sentence_id}: '{sentence_text[:100]}{'...' if len(sentence_text) > 100 else ''}' (good_grammar={has_good_grammar}, multi_speaker={has_multiple_speakers})")
                
                current_sentence_id += 1
                words_in_current_sentence = []
                sentence_start_idx = idx + 1
        
        # Process last sentence if any words remain
        if words_in_current_sentence:
            sentence_text = ' '.join(words_in_current_sentence)
            has_good_grammar = self._has_good_grammar(sentence_text)
            sentence_words_df = self.df.iloc[sentence_start_idx:]
            has_multiple_speakers = self._has_multiple_speakers(sentence_words_df)
            
            for sentence_idx in range(sentence_start_idx, len(self.df)):
                self.df.at[sentence_idx, 'has_good_grammar'] = has_good_grammar
                self.df.at[sentence_idx, 'has_multiple_speakers'] = has_multiple_speakers
        
        # Also check grammar and multi-speaker at the segment level
        self._check_segment_attributes()
        
        # After all analysis is complete, assign category-based speaker types
        self._assign_category_based_speaker_types()
        
        logger.info(f"[{self.content_id}] Assigned {current_sentence_id} sentence IDs to words using enhanced punctuation from Stage 3")
        
        # Log grammar statistics
        good_grammar_count = self.df['has_good_grammar'].sum()
        multi_speaker_count = self.df['has_multiple_speakers'].sum()
        logger.info(f"[{self.content_id}] Grammar analysis: {good_grammar_count}/{len(self.df)} words have good grammar, {multi_speaker_count}/{len(self.df)} words potentially have multiple speakers")

    def _check_segment_attributes(self) -> None:
        """
        Check grammar and multi-speaker attributes at the segment level.
        This provides segment-level analysis in addition to sentence-level.
        """
        if len(self.df) == 0:
            return
            
        # Add segment-level columns if they don't exist
        if 'segment_has_good_grammar' not in self.df.columns:
            self.df['segment_has_good_grammar'] = False
        if 'segment_has_multiple_speakers' not in self.df.columns:
            self.df['segment_has_multiple_speakers'] = False
        if 'segment_within_1s_of_non_majority_speaker' not in self.df.columns:
            self.df['segment_within_1s_of_non_majority_speaker'] = False
        
        # Process each segment
        segment_indices = self.df['segment_index'].unique()
        
        for seg_idx in segment_indices:
            # Get all words in this segment
            segment_mask = self.df['segment_index'] == seg_idx
            segment_words = self.df[segment_mask]
            
            if len(segment_words) == 0:
                continue
                
            # Get segment text
            segment_text = ' '.join(segment_words['text'].tolist())
            
            # Check grammar quality for the segment
            segment_has_good_grammar = self._has_good_grammar(segment_text)
            
            # Check for multiple speakers in the segment
            segment_has_multiple_speakers = self._has_multiple_speakers(segment_words)
            
            # Check if segment is within 1.0s of non-majority speaker
            segment_within_1s_of_non_majority = self._check_1s_of_non_majority_speaker(segment_words)
            
            # Apply flags to all words in the segment
            self.df.loc[segment_mask, 'segment_has_good_grammar'] = segment_has_good_grammar
            self.df.loc[segment_mask, 'segment_has_multiple_speakers'] = segment_has_multiple_speakers
            self.df.loc[segment_mask, 'segment_within_1s_of_non_majority_speaker'] = segment_within_1s_of_non_majority
        
        # Log segment statistics
        segments_with_good_grammar = self.df.groupby('segment_index')['segment_has_good_grammar'].first().sum()
        segments_with_multi_speaker = self.df.groupby('segment_index')['segment_has_multiple_speakers'].first().sum()
        segments_within_1s_non_majority = self.df.groupby('segment_index')['segment_within_1s_of_non_majority_speaker'].first().sum()
        total_segments = len(segment_indices)
        
        logger.info(f"[{self.content_id}] Segment analysis: {segments_with_good_grammar}/{total_segments} segments have good grammar, {segments_with_multi_speaker}/{total_segments} segments potentially have multiple speakers")
        logger.info(f"[{self.content_id}] Edge analysis: {segments_within_1s_non_majority}/{total_segments} segments are within 1.0s of non-majority speaker")
        
        # Generate and log the 2x2 category breakdown
        self._log_grammar_speaker_breakdown()

    def _assign_category_based_speaker_types(self) -> None:
        """
        Assign category-based speaker types based on grammar and speaker analysis.
        
        Categories:
        - BAD_GRAMMAR_SINGLE: Poor grammar, single speaker
        - BAD_GRAMMAR_MULTI: Poor grammar, multiple speakers  
        - GOOD_GRAMMAR_SINGLE: Good grammar, single speaker
        - GOOD_GRAMMAR_MULTI: Good grammar, multiple speakers
        """
        if len(self.df) == 0:
            return
        
        logger.debug(f"[{self.content_id}] Assigning category-based speaker types based on grammar and speaker analysis")
        
        # Track category assignments
        category_counts = {
            'BAD_GRAMMAR_SINGLE': 0,
            'BAD_GRAMMAR_MULTI': 0, 
            'GOOD_GRAMMAR_SINGLE': 0,
            'GOOD_GRAMMAR_MULTI': 0
        }
        
        # Assign speaker types based on segment-level attributes
        for idx in self.df.index:
            has_good_grammar = self.df.at[idx, 'segment_has_good_grammar']
            has_multiple_speakers = self.df.at[idx, 'segment_has_multiple_speakers']
            is_unsafe = self.df.at[idx, 'segment_within_1s_of_non_majority_speaker']
            
            # Determine category
            # IMPORTANT: Treat unsafe segments as multi-speaker for Stage 9 processing
            if is_unsafe:
                # Unsafe segments should be treated as multi-speaker regardless of actual speaker count
                if has_good_grammar:
                    speaker_type = 'GOOD_GRAMMAR_MULTI'
                else:
                    speaker_type = 'BAD_GRAMMAR_MULTI'
                    
                # Log why it's unsafe (only for first few words of segment to avoid spam)
                if idx == self.df[self.df['segment_index'] == self.df.at[idx, 'segment_index']].index[0]:
                    logger.debug(f"[{self.content_id}] Segment {self.df.at[idx, 'segment_index']} marked as {speaker_type} due to unsafe=True")
            else:
                # Safe segments use normal categorization
                if has_good_grammar and not has_multiple_speakers:
                    speaker_type = 'GOOD_GRAMMAR_SINGLE'
                elif has_good_grammar and has_multiple_speakers:
                    speaker_type = 'GOOD_GRAMMAR_MULTI'
                    # Log why it's multi (only for first few words of segment)
                    if idx == self.df[self.df['segment_index'] == self.df.at[idx, 'segment_index']].index[0]:
                        logger.debug(f"[{self.content_id}] Segment {self.df.at[idx, 'segment_index']} marked as GOOD_GRAMMAR_MULTI due to has_multiple_speakers=True")
                elif not has_good_grammar and not has_multiple_speakers:
                    speaker_type = 'BAD_GRAMMAR_SINGLE'
                else:  # not has_good_grammar and has_multiple_speakers
                    speaker_type = 'BAD_GRAMMAR_MULTI'
            
            # Update the word
            self.df.at[idx, 'speaker_current'] = speaker_type
            self.df.at[idx, 'resolution_method'] = 'stage3_category_assignment'
            self.df.at[idx, 'processing_status'] = 'category_assigned'
            
            # Update assignment history
            current_history = self.df.at[idx, 'assignment_history']
            if not isinstance(current_history, list):
                current_history = []
            
            assignment_entry = {
                'stage': 'stage3_category_assignment',
                'timestamp': time.time(),
                'speaker': speaker_type,
                'method': 'grammar_speaker_analysis',
                'confidence': 1.0,  # High confidence in categorization
                'reason': f'Category assignment based on grammar={has_good_grammar}, multi_speaker={has_multiple_speakers}, unsafe={is_unsafe}'
            }
            current_history.append(assignment_entry)
            self.df.at[idx, 'assignment_history'] = current_history
            
            # Update metadata
            current_metadata = self.df.at[idx, 'metadata']
            if isinstance(current_metadata, dict):
                current_metadata['category_assignment'] = {
                    'category': speaker_type,
                    'has_good_grammar': has_good_grammar,
                    'has_multiple_speakers': has_multiple_speakers,
                    'is_unsafe': is_unsafe,
                    'assignment_timestamp': time.time()
                }
            else:
                self.df.at[idx, 'metadata'] = {
                    'category_assignment': {
                        'category': speaker_type,
                        'has_good_grammar': has_good_grammar,
                        'has_multiple_speakers': has_multiple_speakers,
                        'is_unsafe': is_unsafe,
                        'assignment_timestamp': time.time()
                    }
                }
            
            category_counts[speaker_type] += 1
        
        # Log category distribution
        total_words = sum(category_counts.values())
        logger.info(f"[{self.content_id}] Category-based speaker type assignments:")
        for category, count in category_counts.items():
            percentage = (count / total_words * 100) if total_words > 0 else 0
            logger.info(f"[{self.content_id}]   - {category}: {count} words ({percentage:.1f}%)")
        
        logger.info(f"[{self.content_id}] All {total_words} words assigned to category-based speaker types")

    def _log_grammar_speaker_breakdown(self) -> None:
        """
        Log a breakdown of segments and words by grammar quality and speaker count.
        Shows a 2x2 matrix: (good/bad grammar) × (single/multi speaker)
        """
        if len(self.df) == 0:
            return
        
        # Segment-level breakdown
        segment_categories = {
            'good_grammar_single_speaker': 0,
            'good_grammar_multi_speaker': 0,
            'bad_grammar_single_speaker': 0,
            'bad_grammar_multi_speaker': 0
        }
        
        # Word-level breakdown
        word_categories = {
            'good_grammar_single_speaker': 0,
            'good_grammar_multi_speaker': 0,
            'bad_grammar_single_speaker': 0,
            'bad_grammar_multi_speaker': 0
        }
        
        # Process each segment
        segment_groups = self.df.groupby('segment_index')
        
        for seg_idx, seg_df in segment_groups:
            # Get segment attributes (should be consistent for all words in segment)
            has_good_grammar = seg_df['segment_has_good_grammar'].iloc[0]
            has_multi_speaker = seg_df['segment_has_multiple_speakers'].iloc[0]
            
            # Categorize segment
            if has_good_grammar and not has_multi_speaker:
                segment_categories['good_grammar_single_speaker'] += 1
            elif has_good_grammar and has_multi_speaker:
                segment_categories['good_grammar_multi_speaker'] += 1
            elif not has_good_grammar and not has_multi_speaker:
                segment_categories['bad_grammar_single_speaker'] += 1
            else:  # not has_good_grammar and has_multi_speaker
                segment_categories['bad_grammar_multi_speaker'] += 1
            
            # Count words in this category
            word_count = len(seg_df)
            if has_good_grammar and not has_multi_speaker:
                word_categories['good_grammar_single_speaker'] += word_count
            elif has_good_grammar and has_multi_speaker:
                word_categories['good_grammar_multi_speaker'] += word_count
            elif not has_good_grammar and not has_multi_speaker:
                word_categories['bad_grammar_single_speaker'] += word_count
            else:
                word_categories['bad_grammar_multi_speaker'] += word_count
        
        # Calculate totals
        total_segments = sum(segment_categories.values())
        total_words = sum(word_categories.values())
        
        # Log the breakdown
        logger.info(f"[{self.content_id}] Grammar × Speaker Category Breakdown:")
        logger.info(f"[{self.content_id}] ┌─────────────────────┬──────────────────┬──────────────────┐")
        logger.info(f"[{self.content_id}] │                     │ Single Speaker   │ Multi Speaker    │")
        logger.info(f"[{self.content_id}] ├─────────────────────┼──────────────────┼──────────────────┤")
        
        # Good grammar row
        logger.info(f"[{self.content_id}] │ Good Grammar        │ {segment_categories['good_grammar_single_speaker']:>3} segments     │ {segment_categories['good_grammar_multi_speaker']:>3} segments     │")
        logger.info(f"[{self.content_id}] │                     │ {word_categories['good_grammar_single_speaker']:>5} words     │ {word_categories['good_grammar_multi_speaker']:>5} words     │")
        
        logger.info(f"[{self.content_id}] ├─────────────────────┼──────────────────┼──────────────────┤")
        
        # Bad grammar row
        logger.info(f"[{self.content_id}] │ Bad Grammar         │ {segment_categories['bad_grammar_single_speaker']:>3} segments     │ {segment_categories['bad_grammar_multi_speaker']:>3} segments     │")
        logger.info(f"[{self.content_id}] │                     │ {word_categories['bad_grammar_single_speaker']:>5} words     │ {word_categories['bad_grammar_multi_speaker']:>5} words     │")
        
        logger.info(f"[{self.content_id}] └─────────────────────┴──────────────────┴──────────────────┘")
        
        # Log percentages
        if total_segments > 0:
            logger.info(f"[{self.content_id}] Segment percentages:")
            logger.info(f"[{self.content_id}]   - Good grammar + Single speaker: {segment_categories['good_grammar_single_speaker']/total_segments*100:.1f}%")
            logger.info(f"[{self.content_id}]   - Good grammar + Multi speaker: {segment_categories['good_grammar_multi_speaker']/total_segments*100:.1f}%")
            logger.info(f"[{self.content_id}]   - Bad grammar + Single speaker: {segment_categories['bad_grammar_single_speaker']/total_segments*100:.1f}%")
            logger.info(f"[{self.content_id}]   - Bad grammar + Multi speaker: {segment_categories['bad_grammar_multi_speaker']/total_segments*100:.1f}%")

    def _get_grammar_speaker_categories(self) -> Dict[str, Any]:
        """
        Get the grammar × speaker category breakdown data.
        Returns dictionary with segment and word counts for each category.
        """
        if len(self.df) == 0:
            return {
                'segments': {
                    'good_grammar_single_speaker': 0,
                    'good_grammar_multi_speaker': 0,
                    'bad_grammar_single_speaker': 0,
                    'bad_grammar_multi_speaker': 0,
                    'total': 0
                },
                'words': {
                    'good_grammar_single_speaker': 0,
                    'good_grammar_multi_speaker': 0,
                    'bad_grammar_single_speaker': 0,
                    'bad_grammar_multi_speaker': 0,
                    'total': 0
                }
            }
        
        # Segment-level breakdown
        segment_categories = {
            'good_grammar_single_speaker': 0,
            'good_grammar_multi_speaker': 0,
            'bad_grammar_single_speaker': 0,
            'bad_grammar_multi_speaker': 0
        }
        
        # Word-level breakdown
        word_categories = {
            'good_grammar_single_speaker': 0,
            'good_grammar_multi_speaker': 0,
            'bad_grammar_single_speaker': 0,
            'bad_grammar_multi_speaker': 0
        }
        
        # Process each segment
        segment_groups = self.df.groupby('segment_index')
        
        for seg_idx, seg_df in segment_groups:
            # Get segment attributes
            has_good_grammar = seg_df['segment_has_good_grammar'].iloc[0]
            has_multi_speaker = seg_df['segment_has_multiple_speakers'].iloc[0]
            
            # Categorize segment
            if has_good_grammar and not has_multi_speaker:
                segment_categories['good_grammar_single_speaker'] += 1
            elif has_good_grammar and has_multi_speaker:
                segment_categories['good_grammar_multi_speaker'] += 1
            elif not has_good_grammar and not has_multi_speaker:
                segment_categories['bad_grammar_single_speaker'] += 1
            else:
                segment_categories['bad_grammar_multi_speaker'] += 1
            
            # Count words in this category
            word_count = len(seg_df)
            if has_good_grammar and not has_multi_speaker:
                word_categories['good_grammar_single_speaker'] += word_count
            elif has_good_grammar and has_multi_speaker:
                word_categories['good_grammar_multi_speaker'] += word_count
            elif not has_good_grammar and not has_multi_speaker:
                word_categories['bad_grammar_single_speaker'] += word_count
            else:
                word_categories['bad_grammar_multi_speaker'] += word_count
        
        # Add totals
        segment_categories['total'] = sum(v for k, v in segment_categories.items() if k != 'total')
        word_categories['total'] = sum(v for k, v in word_categories.items() if k != 'total')
        
        return {
            'segments': segment_categories,
            'words': word_categories
        }

    def get_words_by_time_range(self, start_time: float, end_time: float) -> pd.DataFrame:
        """Get words within a specific time range."""
        mask = (self.df['start'] >= start_time) & (self.df['end'] <= end_time)
        return self.df[mask].copy()
    
    def get_words_by_sentence_id(self, sentence_id: int) -> pd.DataFrame:
        """Get all words belonging to a specific sentence."""
        return self.df[self.df['sentence_id'] == sentence_id].copy()
    
    def get_sentences_with_unknown_words(self) -> List[int]:
        """Get list of sentence IDs that contain UNKNOWN words or category-based assignments."""
        category_types = ['UNKNOWN', 'BAD_GRAMMAR_SINGLE', 'BAD_GRAMMAR_MULTI', 'GOOD_GRAMMAR_SINGLE', 'GOOD_GRAMMAR_MULTI']
        unknown_mask = (self.df['speaker_current'].isin(category_types)) | (self.df['speaker_current'].isna())
        unknown_sentences = self.df[unknown_mask]['sentence_id'].unique().tolist()
        return sorted(unknown_sentences)
    
    def get_sentence_info(self, sentence_id: int) -> Dict:
        """Get comprehensive information about a sentence."""
        sentence_words = self.get_words_by_sentence_id(sentence_id)
        if len(sentence_words) == 0:
            return {}
        
        return {
            'sentence_id': sentence_id,
            'word_count': len(sentence_words),
            'start_time': sentence_words['start'].min(),
            'end_time': sentence_words['end'].max(),
            'duration': sentence_words['end'].max() - sentence_words['start'].min(),
            'text': ' '.join(sentence_words['text'].tolist()),
            'speaker_counts': sentence_words['speaker_current'].value_counts().to_dict(),
            'unknown_word_count': sentence_words['speaker_current'].isin(['UNKNOWN', 'BAD_GRAMMAR_SINGLE', 'BAD_GRAMMAR_MULTI', 'GOOD_GRAMMAR_SINGLE', 'GOOD_GRAMMAR_MULTI']).sum(),
            'segment_indices': sentence_words['segment_index'].unique().tolist()
        }
    
    def get_unknown_words(self) -> pd.DataFrame:
        """Get all words that still have UNKNOWN or category-based speaker assignment."""
        category_types = ['UNKNOWN', 'BAD_GRAMMAR_SINGLE', 'BAD_GRAMMAR_MULTI', 'GOOD_GRAMMAR_SINGLE', 'GOOD_GRAMMAR_MULTI']
        return self.df[self.df['speaker_current'].isin(category_types)].copy()
    
    def get_resolved_words(self) -> pd.DataFrame:
        """Get all words that have been assigned to actual speakers (not category-based assignments)."""
        category_types = ['UNKNOWN', 'BAD_GRAMMAR_SINGLE', 'BAD_GRAMMAR_MULTI', 'GOOD_GRAMMAR_SINGLE', 'GOOD_GRAMMAR_MULTI']
        return self.df[~self.df['speaker_current'].isin(category_types)].copy()

    def assign_speaker_to_words(self, word_ids: List[str], speaker: str, method: str, confidence: float = 1.0, stage: str = "unknown", reason: str = None) -> int:
        """Assign a speaker to specific words and track the assignment history."""
        if not word_ids:
            return 0
        
        # Find words to update
        mask = self.df['word_id'].isin(word_ids)
        words_to_update = mask.sum()
        
        if words_to_update == 0:
            logger.warning(f"[{self.content_id}] No matching words found for speaker assignment")
            return 0
        
        # Track previous assignments for changed words
        changed_words = []
        for idx in self.df[mask].index:
            previous_speaker = self.df.at[idx, 'speaker_current']
            if previous_speaker != speaker:
                changed_words.append({
                    'word_id': self.df.at[idx, 'word_id'],
                    'text': self.df.at[idx, 'text'],
                    'start': self.df.at[idx, 'start'],
                    'previous_speaker': previous_speaker,
                    'new_speaker': speaker
                })
        
        # Update the words
        self.df.loc[mask, 'speaker_current'] = speaker
        self.df.loc[mask, 'resolution_method'] = method
        self.df.loc[mask, 'processing_status'] = 'speaker_assigned'
        
        # Set confidence in the assignment_confidence column (add if it doesn't exist)
        if 'assignment_confidence' not in self.df.columns:
            self.df['assignment_confidence'] = 0.0
        self.df.loc[mask, 'assignment_confidence'] = confidence
        
        # Update metadata and assignment history
        for idx in self.df[mask].index:
            current_metadata = self.df.at[idx, 'metadata']
            if isinstance(current_metadata, dict):
                current_metadata['assignment_confidence'] = confidence
                current_metadata['assignment_method'] = method
            else:
                self.df.at[idx, 'metadata'] = {
                    'assignment_confidence': confidence,
                    'assignment_method': method
                }
            
            # Update assignment history
            current_history = self.df.at[idx, 'assignment_history']
            if not isinstance(current_history, list):
                current_history = []
            
            assignment_entry = {
                'stage': stage,
                'timestamp': time.time(),
                'speaker': speaker,
                'method': method,
                'confidence': confidence,
                'reason': reason or f"Speaker assignment via {method}"
            }
            current_history.append(assignment_entry)
            self.df.at[idx, 'assignment_history'] = current_history
        
        # Log assignment changes with detail
        if changed_words:
            logger.info(f"[{self.content_id}] [{stage}] Changed {len(changed_words)} word assignments:")
            for word in changed_words[:5]:  # Show first 5 changes
                logger.info(f"  [{word['start']:.2f}s] '{word['text']}' {word['previous_speaker']} -> {word['new_speaker']}")
            if len(changed_words) > 5:
                logger.info(f"  ... and {len(changed_words) - 5} more changes")
        
        total_assigned = words_to_update - len(changed_words)
        if total_assigned > 0:
            logger.info(f"[{self.content_id}] [{stage}] Assigned {total_assigned} new words to '{speaker}' using '{method}'")
        
        return words_to_update
    
    def assign_speaker_to_word_by_index(self, idx: int, speaker: str, method: str, confidence: float = 1.0, stage: str = "unknown", reason: str = None) -> bool:
        """Assign a speaker to a single word by DataFrame index and track the assignment history."""
        if idx not in self.df.index:
            return False
        
        # Get word info for logging
        word_info = self.df.loc[idx]
        previous_speaker = word_info['speaker_current']
        
        # Update the word
        self.df.at[idx, 'speaker_current'] = speaker
        self.df.at[idx, 'resolution_method'] = method
        self.df.at[idx, 'processing_status'] = 'speaker_assigned'
        
        # Set confidence
        if 'assignment_confidence' not in self.df.columns:
            self.df['assignment_confidence'] = 0.0
        self.df.at[idx, 'assignment_confidence'] = confidence
        
        # Update metadata
        current_metadata = self.df.at[idx, 'metadata']
        if isinstance(current_metadata, dict):
            current_metadata['assignment_confidence'] = confidence
            current_metadata['assignment_method'] = method
        else:
            current_metadata = {
                'assignment_confidence': confidence,
                'assignment_method': method
            }
            self.df.at[idx, 'metadata'] = current_metadata
        
        # Update assignment history
        current_history = self.df.at[idx, 'assignment_history']
        if not isinstance(current_history, list):
            current_history = []
        
        assignment_entry = {
            'stage': stage,
            'timestamp': time.time(),
            'speaker': speaker,
            'method': method,
            'confidence': confidence,
            'reason': reason or f"Speaker assignment via {method}"
        }
        current_history.append(assignment_entry)
        self.df.at[idx, 'assignment_history'] = current_history
        
        # Log if speaker changed
        if previous_speaker != speaker:
            logger.debug(f"[{self.content_id}] [{stage}] [{word_info['start']:.2f}s] '{word_info['text']}' {previous_speaker} -> {speaker} ({method})")
        
        return True
    
    def update_segment_with_retranscription(self, segment_index: int, retranscribed_words: List[Dict], 
                                           original_category: str = 'BAD_GRAMMAR_SINGLE',
                                           method_suffix: str = 'retranscribed',
                                           stage: str = 'unknown',
                                           remove_overlapping: bool = True) -> int:
        """
        Update a segment with re-transcribed words, optionally removing overlapping original words.
        
        This is a common operation in stages 5 and 9 when re-transcribing segments with bad grammar.
        
        Args:
            segment_index: The segment to update
            retranscribed_words: List of word dictionaries with keys: text, start, end, speaker, confidence
            original_category: Original speaker category to look for when removing (e.g., 'BAD_GRAMMAR_SINGLE')
            method_suffix: Suffix for the resolution method (e.g., 'retranscribed')
            stage: Stage name for tracking
            remove_overlapping: Whether to remove overlapping words with original_category
            
        Returns:
            Number of words added
        """
        if not retranscribed_words:
            return 0
            
        words_added = 0
        
        # Remove overlapping words if requested
        if remove_overlapping and retranscribed_words:
            retrans_start = min(w['start'] for w in retranscribed_words)
            retrans_end = max(w['end'] for w in retranscribed_words)
            
            # Find words to remove (original category words in the re-transcribed range)
            words_to_remove_mask = (
                (self.df['segment_index'] == segment_index) &
                (self.df['speaker_current'] == original_category) &
                (self.df['start'] >= retrans_start - 0.5) &
                (self.df['end'] <= retrans_end + 0.5)
            )
            
            removed_word_ids = self.df[words_to_remove_mask]['word_id'].tolist()
            if removed_word_ids:
                logger.debug(f"[{self.content_id}] Removing {len(removed_word_ids)} {original_category} words from segment {segment_index} that overlap with re-transcription")
                self.df = self.df[~self.df['word_id'].isin(removed_word_ids)]
        
        # Prepare new words for DataFrame
        new_words_data = []
        for i, word in enumerate(retranscribed_words):
            # Generate a unique word ID
            word_id = f"w_{segment_index}_{len(self.df)}_{int(word['start']*1000)}_{i}"
            
            # Get segment-level properties from existing words in this segment
            seg_mask = self.df['segment_index'] == segment_index
            if seg_mask.any():
                seg_has_good_grammar = self.df[seg_mask]['segment_has_good_grammar'].iloc[0]
                seg_has_multiple_speakers = self.df[seg_mask]['segment_has_multiple_speakers'].iloc[0]
                seg_within_1s = self.df[seg_mask]['segment_within_1s_of_non_majority_speaker'].iloc[0]
            else:
                seg_has_good_grammar = False
                seg_has_multiple_speakers = False
                seg_within_1s = False
            
            new_words_data.append({
                'word_id': word_id,
                'text': word.get('text', '').strip(),
                'start': word.get('start', 0),
                'end': word.get('end', 0),
                'segment_index': segment_index,
                'speaker_current': word.get('speaker', 'UNKNOWN'),
                'speaker_original': original_category,  # Track original category
                'resolution_method': f"{original_category.lower()}_{method_suffix}",
                'assignment_confidence': word.get('confidence', 0.8),
                'processing_status': 'speaker_assigned',
                'segment_has_good_grammar': seg_has_good_grammar,
                'segment_has_multiple_speakers': seg_has_multiple_speakers,
                'segment_within_1s_of_non_majority_speaker': seg_within_1s,
                'metadata': {
                    f'retranscribed_{stage}': True,
                    'original_category': original_category,
                    'assignment_confidence': word.get('confidence', 0.8),
                    'assignment_method': f"{original_category.lower()}_{method_suffix}",
                    **word.get('metadata', {})  # Include any additional metadata
                },
                'assignment_history': [{
                    'stage': stage,
                    'timestamp': time.time(),
                    'speaker': word.get('speaker', 'UNKNOWN'),
                    'method': f"{original_category.lower()}_{method_suffix}",
                    'confidence': word.get('confidence', 0.8),
                    'reason': word.get('reason', f"Re-transcribed in {stage}")
                }]
            })
            words_added += 1
        
        # Add new words to the DataFrame
        if new_words_data:
            new_words_df = pd.DataFrame(new_words_data)
            
            # Ensure all required columns exist in new_words_df
            for col in self.df.columns:
                if col not in new_words_df.columns:
                    new_words_df[col] = None
            
            self.df = pd.concat([self.df, new_words_df], ignore_index=True)
            self.df = self.df.sort_values('start').reset_index(drop=True)
            
            logger.info(f"[{self.content_id}] Updated segment {segment_index} with {len(new_words_data)} re-transcribed words")
        
        return words_added
    
    def extract_words_from_retranscription(self, retranscription_result: Dict, 
                                          default_speaker: str = 'UNKNOWN') -> List[Dict]:
        """
        Extract words from a re-transcription result, handling various formats.
        
        Args:
            retranscription_result: Result from re-transcription (may have 'data' wrapper)
            default_speaker: Default speaker to use if not specified
            
        Returns:
            List of word dictionaries with normalized format
        """
        words = []
        
        # Debug logging
        logger.debug(f"[{self.content_id}] extract_words_from_retranscription called with default_speaker={default_speaker}")
        logger.debug(f"[{self.content_id}] Result keys: {list(retranscription_result.keys())}")
        
        # Handle wrapped results (with 'data' key)
        if 'data' in retranscription_result:
            data = retranscription_result['data']
            logger.debug(f"[{self.content_id}] Found 'data' wrapper, data keys: {list(data.keys())}")
        else:
            data = retranscription_result
            logger.debug(f"[{self.content_id}] No 'data' wrapper, using result directly")
        
        # First try direct words array
        words_from_result = data.get('words', [])
        logger.debug(f"[{self.content_id}] Direct words count: {len(words_from_result)}")
        
        # If no words at top level, check in segments
        if not words_from_result:
            logger.debug(f"[{self.content_id}] No direct words, checking {len(data.get('segments', []))} segments")
            for i, segment in enumerate(data.get('segments', [])):
                segment_words = segment.get('words', [])
                logger.debug(f"[{self.content_id}] Segment {i} has {len(segment_words)} words")
                words_from_result.extend(segment_words)
        
        logger.debug(f"[{self.content_id}] Total words to process: {len(words_from_result)}")
        
        # Normalize word format
        for word_info in words_from_result:
            # Handle different word formats (word vs text key)
            word_text = word_info.get('word', word_info.get('text', '')).strip()
            if word_text:  # Only add non-empty words
                words.append({
                    'text': word_text,
                    'start': word_info.get('start', 0),
                    'end': word_info.get('end', 0),
                    'speaker': word_info.get('speaker', default_speaker),
                    'confidence': word_info.get('confidence', word_info.get('probability', 0.8))
                })
            else:
                logger.debug(f"[{self.content_id}] Skipping empty word: {word_info}")
        
        logger.debug(f"[{self.content_id}] Extracted {len(words)} non-empty words")
        return words
    
    def generate_assignment_history_report(self, max_words: int = None) -> List[str]:
        """Generate a detailed assignment history report for debugging."""
        if self.df.empty:
            return ["No words in table"]
        
        report = []
        report.append("DETAILED ASSIGNMENT HISTORY REPORT")
        report.append("=" * 80)
        report.append(f"Content ID: {self.content_id}")
        report.append(f"Total Words: {len(self.df)}")
        report.append(f"Report Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 80)
        report.append("")
        
        # Sort by start time for chronological view
        sorted_df = self.df.sort_values('start')
        
        # Get speaker statistics
        speaker_counts = sorted_df['speaker_current'].value_counts()
        report.append("SPEAKER DISTRIBUTION:")
        for speaker, count in speaker_counts.items():
            report.append(f"  {speaker}: {count} words")
        report.append("")
        
        words_processed = 0
        for idx, word in sorted_df.iterrows():
            word_start = word['start']
            word_text = word['text']
            current_speaker = word['speaker_current']
            assignment_history = word.get('assignment_history', [])
            
            # Word header with more details
            word_end = word.get('end', word_start)
            confidence = word.get('assignment_confidence', 0.0)
            method = word.get('resolution_method', 'unknown')
            
            header = f"[{word_start:.2f}s-{word_end:.2f}s] '{word_text}' -> {current_speaker}"
            if not pd.isna(confidence):
                header += f" (final_conf={confidence:.2f}, method={method})"
            report.append(header)
            
            # Assignment history
            if isinstance(assignment_history, list) and assignment_history:
                for i, assignment in enumerate(assignment_history):
                    stage = assignment.get('stage', 'unknown')
                    speaker = assignment.get('speaker', 'UNKNOWN')
                    method = assignment.get('method', 'unknown')
                    confidence = assignment.get('confidence', 0.0)
                    reason = assignment.get('reason', 'No reason given')
                    timestamp = assignment.get('timestamp', 0)
                    
                    # Format timestamp if available
                    time_str = ""
                    if timestamp:
                        time_str = f" [{time.strftime('%H:%M:%S', time.localtime(timestamp))}]"
                    
                    prefix = "  └─" if i == len(assignment_history) - 1 else "  ├─"
                    report.append(f"{prefix} {stage}: {speaker} ({method}, conf={confidence:.2f}){time_str} - {reason}")
            else:
                report.append("  └─ No assignment history available")
            
            # Add metadata if interesting
            metadata = word.get('metadata', {})
            if isinstance(metadata, dict):
                interesting_metadata = []
                if metadata.get('boundary_tolerance_assignment'):
                    interesting_metadata.append(f"boundary_dist={metadata.get('boundary_distance', 0):.3f}s")
                if metadata.get('sentence_consolidated'):
                    interesting_metadata.append(f"consolidated_from={metadata.get('original_speaker', 'unknown')}")
                if metadata.get('orphan_reassigned'):
                    interesting_metadata.append(f"orphan_type={metadata.get('original_orphan_type', 'unknown')}")
                
                if interesting_metadata:
                    report.append(f"  📝 Metadata: {', '.join(interesting_metadata)}")
            
            report.append("")  # Blank line between words
            words_processed += 1
            
            # Apply limit if specified
            if max_words and words_processed >= max_words:
                remaining = len(sorted_df) - words_processed
                if remaining > 0:
                    report.append(f"... (showing first {max_words} words, {remaining} more words not shown)")
                    report.append("To see all words, increase max_words parameter or remove limit")
                break
        
        # Summary at the end
        report.append("=" * 80)
        report.append(f"SUMMARY: Processed {words_processed} of {len(sorted_df)} words")
        if max_words:
            report.append(f"Report was limited to {max_words} words")
        report.append("=" * 80)
        
        return report
    
    def get_speaker_statistics(self) -> Dict[str, Any]:
        """Get statistics about speaker assignments."""
        if self.df.empty:
            return {
                'unknown_words': 0,
                'speaker_distribution': {},
                'total_words': 0,
                'grammar_stats': {},
                'multi_speaker_stats': {}
            }
        
        # Count words by speaker
        speaker_counts = self.df['speaker_current'].value_counts().to_dict()
        
        # Count unknown words
        unknown_words = speaker_counts.get('UNKNOWN', 0)
        
        # Get distribution without UNKNOWN
        speaker_distribution = {k: v for k, v in speaker_counts.items()}
        
        # Grammar statistics
        grammar_stats = {
            'words_with_good_grammar': self.df['has_good_grammar'].sum(),
            'segments_with_good_grammar': self.df.groupby('segment_index')['segment_has_good_grammar'].first().sum(),
            'sentences_with_good_grammar': self.df.groupby('sentence_id')['has_good_grammar'].first().sum() if 'sentence_id' in self.df.columns else 0
        }
        
        # Multi-speaker statistics
        multi_speaker_stats = {
            'words_with_multiple_speakers': self.df['has_multiple_speakers'].sum(),
            'segments_with_multiple_speakers': self.df.groupby('segment_index')['segment_has_multiple_speakers'].first().sum(),
            'sentences_with_multiple_speakers': self.df.groupby('sentence_id')['has_multiple_speakers'].first().sum() if 'sentence_id' in self.df.columns else 0
        }
        
        return {
            'unknown_words': unknown_words,
            'speaker_distribution': speaker_distribution,
            'total_words': len(self.df),
            'grammar_stats': grammar_stats,
            'multi_speaker_stats': multi_speaker_stats
        }
    
    def get_words_needing_grammar_improvement(self) -> pd.DataFrame:
        """Get all words that don't have good grammar."""
        return self.df[~self.df['has_good_grammar']].copy()
    
    def get_words_with_potential_multi_speaker(self) -> pd.DataFrame:
        """Get all words that potentially have multiple speakers."""
        return self.df[self.df['has_multiple_speakers']].copy()
    
    def get_words_by_category(self, category: str) -> pd.DataFrame:
        """Get all words assigned to a specific category."""
        valid_categories = ['BAD_GRAMMAR_SINGLE', 'BAD_GRAMMAR_MULTI', 'GOOD_GRAMMAR_SINGLE', 'GOOD_GRAMMAR_MULTI', 'UNKNOWN']
        if category not in valid_categories:
            raise ValueError(f"Invalid category: {category}. Must be one of {valid_categories}")
        return self.df[self.df['speaker_current'] == category].copy()
    
    def get_category_statistics(self) -> Dict[str, int]:
        """Get count of words in each category."""
        categories = ['BAD_GRAMMAR_SINGLE', 'BAD_GRAMMAR_MULTI', 'GOOD_GRAMMAR_SINGLE', 'GOOD_GRAMMAR_MULTI', 'UNKNOWN']
        stats = {}
        for category in categories:
            stats[category] = len(self.df[self.df['speaker_current'] == category])
        return stats
    
    def get_segments_needing_grammar_improvement(self) -> List[int]:
        """Get segment indices that don't have good grammar."""
        if self.df.empty:
            return []
        segments_needing_grammar = self.df[~self.df['segment_has_good_grammar']]['segment_index'].unique()
        return sorted(segments_needing_grammar.tolist())
    
    def get_segments_with_potential_multi_speaker(self) -> List[int]:
        """Get segment indices that potentially have multiple speakers."""
        if self.df.empty:
            return []
        multi_speaker_segments = self.df[self.df['segment_has_multiple_speakers']]['segment_index'].unique()
        return sorted(multi_speaker_segments.tolist())
    
    def save_to_file(self, file_path: Path) -> None:
        """Save the word table to a JSON file."""
        logger.info(f"[{self.content_id}] Saving WordTable to {file_path}")
        
        # Convert DataFrame to dictionary format for JSON serialization
        data = {
            'content_id': self.content_id,
            'total_words': len(self.df),
            'words': self.df.to_dict('records')
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"[{self.content_id}] WordTable saved with {len(self.df)} words")

    def __len__(self) -> int:
        """Return the number of words in the table."""
        return len(self.df)


class SegmentTable:
    """
    Tracks diarization segments through the stitch pipeline.
    
    Each segment has:
    - segment_id: Unique identifier
    - start/end: Timing information
    - speaker: Speaker assigned by diarization
    - confidence: Diarization confidence score
    - processing_status: Current processing status
    """
    
    def __init__(self, content_id: str):
        """Initialize a new SegmentTable."""
        self.content_id = content_id
        self.df = pd.DataFrame(columns=[
            'segment_id',        # Unique identifier for this segment
            'start',             # Start time in seconds
            'end',               # End time in seconds
            'speaker',           # Speaker assigned by diarization
            'confidence',        # Diarization confidence score
            'text',              # Text content (if available)
            'duration',          # Segment duration in seconds
            'processing_status', # Current processing status
            'metadata'           # Additional metadata dict
        ])
        
        logger.info(f"[{content_id}] SegmentTable initialized")
    
    def initialize_from_diarization_data(self, diarization_segments: List[Dict]) -> int:
        """Initialize from diarization segment data."""
        logger.info(f"[{self.content_id}] Initializing SegmentTable from {len(diarization_segments)} segments")
        
        segments_data = []
        
        for segment_idx, segment in enumerate(diarization_segments):
            segment_id = str(uuid.uuid4())
            
            start_time = float(segment.get('start', 0))
            end_time = float(segment.get('end', 0))
            duration = end_time - start_time
            
            segment_entry = {
                'segment_id': segment_id,
                'start': start_time,
                'end': end_time,
                'speaker': segment.get('speaker', 'UNKNOWN'),
                'confidence': float(segment.get('confidence', 1.0)),
                'text': segment.get('text', ''),
                'duration': duration,
                'processing_status': 'initialized',
                'metadata': {
                    'original_index': segment_idx,
                    'source': 'diarization'
                }
            }
            
            segments_data.append(segment_entry)
        
        # Create DataFrame
        if segments_data:
            self.df = pd.DataFrame(segments_data)
            # Sort by start time
            self.df = self.df.sort_values('start').reset_index(drop=True)
        
        logger.info(f"[{self.content_id}] SegmentTable initialized with {len(segments_data)} segments")
        return len(segments_data)

    def find_overlapping_segments(self, start_time: float, end_time: float) -> pd.DataFrame:
        """Find segments that overlap with the given time range."""
        mask = (self.df['start'] < end_time) & (self.df['end'] > start_time)
        overlapping = self.df[mask].copy()
        
        # Calculate overlap duration
        overlapping['overlap_start'] = overlapping['start'].clip(lower=start_time)
        overlapping['overlap_end'] = overlapping['end'].clip(upper=end_time)
        overlapping['overlap_duration'] = overlapping['overlap_end'] - overlapping['overlap_start']
        overlapping['overlap_ratio'] = overlapping['overlap_duration'] / overlapping['duration']
        
        return overlapping

    def clean_data(self) -> Dict[str, Any]:
        """
        Clean the segment table data including removing invalid segments.
        
        Returns:
            Dictionary with cleaning statistics
        """
        logger.info(f"[{self.content_id}] Cleaning SegmentTable data")
        
        initial_count = len(self.df)
        
        # Remove segments with invalid timing
        invalid_timing_mask = self.df['end'] <= self.df['start']
        invalid_timing_count = invalid_timing_mask.sum()
        self.df = self.df[~invalid_timing_mask]
        
        # Remove segments with zero or negative duration
        zero_duration_mask = self.df['duration'] <= 0
        zero_duration_count = zero_duration_mask.sum()
        self.df = self.df[~zero_duration_mask]
        
        # Remove duplicate segments (same timing and speaker)
        duplicate_mask = self.df.duplicated(subset=['start', 'end', 'speaker'])
        duplicate_count = duplicate_mask.sum()
        self.df = self.df[~duplicate_mask]
        
        # Reset index after cleaning
        self.df = self.df.reset_index(drop=True)
        
        final_count = len(self.df)
        
        cleaning_stats = {
            'initial_segments': initial_count,
            'invalid_timing_removed': invalid_timing_count,
            'zero_duration_removed': zero_duration_count,
            'duplicate_segments_removed': duplicate_count,
            'final_segments': final_count,
            'total_removed': initial_count - final_count
        }
        
        logger.debug(f"[{self.content_id}] Segment complete: {initial_count} -> {final_count} segments")
        logger.info(f"[{self.content_id}] Removed: {invalid_timing_count} invalid timing, {zero_duration_count} zero duration, {duplicate_count} duplicates")
        
        return cleaning_stats
    
    def save_to_file(self, file_path: Path) -> None:
        """Save the segment table to a JSON file."""
        logger.info(f"[{self.content_id}] Saving SegmentTable to {file_path}")
        
        # Convert DataFrame to dictionary format for JSON serialization
        data = {
            'content_id': self.content_id,
            'total_segments': len(self.df),
            'segments': self.df.to_dict('records')
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"[{self.content_id}] SegmentTable saved with {len(self.df)} segments")

    def __len__(self) -> int:
        """Return the number of segments in the table."""
        return len(self.df)


def tables_stage(content_id: str,
                 cleaned_diarization_data: List[Dict],
                 cleaned_transcript_data: List[Dict]) -> Dict[str, Any]:
    """
    Main entry point for Stage 3: Create tracking tables and categorize words.
    
    This is the primary method called by the stitch pipeline. It creates the WordTable
    which tracks every word through all processing stages, and categorizes words based
    on grammar quality and speaker complexity for efficient processing.
    
    Args:
        content_id: Content ID to process (e.g., "Bdb001")
        cleaned_diarization_data: Cleaned diarization from Stage 2 (dict with 'segments' or list)
        cleaned_transcript_data: Cleaned transcript from Stage 2 (dict with 'segments' or list)
        
    Returns:
        Dictionary containing:
        - status: 'success' or 'error'
        - data: Dict with word_table and segment_table objects
        - stats: Table creation statistics and category breakdown
        - error: Error message if status is 'error'
        
    Example:
        result = tables_stage("Bdb001", cleaned_diar, cleaned_trans)
        if result['status'] == 'success':
            word_table = result['data']['word_table']
            categories = result['stats']['category_breakdown']
    """
    start_time = time.time()
    
    logger.info(f"[{content_id}] Starting Stage 3: Table Creation (from cleaned data)")
    
    result = {
        'status': 'pending',
        'content_id': content_id,
        'stage': 'table_creation',
        'data': {
            'word_table': None,
            'segment_table': None
        },
        'stats': {},
        'error': None
    }
    
    try:
        # Validate input data
        if not cleaned_diarization_data:
            raise ValueError("No cleaned diarization data available")
        if not cleaned_transcript_data:
            raise ValueError("No cleaned transcript data available")
        
        # Extract segments from data structures if needed
        diar_segments = cleaned_diarization_data.get('segments', cleaned_diarization_data)
        trans_segments = cleaned_transcript_data.get('segments', cleaned_transcript_data)
        
        logger.debug(f"[{content_id}] Processing")
        logger.debug(f"[{content_id}] Processing")
        logger.info(f"[{content_id}] diar_segments type: {type(diar_segments)}")
        logger.info(f"[{content_id}] trans_segments type: {type(trans_segments)}")
        if diar_segments and len(diar_segments) > 0:
            logger.info(f"[{content_id}] First diarization segment: {diar_segments[0]}")
        else:
            logger.warning(f"[{content_id}] diar_segments is empty or None: {diar_segments}")
        
        # Create and initialize WordTable
        logger.info(f"[{content_id}] Creating WordTable from CLEANED transcript data")
        
        word_table = WordTable(content_id)
        # Set diarization data BEFORE initializing to ensure it's available during sentence assignment
        word_table.set_diarization_data(diar_segments)
        
        word_count = word_table.initialize_from_transcript_data(trans_segments)
        
        # Create and initialize SegmentTable
        logger.info(f"[{content_id}] Creating SegmentTable from CLEANED diarization data")
        segment_table = SegmentTable(content_id)
        segment_count = segment_table.initialize_from_diarization_data(diar_segments)
        
        # Calculate speaker duration statistics
        speaker_durations = _calculate_speaker_durations(segment_table, content_id)
        
        # Store table objects and statistics
        result['data']['word_table'] = word_table
        result['data']['segment_table'] = segment_table
        
        stage_duration = time.time() - start_time
        # Get grammar and speaker category breakdown
        category_breakdown = word_table._get_grammar_speaker_categories() if hasattr(word_table, '_get_grammar_speaker_categories') else {}
        
        result['stats'] = {
            'duration': stage_duration,
            'words_created': word_count,
            'segments_created': segment_count,
            'word_table_size': len(word_table.df) if hasattr(word_table, 'df') else 0,
            'segment_table_size': len(segment_table.df) if hasattr(segment_table, 'df') else 0,
            'speaker_durations': speaker_durations,
            'category_breakdown': category_breakdown
        }
        
        result['status'] = 'success'
        
        logger.info(f"[{content_id}] Stage 3 completed successfully in {stage_duration:.2f}s")
        logger.info(f"[{content_id}] Created WordTable with {word_count} words from CLEANED data")
        logger.info(f"[{content_id}] Created SegmentTable with {segment_count} segments from CLEANED data")
        logger.info(f"[{content_id}] WordTable established as central tracking mechanism")
        
        return result
        
    except Exception as e:
        logger.error(f"[{content_id}] Stage 3 failed: {str(e)}")
        logger.error(f"[{content_id}] Error details:", exc_info=True)
        
        result.update({
            'status': 'error',
            'error': str(e),
            'duration': time.time() - start_time
        })
        return result


def get_word_table_summary(word_table: 'WordTable') -> Dict[str, Any]:
    """
    Get a comprehensive summary of the WordTable for pipeline tracking.
    
    Args:
        word_table: WordTable instance
        
    Returns:
        Dictionary with comprehensive tracking information
    """
    if not hasattr(word_table, 'df') or word_table.df.empty:
        return {'status': 'empty', 'total_words': 0}
    
    summary = {
        'status': 'ready',
        'content_id': word_table.content_id,
        'total_words': len(word_table.df),
        'time_span': {
            'start': float(word_table.df['start'].min()),
            'end': float(word_table.df['end'].max()),
            'duration': float(word_table.df['end'].max() - word_table.df['start'].min())
        },
        'speaker_assignments': word_table.df['speaker_current'].value_counts().to_dict(),
        'processing_status': word_table.df['processing_status'].value_counts().to_dict(),
        'resolution_methods': word_table.df['resolution_method'].value_counts().to_dict(),
        'confidence_stats': {
            'mean': float(word_table.df['confidence'].mean()),
            'min': float(word_table.df['confidence'].min()),
            'max': float(word_table.df['confidence'].max()),
            'std': float(word_table.df['confidence'].std())
        },
        'segments_represented': len(word_table.df['segment_index'].unique()),
        'tracking_readiness': {
            'unique_word_ids': len(word_table.df['word_id'].unique()),
            'words_with_valid_timing': len(word_table.df[word_table.df['start'] < word_table.df['end']]),
            'words_ready_for_assignment': len(word_table.df[word_table.df['speaker_current'].isin(['BAD_GRAMMAR_SINGLE', 'BAD_GRAMMAR_MULTI', 'GOOD_GRAMMAR_SINGLE', 'GOOD_GRAMMAR_MULTI'])])
        }
    }
    
    return summary


def _calculate_speaker_durations(segment_table: 'SegmentTable', content_id: str) -> Dict[str, Any]:
    """
    Calculate duration statistics for each speaker from diarization segments.
    
    Args:
        segment_table: SegmentTable with diarization segments
        content_id: Content ID for logging
        
    Returns:
        Dictionary with speaker duration statistics
    """
    if segment_table.df.empty:
        return {}
    
    # Group by speaker and calculate durations
    speaker_groups = segment_table.df.groupby('speaker')
    speaker_durations = {}
    
    total_duration = 0
    for speaker, segments in speaker_groups:
        speaker_duration = segments['duration'].sum()
        segment_count = len(segments)
        
        speaker_durations[speaker] = {
            'total_duration': float(speaker_duration),
            'segment_count': int(segment_count),
            'average_segment_duration': float(speaker_duration / segment_count) if segment_count > 0 else 0,
            'min_segment_duration': float(segments['duration'].min()),
            'max_segment_duration': float(segments['duration'].max())
        }
        
        total_duration += speaker_duration
    
    # Calculate percentages and log
    logger.info(f"[{content_id}] Speaker durations from extended diarization segments:")
    for speaker, stats in sorted(speaker_durations.items()):
        percentage = (stats['total_duration'] / total_duration * 100) if total_duration > 0 else 0
        logger.info(f"[{content_id}] - {speaker}: {stats['total_duration']:.1f}s ({percentage:.1f}%), "
                   f"{stats['segment_count']} segments, avg {stats['average_segment_duration']:.1f}s")
    
    # Add summary statistics
    speaker_durations['_summary'] = {
        'total_duration': float(total_duration),
        'total_speakers': len(speaker_durations) - 1,  # Exclude _summary itself
        'duration_by_percentage': {
            speaker: float(stats['total_duration'] / total_duration * 100) if total_duration > 0 else 0
            for speaker, stats in speaker_durations.items()
            if speaker != '_summary'
        }
    }
    
    return speaker_durations


def update_word_processing_status(word_table: 'WordTable', stage_name: str, word_ids: List[str] = None) -> int:
    """
    Update processing status for words as they progress through pipeline stages.
    
    Args:
        word_table: WordTable instance
        stage_name: Name of the processing stage
        word_ids: Specific word IDs to update (if None, updates all)
        
    Returns:
        Number of words updated
    """
    if word_ids is None:
        # Update all words
        mask = word_table.df.index
        updated_count = len(word_table.df)
    else:
        # Update specific words
        mask = word_table.df['word_id'].isin(word_ids)
        updated_count = mask.sum()
    
    if updated_count > 0:
        word_table.df.loc[mask, 'processing_status'] = f'processed_stage_{stage_name}'
        
        # Update metadata with stage progression
        for idx in word_table.df[mask].index:
            current_metadata = word_table.df.at[idx, 'metadata']
            if isinstance(current_metadata, dict):
                if 'stage_progression' not in current_metadata:
                    current_metadata['stage_progression'] = []
                current_metadata['stage_progression'].append({
                    'stage': stage_name,
                    'timestamp': time.time()
                })
            else:
                word_table.df.at[idx, 'metadata'] = {
                    'stage_progression': [{
                        'stage': stage_name,
                        'timestamp': time.time()
                    }]
                }
    
    return updated_count
