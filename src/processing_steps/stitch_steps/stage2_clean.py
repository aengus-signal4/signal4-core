#!/usr/bin/env python3
"""
Stage 2: Data Cleaning
======================

Second stage of the stitch pipeline that cleans raw data from Stage 1.

Key Responsibilities:
- Remove chunk boundary overlaps (duplicates at ~298s boundaries)
- Clean excessive word/phrase repetitions ("uh uh uh..." → "uh uh uh")
- Filter invalid diarization segments (zero duration, invalid timing)
- Remove duplicate diarization segments
- Rebuild word arrays to match cleaned text
- Save debug outputs for boundary analysis

Input:
- Raw diarization segments from Stage 1
- Raw transcript segments with word-level timing

Output:
- Cleaned diarization segments
- Cleaned transcript segments with overlaps merged
- combined_transcript_cleaned.json for debugging

Methods:
- clean_stage(): Main entry point called by stitch pipeline
- _clean_transcript_segments(): Handles chunk overlaps and repetitions
- _clean_diarization_segments(): Validates and deduplicates diarization
- _clean_repetitive_text(): Removes excessive repetitions
- _vectorized_timing_deduplication(): Fast duplicate detection using numpy
"""

import logging
import time
import json
from pathlib import Path
from src.utils.paths import get_project_root, get_config_path
from typing import Dict, Any, List, Set, Tuple
from collections import Counter
import numpy as np

from src.utils.logger import setup_worker_logger
from stitch_steps.util_stitch import smart_join_words

logger = setup_worker_logger('stitch')
logger.setLevel(logging.INFO)


def normalize_word(word):
    """Normalize a word for comparison by removing punctuation and converting to lowercase."""
    import re
    return re.sub(r'[^\w]', '', word.lower())


def _detect_word_spacing_model(transcript_segments: List[Dict]) -> str:
    """
    Detect whether transcript words have leading spaces (Parakeet) or not (Whisper).

    Args:
        transcript_segments: List of transcript segments with word-level data

    Returns:
        'parakeet' if words have leading spaces, 'whisper' if they don't
    """
    # Sample up to 50 words from various segments
    sample_words = []
    for seg in transcript_segments[:10]:  # Check first 10 segments
        words = seg.get('words', [])
        for word_data in words[:10]:  # Check first 10 words per segment
            word_text = word_data.get('word', '')
            if word_text and len(word_text) > 1:  # Skip empty and single-char
                sample_words.append(word_text)
                if len(sample_words) >= 50:
                    break
        if len(sample_words) >= 50:
            break

    if not sample_words:
        return 'whisper'  # Default to whisper behavior

    # Count how many words start with a space
    words_with_leading_space = sum(1 for w in sample_words if w[0] == ' ')

    # If more than 70% have leading spaces, it's Parakeet
    if words_with_leading_space / len(sample_words) > 0.7:
        return 'parakeet'
    else:
        return 'whisper'


def _vectorized_timing_deduplication(all_words_with_timing, content_id):
    """
    Vectorized implementation of timing-based deduplication.
    
    Args:
        all_words_with_timing: List of word dictionaries with timing info
        content_id: Content ID for logging
        
    Returns:
        Number of duplicates removed
    """
    from collections import defaultdict
    
    if not all_words_with_timing:
        return 0
        
    timing_duplicates_removed = 0
    
    # Convert to numpy arrays for vectorized operations
    n_words = len(all_words_with_timing)
    starts = np.array([w['start'] for w in all_words_with_timing])
    ends = np.array([w['end'] for w in all_words_with_timing])
    seg_indices = np.array([w['seg_idx'] for w in all_words_with_timing])
    word_indices = np.array([w['word_idx'] for w in all_words_with_timing])
    normalized_words = [w['normalized'] for w in all_words_with_timing]
    
    # Sort by start time
    sort_indices = np.argsort(starts)
    starts = starts[sort_indices]
    ends = ends[sort_indices]
    seg_indices = seg_indices[sort_indices]
    word_indices = word_indices[sort_indices]
    normalized_words = [normalized_words[i] for i in sort_indices]
    all_words_with_timing = [all_words_with_timing[i] for i in sort_indices]
    
    # Track which words to remove
    to_remove = np.zeros(n_words, dtype=bool)
    
    # Group by normalized word for efficient comparison
    word_groups = defaultdict(list)
    for i in range(n_words):
        word_groups[normalized_words[i]].append(i)
    
    # Process each group of same words
    for word, indices in word_groups.items():
        if len(indices) < 2:
            continue
            
        # Convert to numpy array for vectorized operations
        indices = np.array(indices)
        n_group = len(indices)
        
        # Extract timing data for this group
        group_starts = starts[indices]
        group_ends = ends[indices]
        group_seg_idx = seg_indices[indices]
        group_word_idx = word_indices[indices]
        
        # Vectorized pairwise comparison using broadcasting
        # Create matrices for comparison
        starts_diff = np.abs(group_starts[:, np.newaxis] - group_starts[np.newaxis, :])
        ends_diff = np.abs(group_ends[:, np.newaxis] - group_ends[np.newaxis, :])
        
        # Calculate overlap durations
        overlap_starts = np.maximum(group_starts[:, np.newaxis], group_starts[np.newaxis, :])
        overlap_ends = np.minimum(group_ends[:, np.newaxis], group_ends[np.newaxis, :])
        overlap_durations = np.maximum(0, overlap_ends - overlap_starts)
        
        # Calculate durations
        durations = group_ends - group_starts
        durations_matrix = durations[:, np.newaxis]
        durations_matrix_T = durations[np.newaxis, :]
        
        # Calculate overlap ratios (avoid division by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            overlap_ratios_1 = np.where(durations_matrix > 0, overlap_durations / durations_matrix, 0)
            overlap_ratios_2 = np.where(durations_matrix_T > 0, overlap_durations / durations_matrix_T, 0)
        
        # Find duplicates based on criteria
        same_start = starts_diff < 0.05
        same_end = ends_diff < 0.05
        high_overlap = (overlap_ratios_1 >= 0.8) | (overlap_ratios_2 >= 0.8)
        
        # Combine criteria - only consider upper triangle (i < j)
        is_duplicate = (same_start | same_end | high_overlap) & (np.arange(n_group)[:, np.newaxis] < np.arange(n_group)[np.newaxis, :])
        
        # For each duplicate pair, remove the one from the later segment
        for i in range(n_group):
            for j in range(i + 1, n_group):
                if is_duplicate[i, j] and not to_remove[indices[i]] and not to_remove[indices[j]]:
                    # Determine which to remove based on segment index
                    if group_seg_idx[i] < group_seg_idx[j]:
                        to_remove[indices[j]] = True
                        removed_idx = indices[j]
                        kept_idx = indices[i]
                    elif group_seg_idx[j] < group_seg_idx[i]:
                        to_remove[indices[i]] = True
                        removed_idx = indices[i]
                        kept_idx = indices[j]
                    else:
                        # Same segment - keep earlier word index
                        if group_word_idx[i] < group_word_idx[j]:
                            to_remove[indices[j]] = True
                            removed_idx = indices[j]
                            kept_idx = indices[i]
                        else:
                            to_remove[indices[i]] = True
                            removed_idx = indices[i]
                            kept_idx = indices[j]
                    
                    timing_duplicates_removed += 1
                    
                    # Log the removal
                    removed_word = all_words_with_timing[removed_idx]
                    kept_word = all_words_with_timing[kept_idx]
                    
                    if same_start[i, j] and same_end[i, j]:
                        logger.debug(f"[{content_id}] Removing duplicate with same start and end time: '{removed_word['text']}' at {removed_word['start']:.3f}-{removed_word['end']:.3f}s (matches earlier {kept_word['start']:.3f}-{kept_word['end']:.3f}s)")
                    elif same_start[i, j]:
                        logger.debug(f"[{content_id}] Removing duplicate with same start time: '{removed_word['text']}' at {removed_word['start']:.3f}-{removed_word['end']:.3f}s (matches earlier {kept_word['start']:.3f}-{kept_word['end']:.3f}s)")
                    elif same_end[i, j]:
                        logger.debug(f"[{content_id}] Removing duplicate with same end time: '{removed_word['text']}' at {removed_word['start']:.3f}-{removed_word['end']:.3f}s (matches earlier {kept_word['start']:.3f}-{kept_word['end']:.3f}s)")
                    else:
                        logger.debug(f"[{content_id}] Removing timing duplicate: '{removed_word['text']}' at {removed_word['start']:.2f}-{removed_word['end']:.2f}s")
    
    # Apply the removal flags back to the original list
    for i, should_remove in enumerate(to_remove):
        all_words_with_timing[i]['to_remove'] = should_remove
    
    return timing_duplicates_removed


class NumpyJsonEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.str_):
            return str(obj)
        elif hasattr(obj, 'item'):  # For numpy scalars
            return obj.item()
        return super().default(obj)


def _is_contentless_segment(text: str) -> bool:
    """
    Check if a segment contains no meaningful content (only artifacts).

    Returns True if the segment is only:
    - Ellipses (... or …)
    - Punctuation
    - Whitespace
    - Very short (1-2 chars)

    Args:
        text: Segment text to check

    Returns:
        True if segment has no meaningful content
    """
    if not text:
        return True

    # Strip whitespace
    text = text.strip()

    if len(text) <= 2:
        return True

    # Replace ellipses and common artifacts
    cleaned = text.replace('...', '').replace('…', '').replace('.', '').replace(',', '').replace('-', '').strip()

    # If nothing left after removing punctuation, it's contentless
    return len(cleaned) == 0


def _remove_unicode_artifacts(text: str) -> str:
    """
    Remove or normalize unicode/non-ASCII artifacts from text.
    - Removes sequences of repeated non-ASCII characters (e.g., '話話話話話')
    - Optionally normalizes common unicode punctuation to ASCII equivalents
    - Removes isolated non-ASCII characters that are not part of valid words
    """
    import re
    import unicodedata

    if not text:
        return text

    # Normalize unicode to NFKC form
    text = unicodedata.normalize('NFKC', text)

    # Remove sequences of 3+ repeated non-ASCII characters (common artifact)
    text = re.sub(r'([^ -~])\1{2,}', '', text)

    # Remove isolated non-ASCII characters surrounded by spaces or punctuation
    text = re.sub(r'(?<=\s)[^\u0000-\u007F](?=\s)', '', text)

    # Optionally, replace common unicode punctuation with ASCII equivalents
    replacements = {
        '\u2019': "'",  # right single quotation mark
        '\u2018': "'",  # left single quotation mark
        '\u201C': '"',  # left double quotation mark
        '\u201D': '"',  # right double quotation mark
        '\u2013': '-',  # en dash
        '\u2014': '-',  # em dash
        '\u2026': '...',  # ellipsis
        '–': '-',
        '—': '-',
        '…': '...',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    # Note: Removed aggressive ASCII-only filter to preserve legitimate accented characters
    # from French, Ukrainian, and other languages. The targeted replacements above
    # should handle problematic punctuation artifacts.

    # Collapse extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def _remove_excessive_unk_tokens(text: str, max_consecutive_unks: int = 5) -> str:
    """
    Remove all <unk> tokens from transcription.

    When transcription fails (poor audio, music, silence), it can output
    thousands of <unk> tokens. These are removed completely as they confuse
    readers and provide no useful information. Original transcripts are preserved
    in raw files if needed.

    Args:
        text: Input text that may contain <unk> tokens
        max_consecutive_unks: Unused parameter (kept for compatibility)

    Returns:
        Text with all <unk> tokens removed
    """
    import re

    if not text or '<unk>' not in text:
        return text

    # Count how many <unk> tokens exist
    unk_count = text.count('<unk>')

    if unk_count == 0:
        return text

    # Remove all <unk> tokens completely
    cleaned = re.sub(r'<unk>\s*', '', text)

    # Collapse multiple spaces that may result from removal
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    if unk_count > 0:
        logger.debug(f"Removed all {unk_count} <unk> tokens from text")

    return cleaned


def _clean_repetitive_text(text: str, max_word_repetition: int = 3, max_ngram_repetition: int = 2, max_ngram_size: int = 5) -> str:
    """
    Clean excessive repetition of words and phrases from text, plus Unicode/non-ASCII artifacts and <unk> tokens.

    Handles cases like:
    - "uh, uh, uh, uh, uh..." -> "uh, uh, uh"
    - "in the house in the house in the house" -> "in the house in the house"
    - Unicode artifacts like "話話話話話..." -> removed entirely
    - <unk> tokens from failed transcription -> removed completely

    Args:
        text: Input text to clean
        max_word_repetition: Maximum allowed consecutive repetitions of the same word
        max_ngram_repetition: Maximum allowed consecutive repetitions of n-grams
        max_ngram_size: Maximum n-gram size to check

    Returns:
        Cleaned text with excessive repetitions and artifacts removed
    """
    if not text:
        return text

    # Step 0a: Remove all <unk> tokens (common in failed transcriptions)
    cleaned_text = _remove_excessive_unk_tokens(text)

    # Step 0b: Remove Unicode/non-ASCII artifacts
    cleaned_text = _remove_unicode_artifacts(cleaned_text)
    
    words = cleaned_text.split()
    if len(words) < 2:
        return cleaned_text
    
    # Step 1: Clean excessive single word repetition
    cleaned_words = _remove_excessive_word_repetition(words, max_word_repetition)
    
    # Step 2: Clean excessive n-gram repetition (phrases)
    cleaned_words = _remove_excessive_ngram_repetition(cleaned_words, max_ngram_repetition, max_ngram_size)
    
    final_text = ' '.join(cleaned_words)
    
    # Log significant cleaning
    original_word_count = len(text.split())
    cleaned_word_count = len(cleaned_words)
    if cleaned_word_count < original_word_count:
        removed_count = original_word_count - cleaned_word_count
        logger.debug(f"Cleaned repetitive text: removed {removed_count} words ({original_word_count} -> {cleaned_word_count})")
        
        # Log specific cleaning for very repetitive cases
        if removed_count > 10:
            logger.debug(f"Cleaned excessive repetition: {removed_count} words removed")
            logger.debug(f"Original text preview: {text[:100]}...")
            logger.debug(f"Cleaned text preview: {final_text[:100]}...")
    
    return final_text


def _remove_excessive_word_repetition(words: List[str], max_repetition: int) -> List[str]:
    """
    Remove excessive consecutive repetition of words.
    
    For example: "uh uh uh uh uh" -> "uh uh uh" (if max_repetition = 3)
    """
    if not words:
        return words
    
    cleaned = []
    current_word = None
    current_count = 0
    
    for word in words:
        word_normalized = word.lower().strip('.,!?;:')  # Normalize for comparison
        
        if word_normalized == current_word:
            current_count += 1
            if current_count <= max_repetition:
                cleaned.append(word)
            # Skip words beyond max_repetition
        else:
            current_word = word_normalized
            current_count = 1
            cleaned.append(word)
    
    return cleaned


def _remove_excessive_ngram_repetition(words: List[str], max_repetition: int, max_ngram_size: int) -> List[str]:
    """
    Remove excessive consecutive repetition of n-grams (2-5 word phrases).
    
    For example: "in the house in the house in the house" -> "in the house in the house"
    (if max_repetition = 2)
    """
    if len(words) < 2:
        return words
    
    cleaned_words = words[:]
    
    # First check if we have a repeating pattern that's longer than max_ngram_size
    # by looking for exact repetitions in the word list
    if len(cleaned_words) >= 4:
        # Try to detect full phrase repetition
        for pattern_len in range(len(cleaned_words) // 2, max_ngram_size, -1):
            pattern = cleaned_words[:pattern_len]
            repetitions = 1
            
            # Check if the pattern repeats
            for i in range(pattern_len, len(cleaned_words), pattern_len):
                if i + pattern_len <= len(cleaned_words):
                    if cleaned_words[i:i+pattern_len] == pattern:
                        repetitions += 1
                    else:
                        break
            
            # If we found a repeating pattern that covers most of the text
            if repetitions > 1 and repetitions * pattern_len >= len(cleaned_words) * 0.8:
                if repetitions > max_repetition:
                    # Keep only max allowed repetitions
                    cleaned_words = cleaned_words[:pattern_len * max_repetition]
                    logger.debug(f"Removed {repetitions - max_repetition} repetitions of {pattern_len}-word pattern: {' '.join(pattern[:5])}...")
                    return cleaned_words
    
    # Standard n-gram processing for smaller patterns
    for n in range(min(max_ngram_size, len(cleaned_words)), 1, -1):
        i = 0
        while i <= len(cleaned_words) - n:
            # Get current n-gram
            ngram = tuple(cleaned_words[i:i+n])
            
            # Count consecutive repetitions
            repetitions = 1
            j = i + n
            while j <= len(cleaned_words) - n:
                next_ngram = tuple(cleaned_words[j:j+n])
                if ngram == next_ngram:
                    repetitions += 1
                    j += n
                else:
                    break
            
            # If we found excessive repetitions, keep only max allowed
            if repetitions > max_repetition:
                # Calculate how many words to remove
                words_to_remove = (repetitions - max_repetition) * n
                # Remove the excess repetitions
                del cleaned_words[i + max_repetition * n : i + repetitions * n]
                logger.debug(f"Removed {repetitions - max_repetition} repetitions of {n}-gram: {' '.join(ngram)}")
            
            i += 1
    
    return cleaned_words


def _clean_transcript_segments(transcript_segments: List[Dict], content_id: str, model_type: str = 'whisper') -> Tuple[List[Dict], Dict[str, Any]]:
    """
    Clean transcript segments using proven chunk overlap handling approach.
    
    Args:
        transcript_segments: List of transcript segment dictionaries
        content_id: Content ID for logging
        
    Returns:
        Tuple of (cleaned_segments, cleaning_stats)
    """
    logger.info(f"[{content_id}] Cleaning {len(transcript_segments)} transcript segments")
    
    # Count initial words for validation
    initial_word_count = 0
    initial_words = []
    for seg in transcript_segments:
        words = seg.get('words', [])
        initial_word_count += len(words)
        initial_words.extend([w.get('word', '').strip() for w in words])
    
    logger.debug(f"[{content_id}] Initial word count: {initial_word_count}")
    if initial_word_count > 0 and len(initial_words) <= 50:  # Only log for small segments
        logger.debug(f"[{content_id}] Initial words: {initial_words}")
    elif initial_word_count > 0:
        logger.debug(f"[{content_id}] First 10 words: {initial_words[:10]}")
        logger.debug(f"[{content_id}] Last 10 words: {initial_words[-10:]}")
    
    # Initialize stats
    total_stats = {
        'segments_processed': len(transcript_segments),
        'segments_kept': 0,
        'overlap_merges': 0,
        'initial_word_count': initial_word_count,
        'duplicates_removed': 0,
        'empty_segments_removed': 0,
        'repetitive_text_cleaned': 0,
        'total_artifacts_removed': 0
    }
    
    # Step 1: Remove empty and contentless segments (ellipses, punctuation-only, etc.)
    non_empty_segments = []
    for segment in transcript_segments:
        text = segment.get('text', '').strip()
        if text and not _is_contentless_segment(text):
            non_empty_segments.append(segment)
        else:
            total_stats['empty_segments_removed'] += 1

    logger.info(f"[{content_id}] After removing empty/contentless segments: {len(non_empty_segments)} segments")
    
    # Step 2: Clean repetitive text in segments
    logger.debug(f"[{content_id}] Cleaning repetitive text artifacts...")
    for i, segment in enumerate(non_empty_segments):
        original_text = segment.get('text', '')
        cleaned_text = _clean_repetitive_text(original_text)
        
        if cleaned_text != original_text:
            segment['text'] = cleaned_text
            total_stats['repetitive_text_cleaned'] += 1
            
            # Also update words if they exist
            if 'words' in segment and segment['words']:
                # Rebuild words array to match cleaned text
                cleaned_words_list = cleaned_text.split()
                original_words = segment['words']
                
                # Simple word matching - this preserves timing for non-repeated words
                new_words = []
                cleaned_idx = 0
                
                for word_data in original_words:
                    word_text = word_data.get('word', '').strip()
                    if cleaned_idx < len(cleaned_words_list) and word_text:
                        # Check if this word appears in the cleaned text at current position
                        if word_text == cleaned_words_list[cleaned_idx]:
                            new_words.append(word_data)
                            cleaned_idx += 1
                        # Otherwise skip this word (it was part of repetition)
                
                segment['words'] = new_words
                logger.debug(f"[{content_id}] Segment {i}: Updated words array after repetition cleaning ({len(original_words)} -> {len(new_words)} words)")
    
    # Step 3: Handle chunk overlaps using proven approach
    final_segments = []
    duplicate_count = 0
    overlap_merges = 0
    
    # Identify chunk boundaries (298s, 596s, 894s, etc.)
    chunk_boundaries = set()
    if non_empty_segments:
        # Calculate boundaries based on standard chunk pattern
        max_time = max(float(seg.get('end', 0)) for seg in non_empty_segments)
        step_size = 298  # chunk_size (300) - chunk_overlap (2)
        
        boundary_time = step_size
        while boundary_time <= max_time + step_size:
            chunk_boundaries.add(boundary_time)
            boundary_time += step_size
    
    if chunk_boundaries:
        logger.debug(f"[{content_id}] Processing chunk boundaries at: {sorted(list(chunk_boundaries)[:5])}{'...' if len(chunk_boundaries) > 5 else ''}")
    
    # Process segments with overlap handling
    i = 0
    while i < len(non_empty_segments):
        current_seg = non_empty_segments[i]
        
        # Check if this segment is near a chunk boundary
        seg_start = float(current_seg.get('start', 0))
        seg_end = float(current_seg.get('end', 0))
        near_boundary = any(abs(seg_start - boundary) < 3.0 or abs(seg_end - boundary) < 3.0 for boundary in chunk_boundaries)
        
        logger.debug(f"[{content_id}] Segment {i}: {seg_start:.1f}-{seg_end:.1f}s, near_boundary: {near_boundary}")
        if chunk_boundaries:
            closest_boundary = min(chunk_boundaries, key=lambda b: min(abs(seg_start - b), abs(seg_end - b)))
            logger.debug(f"  Closest boundary: {closest_boundary:.1f}s, distance: {min(abs(seg_start - closest_boundary), abs(seg_end - closest_boundary)):.1f}s")
        
        if near_boundary and i < len(non_empty_segments) - 1:
            # Look ahead for potential overlap duplicates
            j = i + 1
            merged = False
            
            while j < len(non_empty_segments) and j < i + 5:  # Check next few segments
                next_seg = non_empty_segments[j]
                
                # Check if segments have overlapping text content
                curr_text = current_seg.get('text', '').strip()
                next_text = next_seg.get('text', '').strip()
                
                # Calculate text overlap
                curr_words = curr_text.split()
                next_words = next_text.split()
                
                if curr_words and next_words:
                    # Check for partial overlap at word level
                    overlap_found = False
                    
                    logger.debug(f"[{content_id}] Checking overlap between segments:")
                    logger.debug(f"  Current: '{curr_text}' ({current_seg.get('start', 0):.1f}s)")  
                    logger.debug(f"  Next: '{next_text}' ({next_seg.get('start', 0):.1f}s)")
                    
                    # Use the normalize_word function defined below
                    
                    curr_normalized = [normalize_word(w) for w in curr_words]
                    next_normalized = [normalize_word(w) for w in next_words]
                    
                    logger.debug(f"  Current normalized: {curr_normalized}")
                    logger.debug(f"  Next normalized: {next_normalized}")
                    
                    for k in range(1, min(len(curr_words), len(next_words), 6)):
                        if curr_normalized[-k:] == next_normalized[:k]:
                            # Found overlap - merge segments by removing overlap from second segment
                            logger.debug(f"[{content_id}] Found {k}-word overlap: '{' '.join(curr_words[-k:])}'")
                            
                            # Simple approach: keep all of current segment, skip overlap from next
                            merged_words = current_seg.get('words', []).copy()
                            next_words_list = next_seg.get('words', [])
                            
                            if next_words_list and len(next_words_list) > k:
                                # Skip the first k words (the overlap) from next segment
                                words_to_add = next_words_list[k:]
                                merged_words.extend(words_to_add)
                                logger.debug(f"[{content_id}] Removed {k} overlapping words from start of next segment")
                            elif next_words_list and len(next_words_list) <= k:
                                # Entire next segment is overlap - skip it entirely
                                logger.debug(f"[{content_id}] Next segment is entirely overlap, skipping all {len(next_words_list)} words")
                            
                            # Build merged text from merged words to ensure consistency
                            if merged_words:
                                if model_type == 'parakeet':
                                    # Parakeet: words already have leading spaces, just join them
                                    merged_text = "".join(word.get('word', '') for word in merged_words)
                                else:
                                    # Whisper: words don't have spaces, use smart_join_words
                                    word_texts = [word.get('word', '').strip() for word in merged_words]
                                    merged_text = smart_join_words(word_texts)
                            else:
                                # Fallback to text-based merge if no word-level data
                                merged_text = curr_text + " " + " ".join(next_words[k:])
                            
                            logger.debug(f"[{content_id}] Merged segment text: '{merged_text[:100]}{'...' if len(merged_text) > 100 else ''}'")
                            words_added = len(merged_words) - len(current_seg.get('words', []))
                            logger.debug(f"[{content_id}] Merged words count: {len(merged_words)} (curr: {len(current_seg.get('words', []))}, added: {words_added}, skipped: {k})")
                            
                            # Create merged segment
                            merged_segment = {
                                'text': merged_text.strip(),
                                'start': current_seg['start'],
                                'end': next_seg['end'],
                                'words': merged_words
                            }
                            
                            # Copy other properties from current segment
                            for key, value in current_seg.items():
                                if key not in ['text', 'start', 'end', 'words']:
                                    merged_segment[key] = value
                            
                            final_segments.append(merged_segment)
                            overlap_merges += 1
                            i = j + 1  # Skip past the merged segment
                            merged = True
                            overlap_found = True
                            break
                    
                    # Also check for complete text duplication
                    if not overlap_found and (curr_text in next_text or next_text in curr_text):
                        # One segment contains the other - keep the longer one
                        if len(next_text) > len(curr_text):
                            final_segments.append(next_seg)
                        else:
                            final_segments.append(current_seg)
                        duplicate_count += 1
                        i = j + 1
                        merged = True
                        break
                
                j += 1
            
            if not merged:
                final_segments.append(current_seg)
                i += 1
        else:
            # Not near boundary or last segment - just add it
            final_segments.append(current_seg)
            i += 1
    
    # Step 4: Final pass to remove duplicate words based on timing overlap
    logger.debug(f"[{content_id}] Performing final timing-based deduplication pass...")
    timing_duplicates_removed = 0
    
    # Collect all words with their timing and segment info
    all_words_with_timing = []
    
    for seg_idx, segment in enumerate(final_segments):
        words = segment.get('words', [])
        for word_idx, word in enumerate(words):
            if word.get('start') is not None and word.get('end') is not None:
                word_text = word.get('word', '').strip()
                normalized = normalize_word(word_text)
                
                word_info = {
                    'word': word,
                    'text': word_text,
                    'normalized': normalized,
                    'start': float(word.get('start', 0)),
                    'end': float(word.get('end', 0)),
                    'seg_idx': seg_idx,
                    'word_idx': word_idx,
                    'to_remove': False
                }
                all_words_with_timing.append(word_info)
    
    # Use vectorized implementation for better performance
    timing_duplicates_removed = _vectorized_timing_deduplication(all_words_with_timing, content_id)
    
    # Remove marked duplicates from segments
    if timing_duplicates_removed > 0:
        for seg_idx, segment in enumerate(final_segments):
            original_words = segment.get('words', [])
            if original_words:
                # Keep only words not marked for removal
                kept_words = []
                for word_idx, word in enumerate(original_words):
                    # Find this word in our timing list
                    should_remove = False
                    for timed_word in all_words_with_timing:
                        if (timed_word['seg_idx'] == seg_idx and 
                            timed_word['word_idx'] == word_idx and 
                            timed_word['to_remove']):
                            should_remove = True
                            break
                    
                    if not should_remove:
                        kept_words.append(word)
                
                segment['words'] = kept_words

                # Rebuild text from remaining words
                if kept_words:
                    if model_type == 'parakeet':
                        # Parakeet: words already have leading spaces, just join them
                        segment['text'] = "".join(w.get('word', '') for w in kept_words)
                    else:
                        # Whisper: words don't have spaces, use smart_join_words
                        word_texts = [w.get('word', '').strip() for w in kept_words]
                        segment['text'] = smart_join_words(word_texts)
                else:
                    segment['text'] = ""
        
        logger.info(f"[{content_id}] Removed {timing_duplicates_removed} duplicate words based on timing overlap")
    
    # Update final stats
    total_stats['segments_kept'] = len(final_segments)
    total_stats['overlap_merges'] = overlap_merges
    total_stats['duplicates_removed'] = duplicate_count
    total_stats['timing_duplicates_removed'] = timing_duplicates_removed
    total_stats['total_artifacts_removed'] = total_stats['empty_segments_removed'] + duplicate_count + total_stats['repetitive_text_cleaned'] + timing_duplicates_removed
    
    # Count total words for compatibility with main stitch.py and validate word preservation
    total_final_words = 0
    final_words = []
    for segment in final_segments:
        words = segment.get('words', [])
        if words:
            total_final_words += len(words)
            final_words.extend([w.get('word', '').strip() for w in words])
    
    total_stats['total_final_words'] = total_final_words
    
    # Critical validation: Check for word loss
    words_lost = initial_word_count - total_final_words
    if words_lost > 0:
        logger.warning(f"[{content_id}] Word count difference during cleaning: {words_lost} words ({initial_word_count} -> {total_final_words})")
        
        # Find specific lost words for debugging
        initial_set = set(initial_words)
        final_set = set(final_words)
        lost_words = initial_set - final_set
        if lost_words:
            logger.debug(f"[{content_id}] Words removed (likely duplicates): {list(lost_words)[:10]}{'...' if len(lost_words) > 10 else ''}")
            
        # Check if these are legitimate duplicates from overlap merging
        if overlap_merges > 0:
            logger.debug(f"[{content_id}] Note: {overlap_merges} overlap merges performed, word count difference may be from removing duplicate words at boundaries")
    
    logger.info(f"[{content_id}] Transcript cleaning: {total_stats['segments_processed']} -> {total_stats['segments_kept']} segments")
    logger.info(f"[{content_id}] Word count: {initial_word_count} -> {total_final_words} ({words_lost} lost)")
    logger.info(f"[{content_id}] Removed {total_stats['empty_segments_removed']} empty segments, {duplicate_count} duplicates, merged {overlap_merges} overlaps")
    logger.info(f"[{content_id}] Cleaned repetitive text in {total_stats['repetitive_text_cleaned']} segments")
    if timing_duplicates_removed > 0:
        logger.info(f"[{content_id}] Removed {timing_duplicates_removed} duplicate words based on timing overlap")
    
    return final_segments, total_stats


def _clean_diarization_segments(diarization_segments: List[Dict], content_id: str) -> Tuple[List[Dict], Dict[str, Any]]:
    """
    Clean diarization segments by removing invalid segments, duplicates, and unknown speakers.
    
    Args:
        diarization_segments: List of diarization segment dictionaries
        content_id: Content ID for logging
        
    Returns:
        Tuple of (cleaned_segments, cleaning_stats)
    """
    logger.info(f"[{content_id}] Cleaning {len(diarization_segments)} diarization segments")
    
    initial_count = len(diarization_segments)
    cleaned_segments = []
    
    # Step 1: Basic validation and remove unknown speakers
    invalid_timing_removed = 0
    zero_duration_removed = 0
    
    for segment in diarization_segments:
        start_time = float(segment.get('start', 0))
        end_time = float(segment.get('end', 0))
        speaker = segment.get('speaker', 'UNKNOWN')
        
        # Keep SPEAKER_99 segments - they will be handled as NEEDS_EMBEDDING later
        # (Previously these were being removed, but they contain valid timing info)
        
        # Skip segments with invalid timing
        if end_time <= start_time:
            invalid_timing_removed += 1
            continue
            
        # Skip segments with zero duration
        duration = end_time - start_time
        if duration <= 0:
            zero_duration_removed += 1
            continue
        
        cleaned_segments.append(segment.copy())
    
    # Step 2: Remove duplicates (same timing and speaker)
    if len(cleaned_segments) > 1:
        unique_segments = []
        seen_segments = set()
        duplicate_count = 0
        
        for segment in cleaned_segments:
            start_time = float(segment.get('start', 0))
            end_time = float(segment.get('end', 0))
            speaker = segment.get('speaker', 'UNKNOWN')
            
            segment_key = (round(start_time, 3), round(end_time, 3), speaker)
            
            if segment_key not in seen_segments:
                seen_segments.add(segment_key)
                unique_segments.append(segment)
            else:
                duplicate_count += 1
        
        cleaned_segments = unique_segments
    else:
        duplicate_count = 0
    
    final_count = len(cleaned_segments)
    
    cleaning_stats = {
        'initial_segments': initial_count,
        'invalid_timing_removed': invalid_timing_removed,
        'zero_duration_removed': zero_duration_removed,
        'duplicate_segments_removed': duplicate_count,
        'final_segments': final_count,
        'total_removed': initial_count - final_count
    }
    
    logger.info(f"[{content_id}] Diarization cleaning: {initial_count} -> {final_count} segments")
    logger.info(f"[{content_id}] Removed: {invalid_timing_removed} invalid timing, {zero_duration_removed} zero duration, "
               f"{duplicate_count} duplicates")
    
    return cleaned_segments, cleaning_stats


def _save_individual_chunks(transcript_data: List[Dict], content_id: str, test_mode: bool = False) -> None:
    """
    Save individual transcript chunks for debugging boundary issues.
    
    Args:
        transcript_data: Raw transcript data from stage 1
        content_id: Content ID for the filename
        test_mode: Whether to save in test mode directory
    """
    if test_mode:
        inputs_dir = get_project_root() / "tests" / "content" / content_id / "inputs"
    else:
        inputs_dir = Path("/tmp/stitch_temp") / content_id / "inputs"
    
    inputs_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract segments from data structure if needed
    segments = transcript_data.get('segments', transcript_data) if isinstance(transcript_data, dict) else transcript_data
    
    # Group segments by chunk (assuming they have chunk info or we can infer from timing)
    chunks_by_index = {}
    
    for segment in segments:
        # Try to infer chunk index from timing (298s chunks with 2s overlap)
        seg_start = float(segment.get('start', 0))
        chunk_index = int(seg_start // 298)  # Rough chunk assignment
        
        if chunk_index not in chunks_by_index:
            chunks_by_index[chunk_index] = []
        chunks_by_index[chunk_index].append(segment)
    
    # Save each chunk
    for chunk_index, chunk_segments in chunks_by_index.items():
        chunk_file = inputs_dir / f"raw_transcript_chunk_{chunk_index}.json"
        
        chunk_data = {
            "content_id": content_id,
            "chunk_index": chunk_index,
            "estimated_time_range": f"{chunk_index * 298:.0f}-{(chunk_index + 1) * 298:.0f}s",
            "segment_count": len(chunk_segments),
            "segments": chunk_segments
        }
        
        with open(chunk_file, 'w') as f:
            json.dump(chunk_data, f, indent=2, cls=NumpyJsonEncoder)
        
        logger.info(f"[{content_id}] Saved chunk {chunk_index} with {len(chunk_segments)} segments to: {chunk_file}")


def _save_combined_transcript(cleaned_transcript_segments: List[Dict], 
                             content_id: str, 
                             test_mode: bool = False) -> str:
    """
    Save the cleaned transcript segments as combined_transcript.json for comparison.
    
    Args:
        cleaned_transcript_segments: List of cleaned transcript segments
        content_id: Content ID for the filename
        test_mode: Whether to save in test mode directory
        
    Returns:
        Path to the saved file
    """
    if test_mode:
        # Save in test inputs directory
        inputs_dir = get_project_root() / "tests" / "content" / content_id / "inputs"
    else:
        # Save in temp directory for production
        inputs_dir = Path("/tmp/stitch_temp") / content_id / "inputs"
    
    inputs_dir.mkdir(parents=True, exist_ok=True)
    
    # Create the combined transcript structure
    combined_transcript = {
        "content_id": content_id,
        "source": "stage2_cleaned",
        "cleaning_applied": True,
        "total_segments": len(cleaned_transcript_segments),
        "segments": cleaned_transcript_segments
    }
    
    # Save the file
    file_path = inputs_dir / "combined_transcript_cleaned.json"
    with open(file_path, 'w') as f:
        json.dump(combined_transcript, f, indent=2, cls=NumpyJsonEncoder)
    
    logger.info(f"[{content_id}] Saved cleaned transcript to: {file_path}")
    return str(file_path)


def clean_stage(content_id: str,
                diarization_data: List[Dict],
                transcript_data: List[Dict],
                test_mode: bool = False) -> Dict[str, Any]:
    """
    Main entry point for Stage 2: Clean raw data and handle chunk boundaries.
    
    This is the primary method called by the stitch pipeline. It performs critical
    cleaning operations on the raw data, especially handling chunk boundary overlaps
    that occur at ~298s intervals due to the chunked transcription process.
    
    Args:
        content_id: Content ID to process (e.g., "Bdb001")
        diarization_data: Raw diarization data from Stage 1 (dict with 'segments' or list)
        transcript_data: Raw transcript data from Stage 1 (dict with 'segments' or list)
        test_mode: If True, saves debug outputs for boundary analysis
        
    Returns:
        Dictionary containing:
        - status: 'success' or 'error'
        - data: Dict with cleaned_diarization_data and cleaned_transcript_data
        - stats: Cleaning statistics (segments removed, words cleaned, etc.)
        - error: Error message if status is 'error'
        
    Example:
        result = clean_stage("Bdb001", diarization_data, transcript_data, test_mode=True)
        if result['status'] == 'success':
            cleaned_diar = result['data']['cleaned_diarization_data']
            cleaned_trans = result['data']['cleaned_transcript_data']
    """
    start_time = time.time()
    
    logger.info(f"[{content_id}] Starting Stage 2: Data Cleaning (Raw Data)")
    
    result = {
        'status': 'pending',
        'content_id': content_id,
        'stage': 'data_cleaning',
        'data': {
            'cleaned_diarization_data': None,
            'cleaned_transcript_data': None
        },
        'stats': {},
        'error': None
    }
    
    try:
        # Validate input data
        if not diarization_data:
            raise ValueError("No diarization data available")
        if not transcript_data:
            raise ValueError("No transcript data available")
        
        # Extract segments from data structures if needed
        diar_segments = diarization_data.get('segments', diarization_data)
        trans_segments = transcript_data.get('segments', transcript_data)

        logger.debug(f"[{content_id}] Processing {len(diar_segments)} diarization segments")
        logger.debug(f"[{content_id}] Processing {len(trans_segments)} transcript segments")

        # Detect transcription model based on word spacing
        model_type = _detect_word_spacing_model(trans_segments)
        logger.info(f"[{content_id}] Detected transcription model type: {model_type}")

        # Save individual chunks for debugging
        logger.info(f"[{content_id}] Saving individual transcript chunks for debugging")
        _save_individual_chunks(transcript_data, content_id, test_mode)

        # Clean transcript data (this is where most artifacts are)
        logger.info(f"[{content_id}] Cleaning transcript data for artifacts and repetitions")
        cleaned_transcript_segments, transcript_cleaning_stats = _clean_transcript_segments(trans_segments, content_id, model_type)
        
        # Clean diarization data (basic validation)
        logger.info(f"[{content_id}] Cleaning diarization data for quality issues")
        cleaned_diarization_segments, diarization_cleaning_stats = _clean_diarization_segments(diar_segments, content_id)
        
        # Save cleaned transcript for comparison
        logger.info(f"[{content_id}] Saving cleaned transcript for comparison")
        transcript_file_path = _save_combined_transcript(cleaned_transcript_segments, content_id, test_mode)
        
        # Prepare cleaned data structures
        if isinstance(diarization_data, dict):
            cleaned_diarization_data = diarization_data.copy()
            cleaned_diarization_data['segments'] = cleaned_diarization_segments
        else:
            cleaned_diarization_data = cleaned_diarization_segments
        
        if isinstance(transcript_data, dict):
            cleaned_transcript_data = transcript_data.copy()
            cleaned_transcript_data['segments'] = cleaned_transcript_segments
        else:
            cleaned_transcript_data = cleaned_transcript_segments
        
        # Store cleaned data
        result['data']['cleaned_diarization_data'] = cleaned_diarization_data
        result['data']['cleaned_transcript_data'] = cleaned_transcript_data
        
        stage_duration = time.time() - start_time
        
        # Count total words for compatibility
        total_final_words = 0
        for segment in cleaned_transcript_segments:
            words = segment.get('words', [])
            if words:
                total_final_words += len(words)
        
        result['stats'] = {
            'duration': stage_duration,
            'transcript_cleaning_stats': transcript_cleaning_stats,
            'diarization_cleaning_stats': diarization_cleaning_stats,
            'final_transcript_segments': len(cleaned_transcript_segments),
            'final_diarization_segments': len(cleaned_diarization_segments),
            'total_final_words': total_final_words,
            'output_files': {
                'cleaned_transcript': transcript_file_path
            }
        }
        
        result['status'] = 'success'
        
        logger.info(f"[{content_id}] Stage 2 completed successfully in {stage_duration:.2f}s")
        
        # Save stage 2 output for debugging in test mode
        if test_mode:
            stage2_output_dir = get_project_root() / "tests" / "content" / content_id / "stage_outputs"
            stage2_output_dir.mkdir(parents=True, exist_ok=True)
            
            stage2_output_file = stage2_output_dir / "stage2_clean_output.json"
            
            # Create detailed output with segment inspection
            stage2_debug_output = {
                "content_id": content_id,
                "stage": "stage2_clean",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "stats": result['stats'],
                "transcript_segments_count": len(cleaned_transcript_segments),
                "diarization_segments_count": len(cleaned_diarization_segments),
                "transcript_segments": []
            }
            
            # Add detailed segment info with word timings
            for idx, segment in enumerate(cleaned_transcript_segments):
                seg_info = {
                    "index": idx,
                    "text": segment.get('text', ''),
                    "start": segment.get('start', 0),
                    "end": segment.get('end', 0),
                    "word_count": len(segment.get('words', [])),
                    "words": []
                }
                
                # Include word details
                words = segment.get('words', [])
                for word in words:
                    word_info = {
                        "word": word.get('word', ''),
                        "start": word.get('start', 0),
                        "end": word.get('end', 0)
                    }
                    seg_info["words"].append(word_info)
                
                stage2_debug_output["transcript_segments"].append(seg_info)
            
            # Save the output
            with open(stage2_output_file, 'w') as f:
                json.dump(stage2_debug_output, f, indent=2, cls=NumpyJsonEncoder)
            
            logger.info(f"[{content_id}] Saved Stage 2 debug output to: {stage2_output_file}")
        
        # Detailed deduplication summary
        total_duplicates = (transcript_cleaning_stats.get('duplicates_removed', 0) + 
                           transcript_cleaning_stats.get('overlap_merges', 0) +
                           diarization_cleaning_stats.get('duplicate_segments_removed', 0))
        
        logger.info(f"[{content_id}] CHUNK BOUNDARY DEDUPLICATION COMPLETE - All duplicate removal done in Stage 2:")
        logger.info(f"[{content_id}] - Transcript overlaps merged: {transcript_cleaning_stats.get('overlap_merges', 0)}")
        logger.info(f"[{content_id}] - Transcript duplicates removed: {transcript_cleaning_stats.get('duplicates_removed', 0)}")
        logger.info(f"[{content_id}] - Empty segments removed: {transcript_cleaning_stats.get('empty_segments_removed', 0)}")
        logger.info(f"[{content_id}] - Diarization segment duplicates: {diarization_cleaning_stats.get('duplicate_segments_removed', 0)}")
        logger.info(f"[{content_id}] - TOTAL SEGMENTS CLEANED: {total_duplicates}")
        
        logger.info(f"[{content_id}] Strategy: Merge overlapping segments at chunk boundaries, keep earlier segment + add missing words")
        logger.info(f"[{content_id}] Other artifacts removed: {transcript_cleaning_stats['total_artifacts_removed']} transcript, {diarization_cleaning_stats['total_removed']} diarization")
        logger.info(f"[{content_id}] Cleaned transcript saved to: {transcript_file_path}")
        logger.info(f"[{content_id}] NOTE: No further deduplication will occur in subsequent stages")
        
        return result
        
    except Exception as e:
        logger.error(f"[{content_id}] Stage 2 failed: {str(e)}")
        logger.error(f"[{content_id}] Error details:", exc_info=True)
        
        result.update({
            'status': 'error',
            'error': str(e),
            'duration': time.time() - start_time
        })
        return result