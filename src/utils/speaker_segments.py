#!/usr/bin/env python3
"""
Speaker Segments Utility

Utility functions for working with speaker-attributed text in embedding segments.
Converts compact speaker_positions format into human-readable speaker-attributed text.
"""

from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def format_speaker_attributed_text(text: str, speaker_positions: Dict[str, List[List[int]]]) -> str:
    """
    Format segment text with speaker attribution using speaker_positions map.

    Args:
        text: The full segment text
        speaker_positions: Dict mapping speaker_id (from speakers table) to character ranges
                          Example: {"2668417": [[0, 280]], "2668422": [[281, 315], [650, 750]]}
                          Note: Keys are speaker database IDs stored as strings (JSONB format)

    Returns:
        Formatted text with speaker IDs, separated by double newlines.
        Example:
            2668417: First chunk of text from speaker 1

            2668422: Text from speaker 2

            2668417: Speaker 1 speaking again
    """
    if not speaker_positions or not text:
        return text

    # Extract all position ranges with their speaker labels
    # Format: [(start, end, speaker_label), ...]
    ranges: List[Tuple[int, int, str]] = []

    for speaker_label, position_list in speaker_positions.items():
        for start, end in position_list:
            ranges.append((start, end, speaker_label))

    # Sort by start position to maintain text order
    ranges.sort(key=lambda x: x[0])

    # Build formatted output
    formatted_parts = []

    for start, end, speaker_label in ranges:
        # Extract text for this range
        speaker_text = text[start:end].strip()

        if speaker_text:  # Only add non-empty text
            formatted_parts.append(f"{speaker_label}: {speaker_text}")

    # Join with double newlines for clear speaker separation
    return "\n\n".join(formatted_parts)


def merge_consecutive_speaker_ranges(speaker_positions: Dict[str, List[List[int]]]) -> Dict[str, List[List[int]]]:
    """
    Merge consecutive or overlapping ranges for each speaker.

    This is useful for cleaning up speaker_positions that may have been
    built incrementally with adjacent ranges.

    Args:
        speaker_positions: Dict mapping speaker labels to character ranges

    Returns:
        Dict with merged ranges for each speaker

    Example:
        Input:  {"SPEAKER_00": [[0, 100], [100, 200], [300, 400]]}
        Output: {"SPEAKER_00": [[0, 200], [300, 400]]}
    """
    merged = {}

    for speaker_label, ranges in speaker_positions.items():
        if not ranges:
            merged[speaker_label] = []
            continue

        # Sort ranges by start position
        sorted_ranges = sorted(ranges, key=lambda x: x[0])

        # Merge consecutive/overlapping ranges
        merged_ranges = [sorted_ranges[0]]

        for start, end in sorted_ranges[1:]:
            last_start, last_end = merged_ranges[-1]

            # If current range starts at or before last range ends, merge them
            if start <= last_end:
                merged_ranges[-1] = [last_start, max(last_end, end)]
            else:
                merged_ranges.append([start, end])

        merged[speaker_label] = merged_ranges

    return merged


def validate_speaker_positions(text: str, speaker_positions: Dict[str, List[List[int]]]) -> Tuple[bool, List[str]]:
    """
    Validate that speaker_positions are valid for the given text.

    Checks:
    - All ranges are within text bounds
    - No overlapping ranges between speakers
    - Ranges are sorted and valid [start, end] with start < end

    Args:
        text: The segment text
        speaker_positions: Dict mapping speaker labels to character ranges

    Returns:
        Tuple of (is_valid, list_of_error_messages)
    """
    errors = []
    text_len = len(text)

    # Track all occupied ranges to detect overlaps
    all_ranges = []

    for speaker_label, ranges in speaker_positions.items():
        for range_idx, (start, end) in enumerate(ranges):
            # Check range validity
            if start < 0:
                errors.append(f"{speaker_label} range {range_idx}: start ({start}) is negative")

            if end > text_len:
                errors.append(f"{speaker_label} range {range_idx}: end ({end}) exceeds text length ({text_len})")

            if start >= end:
                errors.append(f"{speaker_label} range {range_idx}: start ({start}) >= end ({end})")

            # Track range for overlap detection
            all_ranges.append((start, end, speaker_label, range_idx))

    # Check for overlaps between different speakers
    all_ranges.sort(key=lambda x: x[0])

    for i in range(len(all_ranges) - 1):
        start1, end1, speaker1, idx1 = all_ranges[i]
        start2, end2, speaker2, idx2 = all_ranges[i + 1]

        # Overlaps if next range starts before current range ends
        if start2 < end1 and speaker1 != speaker2:
            errors.append(
                f"Overlap detected: {speaker1} range {idx1} [{start1}:{end1}] "
                f"overlaps with {speaker2} range {idx2} [{start2}:{end2}]"
            )

    return len(errors) == 0, errors


def get_speakers_from_positions(speaker_positions: Dict[str, List[List[int]]]) -> List[str]:
    """
    Extract sorted list of speaker labels from speaker_positions.

    Args:
        speaker_positions: Dict mapping speaker labels to character ranges

    Returns:
        Sorted list of speaker labels (e.g., ["SPEAKER_00", "SPEAKER_01"])
    """
    return sorted(speaker_positions.keys())


def calculate_speaker_text_coverage(text: str, speaker_positions: Dict[str, List[List[int]]]) -> Dict[str, float]:
    """
    Calculate what percentage of text each speaker covers.

    Args:
        text: The segment text
        speaker_positions: Dict mapping speaker labels to character ranges

    Returns:
        Dict mapping speaker labels to coverage percentage (0.0 to 1.0)
    """
    if not text or not speaker_positions:
        return {}

    text_len = len(text)
    coverage = {}

    for speaker_label, ranges in speaker_positions.items():
        total_chars = sum(end - start for start, end in ranges)
        coverage[speaker_label] = total_chars / text_len if text_len > 0 else 0.0

    return coverage
