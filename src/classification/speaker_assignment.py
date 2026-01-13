#!/usr/bin/env python3
"""Speaker assignment module for embedding segments.

This module provides functions to assign speakers to portions of text in embedding segments
based on the associated speaker transcriptions.
"""

import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class SpeakerSegment:
    """Represents a portion of text attributed to a speaker"""
    speaker_label: str
    text: str
    start_offset: int
    end_offset: int
    confidence: float = 1.0

def assign_speakers_to_segment(
    segment_text: str,
    transcripts: List[Dict],
    segment_start: float,
    segment_end: float
) -> List[SpeakerSegment]:
    """
    Assign speakers to portions of segment text based on transcript boundaries.
    
    Args:
        segment_text: The text of the embedding segment
        transcripts: List of transcript dictionaries with speaker info
        segment_start: Start time of the segment
        segment_end: End time of the segment
    
    Returns:
        List of SpeakerSegment objects with speaker attributions
    """
    
    if not transcripts:
        # No speaker info available
        return [SpeakerSegment(
            speaker_label="UNKNOWN",
            text=segment_text,
            start_offset=0,
            end_offset=len(segment_text),
            confidence=0.0
        )]
    
    assignments = []
    current_pos = 0
    
    for i, trans in enumerate(transcripts):
        # Get transcript info
        trans_text = trans.get('trans_text', '')
        speaker_label = trans.get('speaker_label', f"SPEAKER_{trans.get('speaker_id', 'UNK')}")
        trans_start = trans.get('trans_start', 0)
        trans_end = trans.get('trans_end', 0)
        alignment_case = trans.get('alignment_case', 'unknown')
        
        if alignment_case == 'fully_contained':
            # Try to find the full transcript text in the segment
            pos = segment_text.find(trans_text, current_pos)
            if pos >= 0:
                assignments.append(SpeakerSegment(
                    speaker_label=speaker_label,
                    text=trans_text,
                    start_offset=pos,
                    end_offset=pos + len(trans_text),
                    confidence=1.0
                ))
                current_pos = pos + len(trans_text)
            else:
                # Try fuzzy matching (first 50 chars)
                sample = trans_text[:50] if len(trans_text) > 50 else trans_text
                pos = segment_text.find(sample, current_pos)
                if pos >= 0:
                    # Find where this transcript likely ends
                    if i < len(transcripts) - 1:
                        # Use next transcript's start as boundary
                        next_sample = transcripts[i+1].get('trans_text', '')[:50]
                        end_pos = segment_text.find(next_sample, pos) if next_sample else -1
                        if end_pos > pos:
                            text_portion = segment_text[pos:end_pos].strip()
                        else:
                            text_portion = segment_text[pos:].strip()
                    else:
                        text_portion = segment_text[pos:].strip()
                    
                    assignments.append(SpeakerSegment(
                        speaker_label=speaker_label,
                        text=text_portion,
                        start_offset=pos,
                        end_offset=pos + len(text_portion),
                        confidence=0.8
                    ))
                    current_pos = pos + len(text_portion)
        
        elif alignment_case == 'starts_before':
            # The segment should start with part of this transcript
            if i < len(transcripts) - 1:
                # Use the beginning of the next transcript as boundary
                next_text_start = transcripts[i+1].get('trans_text', '')[:50]
                boundary_pos = segment_text.find(next_text_start, current_pos)
                if boundary_pos > current_pos:
                    text_portion = segment_text[current_pos:boundary_pos].strip()
                else:
                    # Fallback: use time proportion
                    overlap_duration = segment_end - segment_start
                    trans_duration = trans_end - trans_start
                    proportion = overlap_duration / trans_duration if trans_duration > 0 else 1
                    estimated_chars = int(len(trans_text) * proportion)
                    text_portion = segment_text[:estimated_chars].strip()
            else:
                # This is the last transcript, take everything from current position
                text_portion = segment_text[current_pos:].strip()
            
            if text_portion:
                assignments.append(SpeakerSegment(
                    speaker_label=speaker_label,
                    text=text_portion,
                    start_offset=current_pos,
                    end_offset=current_pos + len(text_portion),
                    confidence=0.7
                ))
                current_pos = current_pos + len(text_portion)
        
        elif alignment_case == 'ends_after':
            # The segment should end with part of this transcript
            remaining_text = segment_text[current_pos:].strip()
            if remaining_text:
                assignments.append(SpeakerSegment(
                    speaker_label=speaker_label,
                    text=remaining_text,
                    start_offset=current_pos,
                    end_offset=len(segment_text),
                    confidence=0.7
                ))
                current_pos = len(segment_text)
        
        elif alignment_case == 'both_outside':
            # Transcript spans entire segment
            assignments.append(SpeakerSegment(
                speaker_label=speaker_label,
                text=segment_text,
                start_offset=0,
                end_offset=len(segment_text),
                confidence=0.9
            ))
            current_pos = len(segment_text)
    
    # Handle any remaining text
    if current_pos < len(segment_text):
        remaining_text = segment_text[current_pos:].strip()
        if remaining_text and assignments:
            # Attribute to last known speaker
            assignments.append(SpeakerSegment(
                speaker_label=assignments[-1].speaker_label,
                text=remaining_text,
                start_offset=current_pos,
                end_offset=len(segment_text),
                confidence=0.5
            ))
        elif remaining_text:
            # No previous speaker, mark as unknown
            assignments.append(SpeakerSegment(
                speaker_label="UNKNOWN",
                text=remaining_text,
                start_offset=current_pos,
                end_offset=len(segment_text),
                confidence=0.0
            ))
    
    return assignments

def format_speaker_attributed_text(speaker_segments: List[SpeakerSegment]) -> str:
    """
    Format speaker segments into a readable speaker-attributed text.
    
    Args:
        speaker_segments: List of SpeakerSegment objects
    
    Returns:
        Formatted text with speaker labels
    """
    
    if not speaker_segments:
        return ""
    
    formatted_parts = []
    current_speaker = None
    current_text = []
    
    for segment in speaker_segments:
        if segment.speaker_label != current_speaker:
            # Speaker change - output accumulated text
            if current_text:
                formatted_parts.append(f"{current_speaker}: {' '.join(current_text)}")
            current_speaker = segment.speaker_label
            current_text = [segment.text]
        else:
            # Same speaker - accumulate text
            current_text.append(segment.text)
    
    # Don't forget the last speaker's text
    if current_text:
        formatted_parts.append(f"{current_speaker}: {' '.join(current_text)}")
    
    return '\n\n'.join(formatted_parts)

def get_speaker_info_summary(speaker_segments: List[SpeakerSegment]) -> Dict:
    """
    Generate a summary of speaker information.
    
    Args:
        speaker_segments: List of SpeakerSegment objects
    
    Returns:
        Dictionary with speaker statistics and info
    """
    
    speaker_stats = {}
    total_chars = 0
    
    for segment in speaker_segments:
        speaker = segment.speaker_label
        text_length = len(segment.text)
        
        if speaker not in speaker_stats:
            speaker_stats[speaker] = {
                'char_count': 0,
                'segment_count': 0,
                'avg_confidence': 0.0
            }
        
        speaker_stats[speaker]['char_count'] += text_length
        speaker_stats[speaker]['segment_count'] += 1
        speaker_stats[speaker]['avg_confidence'] += segment.confidence
        total_chars += text_length
    
    # Calculate averages and percentages
    for speaker in speaker_stats:
        stats = speaker_stats[speaker]
        stats['avg_confidence'] = stats['avg_confidence'] / stats['segment_count']
        stats['percentage'] = (stats['char_count'] / total_chars * 100) if total_chars > 0 else 0
    
    return {
        'speakers': list(speaker_stats.keys()),
        'speaker_count': len(speaker_stats),
        'speaker_stats': speaker_stats,
        'total_characters': total_chars
    }