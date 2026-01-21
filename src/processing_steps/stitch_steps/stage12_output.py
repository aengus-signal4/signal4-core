#!/usr/bin/env python3
"""
Stage 12: Output Generation and Database Storage
================================================

Final stage of the stitch pipeline that generates outputs and saves results to database.

Key Responsibilities:
- Generate readable transcript with speaker labels and timestamps
- Create detailed word-by-word analysis showing assignment methods and confidence
- Generate speaker turns for file output (debugging only)
- Generate sentences (atomic unit) and save to database
- Create assignment history reports and visualizations for debugging

Input:
- WordTable from Stage 11 with final speaker assignments and cleanup
- Output directory for file generation (optional)
- Test mode flag for database simulation

Output:
- Readable transcript: [30s] [00:30] [SPEAKER_00]: {speaker text}
- Detailed transcript: Word-by-word breakdown with timing, methods, confidence
- Speaker turns: For file output/debugging (not saved to database)
- Sentences: Atomic units saved to sentences table
- Assignment history: Complete debugging report of all assignments

Key Components:
- format_timestamp(): Dual timestamp formatting ([30s] [00:30])
- generate_readable_transcript(): Clean speaker-labeled transcript
- generate_detailed_transcript(): Technical word-by-word analysis
- generate_speaker_turns(): Speaker turns for file output
- generate_sentences(): Sentence-level atomic units
- save_sentences_to_database(): Database persistence for sentences

Methods:
- stage12_output(): Main entry point called by stitch pipeline
- format_method_8char(): Standardized method name formatting
- _get_comprehensive_speaker_mappings(): Multi-source speaker database lookups

Performance:
- Minimal computational cost (1-2% of pipeline time)
- Handles large transcripts efficiently with streaming processing
- Comprehensive database integration with proper error handling
- Generates multiple output formats for different use cases

Special Features:
- Multi-layer speaker identification (universal name, global ID, diarization)
- Sankey visualization generation in test mode
- Comprehensive speaker mapping from multiple sources
- Assignment history tracking for debugging
- Test mode database simulation
"""

import logging
import time
from pathlib import Path
from src.utils.paths import get_project_root, get_config_path
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import pandas as pd
import yaml
import numpy as np
import re
import hashlib
import json

from src.utils.logger import setup_worker_logger
from src.utils.config import load_config
from .stage3_tables import WordTable
from src.database.models import Content, Sentence
from src.database.session import get_session
from .util_stitch import smart_join_words

logger = setup_worker_logger('stitch')

def format_timestamp(seconds: float) -> str:
    """Format timestamp in both [30s] and [MM:SS] formats."""
    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)
    return f"[{int(seconds)}s] [{minutes:02d}:{remaining_seconds:02d}]"

def generate_readable_transcript(word_table: WordTable) -> str:
    """
    Generate a readable transcript with timestamps and speaker labels.
    Includes ALL speakers (including UNKNOWN and category labels) for transparency.
    Format: [30s] [00:30] [Speaker 12345] [SPEAKER_a1b2c3d4] [SPEAKER_00]: {text}
    """
    if word_table is None or word_table.df is None:
        logger.error("Word table or dataframe is None in generate_readable_transcript")
        return "ERROR: No word table data available"
    
    # Group words by speaker segments
    transcript_lines = []
    current_speaker = None
    current_speaker_info = {}
    current_text = []
    segment_start_time = None
    last_end_time = 0
    
    # Include ALL words with any speaker assignment (including UNKNOWN and category labels)
    # This ensures complete transparency in the final transcript
    valid_words = word_table.df[
        (word_table.df['speaker_current'].notna()) & 
        (word_table.df['speaker_current'] != '')
    ].copy()
    
    if len(valid_words) == 0:
        logger.error("No words with speaker assignments found")
        return "ERROR: No words with speaker assignments"
    
    logger.info(f"Generating transcript from {len(valid_words)} words with speaker assignments (filtered from {len(word_table.df)} total)")
    sorted_words = valid_words.sort_values('start')
    
    for _, word in sorted_words.iterrows():
        # Extract speaker information from metadata if available
        metadata = word.get('metadata', {})
        if isinstance(metadata, dict):
            speaker_mapping = metadata.get('speaker_db_mapping', {})
        else:
            speaker_mapping = {}
        
        # Check if this is a new speaker or significant time gap (>2s)
        is_new_segment = (
            word['speaker_current'] != current_speaker
        )
        
        if is_new_segment:
            # Output previous segment if exists
            if current_text and current_speaker:
                timestamp = format_timestamp(segment_start_time)
                
                # Build speaker label with all identifiers
                speaker_parts = []
                
                # Universal name (e.g., Speaker 12345)
                if current_speaker_info.get('universal_name'):
                    speaker_parts.append(f"[{current_speaker_info['universal_name']}]")
                
                # Global ID (e.g., SPEAKER_a1b2c3d4)
                if current_speaker_info.get('global_id'):
                    speaker_parts.append(f"[{current_speaker_info['global_id']}]")
                
                # Diarization speaker (e.g., SPEAKER_00)
                speaker_parts.append(f"[{current_speaker}]")
                
                speaker_label = " ".join(speaker_parts)
                transcript_lines.append(f"{timestamp} {speaker_label}: {smart_join_words(current_text)}")
            
            # Reset for new segment
            current_speaker = word['speaker_current']
            current_speaker_info = speaker_mapping.copy()
            current_text = [word['text']]
            segment_start_time = word['start']
        else:
            # Continue current segment
            current_text.append(word['text'])
        
        last_end_time = word['end']
    
    # Add final segment
    if current_text and current_speaker:
        timestamp = format_timestamp(segment_start_time)
        
        # Build speaker label with all identifiers
        speaker_parts = []
        
        # Universal name (e.g., Speaker 12345)
        if current_speaker_info.get('universal_name'):
            speaker_parts.append(f"[{current_speaker_info['universal_name']}]")
        
        # Global ID (e.g., SPEAKER_a1b2c3d4)
        if current_speaker_info.get('global_id'):
            speaker_parts.append(f"[{current_speaker_info['global_id']}]")
        
        # Diarization speaker (e.g., SPEAKER_00)
        speaker_parts.append(f"[{current_speaker}]")
        
        speaker_label = " ".join(speaker_parts)
        transcript_lines.append(f"{timestamp} {speaker_label}: {smart_join_words(current_text)}")
    
    return "\n".join(transcript_lines)

def format_method_8char(method) -> str:
    """Format method name to exactly 8 characters for alignment."""
    # Handle non-string values (e.g., NaN, None, floats)
    if method is None or (isinstance(method, float) and pd.isna(method)):
        method = 'none'
    elif not isinstance(method, str):
        method = str(method)
    
    # Map of method names to 8-character versions
    method_map = {
        # Stage 5: Overlap-Based Speaker Assignment
        'diarization_overlap': '5_diar_ov',
        'no_diarization_overlap': '5_no_diar',
        'multi_speaker_overlap': '5_multi_s',
        'near_speaker_transition': '5_near_tr',
        'same_speaker_absorption': '5_absorb',
        
        # Stage 6: LLM-Based Speaker Assignment
        'obvious_context_assignment': '6_obvious',
        'llm_option_selection': '6_llm_opt',
        
        # Legacy Stage 5: Speaker Assignment (keeping for compatibility)
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
        
        # Stage 11: LLM Coherence
        'llm_before': '9_llm_bef',
        'llm_after': '9_llm_aft',
        'llm_split': '9_llm_spl',
        'llm_split_first': '9_llm_sp1',
        'llm_split_second': '9_llm_sp2',
        'llm_split_fallback_before': '9_llm_fb1',
        'same_speaker_context': '9_same_sp',
        
        # Stage 10: Sentence Consolidation
        'llm_leave_split': '10_lv_spl',
        'llm_consolidate_previous': '10_con_pr',
        'llm_consolidate_following': '10_con_fo',
        
        # Default/initial
        'none': 'none    ',
        
        # Add any other methods encountered
    }
    
    # Return mapped version or truncate/pad to 8 chars
    if method in method_map:
        return method_map[method]
    else:
        # Truncate or pad to exactly 8 characters
        if len(method) > 8:
            return method[:8]
        else:
            return method.ljust(8)

def generate_detailed_transcript(word_table: WordTable) -> str:
    """
    Generate a detailed word-by-word transcript showing speaker assignment reasoning.
    
    Shows for each word:
    - Speaker assignment and confidence
    - Assignment method (Stage 5 overlap, Stage 6 LLM, etc.)
    - Method-specific details:
      * Stage 5: boundary tolerance (tol=0.35s), overlap ratio (ovr=85%)
      * Stage 6: LLM option selected (opt=A), split point (split=2)
    - Assignment reasoning/metadata
    
    Format: [word] [timing] [method] [confidence] [details]
    Example: hello                [45.23s] [00:45] [5_diar_ov] [0.9] tol=0.35s ovr=85%
    """
    detailed_lines = []
    
    # Include ALL words with any speaker assignment (including UNKNOWN and category labels)
    # This ensures complete transparency in the final transcript
    valid_words = word_table.df[
        (word_table.df['speaker_current'].notna()) & 
        (word_table.df['speaker_current'] != '')
    ].copy()
    
    if len(valid_words) == 0:
        logger.error("No words with speaker assignments found for detailed transcript")
        return "ERROR: No words with speaker assignments"
    
    sorted_words = valid_words.sort_values('start')
    
    current_speaker = None
    current_speaker_words = []
    speaker_start_time = None
    
    for _, word in sorted_words.iterrows():
        # Check if this is a new speaker segment
        if word['speaker_current'] != current_speaker:
            # Output previous speaker segment if exists
            if current_speaker is not None and current_speaker_words:
                # Create speaker header line
                timestamp = format_timestamp(speaker_start_time)
                speaker_label = f"[{current_speaker}]" if current_speaker != "UNKNOWN" else "[UNKNOWN]"
                text = smart_join_words([w['text'] for w in current_speaker_words])
                detailed_lines.append(f"{timestamp} {speaker_label} {text}")
                
                # Add indented word details
                for w in current_speaker_words:
                    word_text = w['text']
                    # Pad word to 20 characters for alignment
                    padded_word = f"{word_text:<20}"
                    
                    word_timestamp = format_timestamp(w['start'])
                    method = format_method_8char(w.get('assignment_method', w.get('resolution_method', 'unknown')))
                    
                    # Get confidence from the correct column or metadata
                    confidence = w.get('assignment_confidence', 0.0)
                    if pd.isna(confidence):
                        # Try getting from metadata if not in main columns
                        metadata = w.get('metadata', {})
                        if isinstance(metadata, dict):
                            confidence = metadata.get('assignment_confidence', 0.0)
                    
                    # Handle NaN confidence values
                    if pd.isna(confidence):
                        confidence_str = "nan"
                    else:
                        confidence_str = f"{confidence:.1f}"
                    
                    # Extract assignment details from metadata
                    assignment_details = ""
                    metadata = w.get('metadata', {})
                    
                    # Debug: Log metadata for first few words to see what's available
                    if len(detailed_lines) < 5:  # Only for first few words
                        logger.debug(f"Word '{w['text']}' metadata keys: {list(metadata.keys()) if isinstance(metadata, dict) else 'not dict'}")
                        if isinstance(metadata, dict) and 'secondary_speaker' in metadata:
                            logger.debug(f"  secondary_speaker: {metadata.get('secondary_speaker')}")
                            logger.debug(f"  secondary_distance: {metadata.get('secondary_distance')}")
                    
                    if isinstance(metadata, dict):
                        # Stage 5 overlap details
                        if 'boundary_tolerance' in metadata:
                            assignment_details += f" tol={metadata['boundary_tolerance']:.2f}s"
                        if 'overlap_ratio' in metadata:
                            assignment_details += f" ovr={metadata['overlap_ratio']:.1%}"
                        
                        # Secondary diarization distance (new Stage 5 feature)
                        if 'secondary_speaker' in metadata and metadata['secondary_speaker']:
                            secondary_speaker = metadata['secondary_speaker']
                            secondary_distance = metadata.get('secondary_distance', float('inf'))
                            expanded_distance = metadata.get('expanded_secondary_distance', float('inf'))
                            secondary_overlap = metadata.get('secondary_overlap_ratio', 0.0)
                            
                            # Show secondary speaker info
                            assignment_details += f" sec={secondary_speaker}"
                            
                            # Show distances (use expanded if significantly different)
                            if expanded_distance != secondary_distance and expanded_distance < float('inf'):
                                assignment_details += f" d={secondary_distance:.2f}s({expanded_distance:.2f}s)"
                            elif secondary_distance < float('inf'):
                                assignment_details += f" d={secondary_distance:.2f}s"
                            
                            # Show secondary overlap if significant
                            if secondary_overlap > 0.01:
                                assignment_details += f" sovr={secondary_overlap:.1%}"
                        
                        # Multi-speaker boundary detection details
                        if 'min_boundary_distance' in metadata:
                            boundary_dist = metadata['min_boundary_distance']
                            assignment_details += f" bdist={boundary_dist:.2f}s"
                        
                        # Stage 6 LLM details
                        if 'selected_option' in metadata:
                            assignment_details += f" opt={metadata['selected_option']}"
                        if 'split_point' in metadata and metadata['split_point'] >= 0:
                            assignment_details += f" split={metadata['split_point']}"
                        
                        # General assignment details
                        if 'reason' in metadata:
                            reason = metadata['reason'][:30] if len(metadata['reason']) > 30 else metadata['reason']
                            assignment_details += f" ({reason})"
                    
                    detailed_lines.append(f"    - {padded_word} [{w['start']:.2f}s] {word_timestamp} [{method}] [{confidence_str}]{assignment_details}")
            
            # Reset for new speaker
            current_speaker = word['speaker_current']
            current_speaker_words = []
            speaker_start_time = word['start']
        
        # Add word to current speaker segment (convert Series to dict for consistent access)
        current_speaker_words.append(word.to_dict())
    
    # Output final speaker segment
    if current_speaker is not None and current_speaker_words:
        timestamp = format_timestamp(speaker_start_time)
        speaker_label = f"[{current_speaker}]" if current_speaker != "UNKNOWN" else "[UNKNOWN]"
        text = smart_join_words([w['text'] for w in current_speaker_words])
        detailed_lines.append(f"{timestamp} {speaker_label} {text}")
        
        # Add indented word details
        for w in current_speaker_words:
            word_text = w['text']
            # Pad word to 20 characters for alignment
            padded_word = f"{word_text:<20}"
            
            word_timestamp = format_timestamp(w['start'])
            method = format_method_8char(w.get('assignment_method', w.get('resolution_method', 'unknown')))
            
            # Get confidence from the correct column or metadata
            confidence = w.get('assignment_confidence', 0.0)
            if pd.isna(confidence):
                # Try getting from metadata if not in main columns
                metadata = w.get('metadata', {})
                if isinstance(metadata, dict):
                    confidence = metadata.get('assignment_confidence', 0.0)
            
            # Handle NaN confidence values
            if pd.isna(confidence):
                confidence_str = "nan"
            else:
                confidence_str = f"{confidence:.1f}"
            
            # Extract assignment details from metadata
            assignment_details = ""
            metadata = w.get('metadata', {})
            
            # Debug: Log metadata for first few words to see what's available
            if len(detailed_lines) < 5:  # Only for first few words
                logger.debug(f"Word '{w['text']}' metadata keys: {list(metadata.keys()) if isinstance(metadata, dict) else 'not dict'}")
                if isinstance(metadata, dict) and 'secondary_speaker' in metadata:
                    logger.debug(f"  secondary_speaker: {metadata.get('secondary_speaker')}")
                    logger.debug(f"  secondary_distance: {metadata.get('secondary_distance')}")
            
            if isinstance(metadata, dict):
                # Stage 5 overlap details
                if 'boundary_tolerance' in metadata:
                    assignment_details += f" tol={metadata['boundary_tolerance']:.2f}s"
                if 'overlap_ratio' in metadata:
                    assignment_details += f" ovr={metadata['overlap_ratio']:.1%}"
                
                # Secondary diarization distance (new Stage 5 feature)
                if 'secondary_speaker' in metadata and metadata['secondary_speaker']:
                    secondary_speaker = metadata['secondary_speaker']
                    secondary_distance = metadata.get('secondary_distance', float('inf'))
                    expanded_distance = metadata.get('expanded_secondary_distance', float('inf'))
                    secondary_overlap = metadata.get('secondary_overlap_ratio', 0.0)
                    
                    # Show secondary speaker info
                    assignment_details += f" sec={secondary_speaker}"
                    
                    # Show distances (use expanded if significantly different)
                    if expanded_distance != secondary_distance and expanded_distance < float('inf'):
                        assignment_details += f" d={secondary_distance:.2f}s({expanded_distance:.2f}s)"
                    elif secondary_distance < float('inf'):
                        assignment_details += f" d={secondary_distance:.2f}s"
                    
                    # Show secondary overlap if significant
                    if secondary_overlap > 0.01:
                        assignment_details += f" sovr={secondary_overlap:.1%}"
                
                # Multi-speaker boundary detection details
                if 'min_boundary_distance' in metadata:
                    boundary_dist = metadata['min_boundary_distance']
                    assignment_details += f" bdist={boundary_dist:.2f}s"
                
                # Stage 6 LLM details
                if 'selected_option' in metadata:
                    assignment_details += f" opt={metadata['selected_option']}"
                if 'split_point' in metadata and metadata['split_point'] >= 0:
                    assignment_details += f" split={metadata['split_point']}"
                
                # General assignment details
                if 'reason' in metadata:
                    reason = metadata['reason'][:30] if len(metadata['reason']) > 30 else metadata['reason']
                    assignment_details += f" ({reason})"
            
            detailed_lines.append(f"    - {padded_word} [{w['start']:.2f}s] {word_timestamp} [{method}] [{confidence_str}]{assignment_details}")
    
    return "\n".join(detailed_lines)

def get_current_stitch_version() -> str:
    """Get the current stitch version from config.
    
    Returns:
        Current stitch version string (e.g., 'stitch_v13' or 'stitch_v13.1')
        
    Note:
        This function supports both major versions (stitch_v13) and sub-versions (stitch_v13.1).
        For version compatibility checking, use the functions in src.utils.version_utils.
    """
    try:
        config_path = get_config_path()
        logger.debug(f"Loading stitch version from config at: {config_path}")
        with open(config_path) as f:
            config = yaml.safe_load(f)
        version = config.get('processing', {}).get('stitch', {}).get('current_version', 'stitch_v1')
        logger.debug(f"Got stitch version from config: {version}")
        return version
    except Exception as e:
        logger.warning(f"Failed to load stitch version from config: {e}, using default 'stitch_v1'")
        logger.warning(f"Config path attempted: {get_project_root() / 'config' / 'config.yaml'}")
        return 'stitch_v1'

def generate_speaker_turns(word_table: WordTable, content_id: str) -> Dict[str, Any]:
    """
    Generate speaker turns from the word table.
    
    A speaker turn is a continuous segment of speech by one speaker.
    Returns a dictionary with metadata and turns ready for database insertion.
    
    This function separates turn generation from database ID lookup for better robustness.
    """
    if word_table is None or word_table.df is None:
        logger.error("Word table or dataframe is None in generate_speaker_turns")
        return {'error': 'No word table data available'}
    
    # Sort words by start time without filtering
    sorted_words = word_table.df.sort_values('start')
    
    if len(sorted_words) == 0:
        logger.error("No words found in word table")
        return {'error': 'No words found'}
    
    # Calculate content duration from word timings
    content_duration = sorted_words['end'].max() - sorted_words['start'].min()
    
    # Get processing start time from metadata if available
    processing_start = None
    processing_time = None
    processing_speed = None
    
    if hasattr(word_table, 'metadata'):
        metadata = word_table.metadata
        if isinstance(metadata, dict):
            processing_start = metadata.get('processing_start')
            processing_time = metadata.get('processing_time')
            
            # Calculate processing speed (e.g. 30x means it processed 300s audio in 10s)
            if processing_time and content_duration:
                processing_speed = content_duration / processing_time
    
    # STEP 1: Generate raw speaker turns (without database IDs)
    raw_turns = []
    current_speaker = None
    current_text = []
    turn_start_time = None
    turn_end_time = None
    
    for _, word in sorted_words.iterrows():
        current_speaker_name = word['speaker_current']
        
        # Check if this is a new speaker
        is_new_turn = current_speaker_name != current_speaker
        
        if is_new_turn:
            # Save previous turn if exists
            if current_text and current_speaker:
                raw_turns.append({
                    'speaker_name': current_speaker,
                    'start_time': turn_start_time,
                    'end_time': turn_end_time,
                    'text': smart_join_words(current_text),
                    'word_count': len(current_text)
                })
            
            # Start new turn
            current_speaker = current_speaker_name
            current_text = [word['text']]
            turn_start_time = word['start']
            turn_end_time = word['end']
        else:
            # Continue current turn
            current_text.append(word['text'])
            turn_end_time = word['end']
    
    # Add final turn
    if current_text and current_speaker:
        raw_turns.append({
            'speaker_name': current_speaker,
            'start_time': turn_start_time,
            'end_time': turn_end_time,
            'text': smart_join_words(current_text),
            'word_count': len(current_text)
        })
    
    logger.info(f"[{content_id}] Generated {len(raw_turns)} raw speaker turns")
    
    # STEP 2: Comprehensive speaker mapping lookup
    speaker_mappings = _get_comprehensive_speaker_mappings(word_table, content_id)
    
    # STEP 3: Convert raw turns to database-ready turns
    final_turns = []
    skipped_turns = []
    turn_index = 0
    
    for turn in raw_turns:
        speaker_name = turn['speaker_name']
        
        if speaker_name in speaker_mappings:
            mapping = speaker_mappings[speaker_name]
            final_turns.append({
                'content_id': content_id,  # Platform content_id for reference
                'speaker_id': mapping['speaker_db_id'],
                'start_time': turn['start_time'],
                'end_time': turn['end_time'],
                'text': turn['text'],
                'turn_index': turn_index,
                'stitch_version': get_current_stitch_version(),
                'diarization_speaker': speaker_name,
                'universal_name': mapping.get('universal_name', 'Unknown'),
                'global_id': mapping.get('global_id', 'Unknown')
            })
            turn_index += 1
        else:
            skipped_turns.append({
                'speaker_name': speaker_name,
                'start_time': turn['start_time'],
                'end_time': turn['end_time'],
                'word_count': turn['word_count'],
                'reason': 'no_database_mapping'
            })
    
    # Log results
    logger.info(f"[{content_id}] Final result: {len(final_turns)} turns with database IDs, {len(skipped_turns)} turns skipped")
    
    if skipped_turns:
        logger.warning(f"[{content_id}] Skipped turns for speakers without database mappings:")
        for skip in skipped_turns:
            logger.warning(f"[{content_id}]   {skip['speaker_name']}: {skip['word_count']} words, {skip['start_time']:.1f}-{skip['end_time']:.1f}s")
    
    # Return dictionary with metadata and turns
    return {
        'metadata': {
            'content_id': content_id,
            'content_duration': content_duration,
            'total_words': len(sorted_words),
            'raw_turns_generated': len(raw_turns),
            'final_turns_with_db_ids': len(final_turns),
            'skipped_turns': len(skipped_turns),
            'processing_start': processing_start,
            'processing_time': processing_time,
            'processing_speed': f"{processing_speed:.1f}x" if processing_speed else None,
            'stitch_version': get_current_stitch_version(),
            'generated_at': pd.Timestamp.now().isoformat()
        },
        'speaker_turns': final_turns,
        'skipped_turns': skipped_turns,
        'speaker_mappings_used': speaker_mappings
    }


def generate_sentences(word_table: WordTable, content_id: str, speaker_mappings: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Generate sentence-level data from the word table with precise word-level timestamps.

    Sentences are the atomic unit for:
    - Emotion detection (sentence-level audio processing)
    - Fine-grained queries ("what did speaker X say when angry")
    - Building embedding segments (source_sentence_ids)

    Each sentence includes:
    - sentence_index: Global index within content (0, 1, 2...)
    - turn_index: Which speaker turn this belongs to
    - sentence_in_turn: Position within turn (0, 1, 2...)
    - Precise word-level start_time and end_time
    - word_count
    - speaker_id (database ID)

    Returns:
        Dictionary with 'sentences' list and 'metadata'
    """
    if word_table is None or word_table.df is None:
        logger.error("Word table or dataframe is None in generate_sentences")
        return {'error': 'No word table data available', 'sentences': []}

    sorted_words = word_table.df.sort_values('start')

    if len(sorted_words) == 0:
        logger.error("No words found in word table")
        return {'error': 'No words found', 'sentences': []}

    stitch_version = get_current_stitch_version()
    sentences = []
    global_sentence_index = 0
    turn_index = 0

    # Group words into speaker turns first
    current_speaker = None
    current_turn_words = []

    all_turns = []

    for _, word in sorted_words.iterrows():
        speaker_name = word['speaker_current']

        if speaker_name != current_speaker:
            # Save previous turn if exists
            if current_turn_words and current_speaker:
                all_turns.append({
                    'speaker_name': current_speaker,
                    'words': current_turn_words,
                    'turn_index': turn_index
                })
                turn_index += 1

            # Start new turn
            current_speaker = speaker_name
            current_turn_words = [word.to_dict()]
        else:
            current_turn_words.append(word.to_dict())

    # Add final turn
    if current_turn_words and current_speaker:
        all_turns.append({
            'speaker_name': current_speaker,
            'words': current_turn_words,
            'turn_index': turn_index
        })

    logger.info(f"[{content_id}] Processing {len(all_turns)} turns into sentences")

    # Process each turn into sentences
    for turn in all_turns:
        speaker_name = turn['speaker_name']
        words = turn['words']
        turn_idx = turn['turn_index']

        # Skip if no speaker mapping
        if speaker_name not in speaker_mappings:
            logger.debug(f"[{content_id}] Skipping turn {turn_idx} - speaker {speaker_name} has no database mapping")
            continue

        speaker_db_id = speaker_mappings[speaker_name]['speaker_db_id']

        # Detect sentence boundaries directly from word stream using punctuation
        # No NLTK text matching - timing comes directly from word indices
        SENTENCE_ENDERS = {'.', '!', '?'}

        sentence_in_turn = 0
        current_sentence_start = 0

        for word_idx, word in enumerate(words):
            word_text = word['text'].strip()

            # Check if this word ends a sentence
            is_sentence_end = (
                word_text in SENTENCE_ENDERS or
                word_text.endswith('.') or
                word_text.endswith('!') or
                word_text.endswith('?')
            )

            # Also end sentence on last word of turn
            is_last_word = (word_idx == len(words) - 1)

            if is_sentence_end or is_last_word:
                # Create sentence from word range [current_sentence_start, word_idx]
                sentence_words = words[current_sentence_start:word_idx + 1]

                if sentence_words:
                    # Build sentence text from words
                    sent_text = smart_join_words([w['text'] for w in sentence_words])

                    # Timing is DIRECTLY from word indices - no matching needed!
                    sent_start_time = sentence_words[0]['start']
                    sent_end_time = sentence_words[-1]['end']

                    sentences.append({
                        'sentence_index': global_sentence_index,
                        'turn_index': turn_idx,
                        'sentence_in_turn': sentence_in_turn,
                        'speaker_id': speaker_db_id,
                        'text': sent_text,
                        'start_time': sent_start_time,
                        'end_time': sent_end_time,
                        'word_count': len(sentence_words),
                        'stitch_version': stitch_version
                    })
                    global_sentence_index += 1
                    sentence_in_turn += 1

                # Start next sentence after this word
                current_sentence_start = word_idx + 1

    logger.info(f"[{content_id}] Generated {len(sentences)} sentences from {len(all_turns)} turns")

    return {
        'metadata': {
            'content_id': content_id,
            'total_sentences': len(sentences),
            'total_turns': len(all_turns),
            'stitch_version': stitch_version,
            'generated_at': pd.Timestamp.now().isoformat()
        },
        'sentences': sentences
    }


def _get_comprehensive_speaker_mappings(word_table: WordTable, content_id: str) -> Dict[str, Dict]:
    """
    Get comprehensive speaker mappings from all available sources.
    
    This function checks multiple sources for speaker database mappings:
    1. word_table.speaker_db_dictionary (from Stage 7)
    2. Individual word metadata (fallback)
    3. Logs diagnostics for debugging
    
    Returns:
        Dictionary mapping speaker names to database info
    """
    logger.info(f"[{content_id}] Building comprehensive speaker mappings")
    
    mappings = {}
    
    # SOURCE 1: Check word_table.speaker_db_dictionary (primary source from Stage 7)
    if hasattr(word_table, 'speaker_db_dictionary') and word_table.speaker_db_dictionary:
        logger.info(f"[{content_id}] Found speaker_db_dictionary with {len(word_table.speaker_db_dictionary)} entries")
        for speaker_name, mapping in word_table.speaker_db_dictionary.items():
            # In test mode, speaker_db_id might be negative (like -100, -101 etc)
            # Only skip if it's None or exactly -1 (the old invalid marker)
            if mapping.get('speaker_db_id') is not None and mapping['speaker_db_id'] != -1:
                mappings[speaker_name] = mapping
                logger.info(f"[{content_id}] Dictionary mapping: {speaker_name} -> {mapping.get('universal_name', 'Unknown')} (db_id: {mapping['speaker_db_id']})")
    else:
        logger.warning(f"[{content_id}] No speaker_db_dictionary found on word_table")
        logger.warning(f"[{content_id}] word_table attributes: {[attr for attr in dir(word_table) if not attr.startswith('_')]}")
    
    # SOURCE 2: Check individual word metadata (fallback source)
    metadata_mappings_found = 0
    for _, word in word_table.df.iterrows():
        speaker_name = word['speaker_current']
        
        # Skip if we already have this speaker or if it's not a real speaker
        if speaker_name in mappings or pd.isna(speaker_name) or speaker_name in ['UNKNOWN', 'MULTI_SPEAKER']:
            continue
        
        # Check word metadata for speaker_db_mapping
        metadata = word.get('metadata', {})
        if isinstance(metadata, dict) and 'speaker_db_mapping' in metadata:
            db_mapping = metadata['speaker_db_mapping']
            # In test mode, speaker_db_id might be negative (like -100, -101 etc)
            # Only skip if it's None or exactly -1 (the old invalid marker)
            if isinstance(db_mapping, dict) and db_mapping.get('speaker_db_id') is not None and db_mapping['speaker_db_id'] != -1:
                mappings[speaker_name] = {
                    'speaker_db_id': db_mapping['speaker_db_id'],
                    'global_id': db_mapping.get('global_id', 'Unknown'),
                    'universal_name': db_mapping.get('universal_name', 'Unknown'),
                    'source': 'word_metadata'
                }
                metadata_mappings_found += 1
                logger.debug(f"[{content_id}] Metadata mapping: {speaker_name} -> {db_mapping.get('universal_name', 'Unknown')} (db_id: {db_mapping['speaker_db_id']})")
    
    if metadata_mappings_found > 0:
        logger.debug(f"[{content_id}] Found word metadata")
    
    # SOURCE 3: Direct database lookup for missing speakers (fallback when speaker_db_dictionary is lost)
    if not mappings:  # Only do this expensive lookup if we have no mappings at all
        logger.warning(f"[{content_id}] No speaker mappings found, attempting direct database lookup")
        from src.database.session import get_session
        from src.database.models import Speaker
        
        # Get all unique speakers in word table
        unique_speakers = set()
        for _, word in word_table.df.iterrows():
            speaker = word['speaker_current']
            if pd.notna(speaker) and speaker != '' and speaker.startswith('SPEAKER_'):
                unique_speakers.add(speaker)
        
        logger.info(f"[{content_id}] Found {len(unique_speakers)} unique speakers in word table: {sorted(unique_speakers)}")
        
        # Look up each speaker in database
        try:
            with get_session() as session:
                for speaker_name in unique_speakers:
                    # Calculate expected hash
                    hash_str = f"{content_id}:{speaker_name}"
                    speaker_hash = hashlib.sha256(hash_str.encode()).hexdigest()[:8]
                    
                    # Look up in database
                    speaker_record = session.query(Speaker).filter_by(speaker_hash=speaker_hash).first()
                    if speaker_record:
                        mappings[speaker_name] = {
                            'speaker_db_id': speaker_record.id,
                            'global_id': speaker_hash,  # Use hash as global_id
                            'universal_name': f"Speaker {speaker_record.id}",
                            'source': 'direct_database_lookup'
                        }
                        logger.info(f"[{content_id}] Direct lookup: {speaker_name} -> Speaker {speaker_record.id} (db_id: {speaker_record.id}, hash: {speaker_hash})")
                    else:
                        logger.warning(f"[{content_id}] Direct lookup failed: {speaker_name} (hash: {speaker_hash}) not found in database")
        except Exception as e:
            logger.error(f"[{content_id}] Error during direct database lookup: {e}")
    
    # SOURCE 4: Get all unique speakers in word table for diagnostics
    all_speakers_in_table = set()
    speaker_word_counts = {}
    for _, word in word_table.df.iterrows():
        speaker = word['speaker_current']
        if pd.notna(speaker) and speaker != '':
            all_speakers_in_table.add(speaker)
            speaker_word_counts[speaker] = speaker_word_counts.get(speaker, 0) + 1
    
    # Log comprehensive diagnostics
    mapped_speakers = set(mappings.keys())
    unmapped_speakers = all_speakers_in_table - mapped_speakers
    
    logger.info(f"[{content_id}] Speaker mapping summary:")
    logger.info(f"[{content_id}]   Total unique speakers in word table: {len(all_speakers_in_table)}")
    logger.info(f"[{content_id}]   Speakers with database mappings: {len(mapped_speakers)}")
    logger.info(f"[{content_id}]   Speakers without database mappings: {len(unmapped_speakers)}")
    
    if mapped_speakers:
        logger.info(f"[{content_id}] Mapped speakers:")
        for speaker in sorted(mapped_speakers):
            mapping = mappings[speaker]
            word_count = speaker_word_counts.get(speaker, 0)
            logger.info(f"[{content_id}]   {speaker} -> {mapping.get('universal_name', 'Unknown')} (db_id: {mapping['speaker_db_id']}, words: {word_count}, source: {mapping.get('source', 'unknown')})")
    
    if unmapped_speakers:
        logger.warning(f"[{content_id}] Unmapped speakers:")
        for speaker in sorted(unmapped_speakers):
            word_count = speaker_word_counts.get(speaker, 0)
            speaker_type = "category" if speaker in ['UNKNOWN', 'MULTI_SPEAKER', 'NEEDS_EMBEDDING', 'NEEDS_LLM', 
                                                   'BAD_GRAMMAR_SINGLE', 'BAD_GRAMMAR_MULTI', 
                                                   'GOOD_GRAMMAR_SINGLE', 'GOOD_GRAMMAR_MULTI'] else "speaker"
            logger.warning(f"[{content_id}]   {speaker} ({speaker_type}, words: {word_count})")
    
    # SOURCE 5: Update speaker metrics for existing speakers
    if mappings:
        logger.info(f"[{content_id}] Updating speaker metrics for {len(mappings)} speakers")
        _update_speaker_metrics(mappings, word_table, content_id)
    
    # Log any unmapped speakers (should only be category labels now)
    real_unmapped_speakers = []
    for speaker in unmapped_speakers:
        if re.match(r'^SPEAKER_\d+$', speaker):
            real_unmapped_speakers.append(speaker)
    
    if real_unmapped_speakers:
        logger.error(f"[{content_id}] ERROR: Found {len(real_unmapped_speakers)} real speakers without database records - stage7 should have created these!")
        for speaker_name in real_unmapped_speakers:
            word_count = speaker_word_counts.get(speaker_name, 0)
            logger.error(f"[{content_id}]   {speaker_name} (words: {word_count}) - missing database record")
    
    # Debug: Check if speaker_db_dictionary exists and log its contents
    if hasattr(word_table, 'speaker_db_dictionary'):
        logger.debug(f"[{content_id}] speaker_db_dictionary contents: {word_table.speaker_db_dictionary}")
    else:
        logger.debug(f"[{content_id}] word_table has no speaker_db_dictionary attribute")
    
    return mappings


def _update_speaker_metrics(mappings: Dict[str, Dict], word_table: WordTable, content_id: str) -> None:
    """
    Update speaker metrics (duration, segment count, word count) based on actual word usage.
    
    Args:
        mappings: Dictionary mapping speaker names to database info
        word_table: WordTable with final speaker assignments
        content_id: Content ID for logging
    """
    from src.database.session import get_session
    from src.database.models import Speaker
    
    try:
        with get_session() as session:
            # Calculate metrics for each speaker
            speaker_metrics = {}
            
            for _, word in word_table.df.iterrows():
                speaker_name = word['speaker_current']
                if pd.notna(speaker_name) and speaker_name in mappings:
                    if speaker_name not in speaker_metrics:
                        speaker_metrics[speaker_name] = {
                            'word_count': 0,
                            'total_duration': 0.0,
                            'segment_count': 0,
                            'first_word_time': float('inf'),
                            'last_word_time': 0.0
                        }
                    
                    metrics = speaker_metrics[speaker_name]
                    metrics['word_count'] += 1
                    metrics['total_duration'] += (word['end'] - word['start'])
                    metrics['first_word_time'] = min(metrics['first_word_time'], word['start'])
                    metrics['last_word_time'] = max(metrics['last_word_time'], word['end'])
            
            # Count segments (continuous speech blocks) for each speaker
            for speaker_name in speaker_metrics:
                segments = 0
                current_segment_speaker = None
                
                for _, word in word_table.df.sort_values('start').iterrows():
                    if word['speaker_current'] == speaker_name:
                        if current_segment_speaker != speaker_name:
                            segments += 1
                            current_segment_speaker = speaker_name
                    else:
                        if current_segment_speaker == speaker_name:
                            current_segment_speaker = None
                
                speaker_metrics[speaker_name]['segment_count'] = segments
            
            # Update database records
            updated_count = 0
            for speaker_name, metrics in speaker_metrics.items():
                if speaker_name in mappings:
                    speaker_db_id = mappings[speaker_name]['speaker_db_id']
                    
                    # Get speaker record
                    speaker = session.query(Speaker).filter_by(id=speaker_db_id).first()
                    if speaker:
                        # Update metrics
                        speaker.duration = metrics['total_duration']
                        speaker.segment_count = metrics['segment_count']
                        
                        # Update metadata with word-level statistics
                        if not speaker.meta_data:
                            speaker.meta_data = {}
                        
                        speaker.meta_data.update({
                            'final_word_count': metrics['word_count'],
                            'final_duration': metrics['total_duration'],
                            'final_segment_count': metrics['segment_count'],
                            'first_word_time': metrics['first_word_time'],
                            'last_word_time': metrics['last_word_time'],
                            'updated_in_stage12': True
                        })
                        
                        updated_count += 1
                        logger.info(f"[{content_id}] Updated {speaker_name} (db_id: {speaker_db_id}): "
                                   f"words={metrics['word_count']}, duration={metrics['total_duration']:.1f}s, "
                                   f"segments={metrics['segment_count']}")
                    else:
                        logger.error(f"[{content_id}] Speaker {speaker_name} not found in database (db_id: {speaker_db_id})")
            
            session.commit()
            logger.info(f"[{content_id}] Successfully updated metrics for {updated_count} speakers")
            
    except Exception as e:
        logger.error(f"[{content_id}] Error updating speaker metrics: {e}")
        logger.error(f"[{content_id}] Speaker metrics update error details:", exc_info=True)

def save_sentences_to_database(sentences: List[Dict[str, Any]], content_id: str, test_mode: bool = False) -> int:
    """
    Save sentences to the database.

    Sentences are the atomic unit for emotion detection and fine-grained queries.
    They are saved to the `sentences` table.

    In test mode, only logs what would be saved without actually writing to the database.

    Returns the number of sentences saved.
    """
    if not sentences:
        logger.warning(f"[{content_id}] No sentences to save")
        return 0

    if test_mode:
        logger.info(f"[{content_id}] TEST MODE: Would save {len(sentences)} sentences to database")
        for i, sent in enumerate(sentences[:5]):  # Log first 5 sentences as examples
            logger.info(f"[{content_id}] TEST MODE: Sentence {i}: speaker_id={sent['speaker_id']}, "
                       f"turn={sent['turn_index']}, time={sent['start_time']:.1f}-{sent['end_time']:.1f}s, "
                       f"text='{sent['text'][:50]}...'")
        if len(sentences) > 5:
            logger.info(f"[{content_id}] TEST MODE: ... and {len(sentences) - 5} more sentences")
        return 0

    saved_count = 0

    try:
        with get_session() as session:
            # Get the content record
            content = session.query(Content).filter_by(content_id=content_id).first()
            if not content:
                logger.error(f"[{content_id}] Content not found in database for content_id: {content_id}")
                return 0

            # Delete existing sentences for this content
            existing_count = session.query(Sentence).filter_by(
                content_id=content.id
            ).count()

            if existing_count > 0:
                logger.info(f"[{content_id}] Deleting {existing_count} existing sentences")
                deleted = session.query(Sentence).filter_by(
                    content_id=content.id
                ).delete()
                logger.info(f"[{content_id}] Actually deleted {deleted} sentences")

            # Create new sentence records
            for sent in sentences:
                sentence_record = Sentence(
                    content_id=content.id,  # Use the primary key, not the platform content_id
                    speaker_id=sent['speaker_id'],
                    sentence_index=sent['sentence_index'],
                    turn_index=sent['turn_index'],
                    sentence_in_turn=sent['sentence_in_turn'],
                    text=sent['text'],
                    start_time=sent['start_time'],
                    end_time=sent['end_time'],
                    word_count=sent['word_count'],
                    stitch_version=sent['stitch_version'],
                    # Emotion fields are NULL initially - populated by stage13 or batch processing
                    emotion=None,
                    emotion_confidence=None,
                    emotion_scores=None,
                    arousal=None,
                    valence=None,
                    dominance=None
                )
                session.add(sentence_record)
                saved_count += 1

            # Update content flags - sentences are now the primary stitched output
            content.is_stitched = True
            content.stitch_version = get_current_stitch_version()
            content.last_updated = datetime.now(timezone.utc)

            session.commit()
            logger.info(f"[{content_id}] Successfully saved {saved_count} sentences to database")

    except Exception as e:
        logger.error(f"[{content_id}] Failed to save sentences to database: {e}")
        logger.error(f"[{content_id}] Error type: {type(e).__name__}")
        logger.error(f"[{content_id}] Error details:", exc_info=True)
        # Log more details about the failure
        logger.error(f"[{content_id}] Failed while processing {len(sentences)} sentences")
        if len(sentences) > 0:
            logger.error(f"[{content_id}] First sentence sample: speaker_id={sentences[0].get('speaker_id')}, "
                        f"text='{sentences[0].get('text', '')[:30]}...'")
        # Try to rollback if session is still active
        try:
            session.rollback()
            logger.error(f"[{content_id}] Database transaction rolled back")
        except:
            pass
        saved_count = 0

    return saved_count


def stage12_output(content_id: str,
                  word_table: WordTable,
                  output_dir: Optional[Path] = None,
                  test_mode: bool = False) -> Dict[str, Any]:
    """
    Execute Stage 11: Generate outputs and save to database.
    
    Args:
        content_id: Content ID to process
        word_table: WordTable with final speaker assignments
        output_dir: Optional directory to save output files
        test_mode: If True, only simulate database operations
        
    Returns:
        Dictionary with generation results
    """
    start_time = time.time()
    stage_name = 'stage9_output'
    
    logger.info(f"[{content_id}] Starting Stage 11: Output Generation")
    
    result = {
        'status': 'pending',
        'content_id': content_id,
        'stage': stage_name,
        'data': {
            'readable_transcript': None,
            'detailed_transcript': None,
            'speaker_turns': None,
            'output_files': [],
            'database_saved': False
        },
        'stats': {},
        'error': None
    }
    
    try:
        # Generate readable transcript
        readable_transcript = generate_readable_transcript(word_table)
        result['data']['readable_transcript'] = readable_transcript
        
        # Generate detailed transcript
        detailed_transcript = generate_detailed_transcript(word_table)
        result['data']['detailed_transcript'] = detailed_transcript
        
        # Generate speaker turns for database
        speaker_turns_dict = generate_speaker_turns(word_table, content_id)
        if 'error' in speaker_turns_dict:
            raise ValueError(f"Speaker turn generation failed: {speaker_turns_dict['error']}")
        
        result['data']['speaker_turns'] = speaker_turns_dict['speaker_turns']
        result['data']['skipped_turns'] = speaker_turns_dict.get('skipped_turns', [])
        result['data']['speaker_mappings'] = speaker_turns_dict.get('speaker_mappings_used', {})
        
        # Log generation results
        metadata = speaker_turns_dict.get('metadata', {})
        logger.info(f"[{content_id}] Speaker turn generation results:")
        logger.info(f"[{content_id}]   Raw turns generated: {metadata.get('raw_turns_generated', 0)}")
        logger.info(f"[{content_id}]   Final turns with database IDs: {metadata.get('final_turns_with_db_ids', 0)}")
        logger.info(f"[{content_id}]   Skipped turns (no database mapping): {metadata.get('skipped_turns', 0)}")
        
        if result['data']['skipped_turns']:
            logger.warning(f"[{content_id}] Some turns were skipped due to missing database mappings - check earlier stages")

        # Generate sentences from word table (atomic unit for emotion detection)
        speaker_mappings = speaker_turns_dict.get('speaker_mappings_used', {})
        sentences_dict = generate_sentences(word_table, content_id, speaker_mappings)
        if 'error' in sentences_dict and sentences_dict.get('sentences', []) == []:
            logger.warning(f"[{content_id}] Sentence generation had issues: {sentences_dict.get('error')}")
        result['data']['sentences'] = sentences_dict.get('sentences', [])

        # Log sentence generation results
        sentences_metadata = sentences_dict.get('metadata', {})
        logger.info(f"[{content_id}] Sentence generation results:")
        logger.info(f"[{content_id}]   Total sentences generated: {sentences_metadata.get('total_sentences', 0)}")
        logger.info(f"[{content_id}]   From turns: {sentences_metadata.get('total_turns', 0)}")

        # Save to files if output directory provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save readable transcript
            readable_path = output_dir / f"{content_id}_transcript.txt"
            with open(readable_path, 'w') as f:
                f.write(readable_transcript)
            result['data']['output_files'].append(str(readable_path))
            
            # Save detailed transcript
            detailed_path = output_dir / f"{content_id}_transcript_detailed.txt"
            with open(detailed_path, 'w') as f:
                f.write(detailed_transcript)
            result['data']['output_files'].append(str(detailed_path))
            
            # Save assignment history report for debugging (no word limit = show all words)
            assignment_history_report = word_table.generate_assignment_history_report(max_words=None)
            history_path = output_dir / f"{content_id}_assignment_history.txt"
            with open(history_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(assignment_history_report))
            result['data']['output_files'].append(str(history_path))
            logger.info(f"[{content_id}] Saved complete assignment history report to {history_path} ({len(assignment_history_report)} lines)")
            
            # Create Sankey diagram and assignment visualizations (only in test mode)
            if test_mode:
                try:
                    from .sankey_visualizer import create_all_assignment_visualizations
                    viz_files = create_all_assignment_visualizations(content_id, word_table, output_dir, test_mode)
                    result['data']['output_files'].extend([str(f) for f in viz_files])
                    result['stats']['visualization_files_created'] = len(viz_files)
                    logger.info(f"[{content_id}] Created {len(viz_files)} assignment visualization files")
                except ImportError as e:
                    logger.warning(f"[{content_id}] Sankey visualization not available: {e}")
                    result['stats']['visualization_files_created'] = 0
                except Exception as e:
                    logger.error(f"[{content_id}] Error creating assignment visualizations: {e}")
                    result['stats']['visualization_files_created'] = 0
            else:
                result['stats']['visualization_files_created'] = 0
            
            # Save speaker turns as JSON
            speaker_turns_path = output_dir / f"{content_id}_speaker_turns.json"
            with open(speaker_turns_path, 'w') as f:
                # Get the speaker turns list
                speaker_turns_list = result['data']['speaker_turns']
                
                # Create the output structure
                output_data = {
                    'metadata': {
                        'content_id': content_id,
                        'generated_at': pd.Timestamp.now().isoformat(),
                        'stitch_version': get_current_stitch_version()
                    },
                    'turns': []
                }
                
                # Convert turns to JSON-serializable format
                for turn in speaker_turns_list:
                    json_turn = {}
                    # Copy and convert values to ensure JSON compatibility
                    for key, value in turn.items():
                        if isinstance(value, (int, float)):
                            json_turn[key] = float(value) if isinstance(value, float) else int(value)
                        else:
                            json_turn[key] = value
                    # Add readable timestamps
                    json_turn['start_timestamp'] = format_timestamp(json_turn['start_time'])
                    json_turn['end_timestamp'] = format_timestamp(json_turn['end_time'])
                    output_data['turns'].append(json_turn)
                
                json.dump(output_data, f, indent=2)
            result['data']['output_files'].append(str(speaker_turns_path))
            
            logger.info(f"[{content_id}] Saved transcript outputs to {output_dir}")
        
        # NOTE: speaker_transcriptions table is deprecated - we now only save to sentences table
        # Speaker turns are kept in result['data']['speaker_turns'] for file output/debugging
        # but are NOT saved to database. Sentences are the atomic unit going forward.
        result['stats']['speaker_turns_saved'] = 0  # Deprecated - no longer saving

        if not result['data']['speaker_turns']:
            # No speaker turns means stitch failed to produce valid output
            if not test_mode:
                result['status'] = 'error'
                result['error'] = 'No speaker turns generated - stitch failed to produce valid output'
                logger.error(f"[{content_id}] Stage 12 failed: No speaker turns generated")
            else:
                logger.warning(f"[{content_id}] No speaker turns generated (test mode)")

        # Save sentences to database (atomic unit - this is the primary output now)
        if result['data'].get('sentences'):
            sentences_saved = save_sentences_to_database(result['data']['sentences'], content_id, test_mode)
            result['data']['sentences_saved'] = sentences_saved > 0
            result['stats']['sentences_saved'] = sentences_saved

            # If not in test mode and database save failed, this is a critical error
            if not test_mode and sentences_saved == 0:
                result['status'] = 'error'
                result['error'] = 'Failed to save sentences to database'
                logger.error(f"[{content_id}] Stage 12 failed: Database save returned 0 saved sentences")
            else:
                logger.info(f"[{content_id}] Saved {sentences_saved} sentences to database")
        else:
            result['stats']['sentences_saved'] = 0
            if not test_mode:
                result['status'] = 'error'
                result['error'] = 'No sentences generated - stitch failed to produce valid output'
                logger.error(f"[{content_id}] Stage 12 failed: No sentences generated")
            else:
                logger.warning(f"[{content_id}] No sentences to save to database (test mode)")
        
        # Calculate statistics
        word_stats = word_table.get_speaker_statistics()
        
        result['stats'].update({
            'duration': time.time() - start_time,
            'total_words': word_stats['total_words'],
            'assigned_words': word_stats['total_words'] - word_stats['unknown_words'],
            'unknown_words': word_stats['unknown_words'],
            'speaker_distribution': word_stats['speaker_distribution'],
            'speaker_turns_generated': len(result['data']['speaker_turns']),
            'speaker_turns_skipped': len(result['data']['skipped_turns']),
            'sentences_generated': len(result['data'].get('sentences', [])),
            'turn_generation_metadata': speaker_turns_dict.get('metadata', {}),
            'sentence_generation_metadata': sentences_dict.get('metadata', {})
        })
        
        # Only set success if not already set to error
        if result['status'] != 'error':
            result['status'] = 'success'
        logger.info(f"[{content_id}] Stage 11 completed successfully")
        
        return result
        
    except Exception as e:
        logger.error(f"[{content_id}] Stage 11 failed: {str(e)}")
        logger.error(f"[{content_id}] Error details:", exc_info=True)
        
        result.update({
            'status': 'error',
            'error': str(e),
            'duration': time.time() - start_time
        })
        return result 