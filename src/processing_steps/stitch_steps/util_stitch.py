#!/usr/bin/env python3
"""
Stitch Stage Utilities
=====================

Shared utilities and helper functions for stitch stages.
Provides consistent handling of:
1. Word table updates
2. Stage configuration
3. Metadata tracking
4. Status updates
"""

import time
import logging
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
from pathlib import Path
from src.utils.paths import get_project_root, get_config_path
import yaml
import torch
import numpy as np

# PyTorch 2.6+ changed default weights_only=True which breaks loading older models
# that contain pickle-serialized objects like pytorch_lightning callbacks.
# Monkey-patch torch.load BEFORE any pyannote imports to ensure the patched version
# is used throughout. This is safe for pyannote models which are from a trusted source.
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from .stage3_tables import WordTable

from src.utils.logger import setup_worker_logger
from src.utils.config import load_config
logger = setup_worker_logger('stitch')


def summarize_speaker_assignments(word_table: WordTable, stage_name: str, content_id: str, test_mode: bool = False) -> Dict[str, int]:
    """
    Summarize the current speaker assignment status in the word table.
    
    Args:
        word_table: WordTable instance
        stage_name: Name of the current stage
        content_id: Content ID for logging
        test_mode: Whether to log the summary
        
    Returns:
        Dictionary with counts for each speaker category
    """
    if word_table is None or word_table.df is None or len(word_table.df) == 0:
        return {}
    
    # Count words by speaker assignment
    speaker_counts = word_table.df['speaker_current'].value_counts().to_dict()
    
    # Separate into categories
    summary = {
        'NEEDS_EMBEDDING': speaker_counts.get('NEEDS_EMBEDDING', 0),
        'NEEDS_LLM': speaker_counts.get('NEEDS_LLM', 0),
        'UNKNOWN': speaker_counts.get('UNKNOWN', 0),
        'MULTI_SPEAKER': speaker_counts.get('MULTI_SPEAKER', 0),
        'BAD_GRAMMAR_SINGLE': speaker_counts.get('BAD_GRAMMAR_SINGLE', 0),
        'BAD_GRAMMAR_MULTI': speaker_counts.get('BAD_GRAMMAR_MULTI', 0),
        'GOOD_GRAMMAR_SINGLE': speaker_counts.get('GOOD_GRAMMAR_SINGLE', 0),
        'GOOD_GRAMMAR_MULTI': speaker_counts.get('GOOD_GRAMMAR_MULTI', 0),
        'resolved_speakers': 0,
        'total_words': len(word_table.df)
    }
    
    # Count actual speaker assignments (SPEAKER_XX)
    for speaker, count in speaker_counts.items():
        if speaker.startswith('SPEAKER_') and len(speaker) > 8:  # e.g., SPEAKER_01
            summary['resolved_speakers'] += count
    
    # Calculate unresolved total
    unresolved_categories = ['NEEDS_EMBEDDING', 'NEEDS_LLM', 'UNKNOWN', 'MULTI_SPEAKER',
                           'BAD_GRAMMAR_SINGLE', 'BAD_GRAMMAR_MULTI', 
                           'GOOD_GRAMMAR_SINGLE', 'GOOD_GRAMMAR_MULTI']
    summary['total_unresolved'] = sum(summary[cat] for cat in unresolved_categories)
    
    # Log summary in test mode
    if test_mode:
        logger.info(f"[{content_id}] ========== STAGE {stage_name.upper()} SUMMARY ==========")
        logger.info(f"[{content_id}] Total words: {summary['total_words']}")
        logger.info(f"[{content_id}] Resolved to speakers: {summary['resolved_speakers']} ({summary['resolved_speakers']/summary['total_words']*100:.1f}%)")
        logger.info(f"[{content_id}] Total unresolved: {summary['total_unresolved']} ({summary['total_unresolved']/summary['total_words']*100:.1f}%)")
        
        if summary['total_unresolved'] > 0:
            logger.info(f"[{content_id}] Unresolved breakdown:")
            
            # Resolution needed categories
            resolution_needed = ['NEEDS_EMBEDDING', 'NEEDS_LLM', 'UNKNOWN']
            resolution_total = sum(summary[cat] for cat in resolution_needed)
            if resolution_total > 0:
                logger.info(f"[{content_id}]   Needing resolution: {resolution_total}")
                for cat in resolution_needed:
                    if summary[cat] > 0:
                        logger.info(f"[{content_id}]     - {cat}: {summary[cat]}")
            
            # Grammar categories (temporary assignments)
            grammar_categories = ['BAD_GRAMMAR_SINGLE', 'BAD_GRAMMAR_MULTI', 
                                'GOOD_GRAMMAR_SINGLE', 'GOOD_GRAMMAR_MULTI']
            grammar_total = sum(summary[cat] for cat in grammar_categories)
            if grammar_total > 0:
                logger.info(f"[{content_id}]   Grammar categories: {grammar_total}")
                for cat in grammar_categories:
                    if summary[cat] > 0:
                        logger.info(f"[{content_id}]     - {cat}: {summary[cat]}")
            
            # Multi-speaker
            if summary['MULTI_SPEAKER'] > 0:
                logger.info(f"[{content_id}]   MULTI_SPEAKER: {summary['MULTI_SPEAKER']}")
        
        logger.info(f"[{content_id}] " + "=" * 50)
    
    return summary


def load_stage_config(stage_name: str) -> Dict:
    """
    Load configuration for a specific stitch stage.
    
    Args:
        stage_name: Name of the stage (e.g., 'word_diarization')
        
    Returns:
        Dict with stage configuration
    """
    try:
        config_path = get_project_root() / 'config' / "stitch_config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        stage_config = config.get('stitch_stages', {}).get(stage_name, {})
        if not stage_config:
            logger.warning(f"No specific configuration found for stage: {stage_name}")
            stage_config = {}
        
        return stage_config
        
    except Exception as e:
        logger.error(f"Error loading stage config: {e}")
        return {}

def update_word_assignment(word_table: WordTable,
                         idx: int,
                         stage_name: str,
                         speaker: Optional[str] = None,
                         confidence: float = 0.0,
                         method: str = 'unknown',
                         metadata: Optional[Dict] = None) -> None:
    """
    Update word table with speaker assignment information.
    
    Args:
        word_table: WordTable instance
        idx: Index of word to update
        stage_name: Name of the stage making the assignment
        speaker: Assigned speaker (if any)
        confidence: Confidence in the assignment
        method: Method used for assignment
        metadata: Additional metadata to store
    """
    # Use the actual column name from WordTable
    speaker_column = 'speaker_current'
    
    # Store previous assignment if exists
    if pd.notna(word_table.df.at[idx, speaker_column]):
        word_table.df.at[idx, 'previous_assignment'] = word_table.df.at[idx, speaker_column]
    
    # Update assignment
    if speaker is not None:
        word_table.df.at[idx, speaker_column] = speaker
    
    word_table.df.at[idx, 'assignment_confidence'] = confidence
    word_table.df.at[idx, 'assignment_method'] = method
    word_table.df.at[idx, 'assignment_stage'] = stage_name
    word_table.df.at[idx, 'assignment_timestamp'] = time.time()
    
    # Update metadata
    current_metadata = word_table.df.at[idx, 'metadata']
    if not isinstance(current_metadata, dict):
        current_metadata = {}
    
    if metadata:
        current_metadata.update(metadata)
    
    if 'stage_progression' not in current_metadata:
        current_metadata['stage_progression'] = []
    
    current_metadata['stage_progression'].append({
        'stage': stage_name,
        'timestamp': time.time(),
        'method': method,
        'confidence': confidence
    })
    
    word_table.df.at[idx, 'metadata'] = current_metadata

def update_word_embedding(word_table: WordTable,
                        idx: int,
                        embedding: Optional[Any] = None,
                        confidence: float = 0.0,
                        method: str = 'unknown',
                        metadata: Optional[Dict] = None) -> None:
    """
    Update word table with embedding information.
    
    Args:
        word_table: WordTable instance
        idx: Index of word to update
        embedding: The embedding vector (if successful)
        confidence: Confidence in the embedding
        method: Method used for embedding extraction
        metadata: Additional metadata to store
    """
    if embedding is not None:
        # Convert numpy array to list for storage in DataFrame
        if hasattr(embedding, 'tolist'):
            embedding = embedding.tolist()
        word_table.df.at[idx, 'speaker_embedding'] = embedding
    
    word_table.df.at[idx, 'embedding_confidence'] = confidence
    word_table.df.at[idx, 'embedding_method'] = method
    word_table.df.at[idx, 'embedding_stage'] = 'word_embeddings'
    word_table.df.at[idx, 'embedding_timestamp'] = time.time()
    
    # Update metadata
    current_metadata = word_table.df.at[idx, 'metadata']
    if not isinstance(current_metadata, dict):
        current_metadata = {}
    
    if metadata:
        current_metadata.update(metadata)
    
    if 'stage_progression' not in current_metadata:
        current_metadata['stage_progression'] = []
    
    current_metadata['stage_progression'].append({
        'stage': 'word_embeddings',
        'timestamp': time.time(),
        'method': method,
        'confidence': confidence
    })
    
    word_table.df.at[idx, 'metadata'] = current_metadata

def update_processing_status(word_table: WordTable, stage_name: str) -> int:
    """
    Update processing status for all words in table.
    
    Args:
        word_table: WordTable instance
        stage_name: Name of the stage
        
    Returns:
        Number of words updated
    """
    updated_count = len(word_table.df)
    word_table.df['processing_status'] = f'processed_stage_{stage_name}'
    
    # Update metadata with stage progression
    for idx in word_table.df.index:
        current_metadata = word_table.df.at[idx, 'metadata']
        if not isinstance(current_metadata, dict):
            current_metadata = {}
        
        if 'stage_progression' not in current_metadata:
            current_metadata['stage_progression'] = []
            
        current_metadata['stage_progression'].append({
            'stage': stage_name,
            'timestamp': time.time()
        })
        
        word_table.df.at[idx, 'metadata'] = current_metadata
    
    return updated_count

def format_stage_stats(content_id: str,
                      stage_name: str,
                      stats: Dict,
                      start_time: float) -> None:
    """
    Format and log stage statistics consistently.
    
    Args:
        content_id: Content ID being processed
        stage_name: Name of the stage
        stats: Statistics dictionary
        start_time: Stage start time
    """
    duration = time.time() - start_time
    
    logger.info(f"[{content_id}] Stage {stage_name} completed in {duration:.2f}s")
    
    for key, value in stats.items():
        if isinstance(value, (int, float)):
            logger.info(f"[{content_id}] - {key}: {value:,}")
        else:
            logger.info(f"[{content_id}] - {key}: {value}")

def propagate_speaker_by_fragment(word_table: WordTable,
                                stage_name: str,
                                context_words: int = 5,
                                confidence_decay: float = 0.9) -> int:
    """
    Propagate speaker assignments within sentence fragments around speaker changes,
    but only to UNKNOWN words and without cascading previously assigned methods.
    Uses the highest confidence assignment in a fragment as the source.
    
    Args:
        word_table: WordTable instance
        stage_name: Name of the stage doing propagation
        context_words: Number of words on each side of speaker changes to consider
        confidence_decay: Factor to decay confidence for propagated assignments
        
    Returns:
        Number of newly assigned words
    """
    from .stage10_split_sentences import get_sentence_fragments_around_speaker_changes
    
    newly_assigned = 0
    
    # Get fragments around speaker changes (ignoring artificial sentence_id)
    fragments = get_sentence_fragments_around_speaker_changes(word_table, context_words=context_words)
    
    if not fragments:
        logger.info("No speaker change fragments found for propagation")
        return 0
    
    logger.info(f"Processing {len(fragments)} speaker change fragments for propagation")
    
    # Process each fragment
    for fragment in fragments:
        word_indices = fragment['word_indices']
        fragment_df = word_table.df.loc[word_indices]
        
        # Skip if no words in fragment
        if len(fragment_df) == 0:
            continue
        
        # Use the actual column name from WordTable
        speaker_column = 'speaker_current'
        
        # Find assigned speakers in this fragment
        assigned_speakers = fragment_df[
            pd.notna(fragment_df[speaker_column]) & 
            (fragment_df[speaker_column] != 'UNKNOWN')
        ][speaker_column].unique()
        
        # For fragments, we expect multiple speakers (that's why they're fragments)
        # We'll propagate from the dominant speaker to nearby UNKNOWN words
        if len(assigned_speakers) == 0:
            continue
        
        # Find the dominant speaker (most words) in the fragment
        speaker_counts = {}
        for speaker in assigned_speakers:
            speaker_counts[speaker] = (fragment_df[speaker_column] == speaker).sum()
        
        dominant_speaker = max(speaker_counts.keys(), key=lambda x: speaker_counts[x])
        
        # Only propagate if the dominant speaker has significantly more words
        total_assigned = sum(speaker_counts.values())
        dominance_ratio = speaker_counts[dominant_speaker] / total_assigned
        
        if dominance_ratio < 0.6:  # Require at least 60% dominance
            logger.debug(f"Skipping fragment {fragment['fragment_id']}: no clear dominant speaker "
                        f"(best: {dominant_speaker} with {dominance_ratio:.1%})")
            continue
        
        # Find the assigned words with highest confidence for the dominant speaker
        confidence_col = 'assignment_confidence' if 'assignment_confidence' in fragment_df.columns else 'confidence'
        dominant_speaker_words = fragment_df[fragment_df[speaker_column] == dominant_speaker]
        
        if len(dominant_speaker_words) == 0:
            continue
            
        max_confidence = dominant_speaker_words[confidence_col].max() if confidence_col in fragment_df.columns else 1.0
        
        # Find unassigned words in fragment (either null or explicitly UNKNOWN)
        unassigned_mask = (
            pd.isna(fragment_df[speaker_column]) | 
            (fragment_df[speaker_column] == 'UNKNOWN')
        )
        unassigned_indices = fragment_df[unassigned_mask].index
        
        # Only propagate to words that don't have previous assignments from other methods
        # (to prevent cascading as mentioned by user)
        for idx in unassigned_indices:
            # Check if this word has been assigned by previous stages
            current_method = word_table.df.at[idx, 'resolution_method']
            if current_method not in ['none', 'initialized', 'unknown']:
                continue  # Skip words already processed by other methods
            
            # Calculate distance from nearest assigned word of dominant speaker
            word_start = word_table.df.at[idx, 'start']
            distances = []
            for assigned_idx in dominant_speaker_words.index:
                assigned_start = word_table.df.at[assigned_idx, 'start']
                distances.append(abs(word_start - assigned_start))
            
            if distances:
                min_distance = min(distances)
                # Apply distance-based confidence decay (closer words get higher confidence)
                distance_decay = max(0.5, 1.0 - (min_distance / 5.0))  # Decay over 5 seconds
                final_confidence = max_confidence * confidence_decay * distance_decay
                
                update_word_assignment(
                    word_table,
                    idx,
                    stage_name,
                    speaker=dominant_speaker,
                    confidence=final_confidence,
                    method='fragment_propagation',
                    metadata={
                        'propagation': {
                            'fragment_id': fragment['fragment_id'],
                            'original_confidence': max_confidence,
                            'decay_factor': confidence_decay,
                            'distance_decay': distance_decay,
                            'min_distance': min_distance,
                            'source_speaker': dominant_speaker,
                            'dominance_ratio': dominance_ratio
                        }
                    }
                )
                newly_assigned += 1
        
        if len(unassigned_indices) > 0:
            logger.debug(f"Fragment {fragment['fragment_id']}: Propagated speaker {dominant_speaker} "
                        f"to {len(unassigned_indices)} words (dominance: {dominance_ratio:.1%}, "
                        f"confidence: {max_confidence:.3f})")
    
    if newly_assigned > 0:
        logger.info(f"Propagated speakers to {newly_assigned} words across {len(fragments)} fragments")
    
    return newly_assigned


def propagate_speaker_by_sentence(word_table: WordTable,
                                stage_name: str,
                                confidence_decay: float = 0.9) -> int:
    """
    DEPRECATED: Use propagate_speaker_by_fragment instead.
    
    Propagate speaker assignments within sentences, but only to UNKNOWN words.
    Uses the highest confidence assignment in a sentence as the source.
    
    This function is deprecated because it relies on artificial sentence_id 
    boundaries. Use propagate_speaker_by_fragment for better results.
    """
    logger.warning(f"Using deprecated propagate_speaker_by_sentence. "
                  f"Consider using propagate_speaker_by_fragment instead.")
    
    return propagate_speaker_by_fragment(word_table, stage_name, confidence_decay=confidence_decay) 

def combine_sequential_turns(word_table: WordTable,
                           stage_name: str,
                           confidence_boost: float = 1.1) -> int:
    """
    Handle 1-word interruptions by reassigning isolated single words to match surrounding context.
    Only processes true interruptions where a single word breaks an otherwise continuous speaker segment.
    
    Args:
        word_table: WordTable instance
        stage_name: Name of the stage doing the combination
        confidence_boost: Factor to boost confidence for absorbed words (default 1.1)
        
    Returns:
        Number of words reassigned
    """
    words_reassigned = 0
    speaker_column = 'speaker_current'
    
    # Create a copy of the speaker assignments to track changes
    original_speakers = word_table.df[speaker_column].copy()
    current_speakers = original_speakers.copy()
    
    # First pass: Handle 1-word interruptions (vectorized)
    # Shift speakers to get previous and next
    prev_speakers = current_speakers.shift(1)
    next_speakers = current_speakers.shift(-1)
    
    # Find 1-word interruptions where prev and next match but current differs
    interruption_mask = (
        (prev_speakers == next_speakers) &  # Adjacent speakers match
        (current_speakers != prev_speakers) &  # Current speaker is different
        (~pd.isna(prev_speakers)) &  # No NaN speakers
        (~pd.isna(current_speakers)) &
        (~pd.isna(next_speakers))
    )
    
    # Get indices of interruptions
    interruption_indices = interruption_mask[interruption_mask].index
    
    # Process interruptions
    for idx in interruption_indices:
        surrounding_speaker = prev_speakers.iloc[idx]  # Could use next_speakers too since they match
        interrupted_speaker = current_speakers.iloc[idx]
        
        # Update speaker assignment
        current_speakers.iloc[idx] = surrounding_speaker
        
        # Update assignment with boosted confidence
        current_confidence = word_table.df.at[idx, 'assignment_confidence']
        if pd.isna(current_confidence):
            current_confidence = 0.5  # Default if no confidence exists
            
        update_word_assignment(
            word_table,
            idx,
            stage_name,
            speaker=surrounding_speaker,
            confidence=min(current_confidence * confidence_boost, 1.0),
            method='interruption_absorbed',
            metadata={
                'turn_combination': {
                    'original_speaker': interrupted_speaker,
                    'absorbed_into': surrounding_speaker,
                    'reason': 'one_word_interruption'
                }
            }
        )
        words_reassigned += 1
    
    if words_reassigned > 0:
        logger.info(f"Handled {words_reassigned} single-word interruptions")
        
    return words_reassigned


# ====================================================================================================
# SHARED AUDIO LOADING UTILITIES
# ====================================================================================================

def load_shared_audio(audio_path: Optional[Path], content_id: str) -> Optional[Tuple[torch.Tensor, int]]:
    """
    Load audio file once for use across multiple stages (5, 8, 10, 11).
    
    Args:
        audio_path: Path to audio file
        content_id: Content ID for logging
        
    Returns:
        Tuple of (waveform, sample_rate) or None if loading fails
    """
    if not audio_path or not audio_path.exists():
        logger.info(f"[{content_id}] No audio file available for shared loading")
        return None
    
    try:
        # Initialize audio loader
        from pyannote.audio import Audio
        audio_loader = Audio(sample_rate=16000, mono="downmix")
        
        logger.debug(f"[{content_id}] Loading audio file for shared use: {audio_path}")
        start_time = time.time()
        
        # Load full audio file
        audio_result = audio_loader(audio_path)
        if isinstance(audio_result, tuple):
            full_waveform, sample_rate = audio_result
        else:
            full_waveform = audio_result
            sample_rate = 16000  # PyAnnote default
        
        load_time = time.time() - start_time
        logger.info(f"[{content_id}] Shared audio loaded: shape={full_waveform.shape}, "
                   f"sample_rate={sample_rate}, time={load_time:.2f}s")
        
        return full_waveform, sample_rate
        
    except Exception as e:
        logger.error(f"[{content_id}] Failed to load shared audio: {e}")
        return None


def initialize_shared_embedding_model(content_id: str, device: str = "auto") -> Optional[torch.nn.Module]:
    """
    Initialize pyannote embedding model for shared use across stages.
    
    Args:
        content_id: Content ID for logging
        device: Device to use ("auto", "cpu", "mps", "cuda")
        
    Returns:
        Initialized embedding model or None if initialization fails
    """
    try:
        # Determine device
        if device == "auto":
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        
        logger.info(f"[{content_id}] Initializing shared embedding model on {device}")
        start_time = time.time()
        
        # Load the pyannote/embedding model
        from pyannote.audio import Model
        embedding_model = Model.from_pretrained("pyannote/embedding").to(device)
        
        init_time = time.time() - start_time
        logger.info(f"[{content_id}] Shared embedding model initialized on {device} in {init_time:.2f}s")
        
        return embedding_model
        
    except Exception as e:
        logger.error(f"[{content_id}] Failed to initialize shared embedding model: {e}")
        return None


def extract_audio_embeddings_batch(segments: list, 
                                  audio_data: Tuple[torch.Tensor, int],
                                  embedding_model: torch.nn.Module,
                                  content_id: str,
                                  max_batch_size: int = 16) -> Dict[str, np.ndarray]:
    """
    Extract embeddings for multiple audio segments in batches.
    
    Args:
        segments: List of segment dicts with 'start_time', 'end_time', 'segment_id'
        audio_data: Tuple of (waveform, sample_rate) from load_shared_audio
        embedding_model: Initialized pyannote embedding model
        content_id: Content ID for logging
        max_batch_size: Maximum batch size for processing
        
    Returns:
        Dict mapping segment_id to embedding array
    """
    if not segments or not audio_data or not embedding_model:
        return {}
    
    full_waveform, sample_rate = audio_data
    embeddings = {}
    
    logger.info(f"[{content_id}] Extracting embeddings for {len(segments)} segments")
    
    # Process in batches
    for batch_start in range(0, len(segments), max_batch_size):
        batch_end = min(batch_start + max_batch_size, len(segments))
        batch_segments = segments[batch_start:batch_end]
        
        try:
            # Extract waveforms for this batch
            batch_waveforms = []
            batch_ids = []
            
            for segment in batch_segments:
                start_time = segment['start_time']
                end_time = segment['end_time']
                segment_id = segment['segment_id']
                
                # Convert to sample indices
                start_sample = int(start_time * sample_rate)
                end_sample = int(end_time * sample_rate)
                
                # Extract segment waveform
                segment_waveform = full_waveform[..., start_sample:end_sample]
                
                # Ensure correct format
                if segment_waveform.dim() == 1:
                    segment_waveform = segment_waveform.unsqueeze(0)
                elif segment_waveform.dim() == 2 and segment_waveform.size(0) > 1:
                    segment_waveform = segment_waveform.mean(dim=0, keepdim=True)
                
                batch_waveforms.append(segment_waveform)
                batch_ids.append(segment_id)
            
            if not batch_waveforms:
                continue
            
            # Stack and pad waveforms
            max_length = max(w.size(1) for w in batch_waveforms)
            padded_waveforms = []
            
            for waveform in batch_waveforms:
                if waveform.size(1) < max_length:
                    padding = max_length - waveform.size(1)
                    padded = torch.nn.functional.pad(waveform, (0, padding))
                    padded_waveforms.append(padded)
                else:
                    padded_waveforms.append(waveform)
            
            stacked_waveforms = torch.stack(padded_waveforms).to(embedding_model.device)
            
            # Get embeddings for batch
            with torch.no_grad():
                raw_embeddings = embedding_model(stacked_waveforms)
            
            # Convert to numpy and normalize
            if isinstance(raw_embeddings, torch.Tensor):
                embeddings_array = raw_embeddings.detach().cpu().numpy()
            else:
                embeddings_array = np.array(raw_embeddings)
            
            # Ensure 2D shape [batch_size, embedding_dim]
            if embeddings_array.ndim > 2:
                embeddings_array = embeddings_array.squeeze()
            elif embeddings_array.ndim == 1:
                embeddings_array = embeddings_array.reshape(1, -1)
            
            # Normalize each embedding
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            embeddings_array = embeddings_array / norms
            
            # Store embeddings by segment ID
            for segment_id, embedding in zip(batch_ids, embeddings_array):
                embeddings[segment_id] = embedding
                
        except Exception as e:
            logger.error(f"[{content_id}] Failed to process embedding batch {batch_start}-{batch_end}: {e}")
            continue
    
    logger.info(f"[{content_id}] Successfully extracted {len(embeddings)} embeddings")
    return embeddings


def create_audio_segments_from_words(word_table: WordTable, 
                                   min_duration: float = 0.15) -> List[Dict[str, Any]]:
    """
    Create audio segments from word table for batch embedding extraction.
    
    Args:
        word_table: WordTable instance
        min_duration: Minimum duration for segments (filter out very short ones)
        
    Returns:
        List of segment dicts with 'start_time', 'end_time', 'segment_id', 'word_indices'
    """
    if word_table is None or word_table.df is None or len(word_table.df) == 0:
        return []
    
    segments = []
    df = word_table.df
    
    for idx, row in df.iterrows():
        duration = row['end'] - row['start']
        
        # Skip very short segments
        if duration < min_duration:
            continue
            
        segments.append({
            'start_time': row['start'],
            'end_time': row['end'],
            'segment_id': f"word_{idx}",
            'word_indices': [idx],
            'duration': duration,
            'text': row['text']
        })
    
    return segments


def create_speaker_segments_from_centroids(speaker_centroids: Dict[str, Any],
                                         word_table: WordTable) -> List[Dict[str, Any]]:
    """
    Create audio segments from speaker centroids for batch embedding extraction.
    
    Args:
        speaker_centroids: Dict with speaker centroid information
        word_table: WordTable instance for timing information
        
    Returns:
        List of segment dicts for centroid speakers
    """
    segments = []
    
    if not speaker_centroids or word_table is None:
        return segments
    
    # Extract segments for each speaker centroid
    for speaker_id, centroid_data in speaker_centroids.items():
        if 'segments' in centroid_data:
            for i, segment_info in enumerate(centroid_data['segments']):
                segments.append({
                    'start_time': segment_info['start'],
                    'end_time': segment_info['end'],
                    'segment_id': f"centroid_{speaker_id}_{i}",
                    'speaker_id': speaker_id,
                    'duration': segment_info['end'] - segment_info['start'],
                    'confidence': segment_info.get('confidence', 1.0)
                })
    
    return segments


def stack_embeddings_efficiently(embeddings_dict: Dict[str, np.ndarray],
                                segment_ids: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Stack embeddings into a batch tensor for efficient processing.
    
    Args:
        embeddings_dict: Dict mapping segment_id to embedding array
        segment_ids: List of segment IDs to stack (in order)
        
    Returns:
        Tuple of (stacked_embeddings, valid_segment_ids)
    """
    valid_embeddings = []
    valid_ids = []
    
    for segment_id in segment_ids:
        if segment_id in embeddings_dict:
            embedding = embeddings_dict[segment_id]
            
            # Ensure consistent shape
            if isinstance(embedding, (list, tuple)):
                embedding = np.array(embedding)
            
            if embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)
            elif embedding.ndim > 2:
                embedding = embedding.squeeze()
                if embedding.ndim == 1:
                    embedding = embedding.reshape(1, -1)
            
            valid_embeddings.append(embedding)
            valid_ids.append(segment_id)
    
    if not valid_embeddings:
        return np.array([]), []
    
    # Stack all embeddings
    stacked = np.vstack(valid_embeddings)
    
    # Normalize the stacked embeddings
    norms = np.linalg.norm(stacked, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    stacked = stacked / norms
    
    return stacked, valid_ids


def compute_similarity_matrix_batch(embeddings1: np.ndarray,
                                   embeddings2: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity matrix between two sets of embeddings efficiently.
    
    Args:
        embeddings1: First set of embeddings [N, D]
        embeddings2: Second set of embeddings [M, D]
        
    Returns:
        Similarity matrix [N, M]
    """
    if embeddings1.size == 0 or embeddings2.size == 0:
        return np.array([])
    
    # Ensure 2D
    if embeddings1.ndim == 1:
        embeddings1 = embeddings1.reshape(1, -1)
    if embeddings2.ndim == 1:
        embeddings2 = embeddings2.reshape(1, -1)
    
    # Normalize embeddings
    norm1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
    norm2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)
    
    norm1[norm1 == 0] = 1
    norm2[norm2 == 0] = 1
    
    embeddings1_norm = embeddings1 / norm1
    embeddings2_norm = embeddings2 / norm2
    
    # Compute cosine similarity matrix
    similarity_matrix = np.dot(embeddings1_norm, embeddings2_norm.T)
    
    return similarity_matrix


def compute_segment_overlap_analysis(word_table, diarization_segments, boundary_touch_threshold: float = 0.05) -> pd.DataFrame:
    """
    Compute whisper segment vs diarization segment overlap analysis.
    This analysis is computed once and shared between stage4 and stage5.
    
    Args:
        word_table: WordTable instance with whisper segment information
        diarization_segments: List of diarization segments
        boundary_touch_threshold: Threshold for considering segments as boundary-touching
        
    Returns:
        DataFrame with columns:
        - segment_index: Whisper segment index
        - assignment_strategy: 'segment_level', 'word_level', or 'no_overlap'
        - overlapping_speakers: Number of speakers with significant overlap
        - primary_speaker: Speaker for single-speaker segments
        - overlap_details: Detailed overlap information
    """
    # Merge consecutive same-speaker diarization segments first
    from .stage5_overlap import merge_consecutive_same_speaker_segments
    merged_diarization_segments = merge_consecutive_same_speaker_segments(diarization_segments)
    
    # Create DataFrames for efficient processing
    diar_df = pd.DataFrame(merged_diarization_segments)
    
    # Get unique whisper segments from word table
    whisper_segments = word_table.df.groupby('segment_index').agg({
        'start': 'min',
        'end': 'max'
    }).reset_index()
    
    logger.info(f"Computing segment overlap analysis: {len(whisper_segments)} whisper segments, "
                f"{len(merged_diarization_segments)} diarization segments")
    
    # Use the vectorized calculation from stage5
    from .stage5_overlap import calculate_segment_overlap_vectorized
    segment_overlap_results = calculate_segment_overlap_vectorized(whisper_segments, diar_df)
    
    # Log analysis summary
    strategy_counts = segment_overlap_results['assignment_strategy'].value_counts()
    logger.info("Segment overlap analysis summary:")
    for strategy, count in strategy_counts.items():
        percentage = (count / len(segment_overlap_results)) * 100
        logger.info(f"  - {strategy}: {count} segments ({percentage:.1f}%)")
    
    return segment_overlap_results


def smart_join_words(words: List[str]) -> str:
    """
    Join words together with proper spacing, handling punctuation correctly.
    
    This function joins words with spaces but avoids adding spaces before
    punctuation marks that should attach to the previous word.
    
    Args:
        words: List of word strings to join
        
    Returns:
        Properly joined text string
        
    Examples:
        >>> smart_join_words(['Hello', ',', 'world', '!'])
        'Hello, world!'
        
        >>> smart_join_words(['Visit', 'goacpr', '.', 'org'])
        'Visit goacpr.org'
        
        >>> smart_join_words(['L', '.', 'A', '.'])
        'L.A.'
    """
    if not words:
        return ""
    
    if len(words) == 1:
        return words[0]
    
    # Punctuation that should attach to the previous word (no space before)
    attach_to_previous = {'.', ',', '!', '?', ';', ':', ')', ']', '}', '%', "'", '"', "'s", "'t", "'d", "'ll", "'re", "'ve", "'m"}

    # Punctuation that should attach to the next word (no space after)
    attach_to_next = {'(', '[', '{', '"', "'", '$', '€', '£', '¥'}

    # Punctuation that should attach to both sides (no spaces on either side)
    attach_both_sides = {'-'}

    # Common domain extensions
    domain_extensions = {'org', 'com', 'net', 'edu', 'gov', 'mil', 'io', 'co', 'tv', 'fm', 'ca', 'uk', 'de', 'fr', 'jp', 'cn', 'au'}
    
    # Build the result
    result = []
    for i, word in enumerate(words):
        if i == 0:
            # First word always gets added
            result.append(word)
        else:
            # Check if we should add a space before this word
            word_lower = word.lower()

            # Handle punctuation that attaches to both sides (like hyphens)
            if word in attach_both_sides:
                result.append(word)
            # Don't add space if this word/punctuation should attach to previous
            elif word in attach_to_previous or word_lower in attach_to_previous:
                result.append(word)
            # Don't add space if previous word should attach to this one
            elif i > 0 and words[i-1] in attach_to_next:
                result.append(word)
            # Don't add space if previous word attaches to both sides (like hyphens)
            elif i > 0 and words[i-1] in attach_both_sides:
                result.append(word)
            # Don't add space if this word starts with a hyphen (part of hyphenated word like "re-envahir")
            elif word.startswith('-'):
                result.append(word)
            # Special case for contractions and possessives
            elif word_lower.startswith("'") or word_lower.startswith("'"):
                result.append(word)
            # Special case for domain names and abbreviations (e.g., "goacpr.org", "L.A.")
            elif i > 1 and words[i-1] == '.' and result and len(result[-1]) > 0 and result[-1][-1] != ' ':
                # Previous word was a period and before that was text without trailing space
                # Check if this looks like part of a domain or abbreviation
                if (word.lower() in ['org', 'com', 'net', 'edu', 'gov', 'mil'] or  # domain extensions
                    (len(word) == 1 and word.isupper()) or  # single letter (like A in L.A.)
                    (i + 1 < len(words) and words[i + 1] == '.')):  # followed by another period
                    # This is likely part of domain/abbreviation, don't add space
                    result.append(word)
                else:
                    # This is a regular word after a period, add space
                    result.append(' ')
                    result.append(word)
            # Special case: handle common website extensions that might be separate words
            elif (word.lower() in ['org', 'com', 'net', 'edu', 'gov', 'mil', 'io', 'co', 'tv', 'fm'] and 
                  i > 0 and words[i-1] == '.' and 
                  i > 1 and result and len(result) >= 2 and result[-2] not in [' ', '\n', '\t']):
                # This is likely a domain extension after a period, don't add space
                result.append(word)
            else:
                # Normal case: add space before word
                result.append(' ')
                result.append(word)
    
    return ''.join(result) 