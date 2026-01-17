#!/usr/bin/env python3
"""
Stage 6: Speaker Assignment with Pyannote Embeddings
===================================================

Sixth stage of the stitch pipeline that performs audio-based speaker identification.

Key Responsibilities:
- Extract audio segments for all words needing speaker assignment
- Generate speaker embeddings using pyannote/embedding model
- Build speaker centroids from diarization ground truth
- Merge similar speakers (>0.8 similarity threshold)
- Assign words based on embedding similarity to centroids
- Handle outlier detection and reassignment
- Process in optimized batches (max 64 segments)

Input:
- WordTable from Stage 5 with some assignments and categorized words
- Audio file (wav format, 16kHz)
- Diarization data for ground truth speakers
- Configuration settings for thresholds and model parameters

Output:
- WordTable with speaker assignments based on audio embeddings
- Speaker centroids for each identified speaker
- Assignment statistics including confidence scores
- Quality metrics for embeddings and assignments

Key Components:
- SpeakerEmbedding: Data class for embedding metadata
- SpeakerCentroid: Data class for speaker representations
- ComprehensiveSpeakerProcessor: Main processing class with batching and clustering

Quality Tiers:
- High quality: segments ≥2.0s (prioritized for centroids)
- Medium quality: segments 1.5-2.0s (decent for centroids)
- Low quality: segments <1.5s (used when necessary)

Performance:
- Most computationally expensive stage (20-25% of pipeline time)
- Uses GPU/MPS acceleration when available
- Batch processing for efficiency (up to 64 segments)
- Shared audio and model resources from pipeline

Methods:
- speaker_stage(): Main entry point called by stitch pipeline
- ComprehensiveSpeakerProcessor.process_comprehensive_speaker_assignment(): Core processing
"""

import logging
import time
import aiohttp
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

from src.utils.paths import get_project_root
from dataclasses import dataclass
from collections import defaultdict, Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN, AgglomerativeClustering
import sys
import json
import argparse
import torch
import os

# Ensure homebrew libraries (ffmpeg for torchcodec/pyannote) are discoverable
_homebrew_lib = '/opt/homebrew/lib'
for _dyld_var in ['DYLD_LIBRARY_PATH', 'DYLD_FALLBACK_LIBRARY_PATH']:
    if _homebrew_lib not in os.environ.get(_dyld_var, ''):
        os.environ[_dyld_var] = f"{_homebrew_lib}:{os.environ.get(_dyld_var, '')}"


from src.utils.logger import setup_worker_logger
from .stage3_tables import WordTable

logger = setup_worker_logger('stitch')
logger.setLevel(logging.INFO)

# Add the project root to Python path
sys.path.append(str(get_project_root()))

# Model server imports removed - using direct model loading only


# Import pyannote models
from pyannote.audio import Audio


@dataclass
class SpeakerEmbedding:
    """Represents a speaker embedding with metadata."""
    segment_id: str
    embedding: np.ndarray
    confidence: float
    duration: float
    word_count: int
    text: str
    diarization_speaker: str
    diarization_confidence: float
    quality_metrics: Dict
    word_ids: List[str]  # Word IDs in this embedding segment


@dataclass
class SpeakerCentroid:
    """Represents a speaker centroid with statistics."""
    speaker_id: str
    centroid: np.ndarray
    embeddings_count: int
    mean_confidence: float
    std_deviation: float
    total_duration: float
    quality_score: float
    quality_source: str  # Track which method was used to create this centroid




class ComprehensiveSpeakerProcessor:
    """
    Complete end-to-end speaker processing with pyannote embeddings,
    intelligent batching, clustering, and outlier detection.
    """
    
    def __init__(self, config: Dict = None, 
                 shared_audio_data: Optional[Tuple] = None,
                 shared_embedding_model: Optional[torch.nn.Module] = None):
        self.config = config or {}
        
        # Shared resources from stitch pipeline
        self.shared_audio_data = shared_audio_data
        self.shared_embedding_model = shared_embedding_model
        
        # Processing parameters
        self.max_batch_size = 64
        self.min_segment_duration = 1.0
        self.min_embedding_quality = 0.5
        self.min_diarization_confidence = 0.7
        self.outlier_threshold = 2.0
        self.min_similarity_threshold = 0.3  # Lowered from 0.6 - be more permissive
        self.cross_speaker_threshold = 0.6   # Raised back up - require strong confidence for reassignment
        self.cross_speaker_min_gain = 0.3    # Require significant similarity gain to override diarization
        self.min_centroid_embeddings = 2
        
        # Model server deprecated - using direct model loading only
        self.model_server_enabled = False
        
        # Local model configuration
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.hf_token = None
        self.pyannote_embedding = shared_embedding_model  # Use shared model if available
        self.audio_loader = None
        
        # Set environment variable for MPS fallback
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
        
        logger.info(f"Device configured: {self.device}")
        if torch.backends.mps.is_available():
            logger.info("MPS backend available and enabled for acceleration")
        
        # Log shared resource usage
        if self.shared_audio_data:
            logger.debug(f"Using pipeline: {type(self.shared_audio_data)}")
        else:
            logger.info("No shared audio data provided")
        
        if self.shared_embedding_model:
            logger.debug(f"Using pipeline: {type(self.shared_embedding_model)}, callable: {callable(self.shared_embedding_model)}")
        else:
            logger.info("No shared embedding model provided")
        
        # Configure from config if available
        if config:
            self._load_config(config)
        
        # Processing state
        self.speaker_centroids: Dict[str, SpeakerCentroid] = {}
        self.outlier_embeddings: List[SpeakerEmbedding] = []
        
        logger.info("Comprehensive speaker processor initialized")
        logger.info(f"  - Max batch size: {self.max_batch_size}")
        logger.info(f"  - Direct model loading only (model server deprecated)")
        logger.info(f"  - Device: {self.device}")
        logger.info(f"  - MPS available: {torch.backends.mps.is_available()}")
        logger.info(f"  - Using shared resources: audio={self.shared_audio_data is not None}, model={self.shared_embedding_model is not None}")
    
    def _load_config(self, config: Dict) -> None:
        """Load configuration parameters."""
        processing_config = config.get('processing', {})
        
        # Get HF token from config
        self.hf_token = processing_config.get('hf_token')
        
        # Override defaults with config values
        speaker_config = processing_config.get('speaker_assignment', {})
        self.max_batch_size = speaker_config.get('max_batch_size', 64)
        self.min_segment_duration = speaker_config.get('min_segment_duration', 1.0)
        self.min_embedding_quality = speaker_config.get('min_embedding_quality', 0.5)
        self.outlier_threshold = speaker_config.get('outlier_threshold', 2.0)
        
        logger.debug(f"Configuration loaded: batch_size={self.max_batch_size}")
    
    def _initialize_local_models(self):
        """Initialize local pyannote models for embedding - uses shared model if available."""
        # Check if we have a valid callable embedding model
        if self.pyannote_embedding is not None and callable(self.pyannote_embedding):
            logger.info("Using existing embedding model (shared or previously initialized)")
            # Initialize audio loader if needed (only if not using shared audio)
            if self.audio_loader is None and self.shared_audio_data is None:
                self.audio_loader = Audio(sample_rate=16000, mono="downmix")
            return

        # Use shared embedding model if available and valid
        if self.shared_embedding_model is not None and callable(self.shared_embedding_model):
            logger.debug(f"Using pipeline: {type(self.shared_embedding_model)}")
            self.pyannote_embedding = self.shared_embedding_model
        else:
            # Clear any invalid shared model reference
            if self.shared_embedding_model is not None:
                logger.warning(f"Shared embedding model is not callable - type: {type(self.shared_embedding_model)}, callable: {callable(self.shared_embedding_model)}")
            else:
                logger.info("No shared embedding model available")
            
            logger.info(f"Initializing local pyannote embedding model on {self.device}...")
            try:
                # Load the pyannote/embedding model directly (not using Inference)
                from pyannote.audio import Model
                model = Model.from_pretrained(
                    "pyannote/embedding",
                    use_auth_token=self.hf_token
                )
                if model is None:
                    raise RuntimeError("Model.from_pretrained returned None")
                
                self.pyannote_embedding = model.to(self.device)
                if self.pyannote_embedding is None:
                    raise RuntimeError("Model.to(device) returned None")
                    
                logger.info(f"Pyannote embedding model loaded successfully on {self.device}")
                
            except Exception as e:
                logger.error(f"Failed to initialize pyannote embedding model: {e}")
                raise RuntimeError(f"Failed to initialize pyannote embedding model: {e}")

        # Final validation that embedding model is properly initialized
        if self.pyannote_embedding is None:
            raise RuntimeError(f"Failed to initialize pyannote embedding model: 'NoneType' object has no attribute 'to'")
        if not callable(self.pyannote_embedding):
            raise RuntimeError(f"Embedding model initialization failed - model is not callable: {type(self.pyannote_embedding)}")
        
        # Initialize audio loader if needed (only if not using shared audio)
        if self.audio_loader is None and self.shared_audio_data is None:
            self.audio_loader = Audio(sample_rate=16000, mono="downmix")
    
    async def process_speakers(self, 
                             word_table: WordTable,
                             audio_path: Optional[Path],
                             content_id: str) -> Dict[str, Any]:
        """
        Complete end-to-end speaker processing pipeline.
        
        Args:
            word_table: WordTable with speaker context analysis
            audio_path: Path to audio file for embeddings
            content_id: Content ID for logging
            
        Returns:
            Processing results with statistics
        """
        start_time = time.time()
        logger.info(f"[{content_id}] Starting comprehensive speaker processing")
        
        # Validate audio path
        if not audio_path or not audio_path.exists():
            raise ValueError(f"Valid audio file required for embedding calculation. Audio path: {audio_path}")
        
        # Initialize local models once at the beginning
        logger.info(f"[{content_id}] Initializing pyannote model once at beginning...")
        self._initialize_local_models()
        logger.info(f"[{content_id}] Model initialization complete")
        
        try:
            # Step 1: Extract viable segments for embedding
            segments = self._extract_embedding_segments(word_table, content_id)
            if not segments:
                return {'status': 'no_segments', 'message': 'No viable segments for embedding'}
            
            # Step 2: Calculate embeddings in optimized batches
            embeddings = await self._calculate_embeddings_batched(segments, audio_path, content_id)
            if not embeddings:
                return {'status': 'no_embeddings', 'message': 'Failed to calculate embeddings'}
            
            # Step 3: Build speaker centroids from diarization ground truth
            centroids = self._build_speaker_centroids(embeddings, content_id)
            if not centroids:
                return {'status': 'no_centroids', 'message': 'Failed to build speaker centroids'}
            
            # Skip speaker reassignment - we trust pyannote diarization
            # Just calculate statistics without modifying assignments
            final_stats = self._calculate_embedding_statistics(embeddings, centroids, content_id)
            
            total_time = time.time() - start_time
            
            # Get comprehensive quality summary
            quality_summary = self._get_quality_summary()
            
            result = {
                'status': 'completed',
                'total_time': total_time,
                'stages': {
                    'segments_extracted': len(segments),
                    'embeddings_calculated': len(embeddings),
                    'centroids_built': len(centroids)
                },
                'final_stats': final_stats,
                'centroids_summary': self._get_centroids_summary(),
                'quality_summary': quality_summary,  # Enhanced quality reporting
                'speaker_centroids': self.get_speaker_centroid_vectors(),  # Add centroid vectors
                'speaker_centroid_data': self._get_centroid_data_for_stage6b(),  # Pass centroid data to stage 6b
            }
            
            # Enhanced logging with quality information
            logger.info(f"[{content_id}] Speaker processing completed in {total_time:.2f}s")
            logger.info(f"[{content_id}] Final assignment rate: {final_stats.get('assignment_rate', 0):.1f}%")
            logger.info(f"[{content_id}] Quality distribution: HIGH={quality_summary['quality_tiers']['HIGH']}, "
                       f"MEDIUM={quality_summary['quality_tiers']['MEDIUM']}, "
                       f"LOW={quality_summary['quality_tiers']['LOW']}, "
                       f"FALLBACK={quality_summary['quality_tiers']['FALLBACK']}")
            logger.info(f"[{content_id}] Quality metrics: avg_quality={quality_summary['quality_statistics']['mean_quality']:.3f}, "
                       f"avg_stability={quality_summary['quality_statistics']['mean_stability']:.3f}, "
                       f"avg_core_confidence={quality_summary['quality_statistics']['mean_core_confidence']:.3f}")
            
            # Add consistency logging for debugging
            for speaker_id, centroid in self.speaker_centroids.items():
                quality_source = getattr(centroid, 'quality_source', 'MISSING')
                embedding_source = getattr(centroid, 'embedding_source', 'MISSING')
                logger.debug(f"[{content_id}] Speaker {speaker_id}: quality_source={quality_source}, "
                           f"embedding_source={embedding_source}, quality_score={centroid.quality_score:.3f}")
            logger.info(f"[{content_id}] Embedding sources: {dict(quality_summary['embedding_sources'])}")
            
            return result
            
        except Exception as e:
            logger.error(f"[{content_id}] Speaker processing failed: {str(e)}")
            logger.error(f"[{content_id}] Error details:", exc_info=True)
            return {
                'status': 'error',
                'error': str(e),
                'total_time': time.time() - start_time
            }
    
    def _extract_embedding_segments(self, word_table: WordTable, content_id: str) -> List[Dict]:
        """
        Extract segments suitable for embedding calculation from word table.
        Uses transcript segments with single-speaker assignments.
        """
        logger.info(f"[{content_id}] Extracting segments for embedding analysis")
        
        # Debug: Log current word table state
        total_words = len(word_table.df)
        speaker_counts = word_table.df['speaker_current'].value_counts()
        assignment_stage_counts = word_table.df.get('assignment_stage', pd.Series()).value_counts()
        
        logger.info(f"[{content_id}] Word table debug - Total words: {total_words}")
        logger.info(f"[{content_id}] Speaker distribution: {dict(speaker_counts)}")
        logger.info(f"[{content_id}] Assignment stage distribution: {dict(assignment_stage_counts)}")
        
        # Get words with single-speaker assignments only (exclude UNKNOWN and MULTI_SPEAKER)
        # Also exclude category labels since they should be treated like UNKNOWN
        # Prioritize high-confidence deterministic assignments from Stage 4 and 5
        excluded_speakers = {'UNKNOWN', 'MULTI_SPEAKER', 'BAD_GRAMMAR_SINGLE', 'BAD_GRAMMAR_MULTI', 
                           'GOOD_GRAMMAR_SINGLE', 'GOOD_GRAMMAR_MULTI', 'NEEDS_EMBEDDING', 'NEEDS_LLM'}
        
        # Check if assignment_stage column exists
        if 'assignment_stage' in word_table.df.columns:
            deterministic_words = word_table.df[
                (~word_table.df['speaker_current'].isin(excluded_speakers)) & 
                (word_table.df['speaker_current'].notna()) &
                (
                    # Stage 4 assignments (slam dunk)
                    (word_table.df['assignment_stage'] == 'stage4') |
                    # Stage 5 assignments (bad grammar single speaker)
                    (word_table.df['assignment_stage'] == 'stage5') |
                    # Legacy stage 5 assignments
                    (word_table.df['assignment_stage'] == 'stage5_overlap')
                )
            ].copy().sort_values('start')
        else:
            # No assignment_stage column, just use any assigned speakers
            deterministic_words = pd.DataFrame()  # Empty, will fall back to known speakers
        
        if deterministic_words.empty:
            logger.warning(f"[{content_id}] No deterministic speaker assignments found, falling back to all known speakers")
            # Fallback to any known speaker words if no deterministic assignments
            # Exclude category labels like UNKNOWN
            excluded_speakers = {'UNKNOWN', 'MULTI_SPEAKER', 'BAD_GRAMMAR_SINGLE', 'BAD_GRAMMAR_MULTI', 
                               'GOOD_GRAMMAR_SINGLE', 'GOOD_GRAMMAR_MULTI', 'NEEDS_EMBEDDING', 'NEEDS_LLM'}
            known_speaker_words = word_table.df[
                (~word_table.df['speaker_current'].isin(excluded_speakers)) & 
                (word_table.df['speaker_current'].notna())
            ].copy().sort_values('start')
            
            # If still no known speakers, try even more permissive - any speaker assignment
            if known_speaker_words.empty:
                logger.warning(f"[{content_id}] No known speakers found, trying any speaker assignment")
                # Check if we have any speaker assignments at all from diarization
                any_speaker_words = word_table.df[
                    (word_table.df['speaker_current'].notna()) &
                    (word_table.df['speaker_current'] != '') &
                    (word_table.df['speaker_current'] != 'UNKNOWN')
                ].copy().sort_values('start')
                
                if any_speaker_words.empty:
                    logger.error(f"[{content_id}] No speaker assignments found at all - check if previous stages are working")
                    # As last resort, try diarization speakers
                    if 'diarization_speaker' in word_table.df.columns:
                        diarization_words = word_table.df[
                            (word_table.df['diarization_speaker'].notna()) &
                            (word_table.df['diarization_speaker'] != '') &
                            (word_table.df['diarization_speaker'] != 'UNKNOWN')
                        ].copy().sort_values('start')
                        if not diarization_words.empty:
                            logger.info(f"[{content_id}] Using diarization speakers as fallback: {len(diarization_words)} words")
                            # Copy diarization speaker to current speaker for processing
                            diarization_words = diarization_words.copy()
                            diarization_words['speaker_current'] = diarization_words['diarization_speaker']
                            known_speaker_words = diarization_words
                        else:
                            logger.error(f"[{content_id}] No diarization speakers found either")
                            known_speaker_words = pd.DataFrame()
                    else:
                        logger.error(f"[{content_id}] No diarization_speaker column found")
                        known_speaker_words = pd.DataFrame()
                else:
                    known_speaker_words = any_speaker_words
        else:
            known_speaker_words = deterministic_words
            # Count words with actual speaker assignments (not categories)
            excluded_speakers = {'UNKNOWN', 'MULTI_SPEAKER', 'BAD_GRAMMAR_SINGLE', 'BAD_GRAMMAR_MULTI', 
                               'GOOD_GRAMMAR_SINGLE', 'GOOD_GRAMMAR_MULTI', 'NEEDS_EMBEDDING', 'NEEDS_LLM'}
            total_known = len(word_table.df[
                (~word_table.df['speaker_current'].isin(excluded_speakers)) &
                (word_table.df['speaker_current'].notna())
            ])
            logger.info(f"[{content_id}] Using {len(known_speaker_words)} deterministic assignments out of {total_known} total known speakers")
        
        if known_speaker_words.empty:
            logger.warning(f"[{content_id}] No words with known speakers found")
            return []
        
        logger.info(f"[{content_id}] Found {len(known_speaker_words)} words for embedding generation")
        
        # Group words by segment and check for single-speaker segments
        agg_dict = {
            'speaker_current': lambda x: x.iloc[0] if x.nunique() == 1 else 'MULTI_SPEAKER',
            'text': lambda x: ' '.join(x),
            'start': 'min',
            'end': 'max',
            'word_id': list
        }
        
        # Only include diarization_confidence if it exists
        if 'diarization_confidence' in known_speaker_words.columns:
            agg_dict['diarization_confidence'] = lambda x: x.mean()
        
        segments_df = known_speaker_words.groupby('segment_index').agg(agg_dict).reset_index()
        
        # Filter to single-speaker segments only
        single_speaker_segments = segments_df[segments_df['speaker_current'] != 'MULTI_SPEAKER']
        logger.info(f"[{content_id}] Found {len(single_speaker_segments)} single-speaker segments out of {len(segments_df)} total segments")
        
        # Convert to list of segment info dictionaries
        embedding_segments = []
        for idx, segment in single_speaker_segments.iterrows():
            segment_info = {
                'segment_id': f"seg_{segment['segment_index']}",
                'speaker': segment['speaker_current'],
                'text': segment['text'],
                'start': segment['start'],
                'end': segment['end'],
                'duration': segment['end'] - segment['start'],
                'word_count': len(segment['word_id']),
                'word_ids': segment['word_id'],
                'diarization_speaker': segment['speaker_current'],  # For compatibility
                'diarization_confidence': segment.get('diarization_confidence', 0.8),  # Default confidence if missing
                'type': 'segment'
            }
            
            # Only include segments with reasonable duration for embeddings
            if segment_info['duration'] >= 0.5:  # At least 0.5 seconds
                embedding_segments.append(segment_info)
        
        logger.info(f"[{content_id}] Created {len(embedding_segments)} embedding segments (filtered for duration >= 0.5s)")
        
        # Log segment statistics
        if embedding_segments:
            durations = [seg['duration'] for seg in embedding_segments]
            speaker_counts = {}
            for seg in embedding_segments:
                speaker = seg['speaker']
                speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
            
            logger.info(f"[{content_id}] Segment duration stats: min={min(durations):.2f}s, max={max(durations):.2f}s, avg={sum(durations)/len(durations):.2f}s")
            logger.info(f"[{content_id}] Segments per speaker: {dict(speaker_counts)}")
        
        return embedding_segments
    
    async def _calculate_embeddings_batched(self, 
                                          segments: List[Dict], 
                                          audio_path: Optional[Path],
                                          content_id: str) -> List[SpeakerEmbedding]:
        """
        Calculate embeddings in optimized batches with intelligent sorting.
        """
        logger.info(f"[{content_id}] Calculating embeddings for {len(segments)} segments")
        
        # Sort segments for optimal batching (by duration and speaker)
        sorted_segments = self._optimize_segment_batching(segments, content_id)
        
        all_embeddings = []
        
        # Load audio once for all batches
        preloaded_audio = None
        if audio_path and audio_path.exists():
            logger.info(f"[{content_id}] Pre-loading audio file for all batches...")
            preloaded_audio = await self._preload_audio(audio_path, content_id)
        
        # Process each optimized batch
        batch_start = 0
        batch_num = 1
        
        while batch_start < len(sorted_segments):
            # Find the end of this batch by looking at duration ratios
            batch_end = batch_start + 1
            min_duration = sorted_segments[batch_start]['duration']
            max_duration = min_duration
            
            while batch_end < len(sorted_segments):
                next_duration = sorted_segments[batch_end]['duration']
                new_max_duration = max(max_duration, next_duration)
                duration_ratio = new_max_duration / min_duration
                
                if duration_ratio > 1.2 or batch_end - batch_start >= self.max_batch_size:
                    break
                    
                max_duration = new_max_duration
                batch_end += 1
            
            batch_segments = sorted_segments[batch_start:batch_end]
            logger.debug(f"[{content_id}] Processing")
            
            if not audio_path or not audio_path.exists():
                raise RuntimeError(f"Audio file not found at {audio_path}. Cannot calculate embeddings without audio.")
            
            # Use preloaded audio for local model
            batch_embeddings = await self._get_embeddings_from_local_model_preloaded(
                preloaded_audio, batch_segments, content_id
            )
            
            if batch_embeddings:
                all_embeddings.extend(batch_embeddings)
            
            batch_start = batch_end
            batch_num += 1
        
        # Quality filtering
        quality_embeddings = self._filter_quality_embeddings(all_embeddings, content_id)
        
        logger.info(f"[{content_id}] Final embeddings: {len(quality_embeddings)} high-quality from {len(all_embeddings)} calculated")
        return quality_embeddings
    
    def _optimize_segment_batching(self, segments: List[Dict], content_id: str) -> List[Dict]:
        """
        Optimize segment ordering for efficient stacked batch processing.
        
        Creates batches of up to 64 segments with no more than 1.2x duration difference
        between the shortest and longest segments in each batch. Pads batches to 64
        when possible while staying within the 1.2x constraint.
        """
        logger.info(f"[{content_id}] Optimizing stacked batching for {len(segments)} segments")
        
        if not segments:
            return segments
        
        # First, sort by duration to enable stacking by similar durations
        segments_by_duration = sorted(segments, key=lambda s: s['duration'])
        
        optimized_batches = []
        current_batch = []
        
        i = 0
        while i < len(segments_by_duration):
            segment = segments_by_duration[i]
            
            # Start a new batch if current batch is empty
            if not current_batch:
                current_batch = [segment]
                i += 1
                continue
            
            # Check if this segment can fit in current batch (1.2x constraint)
            min_duration = min(s['duration'] for s in current_batch)
            max_duration = max(s['duration'] for s in current_batch)
            new_max_duration = max(max_duration, segment['duration'])
            
            # Check if adding this segment violates the 1.2x constraint
            duration_ratio = new_max_duration / min_duration
            
            if duration_ratio <= 1.2 and len(current_batch) < self.max_batch_size:
                # Segment fits within constraints, add to current batch
                current_batch.append(segment)
                i += 1
            else:
                # Can't add this segment, try to pad current batch to 64 if possible
                padded_batch = self._pad_batch_to_target(
                    current_batch, segments_by_duration[i:], content_id
                )
                
                # Remove padded segments from remaining segments
                padded_segment_ids = {s['segment_id'] for s in padded_batch}
                original_segment_ids = {s['segment_id'] for s in current_batch}
                added_segment_ids = padded_segment_ids - original_segment_ids
                
                # Skip the segments we added during padding
                j = i
                while j < len(segments_by_duration):
                    if segments_by_duration[j]['segment_id'] in added_segment_ids:
                        segments_by_duration.pop(j)
                    else:
                        j += 1
                
                optimized_batches.append(padded_batch)
                current_batch = []
                # Don't increment i, process the current segment in next iteration
        
        # Add any remaining batch
        if current_batch:
            # Try to pad the final batch
            final_batch = self._pad_batch_to_target(current_batch, [], content_id)
            optimized_batches.append(final_batch)
        
        # Flatten all batches back into a single list
        final_segments = []
        for batch in optimized_batches:
            final_segments.extend(batch)
        
        # Log batching statistics
        batch_sizes = [len(batch) for batch in optimized_batches]
        batch_duration_ratios = []
        
        for batch in optimized_batches:
            if len(batch) > 1:
                durations = [s['duration'] for s in batch]
                ratio = max(durations) / min(durations)
                batch_duration_ratios.append(ratio)
        
        logger.info(f"[{content_id}] Stacked batching optimization complete:")
        logger.info(f"[{content_id}] - Created {len(optimized_batches)} optimized batches")
        logger.info(f"[{content_id}] - Batch sizes: {batch_sizes}")
        logger.info(f"[{content_id}] - Average batch size: {sum(batch_sizes) / len(batch_sizes):.1f}")
        logger.info(f"[{content_id}] - Max duration ratios: {[f'{ratio:.2f}' for ratio in batch_duration_ratios[:5]]}")
        
        return final_segments
    
    def _pad_batch_to_target(self, current_batch: List[Dict], remaining_segments: List[Dict], content_id: str) -> List[Dict]:
        """
        Try to pad a batch towards the target size (64) while maintaining 1.2x duration constraint.
        
        Args:
            current_batch: Current segments in the batch
            remaining_segments: Available segments to choose from
            content_id: Content ID for logging
            
        Returns:
            Padded batch (may be same as input if no suitable segments found)
        """
        if len(current_batch) >= self.max_batch_size:
            return current_batch
        
        padded_batch = current_batch.copy()
        min_duration = min(s['duration'] for s in current_batch)
        max_duration = max(s['duration'] for s in current_batch)
        
        # Find segments that can fit within 1.2x constraint
        candidates = []
        for segment in remaining_segments:
            # Calculate new ratio if we add this segment
            new_min = min(min_duration, segment['duration'])
            new_max = max(max_duration, segment['duration'])
            new_ratio = new_max / new_min
            
            if new_ratio <= 1.2:
                candidates.append((segment, new_ratio))
        
        # Sort candidates by how much they help us reach target size
        # Prefer segments that are closer to existing durations
        candidates.sort(key=lambda x: abs(x[0]['duration'] - (min_duration + max_duration) / 2))
        
        # Add candidates until we reach max_batch_size or run out
        for segment, ratio in candidates:
            if len(padded_batch) >= self.max_batch_size:
                break
            
            # Double-check the constraint with current batch state
            current_min = min(s['duration'] for s in padded_batch)
            current_max = max(s['duration'] for s in padded_batch)
            new_ratio = max(current_max, segment['duration']) / min(current_min, segment['duration'])
            
            if new_ratio <= 1.2:
                padded_batch.append(segment)
        
        if len(padded_batch) > len(current_batch):
            logger.debug(f"[{content_id}] Padded batch from {len(current_batch)} to {len(padded_batch)} segments")
        
        return padded_batch
    
    
    async def _preload_audio(self, audio_path: Path, content_id: str) -> Tuple[torch.Tensor, int]:
        """Preload audio file once for all batches - uses shared audio if available."""
        try:
            # Use shared audio data if available
            if self.shared_audio_data is not None:
                logger.info(f"[{content_id}] Using shared audio data (pre-loaded by pipeline)")
                return self.shared_audio_data
            
            # Fallback to loading audio ourselves
            if self.audio_loader is None:
                logger.error(f"[{content_id}] Audio loader not initialized! This should not happen.")
                raise RuntimeError("Audio loader not initialized")
            
            # Load full audio file once
            audio_result = self.audio_loader(audio_path)
            if isinstance(audio_result, tuple):
                full_waveform, sample_rate = audio_result
            else:
                full_waveform = audio_result
                sample_rate = 16000  # PyAnnote default
            
            logger.info(f"[{content_id}] Pre-loaded audio file: shape={full_waveform.shape}, sample_rate={sample_rate}")
            return full_waveform, sample_rate
            
        except Exception as e:
            logger.error(f"[{content_id}] Failed to preload audio: {e}")
            raise RuntimeError(f"Failed to preload audio: {e}")
    
    async def _get_embeddings_from_local_model_preloaded(self,
                                                       preloaded_audio: Tuple[torch.Tensor, int],
                                                       segments: List[Dict],
                                                       content_id: str) -> List[SpeakerEmbedding]:
        """Calculate embeddings using preloaded audio data."""
        try:
            # Ensure models are initialized before processing
            if self.pyannote_embedding is None:
                logger.info(f"[{content_id}] Local models not initialized, initializing now...")
                self._initialize_local_models()
            
            full_waveform, sample_rate = preloaded_audio
            
            # Sort segments by duration for optimal batching
            sorted_segments = sorted(segments, key=lambda s: s['duration'])
            
            # Process in optimized batches
            embeddings = []
            current_batch = []
            current_batch_waveforms = []
            current_batch_segments = []
            
            for segment in sorted_segments:
                # Calculate segment boundaries in samples
                start_sample = int(segment['start'] * sample_rate)
                end_sample = int(segment['end'] * sample_rate)
                
                # Extract segment waveform from preloaded audio
                segment_waveform = full_waveform[..., start_sample:end_sample]
                
                # Convert waveform to tensor if needed
                if isinstance(segment_waveform, np.ndarray):
                    segment_waveform = torch.from_numpy(segment_waveform)
                
                # Ensure waveform is on CPU and has correct shape
                segment_waveform = segment_waveform.cpu()
                if segment_waveform.dim() == 1:
                    segment_waveform = segment_waveform.unsqueeze(0)  # Add channel dimension
                elif segment_waveform.dim() == 2 and segment_waveform.size(0) > 1:
                    segment_waveform = segment_waveform.mean(dim=0, keepdim=True)  # Convert to mono
                
                # Check if this segment can be added to current batch
                if current_batch:
                    min_duration = min(s['duration'] for s in current_batch)
                    max_duration = max(s['duration'] for s in current_batch)
                    new_max_duration = max(max_duration, segment['duration'])
                    duration_ratio = new_max_duration / min_duration
                    
                    if duration_ratio <= 1.2 and len(current_batch) < self.max_batch_size:
                        # Add to current batch
                        current_batch.append(segment)
                        current_batch_waveforms.append(segment_waveform)
                        current_batch_segments.append(segment)
                    else:
                        # Process current batch and start new one
                        batch_embeddings = await self._process_batch(
                            current_batch_waveforms,
                            current_batch_segments,
                            sample_rate,
                            content_id
                        )
                        if batch_embeddings:
                            embeddings.extend(batch_embeddings)
                        
                        # Start new batch with current segment
                        current_batch = [segment]
                        current_batch_waveforms = [segment_waveform]
                        current_batch_segments = [segment]
                else:
                    # First segment in a new batch
                    current_batch = [segment]
                    current_batch_waveforms = [segment_waveform]
                    current_batch_segments = [segment]
            
            # Process any remaining segments
            if current_batch:
                batch_embeddings = await self._process_batch(
                    current_batch_waveforms,
                    current_batch_segments,
                    sample_rate,
                    content_id
                )
                if batch_embeddings:
                    embeddings.extend(batch_embeddings)
            
            if embeddings:
                logger.info(f"[{content_id}] Successfully calculated {len(embeddings)} embeddings with preloaded audio")
                return embeddings
            else:
                raise RuntimeError("No valid embeddings calculated")
            
        except Exception as e:
            logger.error(f"[{content_id}] Preloaded model embedding calculation failed: {e}")
            raise RuntimeError(f"Failed to calculate embeddings with preloaded audio: {e}")
    
    async def _get_embeddings_from_local_model(self,
                                             audio_path: Path,
                                             segments: List[Dict],
                                             content_id: str) -> List[SpeakerEmbedding]:
        """Calculate embeddings using local pyannote model with stacked batching."""
        try:
            # Models should already be initialized, but double-check and initialize if needed
            if self.pyannote_embedding is None or self.audio_loader is None:
                logger.info(f"[{content_id}] Local models not initialized, initializing now...")
                self._initialize_local_models()
            
            # Load full audio file once
            audio_result = self.audio_loader(audio_path)
            if isinstance(audio_result, tuple):
                full_waveform, sample_rate = audio_result
            else:
                full_waveform = audio_result
                sample_rate = 16000  # PyAnnote default
            
            logger.info(f"[{content_id}] Loaded full audio file: shape={full_waveform.shape}, sample_rate={sample_rate}")
            
            # Sort segments by duration for optimal batching
            sorted_segments = sorted(segments, key=lambda s: s['duration'])
            
            # Process in optimized batches
            embeddings = []
            current_batch = []
            current_batch_waveforms = []
            current_batch_segments = []
            
            for segment in sorted_segments:
                # Calculate segment boundaries in samples
                start_sample = int(segment['start'] * sample_rate)
                end_sample = int(segment['end'] * sample_rate)
                
                # Extract segment waveform
                segment_waveform = full_waveform[..., start_sample:end_sample]
                
                # Convert waveform to tensor if needed
                if isinstance(segment_waveform, np.ndarray):
                    segment_waveform = torch.from_numpy(segment_waveform)
                
                # Ensure waveform is on CPU and has correct shape
                segment_waveform = segment_waveform.cpu()
                if segment_waveform.dim() == 1:
                    segment_waveform = segment_waveform.unsqueeze(0)  # Add channel dimension
                elif segment_waveform.dim() == 2 and segment_waveform.size(0) > 1:
                    segment_waveform = segment_waveform.mean(dim=0, keepdim=True)  # Convert to mono
                
                # Check if this segment can be added to current batch
                if current_batch:
                    min_duration = min(s['duration'] for s in current_batch)
                    max_duration = max(s['duration'] for s in current_batch)
                    new_max_duration = max(max_duration, segment['duration'])
                    duration_ratio = new_max_duration / min_duration
                    
                    if duration_ratio <= 1.2 and len(current_batch) < self.max_batch_size:
                        # Add to current batch
                        current_batch.append(segment)
                        current_batch_waveforms.append(segment_waveform)
                        current_batch_segments.append(segment)
                    else:
                        # Process current batch and start new one
                        batch_embeddings = await self._process_batch(
                            current_batch_waveforms,
                            current_batch_segments,
                            sample_rate,
                            content_id
                        )
                        if batch_embeddings:
                            embeddings.extend(batch_embeddings)
                        
                        # Start new batch with current segment
                        current_batch = [segment]
                        current_batch_waveforms = [segment_waveform]
                        current_batch_segments = [segment]
                else:
                    # First segment in a new batch
                    current_batch = [segment]
                    current_batch_waveforms = [segment_waveform]
                    current_batch_segments = [segment]
            
            # Process any remaining segments
            if current_batch:
                batch_embeddings = await self._process_batch(
                    current_batch_waveforms,
                    current_batch_segments,
                    sample_rate,
                    content_id
                )
                if batch_embeddings:
                    embeddings.extend(batch_embeddings)
            
            if embeddings:
                logger.info(f"[{content_id}] Successfully calculated {len(embeddings)} embeddings with local model")
                return embeddings
            else:
                raise RuntimeError("No valid embeddings calculated")
            
        except Exception as e:
            logger.error(f"[{content_id}] Local model embedding calculation failed: {e}")
            # If models failed to initialize, try one more time
            if "Local models not initialized" in str(e) or self.pyannote_embedding is None:
                logger.info(f"[{content_id}] Attempting to reinitialize local models...")
                try:
                    self._initialize_local_models()
                    # Don't recursively retry - just re-raise if still failing
                except Exception as init_e:
                    logger.error(f"[{content_id}] Failed to reinitialize local models: {init_e}")
            raise RuntimeError(f"Failed to calculate embeddings with local model: {e}")
    
    async def _process_batch(self,
                           batch_waveforms: List[torch.Tensor],
                           batch_segments: List[Dict],
                           sample_rate: int,
                           content_id: str) -> List[SpeakerEmbedding]:
        """Process a batch of waveforms to get embeddings."""
        try:
            # Stack waveforms into a single batch
            max_length = max(w.size(1) for w in batch_waveforms)
            padded_waveforms = []
            
            for waveform in batch_waveforms:
                if waveform.size(1) < max_length:
                    padding = max_length - waveform.size(1)
                    padded = torch.nn.functional.pad(waveform, (0, padding))
                    padded_waveforms.append(padded)
                else:
                    padded_waveforms.append(waveform)
            
            stacked_waveforms = torch.stack(padded_waveforms).to(self.device)
            
            logger.debug(f"[{content_id}] Processing batch of {len(batch_segments)} segments")
            logger.debug(f"[{content_id}] Batch waveform shape: {stacked_waveforms.shape}")
            logger.debug(f"[{content_id}] Waveform device: {stacked_waveforms.device}")
            
            # Get embeddings for batch
            try:
                # Validate that the embedding model is callable
                if self.pyannote_embedding is None:
                    raise RuntimeError("Pyannote embedding model is None - model initialization failed")
                if not callable(self.pyannote_embedding):
                    raise RuntimeError(f"Pyannote embedding model is not callable - got {type(self.pyannote_embedding)}")
                
                # Store a reference to prevent potential race conditions
                embedding_model = self.pyannote_embedding
                if embedding_model is None or not callable(embedding_model):
                    raise RuntimeError(f"Embedding model became None or non-callable between checks - type: {type(embedding_model)}")
                
                # Ensure model is in eval mode
                if hasattr(embedding_model, 'eval'):
                    embedding_model.eval()
                
                # CRITICAL FIX: Clear any cached states in the model
                # PyAnnote models can have internal buffers that need clearing
                if hasattr(embedding_model, 'reset') or hasattr(embedding_model, 'reset_parameters'):
                    if hasattr(embedding_model, 'reset'):
                        embedding_model.reset()
                    elif hasattr(embedding_model, 'reset_parameters'):
                        embedding_model.reset_parameters()
                
                # Additional fix: ensure each input is processed independently
                # Process one at a time if we detect the problematic pattern
                with torch.no_grad():
                    # First try batch processing
                    raw_embeddings = embedding_model(stacked_waveforms)
                    
                    # Check if we got identical embeddings (shouldn't happen)
                    if isinstance(raw_embeddings, torch.Tensor) and raw_embeddings.shape[0] > 1:
                        first_emb = raw_embeddings[0].cpu().numpy()
                        all_same = all(torch.allclose(raw_embeddings[i].cpu(), raw_embeddings[0].cpu(), rtol=1e-5, atol=1e-7)
                                      for i in range(1, raw_embeddings.shape[0]))
                        
                        if all_same:
                            logger.warning(f"[{content_id}] Batch processing produced identical embeddings, falling back to individual processing")
                            # Process individually as fallback
                            individual_embeddings = []
                            for i, waveform in enumerate(batch_waveforms):
                                # Ensure waveform is properly shaped and on device
                                if waveform.dim() == 1:
                                    waveform = waveform.unsqueeze(0)
                                waveform = waveform.to(self.device)
                                if waveform.dim() == 2:
                                    waveform = waveform.unsqueeze(0)  # Add batch dimension
                                
                                # Get individual embedding
                                individual_emb = embedding_model(waveform)
                                individual_embeddings.append(individual_emb)
                            
                            # Stack individual results
                            raw_embeddings = torch.cat(individual_embeddings, dim=0)
                
                # Convert to numpy array
                if isinstance(raw_embeddings, torch.Tensor):
                    embeddings_array = raw_embeddings.detach().cpu().numpy()
                else:
                    embeddings_array = np.array(raw_embeddings)
                
                # Ensure 2D shape [batch_size, embedding_dim]
                if embeddings_array.ndim > 2:
                    embeddings_array = embeddings_array.squeeze()
                elif embeddings_array.ndim == 1:
                    embeddings_array = embeddings_array.reshape(1, -1)
                
                # Check for problematic identical embeddings BEFORE normalization
                if embeddings_array.shape[0] > 1:
                    # Check if all embeddings in batch are identical (shouldn't happen)
                    first_emb = embeddings_array[0]
                    all_identical = all(np.allclose(embeddings_array[i], first_emb, rtol=1e-5, atol=1e-7) 
                                       for i in range(1, embeddings_array.shape[0]))
                    if all_identical:
                        logger.error(f"[{content_id}] WARNING: All {embeddings_array.shape[0]} embeddings in batch are identical!")
                        logger.error(f"[{content_id}] First embedding values: {embeddings_array[0][:10]}")
                        logger.error(f"[{content_id}] Waveform stats - min: {stacked_waveforms.min():.4f}, max: {stacked_waveforms.max():.4f}, mean: {stacked_waveforms.mean():.4f}")
                
                # Check for the known problematic default embedding pattern
                DEFAULT_EMBEDDING_START = np.array([
                    0.03165537, 0.01190357, 0.04379321, 0.0312836, -0.019626,
                    0.05465591, -0.03783079, -0.02868066, 0.01361196, 0.04733006
                ], dtype=np.float32)
                
                # Normalize each embedding to unit length
                norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
                
                # Check for zero-norm embeddings (shouldn't happen with real audio)
                zero_norm_count = np.sum(norms == 0)
                if zero_norm_count > 0:
                    logger.warning(f"[{content_id}] Found {zero_norm_count} embeddings with zero norm!")
                
                norms[norms == 0] = 1  # Avoid division by zero
                embeddings_array = embeddings_array / norms
                
                # After normalization, check for the known problematic pattern
                problematic_count = 0
                for i, emb in enumerate(embeddings_array):
                    if len(emb) >= 10 and np.allclose(emb[:10], DEFAULT_EMBEDDING_START, rtol=1e-5, atol=1e-7):
                        problematic_count += 1
                        if problematic_count == 1:  # Log details for first occurrence
                            segment = batch_segments[i] if i < len(batch_segments) else None
                            logger.error(f"[{content_id}] DETECTED PROBLEMATIC DEFAULT EMBEDDING!")
                            if segment:
                                logger.error(f"[{content_id}] Segment: {segment.get('segment_id', 'unknown')}, duration: {segment.get('duration', 0):.2f}s")
                            logger.error(f"[{content_id}] Waveform shape: {batch_waveforms[i].shape if i < len(batch_waveforms) else 'N/A'}")
                
                if problematic_count > 0:
                    logger.error(f"[{content_id}] Total problematic embeddings in batch: {problematic_count}/{embeddings_array.shape[0]}")
                
                logger.debug(f"[{content_id}] Processed embeddings shape: {embeddings_array.shape}")
                
            except Exception as e:
                logger.error(f"[{content_id}] Failed to process pyannote output: {e}")
                if 'raw_embeddings' in locals():
                    logger.error(f"[{content_id}] Raw output type: {type(raw_embeddings)}")
                    logger.error(f"[{content_id}] Raw output: {raw_embeddings}")
                raise e
            
            # Define the known problematic default embedding pattern for filtering
            DEFAULT_EMBEDDING_START = np.array([
                0.03165537, 0.01190357, 0.04379321, 0.0312836, -0.019626,
                0.05465591, -0.03783079, -0.02868066, 0.01361196, 0.04733006
            ], dtype=np.float32)
            
            # Convert to SpeakerEmbedding objects
            embeddings = []
            skipped_count = 0
            for segment, embedding_vector in zip(batch_segments, embeddings_array):
                # Validate embedding
                if not np.isnan(embedding_vector).any():
                    # CRITICAL: Skip problematic default embeddings
                    if len(embedding_vector) >= 10 and np.allclose(embedding_vector[:10], DEFAULT_EMBEDDING_START, rtol=1e-5, atol=1e-7):
                        skipped_count += 1
                        logger.warning(f"[{content_id}] Skipping problematic default embedding for segment {segment['segment_id']}")
                        continue
                    
                    quality_metrics = self._calculate_quality_metrics(segment, embedding_vector)
                    
                    speaker_embedding = SpeakerEmbedding(
                        segment_id=segment['segment_id'],
                        embedding=embedding_vector,  # Already normalized
                        confidence=quality_metrics['overall_confidence'],
                        duration=segment['duration'],
                        word_count=segment['word_count'],
                        text=segment['text'],
                        diarization_speaker=segment['diarization_speaker'],
                        diarization_confidence=segment['diarization_confidence'],
                        quality_metrics=quality_metrics,
                        word_ids=segment.get('word_ids', [])
                    )
                    
                    embeddings.append(speaker_embedding)
            
            if skipped_count > 0:
                logger.error(f"[{content_id}] Skipped {skipped_count} segments with problematic default embeddings")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"[{content_id}] Batch processing failed: {e}")
            logger.error(f"[{content_id}] Error details:", exc_info=True)
            return []
    
    
    def _calculate_core_signature_confidence(self, embeddings: List[SpeakerEmbedding], centroid: np.ndarray, content_id: str) -> float:
        """Calculate confidence that the centroid represents the core vocal signature."""
        if not embeddings or centroid is None:
            return 0.0
        
        # Calculate similarity of each embedding to the centroid
        similarities = []
        for emb in embeddings:
            similarity = cosine_similarity(
                emb.embedding.reshape(1, -1),
                centroid.reshape(1, -1)
            )[0, 0]
            similarities.append(similarity)
        
        if not similarities:
            return 0.0
        
        # Core signature confidence is based on:
        # 1. Mean similarity (how well centroid represents embeddings)
        # 2. Consistency (low standard deviation)
        # 3. Number of embeddings (more data = more confidence)
        
        mean_similarity = np.mean(similarities)
        std_similarity = np.std(similarities)
        embedding_count = len(embeddings)
        
        # Consistency score (inverse of standard deviation)
        consistency_score = 1.0 / (1.0 + std_similarity)
        
        # Count bonus (more embeddings = more confidence, plateau at 10)
        count_bonus = min(1.0, embedding_count / 10.0)
        
        # Combined confidence
        core_confidence = mean_similarity * consistency_score * count_bonus
        
        logger.debug(f"[{content_id}] Core signature confidence: {core_confidence:.3f} "
                    f"(mean_sim={mean_similarity:.3f}, consistency={consistency_score:.3f}, "
                    f"count_bonus={count_bonus:.3f})")
        
        return core_confidence
    
    def _calculate_centroid_stability_score(self, selected_embeddings: List[SpeakerEmbedding], centroid: np.ndarray) -> float:
        """Calculate stability score measuring how consistent the selected embeddings are."""
        if len(selected_embeddings) < 2:
            return 1.0  # Perfect stability for single embedding
        
        # Calculate pairwise similarities between all selected embeddings
        embedding_vectors = np.array([emb.embedding for emb in selected_embeddings])
        pairwise_similarities = cosine_similarity(embedding_vectors)
        
        # Get upper triangle (excluding diagonal) for unique pairs
        upper_triangle = np.triu(pairwise_similarities, k=1)
        pairwise_sims = upper_triangle[upper_triangle > 0]
        
        if len(pairwise_sims) == 0:
            return 0.0
        
        # Stability is based on mean pairwise similarity
        mean_pairwise_sim = np.mean(pairwise_sims)
        std_pairwise_sim = np.std(pairwise_sims)
        
        # High stability = high mean similarity, low variance
        stability_score = mean_pairwise_sim * (1.0 / (1.0 + std_pairwise_sim))
        
        return stability_score
    
    def _build_centroid_high_quality(self, speaker_embeddings: List[SpeakerEmbedding], content_id: str) -> Optional[Dict]:
        """Build centroid using tight clustering approach for high-quality embeddings."""
        result = self._find_tight_cluster_centroid(
            speaker_embeddings[0].diarization_speaker, speaker_embeddings, content_id
        )
        
        # Add quality metadata to the centroid
        if result and result.get('centroid'):
            centroid = result['centroid']
            
            # Calculate comprehensive quality metrics for this high-quality centroid
            quality_metrics = self._calculate_comprehensive_quality_score(
                speaker_embeddings, centroid.centroid, 'tight_cluster', content_id
            )
            
            # Store quality metadata and source info
            centroid.quality_metadata = quality_metrics
            centroid.embedding_source = 'tight_cluster'
            
            # Update quality score with comprehensive calculation
            centroid.quality_score = quality_metrics['overall_quality']
            
            # Store quality metrics in result
            result['quality_metrics'] = quality_metrics
        
        return result
    
    def _build_centroid_medium_quality(self, speaker_embeddings: List[SpeakerEmbedding], content_id: str) -> Optional[Dict]:
        """Build centroid using duration-weighted averaging for medium-quality embeddings."""
        if not speaker_embeddings:
            return None
        
        speaker_id = speaker_embeddings[0].diarization_speaker
        
        # Use duration-weighted averaging
        embedding_vectors = np.array([emb.embedding for emb in speaker_embeddings])
        durations = np.array([emb.duration for emb in speaker_embeddings])
        confidences = np.array([emb.confidence for emb in speaker_embeddings])
        
        # Weight by duration (longer segments get more influence)
        weights = durations / np.sum(durations)
        weighted_centroid = np.average(embedding_vectors, axis=0, weights=weights)
        centroid_vector = weighted_centroid / np.linalg.norm(weighted_centroid)
        
        # Calculate quality metrics
        quality_metrics = self._calculate_comprehensive_quality_score(
            speaker_embeddings, centroid_vector, 'duration_weighted', content_id
        )
        
        # Create centroid
        centroid = SpeakerCentroid(
            speaker_id=speaker_id,
            centroid=centroid_vector,
            embeddings_count=len(speaker_embeddings),
            mean_confidence=np.mean(confidences),
            std_deviation=np.std(confidences),
            total_duration=np.sum(durations),
            quality_score=quality_metrics['overall_quality'],
            quality_source='MEDIUM'
        )
        
        # Store quality metadata and source info
        centroid.quality_metadata = quality_metrics
        centroid.embedding_source = 'duration_weighted'
        
        stability_score = self._calculate_centroid_stability_score(speaker_embeddings, centroid_vector)
        
        return {
            'centroid': centroid,
            'embeddings_used': len(speaker_embeddings),
            'tightness_score': stability_score,
            'cluster_source': 'duration_weighted',
            'outliers_detected': 0,
            'quality_metrics': quality_metrics
        }
    
    def _build_centroid_single_best(self, speaker_embeddings: List[SpeakerEmbedding], content_id: str) -> Optional[Dict]:
        """Build centroid using single best embedding for low-quality speakers."""
        if not speaker_embeddings:
            return None
        
        speaker_id = speaker_embeddings[0].diarization_speaker
        
        # Select best embedding based on combined score
        best_embedding = None
        best_score = -1
        
        for emb in speaker_embeddings:
            # Combined score: duration * confidence
            score = emb.duration * emb.confidence
            if score > best_score:
                best_score = score
                best_embedding = emb
        
        if best_embedding is None:
            return None
        
        # Use the best embedding as centroid
        centroid_vector = best_embedding.embedding / np.linalg.norm(best_embedding.embedding)
        
        # Calculate quality metrics
        quality_metrics = self._calculate_comprehensive_quality_score(
            [best_embedding], centroid_vector, 'single_best', content_id
        )
        
        # Create centroid
        centroid = SpeakerCentroid(
            speaker_id=speaker_id,
            centroid=centroid_vector,
            embeddings_count=1,
            mean_confidence=best_embedding.confidence,
            std_deviation=0.0,
            total_duration=best_embedding.duration,
            quality_score=quality_metrics['overall_quality'],
            quality_source='LOW'
        )
        
        # Store quality metadata and source info
        centroid.quality_metadata = quality_metrics
        centroid.embedding_source = 'single_best'
        
        return {
            'centroid': centroid,
            'embeddings_used': 1,
            'tightness_score': 1.0,  # Perfect tightness for single embedding
            'cluster_source': 'single_best',
            'outliers_detected': len(speaker_embeddings) - 1,
            'quality_metrics': quality_metrics
        }
    
    def _build_centroid_fallback(self, speaker_embeddings: List[SpeakerEmbedding], content_id: str) -> Optional[Dict]:
        """Build fallback centroid using all available embeddings for very low-quality speakers."""
        if not speaker_embeddings:
            return None
        
        speaker_id = speaker_embeddings[0].diarization_speaker
        
        # Simple average of all embeddings
        embedding_vectors = np.array([emb.embedding for emb in speaker_embeddings])
        durations = np.array([emb.duration for emb in speaker_embeddings])
        confidences = np.array([emb.confidence for emb in speaker_embeddings])
        
        mean_embedding = np.mean(embedding_vectors, axis=0)
        centroid_vector = mean_embedding / np.linalg.norm(mean_embedding)
        
        # Calculate quality metrics (will be low due to fallback source)
        quality_metrics = self._calculate_comprehensive_quality_score(
            speaker_embeddings, centroid_vector, 'fallback', content_id
        )
        
        # Create centroid
        centroid = SpeakerCentroid(
            speaker_id=speaker_id,
            centroid=centroid_vector,
            embeddings_count=len(speaker_embeddings),
            mean_confidence=np.mean(confidences),
            std_deviation=np.std(confidences),
            total_duration=np.sum(durations),
            quality_score=quality_metrics['overall_quality'],
            quality_source='FALLBACK'
        )
        
        # Store quality metadata and source info
        centroid.quality_metadata = quality_metrics
        centroid.embedding_source = 'fallback'
        
        stability_score = self._calculate_centroid_stability_score(speaker_embeddings, centroid_vector)
        
        return {
            'centroid': centroid,
            'embeddings_used': len(speaker_embeddings),
            'tightness_score': stability_score,
            'cluster_source': 'fallback',
            'outliers_detected': 0,
            'quality_metrics': quality_metrics
        }
    
    def _calculate_comprehensive_quality_score(self, embeddings: List[SpeakerEmbedding], centroid: np.ndarray, 
                                          quality_source: str, content_id: str) -> Dict[str, float]:
        """Calculate comprehensive quality metrics for speaker centroid."""
        if not embeddings or centroid is None:
            return {
                'overall_quality': 0.0,
                'duration_quality': 0.0,
                'confidence_quality': 0.0,
                'stability_score': 0.0,
                'core_signature_confidence': 0.0,
                'embedding_count': 0
            }
        
        # Basic quality metrics
        total_duration = sum(emb.duration for emb in embeddings)
        mean_confidence = np.mean([emb.confidence for emb in embeddings])
        embedding_count = len(embeddings)
        
        # Duration quality (optimal at 10+ seconds)
        duration_quality = min(1.0, total_duration / 10.0)
        
        # Confidence quality (based on mean confidence)
        confidence_quality = mean_confidence
        
        # Stability score (how consistent embeddings are)
        stability_score = self._calculate_centroid_stability_score(embeddings, centroid)
        
        # Core signature confidence (how well centroid represents voice)
        core_signature_confidence = self._calculate_core_signature_confidence(embeddings, centroid, content_id)
        
        # Overall quality based on source type and metrics
        source_weights = {
            'tight_cluster': 1.0,
            'duration_weighted': 0.8,
            'single_best': 0.6,
            'fallback': 0.3
        }
        
        source_weight = source_weights.get(quality_source, 0.5)
        
        # Weighted combination of quality components
        overall_quality = source_weight * (
            duration_quality * 0.25 +
            confidence_quality * 0.25 +
            stability_score * 0.25 +
            core_signature_confidence * 0.25
        )
        
        return {
            'overall_quality': overall_quality,
            'duration_quality': duration_quality,
            'confidence_quality': confidence_quality,
            'stability_score': stability_score,
            'core_signature_confidence': core_signature_confidence,
            'embedding_count': embedding_count,
            'total_duration': total_duration,
            'source_weight': source_weight
        }
    
    def _calculate_quality_metrics(self, segment_info: Dict, embedding: np.ndarray) -> Dict:
        """Calculate quality metrics for an embedding."""
        # Basic embedding metrics
        embedding_norm = np.linalg.norm(embedding)
        embedding_std = np.std(embedding)
        
        # Duration-based quality
        duration = segment_info['duration']
        duration_quality = min(1.0, duration / 3.0)  # Optimal at 3+ seconds
        
        # Word count based quality
        word_count = segment_info['word_count']
        word_quality = min(1.0, word_count / 8.0)  # Optimal at 8+ words
        
        # Diarization confidence
        diarization_confidence = segment_info['diarization_confidence']
        
        # Context quality (high for single-speaker)
        context_quality = 0.9
        
        # Overall confidence calculation
        overall_confidence = (
            duration_quality * 0.25 +
            word_quality * 0.15 +
            diarization_confidence * 0.35 +
            context_quality * 0.25
        )
        
        return {
            'overall_confidence': overall_confidence,
            'duration_quality': duration_quality,
            'word_quality': word_quality,
            'diarization_confidence': diarization_confidence,
            'context_quality': context_quality,
            'embedding_norm': embedding_norm,
            'embedding_std': embedding_std
        }
    
    def _calculate_adaptive_quality_thresholds(self, speaker_embeddings: List[SpeakerEmbedding], content_id: str) -> Dict[str, float]:
        """Calculate adaptive quality thresholds based on speaker-relative quality distribution."""
        if not speaker_embeddings:
            return {'confidence': 0.5, 'duration': 1.0}
        
        confidences = [emb.confidence for emb in speaker_embeddings]
        durations = [emb.duration for emb in speaker_embeddings]
        
        # Calculate percentiles for adaptive thresholds
        conf_30th = np.percentile(confidences, 30)
        conf_70th = np.percentile(confidences, 70)
        dur_30th = np.percentile(durations, 30)
        dur_70th = np.percentile(durations, 70)
        
        # Ensure minimum reasonable thresholds
        adaptive_conf_threshold = max(0.3, conf_30th)
        adaptive_dur_threshold = max(0.5, dur_30th)
        
        # Log the adaptive thresholds for debugging
        logger.debug(f"[{content_id}] Adaptive thresholds for {len(speaker_embeddings)} embeddings:")
        logger.debug(f"[{content_id}] - Confidence: 30th={conf_30th:.3f}, 70th={conf_70th:.3f}, adaptive={adaptive_conf_threshold:.3f}")
        logger.debug(f"[{content_id}] - Duration: 30th={dur_30th:.1f}s, 70th={dur_70th:.1f}s, adaptive={adaptive_dur_threshold:.1f}s")
        
        return {
            'confidence': adaptive_conf_threshold,
            'duration': adaptive_dur_threshold,
            'confidence_70th': conf_70th,
            'duration_70th': dur_70th
        }
    
    def _select_quality_embeddings_adaptive(self, speaker_embeddings: List[SpeakerEmbedding], content_id: str) -> Dict[str, List[SpeakerEmbedding]]:
        """Select embeddings using adaptive quality tiers to maximize core signature capture."""
        if not speaker_embeddings:
            return {'high': [], 'medium': [], 'low': [], 'fallback': []}
        
        thresholds = self._calculate_adaptive_quality_thresholds(speaker_embeddings, content_id)
        
        # Classify embeddings into quality tiers
        high_quality = []
        medium_quality = []
        low_quality = []
        fallback_quality = []
        
        for emb in speaker_embeddings:
            # High quality: segments >= 2.0s with good confidence
            if emb.duration >= 2.0 and emb.confidence >= thresholds['confidence_70th']:
                high_quality.append(emb)
            # Medium quality: segments 1.5-2.0s with decent confidence
            elif emb.duration >= 1.5 and emb.confidence >= thresholds['confidence']:
                medium_quality.append(emb)
            # Low quality: segments < 1.5s with minimum viable thresholds
            elif emb.confidence >= 0.3 and emb.duration >= 0.5:
                low_quality.append(emb)
            # Fallback: everything else
            else:
                fallback_quality.append(emb)
        
        return {
            'high': high_quality,
            'medium': medium_quality,
            'low': low_quality,
            'fallback': fallback_quality
        }
    
    def _filter_quality_embeddings(self, embeddings: List[SpeakerEmbedding], content_id: str) -> List[SpeakerEmbedding]:
        """Filter embeddings by quality threshold - now uses adaptive selection."""
        if not embeddings:
            return embeddings
        
        # Group by speaker for adaptive selection
        speaker_groups = defaultdict(list)
        for emb in embeddings:
            speaker_groups[emb.diarization_speaker].append(emb)
        
        adaptive_embeddings = []
        for speaker_id, speaker_embeddings in speaker_groups.items():
            # Use adaptive selection to get best embeddings for each speaker
            quality_tiers = self._select_quality_embeddings_adaptive(speaker_embeddings, content_id)
            
            # Select embeddings in priority order: high > medium > low > fallback
            selected_embeddings = []
            if quality_tiers['high']:
                selected_embeddings = quality_tiers['high']
            elif quality_tiers['medium']:
                selected_embeddings = quality_tiers['medium']
            elif quality_tiers['low']:
                selected_embeddings = quality_tiers['low']
            else:
                selected_embeddings = quality_tiers['fallback']
            
            adaptive_embeddings.extend(selected_embeddings)
            
            logger.debug(f"[{content_id}] Speaker {speaker_id} adaptive selection: "
                        f"high={len(quality_tiers['high'])}, medium={len(quality_tiers['medium'])}, "
                        f"low={len(quality_tiers['low'])}, fallback={len(quality_tiers['fallback'])}, "
                        f"selected={len(selected_embeddings)}")
        
        logger.info(f"[{content_id}] Adaptive quality selection completed:")
        logger.info(f"[{content_id}] - Total embeddings: {len(embeddings)} calculated, {len(adaptive_embeddings)} selected")
        logger.info(f"[{content_id}] - Selection strategy: best available tier per speaker")
        logger.info(f"[{content_id}] - Speakers processed: {len(speaker_groups)}")
        
        return adaptive_embeddings
    
    def _build_speaker_centroids(self, embeddings: List[SpeakerEmbedding], content_id: str) -> Dict[str, SpeakerCentroid]:
        """
        Build speaker centroids using only high-quality embeddings from transcript segments.
        Prioritizes longer, higher-confidence segments for more stable centroids.
        """
        logger.info(f"[{content_id}] Building speaker centroids using tiered quality-aware approach from {len(embeddings)} segment-based embeddings")
        
        # Group embeddings by speaker
        # Exclude category labels and non-speaker assignments
        excluded_speakers = {'UNKNOWN', 'MULTI_SPEAKER', 'BAD_GRAMMAR_SINGLE', 'BAD_GRAMMAR_MULTI', 
                           'GOOD_GRAMMAR_SINGLE', 'GOOD_GRAMMAR_MULTI', 'NEEDS_EMBEDDING', 'NEEDS_LLM'}
        speaker_groups = defaultdict(list)
        for embedding in embeddings:
            speaker_id = embedding.diarization_speaker
            if speaker_id not in excluded_speakers:
                speaker_groups[speaker_id].append(embedding)
        
        # Analyze embedding quality distribution with segment-based metrics
        total_embeddings = sum(len(group) for group in speaker_groups.values())
        
        # Count embeddings by fixed quality thresholds (for reference)
        fixed_high_quality = sum(len([emb for emb in group if emb.duration >= 2.0 and emb.confidence >= 0.7]) 
                                    for group in speaker_groups.values())
        fixed_medium_quality = sum(len([emb for emb in group if 1.5 <= emb.duration < 2.0 and emb.confidence >= 0.6])
                                       for group in speaker_groups.values())
        fixed_low_quality = sum(len([emb for emb in group if 0.5 <= emb.duration < 1.5 and emb.confidence >= 0.5])
                                    for group in speaker_groups.values())
        
        logger.info(f"[{content_id}] Total embeddings for real speakers: {total_embeddings} "
                   f"(from {len(embeddings)} total including excluded categories)")
        logger.info(f"[{content_id}] Fixed threshold quality distribution (for reference):")
        logger.info(f"[{content_id}] - High (\u22652.0s, conf\u22650.7): {fixed_high_quality}")
        logger.info(f"[{content_id}] - Medium (1.5-2.0s, conf\u22650.6): {fixed_medium_quality}")
        logger.info(f"[{content_id}] - Low (0.5-1.5s, conf\u22650.5): {fixed_low_quality}")
        logger.info(f"[{content_id}] Note: Actual centroid building uses adaptive per-speaker thresholds")
        
        centroids = {}
        quality_stats = {
            'high_quality': 0,
            'medium_quality': 0, 
            'single_best': 0,
            'fallback': 0
        }
        
        for speaker_id, speaker_embeddings in speaker_groups.items():
            # Use adaptive quality selection to determine best approach
            quality_tiers = self._select_quality_embeddings_adaptive(speaker_embeddings, content_id)
            
            centroid_data = None
            
            # Tier 1: High Quality - Use tight clustering if enough high-quality embeddings
            if len(quality_tiers['high']) >= 2:
                logger.debug(f"[{content_id}] Using HIGH QUALITY approach for {speaker_id} with {len(quality_tiers['high'])} embeddings")
                centroid_data = self._build_centroid_high_quality(quality_tiers['high'], content_id)
                if centroid_data:
                    quality_stats['high_quality'] += 1
            
            # Tier 1b: Single High Quality - Use single high-quality embedding as centroid
            elif len(quality_tiers['high']) == 1:
                logger.debug(f"[{content_id}] Using SINGLE HIGH QUALITY embedding for {speaker_id}")
                # Use the medium quality function which handles single embeddings well
                centroid_data = self._build_centroid_medium_quality(quality_tiers['high'], content_id)
                if centroid_data:
                    quality_stats['high_quality'] += 1
            
            # Tier 2: Medium Quality - Use duration-weighted averaging
            elif len(quality_tiers['medium']) >= 1:
                logger.debug(f"[{content_id}] Using MEDIUM QUALITY approach for {speaker_id} with {len(quality_tiers['medium'])} embeddings")
                centroid_data = self._build_centroid_medium_quality(quality_tiers['medium'], content_id)
                if centroid_data:
                    quality_stats['medium_quality'] += 1
            
            # Tier 3: Single Best - Use best single embedding from low quality
            elif len(quality_tiers['low']) >= 1:
                logger.debug(f"[{content_id}] Using SINGLE BEST approach for {speaker_id} with {len(quality_tiers['low'])} embeddings")
                centroid_data = self._build_centroid_single_best(quality_tiers['low'], content_id)
                if centroid_data:
                    quality_stats['single_best'] += 1
            
            # Tier 4: Fallback - Use all available embeddings
            elif len(quality_tiers['fallback']) >= 1:
                logger.debug(f"[{content_id}] Using FALLBACK approach for {speaker_id} with {len(quality_tiers['fallback'])} embeddings")
                centroid_data = self._build_centroid_fallback(quality_tiers['fallback'], content_id)
                if centroid_data:
                    quality_stats['fallback'] += 1
            
            # Store centroid if successfully created
            if centroid_data:
                centroids[speaker_id] = centroid_data['centroid']
                
                # Log quality information
                quality_metrics = centroid_data.get('quality_metrics', {})
                logger.info(f"[{content_id}] Speaker {speaker_id} centroid: "
                           f"method={centroid_data['cluster_source']}, "
                           f"embeddings={centroid_data['embeddings_used']}, "
                           f"quality={quality_metrics.get('overall_quality', 0):.3f}, "
                           f"stability={quality_metrics.get('stability_score', 0):.3f}, "
                           f"core_confidence={quality_metrics.get('core_signature_confidence', 0):.3f}")
            else:
                logger.warning(f"[{content_id}] Failed to create centroid for speaker {speaker_id}")
            
        
        # Log quality statistics
        logger.info(f"[{content_id}] Tiered centroid building complete:")
        logger.info(f"[{content_id}] - High quality (tight clustering): {quality_stats['high_quality']} speakers")
        logger.info(f"[{content_id}] - Medium quality (duration-weighted): {quality_stats['medium_quality']} speakers")
        logger.info(f"[{content_id}] - Single best embedding: {quality_stats['single_best']} speakers")
        logger.info(f"[{content_id}] - Fallback (all embeddings): {quality_stats['fallback']} speakers")
        logger.info(f"[{content_id}] - Total centroids created: {len(centroids)}/{len(speaker_groups)} speakers")
        
        self.speaker_centroids = centroids
        
        return centroids
    
    def _create_fallback_centroid(self, 
                                speaker_id: str, 
                                speaker_embeddings: List[SpeakerEmbedding], 
                                content_id: str) -> Optional[SpeakerCentroid]:
        """
        Create a fallback centroid for speakers without high-quality embeddings.
        Uses ALL available embeddings regardless of quality.
        """
        if not speaker_embeddings:
            logger.warning(f"[{content_id}] No embeddings available for fallback centroid for {speaker_id}")
            return None
        
        # Use ALL embeddings, regardless of quality thresholds
        embedding_vectors = np.array([emb.embedding for emb in speaker_embeddings])
        durations = np.array([emb.duration for emb in speaker_embeddings])
        confidences = np.array([emb.confidence for emb in speaker_embeddings])
        
        # Simple average of all embeddings
        mean_embedding = np.mean(embedding_vectors, axis=0)
        centroid_vector = mean_embedding / np.linalg.norm(mean_embedding)
        
        # Calculate basic statistics
        mean_confidence = np.mean(confidences)
        std_deviation = np.std(confidences) if len(confidences) > 1 else 0.0
        total_duration = np.sum(durations)
        
        # Quality score is low since this is a fallback
        quality_score = min(0.3, mean_confidence)  # Cap at 0.3 to indicate low quality
        
        # SpeakerCentroid is already defined in this file
        centroid = SpeakerCentroid(
            speaker_id=speaker_id,
            centroid=centroid_vector,
            embeddings_count=len(speaker_embeddings),
            mean_confidence=mean_confidence,
            std_deviation=std_deviation,
            total_duration=total_duration,
            quality_score=quality_score,  # Marked as low quality
            quality_source='FALLBACK'
        )
        
        logger.info(f"[{content_id}] Fallback centroid for {speaker_id}: "
                   f"{len(speaker_embeddings)} embeddings, "
                   f"total_duration={total_duration:.1f}s, "
                   f"mean_conf={mean_confidence:.3f}, "
                   f"quality={quality_score:.3f} (fallback)")
        
        return centroid
    
    
    def _calculate_embedding_statistics(self, 
                                      embeddings: List[SpeakerEmbedding], 
                                      centroids: Dict[str, SpeakerCentroid],
                                      content_id: str) -> Dict[str, Any]:
        """
        Calculate statistics about embeddings and centroids without reassigning speakers.
        """
        logger.info(f"[{content_id}] Calculating embedding statistics")
        
        # Calculate similarity statistics
        similarity_stats = []
        for embedding in embeddings:
            speaker = embedding.diarization_speaker
            if speaker in centroids:
                centroid = centroids[speaker]
                similarity = cosine_similarity(
                    embedding.embedding.reshape(1, -1),
                    centroid.centroid.reshape(1, -1)
                )[0, 0]
                similarity_stats.append({
                    'speaker': speaker,
                    'similarity': similarity,
                    'duration': embedding.duration,
                    'confidence': embedding.confidence
                })
        
        # Aggregate statistics
        if similarity_stats:
            avg_similarity = np.mean([s['similarity'] for s in similarity_stats])
            min_similarity = np.min([s['similarity'] for s in similarity_stats])
            max_similarity = np.max([s['similarity'] for s in similarity_stats])
        else:
            avg_similarity = min_similarity = max_similarity = 0.0
        
        # Speaker distribution
        speaker_counts = Counter(emb.diarization_speaker for emb in embeddings)
        
        return {
            'embeddings_processed': len(embeddings),
            'centroids_built': len(centroids),
            'avg_similarity': avg_similarity,
            'min_similarity': min_similarity,
            'max_similarity': max_similarity,
            'speaker_distribution': dict(speaker_counts),
            'total_duration': sum(emb.duration for emb in embeddings),
            'assignment_rate': 100.0  # Always 100% since we trust diarization
        }
    
    def _find_tight_cluster_centroid(self, 
                                   speaker_id: str,
                                   speaker_embeddings: List[SpeakerEmbedding],
                                   content_id: str) -> Optional[Dict]:
        """
        Find the tightest cluster of embeddings and build a centroid from it.
        Uses DBSCAN clustering to identify dense regions of similar embeddings.
        """
        # Extract embedding vectors and metadata
        embedding_vectors = np.array([emb.embedding for emb in speaker_embeddings])
        durations = np.array([emb.duration for emb in speaker_embeddings])
        confidences = np.array([emb.confidence for emb in speaker_embeddings])
        
        # Try Agglomerative Clustering first (often works better for speaker embeddings)
        # Use cosine similarity and aim for clusters that contain ~30-50% of embeddings
        n_embeddings = len(speaker_embeddings)
        
        if n_embeddings >= 10:
            # For speakers with many embeddings, try to find natural clusters
            # Start with n_clusters that would give us ~30-40% per cluster on average
            n_clusters_to_try = max(2, min(5, n_embeddings // 15))
            
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters_to_try,
                metric='cosine',
                linkage='average'
            )
            labels = clustering.fit_predict(embedding_vectors)
            
            # Find the largest cluster
            unique_labels, counts = np.unique(labels, return_counts=True)
            largest_cluster_idx = unique_labels[np.argmax(counts)]
            largest_cluster_size = np.max(counts)
            
            logger.info(f"[{content_id}] Agglomerative clustering for {speaker_id}: "
                       f"{n_clusters_to_try} clusters, largest has {largest_cluster_size}/{n_embeddings} embeddings")
            
            # If the largest cluster has reasonable size (20-70% of embeddings), use it
            if 0.2 <= largest_cluster_size / n_embeddings <= 0.7:
                unique_labels = set([largest_cluster_idx])
            else:
                # Fall back to DBSCAN
                unique_labels = set()
        else:
            # For speakers with few embeddings, skip clustering
            unique_labels = set()
        
        # If agglomerative didn't work well, try DBSCAN
        if not unique_labels:
            # Perform DBSCAN clustering
            # eps=0.5 means embeddings must be at least 50% similar (1-0.5=0.5 cosine similarity)
            # min_samples=max(2, len/20) means we need at least 2 samples or 5% of embeddings to form a cluster
            min_samples_adaptive = max(2, len(speaker_embeddings) // 20)
            clustering = DBSCAN(eps=0.5, min_samples=min_samples_adaptive, metric='cosine')
            labels = clustering.fit_predict(embedding_vectors)
            
            # Log clustering results
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            logger.info(f"[{content_id}] DBSCAN for {speaker_id}: {n_clusters} clusters found, "
                       f"{n_noise}/{len(labels)} noise points, min_samples={min_samples_adaptive}, eps=0.5")
            
            # Find valid clusters (excluding noise label -1)
            unique_labels = set(labels)
            unique_labels.discard(-1)
        
        if not unique_labels:
            # No tight clusters found, fall back to using high-quality turn-based embeddings
            logger.info(f"[{content_id}] No tight clusters found for {speaker_id}, using high-quality turn selection")
            
            # Prioritize embeddings by quality for turn-based processing
            # 1. High-quality: duration >= 2.0s AND confidence >= 0.7
            # 2. Medium-quality: duration >= 1.5s AND confidence >= 0.6
            # 3. Backup: duration >= 0.5s AND confidence >= 0.5
            
            high_quality_mask = (durations >= 2.0) & (confidences >= 0.7)
            medium_quality_mask = (durations >= 1.5) & (confidences >= 0.6)
            backup_quality_mask = (durations >= 0.5) & (confidences >= 0.5)
            
            high_quality_count = np.sum(high_quality_mask)
            medium_quality_count = np.sum(medium_quality_mask)
            backup_quality_count = np.sum(backup_quality_mask)
            total_count = len(speaker_embeddings)
            
            logger.info(f"[{content_id}] Quality breakdown for {speaker_id}: "
                       f"high={high_quality_count}, medium={medium_quality_count}, backup={backup_quality_count}, total={total_count}")
            
            # Select embeddings based on available quality
            if high_quality_count >= self.min_centroid_embeddings:
                # Use high-quality embeddings
                selected_embeddings = [emb for emb, hq in zip(speaker_embeddings, high_quality_mask) if hq]
                selected_vectors = embedding_vectors[high_quality_mask]
                cluster_source = f"high_quality_turns_{high_quality_count}"
                logger.info(f"[{content_id}] Using {high_quality_count} high-quality turn embeddings for {speaker_id}")
                
            elif medium_quality_count >= self.min_centroid_embeddings:
                # Use medium-quality embeddings
                selected_embeddings = [emb for emb, mq in zip(speaker_embeddings, medium_quality_mask) if mq]
                selected_vectors = embedding_vectors[medium_quality_mask]
                cluster_source = f"medium_quality_turns_{medium_quality_count}"
                logger.info(f"[{content_id}] Using {medium_quality_count} medium-quality turn embeddings for {speaker_id}")
                
            elif backup_quality_count >= self.min_centroid_embeddings:
                # Use backup quality embeddings
                selected_embeddings = [emb for emb, bq in zip(speaker_embeddings, backup_quality_mask) if bq]
                selected_vectors = embedding_vectors[backup_quality_mask]
                cluster_source = f"backup_quality_turns_{backup_quality_count}"
                logger.info(f"[{content_id}] Using {backup_quality_count} backup-quality turn embeddings for {speaker_id}")
                
            else:
                # Use all embeddings as last resort
                selected_embeddings = speaker_embeddings
                selected_vectors = embedding_vectors
                cluster_source = f"all_turns_fallback_{total_count}"
                logger.warning(f"[{content_id}] Using all {total_count} turn embeddings for {speaker_id} (insufficient high-quality)")
        else:
            # Find the best cluster based on multiple criteria
            best_cluster_label = None
            best_cluster_score = -1
            cluster_info = []
            
            for label in unique_labels:
                mask = labels == label
                cluster_size = np.sum(mask)
                cluster_durations = durations[mask]
                cluster_confidences = confidences[mask]
                
                # Calculate cluster quality score
                # Consider: size, total duration, average confidence
                size_score = np.log1p(cluster_size)
                duration_score = np.log1p(np.sum(cluster_durations))
                confidence_score = np.mean(cluster_confidences)
                
                # Combined score emphasizing larger clusters with more speech
                cluster_score = size_score * duration_score * confidence_score
                
                cluster_info.append({
                    'label': label,
                    'size': cluster_size,
                    'total_duration': np.sum(cluster_durations),
                    'avg_confidence': confidence_score,
                    'score': cluster_score
                })
                
                if cluster_score > best_cluster_score:
                    best_cluster_score = cluster_score
                    best_cluster_label = label
            
            # Extract best cluster
            cluster_mask = labels == best_cluster_label
            selected_count = np.sum(cluster_mask)
            total_count = len(speaker_embeddings)
            
            # Check if the selected cluster is too small compared to total embeddings
            selection_percentage = selected_count / total_count
            
            # If best cluster is less than 10% of embeddings and we have many embeddings, 
            # consider combining multiple clusters or using more embeddings
            if total_count >= 20 and selection_percentage < 0.1:
                logger.info(f"[{content_id}] Best cluster for {speaker_id} only contains "
                           f"{selected_count}/{total_count} ({selection_percentage:.1%}) embeddings")
                
                # Try combining top clusters to get more coverage
                cluster_info.sort(key=lambda x: x['score'], reverse=True)
                combined_mask = np.zeros(len(speaker_embeddings), dtype=bool)
                
                for cluster in cluster_info:
                    cluster_mask = labels == cluster['label']
                    combined_mask |= cluster_mask
                    combined_count = np.sum(combined_mask)
                    
                    # Stop when we have enough embeddings (at least 10% or 50 embeddings)
                    if combined_count >= max(int(total_count * 0.1), min(50, total_count // 2)):
                        break
                
                selected_embeddings = [emb for emb, in_cluster in zip(speaker_embeddings, combined_mask) if in_cluster]
                selected_vectors = embedding_vectors[combined_mask]
                cluster_source = f"combined_clusters_{len([c for c in cluster_info if np.any(labels == c['label'])])}"
                
                logger.info(f"[{content_id}] Combined multiple clusters for {speaker_id}: "
                           f"{len(selected_embeddings)}/{len(speaker_embeddings)} embeddings")
            else:
                # Use the single best cluster
                selected_embeddings = [emb for emb, in_cluster in zip(speaker_embeddings, cluster_mask) if in_cluster]
                selected_vectors = embedding_vectors[cluster_mask]
                cluster_source = f"dbscan_cluster_{best_cluster_label}"
                
                logger.info(f"[{content_id}] Selected cluster {best_cluster_label} for {speaker_id}: "
                           f"{len(selected_embeddings)}/{len(speaker_embeddings)} embeddings")
        
        if len(selected_embeddings) < self.min_centroid_embeddings:
            return None
        
        # Calculate centroid from selected embeddings
        centroid_vector = np.mean(selected_vectors, axis=0)
        centroid_vector = centroid_vector / np.linalg.norm(centroid_vector)
        
        # Calculate statistics
        selected_durations = [emb.duration for emb in selected_embeddings]
        selected_confidences = [emb.confidence for emb in selected_embeddings]
        
        # Calculate tightness score (how similar embeddings are to centroid)
        similarities = cosine_similarity([centroid_vector], selected_vectors)[0]
        tightness_score = 1.0 / (1.0 + np.std(similarities))
        quality_score = np.mean(similarities) * tightness_score
        
        # Count outliers (embeddings not in any cluster)
        outliers_detected = np.sum(labels == -1)
        
        centroid = SpeakerCentroid(
            speaker_id=speaker_id,
            centroid=centroid_vector,
            embeddings_count=len(selected_embeddings),
            mean_confidence=np.mean(selected_confidences),
            std_deviation=np.std(similarities),
            total_duration=np.sum(selected_durations),
            quality_score=quality_score,
            quality_source='HIGH'
        )
        
        return {
            'centroid': centroid,
            'embeddings_used': len(selected_embeddings),
            'tightness_score': tightness_score,
            'cluster_source': cluster_source,
            'outliers_detected': outliers_detected
        }
    
    
    
    
    
    
    
    def _get_centroids_summary(self) -> Dict:
        """Get summary of speaker centroids."""
        if not self.speaker_centroids:
            return {'status': 'no_centroids'}
        
        summary = {
            'status': 'ready',
            'total_speakers': len(self.speaker_centroids),
            'speakers': {}
        }
        
        for speaker_id, centroid in self.speaker_centroids.items():
            summary['speakers'][speaker_id] = {
                'embeddings_count': centroid.embeddings_count,
                'mean_confidence': centroid.mean_confidence,
                'quality_score': centroid.quality_score,
                'total_duration': centroid.total_duration,
                'std_deviation': centroid.std_deviation
            }
        
        return summary
    
    def get_speaker_centroid_vectors(self) -> Dict[str, np.ndarray]:
        """Get speaker centroid vectors for use in other stages."""
        if not self.speaker_centroids:
            return {}
        
        centroid_vectors = {}
        for speaker_id, centroid in self.speaker_centroids.items():
            centroid_vectors[speaker_id] = centroid.centroid
        
        return centroid_vectors
    
    def _get_quality_summary(self) -> Dict[str, Any]:
        """Get comprehensive quality summary of all speaker centroids."""
        if not self.speaker_centroids:
            return {'total_speakers': 0}
        
        quality_summary = {
            'total_speakers': len(self.speaker_centroids),
            'quality_tiers': {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'FALLBACK': 0},
            'embedding_sources': {},
            'quality_statistics': {
                'mean_quality': 0.0,
                'mean_stability': 0.0,
                'mean_core_confidence': 0.0,
                'total_duration': 0.0,
                'total_embeddings': 0
            }
        }
        
        quality_scores = []
        stability_scores = []
        core_confidences = []
        
        for speaker_id, centroid in self.speaker_centroids.items():
            # Quality tier - use the stored quality_source instead of recalculating
            quality_source = getattr(centroid, 'quality_source', 'UNKNOWN')
            if quality_source in quality_summary['quality_tiers']:
                quality_summary['quality_tiers'][quality_source] += 1
            else:
                # Fallback to score-based assignment for backward compatibility
                quality_score = centroid.quality_score
                if quality_score >= 0.8:
                    quality_summary['quality_tiers']['HIGH'] += 1
                elif quality_score >= 0.5:
                    quality_summary['quality_tiers']['MEDIUM'] += 1
                elif quality_score >= 0.2:
                    quality_summary['quality_tiers']['LOW'] += 1
                else:
                    quality_summary['quality_tiers']['FALLBACK'] += 1
            
            # Embedding source
            source = getattr(centroid, 'embedding_source', 'unknown')
            quality_summary['embedding_sources'][source] = quality_summary['embedding_sources'].get(source, 0) + 1
            
            # Statistics
            quality_scores.append(centroid.quality_score)
            quality_summary['quality_statistics']['total_duration'] += centroid.total_duration
            quality_summary['quality_statistics']['total_embeddings'] += centroid.embeddings_count
            
            # Extract stability and core confidence from metadata
            metadata = getattr(centroid, 'quality_metadata', {})
            stability_scores.append(metadata.get('stability_score', 0.0))
            core_confidences.append(metadata.get('core_signature_confidence', 0.0))
        
        # Calculate averages
        if quality_scores:
            quality_summary['quality_statistics']['mean_quality'] = np.mean(quality_scores)
            quality_summary['quality_statistics']['mean_stability'] = np.mean(stability_scores)
            quality_summary['quality_statistics']['mean_core_confidence'] = np.mean(core_confidences)
        
        return quality_summary
    
    def _get_centroid_data_for_stage6b(self) -> Dict[str, Dict]:
        """Get centroid data formatted for Stage 6b database integration with comprehensive quality metadata."""
        if not self.speaker_centroids:
            return {}
        
        centroid_data = {}
        for speaker_id, centroid in self.speaker_centroids.items():
            # Get comprehensive quality metadata if available
            quality_metadata = getattr(centroid, 'quality_metadata', {})
            
            # Determine quality tier based on quality score
            quality_score = centroid.quality_score
            if quality_score >= 0.8:
                quality_tier = "HIGH"
            elif quality_score >= 0.5:
                quality_tier = "MEDIUM"
            elif quality_score >= 0.2:
                quality_tier = "LOW"
            else:
                quality_tier = "FALLBACK"
            
            centroid_data[speaker_id] = {
                'centroid': centroid.centroid,
                'embeddings_count': centroid.embeddings_count,
                'mean_confidence': centroid.mean_confidence,
                'total_duration': centroid.total_duration,
                'quality_score': centroid.quality_score,
                'std_deviation': centroid.std_deviation,
                'diarization_speaker': speaker_id,  # Local diarization speaker ID (SPEAKER_00, etc.)
                
                # Enhanced quality metadata for database storage
                'quality_metadata': {
                    'quality_tier': quality_tier,
                    'embedding_source': getattr(centroid, 'embedding_source', 'unknown'),
                    'core_signature_confidence': quality_metadata.get('core_signature_confidence', 0.0),
                    'stability_score': quality_metadata.get('stability_score', 0.0),
                    'duration_quality': quality_metadata.get('duration_quality', 0.0),
                    'confidence_quality': quality_metadata.get('confidence_quality', 0.0),
                    'algorithm_version': 'stage6_enhanced_quality_v1',
                    'quality_components': quality_metadata.get('quality_components', {}),
                    'source_weight': quality_metadata.get('source_weight', 1.0)
                }
            }
        
        return centroid_data


async def speaker_stage(content_id: str,
                       word_table: WordTable,
                       speaker_context_stats: Dict[str, int],
                       config: Dict,
                       audio_path: Optional[str] = None,
                       test_mode: bool = False,
                       shared_audio_data: Optional[Tuple] = None,
                       shared_embedding_model: Optional[torch.nn.Module] = None) -> Dict[str, Any]:
    """
    Main entry point for Stage 6: Speaker Assignment with Pyannote Embeddings.
    
    This is the primary method called by the stitch pipeline. It performs comprehensive
    speaker assignment using audio embeddings, building speaker centroids from diarization
    ground truth and assigning words based on acoustic similarity. This is the most
    computationally expensive stage but provides high-quality speaker assignments.
    
    Args:
        content_id: Content ID being processed (e.g., "Bdb001")
        word_table: WordTable from Stage 5 with categorized words and some assignments
        speaker_context_stats: Dictionary with speaker statistics (can be empty)
        config: Configuration dictionary with model settings and thresholds
        audio_path: Optional path to audio file (uses shared_audio_data if provided)
        test_mode: If True, saves detailed outputs and uses test audio paths
        shared_audio_data: Optional tuple of (audio_data, sample_rate) from pipeline
        shared_embedding_model: Optional pre-loaded embedding model from pipeline
        
    Returns:
        Dictionary containing:
            - status: 'success' or 'error'
            - data: Dict with updated word_table, assignment_result, assignment_summary
            - stats: Processing statistics and performance metrics
            - error: Error message if status is 'error'
            
    Example:
        result = await speaker_stage("Bdb001", word_table, {}, config, test_mode=True)
        if result['status'] == 'success':
            centroids = result['data']['assignment_result']['speaker_centroids']
    """
    start_time = time.time()
    
    logger.info(f"[{content_id}] Starting Stage 7: Comprehensive Speaker Assignment")
    
    result = {
        'status': 'pending',
        'content_id': content_id,
        'stage': 'speaker_assignment',
        'data': {
            'word_table': None,
            'assignment_result': None,
            'assignment_summary': None
        },
        'stats': {},
        'error': None
    }
    
    try:
        # Validate inputs
        if word_table is None:
            raise ValueError("No word table available")
        
        # Speaker context stats is optional now that we don't have stage 6 speaker context
        if speaker_context_stats is None:
            speaker_context_stats = {}
        
        # If speaker_context_stats is empty (from new pipeline), analyze word table directly
        if not speaker_context_stats:
            # Count speaker assignments in current word table
            speaker_counts = word_table.df['speaker_current'].value_counts()
            # Exclude category labels when counting actual speaker assignments
            excluded_speakers = {'UNKNOWN', 'MULTI_SPEAKER', 'BAD_GRAMMAR_SINGLE', 'BAD_GRAMMAR_MULTI', 
                               'GOOD_GRAMMAR_SINGLE', 'GOOD_GRAMMAR_MULTI'}
            single_speaker_count = sum(count for speaker, count in speaker_counts.items() 
                                      if speaker not in excluded_speakers)
            unknown_count = speaker_counts.get('UNKNOWN', 0)
            multi_speaker_count = speaker_counts.get('MULTI_SPEAKER', 0)
            
            speaker_context_stats = {
                'single_speaker': single_speaker_count,
                'multi_speaker': multi_speaker_count,
                'unknown': unknown_count
            }
            logger.info(f"[{content_id}] Computed speaker stats from word table")
        
        logger.info(f"[{content_id}] Speaker context stats: single={speaker_context_stats.get('single_speaker', 0)}, "
                   f"multi={speaker_context_stats.get('multi_speaker', 0)}, "
                   f"unknown={speaker_context_stats.get('unknown', 0)}")
        
        # Always proceed with comprehensive speaker processor - it handles its own internal logic
        logger.info(f"[{content_id}] Proceeding with comprehensive speaker assignment")
        
        # Initialize comprehensive speaker processor with shared resources
        speaker_processor = ComprehensiveSpeakerProcessor(
            config=config,
            shared_audio_data=shared_audio_data,
            shared_embedding_model=shared_embedding_model
        )
        
        # Determine audio path for embeddings
        processed_audio_path = None
        if audio_path and Path(audio_path).exists():
            processed_audio_path = Path(audio_path)
            logger.info(f"[{content_id}] Using audio file: {processed_audio_path}")
        elif test_mode:
            # In test mode, check for audio file in test directory
            test_audio_path = get_project_root() / "tests" / "content" / content_id / "inputs" / "audio.wav"
            if test_audio_path.exists():
                processed_audio_path = test_audio_path
                logger.info(f"[{content_id}] Found test audio file: {processed_audio_path}")
            else:
                raise ValueError(f"No audio file found for test mode. Expected at: {test_audio_path}")
        else:
            raise ValueError(f"No audio file available. Audio path provided: {audio_path}")
        
        # Perform comprehensive speaker processing
        # TODO: Integrate shared_audio_data and shared_embedding_model into processor
        # For now, keep existing implementation - full integration would require 
        # modifying ComprehensiveSpeakerProcessor to accept preloaded audio
        assignment_result = await speaker_processor.process_speakers(
            word_table, processed_audio_path, content_id
        )
        
        # Create assignment summary
        assignment_summary = {
            'status': assignment_result.get('status', 'unknown'),
            'processing_stages': assignment_result.get('stages', {}),
            'centroids_summary': assignment_result.get('centroids_summary', {}),
            'final_statistics': assignment_result.get('final_stats', {})
        }
        
        # Update word processing status to track stage completion
        words_updated = _update_word_processing_status(word_table, 'speaker')
        
        # Store results
        result['data']['word_table'] = word_table
        result['data']['assignment_result'] = assignment_result
        result['data']['assignment_summary'] = assignment_summary
        result['data']['speaker_centroids'] = assignment_result.get('speaker_centroids', {})
        result['data']['speaker_centroid_data'] = assignment_result.get('speaker_centroid_data', {})
        
        stage_duration = time.time() - start_time
        result['stats'] = {
            'duration': stage_duration,
            'assignment_result': assignment_result,
            'assignment_summary': assignment_summary,
            'audio_used': processed_audio_path is not None,
            'words_status_updated': words_updated
        }
        
        result['status'] = 'success'
        
        logger.info(f"[{content_id}] Stage 7 completed successfully in {stage_duration:.2f}s")
        logger.info(f"[{content_id}] Speaker assignment completed: {assignment_result.get('status', 'unknown')}")
        if assignment_result.get('status') == 'completed':
            final_stats = assignment_result.get('final_stats', {})
            logger.info(f"[{content_id}] - Assignment rate: {final_stats.get('assignment_rate', 0):.1f}%")
            logger.info(f"[{content_id}] - Speakers found: {final_stats.get('unique_speakers', 0)}")
            logger.info(f"[{content_id}] - Outliers detected: {final_stats.get('outlier_count', 0)}")
        
        return result
        
    except Exception as e:
        logger.error(f"[{content_id}] Stage 7 failed: {str(e)}")
        logger.error(f"[{content_id}] Error details:", exc_info=True)
        
        result.update({
            'status': 'error',
            'error': str(e),
            'duration': time.time() - start_time
        })
        return result


def _update_word_processing_status(word_table: WordTable, stage_name: str, word_ids: List[str] = None) -> int:
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
        mask = slice(None)  # Select all rows
        updated_count = len(word_table.df)
        indices_to_update = word_table.df.index
    else:
        # Update specific words
        mask = word_table.df['word_id'].isin(word_ids)
        updated_count = mask.sum()
        indices_to_update = word_table.df[mask].index
    
    if updated_count > 0:
        word_table.df.loc[mask, 'processing_status'] = f'processed_stage_{stage_name}'
        
        # Update metadata with stage progression
        for idx in indices_to_update:
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