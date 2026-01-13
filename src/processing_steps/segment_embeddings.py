#!/usr/bin/env python3
"""
Embedding Segmentation and Generation
=====================================

Final pipeline step that:
1. Loads SpeakerTranscription records from database
2. Segments them into retrieval-optimized chunks using XLM-R similarity model
3. Generates embeddings for each segment using E5 model
4. Saves segments with embeddings to database

This is the final step in the processing pipeline.
"""

import sys
from pathlib import Path

from src.utils.paths import get_project_root
from src.utils.config import load_config
import asyncio
import logging
import json
import yaml
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
import numpy as np
import os
import aiohttp
import time

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.append(str(get_project_root()))

from src.utils.logger import setup_worker_logger
from src.database.session import get_session
from src.database.models import Content, SpeakerTranscription, EmbeddingSegment, Speaker
from src.utils.gpu_embedding_utils import GPUEmbeddingGenerator, DualModelEmbeddingSegmenter
from src.storage.s3_utils import S3Storage, S3StorageConfig, create_s3_storage_from_config
from src.processing_steps.stitch_steps.stage1_load import _load_chunks_from_local_or_s3
from difflib import SequenceMatcher
import tempfile
from rapidfuzz import fuzz
import pandas as pd

# Try to import NLTK and download punkt if needed
try:
    import nltk
    nltk.data.find('tokenizers/punkt')
except LookupError:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

from nltk.tokenize import sent_tokenize

logger = setup_worker_logger('segment_embeddings')

class BeamState:
    """Represents a state in beam search for text segmentation."""
    
    def __init__(self):
        self.completed_segments = []  # List of completed segment dicts
        self.current_sentences = []    # Sentences in current segment
        self.current_tokens = 0        # Token count of current segment
        self.current_source_ids = []   # Source transcription IDs
        self.current_speaker_ids = set()  # Speakers in current segment
        self.total_score = 0.0         # Total score (lower is better)
        self.segment_boundaries = []   # Indices where segments end
        
    def copy(self):
        """Create a deep copy for branching."""
        new_state = BeamState()
        new_state.completed_segments = self.completed_segments.copy()
        new_state.current_sentences = self.current_sentences.copy()
        new_state.current_tokens = self.current_tokens
        new_state.current_source_ids = self.current_source_ids.copy()
        new_state.current_speaker_ids = self.current_speaker_ids.copy()
        new_state.total_score = self.total_score
        new_state.segment_boundaries = self.segment_boundaries.copy()
        return new_state
    
    def __repr__(self):
        return (f"BeamState(segments={len(self.completed_segments)}, "
                f"current_tokens={self.current_tokens}, score={self.total_score:.2f})")

class EmbeddingSegmenter:
    """Segments speaker transcriptions and generates embeddings using beam search."""
    
    def __init__(self):
        """Initialize the segmenter with dual models."""
        # Load config
        self.config = load_config()
        
        # Model server configuration
        self.model_server_enabled = self.config.get('processing', {}).get('model_server', {}).get('enabled', False)
        self.model_server_url = self.config.get('processing', {}).get('model_server', {}).get('url', 'http://localhost:8002')
        self.embed_text_endpoint = f"{self.model_server_url}/embed_text"
        
        # Segmentation parameters
        seg_config = self.config.get('embedding', {}).get('embedding_segmentation', {})
        self.min_tokens = seg_config.get('min_tokens', 50)
        self.target_tokens = seg_config.get('target_tokens', 250)
        self.max_tokens = seg_config.get('max_tokens', 400)
        self.coherence_threshold = seg_config.get('coherence_threshold', 0.7)
        self.combine_speakers = seg_config.get('combine_speakers', True)
        self.max_combine_gap = seg_config.get('max_combine_gap', 3.0)
        
        # Embedding model configuration - Qwen3-0.6B as default, optional Qwen3-4B for alt
        self.embedding_model = seg_config.get('embedding_model', 'Qwen/Qwen3-Embedding-0.6B')
        self.use_alternative_embedding = seg_config.get('use_alternative_embedding', False)
        self.alternative_model = seg_config.get('alternative_model', 'Qwen/Qwen3-Embedding-4B')
        self.alternative_model_dim = seg_config.get('alternative_model_dim', 2000)
        self.save_to_alt_column = seg_config.get('save_to_alt_column', True)
        
        # Beam search parameters
        self.beam_width = seg_config.get('beam_width', 5)
        self.lookahead_sentences = seg_config.get('lookahead_sentences', 3)
        
        # Initialize embedding models - now using Qwen3-0.6B as default
        logger.info(f"Using Qwen3-0.6B embedding model: {self.embedding_model}")
        self._init_primary_embedding_model()
        
        # Initialize alternative embedding model lazily - only when needed
        self.alt_embed_generator = None
        self.alt_qwen_model = None
        if self.alternative_model:
            logger.info(f"Alternative embedding model available for lazy loading: {self.alternative_model}")
        else:
            logger.info("No alternative embedding model configured")
        
        # Still use XLM-R for similarity during segmentation
        self._init_similarity_model()
        
        # Add cache for pre-computed embeddings (using similarity model)
        self.sentence_embeddings = {}  # Cache for sentence embeddings
        self.sentence_embeddings_array = None  # Numpy array of all embeddings (normalized)
        self.sentence_texts = []  # List of sentences in order
        self.sentence_embeddings_normalized = None  # Normalized embeddings for fast composition
        
        logger.info(f"Initialized with params: min={self.min_tokens}, target={self.target_tokens}, "
                   f"max={self.max_tokens}, beam_width={self.beam_width}")
        
        # Log the model configuration
        qwen_device = getattr(self.embed_generator, 'device', 'unknown')
        xlmr_device = getattr(self.similarity_generator, 'device', 'unknown') if self.similarity_generator else 'none'
        logger.info(f"Using Qwen3-0.6B on {qwen_device} + XLM-R on {xlmr_device} GPU-accelerated approach")
        
        # Initialize similarity calculation tracking
        self._direct_similarity_count = 0
        self._similarity_cache = {}
        
        # Initialize Whisper chunk cache for precise timing
        self._whisper_chunks_cache = {}
        
        # Initialize S3 storage for precise timing
        s3_config = S3StorageConfig(
            endpoint_url=self.config['storage']['s3']['endpoint_url'],
            access_key=self.config['storage']['s3']['access_key'],
            secret_key=self.config['storage']['s3']['secret_key'],
            bucket_name=self.config['storage']['s3']['bucket_name'],
            use_ssl=self.config['storage']['s3']['use_ssl']
        )
        self.s3_storage_for_timing = S3Storage(s3_config)
    
    def _init_primary_embedding_model(self):
        """Initialize primary Qwen3 embedding model (0.6B) for embeddings."""
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            
            logger.info(f"Loading primary model {self.embedding_model}...")
            
            # Always use Qwen3-0.6B as primary model (1024 dimensions)
            self.qwen_model = SentenceTransformer(self.embedding_model, trust_remote_code=True)
            
            # Move to MPS if available
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            self.qwen_model = self.qwen_model.to(device)
            
            # Test to get embedding dimension
            test_embedding = self.qwen_model.encode(["test"], convert_to_numpy=True)
            self.embedding_dimension = test_embedding.shape[1]
            
            logger.info(f"Primary Qwen model loaded on {device}, embedding dimension: {self.embedding_dimension}")
            
            # Create wrapper for compatibility with existing code
            self.embed_generator = self
            self.device = device
            
        except Exception as e:
            logger.error(f"Failed to initialize primary Qwen model: {e}")
            raise
    
    def _init_alternative_embedding_model(self):
        """Initialize alternative Qwen3 embedding model (4B) for embeddings - called lazily when needed."""
        if self.alt_embed_generator is not None:
            return  # Already initialized
        
        if not self.alternative_model:
            logger.warning("No alternative model configured, cannot initialize")
            return
            
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            
            logger.info(f"Lazy loading alternative model {self.alternative_model}...")
            
            # Configure truncate_dim for Qwen3-4B model to get configured dimensions
            if 'Qwen3-Embedding-4B' in self.alternative_model:
                logger.info(f"Using Qwen3-4B model with truncate_dim={self.alternative_model_dim} for {self.alternative_model_dim}-dimensional embeddings")
                self.alt_qwen_model = SentenceTransformer(self.alternative_model, trust_remote_code=True, truncate_dim=self.alternative_model_dim)
            else:
                # Other alternative models - use configured dimension if specified
                if hasattr(self, 'alternative_model_dim') and self.alternative_model_dim:
                    self.alt_qwen_model = SentenceTransformer(self.alternative_model, trust_remote_code=True, truncate_dim=self.alternative_model_dim)
                else:
                    self.alt_qwen_model = SentenceTransformer(self.alternative_model, trust_remote_code=True)
            
            # Move to MPS if available
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            self.alt_qwen_model = self.alt_qwen_model.to(device)
            
            # Test to get embedding dimension
            test_embedding = self.alt_qwen_model.encode(["test"], convert_to_numpy=True)
            self.alt_embedding_dimension = test_embedding.shape[1]
            
            logger.info(f"Alternative Qwen model loaded on {device}, embedding dimension: {self.alt_embedding_dimension}")
            
            # Create wrapper class for alternative embeddings
            class AltEmbedGenerator:
                def __init__(self, model):
                    self.model = model
                    self.embedding_dimension = self.model.get_sentence_embedding_dimension()
                    self.device = model.device
                
                def generate_embeddings(self, texts, show_progress=False):
                    return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=show_progress)
            
            self.alt_embed_generator = AltEmbedGenerator(self.alt_qwen_model)
            logger.info(f"Alternative embedding model ready for use")
            
        except Exception as e:
            logger.error(f"Failed to initialize alternative Qwen model: {e}")
            # Don't raise - alternative embeddings are optional
            self.alt_embed_generator = None
            self.alt_qwen_model = None
            logger.warning(f"Alternative embeddings disabled due to initialization error")
    
    def _init_similarity_model(self):
        """Initialize XLM-R model for similarity calculations during segmentation."""
        try:
            from src.utils.gpu_embedding_utils import GPUEmbeddingGenerator
            
            # Use XLM-R for similarity calculations
            similarity_model_name = "sentence-transformers/paraphrase-xlm-r-multilingual-v1"
            self.similarity_generator = GPUEmbeddingGenerator(model_name=similarity_model_name)
            
            logger.info(f"XLM-R model initialized for similarity (dimension: {self.similarity_generator.embedding_dimension})")
            
        except Exception as e:
            logger.warning(f"Failed to initialize similarity model, will use embeddings for similarity: {e}")
            self.similarity_generator = None
    
    def generate_embeddings(self, texts, show_progress=False):
        """Generate embeddings using primary Qwen3 model (0.6B)."""
        return self.qwen_model.encode(texts, convert_to_numpy=True, show_progress_bar=show_progress)
    
    def generate_alternative_embeddings(self, texts, show_progress=False):
        """Generate embeddings using alternative Qwen3 model (4B)."""
        # Lazy initialization of alternative model when first needed
        if self.alt_embed_generator is None:
            logger.info("Alternative embeddings requested - initializing alternative model...")
            self._init_alternative_embedding_model()
        
        if self.alt_embed_generator is None:
            logger.warning("Alternative embedding generator not available after initialization attempt")
            return None
        
        return self.alt_embed_generator.generate_embeddings(texts, show_progress=show_progress)
    
    def _update_alternative_embeddings(self, content_id: str, existing_segments: List) -> Dict:
        """Update existing segments with alternative embeddings."""
        try:
            logger.info(f"[{content_id}] Updating alternative embeddings for existing segments")
            
            # Filter segments needing alternative embeddings
            segments_to_update = [seg for seg in existing_segments if seg.embedding_alt is None]
            
            if not segments_to_update:
                return {
                    'status': 'success',
                    'segment_count': len(existing_segments),
                    'note': 'All segments already have alternative embeddings'
                }
            
            # Extract texts for embedding
            texts = [seg.text for seg in segments_to_update]
            
            # Generate embeddings in batches
            batch_size = 50
            updated_count = 0
            
            with get_session() as session:
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i+batch_size]
                    batch_segments = segments_to_update[i:i+batch_size]
                    
                    # Generate alternative embeddings (will trigger lazy loading)
                    logger.info(f"[{content_id}] Generating alternative embeddings for batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
                    embeddings = self.generate_alternative_embeddings(batch_texts, show_progress=False)
                    
                    # Update segments
                    for seg, embedding in zip(batch_segments, embeddings):
                        db_seg = session.query(EmbeddingSegment).filter_by(id=seg.id).first()
                        if db_seg:
                            db_seg.embedding_alt = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
                            db_seg.embedding_alt_model = self.alternative_model
                            updated_count += 1
                    
                    session.commit()
                    logger.info(f"[{content_id}] Updated {updated_count}/{len(segments_to_update)} segments")
            
            return {
                'status': 'success',
                'segments_updated': updated_count,
                'total_segments': len(existing_segments),
                'model': self.alternative_model,
                'note': 'Added alternative embeddings to existing segments'
            }
            
        except Exception as e:
            logger.error(f"[{content_id}] Error updating alternative embeddings: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

    async def _get_embeddings_from_model_server(self, texts: List[str]) -> Optional[List[List[float]]]:
        """Try to get text embeddings from model server"""
        try:
            async with aiohttp.ClientSession() as session:
                data = {"texts": texts}
                
                logger.debug(f"Trying model server for {len(texts)} text embeddings: {self.embed_text_endpoint}")
                start_time = time.time()
                
                async with session.post(
                    self.embed_text_endpoint,
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=60)  # 1 minute timeout for text embeddings
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        elapsed = time.time() - start_time
                        logger.debug(f"Model server text embeddings completed in {elapsed:.2f}s")
                        return result.get('embeddings', [])
                    else:
                        error_text = await response.text()
                        raise Exception(f"Model server error ({response.status}): {error_text}")
                        
        except Exception as e:
            logger.warning(f"Model server text embeddings failed: {str(e)}")
            return None

    def _generate_similarity_embeddings(self, texts: List[str]) -> Optional[np.ndarray]:
        """Generate similarity embeddings using XLM-R model or Qwen3 fallback"""
        if self.similarity_generator:
            logger.debug(f"Using XLM-R similarity model for {len(texts)} texts")
            return self.similarity_generator.generate_embeddings(texts)
        else:
            logger.debug(f"Using Qwen3 model fallback for similarity of {len(texts)} texts")
            return self.generate_embeddings(texts)
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using Qwen3 tokenizer."""
        if not text:
            return 0
        
        # Use Qwen3 tokenizer
        tokenizer = self.qwen_model.tokenizer
        tokens = tokenizer(
            text, 
            add_special_tokens=False, 
            return_tensors=None,
            truncation=True,
            max_length=512
        )
        return len(tokens['input_ids'])
    
    def _precompute_sentence_embeddings(self, all_sentences: List[str]) -> None:
        """Pre-compute similarity embeddings for all sentences to be used in beam search."""
        logger.info(f"Pre-computing similarity embeddings for {len(all_sentences)} sentences")
        
        # Generate similarity embeddings for all sentences in one batch
        embeddings = self._generate_similarity_embeddings(all_sentences)
        
        if embeddings is None:
            raise RuntimeError("Failed to generate similarity embeddings for sentences")
        
        # Convert to numpy array for efficient operations
        self.sentence_embeddings_array = np.array(embeddings)
        self.sentence_texts = all_sentences
        
        # Normalize embeddings for efficient cosine similarity (if not already normalized)
        norms = np.linalg.norm(self.sentence_embeddings_array, axis=1, keepdims=True)
        # Only normalize if embeddings aren't already normalized
        if not np.allclose(norms, 1.0, atol=1e-6):
            self.sentence_embeddings_normalized = self.sentence_embeddings_array / norms
            logger.debug("Normalized similarity embeddings for cosine similarity")
        else:
            self.sentence_embeddings_normalized = self.sentence_embeddings_array.copy()
            logger.debug("Similarity embeddings already normalized")
        
        # Store the original (non-normalized) array for backward compatibility
        # But use normalized for hierarchical similarity calculations
        
        model_name = "XLM-R" if self.similarity_generator else "E5"
        logger.info(f"Successfully pre-computed {model_name} similarity embeddings for {len(all_sentences)} sentences")

    def _calculate_similarity_batch(self, indices1: List[int], indices2: List[int]) -> np.ndarray:
        """Calculate similarity between batches of sentences using pre-computed embeddings."""
        if self.sentence_embeddings_normalized is None:
            raise RuntimeError("Embeddings not pre-computed")
        
        # Get embeddings for both batches
        emb1 = self.sentence_embeddings_normalized[indices1]
        emb2 = self.sentence_embeddings_normalized[indices2]
        
        # Calculate cosine similarity using matrix multiplication
        # Since embeddings are normalized, dot product equals cosine similarity
        return np.dot(emb1, emb2.T)
    
    def _calculate_hierarchical_similarity(self, sentence_indices1: List[int], sentence_indices2: List[int]) -> float:
        """Calculate similarity between groups of sentences using hierarchical embedding composition.
        
        This is much faster than generating new embeddings for combined texts.
        Instead, we average the pre-computed embeddings of constituent sentences.
        """
        if self.sentence_embeddings_normalized is None:
            raise RuntimeError("Normalized embeddings not pre-computed")
        
        # Get embeddings for both groups and average them
        emb1 = self.sentence_embeddings_normalized[sentence_indices1].mean(axis=0)
        emb2 = self.sentence_embeddings_normalized[sentence_indices2].mean(axis=0)
        
        # Calculate cosine similarity (embeddings are already normalized, so just dot product)
        similarity = np.dot(emb1, emb2)
        return float(similarity)
    
    def _calculate_text_to_sentences_similarity(self, text: str, sentence_indices: List[int]) -> float:
        """Calculate similarity between arbitrary text and a group of sentences.
        
        Uses hierarchical composition for the sentence group, but generates embedding for the text.
        """
        if self.sentence_embeddings_normalized is None:
            raise RuntimeError("Normalized embeddings not pre-computed")
        
        # Generate embedding for the text (fallback to direct calculation)
        if self.similarity_generator:
            text_embeddings = self.similarity_generator.generate_similarity_embeddings([text])
        else:
            text_embeddings = self.embed_generator.generate_embeddings([text])
        
        if text_embeddings is None or len(text_embeddings) == 0:
            return 0.0
        
        # Normalize the text embedding
        text_emb = np.array(text_embeddings[0])
        text_emb = text_emb / np.linalg.norm(text_emb)
        
        # Average the sentence embeddings
        sentences_emb = self.sentence_embeddings_normalized[sentence_indices].mean(axis=0)
        
        # Calculate similarity
        similarity = np.dot(text_emb, sentences_emb)
        return float(similarity)
    
    def _get_sentence_indices_for_text(self, text: str) -> List[int]:
        """Get sentence indices that correspond to a given text.
        
        This handles the case where text is a combination of multiple sentences.
        """
        # Split the text into sentences
        sentences_in_text = sent_tokenize(text) if text else []
        indices = []
        
        for sentence in sentences_in_text:
            sentence = sentence.strip()
            if sentence in self.sentence_texts:
                indices.append(self.sentence_texts.index(sentence))
        
        return indices

    def _calculate_similarity_with_cache(self, text1: str, text2: str) -> float:
        """Calculate similarity with intelligent caching for beam search efficiency."""
        # Create a cache key for this pair
        cache_key = (text1, text2)
        reverse_cache_key = (text2, text1)  # Similarity is symmetric
        
        # Check if we've already computed this pair
        if not hasattr(self, '_similarity_cache'):
            self._similarity_cache = {}
        
        if cache_key in self._similarity_cache:
            return self._similarity_cache[cache_key]
        
        if reverse_cache_key in self._similarity_cache:
            return self._similarity_cache[reverse_cache_key]
        
        # Try the original method first (pre-computed embeddings)
        try:
            idx1 = self.sentence_texts.index(text1)
            idx2 = self.sentence_texts.index(text2)
            similarity = self._calculate_similarity_batch([idx1], [idx2])[0][0]
            similarity = float(similarity)
        except ValueError:
            # Direct calculation for new combinations
            if self.similarity_generator:
                similarity = self.similarity_generator.calculate_similarity(text1, text2)
            else:
                # E5 fallback
                embeddings = self.embed_generator.generate_embeddings([text1, text2])
                if embeddings is None or len(embeddings) < 2:
                    similarity = 0.0
                else:
                    emb1 = embeddings[0]
                    emb2 = embeddings[1]
                    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                    similarity = float(similarity)
        
        # Cache the result
        self._similarity_cache[cache_key] = similarity
        return similarity
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts using hierarchical embedding composition."""
        try:
            # Try to get sentence indices for both texts
            indices1 = self._get_sentence_indices_for_text(text1)
            indices2 = self._get_sentence_indices_for_text(text2)
            
            # Case 1: Both texts can be represented by sentence indices (most common in beam search)
            if indices1 and indices2:
                similarity = self._calculate_hierarchical_similarity(indices1, indices2)
                logger.debug(f"Used hierarchical similarity for {len(indices1)} vs {len(indices2)} sentences: {similarity:.3f}")
                return similarity
            
            # Case 2: One text is sentences, other is arbitrary text
            elif indices1 and not indices2:
                similarity = self._calculate_text_to_sentences_similarity(text2, indices1)
                logger.debug(f"Used text-to-sentences similarity: {similarity:.3f}")
                return similarity
            
            elif indices2 and not indices1:
                similarity = self._calculate_text_to_sentences_similarity(text1, indices2)
                logger.debug(f"Used text-to-sentences similarity: {similarity:.3f}")
                return similarity
            
            # Case 3: Neither text maps to sentences - fallback to direct calculation
            else:
                self._direct_similarity_count += 1
                logger.debug(f"Neither text maps to sentences, using direct similarity calculation")
                
                # Use appropriate model for similarity calculation
                if self.similarity_generator:
                    similarity = self.similarity_generator.calculate_similarity(text1, text2)
                    logger.debug(f"Used XLM-R direct similarity: {similarity:.3f}")
                    return similarity
                else:
                    # E5 fallback
                    embeddings = self.embed_generator.generate_embeddings([text1, text2])
                    if embeddings is None or len(embeddings) < 2:
                        return 0.0
                    
                    emb1 = embeddings[0]
                    emb2 = embeddings[1]
                    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                    logger.debug(f"Used E5 direct similarity: {similarity:.3f}")
                    return float(similarity)
                
        except Exception as e:
            logger.warning(f"Error calculating similarity: {e}")
            return 0.0
    
    def _segment_transcriptions(self, transcriptions: List[SpeakerTranscription], content_id: str = None) -> List[Dict]:
        """Segment speaker transcriptions into embedding-optimized chunks using beam search."""
        if not transcriptions:
            return []
        
        logger.info(f"Using beam search segmentation (beam_width={self.beam_width})")
        return self._segment_with_beam_search(transcriptions, content_id)
    
    
    def _segment_with_beam_search(self, transcriptions: List[SpeakerTranscription], content_id: str = None) -> List[Dict]:
        """Segment using efficient beam search with pre-computed sentence embeddings."""
        logger.info(f"Starting beam search segmentation with {len(transcriptions)} transcriptions")
        
        # Convert transcriptions to sentence-based representation
        all_sentences = []
        sentence_metadata = []
        
        for trans_idx, trans in enumerate(transcriptions):
            sentences = sent_tokenize(trans.text) if trans.text else []
            for sent_idx, sentence in enumerate(sentences):
                all_sentences.append(sentence)
                sentence_metadata.append({
                    'transcription_idx': trans_idx,
                    'transcription_id': trans.id,
                    'speaker_id': trans.speaker_id,
                    'sent_in_trans': sent_idx,
                    'trans_start': trans.start_time,
                    'trans_end': trans.end_time,
                    'sentence_index': len(all_sentences)  # Global sentence index
                })
        
        if not all_sentences:
            logger.warning("No sentences found in transcriptions")
            return []
        
        logger.info(f"Processing {len(all_sentences)} sentences with beam search (beam_width={self.beam_width})")
        
        # Pre-compute embeddings for all sentences
        self._precompute_sentence_embeddings(all_sentences)
        
        # Use a simpler approach: greedy segmentation with lookahead
        segments = []
        current_sentences = []
        current_tokens = 0
        current_source_ids = []
        current_speaker_ids = set()
        
        for sent_idx, sentence in enumerate(all_sentences):
            if sent_idx % 200 == 0:
                logger.info(f"Processing sentence {sent_idx}/{len(all_sentences)} ({sent_idx/len(all_sentences)*100:.1f}%)")
            
            # Add sentence to current segment
            current_sentences.append(sentence)
            additional_tokens = self._count_tokens(sentence)
            current_tokens += additional_tokens
            
            # Update metadata
            meta = sentence_metadata[sent_idx]
            current_source_ids.append(meta['transcription_id'])
            current_speaker_ids.add(meta['speaker_id'])
            
            # Check if we should end the segment
            should_segment = False
            
            # Reason 1: Token limit reached (use conservative limit)
            max_safe_tokens = min(self.max_tokens, 350)  # Conservative limit to prevent tokenizer issues
            if current_tokens >= max_safe_tokens:
                should_segment = True
                reason = f"token_limit_{current_tokens}"
            
            # Reason 2: Target size reached and low coherence with next few sentences
            elif current_tokens >= self.target_tokens:
                # Look ahead to see if similarity drops
                if self._should_segment_based_on_lookahead(current_sentences, sent_idx + 1, all_sentences, min(self.lookahead_sentences, 3)):
                    should_segment = True
                    reason = f"target_reached_lookahead_{current_tokens}"
            
            # Reason 3: Speaker change (if not combining speakers)
            elif not self.combine_speakers and sent_idx + 1 < len(all_sentences):
                next_speaker = sentence_metadata[sent_idx + 1]['speaker_id']
                if next_speaker not in current_speaker_ids:
                    should_segment = True
                    reason = f"speaker_change_{list(current_speaker_ids)}_{next_speaker}"
            
            # Create segment if needed
            if should_segment and len(current_sentences) > 0:
                # Get sentence indices for this segment
                sentence_indices = list(range(sent_idx - len(current_sentences) + 1, sent_idx + 1))
                
                # Create short segment type (max 20 chars for database constraint)
                short_reason = reason.split('_')[0][:6]  # First 6 chars of reason type
                segment = self._create_segment_from_sentences(
                    current_sentences, sentence_indices, sentence_metadata,
                    transcriptions, f"beam_{short_reason}", content_id  # e.g., "beam_token", "beam_target"
                )
                
                if segment:
                    segments.append(segment)
                
                # Reset for next segment
                current_sentences = []
                current_tokens = 0
                current_source_ids = []
                current_speaker_ids = set()
        
        # Handle remaining sentences
        if current_sentences:
            sentence_indices = list(range(len(all_sentences) - len(current_sentences), len(all_sentences)))
            segment = self._create_segment_from_sentences(
                current_sentences, sentence_indices, sentence_metadata,
                transcriptions, "beam_final", content_id
            )
            if segment:
                segments.append(segment)
        
        logger.info(f"Created {len(segments)} segments using beam search")
        return segments
    
    def _should_segment_based_on_lookahead(self, current_sentences: List[str], start_idx: int, 
                                         all_sentences: List[str], lookahead: int) -> bool:
        """Check if we should segment based on similarity with upcoming sentences."""
        if start_idx >= len(all_sentences):
            return True  # End of content
        
        # Get current segment text
        current_text = ' '.join(current_sentences)
        
        # Check similarity with next few sentences
        total_similarity = 0.0
        valid_comparisons = 0
        
        for i in range(lookahead):
            if start_idx + i >= len(all_sentences):
                break
                
            next_sentence = all_sentences[start_idx + i]
            similarity = self._calculate_fast_similarity(current_text, next_sentence, current_sentences, start_idx + i)
            total_similarity += similarity
            valid_comparisons += 1
        
        if valid_comparisons == 0:
            return True
        
        avg_similarity = total_similarity / valid_comparisons
        
        # Segment if average similarity is below threshold
        return avg_similarity < self.coherence_threshold
    
    def _calculate_fast_similarity(self, current_text: str, next_sentence: str, 
                                 current_sentences: List[str], next_sent_idx: int) -> float:
        """Fast similarity calculation using pre-computed embeddings."""
        try:
            # Get indices for current sentences
            current_indices = []
            for sent in current_sentences:
                if sent in self.sentence_texts:
                    current_indices.append(self.sentence_texts.index(sent))
            
            # Get index for next sentence
            if next_sentence in self.sentence_texts:
                next_idx = self.sentence_texts.index(next_sentence)
            else:
                return 0.5  # Default similarity if not found
            
            if not current_indices:
                return 0.5
            
            # Use hierarchical similarity (efficient with pre-computed embeddings)
            return self._calculate_hierarchical_similarity(current_indices, [next_idx])
            
        except Exception as e:
            logger.debug(f"Error in fast similarity calculation: {e}")
            return 0.5  # Default similarity on error
    
    def _can_add_to_current_segment(self, state: BeamState, sentence: str, sent_idx: int, sentence_metadata: List[Dict]) -> bool:
        """Check if sentence can be added to current segment."""
        # Can't add to empty segment
        if not state.current_sentences:
            return True
        
        # Check token limits
        additional_tokens = self._count_tokens(sentence)
        if state.current_tokens + additional_tokens > self.max_tokens:
            return False
        
        return True
    
    def _should_start_new_segment(self, state: BeamState, sentence: str, sent_idx: int, 
                                sentence_metadata: List[Dict], all_sentences: List[str]) -> bool:
        """Determine if we should start a new segment."""
        # Must have current content to start new segment
        if not state.current_sentences:
            return False
        
        # Check if current segment meets minimum requirements
        if state.current_tokens < self.min_tokens:
            return False
        
        # Calculate semantic coherence with current segment
        current_segment_text = ' '.join(state.current_sentences)
        similarity = self._calculate_segment_coherence(current_segment_text, sentence, state.current_sentences, sent_idx)
        
        # Start new segment if similarity is low
        if similarity < self.coherence_threshold:
            return True
        
        # Start new segment if we've reached target size and similarity is reasonable
        if state.current_tokens >= self.target_tokens and similarity < self.coherence_threshold * 1.2:
            return True
        
        # Speaker change consideration (if enabled)
        if not self.combine_speakers:
            current_speaker = sentence_metadata[sent_idx]['speaker_id']
            if current_speaker not in state.current_speaker_ids:
                return True
        
        return False
    
    def _calculate_segment_coherence(self, current_segment_text: str, new_sentence: str, 
                                   current_sentences: List[str], new_sent_idx: int) -> float:
        """Calculate coherence between current segment and new sentence using pre-computed embeddings."""
        try:
            # Get indices for current sentences
            current_indices = []
            for sent in current_sentences:
                if sent in self.sentence_texts:
                    current_indices.append(self.sentence_texts.index(sent))
            
            # Get index for new sentence
            if new_sentence in self.sentence_texts:
                new_idx = self.sentence_texts.index(new_sentence)
            else:
                # Fallback to direct similarity calculation
                return self._calculate_similarity(current_segment_text, new_sentence)
            
            if not current_indices:
                return 0.0
            
            # Use hierarchical similarity (average current embeddings vs new embedding)
            return self._calculate_hierarchical_similarity(current_indices, [new_idx])
            
        except Exception as e:
            logger.warning(f"Error calculating segment coherence: {e}")
            return 0.0
    
    def _add_sentence_to_current_segment(self, state: BeamState, sentence: str, sent_idx: int, 
                                       sentence_metadata: List[Dict], all_sentences: List[str]):
        """Add sentence to current segment in state."""
        state.current_sentences.append(sentence)
        state.current_tokens += self._count_tokens(sentence)
        
        # Update metadata
        meta = sentence_metadata[sent_idx]
        state.current_source_ids.append(meta['transcription_id'])
        state.current_speaker_ids.add(meta['speaker_id'])
        
        # Update score based on coherence
        if len(state.current_sentences) > 1:
            # Calculate penalty for adding this sentence (lower similarity = higher penalty)
            prev_sentences = state.current_sentences[:-1]
            prev_segment_text = ' '.join(prev_sentences)
            coherence = self._calculate_segment_coherence(prev_segment_text, sentence, prev_sentences, sent_idx)
            penalty = (1.0 - coherence) * 0.1  # Small penalty for low coherence
            state.total_score += penalty
    
    def _start_new_segment(self, state: BeamState, sentence: str, sent_idx: int, sentence_metadata: List[Dict]):
        """Start a new segment in state."""
        state.current_sentences = [sentence]
        state.current_tokens = self._count_tokens(sentence)
        
        # Update metadata
        meta = sentence_metadata[sent_idx]
        state.current_source_ids = [meta['transcription_id']]
        state.current_speaker_ids = {meta['speaker_id']}
        
        # No penalty for starting new segment
    
    def _finalize_current_segment(self, state: BeamState, sentence_metadata: List[Dict], transcriptions: List[SpeakerTranscription]):
        """Finalize current segment and add to completed segments."""
        if not state.current_sentences:
            return
        
        # Find sentence indices for timing calculation
        sentence_indices = []
        for sent in state.current_sentences:
            for i, meta in enumerate(sentence_metadata):
                if i < len(self.sentence_texts) and self.sentence_texts[i] == sent:
                    sentence_indices.append(i)
                    break
        
        if sentence_indices:
            # Create segment using existing method
            segment = self._create_segment_from_sentences(
                state.current_sentences, sentence_indices, sentence_metadata, 
                transcriptions, f"beam_search_seg_{len(state.completed_segments)}"
            )
            
            if segment:
                state.completed_segments.append(segment)
        
        # Reset current segment
        state.current_sentences = []
        state.current_tokens = 0
        state.current_source_ids = []
        state.current_speaker_ids = set()
    
    def _select_best_beam_candidates(self, candidates: List[BeamState]) -> List[BeamState]:
        """Select best candidates for beam search continuation."""
        if not candidates:
            return []
        
        # Sort by total score (lower is better)
        candidates.sort(key=lambda x: x.total_score)
        
        # Return top beam_width candidates
        return candidates[:self.beam_width]
    
    
    def _create_segment_from_sentences(self, sentences: List[str], sentence_indices: List[int],
                                     sentence_metadata: List[Dict], 
                                     transcriptions: List[SpeakerTranscription],
                                     reason: str, content_id: str = None) -> Dict:
        """Create a segment dictionary from sentences with precise timing when possible."""
        if not sentences or not sentence_indices:
            return {}
        
        segment_text = ' '.join(sentences)
        token_count = self._count_tokens(segment_text)
        
        # Try precise timing first if content_id is available
        start_time = None
        end_time = None
        timing_method = 'linear_interpolation'  # Default fallback
        
        if content_id:
            # Load Whisper chunks for precise timing
            whisper_segments = self._load_whisper_chunks_for_content(content_id)
            
            if whisper_segments:
                # Try to find precise timing for the entire segment text
                precise_start, precise_end = self._find_precise_word_timing(whisper_segments, segment_text)
                
                if precise_start is not None and precise_end is not None:
                    start_time = precise_start
                    end_time = precise_end
                    timing_method = 'whisper_word_precise'
                    logger.debug(f"[{content_id}] Using precise word-level timing for segment: {start_time:.2f}s - {end_time:.2f}s")
                else:
                    # Try sentence-by-sentence precise timing
                    first_sentence = sentences[0] if sentences else ""
                    last_sentence = sentences[-1] if sentences else ""
                    
                    first_precise, _ = self._find_precise_word_timing(whisper_segments, first_sentence)
                    _, last_precise = self._find_precise_word_timing(whisper_segments, last_sentence)
                    
                    if first_precise is not None and last_precise is not None:
                        start_time = first_precise
                        end_time = last_precise
                        timing_method = 'whisper_sentence_precise'
                        logger.debug(f"[{content_id}] Using precise sentence-level timing for segment: {start_time:.2f}s - {end_time:.2f}s")
        
        # Fallback to linear interpolation if precise timing failed
        if start_time is None or end_time is None:
            # Get timing from first and last sentences
            first_meta = sentence_metadata[sentence_indices[0]]
            last_meta = sentence_metadata[sentence_indices[-1]]
            
            first_trans = transcriptions[first_meta['transcription_idx']]
            last_trans = transcriptions[last_meta['transcription_idx']]
            
            # Use new method with precise timing capability
            if content_id:
                start_time = self._estimate_sentence_time_with_precision(
                    content_id, first_trans, first_meta['sent_in_trans'], 
                    len(sent_tokenize(first_trans.text)), use_start=True
                )
                
                end_time = self._estimate_sentence_time_with_precision(
                    content_id, last_trans, last_meta['sent_in_trans'],
                    len(sent_tokenize(last_trans.text)), use_start=False
                )
                timing_method = 'mixed_timing'  # Mix of precise and interpolation
            else:
                # Legacy fallback
                start_time = self._estimate_sentence_time(
                    first_trans, first_meta['sent_in_trans'], 
                    len(sent_tokenize(first_trans.text)), use_start=True
                )
                
                end_time = self._estimate_sentence_time(
                    last_trans, last_meta['sent_in_trans'],
                    len(sent_tokenize(last_trans.text)), use_start=False
                )
                timing_method = 'linear_interpolation'
        
        # Collect metadata
        source_ids = list(set(meta['transcription_id'] for meta in 
                            [sentence_metadata[i] for i in sentence_indices]))
        speaker_ids = list(set(meta['speaker_id'] for meta in 
                             [sentence_metadata[i] for i in sentence_indices]))
        
        return {
            'text': segment_text,
            'start_time': start_time,
            'end_time': end_time,
            'token_count': token_count,
            'segment_type': reason,
            'source_ids': source_ids,
            'speaker_ids': speaker_ids,
            'sentence_count': len(sentences),
            'segmentation_reason': reason,
            'timing_method': timing_method
        }
    
    def _load_whisper_chunks_for_content(self, content_id: str) -> List[Dict]:
        """Load original Whisper chunks for precise timing lookup."""
        if content_id in self._whisper_chunks_cache:
            return self._whisper_chunks_cache[content_id]
        
        logger.info(f"[{content_id}] Loading Whisper chunks for precise timing")
        
        try:
            # Create temp directory
            with tempfile.TemporaryDirectory(prefix=f"timing_{content_id}_") as temp_dir:
                temp_path = Path(temp_dir)
                
                # Load chunks using the same logic as stitch stage1
                transcript_segments = _load_chunks_from_local_or_s3(
                    s3_storage=self.s3_storage_for_timing,
                    content_id=content_id,
                    temp_dir=temp_path,
                    test_mode=False,
                    prefer_local=True
                )
                
                self._whisper_chunks_cache[content_id] = transcript_segments
                logger.info(f"[{content_id}] Loaded {len(transcript_segments)} Whisper segments for precise timing")
                return transcript_segments
                
        except Exception as e:
            logger.warning(f"[{content_id}] Could not load Whisper chunks for precise timing: {e}")
            return []
    
    def _find_precise_word_timing(self, whisper_segments: List[Dict], target_text: str) -> Tuple[Optional[float], Optional[float]]:
        """Find precise start and end timing using vectorized word matching."""
        
        if not whisper_segments or not target_text.strip():
            return None, None
        
        # Extract all words with timestamps from all Whisper segments - optimized
        all_words = []
        punctuation_table = str.maketrans('', '', '.,!?":;()[]{}')
        
        for whisper_seg in whisper_segments:
            words = whisper_seg.get('words', [])
            for word_data in words:
                word_text = word_data.get('word', '')
                start_time = word_data.get('start')
                end_time = word_data.get('end')
                
                if word_text and start_time is not None and end_time is not None:
                    # Fast text cleaning using translate
                    clean_text = word_text.lower().strip().translate(punctuation_table).strip()
                    if clean_text:
                        all_words.append({
                            'text': clean_text,
                            'start': start_time,
                            'end': end_time,
                            'original': word_text
                        })
        
        if not all_words:
            logger.debug(f"No words found in Whisper segments for timing lookup")
            return None, None
        
        # Clean target text for matching - optimized
        target_words = []
        for w in target_text.split():
            clean_w = w.lower().strip().translate(punctuation_table).strip()
            if clean_w:
                target_words.append(clean_w)
        
        if not target_words:
            return None, None
        
        logger.debug(f"Looking for {len(target_words)} target words in {len(all_words)} Whisper words")
        
        # Use vectorized word matching
        return self._vectorized_word_sequence_match(all_words, target_words)
    
    def _vectorized_word_sequence_match(self, all_words: List[Dict], target_words: List[str]) -> Tuple[Optional[float], Optional[float]]:
        """Vectorized word sequence matching with multiple fallback strategies."""
        if not all_words or not target_words:
            return None, None
        
        # Convert to numpy arrays for vectorized operations
        word_texts = np.array([w['text'] for w in all_words], dtype=object)
        word_starts = np.array([w['start'] for w in all_words], dtype=np.float32)
        word_ends = np.array([w['end'] for w in all_words], dtype=np.float32)
        
        # Strategy 1: Exact sequence matching (fastest)
        result = self._vectorized_exact_match(word_texts, word_starts, word_ends, target_words)
        if result[0] is not None:
            logger.debug("Vectorized exact match found")
            return result
            
        # Strategy 2: Fuzzy sequence matching with rapidfuzz (fast)
        result = self._vectorized_fuzzy_match(word_texts, word_starts, word_ends, target_words)
        if result[0] is not None:
            logger.debug("Vectorized fuzzy match found")
            return result
            
        # Strategy 3: Partial match (first/last words)
        result = self._vectorized_partial_match(word_texts, word_starts, word_ends, target_words)
        if result[0] is not None:
            logger.debug("Vectorized partial match found")
            return result
            
        return None, None
    
    def _vectorized_exact_match(self, word_texts: np.ndarray, word_starts: np.ndarray, word_ends: np.ndarray, target_words: List[str]) -> Tuple[Optional[float], Optional[float]]:
        """Fast exact sequence matching using numpy."""
        target_len = len(target_words)
        if target_len > len(word_texts):
            return None, None
            
        # Vectorized string comparison
        for i in range(len(word_texts) - target_len + 1):
            window = word_texts[i:i+target_len]
            if np.array_equal(window, target_words):
                return float(word_starts[i]), float(word_ends[i+target_len-1])
        
        return None, None
    
    def _vectorized_fuzzy_match(self, word_texts: np.ndarray, word_starts: np.ndarray, word_ends: np.ndarray, target_words: List[str]) -> Tuple[Optional[float], Optional[float]]:
        """Fast fuzzy sequence matching using rapidfuzz."""
        target_len = len(target_words)
        if target_len > len(word_texts):
            return None, None
            
        tolerance = max(1, int(target_len * 0.2))  # 20% tolerance
        min_matches = int(target_len * 0.8)  # 80% match threshold
        
        for i in range(len(word_texts) - target_len + tolerance):
            matches = 0
            match_positions = []
            
            # Check up to target_len + tolerance positions
            window_size = min(target_len + tolerance, len(word_texts) - i)
            window_texts = word_texts[i:i+window_size]
            
            # Use rapidfuzz for batch string similarity
            target_idx = 0
            for j, word_text in enumerate(window_texts):
                if target_idx < target_len:
                    target_word = target_words[target_idx]
                    
                    # Fast similarity check using rapidfuzz
                    if (word_text == target_word or 
                        target_word in word_text or word_text in target_word or
                        fuzz.ratio(word_text, target_word) > 80):
                        matches += 1
                        match_positions.append(i + j)
                        target_idx += 1
            
            if matches >= min_matches and match_positions:
                start_pos = match_positions[0]
                end_pos = match_positions[-1]
                return float(word_starts[start_pos]), float(word_ends[end_pos])
        
        return None, None
    
    def _vectorized_partial_match(self, word_texts: np.ndarray, word_starts: np.ndarray, word_ends: np.ndarray, target_words: List[str]) -> Tuple[Optional[float], Optional[float]]:
        """Fast partial matching using first and last words."""
        if len(target_words) < 3:
            return None, None
            
        # Use first 3 and last 3 words
        start_words = target_words[:3]
        end_words = target_words[-3:]
        
        start_time = self._find_word_group_vectorized(word_texts, word_starts, word_ends, start_words, get_start=True)
        end_time = self._find_word_group_vectorized(word_texts, word_starts, word_ends, end_words, get_start=False)
        
        return start_time, end_time
    
    def _find_word_group_vectorized(self, word_texts: np.ndarray, word_starts: np.ndarray, word_ends: np.ndarray, search_words: List[str], get_start: bool = True) -> Optional[float]:
        """Vectorized search for a group of words."""
        search_len = len(search_words)
        min_matches = max(1, int(search_len * 0.7))  # 70% threshold
        
        for i in range(len(word_texts) - search_len + 1):
            matches = 0
            match_position = None
            
            window = word_texts[i:i+search_len]
            for j, (word_text, search_word) in enumerate(zip(window, search_words)):
                # Fast comparison using rapidfuzz
                if (word_text == search_word or 
                    search_word in word_text or word_text in search_word or
                    fuzz.ratio(word_text, search_word) > 70):
                    matches += 1
                    if match_position is None:
                        match_position = i + j
            
            if matches >= min_matches and match_position is not None:
                if get_start:
                    return float(word_starts[match_position])
                else:
                    last_match_pos = min(match_position + search_len - 1, len(word_texts) - 1)
                    return float(word_ends[last_match_pos])
        
        return None

    
    def _estimate_sentence_time_with_precision(self, content_id: str, transcription: SpeakerTranscription,
                                             sent_idx: int, total_sentences: int,
                                             use_start: bool = True) -> float:
        """Estimate time for a sentence with precise timing when possible."""
        # Try to get precise timing first
        if total_sentences > 1:
            # Get all sentences from this transcription
            sentences = sent_tokenize(transcription.text) if transcription.text else []
            if sent_idx < len(sentences):
                target_sentence = sentences[sent_idx]
                
                # Load Whisper chunks for this content
                whisper_segments = self._load_whisper_chunks_for_content(content_id)
                
                if whisper_segments:
                    # Try to find precise timing for this sentence
                    precise_start, precise_end = self._find_precise_word_timing(whisper_segments, target_sentence)
                    
                    if precise_start is not None and precise_end is not None:
                        # Use precise timing
                        result_time = precise_start if use_start else precise_end
                        logger.debug(f"[{content_id}] Using precise timing for sentence {sent_idx}: {result_time:.2f}s")
                        return result_time
        
        # Fallback to linear interpolation (old method)
        duration = transcription.end_time - transcription.start_time
        
        if total_sentences <= 1:
            return transcription.start_time if use_start else transcription.end_time
        
        # Linear interpolation based on sentence position
        if use_start:
            position = sent_idx / total_sentences
        else:
            position = (sent_idx + 1) / total_sentences
        
        fallback_time = transcription.start_time + (duration * position)
        logger.debug(f"[{content_id}] Using fallback linear interpolation for sentence {sent_idx}: {fallback_time:.2f}s")
        return fallback_time
    
    def _estimate_sentence_time(self, transcription: SpeakerTranscription,
                               sent_idx: int, total_sentences: int,
                               use_start: bool = True) -> float:
        """Legacy method - estimate time for a sentence within a transcription."""
        # This method is kept for compatibility but should use the new precise timing when content_id is available
        duration = transcription.end_time - transcription.start_time
        
        if total_sentences <= 1:
            return transcription.start_time if use_start else transcription.end_time
        
        # Linear interpolation based on sentence position
        if use_start:
            position = sent_idx / total_sentences
        else:
            position = (sent_idx + 1) / total_sentences
        
        return transcription.start_time + (duration * position)
    
    def _save_segments_to_s3(self, content_id: str, segments: List[Dict]) -> bool:
        """Save semantic segments to S3 as JSON."""
        try:
            # Create fresh S3 storage connection
            s3_storage = create_s3_storage_from_config(self.config['storage']['s3'])
            
            # Extract common metadata from first segment (if available)
            common_metadata = {}
            if segments:
                # Assuming all segments share the same configuration
                first_segment = segments[0]
                if 'segment_type' in first_segment:
                    # Extract segmentation method from segment type
                    if 'beam' in first_segment['segment_type']:
                        common_metadata['segmentation_method'] = 'beam_search'
                    else:
                        common_metadata['segmentation_method'] = 'sentence_similarity'
                
                common_metadata.update({
                    'similarity_model': 'XLM-R' if self.similarity_generator else 'Qwen3-0.6B',
                    'embedding_model': 'Qwen3-0.6B',
                    'coherence_threshold': self.coherence_threshold,
                    'target_tokens': self.target_tokens,
                    'min_tokens': self.min_tokens,
                    'max_tokens': self.max_tokens,
                    'beam_width': self.beam_width,
                    'lookahead_sentences': self.lookahead_sentences
                })
            
            # Prepare segments data for JSON output
            segments_data = {
                "content_id": content_id,
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "pipeline_version": "segment_embeddings_v3_precise_timing",
                "metadata": {
                    "total_segments": len(segments),
                    **common_metadata
                },
                "segments": []
            }
            
            # Add segment data (without embeddings for JSON file)
            for idx, segment in enumerate(segments):
                segment_data = {
                    "segment_index": idx,
                    "text": segment['text'],
                    "start_time": segment['start_time'],
                    "end_time": segment['end_time'],
                    "token_count": segment['token_count'],
                    "segment_type": segment['segment_type'],
                    "speaker_ids": segment['speaker_ids'],
                    "source_transcription_ids": segment.get('source_ids', [])
                }
                segments_data["segments"].append(segment_data)
            
            # Create temporary file
            temp_path = Path(f"/tmp/semantic_segments_{content_id}.json")
            with open(temp_path, 'w') as f:
                json.dump(segments_data, f, indent=2)
            
            # Upload to S3
            s3_key = f"content/{content_id}/semantic_segments.json"
            success = s3_storage.upload_file(str(temp_path), s3_key)
            
            # Cleanup temp file
            temp_path.unlink()
            
            if success:
                logger.info(f"[{content_id}] Successfully saved semantic segments to S3: {s3_key}")
                return True
            else:
                logger.error(f"[{content_id}] Failed to upload semantic segments to S3")
                return False
                
        except Exception as e:
            logger.error(f"[{content_id}] Error saving segments to S3: {e}")
            return False
    
    def _should_use_alternative_embeddings_for_content(self, content) -> bool:
        """Check if alternative embeddings should be used for this content based on project settings."""
        # First check if alternative embedding is configured and can be initialized
        should_use_alt = False
            
        # Check if content has projects and if any of them enable alternative embeddings
        if hasattr(content, 'projects') and content.projects:
            # Parse comma-separated projects string
            if isinstance(content.projects, str):
                project_list = content.projects if content.projects else []
            elif isinstance(content.projects, list):
                project_list = content.projects
            else:
                project_list = [str(content.projects)]
            
            # Get active projects config
            active_projects = self.config.get('active_projects', {})
            
            for project in project_list:
                project_config = active_projects.get(project, {})
                if project_config.get('use_alternative_embeddings', False):
                    logger.info(f"[{content.content_id}] Project '{project}' has alternative embeddings enabled")
                    should_use_alt = True
                    break
            
            if not should_use_alt:
                logger.info(f"[{content.content_id}] No projects with alternative embeddings enabled: {project_list}")
        else:
            logger.info(f"[{content.content_id}] No projects associated with content, using global config")
            should_use_alt = self.use_alternative_embedding
        
        # If alternative embeddings are needed, try to initialize the model
        if should_use_alt:
            if self.alt_embed_generator is None:
                logger.info(f"[{content.content_id}] Alternative embeddings needed - initializing alternative model...")
                self._init_alternative_embedding_model()
            
            # Check if initialization was successful
            if self.alt_embed_generator is None:
                logger.warning(f"[{content.content_id}] Alternative embedding model could not be initialized, falling back to primary embeddings")
                return False
                
        return should_use_alt
    
    async def process_content(self, content_id: str, rewrite: bool = False, 
                            stitch_version: str = None, embedding_version: str = None) -> Dict:
        """Process content to create embedding segments."""
        logger.info(f"[{content_id}] Starting embedding segmentation")
        
        try:
            # Clear any cached embeddings from previous runs
            self.sentence_embeddings = {}
            self.sentence_embeddings_array = None
            self.sentence_embeddings_normalized = None
            self.sentence_texts = []
            # Clear similarity cache for new content
            if hasattr(self, '_similarity_cache'):
                self._similarity_cache.clear()
            # Reset similarity calculation counter
            self._direct_similarity_count = 0
            
            # Initialize S3 storage for file checking
            s3_config = S3StorageConfig(
                endpoint_url=self.config['storage']['s3']['endpoint_url'],
                access_key=self.config['storage']['s3']['access_key'],
                secret_key=self.config['storage']['s3']['secret_key'],
                bucket_name=self.config['storage']['s3']['bucket_name'],
                use_ssl=self.config['storage']['s3']['use_ssl']
            )
            s3_storage = S3Storage(s3_config)
            
            with get_session() as session:
                # Get content record
                content = session.query(Content).filter_by(content_id=content_id).first()
                if not content:
                    return {'status': 'error', 'error': f'Content {content_id} not found'}
                
                # Check if alternative embeddings should be used for this content
                use_alt_for_this_content = self._should_use_alternative_embeddings_for_content(content)
                
                # Update model info based on actual usage
                if use_alt_for_this_content:
                    model_info = f"XLM-R + Qwen3-0.6B + {self.alternative_model}" if self.similarity_generator else f"Qwen3-0.6B + {self.alternative_model}"
                else:
                    model_info = f"XLM-R + Qwen3-0.6B" if self.similarity_generator else "Qwen3-0.6B only"
                logger.info(f"[{content_id}] Using model configuration: {model_info}")
                
                # Check if already processed
                if not rewrite:
                    # When using alternative embeddings, check processing status
                    if use_alt_for_this_content and self.save_to_alt_column:
                        # Check for existing segments that need alternative embeddings
                        existing_segments = session.query(EmbeddingSegment).filter_by(
                            content_id=content.id
                        ).all()
                        
                        if existing_segments:
                            # Check if they already have both primary and alternative embeddings
                            segments_with_primary = sum(1 for seg in existing_segments if seg.embedding is not None)
                            segments_with_alt = sum(1 for seg in existing_segments if seg.embedding_alt is not None)
                            
                            if segments_with_primary == len(existing_segments) and segments_with_alt == len(existing_segments):
                                logger.info(f"[{content_id}] All {len(existing_segments)} segments have both primary and alternative embeddings. Skipping.")
                                return {
                                    'status': 'skipped',
                                    'segment_count': len(existing_segments),
                                    'note': 'Both primary and alternative embeddings already exist'
                                }
                            elif segments_with_alt > 0 and segments_with_primary == len(existing_segments):
                                logger.info(f"[{content_id}] {segments_with_alt}/{len(existing_segments)} segments have alternative embeddings. Updating remaining...")
                                # Update only segments missing alternative embeddings
                                return self._update_alternative_embeddings(content_id, existing_segments)
                            elif segments_with_primary == 0:
                                logger.info(f"[{content_id}] No segments with primary embeddings found. Processing from scratch...")
                                # Process completely from scratch
                            else:
                                logger.info(f"[{content_id}] Found {segments_with_primary} segments with primary embeddings, {segments_with_alt} with alternative. Adding missing embeddings...")
                                # Add alternative embeddings to all existing segments
                                return self._update_alternative_embeddings(content_id, existing_segments)
                    else:
                        # Standard check for main embeddings
                        existing_count = session.query(EmbeddingSegment).filter_by(
                            content_id=content.id
                        ).count()
                        
                        # Also check if S3 file exists (original or compressed)
                        s3_key = f"content/{content_id}/semantic_segments.json"
                        s3_key_gz = f"content/{content_id}/semantic_segments.json.gz"
                        s3_file_exists = s3_storage.file_exists(s3_key) or s3_storage.file_exists(s3_key_gz)
                        
                        if existing_count > 0 and s3_file_exists:
                            logger.info(f"[{content_id}] Already has {existing_count} embedding segments and S3 file exists. Skipping.")
                            return {
                                'status': 'skipped',
                                'segment_count': existing_count
                            }
                        elif existing_count > 0 and not s3_file_exists:
                            logger.warning(f"[{content_id}] Has {existing_count} embedding segments but S3 file missing. Auto-enabling rewrite mode to fix mismatch.")
                            rewrite = True  # Automatically enable rewrite mode
                        elif existing_count == 0 and s3_file_exists:
                            logger.warning(f"[{content_id}] S3 file exists but no database records. Auto-enabling rewrite mode to fix mismatch.")
                            rewrite = True  # Automatically enable rewrite mode
                
                # Load speaker transcriptions
                transcriptions = session.query(SpeakerTranscription).filter_by(
                    content_id=content.id
                ).order_by(SpeakerTranscription.start_time).all()
                
                if not transcriptions:
                    return {
                        'status': 'error',
                        'error': 'No speaker transcriptions found'
                    }
                
                logger.info(f"[{content_id}] Found {len(transcriptions)} speaker transcriptions")
                
                # Segment transcriptions (uses XLM-R for similarity if available)
                similarity_model = "XLM-R" if self.similarity_generator else "E5"
                segmentation_method = "beam_search"
                logger.info(f"[{content_id}] Using {similarity_model} model for {segmentation_method} calculations")
                segments = self._segment_transcriptions(transcriptions, content_id)
                logger.info(f"[{content_id}] Created {len(segments)} segments using {segmentation_method} with {similarity_model} similarity")
                
                # Save segments to S3 for pipeline manager to detect
                if not self._save_segments_to_s3(content_id, segments):
                    logger.warning(f"[{content_id}] Failed to save segments to S3, continuing with database storage")
                
                # Generate embeddings and save
                if rewrite:
                    # Delete existing segments
                    session.query(EmbeddingSegment).filter_by(content_id=content.id).delete()
                    session.flush()
                
                saved_count = 0
                # Generate primary embeddings using Qwen3-0.6B model for all segments at once
                logger.info(f"[{content_id}] Generating primary Qwen3-0.6B embeddings for {len(segments)} segments")
                segment_texts = [seg['text'] for seg in segments]
                primary_embeddings = self.embed_generator.generate_embeddings(segment_texts, show_progress=True)
                
                if primary_embeddings is None or len(primary_embeddings) == 0:
                    logger.error(f"[{content_id}] Failed to generate primary Qwen3-0.6B embeddings")
                    return {
                        'status': 'error',
                        'error': 'Failed to generate primary Qwen3-0.6B embeddings'
                    }
                
                logger.info(f"[{content_id}] Successfully generated primary Qwen3-0.6B embeddings for all segments")
                
                # Generate alternative embeddings if enabled for this content
                alternative_embeddings = None
                if use_alt_for_this_content:
                    logger.info(f"[{content_id}] Generating alternative {self.alternative_model} embeddings for {len(segments)} segments")
                    alternative_embeddings = self.generate_alternative_embeddings(segment_texts, show_progress=True)
                    
                    if alternative_embeddings is None or len(alternative_embeddings) == 0:
                        logger.warning(f"[{content_id}] Failed to generate alternative embeddings, continuing with primary only")
                        alternative_embeddings = None
                    else:
                        logger.info(f"[{content_id}] Successfully generated alternative {self.alternative_model} embeddings for all segments")
                else:
                    logger.info(f"[{content_id}] Alternative embeddings not enabled for this content's projects")
                
                for idx, segment in enumerate(segments):
                    primary_embedding = primary_embeddings[idx]
                    alt_embedding = alternative_embeddings[idx] if alternative_embeddings is not None else None
                    # Create metadata
                    metadata = {
                        'speaker_ids': segment['speaker_ids'],
                        'segment_method': 'sentence_similarity',
                        'similarity_model': similarity_model,
                        'primary_embedding_model': 'Qwen3-0.6B',
                        'alternative_embedding_model': self.alternative_model if alt_embedding is not None else None,
                        'dual_embeddings': alt_embedding is not None,
                        'coherence_threshold': self.coherence_threshold,
                        'target_tokens': self.target_tokens,
                        'timing_method': segment.get('timing_method', 'linear_interpolation'),
                        'precise_timing_enabled': True,
                        'pipeline_version': 'v3_precise_timing'
                    }
                    
                    # Generate segment hash for deduplication
                    import hashlib
                    segment_data = f"{content_id}:{idx}:{segment['text'][:100]}"
                    segment_hash = hashlib.sha256(segment_data.encode()).hexdigest()[:8]
                    
                    # Check if segment already exists
                    existing_segment = session.query(EmbeddingSegment).filter_by(
                        segment_hash=segment_hash
                    ).first()
                    
                    if existing_segment:
                        logger.debug(f"[{content_id}] Segment {idx} with hash {segment_hash} already exists, skipping")
                        saved_count += 1  # Count it as saved since it exists
                        continue
                    
                    # Get speaker hashes for the speakers involved in this segment (optional field)
                    speaker_hashes = None
                    if segment.get('speaker_ids'):
                        try:
                            speakers_query = session.query(Speaker.speaker_hash).filter(
                                Speaker.id.in_(segment['speaker_ids'])
                            ).all()
                            speaker_hashes = [s.speaker_hash for s in speakers_query if s.speaker_hash]
                            if not speaker_hashes:  # Empty list becomes None for nullable field
                                speaker_hashes = None
                        except Exception as e:
                            logger.debug(f"Could not fetch speaker hashes: {e}")
                            speaker_hashes = None
                    
                    # Save both primary and alternative embeddings
                    db_segment = EmbeddingSegment(
                        content_id=content.id,
                        segment_index=idx,
                        text=segment['text'],
                        start_time=segment['start_time'],
                        end_time=segment['end_time'],
                        token_count=segment['token_count'],
                        segment_type=segment['segment_type'],
                        source_transcription_ids=segment['source_ids'],
                        source_start_char=segment.get('source_start_char'),
                        source_end_char=segment.get('source_end_char'),
                        # Primary embedding (Qwen3-0.6B)
                        embedding=primary_embedding.tolist() if isinstance(primary_embedding, np.ndarray) else None,
                        # Alternative embedding (Qwen3-4B) if available
                        embedding_alt=alt_embedding.tolist() if isinstance(alt_embedding, np.ndarray) else None,
                        embedding_alt_model=self.alternative_model if alt_embedding is not None else None,
                        meta_data=metadata,
                        created_at=datetime.now(timezone.utc),
                        stitch_version=stitch_version,
                        embedding_version=embedding_version or f"segment_embeddings_v3_precise_timing",
                        segment_hash=segment_hash,
                        content_id_string=content_id,
                        source_speaker_hashes=speaker_hashes,
                        speaker_positions=segment.get('speaker_positions')  # Add speaker position mapping
                    )
                    session.add(db_segment)
                    saved_count += 1
                    
                    if saved_count % 50 == 0:
                        session.flush()  # Periodic flush for large content
                        logger.info(f"[{content_id}] Saved {saved_count}/{len(segments)} segments")
                
                # Update content status and segment version
                content.is_embedded = True
                
                # Update segment_version in meta_data to match config
                current_segment_version = self.config.get('processing', {}).get('segment', {}).get('current_version', 'segment_v1')
                meta_data = dict(content.meta_data) if content.meta_data else {}
                meta_data['segment_version'] = current_segment_version
                content.meta_data = meta_data
                
                session.commit()
                
                # Update status message based on what was generated
                if alternative_embeddings is not None:
                    logger.info(f"[{content_id}] Successfully created {saved_count} embedding segments with both primary (Qwen3-0.6B) and alternative ({self.alternative_model}) embeddings")
                else:
                    logger.info(f"[{content_id}] Successfully created {saved_count} embedding segments with primary embeddings only using {model_info}")
                
                # Log similarity calculation statistics
                if hasattr(self, '_direct_similarity_count'):
                    total_sentences = len(all_sentences) if 'all_sentences' in locals() else 0
                    logger.info(f"[{content_id}] Similarity calculation stats: {self._direct_similarity_count} direct calculations from {total_sentences} sentences")
                    if hasattr(self, '_similarity_cache'):
                        logger.info(f"[{content_id}] Similarity cache size: {len(self._similarity_cache)} cached pairs")
                
                # Log timing method statistics
                timing_methods = {}
                for segment in segments:
                    method = segment.get('timing_method', 'linear_interpolation')
                    timing_methods[method] = timing_methods.get(method, 0) + 1
                
                logger.info(f"[{content_id}] Timing method statistics: {timing_methods}")
                precise_timing_count = timing_methods.get('whisper_word_precise', 0) + timing_methods.get('whisper_sentence_precise', 0)
                total_segments = len(segments)
                precise_percentage = (precise_timing_count / total_segments) * 100 if total_segments > 0 else 0
                logger.info(f"[{content_id}] Precise timing coverage: {precise_timing_count}/{total_segments} segments ({precise_percentage:.1f}%)")
                
                return {
                    'status': 'success',
                    'segment_count': saved_count,
                    'source_transcription_count': len(transcriptions),
                    'similarity_model': similarity_model,
                    'primary_embedding_model': 'Qwen3-0.6B',
                    'alternative_embedding_model': self.alternative_model if alternative_embeddings is not None else None,
                    'dual_embeddings_generated': alternative_embeddings is not None,
                    'alternative_embeddings_enabled_for_content': use_alt_for_this_content,
                    'models_used': model_info,
                    'timing_methods': timing_methods,
                    'precise_timing_coverage': f"{precise_timing_count}/{total_segments} ({precise_percentage:.1f}%)",
                    'pipeline_version': 'v3_precise_timing'
                }
                
        except Exception as e:
            logger.error(f"[{content_id}] Error in embedding segmentation: {e}", exc_info=True)
            # Ensure session is rolled back to avoid "transaction has been rolled back" errors
            if 'session' in locals():
                try:
                    session.rollback()
                except:
                    pass  # Session may already be closed
            return {
                'status': 'error',
                'error': str(e)
            }

    async def test_mode_process(self, content_id: str) -> Dict:
        """Test mode processing with downloaded inputs and saved outputs."""
        from pathlib import Path
        import shutil
        
        logger.info(f"[{content_id}] Starting TEST MODE processing")
        
        # Create test directory structure
        test_output_dir = get_project_root() / "tests" / "content" / content_id
        test_inputs_dir = test_output_dir / "inputs"
        test_outputs_dir = test_output_dir / "outputs"
        
        test_output_dir.mkdir(parents=True, exist_ok=True)
        test_inputs_dir.mkdir(exist_ok=True)
        test_outputs_dir.mkdir(exist_ok=True)
        
        logger.info(f"[{content_id}] TEST MODE: Using directory {test_output_dir}")
        
        try:
            # Download speaker transcriptions from database and save to inputs
            with get_session() as session:
                content = session.query(Content).filter_by(content_id=content_id).first()
                if not content:
                    return {'status': 'error', 'error': f'Content {content_id} not found'}
                
                transcriptions = session.query(SpeakerTranscription).filter_by(
                    content_id=content.id
                ).order_by(SpeakerTranscription.start_time).all()
                
                if not transcriptions:
                    return {'status': 'error', 'error': 'No speaker transcriptions found'}
                
                # Save transcriptions to test inputs
                transcriptions_data = []
                for trans in transcriptions:
                    transcriptions_data.append({
                        'id': trans.id,
                        'text': trans.text,
                        'start_time': trans.start_time,
                        'end_time': trans.end_time,
                        'speaker_id': trans.speaker_id,
                        'turn_index': trans.turn_index,
                        'stitch_version': trans.stitch_version
                    })
                
                inputs_file = test_inputs_dir / "speaker_transcriptions.json"
                with open(inputs_file, 'w') as f:
                    json.dump({
                        'content_id': content_id,
                        'transcription_count': len(transcriptions_data),
                        'transcriptions': transcriptions_data
                    }, f, indent=2)
                
                logger.info(f"[{content_id}] TEST MODE: Saved {len(transcriptions_data)} transcriptions to {inputs_file}")
            
            # Process the content normally
            result = await self.process_content(content_id, rewrite=True)
            
            if result['status'] == 'success':
                # Save additional test outputs
                with get_session() as session:
                    embedding_segments = session.query(EmbeddingSegment).filter_by(
                        content_id=content.id
                    ).order_by(EmbeddingSegment.segment_index).all()
                    
                    # Save embedding segments
                    segments_data = []
                    for seg in embedding_segments:
                        segments_data.append({
                            'segment_index': seg.segment_index,
                            'text': seg.text,
                            'start_time': seg.start_time,
                            'end_time': seg.end_time,
                            'token_count': seg.token_count,
                            'segment_type': seg.segment_type,
                            'source_transcription_ids': seg.source_transcription_ids,
                            'source_start_char': seg.source_start_char,
                            'source_end_char': seg.source_end_char,
                            'embedding_dim': len(seg.embedding) if seg.embedding is not None else 0,
                            'meta_data': seg.meta_data
                        })
                    
                    segments_file = test_outputs_dir / "embedding_segments.json"
                    with open(segments_file, 'w') as f:
                        json.dump({
                            'content_id': content_id,
                            'segment_count': len(segments_data),
                            'segments': segments_data
                        }, f, indent=2)
                    
                    # Create readable summary
                    summary_lines = [
                        f"EMBEDDING SEGMENTATION TEST RESULTS",
                        f"Content ID: {content_id}",
                        f"Input transcriptions: {len(transcriptions_data)}",
                        f"Output segments: {len(segments_data)}",
                        f"",
                        f"SEGMENTATION BREAKDOWN:",
                    ]
                    
                    segment_types = {}
                    for seg in segments_data:
                        seg_type = seg['segment_type']
                        segment_types[seg_type] = segment_types.get(seg_type, 0) + 1
                    
                    for seg_type, count in segment_types.items():
                        summary_lines.append(f"  {seg_type}: {count} segments")
                    
                    summary_lines.extend([
                        f"",
                        f"SAMPLE SEGMENTS:",
                    ])
                    
                    for i, seg in enumerate(segments_data[:5]):
                        summary_lines.extend([
                            f"",
                            f"Segment {seg['segment_index']} ({seg['segment_type']}):",
                            f"  Time: {seg['start_time']:.2f}s - {seg['end_time']:.2f}s",
                            f"  Tokens: {seg['token_count']}",
                            f"  Text: {seg['text'][:100]}{'...' if len(seg['text']) > 100 else ''}"
                        ])
                    
                    summary_file = test_outputs_dir / "test_summary.txt"
                    with open(summary_file, 'w') as f:
                        f.write('\n'.join(summary_lines))
                    
                    logger.info(f"[{content_id}] TEST MODE: Created test summary at {summary_file}")
                
                # Update result with test paths
                result.update({
                    'test_directory': str(test_output_dir),
                    'test_inputs': str(test_inputs_dir),
                    'test_outputs': str(test_outputs_dir)
                })
            
            return result
            
        except Exception as e:
            logger.error(f"[{content_id}] TEST MODE error: {e}", exc_info=True)
            return {'status': 'error', 'error': f'Test mode error: {str(e)}'}

async def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description='Segment transcriptions and generate embeddings')
    parser.add_argument('--content', required=True, help='Content ID to process')
    parser.add_argument('--rewrite', action='store_true', help='Force reprocess')
    parser.add_argument('--test', action='store_true', help='Test mode: download inputs and save outputs locally')
    parser.add_argument('--qwen4b', action='store_true', help='Use Qwen3-Embedding-4B model instead of default 0.6B')
    args = parser.parse_args()
    
    # Create segmenter - command line flag overrides config
    segmenter = EmbeddingSegmenter()
    if args.qwen4b:
        segmenter.use_alternative_embedding = True
        segmenter.alternative_model = "Qwen/Qwen3-Embedding-4B"
        segmenter.alternative_model_dim = 2000
        segmenter.save_to_alt_column = True
        # Model will be initialized lazily when first needed
        logger.info("Command-line override: Enabling alternative Qwen3-Embedding-4B model (lazy loading)")
    
    if args.test:
        result = await segmenter.test_mode_process(args.content)
    else:
        result = await segmenter.process_content(args.content, args.rewrite)
    
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'success' else 1

if __name__ == "__main__":
    asyncio.run(main())