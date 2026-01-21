#!/usr/bin/env python3
"""
Unified Theme Classification Pipeline with MLX and FAISS

This script combines semantic search (FAISS CPU) with keyword matching for efficient theme classification.
Uses MLX for direct Apple Silicon LLM inference and FAISS CPU for vector similarity search.

Features:
- PostgreSQL GIN text index for fast keyword matching
- FAISS CPU for semantic vector search (supports index persistence)
- MLX for local LLM inference (no HTTP server needed)
- Speaker attribution with anonymous speaker labels

Usage:
    python theme_classifier.py \
        --subthemes-csv projects/CPRMV/subthemes_complete.csv \
        --keywords-file projects/CPRMV/keywords.txt \
        --output-csv outputs/theme_results.csv \
        --similarity-threshold 0.6 \
        --model-name tier_2 \
        --use-faiss
"""

import sys
import os
import pandas as pd
import csv
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from src.utils.paths import get_project_root, get_config_path
import argparse
from tqdm import tqdm
import time
import re
import asyncio

# MLX imports for direct model inference (fallback only)
try:
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

# Add project root to path
sys.path.append(str(get_project_root()))

from src.database.session import get_session
from src.database.models import EmbeddingSegment, Content
from sqlalchemy import text
from src.utils.logger import setup_worker_logger
from sentence_transformers import SentenceTransformer
from src.classification.speaker_assignment import (
    assign_speakers_to_segment,
    format_speaker_attributed_text,
    get_speaker_info_summary,
    SpeakerSegment
)

logger = setup_worker_logger('unified_theme_classifier')

try:
    import faiss
    logger.info("Using FAISS CPU for vector search")
except ImportError:
    logger.error("FAISS not installed. Install with: pip install faiss-cpu")
    raise

# ============================================================================
# Configuration Classes
# ============================================================================

@dataclass
class UnifiedConfig:
    """Configuration for unified theme classification pipeline with MLX and FAISS"""
    subthemes_csv: str
    output_csv: str
    keywords_file: Optional[str] = None
    similarity_threshold: float = 0.6
    keyword_boost: float = 0.1
    keyword_threshold_reduction: float = 0.04  # More permissive threshold for keyword matches
    model_name: str = "tier_2"  # MLX model: tier_1 (80B), tier_2 (4B), tier_3 (8B)
    embedding_model: str = "Qwen/Qwen3-Embedding-4B"  # Embedding model for semantic search
    save_intermediate: bool = True
    use_gpu: bool = True  # MLX uses Apple Silicon GPU by default
    project: Optional[str] = None
    force_language: Optional[str] = None

    # Model server configuration
    model_server_url: Optional[str] = None  # e.g., "http://10.0.0.34:8004"
    use_model_server: bool = True  # Prefer model_server over llm_server_mlx

    # Test mode
    test_mode: Optional[int] = None  # If set, limits total candidates for testing

    # Checkpoint configuration
    checkpoint_file: Optional[str] = None
    resume: bool = True  # Resume from checkpoint if available

    # FAISS configuration
    use_faiss: bool = False  # Use FAISS CPU instead of PostgreSQL HNSW
    faiss_index_path: Optional[str] = None  # Path to save/load FAISS index

    # Cache configuration
    skip_stage1_cache: bool = False  # Force fresh Stage 1 search

    # Cleanup configuration
    cleanup_stale_segments: bool = False  # Remove segment IDs that no longer exist in DB

@dataclass
class SearchCandidate:
    """Unified candidate structure for search results"""
    segment_id: int
    content_id: str
    episode_title: str
    episode_channel: str
    segment_text: str
    start_time: float
    end_time: float
    segment_index: int
    
    # Search metadata
    similarity_score: float
    matched_via: str  # 'keyword', 'semantic', or 'both'
    matched_keywords: List[str] = None
    matching_themes: List[str] = None
    original_url: str = None
    
    # Speaker attribution fields
    speaker_attributed_text: str = None
    speaker_segments: List[SpeakerSegment] = None
    speaker_info: Dict[str, Any] = None
    source_sentence_ids: List[int] = None  # Sentence indices from sentences table
    sentence_texts: List[str] = None  # Sentence texts from sentences table
    speaker_ids: List[int] = None  # Speaker IDs from sentences table
    
    # LLM classification results (filled later)
    theme_ids: List[int] = None
    theme_names: List[str] = None
    theme_confidence: float = 0.0
    theme_reasoning: str = ""
    subtheme_results: Dict[int, Dict[str, Any]] = None
    
    # Validation results (filled in stage 4)
    validation_results: Dict[int, Dict[str, Any]] = None

@dataclass
class ClassificationResult:
    """Result from LLM classification"""
    theme_id: int = 0
    theme_name: str = ""
    theme_ids: List[int] = None
    theme_names: List[str] = None
    subtheme_ids: List[int] = None
    subtheme_names: List[str] = None
    confidence: float = 0.0
    reasoning: str = ""

# ============================================================================
# LLM Classifier (Integrated from stage2_llm_classifier.py)
# ============================================================================

class LLMClassifier:
    """LLM-based classifier for themes and sub-themes using unified LLM client"""

    def __init__(
        self,
        themes_csv: str,
        model_name: str = "tier_2",
        use_gpu: bool = True,  # MLX uses Apple Silicon GPU by default (fallback only)
        llm_endpoint: str = None,  # Deprecated
        project: Optional[str] = None,
        model_server_url: Optional[str] = None,  # Deprecated - uses unified client
        use_model_server: bool = True  # Always uses unified client now
    ):
        from src.utils.llm_client import LLMClient, create_classification_client

        self.project = project
        self.model_name = model_name

        # Model name mapping (similar to llm_server_mlx.py) - kept for local fallback
        self.model_configs = {
            "mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit": {
                "aliases": ["qwen3:80b", "80b", "large", "tier_1", "tier1", "best"],
            },
            "mlx-community/Qwen3-4B-Instruct-2507-4bit": {
                "aliases": ["qwen3:4b-instruct", "qwen3:4b", "4b", "medium", "tier_2", "tier2", "balanced"],
            },
            "mlx-community/LFM2-8B-A1B-4bit": {
                "aliases": ["lfm2:8b", "8b", "small", "fast", "tier_3", "tier3", "fastest"],
            }
        }

        # Use unified LLM client for classification (tier_2, priority 3)
        self._llm_client = create_classification_client(tier=model_name)
        self.model = None  # No local model by default
        self.tokenizer = None
        logger.info(f"Using unified LLM client (tier: {model_name})")

        self.themes, self.subthemes = self._load_themes(themes_csv)

    def _resolve_model_name(self, requested_model: str) -> str:
        """Resolve model alias to full model name."""
        # Check if it's already a full model name
        if requested_model in self.model_configs:
            return requested_model

        # Check aliases
        for model_path, config in self.model_configs.items():
            if requested_model in config.get('aliases', []):
                return model_path

        # Default to 4B model
        logger.warning(f"Unknown model '{requested_model}', defaulting to 4B model")
        return "mlx-community/Qwen3-4B-Instruct-2507-4bit"
    
    def _load_themes(self, csv_path: str) -> Tuple[Dict[int, Dict], Dict[int, Dict]]:
        """Load themes and subthemes from CSV"""
        df = pd.read_csv(csv_path)
        
        themes = {}
        subthemes = {}
        
        for theme_id in df['theme_id'].unique():
            theme_df = df[df['theme_id'] == theme_id]
            theme_info = theme_df.iloc[0]
            
            themes[theme_id] = {
                'id': theme_id,
                'name': theme_info['theme_name'],
                'description': theme_info['theme_description']
            }
            
            # Collect subthemes
            theme_subthemes = []
            for _, row in theme_df.iterrows():
                if pd.notna(row.get('subtheme_id')):
                    subtheme = {
                        'id': row['subtheme_id'],
                        'name': row['subtheme_name'],
                        'description': row.get('subtheme_description_short', row['subtheme_description'])
                    }
                    theme_subthemes.append(subtheme)
            
            if theme_subthemes:
                subthemes[theme_id] = theme_subthemes
        
        return themes, subthemes

    def _call_llm_sync(self, prompt: str, priority: int = 2) -> str:
        """Synchronous wrapper for async _call_llm"""
        import asyncio

        # Check if we're already in an event loop
        try:
            loop = asyncio.get_running_loop()
            # We're in a running loop, use nest_asyncio
            import nest_asyncio
            nest_asyncio.apply()
            return asyncio.run(self._call_llm(prompt, priority))
        except RuntimeError:
            # No running loop, safe to use run_until_complete
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            return loop.run_until_complete(self._call_llm(prompt, priority))

    async def _call_llm(self, prompt: str, priority: int = 2) -> str:
        """Call LLM via unified client."""
        logger.debug("="*60)
        logger.debug("LLM PROMPT:")
        logger.debug(prompt)
        logger.debug("="*60)

        # Build messages for chat template
        messages = [
            {"role": "system", "content": "You are a precise classification assistant. Respond only with the requested numbers."},
            {"role": "user", "content": prompt}
        ]

        try:
            llm_response = await self._llm_client.call(
                messages=messages,
                temperature=0.1,
                max_tokens=50,
                priority=priority,
            )
            logger.debug(f"LLM RESPONSE: {llm_response}")
            logger.debug("="*60)
            return llm_response

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            # Fallback to local MLX if available
            if self.model is not None and MLX_AVAILABLE:
                logger.warning("Falling back to local MLX model")
                try:
                    if self.tokenizer.chat_template is not None:
                        formatted_prompt = self.tokenizer.apply_chat_template(
                            messages,
                            add_generation_prompt=True
                        )
                    else:
                        formatted_prompt = "\n\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

                    sampler = make_sampler(temp=0.1, top_p=0.9)
                    response_text = generate(
                        self.model,
                        self.tokenizer,
                        prompt=formatted_prompt,
                        max_tokens=50,
                        sampler=sampler,
                        verbose=False
                    )
                    return response_text.strip()
                except Exception as mlx_error:
                    logger.error(f"Local MLX fallback also failed: {mlx_error}")
                    raise
            raise

    async def _call_llm_batch(self, prompts: List[str], priority: int = 2, max_chunk_size: int = 50) -> List[str]:
        """Call LLM with multiple prompts via unified client batch endpoint.

        Args:
            prompts: List of prompts to process
            priority: Priority for model server queue
            max_chunk_size: Maximum requests to send at once
        """
        if not prompts:
            return []

        # Build message batches
        message_batches = [
            [
                {"role": "system", "content": "You are a precise classification assistant. Respond only with the requested numbers."},
                {"role": "user", "content": prompt}
            ]
            for prompt in prompts
        ]

        return await self._llm_client.call_batch(
            message_batches=message_batches,
            batch_size=max_chunk_size,
            temperature=0.1,
            max_tokens=50,
            priority=priority,
        )
    
    def classify_theme(self, text: str, context: Optional[Dict] = None, segment_id: Optional[int] = None) -> ClassificationResult:
        """Classify text into themes"""
        if not text or not text.strip():
            return ClassificationResult()
        
        themes_list = "\n".join([
            f"{tid}. {tinfo['description']}"
            for tid, tinfo in sorted(self.themes.items())
        ])
        
        prompt = f"""Classify the following text segment into relevant themes.
You may select multiple themes if applicable.
Respond with ONLY the theme numbers separated by commas (e.g., "1,3,5").
If no themes apply, respond with "0".

THEMES:
{themes_list}

TEXT TO CLASSIFY:
{text[:2000]}

THEME NUMBERS (comma-separated):"""
        
        try:
            response = self._call_llm_sync(prompt, priority=1)  # Priority 1 for theme/subtheme classification
            
            # Extract theme numbers
            if response == "0":
                result = ClassificationResult(reasoning="No themes identified")
            else:
                numbers = re.findall(r'\d+', response)
                theme_ids = []
                seen = set()
                for num_str in numbers:
                    num = int(num_str)
                    if num in self.themes and num not in seen:
                        theme_ids.append(num)
                        seen.add(num)
                
                if not theme_ids:
                    result = ClassificationResult(reasoning="No valid themes identified")
                else:
                    theme_names = [self.themes[tid]['name'] for tid in theme_ids]
                    
                    result = ClassificationResult(
                        theme_id=theme_ids[0],
                        theme_ids=theme_ids,
                        theme_name=theme_names[0],
                        theme_names=theme_names,
                        confidence=0.9 if len(theme_ids) == 1 else 0.7,
                        reasoning=f"Classified as {', '.join(theme_names)}"
                    )
            
            return result
            
        except Exception as e:
            logger.error(f"Theme classification error: {e}")
            return ClassificationResult(reasoning=f"Error: {str(e)}")
    
    def classify_subtheme(self, text: str, theme_id: int, context: Optional[Dict] = None, segment_id: Optional[int] = None) -> ClassificationResult:
        """Classify text into subthemes within a theme"""
        if not text or theme_id not in self.themes:
            return ClassificationResult()
        
        if theme_id not in self.subthemes:
            return ClassificationResult(reasoning=f"No subthemes for theme {theme_id}")
        
        # Create simple numbering for subthemes
        subtheme_mapping = {}
        subthemes_list = []
        for i, st in enumerate(self.subthemes[theme_id], 1):
            subtheme_mapping[i] = st['id']
            subthemes_list.append(f"{i}. {st['description']}")
        
        theme_info = self.themes[theme_id]
        prompt = f"""Classify the following text segment into relevant themes.
You may select multiple themes if applicable.
Respond with ONLY the theme numbers separated by commas (e.g., "1,3,5").
If no themes apply, respond with "0".

THEMES for {theme_info['name']}:
{chr(10).join(subthemes_list)}

TEXT TO CLASSIFY:
{text}

THEME NUMBERS (comma-separated):"""
        
        try:
            response = self._call_llm_sync(prompt, priority=1)  # Priority 1 for theme/subtheme classification
            
            numbers = re.findall(r'\d+', response)
            subtheme_ids = []
            for num_str in numbers:
                num = int(num_str)
                if num in subtheme_mapping:
                    subtheme_ids.append(subtheme_mapping[num])
            
            if not subtheme_ids:
                result = ClassificationResult(reasoning="No subthemes identified")
            else:
                available_subthemes = {st['id']: st for st in self.subthemes[theme_id]}
                subtheme_names = [
                    available_subthemes[sid]['name']
                    for sid in subtheme_ids
                    if sid in available_subthemes
                ]
                
                result = ClassificationResult(
                    subtheme_ids=subtheme_ids,
                    subtheme_names=subtheme_names,
                    confidence=0.8 if len(subtheme_ids) <= 2 else 0.6,
                    reasoning=f"Identified {len(subtheme_ids)} subthemes"
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Subtheme classification error: {e}")
            return ClassificationResult(reasoning=f"Error: {str(e)}")

# ============================================================================
# FAISS Index Loader
# ============================================================================

class FAISSIndexLoader:
    """Load embeddings and build FAISS index for fast semantic search"""

    def __init__(self, project: str, embedding_dim: int = 2000, limit: Optional[int] = None):
        self.project = project
        self.embedding_dim = embedding_dim
        self.limit = limit
        self.index = None
        self.segment_id_mapping = {}  # array_index -> segment_id
        self.reverse_mapping = {}  # segment_id -> array_index
        self.embeddings = None

    def load_embeddings_from_db(self) -> Tuple[np.ndarray, Dict[int, int]]:
        """Load embeddings from PostgreSQL for the specified project using streaming/chunked approach"""
        if self.limit:
            logger.info(f"Loading up to {self.limit} embeddings for project {self.project} from database (test mode)...")
        else:
            logger.info(f"Loading all embeddings for project {self.project} from database...")

        with get_session() as session:
            # Add LIMIT clause if specified
            limit_clause = f"LIMIT {self.limit}" if self.limit else ""

            # PASS 1: Get count to pre-allocate array
            count_query = text(f"""
                SELECT COUNT(*)
                FROM embedding_segments es
                JOIN content c ON es.content_id = c.id
                WHERE :project = ANY(c.projects)
                AND es.embedding_alt IS NOT NULL
                {limit_clause}
            """)

            count_result = session.execute(count_query, {'project': self.project})
            total_count = count_result.scalar()

            logger.info(f"Found {total_count:,} segments with embeddings")

            if total_count == 0:
                logger.warning("No segments found!")
                return np.array([]).reshape(0, self.embedding_dim).astype(np.float32), {}

            # Pre-allocate numpy array (saves memory compared to list + vstack)
            embeddings = np.zeros((total_count, self.embedding_dim), dtype=np.float32)
            logger.info(f"Pre-allocated array: {embeddings.shape}, {embeddings.nbytes / 1024 / 1024:.2f} MB")

            # PASS 2: Stream embeddings in chunks using server-side cursor
            data_query = text(f"""
                SELECT es.id, es.embedding_alt
                FROM embedding_segments es
                JOIN content c ON es.content_id = c.id
                WHERE :project = ANY(c.projects)
                AND es.embedding_alt IS NOT NULL
                ORDER BY es.id
                {limit_clause}
            """)

            # Use yield_per for server-side cursor (streaming)
            chunk_size = 50000
            result = session.execute(data_query, {'project': self.project})
            result = result.yield_per(chunk_size)

            idx = 0
            with tqdm(total=total_count, desc="Loading embeddings", unit="emb") as pbar:
                for segment_id, embedding_vector in result:
                    # Convert vector to numpy array
                    if isinstance(embedding_vector, str):
                        embedding_str = embedding_vector.strip('[]')
                        embedding_array = np.array([float(x) for x in embedding_str.split(',')], dtype=np.float32)
                    else:
                        embedding_array = np.array(embedding_vector, dtype=np.float32)

                    # Direct write to pre-allocated array (no intermediate list)
                    if idx >= total_count:
                        logger.warning(f"Received more rows ({idx+1}) than COUNT query returned ({total_count}), stopping")
                        break

                    embeddings[idx] = embedding_array
                    self.segment_id_mapping[idx] = segment_id
                    self.reverse_mapping[segment_id] = idx

                    idx += 1
                    pbar.update(1)

                    # Progress logging every 100k
                    if idx % 100000 == 0:
                        logger.info(f"  Loaded {idx:,}/{total_count:,} embeddings ({idx/total_count*100:.1f}%)")

            logger.info(f"Loaded embeddings array: {embeddings.shape}, {embeddings.nbytes / 1024 / 1024:.2f} MB")
            return embeddings, self.segment_id_mapping

    def build_index(self, embeddings: np.ndarray):
        """Build FAISS CPU index for cosine similarity search"""
        logger.info(f"Building FAISS CPU index for {embeddings.shape[0]} embeddings...")
        logger.info(f"  Embeddings dtype: {embeddings.dtype}, shape: {embeddings.shape}")
        logger.info(f"  Memory size: {embeddings.nbytes / 1024 / 1024:.2f} MB")
        dimension = embeddings.shape[1]

        # Ensure embeddings are contiguous in memory and C-ordered
        if not embeddings.flags['C_CONTIGUOUS']:
            logger.info("Converting embeddings to C-contiguous array...")
            embeddings = np.ascontiguousarray(embeddings)

        # Normalize embeddings for cosine similarity in chunks to avoid memory issues
        logger.info("Normalizing embeddings in chunks...")
        norm_chunk_size = 100000
        for i in range(0, embeddings.shape[0], norm_chunk_size):
            end_idx = min(i + norm_chunk_size, embeddings.shape[0])
            chunk = embeddings[i:end_idx]
            norms = np.linalg.norm(chunk, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embeddings[i:end_idx] = chunk / norms
            if (i // norm_chunk_size) % 10 == 0:
                logger.info(f"  Normalized {end_idx}/{embeddings.shape[0]} vectors...")
        logger.info(f"Normalization complete, all vectors now have unit length")

        # Use IndexFlatIP (Inner Product) for cosine similarity with normalized vectors
        logger.info("Creating FAISS IndexFlatIP for Inner Product search...")
        index = faiss.IndexFlatIP(dimension)

        # Add embeddings in chunks
        chunk_size = 100000  # Process 100k at a time
        logger.info("Adding embeddings to index in chunks...")
        for i in range(0, embeddings.shape[0], chunk_size):
            end_idx = min(i + chunk_size, embeddings.shape[0])
            chunk = embeddings[i:end_idx]
            # Ensure chunk is contiguous
            if not chunk.flags['C_CONTIGUOUS']:
                chunk = np.ascontiguousarray(chunk)
            index.add(chunk)
            logger.info(f"  Added {end_idx}/{embeddings.shape[0]} embeddings to index ({(end_idx/embeddings.shape[0]*100):.1f}%)")

        logger.info(f"FAISS index built successfully: {index.ntotal} vectors, dimension {dimension}")
        return index, embeddings

    def load_or_build_index(self, index_path: Optional[str] = None):
        """Load existing FAISS index or build new one

        Note: In test mode with a limit, we won't save the index to avoid overwriting
        a full production index with a limited test index.
        """
        # Try to load existing index if path provided and not in test mode
        if index_path and Path(index_path).exists() and not self.limit:
            logger.info(f"Loading existing FAISS index from {index_path}...")
            try:
                self.index = faiss.read_index(index_path)
                # Also need to load mappings
                mapping_path = str(Path(index_path).with_suffix('.mapping.json'))
                if Path(mapping_path).exists():
                    with open(mapping_path, 'r') as f:
                        mapping_data = json.load(f)
                        self.segment_id_mapping = {int(k): v for k, v in mapping_data['segment_id_mapping'].items()}
                        self.reverse_mapping = {v: int(k) for k, v in mapping_data['segment_id_mapping'].items()}
                    logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
                    return self.index
                else:
                    logger.warning(f"Mapping file not found at {mapping_path}, rebuilding index")
            except Exception as e:
                logger.warning(f"Failed to load index: {e}, rebuilding...")

        # Build new index
        logger.info("Building new FAISS index from database embeddings...")
        self.embeddings, self.segment_id_mapping = self.load_embeddings_from_db()
        self.reverse_mapping = {v: k for k, v in self.segment_id_mapping.items()}
        self.index, self.embeddings = self.build_index(self.embeddings)

        # Save index if path provided and not in test mode
        if index_path and not self.limit:
            self.save_index(index_path)

        logger.info("FAISS index ready for CPU search")
        return self.index

    def save_index(self, index_path: str):
        """Save FAISS index and mappings to disk"""
        if self.index is None:
            logger.warning("No index to save")
            return

        try:
            # Save FAISS index
            logger.info(f"Saving FAISS index to {index_path}...")
            faiss.write_index(self.index, index_path)

            # Save mappings
            mapping_path = str(Path(index_path).with_suffix('.mapping.json'))
            with open(mapping_path, 'w') as f:
                json.dump({
                    'segment_id_mapping': {str(k): v for k, v in self.segment_id_mapping.items()}
                }, f)

            logger.info(f"Saved FAISS index and mappings")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")

    def search(self, query_embeddings: np.ndarray, k: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """Search FAISS index

        Args:
            query_embeddings: Query vectors, shape (n_queries, dim)
            k: Number of nearest neighbors to return

        Returns:
            scores: Similarity scores, shape (n_queries, k)
            indices: Array indices, shape (n_queries, k)
        """
        if self.index is None:
            raise ValueError("Index not loaded")

        # Normalize query embeddings for cosine similarity
        query_embeddings = query_embeddings.astype(np.float32)
        faiss.normalize_L2(query_embeddings)

        # Search - FAISS returns (distances, indices) as numpy arrays
        scores, indices = self.index.search(query_embeddings, k)

        return scores, indices

    def get_segment_ids(self, array_indices: np.ndarray) -> List[int]:
        """Convert array indices to segment IDs"""
        return [self.segment_id_mapping.get(idx, -1) for idx in array_indices]

# ============================================================================
# Main Unified Pipeline
# ============================================================================

class UnifiedThemeClassifier:
    """Unified pipeline with keyword and semantic search"""

    def __init__(self, config: UnifiedConfig):
        self.config = config

        # Lazy-loaded models (loaded on demand)
        self.embedding_model = None
        self.faiss_loader = None
        self.classifier = None

        # Load themes and keywords
        self.theme_descriptions = self._load_theme_descriptions(config.subthemes_csv)
        self.keywords = self._load_keywords(config.keywords_file) if config.keywords_file else {}

        # Prepare output
        self.output_path = Path(config.output_csv)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Setup checkpoint file - use incremental naming to match the incremental CSV
        if not config.checkpoint_file:
            checkpoint_name = self.output_path.stem + "_incremental_checkpoint.json"
            self.checkpoint_path = self.output_path.parent / checkpoint_name
        else:
            self.checkpoint_path = Path(config.checkpoint_file)

        # Track processed segments
        self.processed_segments = set()

        logger.info(f"Initialized unified classifier (lazy model loading)")
        logger.info(f"Search mode: {'FAISS' if config.use_faiss else 'PostgreSQL HNSW'}")
        logger.info(f"Themes: {len(self.theme_descriptions['en'])}")
        logger.info(f"Keywords loaded: {bool(self.keywords)}")

    def _load_embedding_model(self):
        """Lazy load embedding model for Stage 1"""
        if self.embedding_model is not None:
            return  # Already loaded

        logger.info(f"Loading embedding model: {self.config.embedding_model}")
        device = 'cpu'  # Keep model on CPU

        # For Qwen models, we need trust_remote_code=True and truncate_dim for 4B model
        if 'Qwen' in self.config.embedding_model:
            if 'Embedding-4B' in self.config.embedding_model:
                self.embedding_model = SentenceTransformer(
                    self.config.embedding_model,
                    trust_remote_code=True,
                    truncate_dim=2000,
                    device=device
                )
                logger.info(f"Using Qwen3-4B embeddings with truncate_dim=2000 on {device.upper()}")
            else:
                self.embedding_model = SentenceTransformer(
                    self.config.embedding_model,
                    trust_remote_code=True,
                    device=device
                )
                logger.info(f"Using Qwen embeddings on {device.upper()}")
        else:
            self.embedding_model = SentenceTransformer(self.config.embedding_model, device=device)
            logger.info(f"Using embedding model on {device.upper()}")

    def _load_faiss_index(self):
        """Lazy load FAISS index for Stage 1"""
        if self.faiss_loader is not None:
            return  # Already loaded

        if not self.config.use_faiss:
            return  # Not using FAISS

        if not self.config.project:
            raise ValueError("Project must be specified when using FAISS mode")

        # In test mode, only load 10x the test candidates to build index
        faiss_limit = None
        if self.config.test_mode:
            faiss_limit = self.config.test_mode * 10
            logger.info(f"TEST MODE: Loading only {faiss_limit} embeddings for FAISS index")

        logger.info(f"Initializing FAISS index for project: {self.config.project}")
        self.faiss_loader = FAISSIndexLoader(
            project=self.config.project,
            embedding_dim=2000,
            limit=faiss_limit
        )
        self.faiss_loader.load_or_build_index(self.config.faiss_index_path)
        logger.info(f"FAISS index ready with {self.faiss_loader.index.ntotal} vectors")

    def _unload_stage1_models(self):
        """Unload embedding model and FAISS index to free memory after Stage 1"""
        logger.info("Unloading Stage 1 models (embedding model + FAISS index) to free memory")

        if self.embedding_model is not None:
            del self.embedding_model
            self.embedding_model = None
            logger.info("  ✓ Embedding model unloaded")

        if self.faiss_loader is not None:
            del self.faiss_loader
            self.faiss_loader = None
            logger.info("  ✓ FAISS index unloaded")

        # Force garbage collection to free memory
        import gc
        gc.collect()
        logger.info("  ✓ Memory freed")

    def _load_llm_classifier(self):
        """Lazy load LLM classifier for Stages 2-4"""
        if self.classifier is not None:
            return  # Already loaded

        logger.info("Loading MLX LLM classifier for theme/subtheme classification")
        self.classifier = LLMClassifier(
            themes_csv=self.config.subthemes_csv,
            model_name=self.config.model_name,
            use_gpu=self.config.use_gpu,
            project=self.config.project,
            model_server_url=self.config.model_server_url,
            use_model_server=self.config.use_model_server
        )
        logger.info("MLX LLM classifier loaded successfully")
    
    def _load_theme_descriptions(self, csv_path: str) -> Dict[str, List[str]]:
        """Load theme descriptions from CSV"""
        df = pd.read_csv(csv_path)
        
        unique_descriptions_en = df['theme_description'].unique().tolist()
        
        unique_descriptions_fr = []
        if 'theme_description_fr' in df.columns:
            unique_descriptions_fr = df['theme_description_fr'].unique().tolist()
        else:
            unique_descriptions_fr = unique_descriptions_en
        
        return {
            'en': unique_descriptions_en,
            'fr': unique_descriptions_fr
        }
    
    def _load_keywords(self, keywords_file: str) -> Dict[str, List[str]]:
        """Load keywords from file
        
        Expected format (one per line):
        keyword1
        keyword2
        keyword3
        
        Or with theme associations:
        1:immigration,immigrant,refugee
        2:climate,warming,carbon
        """
        keywords = {'global': []}
        
        if not keywords_file or not Path(keywords_file).exists():
            return keywords
        
        with open(keywords_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                if ':' in line:
                    # Theme-specific keywords
                    theme_part, keywords_part = line.split(':', 1)
                    theme_id = theme_part.strip()
                    theme_keywords = [k.strip() for k in keywords_part.split(',')]
                    keywords[theme_id] = theme_keywords
                else:
                    # Global keywords
                    keywords['global'].append(line)
        
        logger.info(f"Loaded {len(keywords['global'])} global keywords")
        theme_specific = len([k for k in keywords if k != 'global'])
        if theme_specific > 0:
            logger.info(f"Loaded keywords for {theme_specific} specific themes")
        
        return keywords
    
    
    def _find_matching_keywords(self, text: str, keywords: List[str]) -> List[str]:
        """Find which keywords match in the text"""
        matched = []
        text_lower = text.lower()
        for keyword in keywords:
            if keyword.lower() in text_lower:
                matched.append(keyword)
        return matched
    
    def _format_speaker_attributed_text(self, segment_text: str, transcription_texts: List[str], speaker_ids: List[int]) -> str:
        """Format segment text with anonymous speaker attribution.
        
        Args:
            segment_text: The original segment text
            transcription_texts: List of transcription texts in order
            speaker_ids: List of corresponding speaker IDs from database
            
        Returns:
            Text formatted as "SPEAKER_00: text... SPEAKER_01: text..."
        """
        if not transcription_texts or not speaker_ids:
            return segment_text
        
        if len(transcription_texts) != len(speaker_ids):
            logger.warning(f"Mismatch between transcription_texts ({len(transcription_texts)}) and speaker_ids ({len(speaker_ids)})")
            return segment_text
        
        # Create mapping of speaker_id to anonymous labels (SPEAKER_00, SPEAKER_01, etc.)
        unique_speakers = []
        speaker_map = {}
        for sid in speaker_ids:
            if sid is not None and sid not in speaker_map:
                speaker_map[sid] = f"SPEAKER_{len(unique_speakers):02d}"
                unique_speakers.append(sid)
        
        # Build speaker-attributed text
        attributed_parts = []
        current_pos = 0
        
        for trans_text, speaker_id in zip(transcription_texts, speaker_ids):
            if trans_text and speaker_id is not None:
                # Get anonymous speaker label
                speaker_label = speaker_map.get(speaker_id, f"SPEAKER_{speaker_id}")
                
                # Find where this transcription text appears in the segment
                trans_pos = segment_text.find(trans_text, current_pos)
                
                if trans_pos >= 0:
                    # Direct match found
                    attributed_parts.append(f"{speaker_label}: {trans_text}")
                    current_pos = trans_pos + len(trans_text)
                else:
                    # Try partial match (first 50 chars)
                    sample = trans_text[:50] if len(trans_text) > 50 else trans_text
                    sample_pos = segment_text.find(sample, current_pos)
                    
                    if sample_pos >= 0:
                        # Find the end of this speaker's portion
                        # Look for the next transcription's beginning
                        next_idx = transcription_texts.index(trans_text) + 1
                        if next_idx < len(transcription_texts) and transcription_texts[next_idx]:
                            next_sample = transcription_texts[next_idx][:50]
                            end_pos = segment_text.find(next_sample, sample_pos)
                            if end_pos > sample_pos:
                                text_portion = segment_text[sample_pos:end_pos].strip()
                            else:
                                text_portion = segment_text[sample_pos:].strip()
                        else:
                            # This is the last transcription
                            text_portion = segment_text[sample_pos:].strip()
                        
                        attributed_parts.append(f"{speaker_label}: {text_portion}")
                        current_pos = sample_pos + len(text_portion)
        
        # If we have speaker attribution, use it; otherwise return original
        if attributed_parts:
            return " ".join(attributed_parts)
        else:
            # Fallback: if no matches found but we have speaker data, 
            # attribute the whole segment to speakers in sequence
            if len(set(speaker_ids)) == 1 and speaker_ids[0] is not None:
                # Single speaker for entire segment
                speaker_label = speaker_map.get(speaker_ids[0], f"SPEAKER_{speaker_ids[0]}")
                return f"{speaker_label}: {segment_text}"
            else:
                # Multiple speakers but couldn't match text - return original
                return segment_text
    
    
    def _load_checkpoint(self) -> Tuple[Set[int], List[SearchCandidate]]:
        """Load already processed segment IDs and candidates from checkpoint"""
        if not self.config.resume or not self.checkpoint_path.exists():
            return set(), []
        
        try:
            with open(self.checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            processed_ids = set(checkpoint_data.get('processed_segment_ids', []))
            
            # Load previously saved candidates if available
            candidates = []
            intermediate_csv = str(self.output_path).replace('.csv', '_checkpoint.csv')
            if Path(intermediate_csv).exists():
                df = pd.read_csv(intermediate_csv)
                logger.info(f"Loaded {len(df)} candidates from checkpoint CSV")
            
            timestamp = checkpoint_data.get('timestamp', 'unknown')
            logger.info(f"Loaded checkpoint from {timestamp}")
            logger.info(f"  Segments processed: {len(processed_ids)}")
            
            return processed_ids, candidates
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}")
            return set(), []
    
    def _save_checkpoint(self, processed_ids: Set[int]):
        """Save simple checkpoint with processed segment IDs"""
        try:
            checkpoint_data = {
                'processed_segment_ids': list(processed_ids),
                'timestamp': datetime.now().isoformat(),
                'total_processed': len(processed_ids)
            }
            
            with open(self.checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            logger.debug(f"Saved checkpoint: {len(processed_ids)} segments processed")
            
        except Exception as e:
            logger.error(f"Could not save checkpoint: {e}")
    
    def _save_candidates_to_csv(self, candidates: List[SearchCandidate], output_path: str, append: bool = False):
        """Save candidates to CSV file (append mode for incremental saves)"""
        mode = 'a' if append else 'w'
        write_header = not append or not Path(output_path).exists()

        # Primary output path
        with open(output_path, mode, newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            if write_header:
                header = [
                    'segment_id', 'content_id', 'episode_title', 'episode_channel',
                    'start_time', 'end_time', 'themes', 'confidence_scores', 'high_confidence_themes', 
                    'matched_via', 'matched_keywords', 'similarity_score', 
                    'segment_text', 'speaker_attributed_text', 'original_url'
                ]
                writer.writerow(header)
            
            for candidate in candidates:
                # Format themes and confidence scores
                theme_list = []
                confidence_list = []
                high_confidence_list = []  # Themes with confidence >= 0.75
                
                if candidate.theme_ids:
                    for theme_id in candidate.theme_ids:
                        # Check validation results
                        validation_results = getattr(candidate, 'validation_results', {}) or {}
                        validation = validation_results.get(theme_id, {})
                        
                        if candidate.subtheme_results and theme_id in candidate.subtheme_results:
                            # Handle both list (batch processing) and dict (old format) structures
                            subtheme_data = candidate.subtheme_results[theme_id]
                            if isinstance(subtheme_data, list):
                                subtheme_ids = subtheme_data
                            else:
                                subtheme_ids = subtheme_data.get('subtheme_ids', [])
                            if subtheme_ids:
                                for sub_id in subtheme_ids:
                                    # sub_id already contains the full ID like "3C", "5F"
                                    theme_list.append(str(sub_id))
                                    
                                    # Get confidence score
                                    conf = validation.get('subthemes_confidence', {}).get(sub_id)
                                    if conf is not None:
                                        # Ensure sub_id is formatted correctly (e.g., "1F", "2B")
                                        confidence_list.append(f"{sub_id}:{conf:.2f}")
                                        if conf >= 0.75:
                                            high_confidence_list.append(str(sub_id))
                            else:
                                theme_list.append(str(theme_id))
                                # Get theme confidence
                                conf = validation.get('theme_confidence')
                                if conf is not None:
                                    confidence_list.append(f"{theme_id}:{conf:.2f}")
                                    if conf >= 0.75:
                                        high_confidence_list.append(str(theme_id))
                        else:
                            theme_list.append(str(theme_id))
                            # Get theme confidence
                            conf = validation.get('theme_confidence')
                            if conf is not None:
                                confidence_list.append(f"{theme_id}:{conf:.2f}")
                                if conf >= 0.75:
                                    high_confidence_list.append(str(theme_id))
                
                themes_str = ', '.join(theme_list) if theme_list else ""
                # Sort confidence list by score (highest first) for better readability
                confidence_list_sorted = sorted(confidence_list, key=lambda x: float(x.split(':')[1]), reverse=True) if confidence_list else []
                confidence_str = ', '.join(confidence_list_sorted) if confidence_list_sorted else ""
                high_conf_str = ', '.join(high_confidence_list) if high_confidence_list else ""
                keywords_str = ', '.join(candidate.matched_keywords) if candidate.matched_keywords else ""
                
                row = [
                    candidate.segment_id,
                    candidate.content_id,
                    candidate.episode_title,
                    candidate.episode_channel,
                    candidate.start_time,
                    candidate.end_time,
                    themes_str,
                    confidence_str,  # e.g., "1:0.85, 2B:0.92, 3C:0.45"
                    high_conf_str,  # Only themes with confidence >= 0.75
                    candidate.matched_via,
                    keywords_str,
                    candidate.similarity_score,
                    candidate.segment_text,
                    candidate.speaker_attributed_text or candidate.segment_text,  # Include speaker-attributed text
                    candidate.original_url or ""  # Include original URL
                ]
                writer.writerow(row)

        # Backup to Desktop (if exists)
        desktop_backup = Path.home() / "Desktop" / Path(output_path).name
        if desktop_backup.parent.exists():
            try:
                with open(desktop_backup, mode, newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)

                    if write_header:
                        header = [
                            'segment_id', 'content_id', 'episode_title', 'episode_channel',
                            'start_time', 'end_time', 'themes', 'confidence_scores', 'high_confidence_themes',
                            'matched_via', 'matched_keywords', 'similarity_score',
                            'segment_text', 'speaker_attributed_text', 'original_url'
                        ]
                        writer.writerow(header)

                    for candidate in candidates:
                        # Format themes and confidence scores (same as primary)
                        theme_list = []
                        confidence_list = []
                        high_confidence_list = []

                        if candidate.theme_ids:
                            for theme_id in candidate.theme_ids:
                                validation_results = getattr(candidate, 'validation_results', {}) or {}
                                validation = validation_results.get(theme_id, {})

                                if validation and validation.get('is_valid', False):
                                    theme_list.append(str(theme_id))
                                    conf = validation.get('confidence', 0.0)
                                    confidence_list.append(f"{theme_id}:{conf:.2f}")

                                    if conf >= 0.75:
                                        high_confidence_list.append(str(theme_id))

                        themes_str = ','.join(theme_list) if theme_list else ''
                        confidence_str = ','.join(confidence_list) if confidence_list else ''
                        high_confidence_str = ','.join(high_confidence_list) if high_confidence_list else ''
                        keywords_str = ','.join(candidate.matched_keywords) if candidate.matched_keywords else ''

                        row = [
                            candidate.segment_id,
                            candidate.content_id,
                            candidate.episode_title,
                            candidate.episode_channel,
                            candidate.start_time,
                            candidate.end_time,
                            themes_str,
                            confidence_str,
                            high_confidence_str,
                            candidate.matched_via,
                            keywords_str,
                            candidate.similarity_score,
                            candidate.segment_text,
                            candidate.speaker_attributed_text or candidate.segment_text,
                            candidate.original_url or ""
                        ]
                        writer.writerow(row)
                logger.debug(f"Backup saved to {desktop_backup}")
            except Exception as e:
                logger.warning(f"Could not save desktop backup: {e}")

    def stage1_search(self) -> List[SearchCandidate]:
        """Stage 1: Keyword + Semantic search for all themes"""
        if self.config.use_faiss:
            return self._stage1_search_faiss()
        else:
            return self._stage1_search_unified()

    def _stage1_search_faiss(self) -> List[SearchCandidate]:
        """FAISS-based search: keywords first (PostgreSQL) → FAISS semantic → gray zone refinement"""
        if self.config.test_mode:
            logger.info(f"Stage 1: TEST MODE - FAISS search targeting {self.config.test_mode} candidates")
        else:
            logger.info("Stage 1: PRODUCTION MODE - FAISS search for all matching content")

        # Load already processed segments
        processed_segment_ids, _ = self._load_checkpoint()

        # Store FAISS results for later use
        faiss_results_map = {}  # segment_id -> (similarity, theme_desc)

        # ================================================================
        # STEP 1: Fast keyword search in PostgreSQL (GIN index)
        # ================================================================
        keyword_segment_ids = set()

        all_keywords = []
        for theme_keywords in self.keywords.values():
            if isinstance(theme_keywords, list):
                all_keywords.extend(theme_keywords)
        all_keywords = list(set(all_keywords))

        if all_keywords:
            logger.info(f"Step 1: Keyword search with {len(all_keywords)} unique keywords...")

            with get_session() as session:
                # Build keyword conditions
                keyword_conditions = []
                for keyword in all_keywords:
                    escaped_keyword = keyword.replace("'", "''")
                    keyword_conditions.append(f"es.text ILIKE '%{escaped_keyword}%'")

                keyword_where = " OR ".join(keyword_conditions)

                # Exclusion for already processed segments
                exclude_condition = ""
                if processed_segment_ids:
                    exclude_ids_str = ','.join(map(str, processed_segment_ids))
                    exclude_condition = f"AND es.id NOT IN ({exclude_ids_str})"

                # Project filter
                project_condition = ""
                if self.config.project:
                    project_condition = f"AND '{self.config.project}' = ANY(c.projects)"

                # Test mode limit
                limit_clause = ""
                if self.config.test_mode:
                    keyword_limit = self.config.test_mode * 10
                    limit_clause = f"LIMIT {keyword_limit}"
                    logger.info(f"  TEST MODE: Limiting keyword search to {keyword_limit} results")

                # Query for keyword segment IDs only
                keyword_query = text(f"""
                    SELECT es.id
                    FROM embedding_segments es
                    JOIN content c ON es.content_id = c.id
                    WHERE ({keyword_where})
                    AND es.embedding_alt IS NOT NULL
                    {exclude_condition}
                    {project_condition}
                    ORDER BY es.id
                    {limit_clause}
                """)

                result = session.execute(keyword_query)
                keyword_segment_ids = {row[0] for row in result}

                logger.info(f"  Found {len(keyword_segment_ids)} keyword matches")

        # ================================================================
        # STEP 2: FAISS semantic search
        # ================================================================
        logger.info("Step 2: FAISS semantic search across all themes...")

        # Track exclusions (use sets to avoid counting duplicates across queries)
        faiss_excluded_segments = set()
        faiss_total_above_threshold = 0

        # Encode theme descriptions
        theme_embeddings_en = []
        theme_embeddings_fr = []

        for desc in self.theme_descriptions['en']:
            query_text = f"Instruct: Retrieve relevant passages.\nQuery: {desc}"
            embedding = self.embedding_model.encode(
                query_text,
                convert_to_numpy=True,
                normalize_embeddings=False  # FAISS will normalize
            )
            theme_embeddings_en.append(embedding)

        for desc in self.theme_descriptions['fr']:
            query_text = f"Instruct: Retrieve relevant passages.\nQuery: {desc}"
            embedding = self.embedding_model.encode(
                query_text,
                convert_to_numpy=True,
                normalize_embeddings=False  # FAISS will normalize
            )
            theme_embeddings_fr.append(embedding)

        # Run FAISS queries for semantic search (use normal threshold)
        normal_threshold = self.config.similarity_threshold
        # Reduce k in test mode for faster testing
        search_k = 100000 if not self.config.test_mode else min(200, self.config.test_mode * 2)

        logger.info(f"  Searching with threshold={normal_threshold}, k={search_k}")

        query_num = 0
        for theme_idx in range(min(9, len(theme_embeddings_en))):
            for lang_idx, (embeddings, lang_name) in enumerate([
                (theme_embeddings_en, 'EN'),
                (theme_embeddings_fr, 'FR')
            ]):
                query_num += 1
                embedding = embeddings[theme_idx]
                desc = self.theme_descriptions['en'][theme_idx]

                try:
                    logger.debug(f"  Query {query_num}/18: Theme {theme_idx+1} ({lang_name}), embedding shape: {embedding.shape}")

                    # FAISS search
                    query_reshaped = embedding.reshape(1, -1).astype(np.float32)
                    logger.debug(f"    Query reshaped: {query_reshaped.shape}, searching k={search_k}")

                    scores, indices = self.faiss_loader.search(query_reshaped, k=search_k)

                    logger.debug(f"    Search successful, got {len(scores[0])} results")

                    for score, idx in zip(scores[0], indices[0]):
                        if score >= normal_threshold and idx >= 0:
                            # Check if idx is within valid range
                            if idx >= len(self.faiss_loader.segment_id_mapping):
                                logger.warning(f"FAISS returned out-of-bounds index {idx} (max: {len(self.faiss_loader.segment_id_mapping)-1}), skipping")
                                continue
                            faiss_total_above_threshold += 1
                            segment_id = self.faiss_loader.segment_id_mapping.get(int(idx))
                            if segment_id:
                                if segment_id in processed_segment_ids:
                                    faiss_excluded_segments.add(segment_id)
                                else:
                                    # Keep highest score per segment
                                    if segment_id not in faiss_results_map or score > faiss_results_map[segment_id][0]:
                                        faiss_results_map[segment_id] = (float(score), desc)
                except Exception as e:
                    logger.error(f"  Query {query_num}/18 FAILED: Theme {theme_idx+1} ({lang_name})")
                    logger.error(f"    Error: {e}")
                    logger.error(f"    Embedding shape: {embedding.shape}")
                    raise

        logger.info(f"  FAISS semantic search found {len(faiss_results_map)} NEW segments above normal threshold ({normal_threshold})")
        if len(faiss_excluded_segments) > 0:
            logger.info(f"  Excluded {len(faiss_excluded_segments)} unique already-processed segments (total matches including duplicates: {faiss_total_above_threshold})")

        # Store semantic matches (all are above normal threshold)
        semantic_match_ids = set(faiss_results_map.keys())

        # ================================================================
        # STEP 3: Check keyword matches against FAISS with relaxed threshold
        # ================================================================
        logger.info(f"Step 3: Checking {len(keyword_segment_ids)} keyword matches against FAISS with relaxed threshold...")

        relaxed_threshold = self.config.similarity_threshold - self.config.keyword_threshold_reduction
        logger.info(f"  Using relaxed threshold: {relaxed_threshold}")

        keyword_validated_ids = set()

        # For each keyword match, look up its best score in FAISS
        if keyword_segment_ids:
            # Load FAISS if not already loaded (needed for keyword validation)
            if self.faiss_loader is None:
                logger.info("  Loading FAISS index for keyword validation...")
                self._load_faiss_index()

            with get_session() as session:
                # Get embeddings for keyword segments that aren't already semantic matches
                keyword_only_ids = keyword_segment_ids - semantic_match_ids

                if keyword_only_ids:
                    # Check each keyword match's similarity to all theme descriptions
                    for segment_id in keyword_only_ids:
                        # Get the segment's embedding from FAISS index
                        if segment_id in self.faiss_loader.reverse_mapping:
                            idx = self.faiss_loader.reverse_mapping[segment_id]
                            segment_embedding = self.faiss_loader.embeddings[idx].reshape(1, -1)

                            # Check similarity against all theme embeddings
                            best_score = 0.0
                            best_theme = None

                            for theme_idx, theme_embedding in enumerate(theme_embeddings_en + theme_embeddings_fr):
                                # Normalize and compute inner product (cosine similarity for normalized vectors)
                                theme_emb_normalized = theme_embedding / np.linalg.norm(theme_embedding)
                                seg_emb_normalized = segment_embedding[0] / np.linalg.norm(segment_embedding[0])
                                score = float(np.dot(seg_emb_normalized, theme_emb_normalized))

                                if score > best_score:
                                    best_score = score
                                    best_theme = self.theme_descriptions['en'][theme_idx % len(self.theme_descriptions['en'])]

                            # If above relaxed threshold, include it
                            if best_score >= relaxed_threshold:
                                keyword_validated_ids.add(segment_id)
                                # Store in faiss_results_map for later use
                                if segment_id not in faiss_results_map or best_score > faiss_results_map[segment_id][0]:
                                    faiss_results_map[segment_id] = (best_score, best_theme)

        logger.info(f"  {len(keyword_validated_ids)} keyword matches validated above relaxed threshold")

        # ================================================================
        # STEP 4: Combine all candidates
        # ================================================================
        # Include semantic matches (≥0.6) + validated keyword matches (≥0.54)
        final_candidate_ids = semantic_match_ids | keyword_validated_ids

        keyword_only_count = len(keyword_validated_ids - semantic_match_ids)
        semantic_only_count = len(semantic_match_ids - keyword_validated_ids)
        both_count = len(keyword_validated_ids & semantic_match_ids)

        logger.info(f"Step 4: Combining candidates:")
        logger.info(f"  {semantic_only_count} semantic-only matches (≥{normal_threshold})")
        logger.info(f"  {keyword_only_count} keyword-only matches (≥{relaxed_threshold})")
        logger.info(f"  {both_count} matches with both keyword and semantic")

        if self.config.test_mode and len(final_candidate_ids) > self.config.test_mode:
            # Sort by similarity and take top N
            sorted_candidates = sorted(
                final_candidate_ids,
                key=lambda sid: faiss_results_map.get(sid, (0.0, ''))[0],
                reverse=True
            )
            final_candidate_ids = set(sorted_candidates[:self.config.test_mode])
            logger.info(f"  TEST MODE: Limited to {len(final_candidate_ids)} candidates")

        # ================================================================
        # STEP 5: Single batch fetch for ALL metadata
        # ================================================================
        logger.info(f"Step 4: Fetching metadata for {len(final_candidate_ids)} candidates...")

        candidates = []

        if final_candidate_ids:
            with get_session() as session:
                query = text("""
                    SELECT
                        es.id as segment_id,
                        es.content_id,
                        es.text,
                        es.start_time,
                        es.end_time,
                        es.segment_index,
                        c.title as episode_title,
                        c.channel_name as episode_channel,
                        es.source_sentence_ids,
                        array_agg(s.text ORDER BY array_position(es.source_sentence_ids, s.sentence_index)) as sentence_texts,
                        array_agg(s.speaker_id ORDER BY array_position(es.source_sentence_ids, s.sentence_index)) as speaker_ids,
                        c.meta_data->>'webpage_url' as original_url
                    FROM embedding_segments es
                    JOIN content c ON es.content_id = c.id
                    LEFT JOIN sentences s ON s.sentence_index = ANY(es.source_sentence_ids) AND s.content_id = es.content_id
                    WHERE es.id = ANY(:segment_ids)
                    GROUP BY es.id, es.content_id, es.text, es.start_time, es.end_time,
                             es.segment_index, es.source_sentence_ids,
                             c.title, c.channel_name, c.meta_data->>'webpage_url'
                """)

                result = session.execute(query, {'segment_ids': list(final_candidate_ids)})

                for row in result:
                    segment_id = row[0]
                    similarity, theme_desc = faiss_results_map.get(segment_id, (0.0, ''))

                    # Determine matched_via
                    is_semantic = segment_id in semantic_match_ids
                    is_keyword = segment_id in keyword_validated_ids

                    if is_semantic and is_keyword:
                        matched_via = 'both'
                        matched_keywords = self._find_matching_keywords(row[2], all_keywords)
                    elif is_keyword:
                        matched_via = 'keyword'
                        matched_keywords = self._find_matching_keywords(row[2], all_keywords)
                    else:
                        matched_via = 'semantic'
                        matched_keywords = []

                    # Format speaker attribution
                    sentence_texts = row[9] if row[9] else []
                    speaker_ids = row[10] if row[10] else []
                    speaker_attributed_text = self._format_speaker_attributed_text(
                        row[2], sentence_texts, speaker_ids
                    )

                    candidate = SearchCandidate(
                        segment_id=segment_id,
                        content_id=row[1],
                        segment_text=row[2],
                        start_time=row[3],
                        end_time=row[4],
                        segment_index=row[5],
                        episode_title=row[6] or "Unknown",
                        episode_channel=row[7] or "Unknown",
                        similarity_score=similarity,
                        matched_via=matched_via,
                        matched_keywords=matched_keywords,
                        matching_themes=[theme_desc] if theme_desc else [],
                        source_sentence_ids=row[8],
                        sentence_texts=sentence_texts,
                        speaker_ids=speaker_ids,
                        speaker_attributed_text=speaker_attributed_text,
                        original_url=row[11]
                    )
                    candidates.append(candidate)

        # Sort by similarity
        candidates.sort(key=lambda c: c.similarity_score, reverse=True)

        logger.info(f"FAISS search complete: {len(candidates)} total candidates")
        if candidates:
            keyword_count = len([c for c in candidates if c.matched_via == 'keyword'])
            semantic_count = len([c for c in candidates if c.matched_via == 'semantic'])
            logger.info(f"  {keyword_count} via keywords, {semantic_count} via semantic")
            logger.info(f"  Similarity range: {candidates[0].similarity_score:.3f} - {candidates[-1].similarity_score:.3f}")

        return candidates

    def _stage1_search_unified(self) -> List[SearchCandidate]:
        """Unified search: keywords first, then semantic across all themes"""
        if self.config.test_mode:
            logger.info(f"Stage 1: TEST MODE - Unified search targeting {self.config.test_mode} candidates")
            total_target = self.config.test_mode
            keyword_target = total_target // 2
            semantic_target = total_target - keyword_target
            logger.info(f"Target distribution: {keyword_target} keywords, {semantic_target} semantic")
        else:
            logger.info("Stage 1: PRODUCTION MODE - Finding all matching content")
            # In production mode, no limits
            keyword_target = None  # No limit
            semantic_target = None  # No limit
        
        # Load already processed segments
        processed_segment_ids, _ = self._load_checkpoint()
        
        all_candidates = {}
        
        # 1. Keyword search - get ALL keywords and search for matches
        all_keywords = []
        for theme_keywords in self.keywords.values():
            if isinstance(theme_keywords, list):
                all_keywords.extend(theme_keywords)
        all_keywords = list(set(all_keywords))  # Remove duplicates
        
        if all_keywords:
            logger.info(f"Searching with {len(all_keywords)} unique keywords...")
            logger.debug(f"Keywords: {all_keywords}")
            
            with get_session() as session:
                # Use all keywords for comprehensive search
                keywords_to_use = all_keywords
                logger.info(f"Using all {len(keywords_to_use)} keywords for SQL query")
                
                keyword_conditions = []
                for keyword in keywords_to_use:
                    # Escape single quotes in keywords for SQL
                    escaped_keyword = keyword.replace("'", "''")
                    keyword_conditions.append(f"es.text ILIKE '%{escaped_keyword}%'")
                
                keyword_where = " OR ".join(keyword_conditions)
                
                # Add exclusion for already processed segments
                exclude_condition = ""
                if processed_segment_ids:
                    exclude_ids_str = ','.join(map(str, processed_segment_ids))
                    exclude_condition = f"AND es.id NOT IN ({exclude_ids_str})"
                
                # Project filter
                project_condition = ""
                if self.config.project:
                    project_condition = f"AND '{self.config.project}' = ANY(c.projects)"
                
                # Build limit clause based on mode
                if keyword_target is None:
                    # Production mode: no limit
                    limit_clause = ""
                    logger.info(f"Executing keyword query with NO LIMIT (production mode)")
                else:
                    # Test mode: use limit
                    limit_clause = f"LIMIT {keyword_target * 2}"
                    logger.info(f"Executing keyword query with limit {keyword_target * 2} (test mode)")
                
                # Single query with optional limit
                keyword_query = text(f"""
                    SELECT
                        es.id as segment_id,
                        es.content_id,
                        es.text,
                        es.start_time,
                        es.end_time,
                        es.segment_index,
                        c.title as episode_title,
                        c.channel_name as episode_channel,
                        es.embedding_alt,
                        es.source_sentence_ids,
                        array_agg(s.text ORDER BY array_position(es.source_sentence_ids, s.sentence_index)) as sentence_texts,
                        array_agg(s.speaker_id ORDER BY array_position(es.source_sentence_ids, s.sentence_index)) as speaker_ids,
                        c.meta_data->>'webpage_url' as original_url
                    FROM embedding_segments es
                    JOIN content c ON es.content_id = c.id
                    LEFT JOIN sentences s ON s.sentence_index = ANY(es.source_sentence_ids) AND s.content_id = es.content_id
                    WHERE ({keyword_where})
                    AND es.embedding_alt IS NOT NULL
                    {exclude_condition}
                    {project_condition}
                    GROUP BY es.id, es.content_id, es.text, es.start_time, es.end_time,
                             es.segment_index, es.source_sentence_ids,
                             c.title, c.channel_name, es.embedding_alt, c.meta_data->>'webpage_url'
                    {limit_clause}
                """)
                
                result = session.execute(keyword_query)
                
                keyword_rows = result.fetchall()
                
                logger.info(f"Found {len(keyword_rows)} keyword matches (before similarity filtering)")
                
                # Now check similarity for keyword matches
                if keyword_rows:
                    # Encode theme descriptions
                    logger.info("Encoding theme descriptions for similarity checking...")
                    theme_embeddings_en = []
                    theme_embeddings_fr = []
                    
                    for desc in self.theme_descriptions['en']:
                        # For Qwen models, use "Instruct: " prefix for query
                        query_text = f"Instruct: Retrieve relevant passages.\nQuery: {desc}" if 'Qwen' in self.config.embedding_model else f"query: {desc}"
                        embedding = self.embedding_model.encode(
                            query_text,
                            convert_to_numpy=True,
                            normalize_embeddings=True
                        )
                        theme_embeddings_en.append(embedding)
                    
                    for desc in self.theme_descriptions['fr']:
                        # For Qwen models, use "Instruct: " prefix for query
                        query_text = f"Instruct: Retrieve relevant passages.\nQuery: {desc}" if 'Qwen' in self.config.embedding_model else f"query: {desc}"
                        embedding = self.embedding_model.encode(
                            query_text,
                            convert_to_numpy=True,
                            normalize_embeddings=True
                        )
                        theme_embeddings_fr.append(embedding)
                    
                    # Check similarity for each keyword match
                    for row in keyword_rows:
                        # Only apply limit in test mode
                        if keyword_target is not None and len(all_candidates) >= keyword_target:
                            break
                        
                        # Parse embedding_alt (4B model with 2000 dimensions)
                        if row[8]:
                            if isinstance(row[8], str):
                                embedding_str = row[8].strip('[]')
                                segment_embedding = np.array([float(x) for x in embedding_str.split(',')])
                            else:
                                segment_embedding = np.array(row[8])
                            
                            # Check similarity against all theme embeddings
                            best_similarity = 0.0
                            matching_themes = []
                            
                            for idx, (desc_en, emb_en, emb_fr) in enumerate(zip(
                                self.theme_descriptions['en'], 
                                theme_embeddings_en, 
                                theme_embeddings_fr
                            )):
                                sim_en = np.dot(segment_embedding, emb_en)
                                sim_fr = np.dot(segment_embedding, emb_fr)
                                theme_similarity = max(sim_en, sim_fr)
                                
                                # Use relaxed threshold for keyword matches (reduced by 0.03)
                                if theme_similarity >= self.config.similarity_threshold - self.config.keyword_threshold_reduction:
                                    matching_themes.append(desc_en)
                                    best_similarity = max(best_similarity, theme_similarity)
                            
                            # Only add if meets threshold
                            if matching_themes:
                                # Extract speaker data and format attribution
                                sentence_texts = row[10] if row[10] else []
                                speaker_ids = row[11] if row[11] else []
                                speaker_attributed_text = self._format_speaker_attributed_text(
                                    row[2], sentence_texts, speaker_ids
                                )

                                candidate = SearchCandidate(
                                    segment_id=row[0],
                                    content_id=row[1],
                                    segment_text=row[2],
                                    start_time=row[3],
                                    end_time=row[4],
                                    segment_index=row[5],
                                    episode_title=row[6] or "Unknown",
                                    episode_channel=row[7] or "Unknown",
                                    similarity_score=best_similarity,
                                    matched_via='keyword',
                                    matched_keywords=self._find_matching_keywords(row[2], keywords_to_use),
                                    matching_themes=matching_themes,
                                    source_sentence_ids=row[9],
                                    sentence_texts=sentence_texts,
                                    speaker_ids=speaker_ids,
                                    speaker_attributed_text=speaker_attributed_text,
                                    original_url=row[12]  # Original URL from meta_data
                                )
                                all_candidates[candidate.segment_id] = candidate
                
                logger.info(f"Kept {len(all_candidates)} keyword matches above similarity threshold")
        
        # 2. Semantic search - 18 separate queries (9 themes x 2 languages)
        # In production mode, we still want semantic search even with no target limit
        semantic_needed = semantic_target if semantic_target is not None else -1  # -1 means unlimited
        
        if semantic_needed != 0:  # Proceed if we need any semantic matches (including unlimited)
            if semantic_target is not None:
                logger.info(f"Adding {semantic_needed} semantic matches via 18 queries (9 themes × 2 languages)...")
            else:
                logger.info("Adding ALL semantic matches above threshold via 18 queries (9 themes × 2 languages)...")
            
            # Encode theme descriptions if not already done
            if 'theme_embeddings_en' not in locals():
                logger.info("Encoding theme descriptions...")
                theme_embeddings_en = []
                theme_embeddings_fr = []
                
                for desc in self.theme_descriptions['en']:
                    # For Qwen models, use "Instruct: " prefix for query
                    query_text = f"Instruct: Retrieve relevant passages.\nQuery: {desc}" if 'Qwen' in self.config.embedding_model else f"query: {desc}"
                    embedding = self.embedding_model.encode(
                        query_text,
                        convert_to_numpy=True,
                        normalize_embeddings=True
                    )
                    theme_embeddings_en.append(embedding)
                
                for desc in self.theme_descriptions['fr']:
                    # For Qwen models, use "Instruct: " prefix for query
                    query_text = f"Instruct: Retrieve relevant passages.\nQuery: {desc}" if 'Qwen' in self.config.embedding_model else f"query: {desc}"
                    embedding = self.embedding_model.encode(
                        query_text,
                        convert_to_numpy=True,
                        normalize_embeddings=True
                    )
                    theme_embeddings_fr.append(embedding)
            
            # Calculate per-query limit
            if semantic_target is not None:
                # Test mode: calculate based on target
                per_query_limit = max(10, (semantic_needed // 18) + 50)
                logger.info(f"Running 18 semantic queries with limit {per_query_limit} each")
            else:
                # Production mode: NO LIMIT
                per_query_limit = None
                logger.info(f"Running 18 semantic queries with NO LIMIT (production mode)")
            
            # Collect all exclude IDs once
            all_exclude_ids = set(processed_segment_ids) if processed_segment_ids else set()
            all_exclude_ids.update(all_candidates.keys())
            
            with get_session() as session:
                # Build base conditions once
                project_condition = ""
                if self.config.project:
                    project_condition = f"AND '{self.config.project}' = ANY(c.projects)"
                
                # Track how many we've added
                semantic_added = 0
                query_count = 0
                
                # Run 18 queries (9 themes x 2 language embeddings)
                for theme_idx in range(min(9, len(theme_embeddings_en))):
                    # Only check limit in test mode
                    if semantic_target is not None and semantic_added >= semantic_target:
                        break
                    
                    for lang_idx, (embeddings, lang_name) in enumerate([
                        (theme_embeddings_en, 'EN'),
                        (theme_embeddings_fr, 'FR')
                    ]):
                        # Only check limit in test mode
                        if semantic_target is not None and semantic_added >= semantic_target:
                            break
                        
                        query_count += 1
                        embedding = embeddings[theme_idx]
                        desc = self.theme_descriptions['en'][theme_idx]
                        
                        if semantic_target is not None:
                            logger.debug(f"  Query {query_count}/18: Theme {theme_idx+1} ({lang_name}), need {semantic_target - semantic_added} more")
                        else:
                            logger.debug(f"  Query {query_count}/18: Theme {theme_idx+1} ({lang_name}), finding all matches")
                        
                        # Build limit clause
                        limit_clause = f"LIMIT {per_query_limit}" if per_query_limit else ""
                        
                        # Single query with optional limit
                        semantic_query = text(f"""
                            SELECT
                                es.id as segment_id,
                                es.content_id,
                                es.text,
                                es.start_time,
                                es.end_time,
                                es.segment_index,
                                c.title as episode_title,
                                c.channel_name as episode_channel,
                                1 - (es.embedding_alt <=> CAST(:embedding AS vector)) as similarity,
                                es.source_sentence_ids,
                                array_agg(s.text ORDER BY array_position(es.source_sentence_ids, s.sentence_index)) as sentence_texts,
                                array_agg(s.speaker_id ORDER BY array_position(es.source_sentence_ids, s.sentence_index)) as speaker_ids,
                                c.meta_data->>'webpage_url' as original_url
                            FROM embedding_segments es
                            JOIN content c ON es.content_id = c.id
                            LEFT JOIN sentences s ON s.sentence_index = ANY(es.source_sentence_ids) AND s.content_id = es.content_id
                            WHERE es.embedding_alt IS NOT NULL
                            AND 1 - (es.embedding_alt <=> CAST(:embedding AS vector)) >= :threshold
                            {project_condition}
                            GROUP BY es.id, es.content_id, es.text, es.start_time, es.end_time,
                                     es.segment_index, es.source_sentence_ids,
                                     c.title, c.channel_name, es.embedding_alt, c.meta_data->>'webpage_url'
                            {limit_clause}
                        """)
                        
                        params = {
                            'embedding': '[' + ','.join(map(str, embedding)) + ']',
                            'threshold': self.config.similarity_threshold
                        }
                        
                        result = session.execute(semantic_query, params)
                        semantic_rows = result.fetchall()
                        
                        added_this_query = 0
                        for row in semantic_rows:
                            similarity = row[8]
                            
                            # Check if already in candidates or excluded
                            if row[0] in all_exclude_ids:
                                continue
                                
                            # Only check limit in test mode
                            if semantic_target is not None and semantic_added >= semantic_target:
                                break
                            
                            # Extract speaker data and format attribution
                            sentence_texts = row[10] if row[10] else []
                            speaker_ids = row[11] if row[11] else []
                            speaker_attributed_text = self._format_speaker_attributed_text(
                                row[2], sentence_texts, speaker_ids
                            )

                            candidate = SearchCandidate(
                                segment_id=row[0],
                                content_id=row[1],
                                segment_text=row[2],
                                start_time=row[3],
                                end_time=row[4],
                                segment_index=row[5],
                                episode_title=row[6] or "Unknown",
                                episode_channel=row[7] or "Unknown",
                                similarity_score=similarity,
                                matched_via='semantic',
                                matched_keywords=[],
                                matching_themes=[desc],
                                source_sentence_ids=row[9],
                                sentence_texts=sentence_texts,
                                speaker_ids=speaker_ids,
                                speaker_attributed_text=speaker_attributed_text,
                                original_url=row[12]  # Original URL from meta_data
                            )
                            
                            all_candidates[candidate.segment_id] = candidate
                            all_exclude_ids.add(candidate.segment_id)
                            semantic_added += 1
                            added_this_query += 1
                        
                        if added_this_query > 0:
                            logger.debug(f"    Added {added_this_query} from this query")
            
            if semantic_target is not None:
                logger.info(f"Added {semantic_added} semantic matches (target was {semantic_target})")
            else:
                logger.info(f"Added {semantic_added} semantic matches (production mode - no limit)")
        
        # Sort and return
        candidates_list = list(all_candidates.values())
        candidates_list.sort(key=lambda c: c.similarity_score, reverse=True)
        
        logger.info(f"Unified search complete: {len(candidates_list)} total candidates")
        if candidates_list:
            keyword_count = len([c for c in candidates_list if c.matched_via == 'keyword'])
            semantic_count = len([c for c in candidates_list if c.matched_via == 'semantic'])
            logger.info(f"  {keyword_count} via keywords, {semantic_count} via semantic")
            logger.info(f"  Similarity range: {candidates_list[0].similarity_score:.3f} - {candidates_list[-1].similarity_score:.3f}")
        
        return candidates_list

    # ============================================================================
    # Per-Segment Classification Methods (Stages 2-4)
    # ============================================================================

    def stage2_classify_themes(self, segment: SearchCandidate) -> SearchCandidate:
        """Stage 2: Classify themes for a single segment"""
        text_to_classify = segment.speaker_attributed_text or segment.segment_text

        try:
            result = self.classifier.classify_theme(
                text=text_to_classify,
                segment_id=segment.segment_id
            )

            # Store results
            segment.theme_ids = result.theme_ids or []
            segment.theme_names = result.theme_names or []
            segment.theme_confidence = result.confidence
            segment.theme_reasoning = result.reasoning

            # Initialize subtheme results
            segment.subtheme_results = {}

            if self.config.test_mode and segment.theme_names:
                logger.info(f"  Segment {segment.segment_id}: Themes: {', '.join(segment.theme_names)}")
            elif self.config.test_mode:
                logger.info(f"  Segment {segment.segment_id}: No themes matched")

        except Exception as e:
            logger.error(f"Theme classification error for segment {segment.segment_id}: {e}")
            segment.theme_ids = []
            segment.theme_names = []
            segment.theme_confidence = 0.0
            segment.theme_reasoning = f"Error: {str(e)}"
            segment.subtheme_results = {}

        return segment

    def stage3_classify_subthemes(self, segment: SearchCandidate) -> SearchCandidate:
        """Stage 3: Classify subthemes for a single segment (for each matched theme)"""
        if not segment.theme_ids:
            segment.subtheme_results = {}
            return segment

        text_to_classify = segment.speaker_attributed_text or segment.segment_text
        segment.subtheme_results = {}

        for theme_id in segment.theme_ids:
            try:
                result = self.classifier.classify_subtheme(
                    text=text_to_classify,
                    theme_id=theme_id,
                    segment_id=segment.segment_id
                )

                # Store subtheme results for this theme
                if result.subtheme_ids:
                    segment.subtheme_results[theme_id] = {
                        'subtheme_ids': result.subtheme_ids,
                        'subtheme_names': result.subtheme_names,
                        'confidence': result.confidence,
                        'reasoning': result.reasoning
                    }
                    logger.debug(
                        f"  Segment {segment.segment_id}, Theme {theme_id}: "
                        f"{len(result.subtheme_ids)} subthemes"
                    )
                else:
                    segment.subtheme_results[theme_id] = {
                        'subtheme_ids': [],
                        'subtheme_names': [],
                        'confidence': 0.0,
                        'reasoning': "No subthemes identified"
                    }

            except Exception as e:
                logger.error(f"Subtheme classification error for segment {segment.segment_id}, theme {theme_id}: {e}")
                segment.subtheme_results[theme_id] = {
                    'subtheme_ids': [],
                    'subtheme_names': [],
                    'confidence': 0.0,
                    'reasoning': f"Error: {str(e)}"
                }

        return segment

    def stage4_validate(self, segment: SearchCandidate) -> SearchCandidate:
        """Stage 4: Validate theme/subtheme assignments for a single segment"""
        if not segment.theme_ids:
            segment.validation_results = {}
            return segment

        text_to_classify = segment.speaker_attributed_text or segment.segment_text
        segment.validation_results = {}

        logger.debug(f"Starting validation for segment {segment.segment_id} with {len(segment.theme_ids)} themes")

        for theme_id in segment.theme_ids:
            theme_info = self.classifier.themes.get(theme_id)
            if not theme_info:
                continue

            # Determine what we're validating (subthemes or just theme)
            if segment.subtheme_results and theme_id in segment.subtheme_results:
                subtheme_data = segment.subtheme_results[theme_id]
                subtheme_ids = subtheme_data.get('subtheme_ids', [])
            else:
                subtheme_ids = []

            try:
                # Validate each classification separately
                if subtheme_ids:
                    # Has subthemes - validate each subtheme
                    logger.debug(f"  Validating {len(subtheme_ids)} subthemes for theme {theme_id}: {subtheme_ids}")
                    subthemes_confidence = {}

                    # Get subthemes list for this theme
                    theme_subthemes = self.classifier.subthemes.get(theme_id, [])

                    for sub_id in subtheme_ids:
                        # Find the subtheme info in the theme's subthemes list
                        subtheme_info = None
                        for st in theme_subthemes:
                            if st['id'] == sub_id:
                                subtheme_info = st
                                break

                        if not subtheme_info:
                            logger.warning(f"    Subtheme {sub_id} not found for theme {theme_id}!")
                            continue

                        description = subtheme_info['description']
                        logger.debug(f"    Validating subtheme {sub_id}: {description[:50]}...")

                        # Call LLM for this specific subtheme validation (priority 3 = lowest)
                        prompt = f"""I am searching for texts that have: {description}

How well does the following text match that description?

Text: {text_to_classify}

Respond with ONLY ONE of these options:
1. Definitely matches
2. Probably matches
3. Unsure
4. Probably does not match
5. Definitely does not match"""

                        response = self.classifier._call_llm_sync(prompt, priority=3).strip().lower()

                        # Parse Likert scale response to confidence score
                        if 'definitely matches' in response or 'definitely match' in response:
                            confidence = 1.0
                            category = "definitely_matches"
                        elif 'probably matches' in response or 'probably match' in response:
                            confidence = 0.75
                            category = "probably_matches"
                        elif 'unsure' in response:
                            confidence = 0.5
                            category = "unsure"
                        elif 'probably does not' in response or 'probably doesn' in response:
                            confidence = 0.25
                            category = "probably_not"
                        elif 'definitely does not' in response or 'definitely doesn' in response:
                            confidence = 0.0
                            category = "definitely_not"
                        else:
                            # Try to extract number if present (fallback)
                            import re
                            match = re.search(r'\b([1-5])\b', response)
                            if match:
                                num = int(match.group(1))
                                confidence_map = {1: 1.0, 2: 0.75, 3: 0.5, 4: 0.25, 5: 0.0}
                                confidence = confidence_map.get(num, 0.5)
                                category = f"option_{num}"
                            else:
                                confidence = 0.5
                                category = "unknown"

                        subthemes_confidence[sub_id] = confidence
                        logger.debug(f"    Subtheme {sub_id}: {category} (confidence={confidence:.2f})")

                    segment.validation_results[theme_id] = {
                        'subthemes_confidence': subthemes_confidence
                    }
                else:
                    # No subthemes - validate just the theme
                    logger.debug(f"  Validating theme {theme_id} (no subthemes)")
                    description = theme_info['description']

                    prompt = f"""I am searching for texts that have: {description}

How well does the following text match that description?

Text: {text_to_classify}

Respond with ONLY ONE of these options:
1. Definitely matches
2. Probably matches
3. Unsure
4. Probably does not match
5. Definitely does not match"""

                    response = self.classifier._call_llm_sync(prompt, priority=3).strip().lower()

                    # Parse Likert scale response to confidence score
                    if 'definitely matches' in response or 'definitely match' in response:
                        confidence = 1.0
                        category = "definitely_matches"
                    elif 'probably matches' in response or 'probably match' in response:
                        confidence = 0.75
                        category = "probably_matches"
                    elif 'unsure' in response:
                        confidence = 0.5
                        category = "unsure"
                    elif 'probably does not' in response or 'probably doesn' in response:
                        confidence = 0.25
                        category = "probably_not"
                    elif 'definitely does not' in response or 'definitely doesn' in response:
                        confidence = 0.0
                        category = "definitely_not"
                    else:
                        # Try to extract number if present (fallback)
                        import re
                        match = re.search(r'\b([1-5])\b', response)
                        if match:
                            num = int(match.group(1))
                            confidence_map = {1: 1.0, 2: 0.75, 3: 0.5, 4: 0.25, 5: 0.0}
                            confidence = confidence_map.get(num, 0.5)
                            category = f"option_{num}"
                        else:
                            confidence = 0.5
                            category = "unknown"

                    segment.validation_results[theme_id] = {
                        'theme_confidence': confidence
                    }
                    logger.debug(f"  Theme {theme_id}: {category} (confidence={confidence:.2f})")

            except Exception as e:
                logger.error(f"Validation error for segment {segment.segment_id}, theme {theme_id}: {e}")
                segment.validation_results[theme_id] = {
                    'theme_confidence': 0.5,
                    'subthemes_confidence': {},
                    'reasoning': f"Error: {str(e)}"
                }

        return segment

    async def process_segment_batch(self, segments: List[SearchCandidate]) -> List[SearchCandidate]:
        """Process batch of segments with batched LLM calls for efficiency"""
        if not segments:
            return []

        # STAGE 2: Batch theme classification
        logger.debug(f"Stage 2: Classifying themes for {len(segments)} segments (batched)")
        theme_prompts = []
        for segment in segments:
            text_to_classify = segment.speaker_attributed_text or segment.segment_text
            themes_list = "\n".join([
                f"{tid}. {tinfo['description']}"
                for tid, tinfo in sorted(self.classifier.themes.items())
            ])
            prompt = f"""Classify the following text segment into relevant themes.
You may select multiple themes if applicable.
Respond with ONLY the theme numbers separated by commas (e.g., "1,3,5").
If no themes apply, respond with "0".

THEMES:
{themes_list}

TEXT TO CLASSIFY:
{text_to_classify[:2000]}

THEME NUMBERS (comma-separated):"""
            theme_prompts.append(prompt)

        # Batch call for all theme classifications
        theme_responses = await self.classifier._call_llm_batch(theme_prompts, priority=1)

        # Parse theme responses and update segments
        for segment, response in zip(segments, theme_responses):
            try:
                if response == "0" or not response:
                    segment.theme_ids = []
                    segment.theme_names = []
                else:
                    numbers = re.findall(r'\d+', response)
                    theme_ids = []
                    seen = set()
                    for num_str in numbers:
                        num = int(num_str)
                        if num in self.classifier.themes and num not in seen:
                            theme_ids.append(num)
                            seen.add(num)

                    segment.theme_ids = theme_ids
                    segment.theme_names = [self.classifier.themes[tid]['name'] for tid in theme_ids] if theme_ids else []

                segment.subtheme_results = {}
                segment.validation_results = {}
            except Exception as e:
                logger.error(f"Error parsing theme response for segment {segment.segment_id}: {e}")
                segment.theme_ids = []
                segment.theme_names = []
                segment.subtheme_results = {}
                segment.validation_results = {}

        # STAGE 3: Batch subtheme classification (only for segments with themes)
        segments_with_themes = [s for s in segments if s.theme_ids]
        if segments_with_themes:
            logger.debug(f"Stage 3: Classifying subthemes for {len(segments_with_themes)} segments (batched)")
            subtheme_prompts = []
            subtheme_metadata = []  # Track (segment, theme_id) for each prompt

            for segment in segments_with_themes:
                text_to_classify = segment.speaker_attributed_text or segment.segment_text
                for theme_id in segment.theme_ids:
                    if theme_id in self.classifier.subthemes:
                        subthemes_list = "\n".join([
                            f"{st['id']}. {st['description']}"
                            for st in self.classifier.subthemes[theme_id]
                        ])
                        prompt = f"""Classify the following text into relevant subthemes for Theme {theme_id}.
You may select multiple subthemes if applicable.
Respond with ONLY the subtheme numbers separated by commas (e.g., "1,2,4").
If no subthemes apply, respond with "0".

SUBTHEMES:
{subthemes_list}

TEXT TO CLASSIFY:
{text_to_classify[:2000]}

SUBTHEME NUMBERS (comma-separated):"""
                        subtheme_prompts.append(prompt)
                        subtheme_metadata.append((segment, theme_id))

            if subtheme_prompts:
                subtheme_responses = await self.classifier._call_llm_batch(subtheme_prompts, priority=1)

                # Parse subtheme responses
                for (segment, theme_id), response in zip(subtheme_metadata, subtheme_responses):
                    try:
                        if response == "0" or not response:
                            segment.subtheme_results[theme_id] = []
                        else:
                            numbers = re.findall(r'\d+', response)
                            subtheme_ids = []
                            seen = set()
                            for num_str in numbers:
                                num = int(num_str)
                                if theme_id in self.classifier.subthemes:
                                    # Check if subtheme ID exists in the list
                                    valid_ids = [st['id'] for st in self.classifier.subthemes[theme_id]]
                                    if num in valid_ids and num not in seen:
                                        subtheme_ids.append(num)
                                        seen.add(num)
                            segment.subtheme_results[theme_id] = subtheme_ids
                    except Exception as e:
                        logger.error(f"Error parsing subtheme response for segment {segment.segment_id}, theme {theme_id}: {e}")
                        segment.subtheme_results[theme_id] = []

        # STAGE 4: Batch validation (only for segments with themes)
        if segments_with_themes:
            logger.debug(f"Stage 4: Validating {len(segments_with_themes)} segments (batched)")
            validation_prompts = []
            validation_metadata = []  # Track (segment, theme_id) for each prompt

            for segment in segments_with_themes:
                text_to_classify = segment.speaker_attributed_text or segment.segment_text
                for theme_id in segment.theme_ids:
                    theme_name = self.classifier.themes[theme_id]['name']
                    theme_desc = self.classifier.themes[theme_id]['description']
                    subtheme_names = []
                    if theme_id in segment.subtheme_results:
                        for sub_id in segment.subtheme_results[theme_id]:
                            if theme_id in self.classifier.subthemes:
                                # Find subtheme by ID in the list
                                for st in self.classifier.subthemes[theme_id]:
                                    if st['id'] == sub_id:
                                        subtheme_names.append(st['name'])
                                        break

                    prompt = f"""Validate if this text segment actually matches the assigned theme and subthemes.

THEME: {theme_name}
DESCRIPTION: {theme_desc}

ASSIGNED SUBTHEMES: {', '.join(subtheme_names) if subtheme_names else 'None'}

TEXT:
{text_to_classify[:2000]}

Does this text segment match the theme and subthemes? Respond with one of:
- "definitely matches" (high confidence)
- "probably matches" (medium-high confidence)
- "unsure" (medium confidence)
- "probably does not match" (medium-low confidence)
- "definitely does not match" (low confidence)

RESPONSE:"""
                    validation_prompts.append(prompt)
                    validation_metadata.append((segment, theme_id))

            if validation_prompts:
                validation_responses = await self.classifier._call_llm_batch(validation_prompts, priority=1)

                # Parse validation responses
                for (segment, theme_id), response in zip(validation_metadata, validation_responses):
                    try:
                        response_lower = response.lower()
                        if 'definitely matches' in response_lower or 'definitely match' in response_lower:
                            confidence = 1.0
                        elif 'probably matches' in response_lower or 'probably match' in response_lower:
                            confidence = 0.75
                        elif 'unsure' in response_lower:
                            confidence = 0.5
                        elif 'probably does not' in response_lower or 'probably doesn' in response_lower:
                            confidence = 0.25
                        elif 'definitely does not' in response_lower or 'definitely doesn' in response_lower:
                            confidence = 0.0
                        else:
                            confidence = 0.5

                        if theme_id not in segment.validation_results:
                            segment.validation_results[theme_id] = {}

                        segment.validation_results[theme_id] = {
                            'is_valid': confidence >= 0.5,
                            'confidence': confidence,
                            'theme_confidence': confidence,
                            'reasoning': response
                        }
                    except Exception as e:
                        logger.error(f"Error parsing validation response for segment {segment.segment_id}, theme {theme_id}: {e}")
                        segment.validation_results[theme_id] = {
                            'is_valid': False,
                            'confidence': 0.5,
                            'theme_confidence': 0.5,
                            'reasoning': f"Error: {str(e)}"
                        }

        return segments

    async def process_segment(self, segment: SearchCandidate) -> SearchCandidate:
        """Process single segment through full pipeline: themes → subthemes → validation"""
        # Stage 2: Theme classification
        segment = self.stage2_classify_themes(segment)

        # Stage 3: Subtheme classification (only if themes found)
        if segment.theme_ids:
            segment = self.stage3_classify_subthemes(segment)

        # Stage 4: Validation (only if themes found)
        if segment.theme_ids:
            segment = self.stage4_validate(segment)

        return segment

    async def process_all_segments(self, candidates: List[SearchCandidate]) -> List[SearchCandidate]:
        """Process all segments through unified theme→subtheme→validation pipeline"""
        logger.info("Stages 2-4: Unified segment-by-segment classification pipeline")
        logger.info(f"Processing {len(candidates)} segments")

        # Load checkpoint
        processed_segment_ids, _ = self._load_checkpoint()

        # Prepare output
        output_csv = str(self.output_path).replace('.csv', '_incremental.csv')
        output_csv_path = Path(output_csv)

        # Check if output file already exists to preserve existing data
        file_already_exists = output_csv_path.exists() and output_csv_path.stat().st_size > 0
        if file_already_exists:
            logger.info(f"Incremental CSV already exists, will append to it: {output_csv}")

        all_results = []
        batch_results = []

        # Filter out already processed segments for accurate progress tracking
        unprocessed_candidates = [c for c in candidates if c.segment_id not in processed_segment_ids]
        logger.info(f"Found {len(unprocessed_candidates)} unprocessed segments out of {len(candidates)} total candidates")

        # Process in batches of 10 (for batched LLM requests to avoid overwhelming model server queue)
        batch_size = 10
        with tqdm(total=len(unprocessed_candidates), desc="Processing Segments", unit="segment") as pbar:
            for batch_start in range(0, len(unprocessed_candidates), batch_size):
                batch_end = min(batch_start + batch_size, len(unprocessed_candidates))
                current_batch = unprocessed_candidates[batch_start:batch_end]

                # Process entire batch with batched LLM calls
                try:
                    processed_batch = await self.process_segment_batch(current_batch)
                    all_results.extend(processed_batch)
                    batch_results.extend(processed_batch)

                except Exception as e:
                    logger.error(f"Error processing batch starting at {batch_start}: {e}")
                    # Add segments with empty results so we don't reprocess
                    for segment in current_batch:
                        segment.theme_ids = []
                        segment.theme_names = []
                        segment.subtheme_results = {}
                        segment.validation_results = {}
                        all_results.append(segment)
                        batch_results.append(segment)

                pbar.update(len(current_batch))

                # Save every 100 segments (10 batches)
                if len(batch_results) >= 100:
                    self._save_candidates_to_csv(
                        batch_results,
                        output_csv,
                        append=file_already_exists  # Always append if file exists
                    )
                    file_already_exists = True  # After first write, file now exists

                    # Update checkpoint
                    batch_segment_ids = [s.segment_id for s in batch_results]
                    processed_segment_ids.update(batch_segment_ids)
                    self._save_checkpoint(processed_segment_ids)

                    logger.info(f"Checkpoint: {len(processed_segment_ids)} segments processed")
                    batch_results = []

        # Final save for remaining segments
        if batch_results:
            self._save_candidates_to_csv(batch_results, output_csv, append=True)
            batch_segment_ids = [s.segment_id for s in batch_results]
            processed_segment_ids.update(batch_segment_ids)
            self._save_checkpoint(processed_segment_ids)

        logger.info(f"Processing complete: {len(all_results)} segments classified")
        return all_results


    def _save_stage1_candidates(self, candidates: List[SearchCandidate], output_path: str):
        """Save Stage 1 search candidates to CSV (no theme/subtheme info yet)"""
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            header = [
                'segment_id', 'content_id', 'episode_title', 'episode_channel',
                'start_time', 'end_time', 'matched_via', 'matched_keywords',
                'similarity_score', 'matching_themes', 'segment_text',
                'speaker_attributed_text', 'original_url'
            ]
            writer.writerow(header)

            for candidate in candidates:
                keywords_str = ', '.join(candidate.matched_keywords) if candidate.matched_keywords else ""
                themes_str = ', '.join(candidate.matching_themes) if candidate.matching_themes else ""

                row = [
                    candidate.segment_id,
                    candidate.content_id,
                    candidate.episode_title,
                    candidate.episode_channel,
                    candidate.start_time,
                    candidate.end_time,
                    candidate.matched_via,
                    keywords_str,
                    candidate.similarity_score,
                    themes_str,
                    candidate.segment_text,
                    candidate.speaker_attributed_text or candidate.segment_text,
                    candidate.original_url or ""
                ]
                writer.writerow(row)

    def _load_stage1_candidates(self, cache_path: str) -> Optional[List[SearchCandidate]]:
        """Load Stage 1 candidates from cache CSV if it exists"""
        if not Path(cache_path).exists():
            return None

        logger.info(f"Loading Stage 1 candidates from cache: {cache_path}")

        try:
            candidates = []
            with open(cache_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)

                for row in reader:
                    # Parse matched_keywords and matching_themes
                    matched_keywords = [k.strip() for k in row['matched_keywords'].split(',') if k.strip()]
                    matching_themes = [t.strip() for t in row['matching_themes'].split(',') if t.strip()]

                    candidate = SearchCandidate(
                        segment_id=int(row['segment_id']),
                        content_id=row['content_id'],
                        segment_text=row['segment_text'],
                        start_time=float(row['start_time']),
                        end_time=float(row['end_time']),
                        segment_index=0,  # Not stored in stage1 cache
                        episode_title=row['episode_title'],
                        episode_channel=row['episode_channel'],
                        similarity_score=float(row['similarity_score']),
                        matched_via=row['matched_via'],
                        matched_keywords=matched_keywords,
                        matching_themes=matching_themes,
                        speaker_attributed_text=row['speaker_attributed_text'],
                        original_url=row['original_url'] if row['original_url'] else None,
                        # Fields that will be populated later
                        source_sentence_ids=None,
                        sentence_texts=None,
                        speaker_ids=None
                    )
                    candidates.append(candidate)

            logger.info(f"Loaded {len(candidates)} candidates from Stage 1 cache")
            return candidates

        except Exception as e:
            logger.error(f"Failed to load Stage 1 cache: {e}")
            return None

    def _cleanup_stale_segments(self, current_segment_ids: Set[int]) -> Dict[str, Any]:
        """
        Remove rows from output CSV that have segment IDs no longer in the database.
        Moves stale entries to a separate archive file.

        Args:
            current_segment_ids: Set of segment IDs from current run (still exist in DB)

        Returns:
            Dict with statistics about cleanup
        """
        if not self.output_path.exists():
            return {'status': 'skipped', 'reason': 'output file does not exist'}

        logger.info(f"Checking for stale segment IDs in {self.output_path}...")

        # Create archive path
        archive_path = self.output_path.parent / f"{self.output_path.stem}_archived{self.output_path.suffix}"

        try:
            # Read existing output
            df = pd.read_csv(self.output_path)

            if 'segment_id' not in df.columns:
                return {'status': 'error', 'reason': 'segment_id column not found'}

            original_count = len(df)

            # Split into current and stale
            current_mask = df['segment_id'].isin(current_segment_ids)
            current_df = df[current_mask]
            stale_df = df[~current_mask]

            stale_count = len(stale_df)
            kept_count = len(current_df)

            logger.info(f"  Total rows: {original_count}")
            logger.info(f"  Current (valid): {kept_count}")
            logger.info(f"  Stale (deleted): {stale_count}")

            if stale_count > 0:
                # Save stale entries to archive
                if archive_path.exists():
                    # Append to existing archive
                    stale_df.to_csv(archive_path, mode='a', header=False, index=False)
                    logger.info(f"  Appended {stale_count} stale entries to {archive_path}")
                else:
                    # Create new archive
                    stale_df.to_csv(archive_path, index=False)
                    logger.info(f"  Created archive with {stale_count} stale entries: {archive_path}")

                # Overwrite output with only current entries
                current_df.to_csv(self.output_path, index=False)
                logger.info(f"  Updated {self.output_path} with {kept_count} current entries")

                return {
                    'status': 'success',
                    'original_count': original_count,
                    'kept_count': kept_count,
                    'stale_count': stale_count,
                    'archive_path': str(archive_path)
                }
            else:
                logger.info("  No stale entries found")
                return {
                    'status': 'success',
                    'original_count': original_count,
                    'kept_count': kept_count,
                    'stale_count': 0
                }

        except Exception as e:
            logger.error(f"Error cleaning up stale segments: {e}")
            return {'status': 'error', 'reason': str(e)}

    def save_results(self, candidates: List[SearchCandidate]):
        """Save final results to CSV"""
        self._save_candidates_to_csv(candidates, self.config.output_csv)
        logger.info(f"Saved final results to {self.config.output_csv}")

        # Clean up checkpoint and intermediate files
        try:
            if self.checkpoint_path.exists():
                self.checkpoint_path.unlink()
            # Keep the LLM results CSV as it contains the incremental results
            logger.info("Cleaned up checkpoint file")
        except Exception as e:
            logger.warning(f"Could not clean up checkpoint file: {e}")
    
    async def run_async(self) -> Dict[str, Any]:
        """Run the complete pipeline with async batch processing"""
        logger.info("="*70)
        logger.info("Starting Unified Theme Classification Pipeline (Async Batch Mode)")
        logger.info(f"Keywords: {'Loaded' if self.keywords else 'None'}")
        if self.keywords:
            logger.debug(f"Global keywords: {self.keywords.get('global', [])}")
            theme_specific = [k for k in self.keywords if k != 'global']
            if theme_specific:
                logger.debug(f"Theme-specific keywords for themes: {theme_specific}")
        logger.info(f"Similarity threshold: {self.config.similarity_threshold}")
        # Calculate dynamic per-theme limits based on mode
        num_themes = len(self.theme_descriptions['en'])
        
        if self.config.test_mode:
            # Test mode: limit to specified number of candidates
            target_candidates = self.config.test_mode
            self.config.keywords_per_theme = max(50, target_candidates // num_themes)
            self.config.semantic_per_theme = max(30, target_candidates // (num_themes * 2))
            logger.info(f"TEST MODE: Limiting to {target_candidates} total candidates")
        else:
            # Production mode: get ALL matching content (no per-theme limits)
            self.config.keywords_per_theme = None  # No limit
            self.config.semantic_per_theme = None  # No limit
            logger.info("PRODUCTION MODE: Finding all matching content")
        
        logger.info(f"Keywords per theme: {self.config.keywords_per_theme or 'unlimited'}")
        logger.info(f"Semantic per theme: {self.config.semantic_per_theme or 'unlimited'}")
        logger.info("="*70)
        
        try:
            # Stage 1: Keyword + Semantic Search
            stage1_cache = str(self.output_path).replace('.csv', '_stage1_candidates.csv')

            # Try to load from cache first (unless skip_stage1_cache is True)
            candidates = None
            if not self.config.skip_stage1_cache:
                candidates = self._load_stage1_candidates(stage1_cache)

            if candidates:
                logger.info(f"✓ Using cached Stage 1 results ({len(candidates)} candidates)")
                logger.info("Skipping Stage 1 model loading (using cached results)")
            else:
                if self.config.skip_stage1_cache:
                    logger.info("Skipping Stage 1 cache (--skip-stage1-cache flag set)")

                # Load Stage 1 models (embedding + FAISS)
                logger.info("\n" + "="*70)
                logger.info("Loading Stage 1 models (embedding + FAISS)")
                logger.info("="*70)
                self._load_embedding_model()
                self._load_faiss_index()

                # Run search if no cache available
                candidates = self.stage1_search()
                if not candidates:
                    logger.warning("No candidates found")
                    return {'status': 'no_candidates'}
                logger.info(f"Stage 1 complete: {len(candidates)} candidates found")

                # Cache Stage 1 results immediately before proceeding
                logger.info(f"Caching Stage 1 results to {stage1_cache}")
                self._save_stage1_candidates(candidates, stage1_cache)
                logger.info("Stage 1 results cached successfully")

                # Unload Stage 1 models to free memory
                logger.info("\n" + "="*70)
                self._unload_stage1_models()
                logger.info("="*70 + "\n")

            # Speaker attribution is already included from the search queries
            # Each candidate already has transcription_texts and speaker_ids populated

            # Print one example of speaker-enriched segment for verification
            for c in candidates[:10]:  # Check first 10 to find one with speaker attribution
                if c.speaker_attributed_text and c.speaker_attributed_text != c.segment_text:
                    logger.info(f"\nExample speaker-enriched segment (ID: {c.segment_id}):")
                    logger.info(f"  Original text (first 200 chars): {c.segment_text[:200]}...")
                    logger.info(f"  Speaker-attributed (first 300 chars): {c.speaker_attributed_text[:300]}...")
                    logger.info(f"  Number of speakers: {len(set(c.speaker_ids)) if c.speaker_ids else 0}")
                    break

            # Load MLX classifier for Stages 2-4
            logger.info("\n" + "="*70)
            logger.info("Loading MLX classifier for theme classification (Stages 2-4)")
            logger.info("="*70)
            self._load_llm_classifier()
            logger.info("")

            # Stages 2-4: Unified segment-by-segment processing
            final_candidates = await self.process_all_segments(candidates)

            # Optional: Clean up stale segments before saving
            if self.config.cleanup_stale_segments:
                logger.info("\n" + "="*70)
                logger.info("Cleaning up stale segment IDs")
                logger.info("="*70)
                current_segment_ids = {c.segment_id for c in final_candidates}
                cleanup_result = self._cleanup_stale_segments(current_segment_ids)

                if cleanup_result['status'] == 'success' and cleanup_result.get('stale_count', 0) > 0:
                    logger.info(f"✅ Archived {cleanup_result['stale_count']} stale segments to {cleanup_result['archive_path']}")
                elif cleanup_result['status'] == 'success':
                    logger.info("✅ No stale segments found")
                else:
                    logger.warning(f"⚠️  Cleanup failed: {cleanup_result.get('reason', 'unknown')}")
                logger.info("")

            # Save results
            self.save_results(final_candidates)
            
            # Calculate validation statistics with confidence scores
            confidence_summary = {
                'total': 0,
                'high': 0,  # >= 0.75
                'medium': 0,  # 0.5 - 0.75
                'low': 0,  # 0.25 - 0.5
                'very_low': 0,  # < 0.25
                'avg': []
            }
            
            for candidate in final_candidates:
                if hasattr(candidate, 'validation_results') and candidate.validation_results:
                    for theme_id, val_result in candidate.validation_results.items():
                        # Theme confidence
                        if val_result.get('theme_confidence') is not None:
                            conf = val_result['theme_confidence']
                            confidence_summary['total'] += 1
                            confidence_summary['avg'].append(conf)
                            if conf >= 0.75:
                                confidence_summary['high'] += 1
                            elif conf >= 0.5:
                                confidence_summary['medium'] += 1
                            elif conf >= 0.25:
                                confidence_summary['low'] += 1
                            else:
                                confidence_summary['very_low'] += 1
                        
                        # Subtheme confidences
                        for sub_id, conf in val_result.get('subthemes_confidence', {}).items():
                            confidence_summary['total'] += 1
                            confidence_summary['avg'].append(conf)
                            if conf >= 0.75:
                                confidence_summary['high'] += 1
                            elif conf >= 0.5:
                                confidence_summary['medium'] += 1
                            elif conf >= 0.25:
                                confidence_summary['low'] += 1
                            else:
                                confidence_summary['very_low'] += 1
            
            avg_confidence = sum(confidence_summary['avg']) / len(confidence_summary['avg']) if confidence_summary['avg'] else 0
            
            logger.info("="*70)
            logger.info("Pipeline completed successfully")
            logger.info(f"Total candidates found: {len(candidates)}")
            themed_count = len([c for c in final_candidates if c.theme_ids])
            logger.info(f"LLM identified themes in: {themed_count} candidates")
            logger.info(f"Validation confidence scores: {confidence_summary['total']} classifications")
            if confidence_summary['total'] > 0:
                logger.info(f"  High (≥0.75): {confidence_summary['high']} ({confidence_summary['high']*100/confidence_summary['total']:.1f}%)")
                logger.info(f"  Medium (0.5-0.75): {confidence_summary['medium']} ({confidence_summary['medium']*100/confidence_summary['total']:.1f}%)")
                logger.info(f"  Low (0.25-0.5): {confidence_summary['low']} ({confidence_summary['low']*100/confidence_summary['total']:.1f}%)")
                logger.info(f"  Very Low (<0.25): {confidence_summary['very_low']} ({confidence_summary['very_low']*100/confidence_summary['total']:.1f}%)")
            logger.info(f"  Average confidence: {avg_confidence:.3f}")
            logger.info(f"Output: {self.config.output_csv}")
            logger.info("="*70)
            
            return {
                'status': 'success',
                'total_candidates': len(candidates),
                'themes_identified': themed_count,
                'output_file': self.config.output_csv
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def run(self) -> Dict[str, Any]:
        """Synchronous wrapper for async pipeline"""
        return asyncio.run(self.run_async())

def main():
    parser = argparse.ArgumentParser(description='Unified Theme Classification with Keyword Support')
    parser.add_argument('--subthemes-csv', required=True, 
                       help='Path to subthemes_complete.csv file')
    parser.add_argument('--keywords-file', 
                       help='Path to keywords file (optional)')
    parser.add_argument('--output-csv', required=True, 
                       help='Output CSV file path')
    
    # Search parameters
    parser.add_argument('--similarity-threshold', type=float, default=0.6,
                       help='Minimum similarity score threshold (default: 0.6)')
    parser.add_argument('--keyword-boost', type=float, default=0.1,
                       help='Score boost for keyword matches (default: 0.1)')
    
    # Model parameters
    parser.add_argument('--model', default='tier_1',
                       help='MLX model for classification: tier_1 (80B), tier_2 (4B), tier_3 (8B) (default: tier_1)')
    parser.add_argument('--embedding-model', default='Qwen/Qwen3-Embedding-4B',
                       help='Embedding model for semantic search (default: Qwen/Qwen3-Embedding-4B)')
    parser.add_argument('--gpu', action='store_true', default=True, help='Use GPU (MLX uses Apple Silicon by default)')
    parser.add_argument('--project', help='Filter results to specific project (e.g., CPRMV)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--test-mode', type=int, metavar='N',
                       help='Test mode: limit to N total candidates (50%% keywords, 50%% semantic via 18 queries)')
    parser.add_argument('--no-checkpoint', action='store_true',
                       help='Disable checkpoint/resume functionality')

    # FAISS parameters
    parser.add_argument('--use-faiss', action='store_true',
                       help='Use FAISS CPU for semantic search instead of PostgreSQL HNSW (requires --project)')
    parser.add_argument('--faiss-index-path',
                       help='Path to save/load FAISS index (optional, enables index persistence)')

    # Cache parameters
    parser.add_argument('--skip-stage1-cache', action='store_true',
                       help='Force fresh Stage 1 search, skip cached results')

    # Cleanup parameters
    parser.add_argument('--cleanup-stale-segments', action='store_true',
                       help='Remove segment IDs from output CSV that no longer exist in database (moves to _archived.csv)')

    args = parser.parse_args()
    
    # Set logging level based on debug flag
    if args.debug:
        logger.setLevel(logging.DEBUG)
        # Also set debug for any parent loggers
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load model_server configuration
    model_server_url = None
    use_model_server = True
    try:
        from src.utils.config import load_config
        app_config = load_config()
        model_server_config = app_config.get('processing', {}).get('model_server', {})
        if model_server_config.get('enabled', False):
            host = model_server_config.get('host')
            port = model_server_config.get('port', 8004)
            if host:
                model_server_url = f"http://{host}:{port}"
                logger.info(f"Model server configured: {model_server_url}")
    except Exception as e:
        logger.warning(f"Could not load model_server config: {e}, will use local MLX")

    # Create configuration
    config = UnifiedConfig(
        subthemes_csv=args.subthemes_csv,
        keywords_file=args.keywords_file,
        output_csv=args.output_csv,
        similarity_threshold=args.similarity_threshold,
        keyword_boost=args.keyword_boost,
        model_name=args.model,
        embedding_model=args.embedding_model,
        use_gpu=args.gpu,
        project=args.project,
        model_server_url=model_server_url,
        use_model_server=use_model_server,
        test_mode=args.test_mode,
        resume=not args.no_checkpoint,
        use_faiss=args.use_faiss,
        faiss_index_path=args.faiss_index_path,
        skip_stage1_cache=args.skip_stage1_cache,
        cleanup_stale_segments=args.cleanup_stale_segments
    )
    
    # Run pipeline
    pipeline = UnifiedThemeClassifier(config)
    results = pipeline.run()
    
    # Print summary
    if results['status'] == 'success':
        print(f"\n✅ Pipeline completed!")
        print(f"📊 Candidates: {results['total_candidates']}")
        print(f"🎯 Themes identified: {results['themes_identified']}")
        print(f"📁 Output: {results['output_file']}")
    else:
        print(f"\n❌ Pipeline failed: {results.get('message', results['status'])}")

if __name__ == "__main__":
    main()