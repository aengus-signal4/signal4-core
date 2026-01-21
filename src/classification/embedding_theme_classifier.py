#!/usr/bin/env python3
"""
Embedding-Focused Theme Classification with LLM Validation

This classifier uses a hierarchical semantic search approach:
1. Theme-level embedding search
2. Subtheme-level refinement
3. LLM validation for precision

Usage:
    python embedding_theme_classifier.py \
        --subthemes-csv projects/CPRMV/subthemes_complete.csv \
        --output-csv outputs/embedding_theme_results.csv \
        --similarity-threshold 0.6 \
        --project CPRMV
"""

import sys
import os
import pandas as pd
import csv
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from src.utils.paths import get_project_root, get_config_path
import argparse
from tqdm import tqdm
import time
import re
import asyncio
import httpx
import hashlib
import pickle

# Add project root to path
sys.path.append(str(get_project_root()))

from src.database.session import get_session
from src.database.models import EmbeddingSegment, Content
from sqlalchemy import text
from src.utils.logger import setup_worker_logger
from sentence_transformers import SentenceTransformer

logger = setup_worker_logger('embedding_theme_classifier')

# ============================================================================
# Configuration Classes
# ============================================================================

@dataclass
class EmbeddingConfig:
    """Configuration for embedding-based theme classification"""
    subthemes_csv: str
    output_csv: str
    
    # Search parameters
    theme_similarity_threshold: float = 0.60
    subtheme_similarity_threshold: float = 0.65
    
    # Model configuration
    embedding_model: str = "Qwen/Qwen3-Embedding-4B"
    llm_model: str = "qwen3:4b-instruct"
    llm_endpoint: str = "http://localhost:8002"
    
    # Processing options
    project: Optional[str] = None
    batch_size: int = 100
    max_segments_per_theme: Optional[int] = None  # None means no limit
    
    # Checkpoint configuration
    checkpoint_file: Optional[str] = None
    resume: bool = True
    
    # Debug/test options
    test_mode: bool = False
    test_limit: int = 100

@dataclass
class SearchCandidate:
    """Candidate segment from semantic search"""
    segment_id: int
    content_id: str
    episode_title: str
    episode_channel: str
    segment_text: str
    start_time: float
    end_time: float
    segment_index: int
    
    # Semantic search results
    theme_similarities: Dict[int, float] = field(default_factory=dict)
    subtheme_similarities: Dict[str, float] = field(default_factory=dict)
    
    # Speaker attribution
    speaker_attributed_text: Optional[str] = None
    source_sentence_ids: Optional[List[int]] = None  # Sentence indices from sentences table
    sentence_texts: Optional[List[str]] = None  # Sentence texts from sentences table
    speaker_ids: Optional[List[int]] = None
    
    # LLM validation results
    validated_themes: List[int] = field(default_factory=list)
    validated_subthemes: List[str] = field(default_factory=list)
    validation_confidence: Dict[str, float] = field(default_factory=dict)

# ============================================================================
# Embedding Theme Classifier
# ============================================================================

class EmbeddingThemeClassifier:
    """Hierarchical embedding-based theme classifier with LLM validation"""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {config.embedding_model}")
        if 'Qwen' in config.embedding_model and 'Embedding-4B' in config.embedding_model:
            self.embedding_model = SentenceTransformer(
                config.embedding_model, 
                trust_remote_code=True, 
                truncate_dim=2000
            )
            logger.info("Using Qwen3-4B embeddings with truncate_dim=2000")
        else:
            self.embedding_model = SentenceTransformer(config.embedding_model)
        
        # Load themes and subthemes
        self.themes, self.subthemes = self._load_themes_and_subthemes()
        
        # Setup embedding cache
        self.embedding_cache_dir = Path(config.output_csv).parent / "embedding_cache"
        self.embedding_cache_dir.mkdir(exist_ok=True)
        
        # Generate or load cached embeddings for themes and subthemes
        self.theme_embeddings, self.subtheme_embeddings = self._load_or_generate_all_embeddings()
        
        # Setup output
        self.output_path = Path(config.output_csv)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup checkpoint
        if not config.checkpoint_file:
            checkpoint_name = self.output_path.stem + "_checkpoint.json"
            self.checkpoint_path = self.output_path.parent / checkpoint_name
        else:
            self.checkpoint_path = Path(config.checkpoint_file)
        
        logger.info(f"Initialized embedding theme classifier")
        logger.info(f"Themes: {len(self.themes)}")
        logger.info(f"Total subthemes: {sum(len(st) for st in self.subthemes.values())}")
    
    def _load_themes_and_subthemes(self) -> Tuple[Dict, Dict]:
        """Load themes and subthemes from CSV"""
        df = pd.read_csv(self.config.subthemes_csv)
        
        themes = {}
        subthemes = {}
        
        for theme_id in df['theme_id'].unique():
            theme_df = df[df['theme_id'] == theme_id]
            theme_info = theme_df.iloc[0]
            
            themes[theme_id] = {
                'id': theme_id,
                'name': theme_info['theme_name'],
                'description': theme_info['theme_description'],
                'description_fr': theme_info.get('theme_description_fr', theme_info['theme_description'])
            }
            
            # Collect subthemes for this theme
            theme_subthemes = []
            for _, row in theme_df.iterrows():
                if pd.notna(row.get('subtheme_id')):
                    subtheme = {
                        'id': row['subtheme_id'],
                        'name': row['subtheme_name'],
                        'description': row.get('subtheme_description_short', row['subtheme_description']),
                        'description_fr': row.get('subtheme_description_fr', row.get('subtheme_description'))
                    }
                    theme_subthemes.append(subtheme)
            
            if theme_subthemes:
                subthemes[theme_id] = theme_subthemes
        
        return themes, subthemes
    
    def _calculate_csv_hash(self) -> str:
        """Calculate SHA256 hash of the CSV file"""
        with open(self.config.subthemes_csv, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    
    def _get_embedding_cache_path(self) -> Path:
        """Get cache file path for all embeddings"""
        model_name = self.config.embedding_model.replace('/', '_').replace('-', '_')
        csv_hash = self._calculate_csv_hash()[:12]  # First 12 chars of hash
        return self.embedding_cache_dir / f"all_embeddings_{model_name}_{csv_hash}.pkl"
    
    def _load_or_generate_all_embeddings(self) -> Tuple[Dict[int, Dict[str, np.ndarray]], Dict[str, Dict[str, np.ndarray]]]:
        """Load all embeddings from single cache file or generate if not cached"""
        cache_path = self._get_embedding_cache_path()
        
        if cache_path.exists():
            try:
                logger.info(f"Loading cached embeddings from {cache_path}")
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                
                theme_embeddings = cached_data.get('theme_embeddings', {})
                subtheme_embeddings = cached_data.get('subtheme_embeddings', {})
                
                # Verify all themes are present
                expected_subtheme_ids = set()
                for theme_subthemes in self.subthemes.values():
                    expected_subtheme_ids.update(st['id'] for st in theme_subthemes)
                
                themes_valid = set(theme_embeddings.keys()) == set(self.themes.keys())
                subthemes_valid = set(subtheme_embeddings.keys()) == expected_subtheme_ids
                
                if themes_valid and subthemes_valid:
                    logger.info(f"Successfully loaded {len(theme_embeddings)} theme and {len(subtheme_embeddings)} subtheme embeddings from cache")
                    return theme_embeddings, subtheme_embeddings
                else:
                    logger.warning("Cached embeddings don't match current themes/subthemes, regenerating")
            except Exception as e:
                logger.warning(f"Failed to load cached embeddings: {e}, regenerating")
        
        # Generate all embeddings
        logger.info("Generating all embeddings...")
        theme_embeddings = self._generate_theme_embeddings()
        subtheme_embeddings = self._generate_subtheme_embeddings()
        
        # Cache all embeddings in single file
        try:
            cache_data = {
                'theme_embeddings': theme_embeddings,
                'subtheme_embeddings': subtheme_embeddings,
                'csv_hash': self._calculate_csv_hash(),
                'model': self.config.embedding_model,
                'created_at': datetime.now().isoformat()
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            logger.info(f"Cached all embeddings to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to cache embeddings: {e}")
        
        return theme_embeddings, subtheme_embeddings
    
    def _generate_theme_embeddings(self) -> Dict[int, Dict[str, np.ndarray]]:
        """Generate embeddings for each theme in both languages"""
        logger.info("Generating theme embeddings...")
        theme_embeddings = {}
        
        for theme_id, theme_info in self.themes.items():
            # English embedding
            query_en = f"Instruct: Given a theme description, retrieve segments discussing this theme\nQuery: {theme_info['description']}"
            embedding_en = self.embedding_model.encode(
                query_en,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            # French embedding
            query_fr = f"Instruct: Given a theme description, retrieve segments discussing this theme\nQuery: {theme_info['description_fr']}"
            embedding_fr = self.embedding_model.encode(
                query_fr,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            theme_embeddings[theme_id] = {
                'en': embedding_en,
                'fr': embedding_fr,
                'description': theme_info['description']
            }
            
            logger.debug(f"  Theme {theme_id} ({theme_info['name']}): embeddings generated")
        
        return theme_embeddings
    
    def _generate_subtheme_embeddings(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Generate embeddings for each subtheme in both languages"""
        logger.info("Generating subtheme embeddings...")
        subtheme_embeddings = {}
        
        for theme_id, theme_subthemes in self.subthemes.items():
            for subtheme in theme_subthemes:
                # English embedding
                query_en = f"Instruct: Given a theme description, retrieve segments discussing this theme\nQuery: {subtheme['description']}"
                embedding_en = self.embedding_model.encode(
                    query_en,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                
                # French embedding
                query_fr = f"Instruct: Given a theme description, retrieve segments discussing this theme\nQuery: {subtheme['description_fr']}"
                embedding_fr = self.embedding_model.encode(
                    query_fr,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                
                subtheme_embeddings[subtheme['id']] = {
                    'en': embedding_en,
                    'fr': embedding_fr,
                    'theme_id': theme_id,
                    'description': subtheme['description']
                }
                
                logger.debug(f"    Subtheme {subtheme['id']} ({subtheme['name']}): embeddings generated")
        
        return subtheme_embeddings
    
    def _format_speaker_attributed_text(self, segment_text: str, transcription_texts: List[str], speaker_ids: List[int]) -> str:
        """Format segment text with anonymous speaker attribution"""
        if not transcription_texts or not speaker_ids:
            return segment_text
        
        if len(transcription_texts) != len(speaker_ids):
            logger.warning(f"Mismatch between transcription_texts and speaker_ids")
            return segment_text
        
        # Create mapping of speaker_id to anonymous labels
        unique_speakers = []
        speaker_map = {}
        for sid in speaker_ids:
            if sid is not None and sid not in speaker_map:
                speaker_map[sid] = f"SPEAKER_{len(unique_speakers):02d}"
                unique_speakers.append(sid)
        
        # Build speaker-attributed text
        attributed_parts = []
        for trans_text, speaker_id in zip(transcription_texts, speaker_ids):
            if trans_text and speaker_id is not None:
                speaker_label = speaker_map.get(speaker_id, f"SPEAKER_{speaker_id}")
                attributed_parts.append(f"{speaker_label}: {trans_text}")
        
        if attributed_parts:
            return " ".join(attributed_parts)
        elif len(set(speaker_ids)) == 1 and speaker_ids[0] is not None:
            speaker_label = speaker_map.get(speaker_ids[0], f"SPEAKER_{speaker_ids[0]}")
            return f"{speaker_label}: {segment_text}"
        else:
            return segment_text
    
    def stage1_theme_search(self) -> Dict[int, List[SearchCandidate]]:
        """Stage 1: Search for segments matching each theme"""
        logger.info("="*70)
        logger.info("Stage 1: Theme-level semantic search")
        logger.info(f"Searching with {len(self.themes)} themes (x2 languages = {len(self.themes)*2} queries)")
        
        theme_candidates = {}
        
        with get_session() as session:
            # Build project filter if specified
            project_condition = ""
            if self.config.project:
                project_condition = f"AND '{self.config.project}' = ANY(c.projects)"
            
            # Build limit clause
            limit_clause = ""
            if self.config.max_segments_per_theme:
                limit_clause = f"LIMIT {self.config.max_segments_per_theme}"
            elif self.config.test_mode:
                limit_clause = f"LIMIT {self.config.test_limit}"
            
            total_found = 0
            
            # Search for each theme
            for theme_id, theme_embeddings in tqdm(self.theme_embeddings.items(), desc="Theme searches"):
                theme_info = self.themes[theme_id]
                logger.info(f"Searching for Theme {theme_id}: {theme_info['name']}")
                
                candidates_for_theme = []
                seen_segments = set()
                
                # Search with both language embeddings
                for lang, embedding in [('en', theme_embeddings['en']), ('fr', theme_embeddings['fr'])]:
                    query = text(f"""
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
                            array_agg(s.speaker_id ORDER BY array_position(es.source_sentence_ids, s.sentence_index)) as speaker_ids
                        FROM embedding_segments es
                        JOIN content c ON es.content_id = c.id
                        LEFT JOIN sentences s ON s.sentence_index = ANY(es.source_sentence_ids) AND s.content_id = es.content_id
                        WHERE es.embedding_alt IS NOT NULL
                        AND 1 - (es.embedding_alt <=> CAST(:embedding AS vector)) >= :threshold
                        {project_condition}
                        GROUP BY es.id, es.content_id, es.text, es.start_time, es.end_time,
                                 es.segment_index, es.source_sentence_ids,
                                 c.title, c.channel_name, es.embedding_alt
                        ORDER BY similarity DESC
                        {limit_clause}
                    """)
                    
                    params = {
                        'embedding': '[' + ','.join(map(str, embedding)) + ']',
                        'threshold': self.config.theme_similarity_threshold
                    }
                    
                    result = session.execute(query, params)
                    rows = result.fetchall()
                    
                    logger.debug(f"  {lang.upper()} query found {len(rows)} segments")
                    
                    for row in rows:
                        segment_id = row[0]
                        if segment_id in seen_segments:
                            continue
                        seen_segments.add(segment_id)
                        
                        # Format speaker attribution
                        sentence_texts = row[10] if row[10] else []
                        speaker_ids = row[11] if row[11] else []
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
                            speaker_attributed_text=speaker_attributed_text,
                            source_sentence_ids=row[9],
                            sentence_texts=sentence_texts,
                            speaker_ids=speaker_ids
                        )
                        
                        # Store theme similarity
                        candidate.theme_similarities[theme_id] = row[8]
                        candidates_for_theme.append(candidate)
                
                theme_candidates[theme_id] = candidates_for_theme
                total_found += len(candidates_for_theme)
                logger.info(f"  Found {len(candidates_for_theme)} unique segments for Theme {theme_id}")
        
        logger.info(f"Stage 1 complete: {total_found} total segment-theme matches")
        return theme_candidates
    
    def stage2_subtheme_refinement(self, theme_candidates: Dict[int, List[SearchCandidate]]) -> List[SearchCandidate]:
        """Stage 2: Refine with subtheme-level search"""
        logger.info("="*70)
        logger.info("Stage 2: Subtheme-level refinement")
        
        all_candidates = {}  # Use dict to deduplicate by segment_id
        
        for theme_id, candidates in theme_candidates.items():
            if not candidates:
                continue
            
            theme_info = self.themes[theme_id]
            logger.info(f"Refining {len(candidates)} candidates for Theme {theme_id}: {theme_info['name']}")
            
            # Get subthemes for this theme
            if theme_id not in self.subthemes:
                logger.info(f"  No subthemes for Theme {theme_id}")
                # Add candidates without subtheme classification
                for candidate in candidates:
                    if candidate.segment_id not in all_candidates:
                        all_candidates[candidate.segment_id] = candidate
                continue
            
            theme_subthemes = self.subthemes[theme_id]
            logger.info(f"  Checking {len(theme_subthemes)} subthemes")
            
            # For each candidate, check similarity with subthemes
            for candidate in candidates:
                # Parse segment embedding
                segment_embedding = None
                
                # We need to get the embedding from database
                with get_session() as session:
                    query = text("""
                        SELECT embedding_alt 
                        FROM embedding_segments 
                        WHERE id = :segment_id
                    """)
                    result = session.execute(query, {'segment_id': candidate.segment_id})
                    row = result.fetchone()
                    if row and row[0]:
                        if isinstance(row[0], str):
                            embedding_str = row[0].strip('[]')
                            segment_embedding = np.array([float(x) for x in embedding_str.split(',')])
                        else:
                            segment_embedding = np.array(row[0])
                
                if segment_embedding is None:
                    continue
                
                # Check similarity with each subtheme
                for subtheme in theme_subthemes:
                    subtheme_id = subtheme['id']
                    subtheme_emb = self.subtheme_embeddings[subtheme_id]
                    
                    # Calculate similarity with both language embeddings
                    sim_en = np.dot(segment_embedding, subtheme_emb['en'])
                    sim_fr = np.dot(segment_embedding, subtheme_emb['fr'])
                    max_sim = max(sim_en, sim_fr)
                    
                    if max_sim >= self.config.subtheme_similarity_threshold:
                        candidate.subtheme_similarities[subtheme_id] = max_sim
                        logger.debug(f"    Segment {candidate.segment_id} matches subtheme {subtheme_id} (sim: {max_sim:.3f})")
                
                # Add to all candidates if not already present
                if candidate.segment_id not in all_candidates:
                    all_candidates[candidate.segment_id] = candidate
                else:
                    # Merge similarities if segment already exists
                    existing = all_candidates[candidate.segment_id]
                    existing.theme_similarities.update(candidate.theme_similarities)
                    existing.subtheme_similarities.update(candidate.subtheme_similarities)
        
        candidates_list = list(all_candidates.values())
        logger.info(f"Stage 2 complete: {len(candidates_list)} unique segments with theme/subtheme matches")
        
        # Log statistics
        with_subthemes = len([c for c in candidates_list if c.subtheme_similarities])
        logger.info(f"  Segments with subthemes: {with_subthemes}")
        logger.info(f"  Segments with themes only: {len(candidates_list) - with_subthemes}")
        
        return candidates_list
    
    async def stage3_llm_validation(self, candidates: List[SearchCandidate]) -> List[SearchCandidate]:
        """Stage 3: LLM validation of semantic matches"""
        logger.info("="*70)
        logger.info("Stage 3: LLM validation")
        logger.info(f"Validating {len(candidates)} candidates")
        
        # Prepare validation requests
        validation_requests = []
        request_mapping = []  # Track which validation belongs to which candidate/theme/subtheme
        
        for candidate_idx, candidate in enumerate(candidates):
            text_to_validate = candidate.speaker_attributed_text or candidate.segment_text
            
            # Validate theme matches
            for theme_id, similarity in candidate.theme_similarities.items():
                theme_info = self.themes[theme_id]
                
                # Create validation prompt
                prompt = f"""Is the following text discussing {theme_info['description'].lower()}?

Text: {text_to_validate[:1500]}

Answer with just YES or NO."""
                
                validation_requests.append({
                    'segment_id': int(candidate.segment_id),
                    'text': text_to_validate,
                    'prompt': prompt,
                    'context': {
                        'type': 'theme',
                        'theme_id': int(theme_id),
                        'similarity': float(similarity)
                    }
                })
                request_mapping.append((candidate_idx, 'theme', theme_id, None))
            
            # Validate subtheme matches
            for subtheme_id, similarity in candidate.subtheme_similarities.items():
                subtheme_info = None
                theme_id = self.subtheme_embeddings[subtheme_id]['theme_id']
                
                # Find subtheme info
                for st in self.subthemes[theme_id]:
                    if st['id'] == subtheme_id:
                        subtheme_info = st
                        break
                
                if subtheme_info:
                    prompt = f"""Is the following text discussing {subtheme_info['description'].lower()}?

Text: {text_to_validate[:1500]}

Answer with just YES or NO."""
                    
                    validation_requests.append({
                        'segment_id': int(candidate.segment_id),
                        'text': text_to_validate,
                        'prompt': prompt,
                        'context': {
                            'type': 'subtheme',
                            'subtheme_id': str(subtheme_id),
                            'theme_id': int(theme_id),
                            'similarity': float(similarity)
                        }
                    })
                    request_mapping.append((candidate_idx, 'subtheme', theme_id, subtheme_id))
        
        logger.info(f"Processing {len(validation_requests)} validation requests")
        
        # Process in batches
        BATCH_SIZE = 100
        batches = [validation_requests[i:i+BATCH_SIZE] for i in range(0, len(validation_requests), BATCH_SIZE)]
        mapping_batches = [request_mapping[i:i+BATCH_SIZE] for i in range(0, len(request_mapping), BATCH_SIZE)]
        
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(300.0),
            limits=httpx.Limits(max_connections=10)
        ) as client:
            
            for batch_idx, (batch, mapping_batch) in enumerate(tqdm(
                zip(batches, mapping_batches), 
                desc="Validation batches", 
                total=len(batches)
            )):
                try:
                    response = await client.post(
                        f"{self.config.llm_endpoint}/batch-classify",
                        json={
                            'phase': 'validation',
                            'batch_idx': batch_idx,
                            'requests': batch
                        }
                    )
                    
                    if response.status_code != 200:
                        logger.error(f"Validation batch {batch_idx} failed: {response.text}")
                        continue
                    
                    batch_results = response.json()
                    
                    # Process validation results
                    for result, (candidate_idx, val_type, theme_id, subtheme_id) in zip(
                        batch_results['results'], 
                        mapping_batch
                    ):
                        if candidate_idx >= len(candidates):
                            continue
                        
                        candidate = candidates[candidate_idx]
                        
                        # Parse YES/NO response
                        response_text = result.get('response', '').strip().upper()
                        is_valid = 'YES' in response_text
                        
                        if val_type == 'theme' and is_valid:
                            if theme_id not in candidate.validated_themes:
                                candidate.validated_themes.append(theme_id)
                                candidate.validation_confidence[f"theme_{theme_id}"] = candidate.theme_similarities[theme_id]
                        
                        elif val_type == 'subtheme' and is_valid:
                            if subtheme_id not in candidate.validated_subthemes:
                                candidate.validated_subthemes.append(subtheme_id)
                                candidate.validation_confidence[f"subtheme_{subtheme_id}"] = candidate.subtheme_similarities[subtheme_id]
                
                except Exception as e:
                    logger.error(f"Error processing validation batch {batch_idx}: {e}")
                    continue
        
        # Filter to only validated candidates
        validated_candidates = [c for c in candidates if c.validated_themes or c.validated_subthemes]
        
        logger.info(f"Stage 3 complete: {len(validated_candidates)}/{len(candidates)} candidates validated")
        
        # Log validation statistics
        theme_validations = sum(len(c.validated_themes) for c in validated_candidates)
        subtheme_validations = sum(len(c.validated_subthemes) for c in validated_candidates)
        logger.info(f"  Validated theme assignments: {theme_validations}")
        logger.info(f"  Validated subtheme assignments: {subtheme_validations}")
        
        return validated_candidates
    
    def save_results(self, candidates: List[SearchCandidate]):
        """Save results to CSV"""
        logger.info(f"Saving results to {self.config.output_csv}")
        
        with open(self.config.output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            header = [
                'segment_id', 'content_id', 'episode_title', 'episode_channel',
                'start_time', 'end_time', 'themes', 'subthemes', 
                'theme_similarities', 'subtheme_similarities',
                'validation_confidence', 'segment_text', 'speaker_attributed_text'
            ]
            writer.writerow(header)
            
            # Write data
            for candidate in candidates:
                # Format themes
                theme_names = []
                for theme_id in candidate.validated_themes:
                    if theme_id in self.themes:
                        theme_names.append(f"{theme_id}:{self.themes[theme_id]['name']}")
                
                # Format subthemes
                subtheme_names = []
                for subtheme_id in candidate.validated_subthemes:
                    # Find subtheme name
                    for theme_id, subthemes in self.subthemes.items():
                        for st in subthemes:
                            if st['id'] == subtheme_id:
                                subtheme_names.append(f"{subtheme_id}:{st['name']}")
                                break
                
                # Format similarities
                theme_sims = ', '.join([f"{tid}:{sim:.3f}" for tid, sim in candidate.theme_similarities.items()])
                subtheme_sims = ', '.join([f"{sid}:{sim:.3f}" for sid, sim in candidate.subtheme_similarities.items()])
                
                # Format confidence
                confidence_str = ', '.join([f"{k}:{v:.3f}" for k, v in candidate.validation_confidence.items()])
                
                row = [
                    candidate.segment_id,
                    candidate.content_id,
                    candidate.episode_title,
                    candidate.episode_channel,
                    candidate.start_time,
                    candidate.end_time,
                    ', '.join(theme_names),
                    ', '.join(subtheme_names),
                    theme_sims,
                    subtheme_sims,
                    confidence_str,
                    candidate.segment_text,
                    candidate.speaker_attributed_text or candidate.segment_text
                ]
                writer.writerow(row)
        
        logger.info(f"Saved {len(candidates)} results")
    
    async def run_async(self) -> Dict[str, Any]:
        """Run the complete pipeline"""
        logger.info("="*70)
        logger.info("Starting Embedding-Focused Theme Classification")
        logger.info(f"Theme similarity threshold: {self.config.theme_similarity_threshold}")
        logger.info(f"Subtheme similarity threshold: {self.config.subtheme_similarity_threshold}")
        if self.config.project:
            logger.info(f"Project filter: {self.config.project}")
        if self.config.test_mode:
            logger.info(f"TEST MODE: Limited to {self.config.test_limit} segments per theme")
        logger.info("="*70)
        
        try:
            # Stage 1: Theme-level search
            theme_candidates = self.stage1_theme_search()
            
            # Stage 2: Subtheme refinement
            refined_candidates = self.stage2_subtheme_refinement(theme_candidates)
            
            if not refined_candidates:
                logger.warning("No candidates found after refinement")
                return {'status': 'no_candidates'}
            
            # Stage 3: LLM validation
            validated_candidates = await self.stage3_llm_validation(refined_candidates)
            
            # Save results
            self.save_results(validated_candidates)
            
            # Summary statistics
            total_segments = len(validated_candidates)
            segments_with_themes = len([c for c in validated_candidates if c.validated_themes])
            segments_with_subthemes = len([c for c in validated_candidates if c.validated_subthemes])
            
            logger.info("="*70)
            logger.info("Pipeline completed successfully")
            logger.info(f"Total validated segments: {total_segments}")
            logger.info(f"Segments with themes: {segments_with_themes}")
            logger.info(f"Segments with subthemes: {segments_with_subthemes}")
            logger.info(f"Output: {self.config.output_csv}")
            logger.info("="*70)
            
            return {
                'status': 'success',
                'total_segments': total_segments,
                'with_themes': segments_with_themes,
                'with_subthemes': segments_with_subthemes,
                'output_file': self.config.output_csv
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'status': 'error', 'message': str(e)}
    
    def run(self) -> Dict[str, Any]:
        """Synchronous wrapper for async pipeline"""
        return asyncio.run(self.run_async())

def main():
    parser = argparse.ArgumentParser(description='Embedding-focused theme classification with LLM validation')
    
    # Required arguments
    parser.add_argument('--subthemes-csv', required=True,
                       help='Path to subthemes_complete.csv file')
    parser.add_argument('--output-csv', required=True,
                       help='Output CSV file path')
    
    # Search parameters
    parser.add_argument('--theme-threshold', type=float, default=0.60,
                       help='Similarity threshold for theme matching (default: 0.60)')
    parser.add_argument('--subtheme-threshold', type=float, default=0.65,
                       help='Similarity threshold for subtheme matching (default: 0.65)')
    
    # Model configuration
    parser.add_argument('--embedding-model', default='Qwen/Qwen3-Embedding-4B',
                       help='Embedding model (default: Qwen/Qwen3-Embedding-4B)')
    parser.add_argument('--llm-model', default='qwen3:4b-instruct',
                       help='LLM model for validation (default: qwen3:4b-instruct)')
    parser.add_argument('--llm-endpoint', default='http://localhost:8002',
                       help='LLM endpoint URL (default: http://localhost:8002)')
    
    # Processing options
    parser.add_argument('--project', help='Filter to specific project (e.g., CPRMV)')
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for LLM validation (default: 100)')
    parser.add_argument('--max-segments', type=int,
                       help='Max segments per theme (default: no limit)')
    
    # Test mode
    parser.add_argument('--test-mode', action='store_true',
                       help='Enable test mode with limited segments')
    parser.add_argument('--test-limit', type=int, default=100,
                       help='Number of segments per theme in test mode (default: 100)')
    
    # Debug
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create configuration
    config = EmbeddingConfig(
        subthemes_csv=args.subthemes_csv,
        output_csv=args.output_csv,
        theme_similarity_threshold=args.theme_threshold,
        subtheme_similarity_threshold=args.subtheme_threshold,
        embedding_model=args.embedding_model,
        llm_model=args.llm_model,
        llm_endpoint=args.llm_endpoint,
        project=args.project,
        batch_size=args.batch_size,
        max_segments_per_theme=args.max_segments,
        test_mode=args.test_mode,
        test_limit=args.test_limit
    )
    
    # Run pipeline
    classifier = EmbeddingThemeClassifier(config)
    results = classifier.run()
    
    # Print summary
    if results['status'] == 'success':
        print(f"\n✅ Classification completed!")
        print(f"📊 Total segments: {results['total_segments']}")
        print(f"🎯 With themes: {results['with_themes']}")
        print(f"🔍 With subthemes: {results['with_subthemes']}")
        print(f"📁 Output: {results['output_file']}")
    else:
        print(f"\n❌ Classification failed: {results.get('message', results['status'])}")

if __name__ == "__main__":
    main()