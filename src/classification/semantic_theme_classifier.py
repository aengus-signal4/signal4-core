#!/usr/bin/env python3
"""
Semantic-Only Theme Classification Pipeline

Pure semantic search approach without keywords:
- Stage 1: FAISS semantic search for themes and subthemes (threshold 0.38)
- Stage 2: DELETED (no LLM theme classification)
- Stage 3: LLM subtheme validation for matched themes (allows "0/none" responses)
- Stage 4: Final validation with Likert scale

Usage:
    python semantic_theme_classifier.py \
        --subthemes-csv projects/CPRMV/subthemes_final.csv \
        --output-csv projects/CPRMV/outputs/semantic_results.csv \
        --project CPRMV \
        --use-faiss \
        --model tier_1 \
        --similarity-threshold 0.38
"""

import sys
import os
import csv
import json
import logging
import numpy as np
import pandas as pd
import argparse
import asyncio
from pathlib import Path
from src.utils.paths import get_project_root, get_config_path
from typing import List, Dict, Set, Optional
from tqdm import tqdm
from datetime import datetime

# Add project root to path
sys.path.append(str(get_project_root()))

from src.utils.logger import setup_worker_logger
from src.database.session import get_session
from sentence_transformers import SentenceTransformer
from sqlalchemy import text

# Import shared components
from src.classification.core import (
    SearchCandidate,
    ClassificationResult,
    SemanticConfig,
    FAISSIndexLoader,
    LLMClassifier,
    enrich_segments_with_metadata,
    DatabaseWriter
)
from src.classification.core.database_utils import reconstruct_speaker_text

logger = setup_worker_logger('semantic_theme_classifier')


class SemanticThemeClassifier:
    """Semantic-only theme classifier using FAISS and LLM with database-backed progress"""

    def __init__(self, config: SemanticConfig, schema_name: str, schema_version: str):
        self.config = config
        self.schema_name = schema_name
        self.schema_version = schema_version

        # Lazy-loaded models
        self.embedding_model = None
        self.faiss_loader = None
        self.classifier = None

        # Database writer for results
        self.db_writer = DatabaseWriter(
            schema_name=schema_name,
            schema_version=schema_version,
            batch_size=50  # Commit every 50 records
        )

        # Load theme descriptions
        self.theme_descriptions = self._load_theme_descriptions(config.subthemes_csv)

        logger.info(f"Initialized semantic theme classifier")
        logger.info(f"  Schema: {schema_name} v{schema_version}")
        logger.info(f"  Themes: {len(self.theme_descriptions['themes'])}")
        logger.info(f"  Subthemes: {len(self.theme_descriptions['subthemes'])}")
        logger.info(f"  Similarity threshold: {config.similarity_threshold}")

    def _load_theme_descriptions(self, csv_path: str) -> Dict:
        """Load theme and subtheme descriptions from CSV"""
        df = pd.read_csv(csv_path)

        themes = {}
        subthemes = {}

        for theme_id in df['theme_id'].unique():
            theme_id = int(theme_id)  # Convert numpy int64 to native Python int
            theme_df = df[df['theme_id'] == theme_id]
            theme_info = theme_df.iloc[0]

            themes[theme_id] = {
                'id': theme_id,
                'name': theme_info['theme_name'],
                'description_en': theme_info['theme_description'],
                'description_fr': theme_info.get('theme_description_fr', theme_info['theme_description'])
            }

            # Collect subthemes
            for _, row in theme_df.iterrows():
                if pd.notna(row.get('subtheme_id')):
                    subtheme_id = str(row['subtheme_id'])  # Keep as string (e.g., "Q1", "Q2")
                    subthemes[subtheme_id] = {
                        'id': subtheme_id,
                        'theme_id': theme_id,
                        'name': row['subtheme_name'],
                        'description_en': row.get('subtheme_description_short', row['subtheme_description']),
                        'description_fr': row.get('subtheme_description_short_fr', row.get('subtheme_description_short', row['subtheme_description']))
                    }

        logger.info(f"Loaded {len(themes)} themes and {len(subthemes)} subthemes")

        return {
            'themes': themes,
            'subthemes': subthemes
        }

    def _load_embedding_model(self):
        """Lazy load embedding model"""
        if self.embedding_model is not None:
            return

        logger.info(f"Loading embedding model: {self.config.embedding_model}")
        device = 'cpu'

        if 'Qwen' in self.config.embedding_model and 'Embedding-4B' in self.config.embedding_model:
            self.embedding_model = SentenceTransformer(
                self.config.embedding_model,
                trust_remote_code=True,
                truncate_dim=2000,
                device=device
            )
        else:
            self.embedding_model = SentenceTransformer(self.config.embedding_model, device=device)

        logger.info("Embedding model loaded")

    def _load_faiss_index(self):
        """Lazy load FAISS index"""
        if self.faiss_loader is not None:
            return

        logger.info("Loading FAISS index...")
        limit = self.config.test_mode if self.config.test_mode else None
        self.faiss_loader = FAISSIndexLoader(
            project=self.config.project,
            embedding_dim=2000,
            limit=limit
        )
        self.faiss_loader.load_or_build_index(self.config.faiss_index_path)
        logger.info("FAISS index loaded")

    def _load_llm_classifier(self):
        """Lazy load LLM classifier"""
        if self.classifier is not None:
            return

        logger.info("Loading LLM classifier...")
        self.classifier = LLMClassifier(
            themes_csv=self.config.subthemes_csv,
            model_name=self.config.model_name,
            use_gpu=self.config.use_gpu,
            project=self.config.project,
            model_server_url=self.config.model_server_url,
            use_model_server=self.config.use_model_server
        )
        logger.info("LLM classifier loaded")

    def get_processed_segments(self) -> Set[int]:
        """Get processed segments from database"""
        return self.db_writer.get_processed_segments()

    def _compute_query_embeddings(self) -> Dict:
        """Compute embeddings for all subtheme queries (EN + FR)"""
        logger.info("Computing embeddings for subtheme queries...")
        query_embeddings = {}

        for subtheme_id, subtheme_info in self.theme_descriptions['subthemes'].items():
            query_embeddings[subtheme_id] = {}
            for lang, desc in [('en', subtheme_info['description_en']), ('fr', subtheme_info['description_fr'])]:
                query_text = f"Instruct: Retrieve relevant passages.\nQuery: {desc}"
                embedding = self.embedding_model.encode(
                    query_text,
                    convert_to_numpy=True,
                    normalize_embeddings=False
                )
                # Convert numpy array to list for JSON storage
                query_embeddings[subtheme_id][lang] = embedding.tolist()

        logger.info(f"Computed embeddings for {len(query_embeddings)} subthemes (EN + FR)")
        return query_embeddings

    def _load_or_compute_query_embeddings(self) -> Dict:
        """Load cached query embeddings from schema or compute them"""
        # Check if schema has cached embeddings
        cached = self.db_writer.schema_data.get('query_embeddings')

        if cached:
            logger.info("Using cached query embeddings from schema")
            return cached

        # Compute embeddings
        logger.info("No cached embeddings found, computing...")
        self._load_embedding_model()
        return self._compute_query_embeddings()

    def stage1_semantic_search(self) -> List[SearchCandidate]:
        """
        Stage 1: Pure semantic search for subthemes ONLY (not themes).
        A segment qualifies if it matches ANY subtheme with score >= threshold.
        Theme assignment is inferred from matched subthemes.
        """
        logger.info("Stage 1: Semantic search for subthemes only (32 queries: 16 EN + 16 FR)")
        logger.info(f"  Threshold: {self.config.similarity_threshold}")

        # Load FAISS index
        self._load_faiss_index()

        # Load or compute query embeddings (loads from cache if available)
        query_embeddings = self._load_or_compute_query_embeddings()

        # Load processed segments from database
        processed_segment_ids = self.get_processed_segments()

        # Track results: segment_id -> {subtheme_scores}
        segment_matches = {}

        threshold = self.config.similarity_threshold
        search_k = 100000 if not self.config.test_mode else min(1000, self.config.test_mode * 10)

        logger.info(f"Searching with k={search_k}")

        # Search subthemes ONLY (EN + FR)
        logger.info("Searching subtheme descriptions...")
        query_count = 0
        for subtheme_id, lang_embeddings in query_embeddings.items():
            for lang, embedding_list in lang_embeddings.items():
                query_count += 1
                # Convert list back to numpy array
                embedding = np.array(embedding_list, dtype=np.float32)

                scores, indices = self.faiss_loader.search(embedding.reshape(1, -1), k=search_k)

                for score, idx in zip(scores[0], indices[0]):
                    if score >= threshold and idx >= 0:
                        segment_id = self.faiss_loader.segment_id_mapping.get(int(idx))
                        if segment_id and segment_id not in processed_segment_ids:
                            if segment_id not in segment_matches:
                                segment_matches[segment_id] = {'subtheme_scores': {}}
                            # Keep highest score for this subtheme
                            if subtheme_id not in segment_matches[segment_id]['subtheme_scores'] or score > segment_matches[segment_id]['subtheme_scores'][subtheme_id]:
                                segment_matches[segment_id]['subtheme_scores'][subtheme_id] = float(score)

        logger.info(f"  Total queries: {query_count}")
        logger.info(f"  Total segments matched: {len(segment_matches)}")

        # Return query embeddings for caching (if not already cached)
        self._query_embeddings_to_cache = query_embeddings if not self.db_writer.schema_data.get('query_embeddings') else None

        # Convert to SearchCandidate objects
        candidates = []
        for segment_id, matches in segment_matches.items():
            # Infer which themes this segment matched from subtheme matches
            matched_theme_ids = set()
            for subtheme_id in matches['subtheme_scores'].keys():
                theme_id = self.theme_descriptions['subthemes'][subtheme_id]['theme_id']
                matched_theme_ids.add(theme_id)

            # Best subtheme score
            best_score = max(matches['subtheme_scores'].values()) if matches['subtheme_scores'] else 0.0

            candidate = SearchCandidate(
                segment_id=segment_id,
                content_id='',  # Will be filled by enrichment
                episode_title='',
                episode_channel='',
                segment_text='',
                start_time=0.0,
                end_time=0.0,
                segment_index=0,
                similarity_score=best_score,
                matched_via='semantic',
                theme_similarities={},  # No theme-level search
                subtheme_similarities=matches['subtheme_scores'],
                theme_ids=list(matched_theme_ids),
                theme_names=[self.theme_descriptions['themes'][tid]['name'] for tid in matched_theme_ids]
            )
            candidates.append(candidate)

        logger.info(f"Stage 1 complete: {len(candidates)} candidates")

        return candidates

    def stage3_classify_subthemes(self, segment: SearchCandidate) -> SearchCandidate:
        """
        Stage 3: LLM subtheme classification for ALL subthemes of matched themes.

        LLM can respond with:
        - Subtheme numbers (e.g., "1,3,5") - specific subthemes apply
        - "0" - theme applies but no specific subthemes
        - "not applicable" - theme does not apply (will be removed)
        """
        if not segment.theme_ids:
            segment.subtheme_results = {}
            return segment

        text_to_classify = segment.speaker_attributed_text or segment.segment_text
        segment.subtheme_results = {}
        themes_to_remove = []

        for theme_id in segment.theme_ids:
            try:
                result = self.classifier.classify_subtheme(
                    text=text_to_classify,
                    theme_id=theme_id,
                    segment_id=segment.segment_id
                )

                # Check if LLM said theme is not applicable
                if result.reasoning == "Theme not applicable":
                    themes_to_remove.append(theme_id)
                    segment.subtheme_results[theme_id] = {
                        'subtheme_ids': [],
                        'subtheme_names': [],
                        'confidence': 0.0,
                        'reasoning': 'Theme not applicable (removed by LLM)'
                    }
                else:
                    # Store subtheme results for this theme
                    segment.subtheme_results[theme_id] = {
                        'subtheme_ids': result.subtheme_ids or [],
                        'subtheme_names': result.subtheme_names or [],
                        'confidence': result.confidence,
                        'reasoning': result.reasoning
                    }

            except Exception as e:
                logger.error(f"Subtheme classification error for segment {segment.segment_id}, theme {theme_id}: {e}")
                segment.subtheme_results[theme_id] = {
                    'subtheme_ids': [],
                    'subtheme_names': [],
                    'confidence': 0.0,
                    'reasoning': f"Error: {str(e)}"
                }

        # Remove themes that LLM marked as not applicable
        if themes_to_remove:
            segment.theme_ids = [tid for tid in segment.theme_ids if tid not in themes_to_remove]
            logger.debug(f"Segment {segment.segment_id}: Removed {len(themes_to_remove)} themes marked as not applicable")

        return segment

    def stage4_validate(self, segment: SearchCandidate) -> SearchCandidate:
        """Stage 4: Validate theme/subtheme assignments with Likert scale"""
        if not segment.theme_ids:
            segment.validation_results = {}
            return segment

        text_to_classify = segment.speaker_attributed_text or segment.segment_text
        segment.validation_results = {}

        for theme_id in segment.theme_ids:
            theme_info = self.classifier.themes.get(theme_id)
            if not theme_info:
                continue

            # Get subthemes for this theme
            if segment.subtheme_results and theme_id in segment.subtheme_results:
                subtheme_data = segment.subtheme_results[theme_id]
                subtheme_ids = subtheme_data.get('subtheme_ids', [])
            else:
                subtheme_ids = []

            try:
                if subtheme_ids:
                    # Validate each subtheme
                    subthemes_confidence = {}
                    theme_subthemes = self.classifier.subthemes.get(theme_id, [])

                    for sub_id in subtheme_ids:
                        subtheme_info = None
                        for st in theme_subthemes:
                            if st['id'] == sub_id:
                                subtheme_info = st
                                break

                        if not subtheme_info:
                            continue

                        description = subtheme_info['description']
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

                        # Parse Likert scale response
                        if 'definitely matches' in response or 'definitely match' in response:
                            confidence = 1.0
                            category = "definitely_matches"
                        elif 'probably matches' in response or 'probably match' in response:
                            confidence = 0.75
                            category = "probably_matches"
                        elif 'unsure' in response:
                            confidence = 0.5
                            category = "unsure"
                        elif 'probably does not' in response or 'probably doesn\'t' in response:
                            confidence = 0.25
                            category = "probably_does_not_match"
                        else:  # definitely does not match
                            confidence = 0.0
                            category = "definitely_does_not_match"

                        subthemes_confidence[sub_id] = {
                            'confidence': confidence,
                            'category': category,
                            'response': response
                        }

                    segment.validation_results[theme_id] = {
                        'subthemes': subthemes_confidence,
                        'overall_confidence': sum(s['confidence'] for s in subthemes_confidence.values()) / len(subthemes_confidence) if subthemes_confidence else 0.0
                    }

            except Exception as e:
                logger.error(f"Validation error for segment {segment.segment_id}, theme {theme_id}: {e}")
                segment.validation_results[theme_id] = {
                    'error': str(e)
                }

        return segment

    async def process_segment(self, segment: SearchCandidate) -> SearchCandidate:
        """Process a single segment through stages 3-4"""
        # Stage 3: Subtheme classification
        segment = self.stage3_classify_subthemes(segment)

        # Stage 4: Validation
        segment = self.stage4_validate(segment)

        return segment

    async def process_all_segments(self, candidates: List[SearchCandidate]) -> List[SearchCandidate]:
        """Process all segments through stages 3-4 and write to database"""
        logger.info(f"Processing {len(candidates)} segments through stages 3-4...")

        # Load LLM classifier
        self._load_llm_classifier()

        # Process segments in batches and write to DB incrementally
        results = []
        batch = []
        batch_size = self.db_writer.batch_size

        with tqdm(total=len(candidates), desc="Processing segments") as pbar:
            for segment in candidates:
                result = await self.process_segment(segment)
                batch.append(result)
                results.append(result)
                pbar.update(1)

                # Write batch to database
                if len(batch) >= batch_size:
                    self.db_writer.write_candidates_batch(batch, commit=True)
                    logger.info(f"Wrote batch of {len(batch)} to database ({len(results)}/{len(candidates)} total)")
                    batch = []

            # Write remaining batch
            if batch:
                self.db_writer.write_candidates_batch(batch, commit=True)
                logger.info(f"Wrote final batch of {len(batch)} to database")

        return results


    def report_stage1_results(self, candidates: List[SearchCandidate]) -> Dict:
        """Generate detailed report of Stage 1 semantic matches"""
        from collections import Counter

        # Count segments by subtheme
        subtheme_counts = Counter()
        for c in candidates:
            for subtheme_id in c.subtheme_similarities.keys():
                subtheme_counts[subtheme_id] += 1

        # Count segments by theme (inferred from subthemes)
        theme_counts = Counter()
        for c in candidates:
            for theme_id in c.theme_ids:
                theme_counts[theme_id] += 1

        # Calculate statistics
        total_segments = len(candidates)
        avg_score = sum(c.similarity_score for c in candidates) / total_segments if total_segments > 0 else 0

        report = {
            'total_segments': total_segments,
            'avg_similarity_score': avg_score,
            'themes': {},
            'subthemes': {}
        }

        # Theme breakdown
        for theme_id, count in sorted(theme_counts.items()):
            theme_info = self.theme_descriptions['themes'][theme_id]
            report['themes'][theme_id] = {
                'name': theme_info['name'],
                'count': count,
                'percentage': (count / total_segments * 100) if total_segments > 0 else 0
            }

        # Subtheme breakdown
        for subtheme_id, count in sorted(subtheme_counts.items()):
            subtheme_info = self.theme_descriptions['subthemes'][subtheme_id]
            theme_id = subtheme_info['theme_id']
            report['subthemes'][subtheme_id] = {
                'name': subtheme_info['name'],
                'theme_id': theme_id,
                'theme_name': self.theme_descriptions['themes'][theme_id]['name'],
                'count': count,
                'percentage': (count / total_segments * 100) if total_segments > 0 else 0
            }

        return report

    def _get_stage1_cache_path(self) -> Path:
        """Get path for Stage 1 candidates cache file"""
        return Path(f"projects/{self.config.project}/{self.schema_name}_{self.schema_version}_stage1_candidates.json")

    def _save_stage1_candidates(self, candidates: List[SearchCandidate]):
        """Save Stage 1 candidates to cache file"""
        cache_path = self._get_stage1_cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to serializable format
        data = []
        for c in candidates:
            data.append({
                'segment_id': c.segment_id,
                'similarity_score': c.similarity_score,
                'matched_via': c.matched_via,
                'theme_similarities': c.theme_similarities,
                'subtheme_similarities': c.subtheme_similarities,
                'theme_ids': c.theme_ids,
                'theme_names': c.theme_names,
            })

        with open(cache_path, 'w') as f:
            json.dump(data, f)
        logger.info(f"Saved {len(candidates)} Stage 1 candidates to {cache_path}")

    def _load_stage1_candidates(self) -> Optional[List[SearchCandidate]]:
        """Load Stage 1 candidates from cache file"""
        cache_path = self._get_stage1_cache_path()
        if not cache_path.exists():
            return None

        with open(cache_path, 'r') as f:
            data = json.load(f)

        # Get already processed segment IDs
        processed_ids = self.get_processed_segments()

        candidates = []
        for item in data:
            if item['segment_id'] in processed_ids:
                continue  # Skip already processed

            candidate = SearchCandidate(
                segment_id=item['segment_id'],
                content_id='',  # Will be filled by enrichment
                episode_title='',
                episode_channel='',
                segment_text='',
                start_time=0.0,
                end_time=0.0,
                segment_index=0,
                similarity_score=item['similarity_score'],
                matched_via=item['matched_via'],
                theme_similarities=item['theme_similarities'],
                subtheme_similarities=item['subtheme_similarities'],
                theme_ids=item['theme_ids'],
                theme_names=item['theme_names'],
            )
            candidates.append(candidate)

        logger.info(f"Loaded {len(candidates)} unprocessed candidates from cache ({len(processed_ids)} already processed)")
        return candidates

    def _row_to_candidate(self, row: Dict, speaker_text: str) -> SearchCandidate:
        """Convert a database row to a SearchCandidate object."""
        # Parse theme_ids from the array - convert to int for consistency
        theme_ids_raw = row.get('theme_ids', [])
        if isinstance(theme_ids_raw, str):
            theme_ids_raw = [theme_ids_raw]
        theme_ids = [int(tid) for tid in theme_ids_raw if tid]

        # Get theme names
        theme_names = []
        for tid in theme_ids:
            if tid in self.theme_descriptions.get('themes', {}):
                theme_names.append(self.theme_descriptions['themes'][tid]['name'])

        return SearchCandidate(
            segment_id=row['segment_id'],
            content_id=str(row.get('content_id', '')),
            episode_title=row.get('episode_title', ''),
            episode_channel=row.get('episode_channel', ''),
            segment_text=row.get('segment_text', ''),
            speaker_attributed_text=speaker_text,
            start_time=row.get('start_time', 0.0),
            end_time=row.get('end_time', 0.0),
            segment_index=row.get('segment_index', 0),
            similarity_score=row.get('max_similarity_score', 0.0),
            matched_via=row.get('matched_via', 'semantic'),
            theme_similarities={},
            subtheme_similarities=row.get('stage1_similarities', {}),
            theme_ids=theme_ids,
            theme_names=theme_names,
        )

    def _process_from_db(self, batch_size: int = 100) -> Dict:
        """
        Process pending candidates directly from database using async concurrent requests.

        This method:
        1. Fetches pending candidates (stage3_results IS NULL) with metadata in one query
        2. Reconstructs speaker-attributed text from speaker_positions
        3. Processes through LLM stages 3-4 with concurrent requests
        4. Updates the row in place

        Args:
            batch_size: Number of candidates to process per batch

        Returns:
            Dict with processing results
        """
        # Run the async version
        return asyncio.run(self._process_from_db_async(batch_size))

    async def _process_from_db_async(self, batch_size: int = 100) -> Dict:
        """Async implementation with batched concurrent LLM requests."""
        import httpx

        self._load_llm_classifier()

        processed = 0
        themes_found = set()

        # Get initial count
        pending_count = self.db_writer.get_pending_count()
        logger.info(f"Processing {pending_count} pending candidates from database (batch_size={batch_size})")
        logger.info(f"Using batched concurrent async requests to model_server")

        # Create progress bar
        pbar = tqdm(total=pending_count, desc="LLM Classification", unit="seg")

        model_server_url = self.config.model_server_url

        # Create async HTTP client with connection pooling
        async with httpx.AsyncClient(timeout=120.0, limits=httpx.Limits(max_connections=20)) as client:
            while True:
                # Fetch batch with metadata
                batch = self.db_writer.get_pending_candidates_with_metadata(batch_size)
                if not batch:
                    break

                # Prepare all candidates in batch
                prepared = []
                for row in batch:
                    try:
                        speaker_text = reconstruct_speaker_text(
                            row.get('segment_text', ''),
                            row.get('speaker_positions')
                        )
                        candidate = self._row_to_candidate(row, speaker_text)
                        prepared.append((row, candidate))
                    except Exception as e:
                        logger.error(f"Error preparing segment {row.get('segment_id')}: {e}")
                        pbar.update(1)
                        processed += 1

                if not prepared:
                    continue

                # ============ STAGE 3: Subtheme Classification ============
                # Flatten all (candidate, theme_id) pairs into individual LLM calls
                stage3_tasks = []
                for row, candidate in prepared:
                    if not candidate.theme_ids:
                        candidate.subtheme_results = {}
                        continue
                    for theme_id in candidate.theme_ids:
                        stage3_tasks.append(
                            self._classify_subtheme_async(client, model_server_url, candidate, theme_id)
                        )

                logger.info(f"Batch: {len(prepared)} candidates → {len(stage3_tasks)} Stage 3 subtheme calls")

                # Fire all Stage 3 calls concurrently
                if stage3_tasks:
                    stage3_results = await asyncio.gather(*stage3_tasks, return_exceptions=True)
                    logger.info(f"Stage 3 complete: {len(stage3_results)} results")

                    # Reassemble results back to candidates
                    result_idx = 0
                    for row, candidate in prepared:
                        if not candidate.theme_ids:
                            continue
                        candidate.subtheme_results = {}
                        themes_to_remove = []
                        for theme_id in candidate.theme_ids:
                            result = stage3_results[result_idx]
                            result_idx += 1
                            if isinstance(result, Exception):
                                logger.error(f"Stage 3 error segment {candidate.segment_id}, theme {theme_id}: {result}")
                                candidate.subtheme_results[theme_id] = {
                                    'subtheme_ids': [], 'subtheme_names': [],
                                    'confidence': 0.0, 'reasoning': f"Error: {result}"
                                }
                            else:
                                candidate.subtheme_results[theme_id] = result
                                if result.get('reasoning') == 'Theme not applicable (removed by LLM)':
                                    themes_to_remove.append(theme_id)
                        # Remove inapplicable themes
                        if themes_to_remove:
                            candidate.theme_ids = [t for t in candidate.theme_ids if t not in themes_to_remove]

                # ============ STAGE 4: Validation ============
                # Flatten validation calls for remaining themes with subthemes
                stage4_tasks = []
                for row, candidate in prepared:
                    candidate.validation_results = {}
                    for theme_id in candidate.theme_ids:
                        subtheme_data = candidate.subtheme_results.get(theme_id, {})
                        if subtheme_data.get('subtheme_ids'):
                            stage4_tasks.append(
                                self._validate_subthemes_async(client, model_server_url, candidate, theme_id)
                            )

                logger.info(f"Stage 4: {len(stage4_tasks)} validation calls")

                # Fire all Stage 4 calls concurrently
                if stage4_tasks:
                    stage4_results = await asyncio.gather(*stage4_tasks, return_exceptions=True)
                    logger.info(f"Stage 4 complete: {len(stage4_results)} results")

                    # Reassemble validation results
                    result_idx = 0
                    for row, candidate in prepared:
                        for theme_id in candidate.theme_ids:
                            subtheme_data = candidate.subtheme_results.get(theme_id, {})
                            if subtheme_data.get('subtheme_ids'):
                                result = stage4_results[result_idx]
                                result_idx += 1
                                if isinstance(result, Exception):
                                    candidate.validation_results[theme_id] = {'likert_score': 3, 'confidence': 0.5}
                                else:
                                    candidate.validation_results[theme_id] = result
                            else:
                                candidate.validation_results[theme_id] = {'likert_score': 4, 'confidence': 0.8}

                # ============ Save all results ============
                for row, candidate in prepared:
                    try:
                        self.db_writer.update_classification_result(
                            row['tc_id'],
                            candidate,
                            embedding=row.get('embedding_alt')
                        )
                        if candidate.theme_ids:
                            themes_found.update(candidate.theme_ids)
                    except Exception as e:
                        logger.error(f"Error saving segment {row.get('segment_id')}: {e}")

                    processed += 1
                    pbar.update(1)

                # Log progress periodically
                if processed % 500 == 0:
                    remaining = self.db_writer.get_pending_count()
                    logger.info(f"Progress: {processed} processed, {remaining} remaining")

        pbar.close()

        logger.info(f"Database processing complete!")
        logger.info(f"  Total segments classified: {processed}")
        logger.info(f"  Unique themes identified: {len(themes_found)}")

        return {
            'status': 'success',
            'total_candidates': processed,
            'themes_identified': len(themes_found),
            'schema_id': self.db_writer.schema_id,
            'schema_name': self.schema_name,
            'schema_version': self.schema_version
        }

    async def _classify_subtheme_async(self, client, model_server_url: str, candidate: SearchCandidate, theme_id: int) -> Dict:
        """Async Stage 3: Classify subthemes for a single theme."""
        text_to_classify = candidate.speaker_attributed_text or candidate.segment_text

        if theme_id not in self.classifier.subthemes:
            return {
                'subtheme_ids': [], 'subtheme_names': [],
                'confidence': 0.0, 'reasoning': f"No subthemes for theme {theme_id}"
            }

        # Build prompt
        subtheme_mapping = {}
        subthemes_list = []
        for i, st in enumerate(self.classifier.subthemes[theme_id], 1):
            subtheme_mapping[i] = st['id']
            subthemes_list.append(f"{i}. {st['description']}")

        theme_info = self.classifier.themes[theme_id]
        prompt = f"""Classify the following text segment into relevant sub-themes for the theme "{theme_info['name']}".

INSTRUCTIONS:
- If the theme does not apply to this text at all, respond with "not applicable"
- If the theme applies but no specific sub-themes match, respond with "0" (theme only)
- If specific sub-themes apply, respond with their numbers separated by commas (e.g., "1,3,5")
- You may select multiple sub-themes if applicable

SUB-THEMES for {theme_info['name']}:
{chr(10).join(subthemes_list)}

TEXT TO CLASSIFY:
{text_to_classify}

RESPONSE (sub-theme numbers, "0" for theme only, or "not applicable"):"""

        # Make async LLM request
        response = await client.post(
            f"{model_server_url}/llm-request",
            json={
                "messages": [
                    {"role": "system", "content": "You are a precise classification assistant. Respond only with the requested numbers."},
                    {"role": "user", "content": prompt}
                ],
                "model": self.config.model_name,
                "priority": 2,
                "temperature": 0.1,
                "max_tokens": 50,
                "top_p": 0.9
            }
        )
        response.raise_for_status()
        result_text = response.json().get('response', '').strip().lower()

        # Parse response
        if 'not applicable' in result_text:
            return {
                'subtheme_ids': [], 'subtheme_names': [],
                'confidence': 0.0, 'reasoning': 'Theme not applicable (removed by LLM)'
            }
        elif result_text == '0':
            return {
                'subtheme_ids': [], 'subtheme_names': [],
                'confidence': 0.5, 'reasoning': 'Theme applies but no specific subthemes identified'
            }
        else:
            # Parse subtheme numbers
            subtheme_ids = []
            subtheme_names = []
            for part in result_text.replace(' ', '').split(','):
                try:
                    num = int(part)
                    if num in subtheme_mapping:
                        st_id = subtheme_mapping[num]
                        subtheme_ids.append(st_id)
                        for st in self.classifier.subthemes[theme_id]:
                            if st['id'] == st_id:
                                subtheme_names.append(st['name'])
                                break
                except ValueError:
                    continue

            return {
                'subtheme_ids': subtheme_ids,
                'subtheme_names': subtheme_names,
                'confidence': 0.8 if subtheme_ids else 0.5,
                'reasoning': f"Identified {len(subtheme_ids)} subthemes"
            }

    async def _validate_subthemes_async(self, client, model_server_url: str, candidate: SearchCandidate, theme_id: int) -> Dict:
        """Async Stage 4: Validate subtheme assignments with Likert scale."""
        text_to_classify = candidate.speaker_attributed_text or candidate.segment_text
        subtheme_data = candidate.subtheme_results.get(theme_id, {})
        subtheme_ids = subtheme_data.get('subtheme_ids', [])

        if not subtheme_ids:
            return {'likert_score': 4, 'confidence': 0.8}

        # Build validation prompt
        theme_info = self.classifier.themes[theme_id]
        subtheme_names = subtheme_data.get('subtheme_names', [])

        prompt = f"""Rate how well the following text matches the theme "{theme_info['name']}" and subthemes: {', '.join(subtheme_names)}.

TEXT:
{text_to_classify}

Rate on a scale of 1-5:
1 = Strongly disagree (no match)
2 = Disagree
3 = Neutral
4 = Agree
5 = Strongly agree (perfect match)

RESPONSE (number 1-5 only):"""

        response = await client.post(
            f"{model_server_url}/llm-request",
            json={
                "messages": [
                    {"role": "system", "content": "You are a precise rating assistant. Respond only with a number 1-5."},
                    {"role": "user", "content": prompt}
                ],
                "model": self.config.model_name,
                "priority": 3,  # Lower priority than classification
                "temperature": 0.1,
                "max_tokens": 10,
                "top_p": 0.9
            }
        )
        response.raise_for_status()
        result_text = response.json().get('response', '').strip()

        # Parse Likert score
        try:
            score = int(result_text[0]) if result_text else 4
            score = max(1, min(5, score))
        except (ValueError, IndexError):
            score = 4

        return {
            'likert_score': score,
            'confidence': 0.8 if score >= 4 else 0.5
        }

    async def _process_candidate_async(self, client, row: Dict, candidate: SearchCandidate) -> SearchCandidate:
        """Process a single candidate through LLM stages asynchronously."""
        if not candidate.theme_ids:
            candidate.subtheme_results = {}
            candidate.validation_results = {}
            return candidate

        text_to_classify = candidate.speaker_attributed_text or candidate.segment_text
        model_server_url = self.config.model_server_url

        # Stage 3: Classify subthemes for each theme
        candidate.subtheme_results = {}
        themes_to_remove = []

        for theme_id in candidate.theme_ids:
            try:
                # Build subtheme prompt
                if theme_id not in self.classifier.subthemes:
                    candidate.subtheme_results[theme_id] = {
                        'subtheme_ids': [],
                        'subtheme_names': [],
                        'confidence': 0.0,
                        'reasoning': f"No subthemes for theme {theme_id}"
                    }
                    continue

                subtheme_mapping = {}
                subthemes_list = []
                for i, st in enumerate(self.classifier.subthemes[theme_id], 1):
                    subtheme_mapping[i] = st['id']
                    subthemes_list.append(f"{i}. {st['description']}")

                theme_info = self.classifier.themes[theme_id]
                prompt = f"""Classify the following text segment into relevant sub-themes for the theme "{theme_info['name']}".

INSTRUCTIONS:
- If the theme does not apply to this text at all, respond with "not applicable"
- If the theme applies but no specific sub-themes match, respond with "0" (theme only)
- If specific sub-themes apply, respond with their numbers separated by commas (e.g., "1,3,5")
- You may select multiple sub-themes if applicable

SUB-THEMES for {theme_info['name']}:
{chr(10).join(subthemes_list)}

TEXT TO CLASSIFY:
{text_to_classify}

RESPONSE (sub-theme numbers, "0" for theme only, or "not applicable"):"""

                # Make async LLM request
                response = await client.post(
                    f"{model_server_url}/llm-request",
                    json={
                        "messages": [
                            {"role": "system", "content": "You are a precise classification assistant. Respond only with the requested numbers."},
                            {"role": "user", "content": prompt}
                        ],
                        "model": self.config.model_name,
                        "priority": 2,
                        "temperature": 0.1,
                        "max_tokens": 50,
                        "top_p": 0.9
                    }
                )
                response.raise_for_status()
                result_text = response.json().get('response', '').strip().lower()

                # Parse response
                if 'not applicable' in result_text:
                    themes_to_remove.append(theme_id)
                    candidate.subtheme_results[theme_id] = {
                        'subtheme_ids': [],
                        'subtheme_names': [],
                        'confidence': 0.0,
                        'reasoning': 'Theme not applicable (removed by LLM)'
                    }
                elif result_text == '0':
                    candidate.subtheme_results[theme_id] = {
                        'subtheme_ids': [],
                        'subtheme_names': [],
                        'confidence': 0.5,
                        'reasoning': 'Theme applies but no specific subthemes identified'
                    }
                else:
                    # Parse subtheme numbers
                    subtheme_ids = []
                    subtheme_names = []
                    for part in result_text.replace(' ', '').split(','):
                        try:
                            num = int(part)
                            if num in subtheme_mapping:
                                st_id = subtheme_mapping[num]
                                subtheme_ids.append(st_id)
                                # Find subtheme name
                                for st in self.classifier.subthemes[theme_id]:
                                    if st['id'] == st_id:
                                        subtheme_names.append(st['name'])
                                        break
                        except ValueError:
                            continue

                    candidate.subtheme_results[theme_id] = {
                        'subtheme_ids': subtheme_ids,
                        'subtheme_names': subtheme_names,
                        'confidence': 0.8 if subtheme_ids else 0.5,
                        'reasoning': f"Identified {len(subtheme_ids)} subthemes"
                    }

            except Exception as e:
                logger.error(f"Subtheme classification error for segment {candidate.segment_id}, theme {theme_id}: {e}")
                candidate.subtheme_results[theme_id] = {
                    'subtheme_ids': [],
                    'subtheme_names': [],
                    'confidence': 0.0,
                    'reasoning': f"Error: {str(e)}"
                }

        # Remove themes marked as not applicable
        if themes_to_remove:
            candidate.theme_ids = [tid for tid in candidate.theme_ids if tid not in themes_to_remove]

        # Stage 4: Validation (simplified - just mark as validated)
        candidate.validation_results = {}
        for theme_id in candidate.theme_ids:
            candidate.validation_results[theme_id] = {
                'likert_score': 4,  # Default to "agree"
                'confidence': 0.8
            }

        return candidate

    # ============ Stage 5: FINAL CHECK ============

    async def _final_check_async(self, client, model_server_url: str, text: str, language: str) -> Dict:
        """
        Stage 5 FINAL CHECK: Verify segment relates to gender-based discourse.

        Uses tier_1 model with academic definition prompt.

        Args:
            client: httpx async client
            model_server_url: URL to model server
            text: Segment text to check
            language: 'en' or 'fr'

        Returns:
            Dict with 'is_relevant' and 'reasoning'
        """
        import re

        # Truncate text if too long
        text = text[:2500] if len(text) > 2500 else text

        # Build prompt
        if language == 'fr':
            prompt = f"""Nous recherchons tous les propos, représentations ou cadrages discursifs qui mobilisent le genre, le sexe ou l'orientation sexuelle pour dénigrer, discréditer, hiérarchiser, disqualifier ou exclure des individus ou des groupes. Cela inclut tant les formes explicites (insultes, menaces, appels à l'exclusion sociale ou à la violence) que les formes ordinaires ou plus implicites qui naturalisent, légitiment ou banalisent les inégalités et les rapports de domination entre les genres et les sexes ou à l'égard des personnes issues des communautés LGBTQ+.

RECHERCHEZ ces éléments spécifiques:
- Affirmations sur les femmes, les hommes, les personnes trans ou LGBTQ+ en tant que groupes
- Affirmations sur le féminisme, les identités trans, ou les droits LGBTQ+
- Langage sur les rôles de genre, les différences biologiques entre sexes, ou l'orientation sexuelle
- Termes comme: trans, genre, féministe, gay, lesbienne, non-binaire, femmes, masculinité, etc.

Le cadrage rhétorique ne change pas la nature du contenu — un texte est pertinent même s'il est présenté comme liberté d'expression, opinion politique, position scientifique ou valeurs traditionnelles.

TEXTE:
{text}

Répondez en JSON avec ce format exact:
{{"reasoning": "votre explication brève", "relevance": "définitivement|probablement|possiblement|probablement pas|définitivement pas"}}"""
        else:
            prompt = f"""We are looking for all statements, representations, or discursive frameworks that use gender, sex, or sexual orientation to denigrate, discredit, hierarchize, disqualify, or exclude individuals or groups. It includes both explicit forms (insults, threats, calls for social exclusion or violence) and ordinary or more implicit forms that naturalize, legitimize, or trivialize inequalities and relationships of domination between genders and sexes or towards people from LGBTQ+ communities.

LOOK FOR these specific elements:
- Claims about women, men, trans people, or LGBTQ+ people as groups
- Claims about feminism, trans identities, or LGBTQ+ rights
- Language about gender roles, biological sex differences, or sexual orientation
- Terms like: trans, gender, feminist, gay, lesbian, non-binary, women, masculinity, etc.

Rhetorical framing does not change the nature of the content — a text is relevant even if presented as free speech, political opinion, scientific position, or traditional values.

TEXT:
{text}

Respond in JSON with this exact format:
{{"reasoning": "your brief explanation", "relevance": "definitely|probably|possibly|probably not|definitely not"}}"""

        response = await client.post(
            f"{model_server_url}/llm-request",
            json={
                "messages": [
                    {"role": "system", "content": "You are an expert analyst identifying gender-based discourse."},
                    {"role": "user", "content": prompt}
                ],
                "model": "tier_1",
                "priority": 1,
                "temperature": 0.1,
                "top_p": 0.9
            }
        )
        response.raise_for_status()
        result_text = response.json().get('response', '').strip()

        # Parse JSON response
        relevance = None
        reasoning = result_text

        try:
            # Try to extract JSON from response (may have extra text around it)
            json_match = re.search(r'\{[^{}]*\}', result_text)
            if json_match:
                parsed = json.loads(json_match.group())
                reasoning = parsed.get('reasoning', result_text)
                relevance_raw = parsed.get('relevance', '').lower()

                # Normalize to standard values
                if any(x in relevance_raw for x in ['definitely not', 'définitivement pas']):
                    relevance = 'definitely not'
                elif any(x in relevance_raw for x in ['probably not', 'probablement pas']):
                    relevance = 'probably not'
                elif any(x in relevance_raw for x in ['possibly', 'possiblement']):
                    relevance = 'possibly'
                elif any(x in relevance_raw for x in ['probably', 'probablement']):
                    relevance = 'probably'
                elif any(x in relevance_raw for x in ['definitely', 'définitivement']):
                    relevance = 'definitely'
        except (json.JSONDecodeError, AttributeError) as e:
            logger.debug(f"JSON parse failed for segment, using fallback: {e}")

        # Derive verdict: top 2 = YES, bottom 3 = NO
        is_relevant = relevance in ['definitely', 'probably']

        return {
            'is_relevant': is_relevant,
            'reasoning': reasoning,
            'relevance': relevance
        }

    def run_stage5_final_check(self, batch_size: int = 50) -> Dict:
        """
        Run Stage 5 FINAL CHECK on all segments that passed Stages 3-4.

        Uses tier_1 model to verify each segment relates to gender-based violence.

        Args:
            batch_size: Number of segments to process per batch

        Returns:
            Dict with processing results
        """
        return asyncio.run(self._run_stage5_async(batch_size))

    async def _run_stage5_async(self, batch_size: int = 50) -> Dict:
        """Async implementation of Stage 5 FINAL CHECK."""
        import httpx

        pending_count = self.db_writer.get_pending_stage5_count()
        if pending_count == 0:
            logger.info("No segments pending Stage 5 FINAL CHECK")
            return {'status': 'all_processed', 'processed': 0}

        logger.info(f"Stage 5 FINAL CHECK: {pending_count} segments to verify")
        logger.info(f"Using tier_1 model with academic definition")

        model_server_url = self.config.model_server_url
        processed = 0
        relevant_count = 0
        not_relevant_count = 0

        pbar = tqdm(total=pending_count, desc="Stage 5 FINAL CHECK", unit="seg")

        async with httpx.AsyncClient(timeout=180.0, limits=httpx.Limits(max_connections=10)) as client:
            while True:
                batch = self.db_writer.get_pending_stage5_candidates(batch_size)
                if not batch:
                    break

                # Prepare async tasks
                tasks = []
                for row in batch:
                    text = row.get('segment_text', '')
                    language = row.get('main_language', 'en')
                    tasks.append(self._final_check_async(client, model_server_url, text, language))

                # Execute all tasks concurrently
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results and update database
                updates = []
                for row, result in zip(batch, results):
                    if isinstance(result, Exception):
                        logger.error(f"Stage 5 error for segment {row['segment_id']}: {result}")
                        updates.append({
                            'tc_id': row['tc_id'],
                            'is_relevant': True,  # Default to relevant on error
                            'reasoning': f"Error: {str(result)}",
                            'model': 'tier_1'
                        })
                    else:
                        updates.append({
                            'tc_id': row['tc_id'],
                            'is_relevant': result['is_relevant'],
                            'reasoning': result.get('reasoning', ''),
                            'relevance': result.get('relevance'),
                            'model': 'tier_1'
                        })
                        if result['is_relevant']:
                            relevant_count += 1
                        else:
                            not_relevant_count += 1

                # Bulk update
                self.db_writer.bulk_update_stage5_results(updates)

                processed += len(batch)
                pbar.update(len(batch))

                if processed % 200 == 0:
                    logger.info(f"Progress: {processed}/{pending_count} - Relevant: {relevant_count}, Not relevant: {not_relevant_count}")

        pbar.close()

        logger.info(f"Stage 5 FINAL CHECK complete!")
        logger.info(f"  Total processed: {processed}")
        logger.info(f"  Relevant (kept): {relevant_count}")
        logger.info(f"  Not relevant (filtered): {not_relevant_count}")
        logger.info(f"  Filter rate: {not_relevant_count / processed * 100:.1f}%" if processed > 0 else "  N/A")

        return {
            'status': 'success',
            'processed': processed,
            'relevant': relevant_count,
            'not_relevant': not_relevant_count,
            'filter_rate': not_relevant_count / processed if processed > 0 else 0
        }

    # ============ Stage 6: FALSE POSITIVE DETECTION ============

    async def _false_positive_check_async(self, client, model_server_url: str, text: str, language: str) -> Dict:
        """
        Stage 6 FALSE POSITIVE CHECK: Identify segments that were incorrectly flagged as relevant.

        Detects three types of false positives:
        1. pro_progressive: Content defending feminist/LGBTQ+ positions
        2. documenting_harm: Journalism/research documenting prejudice without promoting it
        3. quote_without_endorsement: Quoting someone's position without endorsing it

        Args:
            client: httpx async client
            model_server_url: URL to model server
            text: Segment text to check
            language: 'en' or 'fr'

        Returns:
            Dict with 'is_false_positive', 'false_positive_type', 'reasoning', 'confidence'
        """
        import re

        # Truncate text if too long
        text = text[:2500] if len(text) > 2500 else text

        if language == 'fr':
            prompt = f"""Ce texte a été signalé comme contenant du contenu lié à la violence ou discrimination basée sur le genre.

TEXTE:
{text}

ÉTAPE 1: Identifiez le contenu problématique détecté (stéréotypes de genre, attitudes anti-LGBTQ+, anti-féminisme, etc.)

ÉTAPE 2: Évaluez - le(s) locuteur(s) dans ce segment ADHÈRENT-ils à ces opinions, ou les REJETTENT-ils / les CRITIQUENT-ils?

ATTENTION - Ces patterns indiquent que le locuteur ADHÈRE aux opinions problématiques:
- Rhétorique de "discrimination inversée" (ex: "on discrimine les hommes blancs", "les vrais victimes sont les hommes")
- Présenter les identités LGBTQ+ comme des "privilèges" dans l'emploi ou l'éducation
- Se plaindre que les hommes/blancs sont "les vrais discriminés"
- Critiquer les initiatives de diversité ou d'équité comme injustes envers les hommes
- Ton sarcastique sur les droits des minorités

Ces patterns NE SONT PAS des critiques progressistes - ce sont des positions masculinistes/anti-féministes.

DISTINCTION IMPORTANTE - Ces patterns indiquent que le locuteur REJETTE/CRITIQUE:
- Observer qu'un double standard existe où les HOMMES sont FAVORISÉS (ex: "les hommes qui ont plusieurs partenaires sont valorisés, mais pas les femmes") = CRITIQUE FÉMINISTE
- Dénoncer que les femmes ou LGBTQ+ sont discriminés = POSITION PROGRESSISTE

La différence clé: Qui est présenté comme la VICTIME?
- Si les HOMMES/BLANCS sont présentés comme victimes → masculinisme → ADHÈRE
- Si les FEMMES/LGBTQ+ sont présentés comme victimes d'un système inégal → féminisme → REJETTE

Échelle de réponse:
- "strongly_holds": Le locuteur exprime et défend clairement ces opinions problématiques
- "holds": Le locuteur semble adhérer à ces opinions
- "leans_holds": Le locuteur semble pencher vers ces opinions, mais avec nuance
- "neutral": Impossible de déterminer la position du locuteur / reportage neutre
- "leans_rejects": Le locuteur semble critiquer ces opinions, mais avec nuance
- "rejects": Le locuteur rejette ou critique ces opinions
- "strongly_rejects": Le locuteur dénonce activement ces opinions depuis une perspective progressiste

Répondez en JSON:
{{"problematic_content": "description brève du contenu détecté", "speaker_stance": "strongly_holds"|"holds"|"leans_holds"|"neutral"|"leans_rejects"|"rejects"|"strongly_rejects", "reasoning": "explication brève"}}"""
        else:
            prompt = f"""This text was flagged as containing content related to gender-based violence or discrimination.

TEXT:
{text}

STEP 1: Identify the problematic content detected (gender stereotypes, anti-LGBTQ+ attitudes, anti-feminism, etc.)

STEP 2: Evaluate - does the speaker(s) in this segment HOLD these views, or REJECT/CRITIQUE them?

WARNING - These patterns indicate the speaker HOLDS problematic views:
- "Reverse discrimination" rhetoric (e.g., "white men are the real victims", "men are discriminated against")
- Framing LGBTQ+ identities as "privileges" in employment or education
- Complaining that men/whites are "the real discriminated group"
- Criticizing diversity or equity initiatives as unfair to men
- Sarcastic tone about minority rights

These patterns are NOT progressive critiques - they are masculinist/anti-feminist positions.

IMPORTANT DISTINCTION - These patterns indicate the speaker REJECTS/CRITIQUES:
- Observing a double standard where MEN are FAVORED (e.g., "men with multiple partners are praised, but women aren't") = FEMINIST CRITIQUE
- Denouncing that women or LGBTQ+ people are discriminated against = PROGRESSIVE POSITION

The key difference: Who is presented as the VICTIM?
- If MEN/WHITES are presented as victims → masculinism → HOLDS
- If WOMEN/LGBTQ+ are presented as victims of an unequal system → feminism → REJECTS

Response scale:
- "strongly_holds": The speaker clearly expresses and defends these problematic views
- "holds": The speaker appears to hold these views
- "leans_holds": The speaker seems to lean toward these views, but with some nuance
- "neutral": Cannot determine speaker's position / neutral reporting
- "leans_rejects": The speaker seems to critique these views, but with nuance
- "rejects": The speaker rejects or critiques these views
- "strongly_rejects": The speaker actively denounces these views from a progressive perspective

Respond in JSON:
{{"problematic_content": "brief description of detected content", "speaker_stance": "strongly_holds"|"holds"|"leans_holds"|"neutral"|"leans_rejects"|"rejects"|"strongly_rejects", "reasoning": "brief explanation"}}"""

        response = await client.post(
            f"{model_server_url}/llm-request",
            json={
                "messages": [
                    {"role": "system", "content": "You are an expert analyst identifying false positives in content classification. Be precise and look for nuance in speaker intent."},
                    {"role": "user", "content": prompt}
                ],
                "model": "tier_1",
                "priority": 1,
                "temperature": 0.1,
                "top_p": 0.9
            }
        )
        response.raise_for_status()
        result_text = response.json().get('response', '').strip()

        # Parse JSON response
        speaker_stance = 'strongly_holds'  # Default to holds view
        problematic_content = None
        reasoning = result_text

        try:
            json_match = re.search(r'\{[^{}]*\}', result_text)
            if json_match:
                parsed = json.loads(json_match.group())
                speaker_stance = parsed.get('speaker_stance', 'strongly_holds').lower().replace(' ', '_')
                problematic_content = parsed.get('problematic_content')
                reasoning = parsed.get('reasoning', result_text)

                # Normalize stance values
                valid_stances = ['strongly_holds', 'holds', 'leans_holds', 'neutral', 'leans_rejects', 'rejects', 'strongly_rejects']
                if speaker_stance not in valid_stances:
                    # Try to map French equivalents
                    stance_map = {
                        'fortement_adhère': 'strongly_holds',
                        'adhère': 'holds',
                        'penche_adhère': 'leans_holds',
                        'neutre': 'neutral',
                        'penche_rejette': 'leans_rejects',
                        'rejette': 'rejects',
                        'fortement_rejette': 'strongly_rejects'
                    }
                    speaker_stance = stance_map.get(speaker_stance, 'strongly_holds')

        except (json.JSONDecodeError, AttributeError) as e:
            logger.debug(f"JSON parse failed for Stage 6, using fallback: {e}")

        # Derive is_false_positive from stance
        # False positive = speaker doesn't clearly hold the problematic views
        # Only 'strongly_holds' and 'holds' are true positives
        is_false_positive = speaker_stance in ['leans_holds', 'neutral', 'leans_rejects', 'rejects', 'strongly_rejects']

        return {
            'is_false_positive': is_false_positive,
            'problematic_content': problematic_content,
            'reasoning': reasoning,
            'speaker_stance': speaker_stance
        }

    def run_stage6_false_positive_check(self, batch_size: int = 50) -> Dict:
        """
        Run Stage 6 FALSE POSITIVE CHECK on segments that passed Stage 5.

        Uses tier_1 model to identify false positives that are:
        - Pro-progressive content (defending feminist/LGBTQ+ positions)
        - Documentation of harm (journalism/research)
        - Quotes without endorsement

        Args:
            batch_size: Number of segments to process per batch

        Returns:
            Dict with processing results
        """
        return asyncio.run(self._run_stage6_async(batch_size))

    async def _run_stage6_async(self, batch_size: int = 50) -> Dict:
        """Async implementation of Stage 6 FALSE POSITIVE CHECK."""
        import httpx

        pending_count = self.db_writer.get_pending_stage6_count()
        if pending_count == 0:
            logger.info("No segments pending Stage 6 FALSE POSITIVE CHECK")
            return {'status': 'all_processed', 'processed': 0}

        logger.info(f"Stage 6 FALSE POSITIVE CHECK: {pending_count} segments to verify")
        logger.info(f"Using tier_1 model to detect false positives")

        model_server_url = self.config.model_server_url
        processed = 0
        false_positive_count = 0
        true_positive_count = 0

        # Track by type
        type_counts = {
            'pro_progressive': 0,
            'documenting_harm': 0,
            'quote_without_endorsement': 0
        }

        pbar = tqdm(total=pending_count, desc="Stage 6 FALSE POSITIVE CHECK", unit="seg")

        async with httpx.AsyncClient(timeout=180.0, limits=httpx.Limits(max_connections=10)) as client:
            while True:
                batch = self.db_writer.get_pending_stage6_candidates(batch_size)
                if not batch:
                    break

                # Prepare async tasks
                tasks = []
                for row in batch:
                    text = row.get('segment_text', '')
                    language = row.get('main_language', 'en')
                    tasks.append(self._false_positive_check_async(client, model_server_url, text, language))

                # Execute all tasks concurrently
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results and update database
                updates = []
                for row, result in zip(batch, results):
                    if isinstance(result, Exception):
                        logger.error(f"Stage 6 error for segment {row['segment_id']}: {result}")
                        updates.append({
                            'tc_id': row['tc_id'],
                            'is_false_positive': False,
                            'false_positive_type': None,
                            'reasoning': f"Error: {str(result)}",
                            'confidence': None,
                            'model': 'tier_1'
                        })
                    else:
                        updates.append({
                            'tc_id': row['tc_id'],
                            'is_false_positive': result['is_false_positive'],
                            'problematic_content': result.get('problematic_content'),
                            'reasoning': result.get('reasoning', ''),
                            'speaker_stance': result.get('speaker_stance'),
                            'model': 'tier_1'
                        })
                        if result['is_false_positive']:
                            false_positive_count += 1
                            fp_type = result.get('false_positive_type')
                            if fp_type in type_counts:
                                type_counts[fp_type] += 1
                        else:
                            true_positive_count += 1

                # Bulk update
                self.db_writer.bulk_update_stage6_results(updates)

                processed += len(batch)
                pbar.update(len(batch))

                if processed % 200 == 0:
                    logger.info(f"Progress: {processed}/{pending_count} - FP: {false_positive_count}, TP: {true_positive_count}")

        pbar.close()

        logger.info(f"Stage 6 FALSE POSITIVE CHECK complete!")
        logger.info(f"  Total processed: {processed}")
        logger.info(f"  True positives (kept): {true_positive_count}")
        logger.info(f"  False positives (filtered): {false_positive_count}")
        logger.info(f"  False positive rate: {false_positive_count / processed * 100:.1f}%" if processed > 0 else "  N/A")
        logger.info(f"  By type:")
        logger.info(f"    - Pro-progressive: {type_counts['pro_progressive']}")
        logger.info(f"    - Documenting harm: {type_counts['documenting_harm']}")
        logger.info(f"    - Quote without endorsement: {type_counts['quote_without_endorsement']}")

        return {
            'status': 'success',
            'processed': processed,
            'true_positives': true_positive_count,
            'false_positives': false_positive_count,
            'false_positive_rate': false_positive_count / processed if processed > 0 else 0,
            'by_type': type_counts
        }

    # ======================== Stage 7: EXPANDED CONTEXT RE-CHECK ========================

    async def _expanded_context_check_async(
        self,
        client,
        model_server_url: str,
        text: str,
        language: str,
        original_stance: str
    ) -> Dict:
        """
        Stage 7: Re-evaluate speaker stance with expanded context window.

        Uses the same prompt as Stage 6 but with more surrounding context
        to better assess speaker stance.

        Args:
            client: httpx async client
            model_server_url: URL to model server
            text: Expanded segment text (±20 seconds context)
            language: 'en' or 'fr'
            original_stance: The stance from Stage 6

        Returns:
            Dict with 'is_false_positive', 'speaker_stance', 'reasoning'
        """
        import re

        # Truncate text if too long (expanded context may be longer)
        text = text[:4000] if len(text) > 4000 else text

        if language == 'fr':
            prompt = f"""Ce texte (avec contexte élargi de ±20 secondes) a été signalé comme contenant du contenu lié à la violence ou discrimination basée sur le genre.

Une première analyse a classé ce segment comme "{original_stance}" (le locuteur semble rejeter ou critiquer les opinions problématiques).

Avec ce contexte élargi, réévaluez la position du/des locuteur(s).

TEXTE AVEC CONTEXTE ÉLARGI:
{text}

ÉTAPE 1: Identifiez le contenu problématique détecté (stéréotypes de genre, attitudes anti-LGBTQ+, anti-féminisme, etc.)

ÉTAPE 2: Avec ce contexte élargi, évaluez - le(s) locuteur(s) ADHÈRENT-ils à ces opinions, ou les REJETTENT-ils / les CRITIQUENT-ils?

ATTENTION - Ces patterns indiquent que le locuteur ADHÈRE aux opinions problématiques:
- Rhétorique de "discrimination inversée" (ex: "on discrimine les hommes blancs", "les vrais victimes sont les hommes")
- Présenter les identités LGBTQ+ comme des "privilèges" dans l'emploi ou l'éducation
- Se plaindre que les hommes/blancs sont "les vrais discriminés"
- Critiquer les initiatives de diversité ou d'équité comme injustes envers les hommes
- Ton sarcastique ou moqueur en discutant des concepts LGBTQ+, du féminisme ou de la diversité
- Dog whistles: langage codé comme "wokisme", "idéologie du genre", "femmes biologiques/vraies femmes", "XX/XY"
- Présenter les groupes anti-féministes ou masculinistes comme victimes de "cancel culture" ou de persécution idéologique

Ces patterns NE SONT PAS des critiques progressistes - ce sont des positions masculinistes/anti-féministes.

DISTINCTION IMPORTANTE - Ces patterns indiquent que le locuteur REJETTE/CRITIQUE:
- Observer qu'un double standard existe où les HOMMES sont FAVORISÉS (ex: "les hommes qui ont plusieurs partenaires sont valorisés, mais pas les femmes") = CRITIQUE FÉMINISTE
- Dénoncer que les femmes ou LGBTQ+ sont discriminés = POSITION PROGRESSISTE

La différence clé: Qui est présenté comme la VICTIME?
- Si les HOMMES/BLANCS sont présentés comme victimes → masculinisme → ADHÈRE
- Si les FEMMES/LGBTQ+ sont présentés comme victimes d'un système inégal → féminisme → REJETTE

Échelle de réponse:
- "strongly_holds": Le locuteur exprime et défend clairement ces opinions problématiques
- "holds": Le locuteur semble adhérer à ces opinions
- "leans_holds": Le locuteur semble pencher vers ces opinions, mais avec nuance
- "neutral": Impossible de déterminer la position du locuteur / reportage neutre
- "leans_rejects": Le locuteur semble critiquer ces opinions, mais avec nuance
- "rejects": Le locuteur rejette ou critique ces opinions
- "strongly_rejects": Le locuteur dénonce activement ces opinions depuis une perspective progressiste

Répondez en JSON:
{{"speaker_stance": "strongly_holds"|"holds"|"leans_holds"|"neutral"|"leans_rejects"|"rejects"|"strongly_rejects", "reasoning": "explication brève incluant ce que le contexte élargi révèle"}}"""
        else:
            prompt = f"""This text (with expanded ±20 second context) was flagged as containing content related to gender-based violence or discrimination.

An initial analysis classified this segment as "{original_stance}" (the speaker appears to reject or critique problematic views).

With this expanded context, re-evaluate the speaker(s) position.

TEXT WITH EXPANDED CONTEXT:
{text}

STEP 1: Identify the problematic content detected (gender stereotypes, anti-LGBTQ+ attitudes, anti-feminism, etc.)

STEP 2: With this expanded context, evaluate - does the speaker(s) HOLD these views, or REJECT/CRITIQUE them?

WARNING - These patterns indicate the speaker HOLDS problematic views:
- "Reverse discrimination" rhetoric (e.g., "white men are the real victims", "men are discriminated against")
- Framing LGBTQ+ identities as "privileges" in employment or education
- Complaining that men/whites are "the real discriminated group"
- Criticizing diversity or equity initiatives as unfair to men
- Sarcastic or mocking tone when discussing LGBTQ+ concepts, feminism, or diversity
- Dog whistles: coded language like "wokism", "gender ideology", "biological women/real women", "XX/XY"
- Framing anti-feminist or masculinist groups as victims of "cancel culture" or ideological persecution

These patterns are NOT progressive critiques - they are masculinist/anti-feminist positions.

IMPORTANT DISTINCTION - These patterns indicate the speaker REJECTS/CRITIQUES:
- Observing a double standard where MEN are FAVORED (e.g., "men with multiple partners are praised, but women aren't") = FEMINIST CRITIQUE
- Denouncing that women or LGBTQ+ people are discriminated against = PROGRESSIVE POSITION

The key difference: Who is presented as the VICTIM?
- If MEN/WHITES are presented as victims → masculinism → HOLDS
- If WOMEN/LGBTQ+ are presented as victims of an unequal system → feminism → REJECTS

Response scale:
- "strongly_holds": The speaker clearly expresses and defends these problematic views
- "holds": The speaker appears to hold these views
- "leans_holds": The speaker seems to lean toward these views, but with some nuance
- "neutral": Cannot determine speaker's position / neutral reporting
- "leans_rejects": The speaker seems to critique these views, but with nuance
- "rejects": The speaker rejects or critiques these views
- "strongly_rejects": The speaker actively denounces these views from a progressive perspective

Respond in JSON:
{{"speaker_stance": "strongly_holds"|"holds"|"leans_holds"|"neutral"|"leans_rejects"|"rejects"|"strongly_rejects", "reasoning": "brief explanation including what the expanded context reveals"}}"""

        response = await client.post(
            f"{model_server_url}/llm-request",
            json={
                "messages": [
                    {"role": "system", "content": "You are an expert analyst re-evaluating speaker stance with expanded context. The original classification may have been wrong due to limited context. Be precise about whether speakers hold or reject problematic views."},
                    {"role": "user", "content": prompt}
                ],
                "model": "tier_1",
                "priority": 1,
                "temperature": 0.1,
                "top_p": 0.9
            }
        )
        response.raise_for_status()
        result_text = response.json().get('response', '').strip()

        # Parse JSON response
        speaker_stance = original_stance  # Default to original if parse fails
        reasoning = result_text

        try:
            json_match = re.search(r'\{[^{}]*\}', result_text)
            if json_match:
                parsed = json.loads(json_match.group())
                speaker_stance = parsed.get('speaker_stance', original_stance).lower().replace(' ', '_')
                reasoning = parsed.get('reasoning', result_text)

                # Normalize stance values
                valid_stances = ['strongly_holds', 'holds', 'leans_holds', 'neutral', 'leans_rejects', 'rejects', 'strongly_rejects']
                if speaker_stance not in valid_stances:
                    speaker_stance = original_stance

        except (json.JSONDecodeError, AttributeError) as e:
            logger.debug(f"JSON parse failed for Stage 7, using original stance: {e}")

        # Derive is_false_positive from stance
        # Only 'strongly_holds' and 'holds' are true positives
        is_false_positive = speaker_stance in ['leans_holds', 'neutral', 'leans_rejects', 'rejects', 'strongly_rejects']

        return {
            'is_false_positive': is_false_positive,
            'reasoning': reasoning,
            'speaker_stance': speaker_stance
        }

    def run_stage7_expanded_context(self, batch_size: int = 50, context_window_seconds: int = 20) -> Dict:
        """
        Run Stage 7 EXPANDED CONTEXT RE-CHECK on Stage 6 false positives.

        Re-evaluates segments with ±20 second context window to catch cases
        where the original segment lacked sufficient context.

        Args:
            batch_size: Number of segments to process per batch
            context_window_seconds: Seconds of context to include on each side

        Returns:
            Dict with processing results
        """
        return asyncio.run(self._run_stage7_async(batch_size, context_window_seconds))

    async def _run_stage7_async(self, batch_size: int = 50, context_window_seconds: int = 20) -> Dict:
        """Async implementation of Stage 7 EXPANDED CONTEXT RE-CHECK."""
        import httpx

        pending_count = self.db_writer.get_pending_stage7_count()
        if pending_count == 0:
            logger.info("No segments pending Stage 7 EXPANDED CONTEXT RE-CHECK")
            return {'status': 'all_processed', 'processed': 0}

        logger.info(f"Stage 7 EXPANDED CONTEXT RE-CHECK: {pending_count} segments to verify")
        logger.info(f"Using tier_1 model with ±{context_window_seconds}s context window")

        model_server_url = self.config.model_server_url
        processed = 0
        upgraded_count = 0  # Changed from false positive to true positive
        confirmed_fp_count = 0  # Remained false positive

        # Track stance changes
        stance_changes = {}

        pbar = tqdm(total=pending_count, desc="Stage 7 EXPANDED CONTEXT", unit="seg")

        async with httpx.AsyncClient(timeout=180.0, limits=httpx.Limits(max_connections=10)) as client:
            while True:
                batch = self.db_writer.get_pending_stage7_candidates(batch_size, context_window_seconds)
                if not batch:
                    break

                # Prepare async tasks
                tasks = []
                for row in batch:
                    text = row.get('expanded_text', row.get('segment_text', ''))
                    language = row.get('main_language', 'en')
                    original_stance = row.get('original_stance', 'neutral')
                    tasks.append(self._expanded_context_check_async(client, model_server_url, text, language, original_stance))

                # Execute all tasks concurrently
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results and update database
                updates = []
                for row, result in zip(batch, results):
                    original_stance = row.get('original_stance', 'neutral')

                    if isinstance(result, Exception):
                        logger.error(f"Stage 7 error for segment {row['segment_id']}: {result}")
                        updates.append({
                            'tc_id': row['tc_id'],
                            'is_false_positive': True,  # Keep as false positive on error
                            'speaker_stance': original_stance,
                            'reasoning': f"Error: {str(result)}",
                            'original_stance': original_stance,
                            'context_window_seconds': context_window_seconds,
                            'model': 'tier_1'
                        })
                        confirmed_fp_count += 1
                    else:
                        new_stance = result.get('speaker_stance', original_stance)
                        is_fp = result['is_false_positive']

                        updates.append({
                            'tc_id': row['tc_id'],
                            'is_false_positive': is_fp,
                            'speaker_stance': new_stance,
                            'reasoning': result.get('reasoning', ''),
                            'original_stance': original_stance,
                            'context_window_seconds': context_window_seconds,
                            'model': 'tier_1'
                        })

                        # Track changes
                        change_key = f"{original_stance} -> {new_stance}"
                        stance_changes[change_key] = stance_changes.get(change_key, 0) + 1

                        if not is_fp:
                            upgraded_count += 1
                        else:
                            confirmed_fp_count += 1

                # Bulk update
                self.db_writer.bulk_update_stage7_results(updates)

                processed += len(batch)
                pbar.update(len(batch))

                if processed % 200 == 0:
                    logger.info(f"Progress: {processed}/{pending_count} - Upgraded: {upgraded_count}, Confirmed FP: {confirmed_fp_count}")

        pbar.close()

        logger.info(f"Stage 7 EXPANDED CONTEXT RE-CHECK complete!")
        logger.info(f"  Total processed: {processed}")
        logger.info(f"  Upgraded to true positive: {upgraded_count} ({upgraded_count / processed * 100:.1f}%)" if processed > 0 else "  N/A")
        logger.info(f"  Confirmed false positive: {confirmed_fp_count} ({confirmed_fp_count / processed * 100:.1f}%)" if processed > 0 else "  N/A")
        logger.info(f"  Stance changes:")
        for change, count in sorted(stance_changes.items(), key=lambda x: -x[1]):
            logger.info(f"    {change}: {count}")

        return {
            'status': 'success',
            'processed': processed,
            'upgraded_to_true_positive': upgraded_count,
            'confirmed_false_positive': confirmed_fp_count,
            'upgrade_rate': upgraded_count / processed if processed > 0 else 0,
            'stance_changes': stance_changes
        }

    def run(self, stage1_only: bool = False, resume: bool = False, stage5_only: bool = False, include_stage5: bool = False, stage6_only: bool = False, include_stage6: bool = False, stage7_only: bool = False, include_stage7: bool = False) -> Dict:
        """Run the complete pipeline with database-backed progress

        Args:
            stage1_only: Only run Stage 1 and report results
            resume: Skip Stage 1 and process pending candidates from database
            stage5_only: Only run Stage 5 FINAL CHECK on segments that passed Stages 3-4
            include_stage5: Run Stage 5 after Stages 3-4 (full pipeline)
            stage6_only: Only run Stage 6 FALSE POSITIVE CHECK on segments that passed Stage 5
            include_stage6: Run Stage 6 after Stage 5 (full pipeline)
            stage7_only: Only run Stage 7 EXPANDED CONTEXT on Stage 6 false positives
            include_stage7: Run Stage 7 after Stage 6 (full pipeline)
        """
        logger.info("Starting semantic theme classification pipeline")
        if stage1_only:
            logger.info("STAGE 1 ONLY MODE: Will skip LLM stages and report matches")
        if resume:
            logger.info("RESUME MODE: Will process pending candidates from database")
        if stage5_only:
            logger.info("STAGE 5 ONLY MODE: Will run FINAL CHECK on processed segments")
        if include_stage5:
            logger.info("FULL PIPELINE MODE: Will include Stage 5 FINAL CHECK")
        if stage6_only:
            logger.info("STAGE 6 ONLY MODE: Will run FALSE POSITIVE CHECK on Stage 5 relevant segments")
        if include_stage6:
            logger.info("FULL PIPELINE MODE: Will include Stage 6 FALSE POSITIVE CHECK")
        if stage7_only:
            logger.info("STAGE 7 ONLY MODE: Will run EXPANDED CONTEXT RE-CHECK on Stage 6 false positives")
        if include_stage7:
            logger.info("FULL PIPELINE MODE: Will include Stage 7 EXPANDED CONTEXT RE-CHECK")

        try:
            # Load or create schema (without embeddings first)
            logger.info(f"Loading schema: {self.schema_name} v{self.schema_version}")
            schema_id = self.db_writer.load_or_create_schema(
                csv_path=self.config.subthemes_csv,
                description=f"Semantic classification run {datetime.now().isoformat()}"
            )
            logger.info(f"Schema loaded (id={schema_id})")

            # Stage 7 only mode
            if stage7_only:
                results = self.run_stage7_expanded_context()
                results['schema_id'] = schema_id
                results['schema_name'] = self.schema_name
                results['schema_version'] = self.schema_version
                return results

            # Stage 6 only mode
            if stage6_only:
                results = self.run_stage6_false_positive_check()
                results['schema_id'] = schema_id
                results['schema_name'] = self.schema_name
                results['schema_version'] = self.schema_version

                # Run Stage 7 if requested
                if include_stage7 and results.get('status') == 'success':
                    logger.info("Running Stage 7 EXPANDED CONTEXT RE-CHECK...")
                    stage7_results = self.run_stage7_expanded_context()
                    results['stage7'] = stage7_results

                return results

            # Stage 5 only mode
            if stage5_only:
                results = self.run_stage5_final_check()
                results['schema_id'] = schema_id
                results['schema_name'] = self.schema_name
                results['schema_version'] = self.schema_version

                # Run Stage 6 if requested
                if include_stage6 and results.get('status') == 'success':
                    logger.info("Running Stage 6 FALSE POSITIVE CHECK...")
                    stage6_results = self.run_stage6_false_positive_check()
                    results['stage6'] = stage6_results

                    # Run Stage 7 if requested
                    if include_stage7 and stage6_results.get('status') == 'success':
                        logger.info("Running Stage 7 EXPANDED CONTEXT RE-CHECK...")
                        stage7_results = self.run_stage7_expanded_context()
                        results['stage7'] = stage7_results

                return results

            # Check for pending candidates in database
            pending_count = self.db_writer.get_pending_count()

            if resume and pending_count > 0:
                logger.info(f"Found {pending_count} pending candidates in database, resuming...")
                results = self._process_from_db()

                # Run Stage 5 if requested
                if include_stage5 and results.get('status') == 'success':
                    logger.info("Running Stage 5 FINAL CHECK...")
                    stage5_results = self.run_stage5_final_check()
                    results['stage5'] = stage5_results

                    # Run Stage 6 if requested
                    if include_stage6 and stage5_results.get('status') == 'success':
                        logger.info("Running Stage 6 FALSE POSITIVE CHECK...")
                        stage6_results = self.run_stage6_false_positive_check()
                        results['stage6'] = stage6_results

                        # Run Stage 7 if requested
                        if include_stage7 and stage6_results.get('status') == 'success':
                            logger.info("Running Stage 7 EXPANDED CONTEXT RE-CHECK...")
                            stage7_results = self.run_stage7_expanded_context()
                            results['stage7'] = stage7_results

                return results

            if resume and pending_count == 0:
                logger.info("No pending candidates in database")
                # Try loading from JSON cache as fallback
                candidates = self._load_stage1_candidates()
                if candidates is None:
                    logger.info("No JSON cache found either. Running Stage 1...")
                elif len(candidates) == 0:
                    logger.info("All candidates already processed!")
                    return {'status': 'all_processed', 'schema_id': schema_id}
                else:
                    # Bulk insert from JSON cache into database
                    logger.info(f"Found {len(candidates)} candidates in JSON cache, inserting into database...")
                    inserted = self.db_writer.bulk_insert_stage1_candidates(candidates)
                    logger.info(f"Inserted {inserted} new candidates")
                    results = self._process_from_db()

                    # Run Stage 5 if requested
                    if include_stage5 and results.get('status') == 'success':
                        logger.info("Running Stage 5 FINAL CHECK...")
                        stage5_results = self.run_stage5_final_check()
                        results['stage5'] = stage5_results

                        # Run Stage 6 if requested
                        if include_stage6 and stage5_results.get('status') == 'success':
                            logger.info("Running Stage 6 FALSE POSITIVE CHECK...")
                            stage6_results = self.run_stage6_false_positive_check()
                            results['stage6'] = stage6_results

                    return results

            # Run Stage 1: FAISS semantic search
            self._query_embeddings_to_cache = None
            candidates = self.stage1_semantic_search()
            if not candidates:
                logger.warning("No candidates found")
                return {'status': 'no_candidates'}

            logger.info(f"Stage 1 complete: {len(candidates)} candidates")

            # Bulk insert Stage 1 results into database
            logger.info("Inserting Stage 1 candidates into database...")
            inserted = self.db_writer.bulk_insert_stage1_candidates(candidates)
            logger.info(f"Inserted {inserted} new candidates (skipped {len(candidates) - inserted} duplicates)")

            # Also save to JSON cache for backward compatibility
            self._save_stage1_candidates(candidates)

            # Cache query embeddings if they were computed
            if self._query_embeddings_to_cache:
                logger.info("Caching query embeddings to schema...")
                self.db_writer.load_or_create_schema(
                    csv_path=self.config.subthemes_csv,
                    query_embeddings=self._query_embeddings_to_cache
                )
                logger.info("Query embeddings cached successfully")

            # If stage1_only, generate report and exit
            if stage1_only:
                report = self.report_stage1_results(candidates)

                # Print report
                print("\n" + "="*70)
                print("STAGE 1 SEMANTIC SEARCH RESULTS")
                print("="*70)
                print(f"Total segments matched: {report['total_segments']}")
                print(f"Average similarity score: {report['avg_similarity_score']:.4f}")
                print(f"Similarity threshold: {self.config.similarity_threshold}")

                print("\n" + "-"*70)
                print("THEMES")
                print("-"*70)
                for theme_id, theme_data in report['themes'].items():
                    print(f"  Theme {theme_id}: {theme_data['name']}")
                    print(f"    Segments: {theme_data['count']} ({theme_data['percentage']:.1f}%)")

                print("\n" + "-"*70)
                print("SUBTHEMES")
                print("-"*70)
                for subtheme_id, subtheme_data in report['subthemes'].items():
                    print(f"  {subtheme_id}: {subtheme_data['name']}")
                    print(f"    Theme: {subtheme_data['theme_name']}")
                    print(f"    Segments: {subtheme_data['count']} ({subtheme_data['percentage']:.1f}%)")
                    print()

                print("="*70)

                return {
                    'status': 'success_stage1_only',
                    'report': report,
                    'schema_id': schema_id,
                    'schema_name': self.schema_name,
                    'schema_version': self.schema_version
                }

            # Process from database (new flow)
            results = self._process_from_db()

            # Run Stage 5 if requested
            if include_stage5 and results.get('status') == 'success':
                logger.info("Running Stage 5 FINAL CHECK...")
                stage5_results = self.run_stage5_final_check()
                results['stage5'] = stage5_results

                # Run Stage 6 if requested
                if include_stage6 and stage5_results.get('status') == 'success':
                    logger.info("Running Stage 6 FALSE POSITIVE CHECK...")
                    stage6_results = self.run_stage6_false_positive_check()
                    results['stage6'] = stage6_results

            return results

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            return {'status': 'error', 'message': str(e)}


def main():
    parser = argparse.ArgumentParser(description='Semantic-Only Theme Classification with Database Storage')
    parser.add_argument('--subthemes-csv', required=True, help='Path to subthemes CSV file')
    parser.add_argument('--schema-name', required=True, help='Schema name (e.g., CPRMV)')
    parser.add_argument('--schema-version', required=True, help='Schema version (e.g., v1.0, 2025-01-15)')
    parser.add_argument('--similarity-threshold', type=float, default=0.38,
                       help='Minimum similarity score threshold (default: 0.38)')
    parser.add_argument('--model', default='tier_1',
                       help='Model tier: tier_1 (80B), tier_2 (4B), tier_3 (8B)')
    parser.add_argument('--embedding-model', default='Qwen/Qwen3-Embedding-4B',
                       help='Embedding model for semantic search')
    parser.add_argument('--project', required=True, help='Project name (e.g., CPRMV)')
    parser.add_argument('--test-mode', type=int, help='Test mode: limit to N segments')
    parser.add_argument('--use-faiss', action='store_true', default=True,
                       help='Use FAISS for semantic search')
    parser.add_argument('--faiss-index-path', help='Path to FAISS index')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--stage1-only', action='store_true',
                       help='Run Stage 1 only (semantic search) and report matches, skip LLM stages')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from Stage 1 cache (skip FAISS loading and semantic search)')
    parser.add_argument('--stage5-only', action='store_true',
                       help='Run Stage 5 FINAL CHECK only (verify segments relate to gender-based violence)')
    parser.add_argument('--include-stage5', action='store_true',
                       help='Include Stage 5 FINAL CHECK after Stages 3-4 (full pipeline)')
    parser.add_argument('--stage6-only', action='store_true',
                       help='Run Stage 6 FALSE POSITIVE CHECK only (detect pro-progressive, documenting harm, quotes without endorsement)')
    parser.add_argument('--include-stage6', action='store_true',
                       help='Include Stage 6 FALSE POSITIVE CHECK after Stage 5 (full pipeline)')
    parser.add_argument('--stage7-only', action='store_true',
                       help='Run Stage 7 EXPANDED CONTEXT only (re-check Stage 6 false positives with ±20s context)')
    parser.add_argument('--include-stage7', action='store_true',
                       help='Include Stage 7 EXPANDED CONTEXT after Stage 6 (full pipeline)')

    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)

    # Load model_server configuration (direct to MLX, port 8004)
    model_server_url = None
    use_model_server = True
    try:
        from src.utils.config import load_config
        app_config = load_config()
        # Use model_server directly (port 8004) for best performance
        model_server_config = app_config.get('processing', {}).get('model_server', {})
        if model_server_config.get('enabled', False):
            host = model_server_config.get('host', '10.0.0.34')
            port = model_server_config.get('port', 8004)
            model_server_url = f"http://{host}:{port}"
            logger.info(f"Model server configured: {model_server_url} (direct to MLX)")
    except Exception as e:
        logger.warning(f"Could not load model_server config: {e}")

    # Fallback to hardcoded if config failed
    if not model_server_url:
        model_server_url = "http://10.0.0.34:8004"
        logger.info(f"Using default model server: {model_server_url}")

    # Create configuration (output_csv no longer needed, but keep for compatibility)
    config = SemanticConfig(
        subthemes_csv=args.subthemes_csv,
        output_csv='',  # Not used anymore
        similarity_threshold=args.similarity_threshold,
        model_name=args.model,
        embedding_model=args.embedding_model,
        project=args.project,
        model_server_url=model_server_url,
        use_model_server=use_model_server,
        test_mode=args.test_mode,
        use_faiss=args.use_faiss,
        faiss_index_path=args.faiss_index_path
    )

    # Run pipeline
    pipeline = SemanticThemeClassifier(
        config=config,
        schema_name=args.schema_name,
        schema_version=args.schema_version
    )
    results = pipeline.run(
        stage1_only=args.stage1_only,
        resume=args.resume,
        stage5_only=args.stage5_only,
        include_stage5=args.include_stage5,
        stage6_only=args.stage6_only,
        include_stage6=args.include_stage6,
        stage7_only=args.stage7_only,
        include_stage7=args.include_stage7
    )

    # Print summary
    if results['status'] == 'success' and args.stage7_only:
        print(f"\n✅ Stage 7 EXPANDED CONTEXT RE-CHECK complete!")
        print(f"📊 Processed: {results['processed']}")
        print(f"⬆️ Upgraded to true positive: {results['upgraded_to_true_positive']}")
        print(f"✗ Confirmed false positive: {results['confirmed_false_positive']}")
        print(f"📈 Upgrade rate: {results['upgrade_rate']*100:.1f}%")
        if 'stance_changes' in results:
            print(f"\n📋 Stance changes:")
            for change, count in sorted(results['stance_changes'].items(), key=lambda x: -x[1]):
                print(f"   {change}: {count}")
        print(f"📋 Schema: {results['schema_name']} v{results['schema_version']} (id={results['schema_id']})")
    elif results['status'] == 'success' and args.stage6_only:
        print(f"\n✅ Stage 6 FALSE POSITIVE CHECK complete!")
        print(f"📊 Processed: {results['processed']}")
        print(f"✓ True positives (kept): {results['true_positives']}")
        print(f"✗ False positives (filtered): {results['false_positives']}")
        print(f"📉 False positive rate: {results['false_positive_rate']*100:.1f}%")
        if 'by_type' in results:
            print(f"\n📋 By type:")
            print(f"   - Pro-progressive: {results['by_type'].get('pro_progressive', 0)}")
            print(f"   - Documenting harm: {results['by_type'].get('documenting_harm', 0)}")
            print(f"   - Quote without endorsement: {results['by_type'].get('quote_without_endorsement', 0)}")
        print(f"📋 Schema: {results['schema_name']} v{results['schema_version']} (id={results['schema_id']})")
    elif results['status'] == 'success' and args.stage5_only:
        print(f"\n✅ Stage 5 FINAL CHECK complete!")
        print(f"📊 Processed: {results['processed']}")
        print(f"✓ Relevant (kept): {results['relevant']}")
        print(f"✗ Not relevant (filtered): {results['not_relevant']}")
        print(f"📉 Filter rate: {results['filter_rate']*100:.1f}%")
        print(f"📋 Schema: {results['schema_name']} v{results['schema_version']} (id={results['schema_id']})")
        if 'stage6' in results:
            s6 = results['stage6']
            print(f"\n--- Stage 6 FALSE POSITIVE CHECK ---")
            print(f"✓ True positives: {s6.get('true_positives', 0)}")
            print(f"✗ False positives: {s6.get('false_positives', 0)}")
            print(f"📉 FP rate: {s6.get('false_positive_rate', 0)*100:.1f}%")
    elif results['status'] == 'success':
        print(f"\n✅ Pipeline completed!")
        print(f"📊 Candidates: {results['total_candidates']}")
        print(f"🎯 Themes identified: {results['themes_identified']}")
        print(f"📋 Schema: {results['schema_name']} v{results['schema_version']} (id={results['schema_id']})")
        print(f"💾 Results stored in theme_classifications table")
        if 'stage5' in results:
            s5 = results['stage5']
            print(f"\n--- Stage 5 FINAL CHECK ---")
            print(f"✓ Relevant: {s5.get('relevant', 0)}")
            print(f"✗ Not relevant: {s5.get('not_relevant', 0)}")
            print(f"📉 Filter rate: {s5.get('filter_rate', 0)*100:.1f}%")
        if 'stage6' in results:
            s6 = results['stage6']
            print(f"\n--- Stage 6 FALSE POSITIVE CHECK ---")
            print(f"✓ True positives: {s6.get('true_positives', 0)}")
            print(f"✗ False positives: {s6.get('false_positives', 0)}")
            print(f"📉 FP rate: {s6.get('false_positive_rate', 0)*100:.1f}%")
            if 'by_type' in s6:
                print(f"   - Pro-progressive: {s6['by_type'].get('pro_progressive', 0)}")
                print(f"   - Documenting harm: {s6['by_type'].get('documenting_harm', 0)}")
                print(f"   - Quote without endorsement: {s6['by_type'].get('quote_without_endorsement', 0)}")
    elif results['status'] == 'success_stage1_only':
        print(f"\n✅ Stage 1 complete!")
        print(f"📋 Schema: {results['schema_name']} v{results['schema_version']} (id={results['schema_id']})")
        print(f"📊 See detailed report above")
    elif results['status'] == 'all_processed':
        print(f"\n✅ All segments already processed!")
        if 'processed' in results:
            print(f"📊 No pending segments")
    else:
        print(f"\n❌ Pipeline failed: {results.get('message', results['status'])}")


if __name__ == "__main__":
    main()
