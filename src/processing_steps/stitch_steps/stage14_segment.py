#!/usr/bin/env python3
"""
Stage 14: Semantic Segmentation for Retrieval
=============================================

Creates retrieval-optimized embedding segments from sentences (after emotion detection).

Pipeline Order:
- Stage 12: Generate speaker turns + sentences → save to database
- Stage 13: Emotion detection on sentences → update sentences with emotion data
- Stage 14: Semantic segmentation → create segments with source_sentence_ids and emotion_summary

Key Responsibilities:
- Segment sentences into retrieval-optimized chunks (50-400 tokens)
- Use beam search with semantic similarity for intelligent boundary detection
- Aggregate emotion data from source sentences into emotion_summary
- Generate EmbeddingSegment records with source_sentence_ids for linking

Input:
- Sentences from Stage 12 (with emotion data from Stage 13)
- WordTable for speaker_db_dictionary lookup
- Content ID for database insertion

Output:
- List of segment dictionaries ready for EmbeddingSegment insertion
- Each segment has: text, timing, token count, speaker_positions, source_sentence_ids, emotion_summary

Configuration (from config.yaml):
- min_tokens: Minimum segment size (default: 50)
- target_tokens: Target segment size (default: 250)
- max_tokens: Maximum segment size (default: 400)
- coherence_threshold: Similarity threshold for segmentation (default: 0.7)
- beam_width: Beam search width (default: 5)
- lookahead_sentences: Sentences to check ahead (default: 3)
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
import time
import yaml
import hashlib
from pathlib import Path
from src.utils.paths import get_project_root, get_config_path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize

from src.utils.logger import setup_worker_logger
from src.utils.config import load_config
from .stage3_tables import WordTable
from .util_stitch import smart_join_words

logger = setup_worker_logger('stitch')

# Try to import NLTK and download punkt if needed
try:
    import nltk
    nltk.data.find('tokenizers/punkt')
except LookupError:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)


class TranscriptSegmenter:
    """Segments word-level transcripts into retrieval-optimized chunks using beam search."""

    def __init__(self, config: Dict = None):
        """Initialize segmenter with config."""
        # Load config if not provided
        if config is None:
            self.config = load_config()
        else:
            self.config = config

        # Segmentation parameters
        seg_config = self.config.get('embedding', {}).get('embedding_segmentation', {})
        self.min_tokens = seg_config.get('min_tokens', 50)
        self.target_tokens = seg_config.get('target_tokens', 250)
        self.max_tokens = seg_config.get('max_tokens', 400)
        self.coherence_threshold = seg_config.get('coherence_threshold', 0.7)
        self.beam_width = seg_config.get('beam_width', 5)
        self.lookahead_sentences = seg_config.get('lookahead_sentences', 3)

        # Initialize similarity model (XLM-R)
        self._init_similarity_model()

        # Sentence embedding cache
        self.sentence_embeddings_normalized = None
        self.sentence_to_index = {}

        logger.info(f"Initialized segmenter: min={self.min_tokens}, target={self.target_tokens}, "
                   f"max={self.max_tokens}, beam_width={self.beam_width}")

    def _init_similarity_model(self):
        """Initialize XLM-R model for similarity calculations."""
        try:
            from sentence_transformers import SentenceTransformer
            import torch

            model_name = "sentence-transformers/paraphrase-xlm-r-multilingual-v1"
            logger.info(f"Loading XLM-R similarity model: {model_name}")
            self.similarity_model = SentenceTransformer(model_name)

            # Move to MPS if available
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            self.similarity_model = self.similarity_model.to(device)

            # Create tokenizer for token counting
            self.tokenizer = self.similarity_model.tokenizer

            logger.info(f"XLM-R model initialized on {device}")

        except Exception as e:
            logger.error(f"Failed to initialize similarity model: {e}")
            raise

    def _count_tokens(self, text: str) -> int:
        """Count tokens using XLM-R tokenizer."""
        if not text:
            return 0

        tokens = self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors=None,
            truncation=True,
            max_length=512
        )
        return len(tokens['input_ids'])

    def _precompute_sentence_embeddings(self, sentences: List[str]) -> None:
        """Pre-compute normalized embeddings for all sentences."""
        logger.info(f"Pre-computing embeddings for {len(sentences)} sentences")

        embeddings = self.similarity_model.encode(sentences, convert_to_numpy=True)
        embeddings_array = np.array(embeddings)

        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        self.sentence_embeddings_normalized = embeddings_array / norms

        # Build O(1) lookup dict
        self.sentence_to_index = {text: idx for idx, text in enumerate(sentences)}

        logger.info(f"Pre-computed embeddings for {len(sentences)} sentences")

    def _calculate_similarity(self, sentence_indices1: List[int], sentence_indices2: List[int]) -> float:
        """Calculate similarity between groups of sentences."""
        if self.sentence_embeddings_normalized is None:
            raise RuntimeError("Embeddings not pre-computed")

        # Average embeddings for each group
        emb1 = self.sentence_embeddings_normalized[sentence_indices1].mean(axis=0)
        emb2 = self.sentence_embeddings_normalized[sentence_indices2].mean(axis=0)

        # Cosine similarity
        return float(np.dot(emb1, emb2))

    def _should_segment(self, current_sentences: List[str], next_idx: int,
                       all_sentences: List[str]) -> bool:
        """Check if we should create segment boundary using lookahead."""
        if next_idx >= len(all_sentences):
            return True

        # Get current sentence indices
        current_indices = [self.sentence_to_index[s] for s in current_sentences
                          if s in self.sentence_to_index]

        if not current_indices:
            return True

        # Calculate average similarity with next few sentences
        total_similarity = 0.0
        valid_comparisons = 0

        for i in range(self.lookahead_sentences):
            if next_idx + i >= len(all_sentences):
                break

            next_sentence = all_sentences[next_idx + i]
            if next_sentence not in self.sentence_to_index:
                continue

            next_idx_lookup = self.sentence_to_index[next_sentence]
            similarity = self._calculate_similarity(current_indices, [next_idx_lookup])
            total_similarity += similarity
            valid_comparisons += 1

        if valid_comparisons == 0:
            return True

        avg_similarity = total_similarity / valid_comparisons
        return avg_similarity < self.coherence_threshold

    def segment_from_sentences(self, sentences: List[Dict], content_id: str,
                               word_table: 'WordTable' = None) -> List[Dict[str, Any]]:
        """
        Create retrieval-optimized segments from pre-computed sentences (from Stage 12).

        This is the preferred entry point when sentences are already generated by stage12.
        Uses the sentence data directly instead of re-extracting from word_table.

        Args:
            sentences: List of sentence dicts from stage12 with:
                - sentence_index: Global index within content
                - turn_index: Speaker turn this belongs to
                - sentence_in_turn: Position within turn
                - speaker_id: Database speaker ID
                - text: Sentence text
                - start_time: Start time (word-level precision)
                - end_time: End time (word-level precision)
                - word_count: Number of words
            content_id: Content ID for metadata
            word_table: Optional WordTable for speaker_db_dictionary lookup

        Returns:
            List of segment dictionaries ready for EmbeddingSegment insertion
        """
        start_time = time.time()
        logger.info(f"[{content_id}] Starting semantic segmentation from {len(sentences)} pre-computed sentences")

        if not sentences:
            logger.warning(f"[{content_id}] No sentences provided for segmentation")
            return []

        # Extract sentence texts and metadata (including emotion from stage13)
        all_sentences = [s['text'] for s in sentences]
        sentence_metadata = []
        for s in sentences:
            sentence_metadata.append({
                'sentence_index': s['sentence_index'],
                'turn_index': s['turn_index'],
                'start_time': s['start_time'],
                'end_time': s['end_time'],
                'speaker_id': s['speaker_id'],  # Already database ID
                'word_count': s.get('word_count', len(s['text'].split())),
                # Emotion data from stage13 (may be None if emotion detection was skipped)
                'emotion': s.get('emotion'),
                'emotion_confidence': s.get('emotion_confidence'),
                'emotion_scores': s.get('emotion_scores'),
                'arousal': s.get('arousal'),
                'valence': s.get('valence'),
                'dominance': s.get('dominance')
            })

        # Pre-compute embeddings
        self._precompute_sentence_embeddings(all_sentences)

        # Beam search segmentation
        segments = []
        current_sentences = []
        current_sentence_indices = []  # Track which sentences (by sentence_index) are in segment
        current_tokens = 0

        for sent_idx, sentence in enumerate(all_sentences):
            if sent_idx % 200 == 0:
                logger.info(f"[{content_id}] Processing sentence {sent_idx}/{len(all_sentences)}")

            # Add sentence to current segment
            current_sentences.append(sentence)
            current_sentence_indices.append(sentences[sent_idx]['sentence_index'])
            current_tokens += self._count_tokens(sentence)

            # Check segmentation conditions
            should_segment = False
            reason = ""

            # Token limit reached
            max_safe_tokens = min(self.max_tokens, 350)
            if current_tokens >= max_safe_tokens:
                should_segment = True
                reason = f"token_limit_{current_tokens}"

            # Target reached and low coherence ahead
            elif current_tokens >= self.target_tokens:
                if self._should_segment(current_sentences, sent_idx + 1, all_sentences):
                    should_segment = True
                    reason = f"coherence_{current_tokens}"

            # Create segment
            if should_segment and current_sentences:
                # Get local indices for this segment
                local_indices = list(range(sent_idx - len(current_sentences) + 1, sent_idx + 1))

                segment = self._create_segment_from_sentences(
                    current_sentences, local_indices, current_sentence_indices,
                    sentence_metadata, content_id, reason, word_table
                )
                if segment:
                    segments.append(segment)

                # Reset
                current_sentences = []
                current_sentence_indices = []
                current_tokens = 0

        # Final segment
        if current_sentences:
            local_indices = list(range(len(all_sentences) - len(current_sentences), len(all_sentences)))

            segment = self._create_segment_from_sentences(
                current_sentences, local_indices, current_sentence_indices,
                sentence_metadata, content_id, "final", word_table
            )
            if segment:
                segments.append(segment)

        duration = time.time() - start_time
        logger.info(f"[{content_id}] Created {len(segments)} segments in {duration:.1f}s")

        return segments

    def _create_segment_from_sentences(self, sentences: List[str], local_indices: List[int],
                                       source_sentence_indices: List[int],
                                       sentence_metadata: List[Dict],
                                       content_id: str, reason: str,
                                       word_table: 'WordTable' = None) -> Dict[str, Any]:
        """Create segment dictionary from pre-computed sentences.

        Args:
            sentences: List of sentence texts in this segment
            local_indices: Indices into the local all_sentences list
            source_sentence_indices: Global sentence_index values for source_sentence_ids
            sentence_metadata: Metadata for all sentences
            content_id: Content ID
            reason: Segmentation reason
            word_table: Optional WordTable for speaker_db_dictionary

        Returns:
            Segment dictionary with source_sentence_ids instead of source_transcription_ids
        """
        if not sentences or not local_indices:
            return {}

        # Get timing from sentence metadata
        first_meta = sentence_metadata[local_indices[0]]
        last_meta = sentence_metadata[local_indices[-1]]

        start_time = float(first_meta['start_time'])
        end_time = float(last_meta['end_time'])

        # Skip zero-duration segments
        if abs(end_time - start_time) < 0.001:
            logger.debug(f"[{content_id}] Skipping zero-duration segment [{start_time:.2f}s - {end_time:.2f}s]")
            return {}

        segment_text = smart_join_words(sentences)
        token_count = self._count_tokens(segment_text)

        # Build speaker_positions map
        speaker_positions = {}
        current_pos = 0

        for local_idx, sentence in zip(local_indices, sentences):
            meta = sentence_metadata[local_idx]
            speaker_db_id = meta.get('speaker_id')  # Already database ID from stage12

            if speaker_db_id:
                sentence_start = segment_text.find(sentence, current_pos)

                if sentence_start >= 0:
                    sentence_end = sentence_start + len(sentence)

                    if speaker_db_id not in speaker_positions:
                        speaker_positions[speaker_db_id] = []
                    speaker_positions[speaker_db_id].append([sentence_start, sentence_end])

                    current_pos = sentence_end

        # Collect unique speaker IDs
        speaker_ids = list(set(meta.get('speaker_id') for meta in [sentence_metadata[i] for i in local_indices] if meta.get('speaker_id')))

        # Log speaker_positions generation
        if speaker_positions:
            num_speakers = len(speaker_positions)
            total_ranges = sum(len(ranges) for ranges in speaker_positions.values())
            logger.debug(f"[{content_id}] Generated speaker_positions: {num_speakers} speakers, {total_ranges} total ranges")

        # Aggregate emotion data from sentences into emotion_summary
        # Only include meaningful emotions (exclude neutral/unknown) with confidence > 0.20
        # Format: {"angry": 2, "happy": 1} - count of each interesting emotion
        emotion_summary = None
        emotion_counts = {}  # emotion -> count
        min_confidence = 0.20
        skip_emotions = {'neutral', 'unknown'}

        for local_idx in local_indices:
            meta = sentence_metadata[local_idx]
            emotion = meta.get('emotion')
            confidence = meta.get('emotion_confidence', 0.0)

            # Only count interesting emotions with sufficient confidence
            if emotion and emotion not in skip_emotions and confidence and confidence > min_confidence:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

        if emotion_counts:
            emotion_summary = emotion_counts

        return {
            'text': segment_text,
            'start_time': start_time,
            'end_time': end_time,
            'token_count': token_count,
            'segment_type': reason,
            'speaker_ids': speaker_ids,
            'speaker_names': [],  # Deprecated - use speaker_positions
            'source_sentence_ids': source_sentence_indices,  # References sentences table
            'source_transcription_ids': [],  # Deprecated - kept for backwards compatibility
            'sentence_count': len(sentences),
            'word_count': sum(sentence_metadata[idx].get('word_count', 0) for idx in local_indices),
            'timing_method': 'sentence_precise',
            'speaker_positions': speaker_positions,
            'emotion_summary': emotion_summary  # Aggregated emotion from source sentences
        }

    def segment_from_word_table(self, word_table: WordTable, content_id: str,
                                speaker_turns: List[Dict] = None) -> List[Dict[str, Any]]:
        """
        Create retrieval-optimized segments from WordTable.

        DEPRECATED: Prefer segment_from_sentences() when sentences are available from stage12.

        This is the legacy entry point that extracts sentences from word_table.
        Uses precise word-level timestamps from WordTable for perfect accuracy.

        Args:
            word_table: WordTable with final speaker assignments and word timestamps
            content_id: Content ID for metadata
            speaker_turns: List of speaker turn dictionaries (with db_id, start_time, end_time) for source mapping

        Returns:
            List of segment dictionaries ready for EmbeddingSegment insertion
        """
        start_time = time.time()
        logger.info(f"[{content_id}] Starting semantic segmentation from word table")

        # Get sorted words with speaker assignments
        valid_words = word_table.df[
            (word_table.df['speaker_current'].notna()) &
            (word_table.df['speaker_current'] != '')
        ].sort_values('start').copy()

        if len(valid_words) == 0:
            logger.warning(f"[{content_id}] No valid words for segmentation")
            return []

        logger.info(f"[{content_id}] Processing {len(valid_words)} words")

        # Convert words to sentences with metadata
        all_sentences = []
        sentence_metadata = []

        # Group consecutive words by speaker and sentence boundaries
        current_text = []
        current_word_indices = []

        for idx, (_, word) in enumerate(valid_words.iterrows()):
            current_text.append(word['text'])
            current_word_indices.append(idx)

            # Check if this is end of sentence (simple heuristic: punctuation or speaker change)
            is_end = (
                word['text'].rstrip() in {'.', '!', '?', '...'}
                or (idx + 1 < len(valid_words) and
                    valid_words.iloc[idx + 1]['speaker_current'] != word['speaker_current'])
                or idx == len(valid_words) - 1
            )

            if is_end and current_text:
                # Create sentence from accumulated words
                sentence_text = smart_join_words(current_text)

                # Tokenize into sentences (may split further)
                nltk_sentences = sent_tokenize(sentence_text)

                # If single sentence, use exact word timing
                if len(nltk_sentences) == 1:
                    first_word = valid_words.iloc[current_word_indices[0]]
                    last_word = valid_words.iloc[current_word_indices[-1]]

                    all_sentences.append(nltk_sentences[0])
                    sentence_metadata.append({
                        'start_time': first_word['start'],
                        'end_time': last_word['end'],
                        'speaker': word['speaker_current'],
                        'word_indices': current_word_indices.copy()
                    })
                else:
                    # Multiple sentences - use PRECISE word-level timing by matching words
                    # Build a mapping of normalized words to their indices for fast lookup
                    available_word_indices = set(current_word_indices)
                    word_text_to_indices = {}
                    for word_idx in current_word_indices:
                        word_obj = valid_words.iloc[word_idx]
                        word_text = word_obj['text'].strip().lower()
                        if word_text not in word_text_to_indices:
                            word_text_to_indices[word_text] = []
                        word_text_to_indices[word_text].append(word_idx)

                    for sent in nltk_sentences:
                        # Match sentence words to actual word indices
                        sent_words = sent.split()
                        sent_word_indices = []

                        for sent_word in sent_words:
                            normalized = sent_word.strip().lower()
                            # Find matching word index (prefer unused ones)
                            if normalized in word_text_to_indices:
                                for word_idx in word_text_to_indices[normalized]:
                                    if word_idx in available_word_indices:
                                        sent_word_indices.append(word_idx)
                                        available_word_indices.remove(word_idx)
                                        break

                        # Get precise timing from matched words
                        if sent_word_indices:
                            first_word = valid_words.iloc[sent_word_indices[0]]
                            last_word = valid_words.iloc[sent_word_indices[-1]]
                            sent_start = first_word['start']
                            sent_end = last_word['end']
                        else:
                            # Fallback: use remaining available words
                            if available_word_indices:
                                remaining = sorted(available_word_indices)
                                first_word = valid_words.iloc[remaining[0]]
                                last_word = valid_words.iloc[remaining[-1]]
                                sent_start = first_word['start']
                                sent_end = last_word['end']
                                sent_word_indices = remaining
                                available_word_indices.clear()
                            else:
                                # Should never happen, but fallback to whole range
                                first_word = valid_words.iloc[current_word_indices[0]]
                                last_word = valid_words.iloc[current_word_indices[-1]]
                                sent_start = first_word['start']
                                sent_end = last_word['end']

                        all_sentences.append(sent)
                        sentence_metadata.append({
                            'start_time': sent_start,
                            'end_time': sent_end,
                            'speaker': word['speaker_current'],
                            'word_indices': sent_word_indices
                        })

                # Reset
                current_text = []
                current_word_indices = []

        if not all_sentences:
            logger.warning(f"[{content_id}] No sentences extracted")
            return []

        logger.info(f"[{content_id}] Extracted {len(all_sentences)} sentences")

        # Pre-compute embeddings
        self._precompute_sentence_embeddings(all_sentences)

        # Beam search segmentation
        segments = []
        current_sentences = []
        current_tokens = 0

        for sent_idx, sentence in enumerate(all_sentences):
            if sent_idx % 200 == 0:
                logger.info(f"[{content_id}] Processing sentence {sent_idx}/{len(all_sentences)}")

            # Add sentence to current segment
            current_sentences.append(sentence)
            current_tokens += self._count_tokens(sentence)

            # Check segmentation conditions
            should_segment = False
            reason = ""

            # Token limit reached
            max_safe_tokens = min(self.max_tokens, 350)
            if current_tokens >= max_safe_tokens:
                should_segment = True
                reason = f"token_limit_{current_tokens}"

            # Target reached and low coherence ahead
            elif current_tokens >= self.target_tokens:
                if self._should_segment(current_sentences, sent_idx + 1, all_sentences):
                    should_segment = True
                    reason = f"coherence_{current_tokens}"

            # Create segment
            if should_segment and current_sentences:
                # Get sentence indices for this segment
                sentence_indices = list(range(sent_idx - len(current_sentences) + 1, sent_idx + 1))

                segment = self._create_segment(
                    current_sentences, sentence_indices,
                    sentence_metadata, content_id, reason,
                    speaker_turns, word_table
                )
                if segment:
                    segments.append(segment)

                # Reset
                current_sentences = []
                current_tokens = 0

        # Final segment
        if current_sentences:
            # Get sentence indices for final segment
            sentence_indices = list(range(len(all_sentences) - len(current_sentences), len(all_sentences)))

            segment = self._create_segment(
                current_sentences, sentence_indices,
                sentence_metadata, content_id, "final",
                speaker_turns, word_table
            )
            if segment:
                segments.append(segment)

        duration = time.time() - start_time
        logger.info(f"[{content_id}] Created {len(segments)} segments in {duration:.1f}s")

        return segments

    def _create_segment(self, sentences: List[str], sentence_indices: List[int],
                       sentence_metadata: List[Dict],
                       content_id: str, reason: str,
                       speaker_turns: List[Dict] = None,
                       word_table: 'WordTable' = None) -> Dict[str, Any]:
        """Create segment dictionary from sentences with precise word-level timing."""
        if not sentences or not sentence_indices:
            return {}

        # Get timing from sentence metadata (already has word-level precision)
        first_meta = sentence_metadata[sentence_indices[0]]
        last_meta = sentence_metadata[sentence_indices[-1]]

        start_time = float(first_meta['start_time'])
        end_time = float(last_meta['end_time'])

        # Skip zero-duration segments (start_time == end_time)
        # These are artifacts from edge cases in word timing and contain no retrievable content
        if abs(end_time - start_time) < 0.001:  # Less than 1ms duration
            logger.debug(f"[{content_id}] Skipping zero-duration segment [{start_time:.2f}s - {end_time:.2f}s]")
            return {}

        segment_text = smart_join_words(sentences)
        token_count = self._count_tokens(segment_text)

        # Build speaker_positions map by tracking character positions
        # Format: {2668417: [[0, 280]], 2668422: [[281, 315], [650, 750]]}
        # Keys are speakers.id (speaker_db_id) for direct joining to speakers table
        speaker_positions = {}
        current_pos = 0

        # Get speaker_db_dictionary for mapping local speaker labels to database IDs
        speaker_db_dict = {}
        if word_table and hasattr(word_table, 'speaker_db_dictionary'):
            speaker_db_dict = word_table.speaker_db_dictionary or {}

        for sent_idx, sentence in zip(sentence_indices, sentences):
            meta = sentence_metadata[sent_idx]
            local_speaker = meta.get('speaker')  # e.g., "SPEAKER_00"

            if local_speaker:
                # Map local speaker label to database ID
                speaker_db_id = None
                if local_speaker in speaker_db_dict:
                    speaker_db_id = speaker_db_dict[local_speaker].get('speaker_db_id')

                if speaker_db_id:
                    # Calculate sentence length in the joined text
                    # smart_join_words may add spaces, so find actual position
                    sentence_start = segment_text.find(sentence, current_pos)

                    if sentence_start >= 0:
                        sentence_end = sentence_start + len(sentence)

                        # Add range to speaker's position list using database ID as key
                        if speaker_db_id not in speaker_positions:
                            speaker_positions[speaker_db_id] = []
                        speaker_positions[speaker_db_id].append([sentence_start, sentence_end])

                        # Update position for next sentence
                        current_pos = sentence_end

        # Log speaker_positions generation
        if speaker_positions:
            num_speakers = len(speaker_positions)
            total_ranges = sum(len(ranges) for ranges in speaker_positions.values())
            speaker_ids = list(speaker_positions.keys())
            logger.debug(f"[{content_id}] Generated speaker_positions: {num_speakers} speakers (IDs: {speaker_ids}), {total_ranges} total ranges")

        # Collect speaker information for backwards compatibility with speaker_names
        speakers = set()
        speaker_db_ids = set()

        for sent_idx in sentence_indices:
            meta = sentence_metadata[sent_idx]
            if 'speaker' in meta:
                speakers.add(meta['speaker'])

        # Map segment to source speaker transcription IDs using PRECISE timing
        # Since both segments and speaker turns are derived from the same words,
        # we can use exact timing boundaries for perfect matching
        source_transcription_ids = []
        if speaker_turns:
            # Define segment time range with tiny tolerance for floating point comparison
            tolerance = 0.01  # 10ms tolerance for floating point precision

            for turn in speaker_turns:
                turn_start = turn.get('start_time', 0)
                turn_end = turn.get('end_time', 0)
                turn_db_id = turn.get('db_id')

                if not turn_db_id:
                    continue

                # Calculate overlap between segment and turn
                overlap_start = max(start_time, turn_start)
                overlap_end = min(end_time, turn_end)
                overlap = max(0, overlap_end - overlap_start)

                # If there's ANY meaningful overlap (>10ms), this turn is part of the segment
                # We use a small threshold to handle floating point precision issues
                if overlap > tolerance:
                    source_transcription_ids.append(turn_db_id)

            # CRITICAL CHECK: If no matches found, the pipeline is broken - FAIL THE STITCH
            if not source_transcription_ids:
                error_msg = (
                    f"FATAL: Failed to match segment [{start_time:.2f}s - {end_time:.2f}s] "
                    f"to any speaker turns. Available turns: {len(speaker_turns)}. "
                    f"This should NEVER happen as segments are derived from speaker turns! "
                    f"Pipeline integrity violated."
                )
                logger.error(f"[{content_id}] {error_msg}")

                # Log turn boundaries for debugging
                if speaker_turns:
                    turn_ranges = [f"[{t.get('start_time', 0):.2f}s - {t.get('end_time', 0):.2f}s]"
                                  for t in speaker_turns[:5]]
                    logger.error(f"[{content_id}] First 5 turn ranges: {', '.join(turn_ranges)}")

                # Raise exception to fail the stitch
                raise RuntimeError(error_msg)

        return {
            'text': segment_text,
            'start_time': start_time,
            'end_time': end_time,
            'token_count': token_count,
            'segment_type': reason,
            'speaker_ids': [],  # Will be populated by caller if needed
            'speaker_names': list(speakers),
            'source_transcription_ids': source_transcription_ids,
            'sentence_count': len(sentences),
            'word_count': sum(len(sentence_metadata[idx].get('word_indices', [])) for idx in sentence_indices),
            'timing_method': 'word_table_precise',
            'speaker_positions': speaker_positions  # Character ranges for each speaker
        }


def get_current_segment_version() -> str:
    """
    Get the current segment version by combining stitch version with embedding suffix.

    Returns:
        Current segment version (e.g., 'segment_v14.2_xlmr')
    """
    try:
        config = load_config()

        # Get stitch version (e.g., 'stitch_v14.2')
        stitch_version = config.get('processing', {}).get('stitch', {}).get('current_version', 'stitch_v14')

        # Get embedding suffix (e.g., '_xlmr')
        embedding_suffix = config.get('processing', {}).get('segment', {}).get('embedding_suffix', '_xlmr')

        # Replace 'stitch_' with 'segment_' and append suffix
        segment_version = stitch_version.replace('stitch_', 'segment_') + embedding_suffix

        return segment_version

    except Exception as e:
        logger.error(f"Failed to get segment version: {e}")
        return "segment_v14_xlmr"  # Fallback


def stage14_segment(content_id: str, word_table: WordTable,
                   sentences: List[Dict],
                   test_mode: bool = False) -> Dict[str, Any]:
    """
    Execute Stage 14: Semantic Segmentation.

    Creates retrieval-optimized embedding segments from sentences (after emotion detection).
    Segments reference sentences via source_sentence_ids and aggregate emotion data.

    Args:
        content_id: Content ID
        word_table: WordTable with final speaker assignments (used for speaker_db_dictionary)
        sentences: List of sentence dictionaries from Stage 12 (with emotion from Stage 13). Each dict contains:
                   - sentence_index: Global index within content
                   - turn_index: Speaker turn this belongs to
                   - sentence_in_turn: Position within turn
                   - speaker_id: Database speaker ID
                   - text: Sentence text
                   - start_time: Start time (word-level precision)
                   - end_time: End time (word-level precision)
                   - word_count: Number of words
                   - emotion: Primary emotion (from Stage 13, may be None)
                   - emotion_confidence: Confidence score (from Stage 13, may be None)
        test_mode: If True, skip database operations

    Returns:
        Dictionary with segmentation results including segments with source_sentence_ids and emotion_summary
    """
    start_time = time.time()

    logger.info(f"[{content_id}] Starting Stage 14: Semantic Segmentation")

    result = {
        'status': 'pending',
        'content_id': content_id,
        'stage': 'stage14_segment',
        'data': {
            'segments': []
        },
        'stats': {},
        'error': None
    }

    try:
        # Require sentences - these come from Stage 12 (possibly with emotion from Stage 13)
        if not sentences:
            raise ValueError("Stage 14 requires sentences. "
                           "Sentences must be generated by stage12_output before segmentation.")

        # Create segmenter
        segmenter = TranscriptSegmenter()

        # Generate segments from pre-computed sentences
        logger.info(f"[{content_id}] Segmenting {len(sentences)} sentences from Stage 12")
        segments = segmenter.segment_from_sentences(sentences, content_id, word_table)

        result['data']['segments'] = segments

        # Calculate statistics
        if segments:
            token_counts = [s['token_count'] for s in segments]
            result['stats'] = {
                'duration': time.time() - start_time,
                'segment_count': len(segments),
                'avg_tokens': np.mean(token_counts),
                'min_tokens': np.min(token_counts),
                'max_tokens': np.max(token_counts),
                'total_words': sum(s.get('word_count', 0) for s in segments),
                'timing_method': 'word_table_precise'
            }

            logger.info(f"[{content_id}] Segmentation stats: {len(segments)} segments, "
                       f"avg {result['stats']['avg_tokens']:.0f} tokens")

        # Save segments to JSON file in test mode for debugging
        if test_mode and segments:
            import json
            test_output_dir = get_project_root() / "tests" / "content" / content_id / "outputs"
            test_output_dir.mkdir(parents=True, exist_ok=True)
            segments_file = test_output_dir / f"{content_id}_segments.json"

            with open(segments_file, 'w') as f:
                json.dump(segments, f, indent=2)

            logger.info(f"[{content_id}] Saved {len(segments)} segments to {segments_file}")

        result['status'] = 'success'
        return result

    except Exception as e:
        logger.error(f"[{content_id}] Stage 14 failed: {str(e)}", exc_info=True)
        result.update({
            'status': 'error',
            'error': str(e),
            'duration': time.time() - start_time
        })
        return result
