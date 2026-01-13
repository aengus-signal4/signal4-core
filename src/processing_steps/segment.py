#!/usr/bin/env python3
"""
Transcript Segmentation
=======================

⚠️  DEPRECATED: This standalone segmentation module is being phased out.
   Segmentation is now integrated into the stitch pipeline as Stage 13.
   See: src/processing_steps/stitch_steps/stage13_segment.py

Migration Path:
- Old: download → convert → transcribe → diarize → stitch → **segment** → embed
- New: download → convert → transcribe → diarize → **stitch (with integrated Stage 13)** → embed

Advantages of Integrated Segmentation:
- Perfect timestamp accuracy (uses word-level timings directly from stitch)
- No S3 re-fetching or timestamp reconstruction needed
- Single processing pass creates both SpeakerTranscription and EmbeddingSegment
- ~200 lines of complex timing code eliminated

This file is kept temporarily for:
- Backward compatibility testing
- Migration validation
- Reference implementation

New implementations should use stitch pipeline Stage 13 instead.

---

Original Pipeline Step (Standalone):
1. Loads SpeakerTranscription records from database
2. Segments them into retrieval-optimized chunks using XLM-R similarity model
3. Saves segment metadata to database (without embeddings)

Embeddings are generated separately by scripts/hydrate_embeddings.py for efficiency.
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
import tempfile
from rapidfuzz import fuzz

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.append(str(get_project_root()))

from src.utils.logger import setup_worker_logger
from src.database.session import get_session
from src.database.models import Content, SpeakerTranscription, EmbeddingSegment, Speaker
from src.utils.gpu_embedding_utils import GPUEmbeddingGenerator
from src.storage.s3_utils import S3Storage, S3StorageConfig
from src.processing_steps.stitch_steps.stage1_load import _load_chunks_from_local_or_s3

# Try to import NLTK and download punkt if needed
try:
    import nltk
    nltk.data.find('tokenizers/punkt')
except LookupError:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

from nltk.tokenize import sent_tokenize

logger = setup_worker_logger('segment')


class TranscriptSegmenter:
    """Segments speaker transcriptions into retrieval-optimized chunks."""

    def __init__(self):
        """Initialize the segmenter."""
        # Load config
        self.config = load_config()

        # Segmentation parameters
        seg_config = self.config.get('embedding', {}).get('embedding_segmentation', {})
        self.min_tokens = seg_config.get('min_tokens', 50)
        self.target_tokens = seg_config.get('target_tokens', 250)
        self.max_tokens = seg_config.get('max_tokens', 400)
        self.coherence_threshold = seg_config.get('coherence_threshold', 0.7)
        self.combine_speakers = seg_config.get('combine_speakers', True)
        self.max_combine_gap = seg_config.get('max_combine_gap', 3.0)

        # Beam search parameters
        self.beam_width = seg_config.get('beam_width', 5)
        self.lookahead_sentences = seg_config.get('lookahead_sentences', 3)

        # Initialize similarity model (XLM-R) for segmentation
        self._init_similarity_model()

        # Add cache for pre-computed embeddings (using similarity model)
        self.sentence_embeddings_array = None
        self.sentence_texts = []
        self.sentence_embeddings_normalized = None
        self.sentence_to_index = {}  # Fast O(1) lookup for sentence indices

        logger.info(f"Initialized with params: min={self.min_tokens}, target={self.target_tokens}, "
                   f"max={self.max_tokens}, beam_width={self.beam_width}")

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

    def _init_similarity_model(self):
        """Initialize XLM-R model for similarity calculations during segmentation."""
        try:
            from sentence_transformers import SentenceTransformer
            import torch

            # Use XLM-R for similarity calculations
            similarity_model_name = "sentence-transformers/paraphrase-xlm-r-multilingual-v1"

            logger.info(f"Loading XLM-R similarity model: {similarity_model_name}")
            self.similarity_model = SentenceTransformer(similarity_model_name)

            # Move to MPS if available
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            self.similarity_model = self.similarity_model.to(device)

            # Also create a tokenizer for token counting (using XLM-R tokenizer)
            self.tokenizer = self.similarity_model.tokenizer

            logger.info(f"XLM-R model initialized on {device} for similarity (dimension: {self.similarity_model.get_sentence_embedding_dimension()})")

        except Exception as e:
            logger.error(f"Failed to initialize similarity model: {e}")
            raise

    def _generate_similarity_embeddings(self, texts: List[str]) -> Optional[np.ndarray]:
        """Generate similarity embeddings using XLM-R model"""
        logger.debug(f"Using XLM-R similarity model for {len(texts)} texts")
        return self.similarity_model.encode(texts, convert_to_numpy=True)

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using XLM-R tokenizer."""
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

        # Build fast lookup dict for O(1) sentence-to-index access
        self.sentence_to_index = {text: idx for idx, text in enumerate(all_sentences)}

        # Normalize embeddings for efficient cosine similarity
        norms = np.linalg.norm(self.sentence_embeddings_array, axis=1, keepdims=True)
        if not np.allclose(norms, 1.0, atol=1e-6):
            self.sentence_embeddings_normalized = self.sentence_embeddings_array / norms
            logger.debug("Normalized similarity embeddings for cosine similarity")
        else:
            self.sentence_embeddings_normalized = self.sentence_embeddings_array.copy()
            logger.debug("Similarity embeddings already normalized")

        logger.info(f"Successfully pre-computed XLM-R similarity embeddings for {len(all_sentences)} sentences")

    def _calculate_hierarchical_similarity(self, sentence_indices1: List[int], sentence_indices2: List[int]) -> float:
        """Calculate similarity between groups of sentences using hierarchical embedding composition."""
        if self.sentence_embeddings_normalized is None:
            raise RuntimeError("Normalized embeddings not pre-computed")

        # Get embeddings for both groups and average them
        emb1 = self.sentence_embeddings_normalized[sentence_indices1].mean(axis=0)
        emb2 = self.sentence_embeddings_normalized[sentence_indices2].mean(axis=0)

        # Calculate cosine similarity
        similarity = np.dot(emb1, emb2)
        return float(similarity)

    def _segment_transcriptions(self, transcriptions: List[SpeakerTranscription], content_id: str = None) -> List[Dict]:
        """Segment speaker transcriptions into retrieval-optimized chunks using beam search."""
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
                    'sentence_index': len(all_sentences)
                })

        if not all_sentences:
            logger.warning("No sentences found in transcriptions")
            return []

        logger.info(f"Processing {len(all_sentences)} sentences with beam search (beam_width={self.beam_width})")

        # Pre-compute embeddings for all sentences
        self._precompute_sentence_embeddings(all_sentences)

        # Greedy segmentation with lookahead
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

            # Reason 1: Token limit reached
            max_safe_tokens = min(self.max_tokens, 350)
            if current_tokens >= max_safe_tokens:
                should_segment = True
                reason = f"token_limit_{current_tokens}"

            # Reason 2: Target size reached and low coherence with next few sentences
            elif current_tokens >= self.target_tokens:
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
                sentence_indices = list(range(sent_idx - len(current_sentences) + 1, sent_idx + 1))

                short_reason = reason.split('_')[0][:6]
                segment = self._create_segment_from_sentences(
                    current_sentences, sentence_indices, sentence_metadata,
                    transcriptions, f"beam_{short_reason}", content_id
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
            return True

        current_text = ' '.join(current_sentences)

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
        return avg_similarity < self.coherence_threshold

    def _calculate_fast_similarity(self, current_text: str, next_sentence: str,
                                 current_sentences: List[str], next_sent_idx: int) -> float:
        """Fast similarity calculation using pre-computed embeddings with O(1) lookups."""
        try:
            # Get indices for current sentences using O(1) dict lookup
            current_indices = [self.sentence_to_index[sent] for sent in current_sentences if sent in self.sentence_to_index]

            # Get index for next sentence using O(1) dict lookup
            next_idx = self.sentence_to_index.get(next_sentence)
            if next_idx is None:
                return 0.5

            if not current_indices:
                return 0.5

            return self._calculate_hierarchical_similarity(current_indices, [next_idx])

        except Exception as e:
            logger.debug(f"Error in fast similarity calculation: {e}")
            return 0.5

    def _create_segment_from_sentences(self, sentences: List[str], sentence_indices: List[int],
                                     sentence_metadata: List[Dict],
                                     transcriptions: List[SpeakerTranscription],
                                     reason: str, content_id: str = None) -> Dict:
        """Create a segment dictionary from sentences with precise timing when possible."""
        if not sentences or not sentence_indices:
            return {}

        segment_text = ' '.join(sentences)
        token_count = self._count_tokens(segment_text)

        # Try precise timing using first and last sentences only (more efficient)
        start_time = None
        end_time = None
        timing_method = 'linear_interpolation'

        if content_id:
            whisper_segments = self._load_whisper_chunks_for_content(content_id)

            if whisper_segments:
                first_sentence = sentences[0] if sentences else ""
                last_sentence = sentences[-1] if sentences else ""

                first_precise, _ = self._find_precise_word_timing(whisper_segments, first_sentence)
                _, last_precise = self._find_precise_word_timing(whisper_segments, last_sentence)

                if first_precise is not None and last_precise is not None:
                    start_time = first_precise
                    end_time = last_precise
                    timing_method = 'whisper_sentence_precise'

        # Fallback to linear interpolation
        if start_time is None or end_time is None:
            first_meta = sentence_metadata[sentence_indices[0]]
            last_meta = sentence_metadata[sentence_indices[-1]]

            first_trans = transcriptions[first_meta['transcription_idx']]
            last_trans = transcriptions[last_meta['transcription_idx']]

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
            with tempfile.TemporaryDirectory(prefix=f"timing_{content_id}_") as temp_dir:
                temp_path = Path(temp_dir)

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

        # Extract all words with timestamps
        all_words = []
        punctuation_table = str.maketrans('', '', '.,!?":;()[]{}')

        for whisper_seg in whisper_segments:
            words = whisper_seg.get('words', [])
            for word_data in words:
                word_text = word_data.get('word', '')
                start_time = word_data.get('start')
                end_time = word_data.get('end')

                if word_text and start_time is not None and end_time is not None:
                    clean_text = word_text.lower().strip().translate(punctuation_table).strip()
                    if clean_text:
                        all_words.append({
                            'text': clean_text,
                            'start': start_time,
                            'end': end_time,
                            'original': word_text
                        })

        if not all_words:
            return None, None

        # Clean target text
        target_words = []
        for w in target_text.split():
            clean_w = w.lower().strip().translate(punctuation_table).strip()
            if clean_w:
                target_words.append(clean_w)

        if not target_words:
            return None, None

        # Use vectorized word matching
        return self._vectorized_word_sequence_match(all_words, target_words)

    def _vectorized_word_sequence_match(self, all_words: List[Dict], target_words: List[str]) -> Tuple[Optional[float], Optional[float]]:
        """Vectorized word sequence matching with multiple fallback strategies."""
        if not all_words or not target_words:
            return None, None

        word_texts = np.array([w['text'] for w in all_words], dtype=object)
        word_starts = np.array([w['start'] for w in all_words], dtype=np.float32)
        word_ends = np.array([w['end'] for w in all_words], dtype=np.float32)

        # Strategy 1: Exact sequence matching
        result = self._vectorized_exact_match(word_texts, word_starts, word_ends, target_words)
        if result[0] is not None:
            return result

        # Strategy 2: Fuzzy sequence matching
        result = self._vectorized_fuzzy_match(word_texts, word_starts, word_ends, target_words)
        if result[0] is not None:
            return result

        # Strategy 3: Partial match (first/last words)
        result = self._vectorized_partial_match(word_texts, word_starts, word_ends, target_words)
        if result[0] is not None:
            return result

        return None, None

    def _vectorized_exact_match(self, word_texts: np.ndarray, word_starts: np.ndarray, word_ends: np.ndarray, target_words: List[str]) -> Tuple[Optional[float], Optional[float]]:
        """Fast exact sequence matching using numpy."""
        target_len = len(target_words)
        if target_len > len(word_texts):
            return None, None

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

        tolerance = max(1, int(target_len * 0.2))
        min_matches = int(target_len * 0.8)

        for i in range(len(word_texts) - target_len + tolerance):
            matches = 0
            match_positions = []

            window_size = min(target_len + tolerance, len(word_texts) - i)
            window_texts = word_texts[i:i+window_size]

            target_idx = 0
            for j, word_text in enumerate(window_texts):
                if target_idx < target_len:
                    target_word = target_words[target_idx]

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

        start_words = target_words[:3]
        end_words = target_words[-3:]

        start_time = self._find_word_group_vectorized(word_texts, word_starts, word_ends, start_words, get_start=True)
        end_time = self._find_word_group_vectorized(word_texts, word_starts, word_ends, end_words, get_start=False)

        return start_time, end_time

    def _find_word_group_vectorized(self, word_texts: np.ndarray, word_starts: np.ndarray, word_ends: np.ndarray, search_words: List[str], get_start: bool = True) -> Optional[float]:
        """Vectorized search for a group of words."""
        search_len = len(search_words)
        min_matches = max(1, int(search_len * 0.7))

        for i in range(len(word_texts) - search_len + 1):
            matches = 0
            match_position = None

            window = word_texts[i:i+search_len]
            for j, (word_text, search_word) in enumerate(zip(window, search_words)):
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

    def _estimate_sentence_time(self, transcription: SpeakerTranscription,
                               sent_idx: int, total_sentences: int,
                               use_start: bool = True) -> float:
        """Estimate time for a sentence within a transcription."""
        duration = transcription.end_time - transcription.start_time

        if total_sentences <= 1:
            return transcription.start_time if use_start else transcription.end_time

        if use_start:
            position = sent_idx / total_sentences
        else:
            position = (sent_idx + 1) / total_sentences

        return transcription.start_time + (duration * position)

    def _save_segments_to_s3(self, content_id: str, segments: List[Dict]) -> bool:
        """Save semantic segments to S3 as JSON."""
        try:
            from src.storage.s3_utils import create_s3_storage_from_config
            s3_storage = create_s3_storage_from_config(self.config['storage']['s3'])

            segments_data = {
                "content_id": content_id,
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "pipeline_version": "segment_v4",
                "metadata": {
                    "total_segments": len(segments),
                    "segmentation_method": "beam_search",
                    "similarity_model": "XLM-R",
                    "coherence_threshold": self.coherence_threshold,
                    "target_tokens": self.target_tokens,
                    "min_tokens": self.min_tokens,
                    "max_tokens": self.max_tokens,
                    "beam_width": self.beam_width,
                    "lookahead_sentences": self.lookahead_sentences
                },
                "segments": []
            }

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

            temp_path = Path(f"/tmp/semantic_segments_{content_id}.json")
            with open(temp_path, 'w') as f:
                json.dump(segments_data, f, indent=2)

            s3_key = f"content/{content_id}/semantic_segments.json"
            success = s3_storage.upload_file(str(temp_path), s3_key)

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

    async def process_content(self, content_id: str, rewrite: bool = False,
                            stitch_version: str = None, segment_version: str = None,
                            test_mode: bool = False) -> Dict:
        """Process content to create segments (without embeddings)."""
        if test_mode:
            logger.info(f"[{content_id}] Starting transcript segmentation (TEST MODE - no writes)")
        else:
            logger.info(f"[{content_id}] Starting transcript segmentation")

        try:
            # Clear caches
            self.sentence_embeddings_array = None
            self.sentence_embeddings_normalized = None
            self.sentence_texts = []
            self.sentence_to_index = {}

            # Initialize S3 storage
            s3_config = S3StorageConfig(
                endpoint_url=self.config['storage']['s3']['endpoint_url'],
                access_key=self.config['storage']['s3']['access_key'],
                secret_key=self.config['storage']['s3']['secret_key'],
                bucket_name=self.config['storage']['s3']['bucket_name'],
                use_ssl=self.config['storage']['s3']['use_ssl']
            )
            s3_storage = S3Storage(s3_config)

            with get_session() as session:
                content = session.query(Content).filter_by(content_id=content_id).first()
                if not content:
                    return {'status': 'error', 'error': f'Content {content_id} not found'}

                # Check if already processed - only skip if BOTH DB and S3 exist (never skip in test mode)
                if not rewrite and not test_mode:
                    existing_count = session.query(EmbeddingSegment).filter_by(
                        content_id=content.id
                    ).count()

                    s3_key = f"content/{content_id}/semantic_segments.json"
                    s3_key_gz = f"content/{content_id}/semantic_segments.json.gz"
                    s3_file_exists = s3_storage.file_exists(s3_key) or s3_storage.file_exists(s3_key_gz)

                    if existing_count > 0 and s3_file_exists:
                        logger.info(f"[{content_id}] Already has {existing_count} segments and S3 file. Skipping.")
                        return {
                            'status': 'skipped',
                            'segment_count': existing_count
                        }
                    elif existing_count > 0 and not s3_file_exists:
                        logger.warning(f"[{content_id}] Has {existing_count} DB segments but S3 file missing. Auto-enabling rewrite.")
                        rewrite = True
                    elif existing_count == 0 and s3_file_exists:
                        logger.warning(f"[{content_id}] Has S3 file but no DB segments. Auto-enabling rewrite.")
                        rewrite = True
                    elif existing_count == 0 and not s3_file_exists:
                        logger.info(f"[{content_id}] No existing segments found. Processing fresh.")
                        # rewrite stays False, but we'll proceed to create segments

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

                # Segment transcriptions
                segments = self._segment_transcriptions(transcriptions, content_id)
                logger.info(f"[{content_id}] Created {len(segments)} segments using beam search")

                # Save segments to S3 or local test directory
                if test_mode:
                    # Write to local tests directory instead of S3
                    test_dir = get_project_root() / "tests" / content_id
                    test_dir.mkdir(parents=True, exist_ok=True)
                    test_file = test_dir / "semantic_segments.json"

                    segments_data = {
                        "content_id": content_id,
                        "processed_at": datetime.now(timezone.utc).isoformat(),
                        "pipeline_version": "segment_v4",
                        "test_mode": True,
                        "metadata": {
                            "total_segments": len(segments),
                            "segmentation_method": "beam_search",
                            "similarity_model": "XLM-R",
                            "coherence_threshold": self.coherence_threshold,
                            "target_tokens": self.target_tokens,
                            "min_tokens": self.min_tokens,
                            "max_tokens": self.max_tokens,
                            "beam_width": self.beam_width,
                            "lookahead_sentences": self.lookahead_sentences
                        },
                        "segments": []
                    }

                    for idx, segment in enumerate(segments):
                        segment_data = {
                            "segment_index": idx,
                            "text": segment['text'],
                            "start_time": segment['start_time'],
                            "end_time": segment['end_time'],
                            "token_count": segment['token_count'],
                            "segment_type": segment['segment_type'],
                            "speaker_ids": segment['speaker_ids'],
                            "source_transcription_ids": segment.get('source_ids', []),
                            "timing_method": segment.get('timing_method', 'linear_interpolation')
                        }
                        segments_data["segments"].append(segment_data)

                    with open(test_file, 'w') as f:
                        json.dump(segments_data, f, indent=2)

                    logger.info(f"[{content_id}] TEST MODE: Saved {len(segments)} segments to {test_file}")
                else:
                    if not self._save_segments_to_s3(content_id, segments):
                        logger.warning(f"[{content_id}] Failed to save segments to S3, continuing with database storage")

                # Save segments WITHOUT embeddings (skip in test mode)
                if test_mode:
                    logger.info(f"[{content_id}] TEST MODE: Skipping database operations for {len(segments)} segments")
                    saved_count = len(segments)
                else:
                    if rewrite:
                        session.query(EmbeddingSegment).filter_by(content_id=content.id).delete()
                        session.flush()

                    # Pre-fetch ALL speaker hashes in one query for efficiency
                    import hashlib
                    all_speaker_ids = set()
                    for segment in segments:
                        if segment.get('speaker_ids'):
                            all_speaker_ids.update(segment['speaker_ids'])

                    speaker_hash_map = {}
                    if all_speaker_ids:
                        speaker_hash_map = dict(
                            session.query(Speaker.id, Speaker.speaker_hash)
                            .filter(Speaker.id.in_(all_speaker_ids))
                            .all()
                        )

                    # Build all segment dicts for bulk insert
                    segment_dicts = []
                    current_timestamp = datetime.now(timezone.utc)

                    for idx, segment in enumerate(segments):
                        segment_data = f"{content_id}:{idx}:{segment['text'][:100]}"
                        segment_hash = hashlib.sha256(segment_data.encode()).hexdigest()[:8]

                        # Get speaker hashes from pre-fetched map
                        speaker_hashes = None
                        if segment.get('speaker_ids'):
                            speaker_hashes = [speaker_hash_map.get(sid) for sid in segment['speaker_ids'] if sid in speaker_hash_map]
                            if not speaker_hashes:
                                speaker_hashes = None

                        metadata = {
                            'speaker_ids': segment['speaker_ids'],
                            'segment_method': 'beam_search',
                            'similarity_model': 'XLM-R',
                            'coherence_threshold': self.coherence_threshold,
                            'target_tokens': self.target_tokens,
                            'timing_method': segment.get('timing_method', 'linear_interpolation'),
                            'precise_timing_enabled': True,
                            'pipeline_version': 'segment_v4',
                            'embeddings_pending': True
                        }

                        segment_dicts.append({
                            'content_id': content.id,
                            'segment_index': idx,
                            'text': segment['text'],
                            'start_time': segment['start_time'],
                            'end_time': segment['end_time'],
                            'token_count': segment['token_count'],
                            'segment_type': segment['segment_type'],
                            'source_transcription_ids': segment['source_ids'],
                            'source_start_char': segment.get('source_start_char'),
                            'source_end_char': segment.get('source_end_char'),
                            'embedding': None,
                            'embedding_alt': None,
                            'embedding_alt_model': None,
                            'meta_data': metadata,
                            'created_at': current_timestamp,
                            'stitch_version': stitch_version,
                            'embedding_version': segment_version or "segment_v4",
                            'segment_hash': segment_hash,
                            'content_id_string': content_id,
                            'source_speaker_hashes': speaker_hashes
                        })

                    # Bulk insert all segments at once (much faster than individual inserts)
                    if segment_dicts:
                        session.bulk_insert_mappings(EmbeddingSegment, segment_dicts)
                        saved_count = len(segment_dicts)
                        logger.info(f"[{content_id}] Bulk inserted {saved_count} segments")
                    else:
                        saved_count = 0

                    # Update content status
                    # Note: Embeddings will be generated separately by hydrate_embeddings.py
                    # is_embedded will be set to True once embeddings are hydrated

                    current_segment_version = self.config.get('processing', {}).get('segment', {}).get('current_version', 'segment_v4')
                    meta_data = dict(content.meta_data) if content.meta_data else {}
                    meta_data['segment_version'] = current_segment_version
                    content.meta_data = meta_data

                    session.commit()

                logger.info(f"[{content_id}] Successfully created {saved_count} segments (embeddings will be generated separately)")

                # Log timing statistics
                timing_methods = {}
                for segment in segments:
                    method = segment.get('timing_method', 'linear_interpolation')
                    timing_methods[method] = timing_methods.get(method, 0) + 1

                logger.info(f"[{content_id}] Timing method statistics: {timing_methods}")

                return {
                    'status': 'success',
                    'segment_count': saved_count,
                    'source_transcription_count': len(transcriptions),
                    'similarity_model': 'XLM-R',
                    'timing_methods': timing_methods,
                    'pipeline_version': 'segment_v4',
                    'embeddings_pending': True
                }

        except Exception as e:
            logger.error(f"[{content_id}] Error in segmentation: {e}", exc_info=True)
            if 'session' in locals():
                try:
                    session.rollback()
                except:
                    pass
            return {
                'status': 'error',
                'error': str(e)
            }


async def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description='Segment transcriptions into retrieval chunks')
    parser.add_argument('--content', required=True, help='Content ID to process')
    parser.add_argument('--rewrite', action='store_true', help='Force reprocess')
    parser.add_argument('--test', action='store_true', help='Test mode: process but skip all writes (DB and S3)')
    args = parser.parse_args()

    segmenter = TranscriptSegmenter()
    result = await segmenter.process_content(args.content, args.rewrite, test_mode=args.test)

    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'success' else 1

if __name__ == "__main__":
    asyncio.run(main())