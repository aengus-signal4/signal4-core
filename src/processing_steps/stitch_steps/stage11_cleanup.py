#!/usr/bin/env python3
"""
Stage 11: Final Cleanup and Grammar Enhancement
===============================================

Eleventh stage of the stitch pipeline that performs final cleanup and text enhancement.

Key Responsibilities:
- Consolidate single-word orphan segments bounded by same speaker
- Apply grammar enhancement (punctuation/capitalization) to bad grammar turns only
  - Uses sophisticated detection to avoid enhancing whisper segmentation artifacts
  - Identifies sentence fragments surrounded by good grammar
- Analyze orphaned word segments at turn boundaries (currently disabled)
- Final check: Assign any remaining UNKNOWN words to nearest diarization segment
- Provide final text polishing while preserving previous stage assignments

Input:
- WordTable from Stage 10 with complete speaker assignments
- Audio metadata and timing information
- Word-level grammar flags from Stage 3

Output:
- WordTable with consolidated single-word orphan assignments
- Enhanced text with improved grammar for bad grammar segments only
- Analysis of split sentences and orphan segments (for reference)
- Final cleanup statistics

Key Components:
- PunctuateAllModel: deepmultilingualpunctuation-based model (with transformers fallback)
- GrammarEnhancer: Comprehensive grammar improvement with NER
- ConversationParsingAnalyzer: LLM-based conversation parsing (disabled)
- Single-word orphan consolidation with timing analysis

Methods:
- stage11_cleanup(): Main entry point called by stitch pipeline
- assign_unknown_words_to_nearest_diarization(): Final check for UNKNOWN word assignment
- PunctuateAllModel.punctuate_text(): deepmultilingual punctuation enhancement
- get_orphaned_word_segments(): Identify orphan segments at turn boundaries

Performance:
- Minimal computational cost (2-3% of pipeline time)
- Only processes bad grammar segments for grammar enhancement
- Single-word consolidation based on timing gaps and speaker patterns
- Conservative approach preserving earlier stage assignments

Special Notes:
- Orphan segment analysis disabled to avoid interfering with previous assignments
- Grammar enhancement only applied to segments flagged as bad grammar in Stage 3
- Single-word consolidation uses flexible timing thresholds (1s/5s)
- Maintains full assignment history for debugging
"""

import logging
import time
import re
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import torch
import numpy as np

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# Grammar enhancement dependencies
try:
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available - grammar enhancement will be skipped")

try:
    from deepmultilingualpunctuation import PunctuationModel
    DEEPMULTILINGUAL_AVAILABLE = True
except ImportError:
    DEEPMULTILINGUAL_AVAILABLE = False
    logging.warning("deepmultilingualpunctuation not available - grammar enhancement will be limited")

try:
    import spacy
    from spacy.lang.en import English
    SPACY_AVAILABLE = True
except (ImportError, KeyError) as e:
    SPACY_AVAILABLE = False
    logging.warning(f"SpaCy not available ({type(e).__name__}: {e}) - NER and truecasing will be skipped")

try:
    from langdetect import detect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logging.warning("langdetect not available - language detection will be skipped")

from src.utils.logger import setup_worker_logger
from src.utils.config import get_llm_server_url, get_model_config, get_temperature_config, get_request_timeout
# Model server imports removed - using direct model loading only
from .stage3_tables import WordTable

logger = setup_worker_logger('stitch')


class PunctuateAllModel:
    """Handles punctuation and capitalization using deepmultilingualpunctuation model."""
    
    def __init__(self, device: Optional[str] = None):
        """Initialize the punctuation model using local models only."""
        self.model = None
        self.fallback_model = None
        self.device = device
        self.use_model_server = False  # Model server deprecated
        
        # Direct local model loading only
        logger.info("PunctuateAllModel initialized - using direct local model loading")
    
    def punctuate_text(self, text: str) -> str:
        """
        Add punctuation and capitalization to text.
        
        Args:
            text: Input text without punctuation
            
        Returns:
            Text with punctuation and capitalization
        """
        if not text or not text.strip():
            return text
        
        # Model server deprecated - using local models only
        
        # Fallback to local models
        return self._punctuate_with_local_models(text)
    
    
    def _punctuate_with_local_models(self, text: str) -> str:
        """Punctuate text using local models."""
        # Initialize local models if not already done
        if self.model is None and self.fallback_model is None:
            self._initialize_local_models()
        
        # Use deepmultilingualpunctuation model if available
        if self.model is not None and hasattr(self, 'model_type') and self.model_type == "deepmultilingual":
            try:
                return self._punctuate_with_deepmultilingual(text)
            except Exception as e:
                logger.error(f"Error with deepmultilingualpunctuation: {e}")
                # Fall through to transformers model
        
        # Use transformers fallback model
        if self.fallback_model is not None:
            try:
                # Check token count for transformers model
                max_length = 512
                token_count = len(self.tokenizer.encode(text, add_special_tokens=True))
                
                if token_count > max_length:
                    logger.debug(f"Text too long ({token_count} tokens), chunking...")
                    return self._punctuate_long_text(text, max_length)
                else:
                    return self._punctuate_with_transformers_model(text)
            except Exception as e:
                logger.error(f"Error in transformers punctuation: {e}")
        
        # If all else fails, return original text
        logger.warning("No punctuation models available - returning original text")
        return text
    
    def _initialize_local_models(self):
        """Initialize local punctuation models."""
        # Try deepmultilingualpunctuation first (better model)
        if DEEPMULTILINGUAL_AVAILABLE:
            try:
                logger.info("Loading deepmultilingualpunctuation model...")
                self.model = PunctuationModel()
                logger.info("deepmultilingualpunctuation model loaded successfully")
                self.model_type = "deepmultilingual"
            except Exception as e:
                logger.error(f"Failed to load deepmultilingualpunctuation model: {e}")
                self.model = None
        
        # Fallback to transformers punctuate-all if deepmultilingual fails
        if self.model is None and TRANSFORMERS_AVAILABLE:
            logger.info("Falling back to transformers punctuate-all model...")
            try:
                # Determine device
                if self.device:
                    self.device = self.device
                elif torch.backends.mps.is_available():
                    self.device = "mps"
                    logger.info("Using MPS (Metal Performance Shaders) for acceleration")
                elif torch.cuda.is_available():
                    self.device = "cuda"
                    logger.info("Using CUDA for acceleration")
                else:
                    self.device = "cpu"
                    logger.info("Using CPU for inference")
                
                # Load model and tokenizer
                model_name = "kredor/punctuate-all"
                logger.debug(f"Loading fallback model: {model_name}")
                
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.fallback_model = AutoModelForTokenClassification.from_pretrained(model_name)
                
                # Move model to device
                if self.device != "cpu":
                    self.fallback_model = self.fallback_model.to(self.device)
                
                # Create pipeline for easier usage
                device_id = 0 if self.device == "cuda" else -1 if self.device == "cpu" else None
                if self.device == "mps":
                    # For MPS, we'll use the model directly instead of pipeline
                    self.pipeline = None
                else:
                    self.pipeline = pipeline(
                        "token-classification",
                        model=self.fallback_model,
                        tokenizer=self.tokenizer,
                        device=device_id
                    )
                
                # The model uses direct punctuation marks as labels
                self.punctuation_marks = {'.', ',', '?', '-', ':'}
                self.model_type = "transformers_fallback"
                logger.info(f"Fallback punctuation model loaded successfully on {self.device}")
                
            except Exception as e:
                logger.error(f"Failed to load fallback punctuation model: {e}")
                self.fallback_model = None
        
        # Models initialized (or failed to initialize)
    
    def _punctuate_with_deepmultilingual(self, text: str) -> str:
        """Apply punctuation using deepmultilingualpunctuation model."""
        try:
            # The deepmultilingual model automatically handles punctuation and capitalization
            result = self.model.restore_punctuation(text)
            
            # Post-process to ensure proper capitalization after sentence-ending punctuation
            result = self._ensure_sentence_capitalization(result)
            
            return result
        except Exception as e:
            logger.error(f"Error in deepmultilingual punctuation: {e}")
            return text
    
    def _punctuate_with_transformers_model(self, text: str) -> str:
        """Apply punctuation using the transformers fallback model."""
        if self.pipeline:
            # Use pipeline for non-MPS devices
            outputs = self.pipeline(text)
        else:
            # Manual inference for MPS
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                logits = self.fallback_model(**inputs).logits
                predictions = torch.argmax(logits, dim=-1)
            
            # Convert predictions to pipeline-like format
            tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            outputs = []
            for i, (token, pred_id) in enumerate(zip(tokens, predictions[0])):
                if token not in ['[CLS]', '[SEP]', '[PAD]']:
                    label = self.fallback_model.config.id2label[pred_id.item()]
                    outputs.append({
                        'entity': label,
                        'word': token,
                        'start': i,
                        'end': i + 1
                    })
        
        # Reconstruct text with punctuation
        return self._reconstruct_with_punctuation(text, outputs)
    
    def _punctuate_long_text(self, text: str, max_length: int) -> str:
        """Handle long text by chunking at sentence boundaries."""
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        punctuated_sentences = []
        
        current_chunk = ""
        for sentence in sentences:
            # Check if adding this sentence would exceed limit
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            token_count = len(self.tokenizer.encode(test_chunk, add_special_tokens=True))
            
            if token_count > max_length and current_chunk:
                # Process current chunk and start new one
                punctuated_sentences.append(self._punctuate_with_transformers_model(current_chunk.strip()))
                current_chunk = sentence
            else:
                current_chunk = test_chunk
        
        # Process final chunk
        if current_chunk:
            punctuated_sentences.append(self._punctuate_with_transformers_model(current_chunk.strip()))
        
        return " ".join(punctuated_sentences)
    
    def _reconstruct_with_punctuation(self, original_text: str, outputs: List[Dict]) -> str:
        """Reconstruct text with punctuation from model outputs."""
        if not outputs:
            return original_text
        
        words = original_text.split()
        result_tokens = []
        current_word_idx = 0
        
        for output in outputs:
            token = output['word']
            entity = output['entity']
            
            # Handle subword tokens
            if token.startswith('##'):
                if result_tokens:
                    result_tokens[-1] += token[2:]
            else:
                # Check if this is a punctuation prediction
                if entity in self.punctuation_marks:
                    if result_tokens:
                        result_tokens[-1] += entity
                else:
                    # Regular word token
                    if current_word_idx < len(words):
                        result_tokens.append(words[current_word_idx])
                        current_word_idx += 1
        
        # Add any remaining words
        while current_word_idx < len(words):
            result_tokens.append(words[current_word_idx])
            current_word_idx += 1
        
        # Capitalize first letter of each sentence
        text = " ".join(result_tokens)
        
        # Use the same capitalization method as deepmultilingual
        return self._ensure_sentence_capitalization(text)
    
    def _ensure_sentence_capitalization(self, text: str) -> str:
        """
        Ensure proper capitalization after sentence-ending punctuation.
        This is a post-processing step to handle cases where the model
        doesn't properly capitalize after periods, exclamation marks, or question marks.
        """
        if not text:
            return text
        
        # First, capitalize the very first character if it's a letter
        if text and text[0].isalpha():
            text = text[0].upper() + text[1:]
        
        # Use regex to find positions after sentence-ending punctuation followed by a space
        # This pattern matches: punctuation (.!?) followed by one or more spaces, 
        # then captures the next character
        pattern = r'([.!?])\s+([a-z])'
        
        def capitalize_match(match):
            # Return the punctuation, space(s), and capitalized letter
            return match.group(1) + ' ' + match.group(2).upper()
        
        # Apply the capitalization
        result = re.sub(pattern, capitalize_match, text)
        
        # Also handle cases where there's punctuation followed by quotes, then a letter
        # e.g., ." a -> ." A or !' a -> !' A
        pattern_with_quotes = r'([.!?]["\'])\s+([a-z])'
        
        def capitalize_match_with_quotes(match):
            return match.group(1) + ' ' + match.group(2).upper()
        
        result = re.sub(pattern_with_quotes, capitalize_match_with_quotes, result)
        
        return result


class GrammarEnhancer:
    """Handles comprehensive grammar enhancement including punctuation, capitalization, and NER."""
    
    def __init__(self):
        """Initialize grammar enhancement models."""
        self.punctuator = PunctuateAllModel()  # Will try deepmultilingual first, then transformers fallback
        self.spacy_models = {}  # Dictionary to store language-specific models
        self.language_model_map = {
            'en': 'en_core_web_sm',
            'fr': 'fr_core_news_sm',
            'es': 'es_core_news_sm',
            'de': 'de_core_news_sm',
            'it': 'it_core_news_sm',
            'pt': 'pt_core_news_sm',
            'nl': 'nl_core_news_sm',
            'zh': 'zh_core_web_sm',
            'ja': 'ja_core_news_sm',
            'ru': 'ru_core_news_sm',      # Russian (3,414 items)
            'uk': 'uk_core_news_sm',      # Ukrainian (1,509 items)
            'ca': 'ca_core_news_sm'       # Catalan (357 items)
        }
        
        # Load English model by default if available
        if SPACY_AVAILABLE:
            self._load_spacy_model('en')
    
    def _load_spacy_model(self, lang: str) -> bool:
        """Load a SpaCy model for the specified language."""
        if lang in self.spacy_models:
            return True  # Already loaded
        
        model_name = self.language_model_map.get(lang)
        if not model_name:
            return False
        
        try:
            import spacy
            self.spacy_models[lang] = spacy.load(model_name)
            logger.info(f"SpaCy {lang} model ({model_name}) loaded successfully")
            return True
        except Exception as e:
            logger.debug(f"Failed to load SpaCy {lang} model ({model_name}): {e}")
            return False
    
    def enhance_speaker_turn(self, text: str) -> str:
        """
        Apply comprehensive grammar enhancement to a speaker turn.
        
        Args:
            text: Input text from a speaker turn
            
        Returns:
            Enhanced text with punctuation, capitalization, and NER
        """
        if not text or not text.strip():
            return text
        
        enhanced_text = text.strip()
        
        # Step 1: Apply punctuation and basic capitalization
        if self.punctuator:
            enhanced_text = self.punctuator.punctuate_text(enhanced_text)
        
        # Step 2: Detect language and apply NER-based capitalization if appropriate
        if SPACY_AVAILABLE and LANGDETECT_AVAILABLE:
            try:
                # Detect language
                detected_lang = detect(text)
                logger.debug(f"Detected language: {detected_lang}")
                
                # Apply NER if we have a model for this language
                if detected_lang in self.language_model_map:
                    # Load model if not already loaded
                    if detected_lang not in self.spacy_models:
                        self._load_spacy_model(detected_lang)
                    
                    # Apply NER capitalization if model is available
                    if detected_lang in self.spacy_models:
                        enhanced_text = self._apply_ner_capitalization(enhanced_text, detected_lang)
                else:
                    logger.debug(f"No SpaCy model available for language: {detected_lang}")
            except Exception as e:
                logger.debug(f"Language detection failed: {e}")
        
        return enhanced_text
    
    def _apply_ner_capitalization(self, text: str, lang: str) -> str:
        """Apply NER-based capitalization improvements using language-specific model."""
        try:
            if lang not in self.spacy_models:
                logger.debug(f"No SpaCy model loaded for language: {lang}")
                return text
            
            nlp = self.spacy_models[lang]
            doc = nlp(text)
            words = text.split()
            
            # Track which words have been processed to avoid conflicts
            processed_positions = set()
            
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT']:
                    # Get the entity text and its component words
                    entity_words = ent.text.split()
                    entity_text_lower = ent.text.lower()
                    
                    # Find the best match for this entity in the word list
                    best_match_pos = None
                    best_match_score = 0
                    
                    for i in range(len(words) - len(entity_words) + 1):
                        # Check if this position is already processed
                        if any(pos in processed_positions for pos in range(i, i + len(entity_words))):
                            continue
                        
                        # Calculate match score for this position
                        match_score = 0
                        for j, ent_word in enumerate(entity_words):
                            if i + j < len(words):
                                word_lower = words[i + j].lower().strip('.,!?;:')
                                ent_word_lower = ent_word.lower().strip('.,!?;:')
                                if word_lower == ent_word_lower:
                                    match_score += 1
                                elif word_lower.startswith(ent_word_lower) or ent_word_lower.startswith(word_lower):
                                    match_score += 0.5
                        
                        # If this is a better match, record it
                        if match_score > best_match_score and match_score >= len(entity_words) * 0.7:
                            best_match_score = match_score
                            best_match_pos = i
                    
                    # Apply capitalization at the best match position
                    if best_match_pos is not None:
                        for j, ent_word in enumerate(entity_words):
                            pos = best_match_pos + j
                            if pos < len(words) and pos not in processed_positions:
                                # Preserve punctuation while capitalizing
                                word = words[pos]
                                # Find non-alphabetic suffix (punctuation)
                                alpha_part = ""
                                punct_part = ""
                                for char in word:
                                    if char.isalpha():
                                        alpha_part += char
                                    else:
                                        punct_part += char
                                
                                # Apply proper title case to the alphabetic part
                                if alpha_part:
                                    # Use title() for proper capitalization
                                    capitalized_word = alpha_part.title() + punct_part
                                    words[pos] = capitalized_word
                                    processed_positions.add(pos)
            
            return " ".join(words)
            
        except Exception as e:
            logger.error(f"Error in NER capitalization: {e}")
            return text



def get_orphaned_word_segments(word_table: WordTable, max_orphan_size: int = 4, context_words: int = 5) -> List[Dict]:
    """
    Identify orphaned word segments - short segments at the beginning or end of speaker turns
    that might actually belong to adjacent speakers.
    
    Args:
        word_table: WordTable with speaker assignments
        max_orphan_size: Maximum number of words to consider as potential orphans
        context_words: Number of context words to include around orphan segments
        
    Returns:
        List of orphan segment dictionaries with metadata
    """
    if word_table is None or word_table.df is None or len(word_table.df) == 0:
        return []
    
    sorted_words = word_table.df.sort_values('start').reset_index(drop=True)
    
    if len(sorted_words) < 3:  # Need at least 3 words to have orphans
        return []
    
    # Group consecutive words by speaker to identify speaker turns
    speaker_turns = []
    current_speaker = sorted_words.iloc[0]['speaker_current']
    turn_start_idx = 0
    
    for i in range(1, len(sorted_words)):
        next_speaker = sorted_words.iloc[i]['speaker_current']
        if next_speaker != current_speaker:
            # End of current turn
            if current_speaker != 'UNKNOWN':
                speaker_turns.append({
                    'speaker': current_speaker,
                    'start_idx': turn_start_idx,
                    'end_idx': i - 1,
                    'word_count': i - turn_start_idx
                })
            
            # Start of new turn
            current_speaker = next_speaker
            turn_start_idx = i
    
    # Add final turn
    if current_speaker != 'UNKNOWN':
        speaker_turns.append({
            'speaker': current_speaker,
            'start_idx': turn_start_idx,
            'end_idx': len(sorted_words) - 1,
            'word_count': len(sorted_words) - turn_start_idx
        })
    
    if len(speaker_turns) < 2:
        logger.info("Need at least 2 speaker turns to identify orphans - no orphans to extract")
        return []
    
    orphan_segments = []
    orphan_id = 0
    
    # Look for short segments that might be orphans
    for i, turn in enumerate(speaker_turns):
        # First, check if entire short turns are incomplete sentences (new logic)
        if turn['word_count'] <= max_orphan_size and i < len(speaker_turns) - 1:
            # Get the text of this short turn
            turn_words = sorted_words.iloc[turn['start_idx']:turn['end_idx'] + 1]
            turn_text = ' '.join(turn_words['text'].tolist()).strip()
            
            # Check if turn doesn't end with proper sentence termination (.!?)
            if not any(turn_text.endswith(punct) for punct in ['.', '!', '?']):
                # Check if next turn looks like it continues this sentence
                next_turn = speaker_turns[i + 1]
                next_turn_words = sorted_words.iloc[next_turn['start_idx']:min(next_turn['start_idx'] + 3, next_turn['end_idx'] + 1)]
                next_start_text = ' '.join(next_turn_words['text'].tolist()).strip()
                
                # Simple continuation check: if next turn starts with lowercase, it's likely a continuation
                # This catches any language without word-specific logic
                is_continuation = next_start_text and next_start_text[0].islower()
                
                if is_continuation:
                    # Treat entire short turn as orphan that should go to next speaker
                    context_start = max(0, turn['start_idx'] - context_words)
                    context_end = min(len(sorted_words) - 1, turn['end_idx'] + context_words)
                    context_words_df = sorted_words.iloc[context_start:context_end + 1]
                    
                    orphan_segments.append({
                        'orphan_id': orphan_id,
                        'type': 'entire_turn_orphan',
                        'current_speaker': turn['speaker'],
                        'previous_speaker': speaker_turns[i - 1]['speaker'] if i > 0 else None,
                        'next_speaker': next_turn['speaker'],
                        'orphan_word_indices': turn_words.index.tolist(),
                        'context_word_indices': context_words_df.index.tolist(),
                        'orphan_text': turn_text,
                        'context_text': ' '.join(context_words_df['text'].tolist()),
                        'orphan_word_count': turn['word_count'],
                        'start_time': turn_words['start'].min(),
                        'end_time': turn_words['end'].max(),
                        'duration': turn_words['end'].max() - turn_words['start'].min(),
                        'turn_total_words': turn['word_count'],
                        'continuation_hint': next_start_text[:30]
                    })
                    orphan_id += 1
            
            continue  # Skip further processing for short turns
        
        # Skip very short overall turns that don't look like continuations
        if turn['word_count'] <= max_orphan_size:
            continue
        
        # Check for orphan at start of turn (might belong to previous speaker)
        if i > 0 and turn['word_count'] > max_orphan_size:
            potential_orphan_end = min(turn['start_idx'] + max_orphan_size - 1, turn['end_idx'])
            orphan_words = sorted_words.iloc[turn['start_idx']:potential_orphan_end + 1]
            
            # Only consider as orphan if it's a meaningful fragment
            if len(orphan_words) >= 2:
                # Get context for analysis
                context_start = max(0, turn['start_idx'] - context_words)
                context_end = min(len(sorted_words) - 1, potential_orphan_end + context_words)
                context_words_df = sorted_words.iloc[context_start:context_end + 1]
                
                previous_turn = speaker_turns[i - 1]
                
                # Get previous speaker's end time and last word for conservative analysis
                previous_turn_words = sorted_words.iloc[previous_turn['start_idx']:previous_turn['end_idx'] + 1]
                previous_speaker_end_time = previous_turn_words.iloc[-1]['end'] if len(previous_turn_words) > 0 else None
                previous_speaker_last_word = previous_turn_words.iloc[-1]['text'] if len(previous_turn_words) > 0 else ''
                
                orphan_segments.append({
                    'orphan_id': orphan_id,
                    'type': 'turn_start_orphan',
                    'current_speaker': turn['speaker'],
                    'previous_speaker': previous_turn['speaker'],
                    'next_speaker': speaker_turns[i + 1]['speaker'] if i + 1 < len(speaker_turns) else None,
                    'orphan_word_indices': orphan_words.index.tolist(),
                    'context_word_indices': context_words_df.index.tolist(),
                    'orphan_text': ' '.join(orphan_words['text'].tolist()),
                    'context_text': ' '.join(context_words_df['text'].tolist()),
                    'orphan_word_count': len(orphan_words),
                    'start_time': orphan_words.iloc[0]['start'],
                    'end_time': orphan_words.iloc[-1]['end'],
                    'duration': orphan_words.iloc[-1]['end'] - orphan_words.iloc[0]['start'],
                    'turn_total_words': turn['word_count'],
                    # Conservative analysis fields
                    'previous_speaker_end_time': previous_speaker_end_time,
                    'previous_speaker_last_word': previous_speaker_last_word
                })
                orphan_id += 1
        
        # Check for orphan at end of turn (might belong to next speaker)
        if i < len(speaker_turns) - 1 and turn['word_count'] > max_orphan_size:
            potential_orphan_start = max(turn['start_idx'], turn['end_idx'] - max_orphan_size + 1)
            orphan_words = sorted_words.iloc[potential_orphan_start:turn['end_idx'] + 1]
            
            # Only consider as orphan if it's a meaningful fragment
            if len(orphan_words) >= 2:
                # Get context for analysis
                context_start = max(0, potential_orphan_start - context_words)
                context_end = min(len(sorted_words) - 1, turn['end_idx'] + context_words)
                context_words_df = sorted_words.iloc[context_start:context_end + 1]
                
                next_turn = speaker_turns[i + 1]
                
                # Get next speaker's start time and first word for conservative analysis
                next_turn_words = sorted_words.iloc[next_turn['start_idx']:next_turn['end_idx'] + 1]
                next_speaker_start_time = next_turn_words.iloc[0]['start'] if len(next_turn_words) > 0 else None
                next_speaker_first_word = next_turn_words.iloc[0]['text'] if len(next_turn_words) > 0 else ''
                
                orphan_segments.append({
                    'orphan_id': orphan_id,
                    'type': 'turn_end_orphan',
                    'current_speaker': turn['speaker'],
                    'previous_speaker': speaker_turns[i - 1]['speaker'] if i > 0 else None,
                    'next_speaker': next_turn['speaker'],
                    'orphan_word_indices': orphan_words.index.tolist(),
                    'context_word_indices': context_words_df.index.tolist(),
                    'orphan_text': ' '.join(orphan_words['text'].tolist()),
                    'context_text': ' '.join(context_words_df['text'].tolist()),
                    'orphan_word_count': len(orphan_words),
                    'start_time': orphan_words.iloc[0]['start'],
                    'end_time': orphan_words.iloc[-1]['end'],
                    'duration': orphan_words.iloc[-1]['end'] - orphan_words.iloc[0]['start'],
                    'turn_total_words': turn['word_count'],
                    # Conservative analysis fields
                    'next_speaker_start_time': next_speaker_start_time,
                    'next_speaker_first_word': next_speaker_first_word
                })
                orphan_id += 1
    
    logger.info(f"Identified {len(orphan_segments)} potential orphan segments from {len(speaker_turns)} speaker turns")
    return orphan_segments


def get_sentences_from_word_table(word_table: WordTable) -> List[Dict]:
    """
    Identify orphaned word segments that might be misassigned between speakers.
    This focuses on segments at turn boundaries that might belong to adjacent speakers.
    
    Args:
        word_table: WordTable with speaker assignments
        
    Returns:
        List of orphan segment dictionaries with metadata
    """
    # Increase max_orphan_size to catch longer sentence fragments like "officer what's this in" (5 words)
    return get_orphaned_word_segments(word_table, max_orphan_size=5, context_words=8)

def identify_split_sentences(orphan_segments: List[Dict]) -> List[Dict]:
    """
    Analyze orphan segments to determine if they should be reassigned to adjacent speakers.
    
    Args:
        orphan_segments: List of orphan segment dictionaries from get_orphaned_word_segments
        
    Returns:
        List of orphan analysis dictionaries with detailed assessment
    """
    analyzed_orphans = []
    
    for orphan in orphan_segments:
        # All orphans are potential candidates for reassignment
        orphan_type = orphan['type']  # 'turn_start_orphan', 'turn_end_orphan', or 'entire_turn_orphan'
        
        # Analyze the orphan for reassignment potential
        reassignment_indicators = []
        
        # Short orphans are more likely to be misassigned
        if orphan['orphan_word_count'] <= 2:
            reassignment_indicators.append('very_short_segment')
        elif orphan['orphan_word_count'] <= 4:
            reassignment_indicators.append('short_segment')
        
        # Orphans that are a small fraction of the total turn might be misassigned
        orphan_percentage = orphan['orphan_word_count'] / orphan['turn_total_words']
        if orphan_percentage < 0.1:  # Less than 10% of the turn
            reassignment_indicators.append('small_fraction_of_turn')
        
        # Check if orphan text seems to continue or start a sentence
        orphan_text = orphan['orphan_text'].strip()
        
        # Common indicators of sentence continuation/start
        continuation_patterns = [
            r"^(and|but|or|so|then|now|well|um|uh|because|since|after|before|when|while|if|unless|although|though|however|therefore|thus|hence)",
            r"^(going|gonna|want|need|have|had|has|will|would|could|should|might|may|can|do|did|does|don't|doesn't|didn't|won't|wouldn't|couldn't|shouldn't)",
            r"^(we're|you're|they're|i'm|he's|she's|it's|that's|there's|here's|what's|who's|where's|when's|why's|how's)",
            r"\w+ing$",  # Words ending in -ing often continue thoughts
            r"^\w+ly\b",  # Adverbs often continue sentences
        ]
        
        sentence_start_patterns = [
            r"^(yeah|yes|no|okay|ok|right|sure|exactly|absolutely|definitely|probably|maybe|perhaps|actually|basically|essentially|fundamentally|honestly|frankly)",
            r"^[A-Z]\w+",  # Proper capitalization might indicate sentence start
        ]
        
        orphan_lower = orphan_text.lower()
        
        # Check for continuation indicators
        for pattern in continuation_patterns:
            if re.search(pattern, orphan_lower):
                reassignment_indicators.append('continuation_pattern')
                break
        
        # Check for sentence start indicators
        for pattern in sentence_start_patterns:
            if re.search(pattern, orphan_text):
                reassignment_indicators.append('sentence_start_pattern')
                break
        
        # Determine likely reassignment target
        if orphan_type == 'turn_start_orphan':
            # Orphan at start of turn - might belong to previous speaker
            candidate_speaker = orphan['previous_speaker']
            reassignment_type = 'assign_to_previous'
        elif orphan_type == 'entire_turn_orphan':
            # Entire short turn that looks incomplete - likely belongs to next speaker
            candidate_speaker = orphan['next_speaker']
            reassignment_type = 'assign_to_following'
        else:  # turn_end_orphan
            # Orphan at end of turn - might belong to next speaker
            candidate_speaker = orphan['next_speaker']
            reassignment_type = 'assign_to_following'
        
        # Calculate confidence based on indicators  
        if orphan_type == 'entire_turn_orphan':
            confidence = 0.8  # Higher base confidence for entire incomplete turns
        else:
            confidence = 0.5  # Base confidence for partial orphans
        
        # PUNCTUATION-BASED HANGING SENTENCE DETECTION (language-neutral)
        punctuation_blocks = []
        
        # Get context around the speaker boundary (4 words each direction)
        context_text = orphan.get('context_text', '')
        orphan_text = orphan.get('orphan_text', '').strip()
        
        if orphan_type == 'turn_start_orphan' and context_text and orphan_text:
            # Find the orphan position in context
            orphan_start = context_text.find(orphan_text)
            if orphan_start >= 0:
                # Get 4 words before orphan (previous speaker's end)
                before_orphan = context_text[:orphan_start].strip()
                before_words = before_orphan.split()[-4:] if before_orphan else []
                
                # Get 4 words after orphan start (including orphan)
                after_start = orphan_start
                after_orphan = context_text[after_start:].strip()
                after_words = after_orphan.split()[:4] if after_orphan else []
                
                # Check if previous speaker ended with complete sentence punctuation
                if before_words:
                    last_word = before_words[-1]
                    # Complete sentence endings - strong signal this is a NEW sentence
                    if any(punct in last_word for punct in ['.', '!', '?', ':', ';']):
                        confidence = 0.05  # Very low confidence - likely new sentence
                        punctuation_blocks.append('prev_complete_sentence')
                    
                    # Incomplete sentence endings - potential continuation
                    elif any(punct in last_word for punct in [',', '-', '–', '—', '...']):
                        # Check if orphan looks like it continues the thought
                        if after_words:
                            first_orphan_word = after_words[0]
                            # If orphan starts with lowercase or continuation word, likely continuation
                            if (first_orphan_word[0].islower() or 
                                first_orphan_word.lower() in ['and', 'but', 'or', 'so', 'then', 'because', 'since']):
                                confidence += 0.4  # Higher confidence for true continuations
                                punctuation_blocks.append('hanging_comma_continuation')
                            else:
                                # Even with comma, if next word is capitalized, might be new sentence
                                confidence -= 0.2
                                punctuation_blocks.append('comma_but_capitalized')
                    
                    # No punctuation at end - check for hanging patterns
                    else:
                        # Last word has no punctuation - could be incomplete
                        # This is where we need to be careful - just because there's no punctuation
                        # doesn't mean it's incomplete. Only boost confidence if there are other signals.
                        pass  # Keep default confidence
                
                # Additional check: Does orphan start mid-sentence (lowercase)?
                if after_words:
                    first_orphan_word = after_words[0]
                    if first_orphan_word[0].islower():
                        confidence += 0.3  # Likely continuation if starts lowercase
                        punctuation_blocks.append('lowercase_start_continuation')
                    elif first_orphan_word[0].isupper():
                        # Uppercase start suggests new sentence, especially after period
                        if 'prev_complete_sentence' in punctuation_blocks:
                            confidence = 0.02  # Extremely low - almost certainly new sentence
                            punctuation_blocks.append('uppercase_after_period')
                        else:
                            confidence -= 0.1  # Slight reduction for uppercase start
                            punctuation_blocks.append('uppercase_start')
        
        # Now add positive indicators
        if 'very_short_segment' in reassignment_indicators:
            confidence += 0.2
        elif 'short_segment' in reassignment_indicators:
            confidence += 0.1
        
        if 'small_fraction_of_turn' in reassignment_indicators:
            confidence += 0.15
        
        if 'continuation_pattern' in reassignment_indicators:
            confidence += 0.25
        
        if 'sentence_start_pattern' in reassignment_indicators and orphan_type == 'turn_end_orphan':
            confidence -= 0.2  # Sentence starts are less likely to be reassigned from end of turn
        
        # Apply punctuation analysis minimum threshold 
        if punctuation_blocks:
            confidence = max(confidence, 0.01)  # Extremely low minimum when punctuation blocks exist
        
        confidence = min(confidence, 0.95)  # Cap confidence
        
        # Log punctuation analysis for debugging
        if punctuation_blocks:
            logger.info(f"Punctuation analysis for orphan '{orphan_text[:50]}': {punctuation_blocks}, confidence: {confidence:.3f}")
        
        # Calculate speaker changes in this orphan segment
        speakers_in_orphan = []
        if 'orphan_word_indices' in orphan:
            # This should be available from the word table context
            speakers_in_orphan = [orphan['current_speaker']]
            if candidate_speaker and candidate_speaker != orphan['current_speaker']:
                speakers_in_orphan.append(candidate_speaker)
        
        change_count = max(0, len(set(speakers_in_orphan)) - 1)  # Number of speaker changes
        
        # Extract prev and next context from the full context_text
        # The context_text includes words before and after the orphan
        context_text = orphan.get('context_text', '')
        orphan_text = orphan.get('orphan_text', '')
        
        # Split context around the orphan text
        prev_context = 'N/A'
        next_context = 'N/A'
        
        if context_text and orphan_text:
            # Find orphan text within context
            orphan_start = context_text.find(orphan_text)
            if orphan_start >= 0:
                # Extract text before orphan
                prev_context = context_text[:orphan_start].strip()
                # Extract text after orphan  
                orphan_end = orphan_start + len(orphan_text)
                next_context = context_text[orphan_end:].strip()
                
                # Clean up and limit length
                if not prev_context:
                    prev_context = 'N/A'
                if not next_context:
                    next_context = 'N/A'
        
        analyzed_orphan = {
            'fragment_id': orphan['orphan_id'],  # Keep consistent naming with rest of pipeline
            'orphan_id': orphan['orphan_id'],
            'type': orphan_type,
            'word_indices': orphan['orphan_word_indices'],
            'context_word_indices': orphan['context_word_indices'],
            'start_time': orphan['start_time'],
            'end_time': orphan['end_time'],
            'duration': orphan['duration'],
            'word_count': orphan['orphan_word_count'],
            'text': orphan['orphan_text'],
            'context_text': orphan['context_text'],
            'prev_context': prev_context,
            'next_context': next_context,
            'current_speaker': orphan['current_speaker'],
            'candidate_speaker': candidate_speaker,
            'previous_speaker': orphan['previous_speaker'],
            'next_speaker': orphan['next_speaker'],
            'reassignment_type': reassignment_type,
            'reassignment_confidence': confidence,
            'reassignment_indicators': reassignment_indicators,
            'turn_total_words': orphan['turn_total_words'],
            'orphan_percentage': orphan_percentage,
            'split_type': 'orphan_reassignment',  # Consistent with pipeline
            'unique_speakers': [orphan['current_speaker'], candidate_speaker] if candidate_speaker else [orphan['current_speaker']],
            'change_count': change_count,  # Number of speaker changes in this segment
            'split_confidence': confidence  # Add this for the output formatter
        }
        
        analyzed_orphans.append(analyzed_orphan)
    
    logger.info(f"Analyzed {len(analyzed_orphans)} orphan segments for potential reassignment")
    
    # Log breakdown by type and confidence
    if analyzed_orphans:
        type_counts = {}
        high_confidence_count = 0
        
        for orphan in analyzed_orphans:
            orphan_type = orphan['type']
            type_counts[orphan_type] = type_counts.get(orphan_type, 0) + 1
            
            if orphan['reassignment_confidence'] > 0.7:
                high_confidence_count += 1
        
        logger.info(f"Orphan types: {dict(type_counts)}")
        logger.info(f"High confidence reassignments (>0.7): {high_confidence_count}/{len(analyzed_orphans)}")
    
    return analyzed_orphans





def apply_consolidation_decision(word_table: WordTable, 
                               orphan_segment: Dict, 
                               decision: Dict) -> int:
    """
    Apply LLM orphan reassignment decision to the word table.
    
    Args:
        word_table: WordTable to modify
        orphan_segment: The orphan segment being processed
        decision: LLM decision dictionary
        
    Returns:
        Number of words updated
    """
    if decision['action'] == 'leave_split':
        # No changes needed
        return 0
    
    word_indices = orphan_segment['word_indices']
    assigned_speaker = decision.get('assigned_speaker')
    method = decision.get('method')
    confidence = decision.get('confidence', 0.7)
    
    if not assigned_speaker or assigned_speaker == 'UNKNOWN' or assigned_speaker is None:
        logger.warning(f"Cannot reassign orphan {orphan_segment.get('fragment_id', 'unknown')}: invalid speaker '{assigned_speaker}'")
        logger.warning(f"  Decision was: {decision}")
        logger.warning(f"  Orphan segment type: {orphan_segment.get('type', 'unknown')}")
        logger.warning(f"  Orphan at turn boundary: prev='{orphan_segment.get('prev_speaker', 'N/A')}', next='{orphan_segment.get('next_speaker', 'N/A')}'")
        return 0
    
    # Update all words in the orphan segment to the assigned speaker using tracked assignment
    words_updated = 0
    for word_idx in word_indices:
        if word_idx in word_table.df.index:
            word_table.assign_speaker_to_word_by_index(
                idx=word_idx,
                speaker=assigned_speaker,
                method=method,
                confidence=confidence,
                stage='stage8_split_sentences',
                reason=f"Orphan segment reassignment via {method}"
            )
            
            # Update metadata with reassignment info
            current_metadata = word_table.df.at[word_idx, 'metadata']
            if isinstance(current_metadata, dict):
                current_metadata['reassignment_confidence'] = confidence
                current_metadata['reassignment_method'] = method
                current_metadata['orphan_reassigned'] = True
                current_metadata['original_orphan_type'] = orphan_segment['type']
            else:
                word_table.df.at[word_idx, 'metadata'] = {
                    'reassignment_confidence': confidence,
                    'reassignment_method': method,
                    'orphan_reassigned': True,
                    'original_orphan_type': orphan_segment['type']
                }
            
            words_updated += 1
    
    logger.info(f"Reassigned orphan {orphan_segment['fragment_id']} to {assigned_speaker}: {words_updated} words updated")
    return words_updated


def format_split_sentence_output(split_sentences: List[Dict], word_table: WordTable) -> str:
    """
    Format split sentences for output similar to transcript_detailed.txt format.
    
    Args:
        split_sentences: List of split sentence dictionaries
        word_table: WordTable for word details
        
    Returns:
        Formatted string for output
    """
    if not split_sentences:
        return "No split sentences detected.\n"
    
    output_lines = []
    output_lines.append("Split Sentence Analysis")
    output_lines.append("=" * 50)
    output_lines.append(f"Found {len(split_sentences)} sentences split across multiple speakers\n")
    
    sorted_words = word_table.df.sort_values('start').reset_index(drop=True)
    
    for split_sentence in split_sentences:
        # Header with sentence info
        start_time = split_sentence['start_time']
        end_time = split_sentence['end_time']
        duration = split_sentence['duration']
        split_type = split_sentence['split_type']
        change_count = split_sentence['change_count']
        
        minutes_start = int(start_time // 60)
        seconds_start = int(start_time % 60)
        minutes_end = int(end_time // 60)
        seconds_end = int(end_time % 60)
        
        confidence = split_sentence.get('split_confidence', 0.0)
        dominant_speaker = split_sentence.get('dominant_speaker', 'UNKNOWN')
        dominant_pct = split_sentence.get('dominant_percentage', 0.0)
        
        output_lines.append(f"[{int(start_time)}s-{int(end_time)}s] [{minutes_start:02d}:{seconds_start:02d}-{minutes_end:02d}:{seconds_end:02d}] "
                           f"[{split_type}] [{change_count} changes] [{duration:.1f}s] [conf:{confidence:.2f}]")
        output_lines.append(f"Sentence: {split_sentence['text']}")
        output_lines.append(f"Speakers: {' -> '.join(split_sentence['unique_speakers'])} (dominant: {dominant_speaker} {dominant_pct:.1%})")
        
        # Show misassignment indicators if any
        indicators = split_sentence.get('misassignment_indicators', [])
        if indicators:
            output_lines.append(f"Indicators: {', '.join(indicators)}")
        
        output_lines.append("")
        
        # Word-by-word breakdown using word_indices
        word_indices = split_sentence['word_indices']
        sentence_words = word_table.df.loc[word_indices].sort_values('start')
        
        for _, word in sentence_words.iterrows():
            word_time = word['start']
            minutes = int(word_time // 60)
            seconds = int(word_time % 60)
            speaker = word['speaker_current']
            method = word.get('resolution_method', 'unknown')
            confidence = word.get('assignment_confidence', 0.0)
            
            # Format method to 8 characters (similar to stage11_output)
            if len(method) > 8:
                method_short = method[:8]
            else:
                method_short = method.ljust(8)
            
            output_lines.append(f"    - [{word_time:.2f}s] [{minutes:02d}:{seconds:02d}] [{speaker}] [{method_short}] [{confidence:.1f}] {word['text']}")
        
        output_lines.append("")
    
    # Summary statistics
    output_lines.append("Summary Statistics")
    output_lines.append("-" * 30)
    
    split_types = {}
    total_duration = 0
    total_words = 0
    
    for split_sentence in split_sentences:
        split_type = split_sentence['split_type']
        split_types[split_type] = split_types.get(split_type, 0) + 1
        total_duration += split_sentence['duration']
        total_words += split_sentence['word_count']
    
    output_lines.append(f"Total split sentences: {len(split_sentences)}")
    output_lines.append(f"Total duration: {total_duration:.1f}s")
    output_lines.append(f"Total words: {total_words}")
    output_lines.append(f"Average duration per split: {total_duration/len(split_sentences):.1f}s")
    output_lines.append("")
    
    output_lines.append("Split types:")
    for split_type, count in sorted(split_types.items()):
        percentage = (count / len(split_sentences)) * 100
        output_lines.append(f"  {split_type}: {count} ({percentage:.1f}%)")
    
    return "\n".join(output_lines)





def _get_grammar_enhancement_regions(turn_words: List[Dict]) -> List[Tuple[int, int]]:
    """
    Identify regions within a speaker turn that need grammar enhancement.
    
    This function finds bad grammar segments and extends the enhancement region
    into neighboring good grammar segments, stopping at punctuation marks.
    
    Requirements:
    - At least 2 consecutive bad grammar segments
    - Total word count of bad grammar segments must be at least 20
    
    Args:
        turn_words: List of word dictionaries in the speaker turn
        
    Returns:
        List of (start_idx, end_idx) tuples for regions to enhance
    """
    if not turn_words:
        return []
    
    # Group consecutive words by segment and grammar status
    segments = []
    current_segment = {
        'words': [],
        'start_idx': 0,
        'has_bad_grammar': False,
        'segment_index': None
    }
    
    for i, word in enumerate(turn_words):
        word_segment_idx = word.get('segment_index')
        word_has_good_grammar = word.get('has_good_grammar', True)
        
        # Check if we need to start a new segment (segment change or grammar status change)
        if (i == 0 or 
            word_segment_idx != current_segment['segment_index'] or
            word_has_good_grammar != (not current_segment['has_bad_grammar'])):
            
            if current_segment['words']:
                current_segment['end_idx'] = i - 1
                segments.append(current_segment)
            
            current_segment = {
                'words': [word],
                'start_idx': i,
                'has_bad_grammar': not word_has_good_grammar,
                'segment_index': word_segment_idx
            }
        else:
            current_segment['words'].append(word)
    
    # Add final segment
    if current_segment['words']:
        current_segment['end_idx'] = len(turn_words) - 1
        segments.append(current_segment)
    
    # Find regions to enhance
    enhancement_regions = []
    i = 0
    
    while i < len(segments):
        segment = segments[i]
        
        if segment['has_bad_grammar']:
            # Found a bad grammar segment - check if it meets our criteria
            
            # Collect all consecutive bad grammar segments
            bad_grammar_segments = [segment]
            j = i + 1
            while j < len(segments) and segments[j]['has_bad_grammar']:
                bad_grammar_segments.append(segments[j])
                j += 1
            
            # Check if we have at least 2 segments and at least 20 words total
            total_bad_grammar_words = sum(len(seg['words']) for seg in bad_grammar_segments)
            num_bad_segments = len(bad_grammar_segments)
            
            if num_bad_segments >= 2 and total_bad_grammar_words >= 20:
                # This meets our criteria for enhancement
                logger.debug(f"Found enhancement candidate: {num_bad_segments} bad grammar segments with {total_bad_grammar_words} total words")
                
                # Log the bad grammar text for debugging
                bad_grammar_text = ' '.join(w.get('text', '') for seg in bad_grammar_segments for w in seg['words'])
                logger.debug(f"Bad grammar text: '{bad_grammar_text}'")
                
                # Find the full sentence containing these bad grammar segments
                # Start from the beginning of the first bad grammar segment
                start_idx = bad_grammar_segments[0]['start_idx']
                
                # Check if there's a sentence boundary right before the bad grammar segments
                # If so, we don't need to extend backward
                if start_idx > 0:
                    prev_word_text = turn_words[start_idx - 1].get('text', '')
                    if any(prev_word_text.rstrip().endswith(p) for p in ['.', '!', '?']):
                        # Previous word ends with sentence-ending punctuation
                        # Start from the bad grammar segment itself
                        logger.debug(f"Bad grammar starts after sentence boundary '{prev_word_text}', no backward extension needed")
                    else:
                        # Extend backward through ALL segments until we find sentence-ending punctuation
                        # This ensures we capture the complete sentence/thought
                        found_start_boundary = False
                        for k in range(start_idx - 1, -1, -1):
                            word_text = turn_words[k].get('text', '')
                            # Check for sentence-ending punctuation at the END of the word
                            if any(word_text.rstrip().endswith(p) for p in ['.', '!', '?']):
                                start_idx = k + 1  # Start after punctuation
                                found_start_boundary = True
                                logger.debug(f"Found start boundary at word '{word_text}' (index {k})")
                                break
                        
                        if not found_start_boundary:
                            # No punctuation found, start from beginning of turn
                            start_idx = 0
                            logger.debug("No start boundary found, starting from beginning of turn")
                
                # Find end of enhancement region
                # Start from the end of the last bad grammar segment
                end_idx = bad_grammar_segments[-1]['end_idx']
                
                # Extend forward through ALL segments until we find sentence-ending punctuation
                # This ensures we capture the complete sentence/thought
                found_end_boundary = False
                for k in range(end_idx, len(turn_words)):
                    word_text = turn_words[k].get('text', '')
                    # Include the word with punctuation at the end, then stop
                    if any(word_text.rstrip().endswith(p) for p in ['.', '!', '?']):
                        end_idx = k
                        found_end_boundary = True
                        logger.debug(f"Found end boundary at word '{word_text}' (index {k})")
                        break
                
                if not found_end_boundary:
                    # No punctuation found, extend to end of turn
                    end_idx = len(turn_words) - 1
                    logger.debug("No end boundary found, extending to end of turn")
                
                enhancement_regions.append((start_idx, end_idx))
            else:
                # Doesn't meet criteria, log why
                logger.debug(f"Skipping enhancement: only {num_bad_segments} bad grammar segments with {total_bad_grammar_words} total words (need 2+ segments and 20+ words)")
            
            # Skip segments we've already processed
            i = j
        else:
            i += 1
    
    # Merge overlapping regions (shouldn't happen with our logic, but just in case)
    merged_regions = []
    for start, end in sorted(enhancement_regions):
        if merged_regions and start <= merged_regions[-1][1] + 1:
            # Overlapping or adjacent, extend the previous region
            merged_regions[-1] = (merged_regions[-1][0], max(merged_regions[-1][1], end))
        else:
            merged_regions.append((start, end))
    
    return merged_regions


def assign_unassigned_words_to_nearest_diarization(word_table: WordTable, content_id: str) -> int:
    """
    Final check: Assign any words without proper speaker assignments to the nearest diarization segment.
    
    This function finds all words that don't have proper SPEAKER_XX assignments (including 
    UNKNOWN, NEEDS_LLM, NEEDS_EMBEDDING, BAD_GRAMMAR_SINGLE, etc.) and assigns them to 
    the speaker from the nearest diarization segment based on timing to ensure 100% speaker coverage.
    
    Args:
        word_table: WordTable with potentially some unassigned words
        content_id: Content ID for logging
        
    Returns:
        Number of words that were reassigned to actual speakers
    """
    if not word_table.diarization_segments:
        logger.warning(f"[{content_id}] No diarization segments available for unassigned word assignment")
        return 0
    
    # Find all words that don't have proper SPEAKER_XX assignments
    # Proper speaker assignments match the pattern SPEAKER_XX (where XX is digits)
    speaker_pattern = r'^SPEAKER_\d+$'
    proper_speaker_mask = word_table.df['speaker_current'].str.match(speaker_pattern, na=False)
    unassigned_mask = ~proper_speaker_mask
    unassigned_words = word_table.df[unassigned_mask]
    
    if len(unassigned_words) == 0:
        logger.info(f"[{content_id}] All words have proper SPEAKER_XX assignments - 100% speaker coverage achieved")
        return 0
    
    # Log the types of unassigned speakers we found
    unassigned_types = unassigned_words['speaker_current'].value_counts()
    logger.info(f"[{content_id}] Found {len(unassigned_words)} words without proper speaker assignments:")
    for speaker_type, count in unassigned_types.items():
        logger.info(f"[{content_id}]   {speaker_type}: {count} words")
    
    logger.info(f"[{content_id}] Assigning these words to nearest diarization segments for 100% speaker coverage")
    
    # Convert diarization segments to numpy arrays for efficient computation
    diar_starts = np.array([seg['start'] for seg in word_table.diarization_segments])
    diar_ends = np.array([seg['end'] for seg in word_table.diarization_segments])
    diar_speakers = np.array([seg['speaker'] for seg in word_table.diarization_segments])
    
    assignments_made = 0
    
    for idx, word in unassigned_words.iterrows():
        word_start = word['start']
        word_end = word['end']
        word_mid = (word_start + word_end) / 2.0
        word_text = word['text']
        
        # Method 1: Check for overlap with any diarization segment
        overlap_start = np.maximum(word_start, diar_starts)
        overlap_end = np.minimum(word_end, diar_ends)
        overlap_durations = np.maximum(0, overlap_end - overlap_start)
        
        # Find segment with maximum overlap
        max_overlap_idx = np.argmax(overlap_durations)
        max_overlap = overlap_durations[max_overlap_idx]
        
        if max_overlap > 0:
            # Word overlaps with a diarization segment - use that speaker
            assigned_speaker = diar_speakers[max_overlap_idx]
            assignment_method = "diarization_overlap"
            confidence = 0.8
            reason = f"Overlaps with diarization segment by {max_overlap:.2f}s"
        else:
            # Method 2: No overlap - find nearest diarization segment by midpoint distance
            # Calculate distance from word midpoint to each segment
            distances_to_start = np.abs(diar_starts - word_mid)
            distances_to_end = np.abs(diar_ends - word_mid)
            distances_to_segment = np.minimum(distances_to_start, distances_to_end)
            
            # Find the nearest segment
            nearest_idx = np.argmin(distances_to_segment)
            nearest_distance = distances_to_segment[nearest_idx]
            assigned_speaker = diar_speakers[nearest_idx]
            assignment_method = "nearest_diarization"
            confidence = max(0.3, 0.8 - (nearest_distance / 10.0))  # Confidence decreases with distance
            reason = f"Nearest diarization segment at {nearest_distance:.2f}s distance"
        
        # Apply the assignment using the WordTable's built-in method
        success = word_table.assign_speaker_to_word_by_index(
            idx=idx,
            speaker=assigned_speaker,
            method=assignment_method,
            confidence=confidence,
            stage='stage11_cleanup',
            reason=reason
        )
        
        if success:
            assignments_made += 1
            original_assignment = word['speaker_current']
            logger.debug(f"[{content_id}] Assigned '{original_assignment}' word '{word_text}' at {word_mid:.2f}s to {assigned_speaker} ({assignment_method}, confidence: {confidence:.2f})")
        else:
            logger.warning(f"[{content_id}] Failed to assign unassigned word '{word_text}' at index {idx}")
    
    logger.info(f"[{content_id}] Final assignment check: successfully assigned {assignments_made}/{len(unassigned_words)} words to nearest diarization segments")
    
    # Verify 100% speaker coverage
    speaker_pattern = r'^SPEAKER_\d+$'
    proper_speaker_mask = word_table.df['speaker_current'].str.match(speaker_pattern, na=False)
    remaining_unassigned = (~proper_speaker_mask).sum()
    
    if remaining_unassigned == 0:
        logger.info(f"[{content_id}] ✓ 100% speaker coverage achieved - all {len(word_table.df)} words have proper SPEAKER_XX assignments")
    else:
        logger.warning(f"[{content_id}] ⚠ Speaker coverage incomplete - {remaining_unassigned} words still lack proper assignments")
        # Log the types of remaining unassigned words for debugging
        remaining_words = word_table.df[~proper_speaker_mask]
        remaining_types = remaining_words['speaker_current'].value_counts()
        for speaker_type, count in remaining_types.items():
            logger.warning(f"[{content_id}]   Remaining {speaker_type}: {count} words")
    
    return assignments_made


def stage11_cleanup(content_id: str,
                           word_table: WordTable,
                           output_dir: Optional[Path] = None,
                           test_mode: bool = False) -> Dict[str, Any]:
    """
    Execute Stage 11: Final cleanup and grammar enhancement.
    
    Args:
        content_id: Content ID to process
        word_table: WordTable with final speaker assignments
        output_dir: Optional directory to save output files
        test_mode: Whether to run in test mode
        
    Returns:
        Dictionary with cleanup and enhancement results
    """
    start_time = time.time()
    stage_name = 'stage11_cleanup'
    
    logger.info(f"[{content_id}] Starting Stage 11: Final Cleanup and Grammar Enhancement")
    
    result = {
        'status': 'pending',
        'content_id': content_id,
        'stage': stage_name,
        'data': {
            'sentences': None,
            'split_sentences': None,
            'split_analysis': None,
            'output_files': []
        },
        'stats': {},
        'error': None
    }
    
    try:
        # Validate inputs
        if word_table is None or word_table.df is None:
            raise ValueError("No word table available")
        
        if len(word_table.df) == 0:
            logger.warning(f"[{content_id}] Word table is empty, no words to process")
            result['status'] = 'success'
            result['data']['word_table'] = word_table
            result['stats'] = {
                'duration': time.time() - start_time,
                'words_processed': 0,
                'speaker_turns_enhanced': 0,
                'grammar_enhancements_applied': 0
            }
            return result
        
        # Step 1: Handle single-word orphans bounded by same speaker
        logger.info(f"[{content_id}] Step 1: Handling single-word orphans bounded by same speaker...")
        
        # Get a copy of the dataframe sorted by time but preserve original index
        df = word_table.df.copy()
        df = df.sort_values('start')  # Don't reset index to preserve original indices
        
        # Track single-word orphan reassignments
        single_word_orphans_found = 0
        single_word_orphans_reassigned = 0
        
        # Convert to list for easier iteration while keeping original indices
        df_rows = [(idx, row) for idx, row in df.iterrows()]
        
        # Identify single-word speaker segments bounded by the same speaker
        for i in range(1, len(df_rows) - 1):  # Skip first and last words
            current_idx, current_word = df_rows[i]
            prev_idx, prev_word = df_rows[i-1]
            next_idx, next_word = df_rows[i+1]
            
            current_speaker = current_word.get('speaker_current', 'UNKNOWN')
            prev_speaker = prev_word.get('speaker_current', 'UNKNOWN')
            next_speaker = next_word.get('speaker_current', 'UNKNOWN')
            
            # Check if this is a single-word orphan (different speaker) bounded by same speaker
            if (current_speaker != prev_speaker and 
                current_speaker != next_speaker and 
                prev_speaker == next_speaker and
                prev_speaker != 'UNKNOWN' and
                current_speaker != 'UNKNOWN'):
                
                single_word_orphans_found += 1
                
                # Additional checks for reassignment confidence
                time_gap_before = current_word['start'] - prev_word['end']
                time_gap_after = next_word['start'] - current_word['end']
                
                # More flexible time gap conditions:
                # - If BOTH gaps are small (< 1s), definitely reassign
                # - If one gap is small (< 1s) and the other is moderate (< 5s), still reassign
                # - Only skip if both gaps are large (>= 5s)
                small_gap_threshold = 1.0
                moderate_gap_threshold = 5.0
                
                both_gaps_small = time_gap_before < small_gap_threshold and time_gap_after < small_gap_threshold
                one_small_one_moderate = (
                    (time_gap_before < small_gap_threshold and time_gap_after < moderate_gap_threshold) or
                    (time_gap_after < small_gap_threshold and time_gap_before < moderate_gap_threshold)
                )
                
                if both_gaps_small or one_small_one_moderate:
                    logger.debug(f"[{content_id}] Reassigning single-word orphan '{current_word['text']}' from {current_speaker} to {prev_speaker} (gaps: before={time_gap_before:.2f}s, after={time_gap_after:.2f}s)")
                    
                    # Update the word table using the original index
                    word_table.df.at[current_idx, 'speaker_current'] = prev_speaker
                    word_table.df.at[current_idx, 'resolution_method'] = 'single_word_orphan_consolidation'
                    word_table.df.at[current_idx, 'assignment_confidence'] = 0.9
                    
                    # Add to assignment history
                    if 'assignment_history' in word_table.df.columns:
                        history = word_table.df.at[current_idx, 'assignment_history']
                        if isinstance(history, list):
                            history.append({
                                'stage': 'stage11_cleanup',
                                'method': 'single_word_orphan_consolidation',
                                'from_speaker': current_speaker,
                                'to_speaker': prev_speaker,
                                'confidence': 0.9,
                                'reason': f'Single word bounded by {prev_speaker} (gaps: before={time_gap_before:.1f}s, after={time_gap_after:.1f}s)'
                            })
                    
                    single_word_orphans_reassigned += 1
                else:
                    logger.debug(f"[{content_id}] Skipping single-word orphan '{current_word['text']}' - both time gaps too large (before: {time_gap_before:.2f}s, after: {time_gap_after:.2f}s, both >= {moderate_gap_threshold}s)")
        
        logger.info(f"[{content_id}] Single-word orphan consolidation: found {single_word_orphans_found}, reassigned {single_word_orphans_reassigned}")
        
        # Update df with the changes from word_table
        df = word_table.df.copy()
        df = df.sort_values('start').reset_index(drop=True)
        
        # Step 2: Apply grammar enhancement to speaker turns with malformed grammar
        logger.info(f"[{content_id}] Step 2: Applying grammar enhancement to speaker turns with malformed grammar...")
        
        # Initialize grammar enhancer
        grammar_enhancer = GrammarEnhancer()
        
        # Group consecutive words by speaker
        speaker_turns = []
        current_speaker = None
        current_turn_words = []
        
        for idx, word in df.iterrows():
            word_speaker = word.get('speaker_current', 'UNKNOWN')
            
            # Start new turn if speaker changed or there's a significant time gap
            if (word_speaker != current_speaker or 
                (current_turn_words and word.get('start', 0) - current_turn_words[-1].get('end', 0) > 2.0)):
                
                # Save previous turn if it exists
                if current_turn_words and current_speaker != 'UNKNOWN':
                    speaker_turns.append({
                        'speaker': current_speaker,
                        'words': current_turn_words.copy(),
                        'start': current_turn_words[0].get('start', 0),
                        'end': current_turn_words[-1].get('end', 0),
                        'text': ' '.join(w.get('text', '') for w in current_turn_words)
                    })
                
                # Start new turn
                current_speaker = word_speaker
                current_turn_words = [word.to_dict()]
            else:
                # Continue current turn
                current_turn_words.append(word.to_dict())
        
        # Add final turn
        if current_turn_words and current_speaker != 'UNKNOWN':
            speaker_turns.append({
                'speaker': current_speaker,
                'words': current_turn_words.copy(),
                'start': current_turn_words[0].get('start', 0),
                'end': current_turn_words[-1].get('end', 0),
                'text': ' '.join(w.get('text', '') for w in current_turn_words)
            })
        
        # Analyze turns for enhancement regions
        total_enhancement_regions = 0
        for turn in speaker_turns:
            regions = _get_grammar_enhancement_regions(turn['words'])
            turn['enhancement_regions'] = regions
            total_enhancement_regions += len(regions)
        
        logger.info(f"[{content_id}] Found {len(speaker_turns)} speaker turns, {total_enhancement_regions} regions need grammar enhancement")
        logger.info(f"[{content_id}] Using targeted enhancement: bad grammar segments + context until punctuation")
        
        # Apply grammar enhancement to specific regions
        enhanced_turns = []
        regions_enhanced = 0
        total_enhanced_words = 0
        grammar_start_time = time.time()
        
        for turn_idx, turn in enumerate(speaker_turns):
            turn_start_time = time.time()
            original_text = turn['text']
            word_count = len(turn['words'])
            enhancement_regions = turn.get('enhancement_regions', [])
            
            # Create a working copy of the turn
            turn_data = {
                'speaker': turn['speaker'],
                'start': turn['start'],
                'end': turn['end'],
                'original_text': original_text,
                'word_count': word_count,
                'enhancement_regions': enhancement_regions,
                'region_enhancements': []
            }
            
            if enhancement_regions:
                # This turn has regions that need enhancement
                logger.debug(f"[{content_id}] Turn {turn_idx+1} for {turn['speaker']} has {len(enhancement_regions)} enhancement regions")
                
                # Enhance each region
                for region_idx, (start_idx, end_idx) in enumerate(enhancement_regions):
                    # Extract region text
                    region_words = turn['words'][start_idx:end_idx+1]
                    region_text = ' '.join(w.get('text', '') for w in region_words)
                    region_word_count = len(region_words)
                    
                    logger.debug(f"[{content_id}]   Region {region_idx+1} [{start_idx}:{end_idx+1}] ({region_word_count} words): '{region_text[:50]}...'")
                    
                    # Apply grammar enhancement to this region
                    enhanced_region_text = grammar_enhancer.enhance_speaker_turn(region_text)
                    
                    # Log before/after in test mode (show all regions being processed)
                    if test_mode:
                        logger.info(f"[TEST MODE] Grammar Enhancement - Turn {turn_idx+1}, Region {region_idx+1}:")
                        logger.info(f"  Speaker: {turn['speaker']}")
                        logger.info(f"  Word count: {region_word_count}")
                        logger.info(f"  Original: {region_text}")
                        logger.info(f"  Enhanced: {enhanced_region_text}")
                        if enhanced_region_text != region_text:
                            logger.info(f"  Status: CHANGED")
                        else:
                            logger.info(f"  Status: NO CHANGE")
                    
                    # Check if enhancement was actually applied
                    if enhanced_region_text != region_text:
                        regions_enhanced += 1
                        total_enhanced_words += region_word_count
                        logger.debug(f"[{content_id}]     -> Enhanced to: '{enhanced_region_text[:50]}...'")
                    else:
                        logger.debug(f"[{content_id}]     -> No change needed")
                    
                    # Store region enhancement data
                    turn_data['region_enhancements'].append({
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'original_text': region_text,
                        'enhanced_text': enhanced_region_text,
                        'word_count': region_word_count,
                        'was_enhanced': enhanced_region_text != region_text
                    })
                
                turn_duration = time.time() - turn_start_time
                turn_data['processing_time'] = turn_duration
                turn_data['had_enhancement_regions'] = True
            else:
                # No enhancement needed for this turn
                turn_data['enhanced_text'] = original_text
                turn_data['processing_time'] = 0.0
                turn_data['had_enhancement_regions'] = False
            
            enhanced_turns.append(turn_data)
        
        grammar_total_time = time.time() - grammar_start_time
        logger.info(f"[{content_id}] Grammar enhancement phase completed in {grammar_total_time:.2f}s: enhanced {regions_enhanced} regions containing {total_enhanced_words} words")
        
        result['data']['enhanced_speaker_turns'] = enhanced_turns
        
        # Apply enhanced text back to individual words in the WordTable
        update_start_time = time.time()
        words_updated_with_enhanced_text = 0
        turns_with_updates = 0
        
        for turn_idx, (turn, turn_data) in enumerate(zip(speaker_turns, enhanced_turns)):
            if turn_data.get('region_enhancements'):
                turn_update_start = time.time()
                turns_with_updates += 1
                
                # Process each enhanced region
                for region_enhancement in turn_data['region_enhancements']:
                    if not region_enhancement['was_enhanced']:
                        continue
                    
                    start_idx = region_enhancement['start_idx']
                    end_idx = region_enhancement['end_idx']
                    original_text = region_enhancement['original_text']
                    enhanced_text = region_enhancement['enhanced_text']
                    
                    # Get the original words in this region
                    original_words = turn['words'][start_idx:end_idx+1]
                    enhanced_words = enhanced_text.split()
                    
                    # Apply enhanced text - handle word count mismatches gracefully
                    word_updates = []
                    
                    if len(enhanced_words) == len(original_words):
                        # Perfect match - do 1:1 word mapping
                        for i, word_data in enumerate(original_words):
                            # Use vectorized boolean indexing to find matching words
                            mask = (
                                (word_table.df['text'] == word_data['text']) &
                                (abs(word_table.df['start'] - word_data['start']) < 0.1)
                            )
                            matching_indices = word_table.df.index[mask]
                            
                            if len(matching_indices) > 0:
                                word_idx = matching_indices[0]  # Take first match
                                word_updates.append((word_idx, enhanced_words[i], word_data['text'], 'exact_match'))
                    
                    elif len(enhanced_words) != len(original_words):
                        # Word count mismatch - use enhanced text but preserve timing
                        logger.debug(f"[{content_id}] Region word count mismatch: {len(enhanced_words)} enhanced vs {len(original_words)} original words")
                        
                        # Strategy: Distribute enhanced words across original word slots
                        enhanced_per_original = len(enhanced_words) / len(original_words)
                        
                        for i, word_data in enumerate(original_words):
                            # Calculate how many enhanced words this original word should get
                            start_enhanced_idx = int(i * enhanced_per_original)
                            end_enhanced_idx = int((i + 1) * enhanced_per_original)
                            
                            # Handle edge case for last word
                            if i == len(original_words) - 1:
                                end_enhanced_idx = len(enhanced_words)
                            
                            # Get the enhanced word(s) for this slot
                            if start_enhanced_idx < len(enhanced_words):
                                if start_enhanced_idx == end_enhanced_idx:
                                    # This slot gets no words (fewer enhanced than original)
                                    enhanced_word_portion = ""
                                    mismatch_type = "empty_slot"
                                else:
                                    # This slot gets one or more enhanced words
                                    enhanced_word_portion = " ".join(enhanced_words[start_enhanced_idx:end_enhanced_idx])
                                    mismatch_type = "distributed"
                            else:
                                enhanced_word_portion = ""
                                mismatch_type = "empty_slot"
                            
                            # Find the corresponding word in the table
                            mask = (
                                (word_table.df['text'] == word_data['text']) &
                                (abs(word_table.df['start'] - word_data['start']) < 0.1)
                            )
                            matching_indices = word_table.df.index[mask]
                            
                            if len(matching_indices) > 0:
                                word_idx = matching_indices[0]
                                if enhanced_word_portion:  # Only update if we have text
                                    word_updates.append((word_idx, enhanced_word_portion, word_data['text'], mismatch_type))
                    
                    # Apply all updates at once using vectorized operations
                    if word_updates:
                        indices = [idx for idx, _, _, _ in word_updates]
                        new_texts = [text for _, text, _, _ in word_updates]
                        original_texts = [orig for _, _, orig, _ in word_updates]
                        mismatch_types = [mtype for _, _, _, mtype in word_updates]
                        
                        # Vectorized text updates
                        word_table.df.loc[indices, 'text'] = new_texts
                        
                        # Vectorized metadata updates
                        for idx, orig_text, mismatch_type in zip(indices, original_texts, mismatch_types):
                            current_metadata = word_table.df.at[idx, 'metadata']
                            if isinstance(current_metadata, dict):
                                current_metadata['grammar_enhanced'] = True
                                current_metadata['original_text'] = orig_text
                                current_metadata['word_mapping'] = mismatch_type
                                current_metadata['enhancement_region'] = f"[{start_idx}:{end_idx+1}]"
                            else:
                                word_table.df.at[idx, 'metadata'] = {
                                    'grammar_enhanced': True,
                                    'original_text': orig_text,
                                    'word_mapping': mismatch_type,
                                    'enhancement_region': f"[{start_idx}:{end_idx+1}]"
                                }
                        
                        words_updated_with_enhanced_text += len(word_updates)
                
                turn_update_duration = time.time() - turn_update_start
                region_count = len(turn_data['region_enhancements'])
                enhanced_region_count = sum(1 for r in turn_data['region_enhancements'] if r['was_enhanced'])
                
                if enhanced_region_count > 0:
                    logger.debug(f"[{content_id}] Updated words in {enhanced_region_count}/{region_count} regions for {turn['speaker']} turn ({turn_update_duration:.3f}s)")
        
        update_total_time = time.time() - update_start_time
        logger.debug(f"[{content_id}] Vectorized in {update_total_time:.2f}s: {words_updated_with_enhanced_text} words updated across {turns_with_updates} enhanced turns")
        
        logger.info(f"[{content_id}] Grammar enhancement completed: {regions_enhanced} regions enhanced across {turns_with_updates} turns, {words_updated_with_enhanced_text} words updated")
        
        # Step 3: Analyze split sentences (DISABLED - focuses on individual segments, not speaker-turns)
        # Orphan segment analysis is disabled because it processes individual word segments
        # rather than full speaker-turns, which can interfere with earlier stage assignments.
        # Stage 11 now focuses on single-word orphan consolidation and grammar enhancement only.
        logger.info(f"[{content_id}] Step 3: Orphan segment analysis disabled - stage focuses on single-word consolidation and grammar enhancement")
        
        # # Get orphaned word segments (focused approach)
        # logger.info(f"[{content_id}] Step 3: Identifying orphaned word segments at turn boundaries...")
        # orphan_segments = get_sentences_from_word_table(word_table)
        # result['data']['orphan_segments'] = orphan_segments
        # 
        # # Analyze orphan segments for reassignment
        # logger.debug(f"[{content_id}] Analyzing for potential reassignment...")
        # split_sentences = identify_split_sentences(orphan_segments)
        # result['data']['split_sentences'] = split_sentences
        
        # Set empty data for orphan analysis since it's disabled
        orphan_segments = []
        split_sentences = []
        result['data']['orphan_segments'] = orphan_segments
        result['data']['split_sentences'] = split_sentences
        
        # LLM conversation parsing analysis (simplified - no acoustic validation)
        consolidation_results = []
        words_consolidated = 0
        sentences_analyzed = 0
        sentences_consolidated = 0
        sentences_left_split = 0
        
        if split_sentences:
            logger.info(f"[{content_id}] Found {len(split_sentences)} orphan segments - LLM conversation parsing disabled (works on individual segments, not speaker-turns)")
            
            # LLM conversation parsing is disabled because it operates on individual orphan segments
            # rather than full speaker-turns. This can reassign words that have already been resolved
            # by earlier stages. Future enhancement should apply LLM to speaker-turns with bad grammar.
            sentences_left_split = len(split_sentences)
            
            # # Initialize conversation parsing analyzer
            # try:
            #     conversation_parser = ConversationParsingAnalyzer()
            #     
            #     for orphan_segment in split_sentences:
            #         sentences_analyzed += 1
            #         orphan_text = orphan_segment['text'][:50]
            #         current_speaker = orphan_segment.get('current_speaker', 'UNKNOWN')
            #         candidate_speaker = orphan_segment.get('candidate_speaker', 'N/A')
            #         reassignment_confidence = orphan_segment.get('reassignment_confidence', 0)
            #         
            #         logger.info(f"[{content_id}] 📝 Orphan {orphan_segment['fragment_id']} ({orphan_segment['type']}): '{orphan_text}...' [{current_speaker} -> {candidate_speaker}?] (conf: {reassignment_confidence:.3f})")
            #         
            #         # HIGH CONFIDENCE THRESHOLD - Only apply LLM to cases with strong evidence
            #         if reassignment_confidence < 0.7:
            #             logger.info(f"[{content_id}]   ⚬ Skipping: low confidence ({reassignment_confidence:.3f} < 0.7)")
            #             sentences_left_split += 1
            #             continue
            #         
            #         # Use LLM to parse conversation into speaker segments
            #         conversation_result = conversation_parser.parse_orphan_conversation(orphan_segment, word_table)
            #         
            #         if conversation_result['status'] != 'success':
            #             logger.warning(f"[{content_id}]   ❌ Conversation parsing failed: {conversation_result['reason']}")
            #             sentences_left_split += 1
            #             continue
            #         
            #         parsed_segments = conversation_result['parsed_segments']
            #         
            #         # Log summary of what the LLM recommended
            #         if parsed_segments:
            #             total_words = sum(seg['word_count'] for seg in parsed_segments)
            #             speaker_breakdown = {}
            #             for seg in parsed_segments:
            #                 speaker = seg['speaker']
            #                 speaker_breakdown[speaker] = speaker_breakdown.get(speaker, 0) + seg['word_count']
            #             breakdown_str = ', '.join([f"{spk}:{count}w" for spk, count in speaker_breakdown.items()])
            #             logger.info(f"[{content_id}]   🎭 LLM recommended: {breakdown_str} (total: {total_words}w)")
            #         
            #         # Apply LLM parsing (no acoustic validation)
            #         decision = {
            #             'status': 'success',
            #             'action': 'apply_conversation_parsing',
            #             'parsed_segments': parsed_segments,
            #             'confidence': 0.6,
            #             'reason': "LLM conversation parsing",
            #             'method': 'llm_conversation_parsing'
            #         }
            #         
            #         logger.info(f"[{content_id}]   🤖 Applying LLM conversation parsing")
            #         consolidation_results.append({
            #             'fragment_id': orphan_segment['fragment_id'],
            #             'decision': decision,
            #             'original_speaker': orphan_segment['current_speaker'],
            #             'candidate_speaker': orphan_segment['candidate_speaker'],
            #             'orphan_type': orphan_segment['type'],
            #             'reassignment_confidence': orphan_segment['reassignment_confidence'],
            #             'analysis_method': 'llm_conversation_parsing'
            #         })
            #         
            #         # Apply the decision
            #         if decision['status'] == 'success' and decision['action'] == 'apply_conversation_parsing':
            #             # Apply conversation parsing by updating word assignments
            #             words_updated = apply_conversation_parsing_decision(word_table, orphan_segment, decision)
            #             words_consolidated += words_updated
            #             sentences_consolidated += 1
            #             
            #             # Create summary of assignments
            #             speaker_assignments = {}
            #             for seg in decision['parsed_segments']:
            #                 speaker = seg['speaker']
            #                 speaker_assignments[speaker] = speaker_assignments.get(speaker, 0) + seg['word_count']
            #             
            #             assignment_summary = ', '.join([f"{speaker}:{count}w" for speaker, count in speaker_assignments.items()])
            #             logger.info(f"[{content_id}] ✓ 🤖 Applied LLM conversation parsing: {words_updated} words updated → {assignment_summary}")
            #         else:
            #             sentences_left_split += 1
            #             logger.info(f"[{content_id}] ⚬ Left unchanged: {decision.get('action', 'unknown')}")
            #     
            # except Exception as e:
            #     logger.error(f"[{content_id}] LLM orphan analysis failed: {str(e)}")
            #     logger.error(f"[{content_id}] Continuing without reassignment...")
        
        result['data']['consolidation_results'] = consolidation_results
        
        # Step 4: Final check - assign any words without proper speaker assignments to nearest diarization segment
        logger.info(f"[{content_id}] Step 4: Final check - ensuring 100% speaker coverage by assigning unassigned words to nearest diarization segments...")
        unassigned_words_assigned = assign_unassigned_words_to_nearest_diarization(word_table, content_id)
        
        # Generate analysis output (after potential consolidation and final assignment)
        logger.info(f"[{content_id}] Generating split sentence analysis...")
        split_analysis = format_split_sentence_output(split_sentences, word_table)
        result['data']['split_analysis'] = split_analysis
        
        # Save to file if output directory provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save split sentences analysis
            split_path = output_dir / f"{content_id}_split_sentences.txt"
            with open(split_path, 'w') as f:
                f.write(split_analysis)
            result['data']['output_files'].append(str(split_path))
            
            logger.info(f"[{content_id}] Saved split sentence analysis to {split_path}")
        
        # Calculate statistics
        split_types = {}
        total_split_duration = 0
        total_split_words = 0
        
        for split_sentence in split_sentences:
            split_type = split_sentence['split_type']
            split_types[split_type] = split_types.get(split_type, 0) + 1
            total_split_duration += split_sentence['duration']
            total_split_words += split_sentence['word_count']
        
        result['stats'] = {
            'duration': time.time() - start_time,
            'single_word_orphans': {
                'found': single_word_orphans_found,
                'reassigned': single_word_orphans_reassigned,
                'reassignment_rate': (single_word_orphans_reassigned / single_word_orphans_found * 100) if single_word_orphans_found > 0 else 0
            },
            'grammar_enhancement': {
                'speaker_turns_processed': len(speaker_turns),
                'total_enhancement_regions': total_enhancement_regions,
                'regions_enhanced': regions_enhanced,
                'turns_with_enhancements': turns_with_updates,
                'words_updated_with_enhanced_text': words_updated_with_enhanced_text,
                'enhancement_rate': (regions_enhanced / total_enhancement_regions * 100) if total_enhancement_regions > 0 else 0,
                'total_words_in_turns': sum(turn['word_count'] for turn in enhanced_turns),
                'enhanced_words': total_enhanced_words
            },
            'split_sentence_analysis': {
                'total_orphan_segments': len(orphan_segments),
                'analyzed_orphan_segments': len(split_sentences),
                'orphan_analysis_rate': (len(split_sentences) / len(orphan_segments) * 100) if orphan_segments else 0,
                'split_types': split_types,
                'total_split_duration': total_split_duration,
                'total_split_words': total_split_words,
                'avg_split_duration': total_split_duration / len(split_sentences) if split_sentences else 0,
                'reassignment_analysis': {
                    'orphans_analyzed': sentences_analyzed,
                    'orphans_reassigned': sentences_consolidated,
                    'orphans_left_unchanged': sentences_left_split,
                    'words_reassigned': words_consolidated,
                    'reassignment_rate': (sentences_consolidated / sentences_analyzed * 100) if sentences_analyzed > 0 else 0
                },
                'analysis_methods': {
                    'llm_conversation_parsing': sentences_consolidated,  # LLM-based conversation parsing
                    'parsing_success_rate': (sentences_consolidated / sentences_analyzed * 100) if sentences_analyzed > 0 else 0
                }
            },
            'final_unassigned_assignment': {
                'unassigned_words_assigned': unassigned_words_assigned,
                'method': 'nearest_diarization_segment'
            },
            'approach': 'grammar_enhancement_plus_split_analysis_plus_100_percent_speaker_coverage'
        }
        
        # Return the updated word table
        result['data']['word_table'] = word_table
        result['status'] = 'success'
        logger.info(f"[{content_id}] Stage 11 completed: {single_word_orphans_reassigned} single-word orphans reassigned, {regions_enhanced} grammar regions enhanced (out of {total_enhancement_regions} identified), {len(split_sentences)} orphan segments analyzed, {unassigned_words_assigned} unassigned words assigned to nearest diarization segments for 100% speaker coverage")
        logger.info(f"[{content_id}] LLM conversation parsing: analyzed 10-word windows + orphan + 10-word windows")
        if sentences_analyzed > 0:
            logger.info(f"[{content_id}] Conversation parsing summary: {sentences_consolidated} reassigned, {sentences_left_split} left unchanged, {words_consolidated} words updated")
            parsing_rate = (sentences_consolidated / sentences_analyzed * 100)
            logger.info(f"[{content_id}] LLM parsing success rate: {parsing_rate:.1f}% ({sentences_consolidated}/{sentences_analyzed})")
        
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